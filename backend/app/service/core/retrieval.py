"""Knowledge-base retrieval entry points used by chat and offline evaluation."""

import logging
import os
import re
import time
from copy import deepcopy
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any
from service.core.evidence_metadata import evidence_metadata, ranking_text

from service.core.constraint_compatibility import evaluate_constraint_compatibility, extract_locators
from service.core.evidence_sufficiency import check_evidence_sufficiency
from service.core.query_expansion import QueryExpansionResult, expand_query
from service.core.query_intent import QueryAnalysis
from service.core.query_contract import QueryRequest, resolve_request
from service.core.query_execution import execute_graph
from service.core.query_runtime import (
    bounded_query, current_runtime, run_independent, QueryBudgetExceeded,
    observation_context, record_observation, failure_details,
    observe_plan_execution, record_execution_node,
)
from service.core.query_planning import (
    QuerySubqueryPlan,
    RetrievalPlan,
    prepare_retrieval_request,
)
from service.core.query_rewrite import rewrite_query
from service.core.rag.nlp.model import rerank_similarity
from service.core.rag.nlp.search_v2 import Dealer, reciprocal_rank_fusion
from service.core.rag.utils.es_conn import ESConnection
from service.core.sequential_retrieval import (
    BridgeExtraction,
    extract_bridge_value,
    resolve_query_template,
)
from service.core.table_context import expand_table_contexts
from service.core.query_validation import QueryValidationError, missing_constraints, shared_scope_terms


DEFAULT_PAGE_SIZE = 5
DEFAULT_SIMILARITY_THRESHOLD = 0.1
DEFAULT_VECTOR_SIMILARITY_WEIGHT = 0.6
DEFAULT_RETRIEVAL_CANDIDATE_SIZE = 100
DEFAULT_RERANK_CANDIDATE_SIZE = 20
DEFAULT_RRF_K = 60
DEFAULT_FINAL_RERANKER_WEIGHT = 0.7
DEFAULT_FINAL_RRF_K = 10
DEFAULT_PARALLEL_COVERAGE_PER_SUBQUERY = 2
DEFAULT_PARALLEL_CANDIDATE_MULTIPLIER = 2


logger = logging.getLogger(__name__)


def _stage_latency(result: dict[str, Any], stage: str) -> float:
    value = (result.get("stage_latency_ms") or {}).get(stage)
    try:
        latency = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, latency)


def _sum_stage_latency(
    results: list[dict[str, Any]],
    stage: str,
) -> float:
    return sum(_stage_latency(result, stage) for result in results)


def _environment_flag(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().casefold() not in {"0", "false", "no", "off"}


def _parallel_coverage_per_subquery() -> int:
    raw_value = os.getenv(
        "RAG_PARALLEL_COVERAGE_PER_SUBQUERY",
        str(DEFAULT_PARALLEL_COVERAGE_PER_SUBQUERY),
    )
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return DEFAULT_PARALLEL_COVERAGE_PER_SUBQUERY
    return value if value > 0 else DEFAULT_PARALLEL_COVERAGE_PER_SUBQUERY


@lru_cache(maxsize=1)
def get_retrieval_dealer() -> Dealer:
    """Create the Elasticsearch-backed dealer only when retrieval is used."""
    return Dealer(dataStore=ESConnection())


def _prepare_query_transform(
    original_question: str,
    query_analysis: QueryAnalysis,
) -> tuple[str, QueryExpansionResult]:
    # 改写用户查询，扩展用户查询
    if query_analysis.need_rewrite:
        rewritten_question = rewrite_query(original_question)
    else:
        rewritten_question = original_question
    if query_analysis.need_rewrite or query_analysis.need_expansion:
        query_transform = expand_query(original_question, rewritten_question)
    else:
        query_transform = QueryExpansionResult.noop(
            original_question,
            reason="RAG 意图识别判定无需改写",
        )
    return rewritten_question, query_transform


class _BudgetedDealer:
    """Request-local query deduplication and leaf transformation."""

    def __init__(self, dealer, *, transform_nodes=False):
        self.dealer = dealer
        self.transform_nodes = transform_nodes

    def retrieval(self, **kwargs):
        runtime = current_runtime()
        key = ("retrieval", repr(kwargs))
        if runtime is not None:
            with runtime._lock:
                cached = runtime.cache.get(key)
            if cached is not None:
                runtime.record(stage="retrieval", phase="retrieval", status="cache_hit", end_reason="cached_result")
                return deepcopy(cached)
        transform = None
        started = time.perf_counter()
        try:
            if runtime is not None:
                runtime.reserve("retrieval")
            if self.transform_nodes:
                query = kwargs["question"]
                transform = expand_query(query, query, allow_model=False)
                kwargs["keyword_question"] = transform.effective_query
            result = self.dealer.retrieval(**kwargs)
        except Exception as error:
            if runtime is not None:
                runtime.record(stage="retrieval", phase="retrieval", status="error", error=type(error).__name__,
                               end_reason="budget_exhausted" if isinstance(error, QueryBudgetExceeded) else "retrieval_failed",
                               **failure_details(error, default_rule="retrieval_failed"),
                               duration_ms=round((time.perf_counter() - started) * 1000, 3))
            raise
        if transform is not None:
            result["query_transform"] = transform.to_dict()
        if runtime is not None:
            runtime.record(stage="retrieval", phase="retrieval", status="ok",
                           end_reason="results_returned" if result.get("chunks") else "empty_results",
                           duration_ms=round((time.perf_counter() - started) * 1000, 3),
                           result_count=len(result.get("chunks") or []))
            with runtime._lock:
                runtime.cache[key] = deepcopy(result)
        return result


def _chunk_key(chunk: dict[str, Any]) -> str:
    chunk_id = str(chunk.get("chunk_id") or "").strip()
    if chunk_id:
        return f"id:{chunk_id}"
    return "content:{document}:{content}".format(
        document=str(chunk.get("doc_id") or chunk.get("docnm_kwd") or ""),
        content=str(chunk.get("content_with_weight") or ""),
    )


def _document_aggregations(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregations: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        document_name = str(chunk.get("docnm_kwd") or "")
        if document_name not in aggregations:
            aggregations[document_name] = {
                "doc_name": document_name,
                "doc_id": str(chunk.get("doc_id") or ""),
                "count": 0,
            }
        aggregations[document_name]["count"] += 1
    return sorted(
        aggregations.values(),
        key=lambda item: (-int(item["count"]), item["doc_name"]),
    )


def _measurement_compatibility_priority(chunk: dict[str, Any]) -> int:
    categories = (chunk.get("constraint_compatibility") or {}).get(
        "categories",
        {},
    )
    dimension_level = str(
        (categories.get("measurement_dimension") or {}).get("level") or ""
    )
    if dimension_level == "conflict":
        return 2

    subject_level = str(
        (categories.get("measurement_subject") or {}).get("level") or ""
    )
    if subject_level:
        return 0 if subject_level == "compatible" else 1

    return {
        "compatible": 0,
        "unknown": 1,
        "conflict": 2,
    }.get(dimension_level, 1)

# 偏好精确的表格/文本数值，而不是图表中的近似估计值
def _approximate_figure_penalty(
    question: str,
    chunk: dict[str, Any],
) -> int:
    """Prefer exact prose/table values over approximate visual estimates."""
    if str(chunk.get("evidence_type_kwd") or "") != "figure":
        return 0
    if not re.search(r"多少|具体数值|精确|分别是", question):
        return 0
    if re.search(r"图\s*\d+|图表|柱|曲线|折线", question):
        return 0
    content = str(chunk.get("content_with_weight") or "")
    return int(bool(re.search(r"约|左右|未经人工核验", content)))

# 不再根据内容判断表格行是否重复，而是优先使用表格id和行号来标识表格行，避免使用表格内容来标识，因为表格内容可能会有重复或相似的情况
def _primary_table_row_signature(chunk: dict[str, Any]) -> str | None:
    """Return the identity of the primary table row represented by a chunk."""

    # 优先使用表格id和行号来标识表格行，避免使用表格内容来标识，因为表格内容可能会有重复或相似的情况
    table_id = str(chunk.get("table_id_kwd") or "").strip()
    row_index = chunk.get("table_row_int")
    if row_index is None or str(row_index).strip() == "":
        row_index = chunk.get("table_row_start_int")
    normalized_row_index = (
        "" if row_index is None else str(row_index).strip()
    )

    # This is the strongest identity: the same row in the same parsed table.
    if table_id and normalized_row_index:
        return f"table-row:{table_id}:{normalized_row_index}"

    # Older table chunks may not have stable table/row identifiers.  Their
    # primary cells are still safer than expanded markdown, which can contain
    # shared headers and neighbouring rows.
    cells = chunk.get("table_cells_kwd")
    if not isinstance(cells, list) or not cells:
        return None
    normalized_cells = [
        re.sub(
            r"[^0-9A-Za-z\u4e00-\u9fff.%()-]+",
            "",
            str(cell),
        ).casefold()
        for cell in cells
    ]
    normalized_cells = [cell for cell in normalized_cells if cell]
    if len(normalized_cells) < 2:
        return None

    document_id = str(
        chunk.get("doc_id") or chunk.get("document_id") or ""
    ).strip()
    return "table-cells:{document}:{cells}".format(
        document=document_id,
        cells="|".join(normalized_cells),
    )


def _duplicates_numeric_fact(
    key: str,
    selected_keys: set[str],
    chunks_by_key: dict[str, dict[str, Any]],
) -> bool:
    signature = _primary_table_row_signature(chunks_by_key[key])
    if signature is None:
        return False
    for selected_key in selected_keys:
        if selected_key == key:
            continue
        selected_signature = _primary_table_row_signature(
            chunks_by_key[selected_key]
        )
        if signature == selected_signature:
            return True
    return False


def _merge_parallel_subqueries(
    *,
    original_question: str,
    subquery_results: list[
        tuple[QuerySubqueryPlan, dict[str, Any]]
    ],
    options: Dealer.RetrievalOptions,
) -> dict[str, Any]:
    merge_started = time.perf_counter()
    chunks_by_key: dict[str, dict[str, Any]] = {}
    subquery_keys: dict[str, list[str]] = {}

    for step, result in subquery_results:
        current_keys = []
        for subquery_rank, source_chunk in enumerate(
            result.get("chunks") or [],
            start=1,
        ):
            key = _chunk_key(source_chunk)
            current_keys.append(key)
            subquery_metadata = {
                "subquery_id": step.id,
                "subquery_rank": subquery_rank,
                "query": step.query,
                "ranking": dict(source_chunk.get("final_ranking") or {}),
                "rerank_score": source_chunk.get("rerank_score"),
                "rrf_score": source_chunk.get("rrf_score"),
                "constraint_compatibility": dict(
                    source_chunk.get("constraint_compatibility") or {}
                ),
            }
            # 同一个chunk可能被多个子查询召回，这里去重
            # parallel_subqueries表示这个chunk是被哪些子查询召回的
            if key not in chunks_by_key:
                chunk = dict(source_chunk)
                chunk["parallel_subqueries"] = [subquery_metadata]
                chunks_by_key[key] = chunk
            else:
                chunks_by_key[key]["parallel_subqueries"].append(
                    subquery_metadata
                )
        subquery_keys[step.id] = current_keys

    candidate_keys = list(chunks_by_key)
    scores_by_key: dict[str, float] = {}
    semantic_ranks: dict[str, int] = {}
    ranking_source = "model"
    ranking_method = "original_question_rerank"
    fallback_reason = None
    # 对全部候选做统一重排，用原始问题使用reranker模型进行重排
    if candidate_keys:
        try:
            scores, _ = rerank_similarity(
                original_question,
                [
                    str(ranking_text(chunks_by_key[key]))
                    for key in candidate_keys
                ],
            )
            scores_by_key = {
                key: float(score)
                for key, score in zip(candidate_keys, scores, strict=True)
            }
            semantic_ranks = {
                key: rank
                for rank, key in enumerate(
                    sorted(
                        candidate_keys,
                        key=lambda item: (-scores_by_key[item], item),
                    ),
                    start=1,
                )
            }
        except Exception as error:
            ranking_source = "fallback"
            ranking_method = "subquery_rank_rrf"
            fallback_reason = f"原问题统一重排失败（{type(error).__name__}）"
            logger.warning(
                "Parallel final reranking failed: %s",
                type(error).__name__,
            )
            rankings = {
                step.id: subquery_keys.get(step.id, [])
                for step, _result in subquery_results
            }
            active_rankings = {
                name: keys for name, keys in rankings.items() if keys
            }
            if active_rankings:
                # 使用rrf把子查询内部的排名融合
                scores_by_key = reciprocal_rank_fusion(
                    active_rankings,
                    {
                        name: 1.0 / len(active_rankings)
                        for name in active_rankings
                    },
                    k=options.rrf_k,
                ).scores
            else:
                scores_by_key = {key: 0.0 for key in candidate_keys}
    else:
        ranking_source = "skipped"

    for key, chunk in chunks_by_key.items():
        # 约束，判断chunk内容是否满足问题里的硬性约束和条件，比如时间、地点
        chunk["constraint_compatibility"] = evaluate_constraint_compatibility(
            original_question,
            str(
                chunk.get("content_with_weight")
                or chunk.get("content_ltks")
                or ""
            ),
            document_name=str(chunk.get("docnm_kwd") or ""),
        ).to_dict()
        chunk["rerank_score"] = (
            scores_by_key[key] if ranking_source == "model" else None
        )
        chunk["similarity"] = chunk["rerank_score"]
        chunk["final_ranking"] = {
            "method": ranking_method,
            "semantic_reranker_rank": semantic_ranks.get(key),
            "final_rerank_score": chunk["rerank_score"],
            "fallback_rrf_score": (
                scores_by_key[key] if ranking_source == "fallback" else None
            ),
        }

    ranked_keys = sorted(
        candidate_keys,
        key=lambda key: (
            chunks_by_key[key]["constraint_compatibility"]["conflict_count"],
            _measurement_compatibility_priority(chunks_by_key[key]),
            _approximate_figure_penalty(
                original_question,
                chunks_by_key[key],
            ),
            -scores_by_key[key],
            key,
        ),
    )

    # Reserve membership, not fixed positions, for the strongest results from
    # every independent facet. A shared chunk counts toward every subquery in
    # which it appeared, while the common reranker still determines ordering.
    per_subquery_quota = min(
        _parallel_coverage_per_subquery(),
        max(1, options.page_size // max(1, len(subquery_results))),
    )
    reserved_keys: set[str] = set()
    coverage_by_subquery: dict[str, list[str]] = {}
    # 为每个子查询保留最强结果的名额，保证并行检索的时候，没有任何一个子查询的结果被淹没
    for step, _result in subquery_results:
        coverage = []
        requested_locators = extract_locators(step.query)
        def locator_priority(key):
            if not requested_locators:
                return 0
            chunk = chunks_by_key[key]
            if requested_locators & extract_locators(str(chunk.get("locator_kwd") or "")):
                return 0
            if requested_locators & extract_locators(str(chunk.get("content_with_weight") or "")):
                return 1
            return 2
        # 子查询的chunk经过重排后的顺序
        coverage_candidates = sorted(
            subquery_keys.get(step.id, []),
            key=lambda key: (
                locator_priority(key),
                _measurement_compatibility_priority(chunks_by_key[key]),
                # 子查询可能省略“多少”等措辞，沿用原问题判断图表估值是否降级。
                _approximate_figure_penalty(original_question, chunks_by_key[key]),
                subquery_keys.get(step.id, []).index(key),
            ),
        )
        has_measurement_match = any(
            _measurement_compatibility_priority(chunks_by_key[key]) == 0
            for key in coverage_candidates
        )
        for key in coverage_candidates:
            if len(coverage) >= per_subquery_quota:
                break
            if requested_locators and locator_priority(key) == 2:
                continue
            measurement_priority = _measurement_compatibility_priority(
                chunks_by_key[key]
            )
            if measurement_priority == 2:
                continue
            if has_measurement_match and measurement_priority != 0:
                continue
            if _duplicates_numeric_fact(key, reserved_keys, chunks_by_key):
                continue
            if (
                key not in reserved_keys
                and len(reserved_keys) >= options.page_size
            ):
                break
            reserved_keys.add(key)
            coverage.append(key)
        # 每个子查询最终保留的chunk id列表，保证每个子查询都有一定的覆盖率
        coverage_by_subquery[step.id] = coverage

    selected_keys = set(reserved_keys)
    for key in ranked_keys:
        if len(selected_keys) >= options.page_size:
            break
        if chunks_by_key[key]["constraint_compatibility"]["conflict_count"]:
            continue
        # 跳过和已选chunk重复数值事实的候选
        if _duplicates_numeric_fact(key, selected_keys, chunks_by_key):
            continue
        selected_keys.add(key)
    ordered_keys = [key for key in ranked_keys if key in selected_keys]
    ordered_keys.extend(key for key in ranked_keys if key not in selected_keys)

    coverage_owners: dict[str, list[str]] = {}
    for subquery_id, keys in coverage_by_subquery.items():
        for key in keys:
            coverage_owners.setdefault(key, []).append(subquery_id)
    for key, chunk in chunks_by_key.items():
        chunk["final_ranking"]["coverage_reserved"] = key in reserved_keys
        chunk["final_ranking"]["coverage_subquery_ids"] = (
            coverage_owners.get(key, [])
        )

    start = (options.page - 1) * options.page_size
    end = options.page * options.page_size
    paged_chunks = [chunks_by_key[key] for key in ordered_keys[start:end]]
    subquery_result_values = [
        result for _step, result in subquery_results
    ]
    initial_retrieval_ms = _sum_stage_latency(
        subquery_result_values,
        "initial_retrieval",
    )
    rerank_ms = _sum_stage_latency(
        subquery_result_values,
        "rerank",
    ) + (time.perf_counter() - merge_started) * 1000

    return {
        "total": sum(
            int(result.get("total") or 0)
            for _step, result in subquery_results
        ),
        "chunks": paged_chunks,
        "doc_aggs": _document_aggregations(paged_chunks),
        "stage_latency_ms": {
            "initial_retrieval": initial_retrieval_ms,
            "rerank": rerank_ms,
        },
        "retrieval_fusion": {
            "method": "parallel_original_question_rerank",
            "coverage_first": False,
            "final_rerank": {
                "query": original_question,
                "source": ranking_source,
                "method": ranking_method,
                "candidate_count": len(candidate_keys),
                "fallback_reason": fallback_reason,
                "coverage_window_size": options.page_size,
                "coverage_per_subquery": per_subquery_quota,
                "coverage_chunk_ids_by_subquery": {
                    subquery_id: [
                        str(chunks_by_key[key].get("chunk_id") or "")
                        for key in keys
                    ]
                    for subquery_id, keys in coverage_by_subquery.items()
                },
                "candidate_diagnostics": [{
                    "chunk_id": str(chunks_by_key[key].get("chunk_id") or key),
                    "merged_rank": rank, "selected": key in ordered_keys[start:end],
                    "coverage_reserved": key in reserved_keys,
                    "conflict_count": chunks_by_key[key]["constraint_compatibility"]["conflict_count"],
                    "drop_reason": None if key in ordered_keys[start:end] else
                        "constraint_conflict" if chunks_by_key[key]["constraint_compatibility"]["conflict_count"] else "merge_selection_or_page_limit",
                } for rank, key in enumerate(ranked_keys[:200], 1)],
                "candidate_diagnostics_truncated": len(ranked_keys) > 200,
            },
            "subqueries": [
                {
                    "subquery_id": step.id,
                    "query": step.query,
                    "retrieval_fusion": result.get("retrieval_fusion") or {},
                    "returned_chunk_count": len(result.get("chunks") or []),
                }
                for step, result in subquery_results
            ],
        },
        "parallel_retrieval": {
            "applied": True,
            "coverage_per_subquery": per_subquery_quota,
            "steps": [
                {"id": step.id, "query": step.query}
                for step, _result in subquery_results
            ],
        },
    }

# 独立查询有界并发执行，单个失败不会抹去其他查询的证据。
def _execute_parallel_retrieval(
    *,
    dealer: Dealer,
    index_names: str | list[str],
    original_question: str,
    retrieval_plan: RetrievalPlan,
    options: Dealer.RetrievalOptions,
) -> tuple[dict[str, Any] | None, str | None]:
    if retrieval_plan.mode != "parallel":
        return None, "查询计划不是并行检索"
    if len(retrieval_plan.subqueries) < 2:
        return None, "并行查询计划缺少可执行的子查询"

    subquery_page_size = min(
        options.rerank_candidate_size,
        max(
            (
                options.page
                * options.page_size
                * DEFAULT_PARALLEL_CANDIDATE_MULTIPLIER
            ),
            _parallel_coverage_per_subquery(),
        ),
    )
    subquery_options = replace(
        options,
        page=1,
        page_size=subquery_page_size,
    )
    subquery_results = []
    def retrieve_step(step):
        with observation_context(node_id=step.id, goal_ids=step.goal_ids):
            return dealer.retrieval(question=step.query, keyword_question=step.query,
                                    tenant_ids=index_names, options=subquery_options)
    results = run_independent({step.id: lambda step=step: retrieve_step(step)
                               for step in retrieval_plan.subqueries})
    failures = []
    for step in retrieval_plan.subqueries:
        result = results[step.id]
        if isinstance(result, Exception):
            failure = record_execution_node(step.id, "budget_exhausted" if isinstance(result, QueryBudgetExceeded) else "failed",
                                           error=result, goal_ids=step.goal_ids, end_reason="retrieval_failed", candidate_count=0)
            failures.append({"id": step.id, "error": type(result).__name__, **failure})
            result = {"total": 0, "chunks": []}
        else:
            record_execution_node(step.id, "succeeded" if result.get("chunks") else "empty",
                                  goal_ids=step.goal_ids, candidate_count=len(result.get("chunks") or []),
                                  end_reason="retrieved" if result.get("chunks") else "empty_results")
        subquery_results.append((step, result))
    if len(failures) == len(retrieval_plan.subqueries):
        return None, "全部独立子查询执行失败"
    merged = _merge_parallel_subqueries(
            original_question=original_question,
            subquery_results=subquery_results,
            options=options,
    )
    if failures:
        merged["parallel_retrieval"]["failed_nodes"] = failures
    return merged, None


def _merge_sequential_hops(
    *,
    original_question: str,
    hop_results: list[tuple[QuerySubqueryPlan, str, dict[str, Any]]],
    bridge: BridgeExtraction,
    options: Dealer.RetrievalOptions,
) -> dict[str, Any]:
    merge_started = time.perf_counter()
    chunks_by_key: dict[str, dict[str, Any]] = {}
    hop_keys: list[list[str]] = []
    representative_keys: list[str] = []

    for hop_number, (step, query, result) in enumerate(hop_results, start=1):
        current_keys = []
        for hop_rank, source_chunk in enumerate(
            result.get("chunks") or [],
            start=1,
        ):
            key = _chunk_key(source_chunk)
            current_keys.append(key)
            hop_metadata = {
                "subquery_id": step.id,
                "hop": hop_number,
                "hop_rank": hop_rank,
                "query": query,
                "ranking": dict(source_chunk.get("final_ranking") or {}),
                "rerank_score": source_chunk.get("rerank_score"),
                "rrf_score": source_chunk.get("rrf_score"),
                "constraint_compatibility": dict(
                    source_chunk.get("constraint_compatibility") or {}
                ),
            }
            if key not in chunks_by_key:
                chunk = dict(source_chunk)
                chunk["sequential_subqueries"] = [hop_metadata]
                chunks_by_key[key] = chunk
            else:
                chunks_by_key[key]["sequential_subqueries"].append(
                    hop_metadata
                )
        hop_keys.append(current_keys)
        # Keep the validated bridge rather than an unrelated first-hop leader.
        # Later hops retain their best unique local result, so background
        # evidence is not lost merely because it scores lower for the full query.
        representative = None
        if hop_number == 1 and 1 <= bridge.source_chunk_index <= len(current_keys):
            representative = current_keys[bridge.source_chunk_index - 1]
        if representative is None:
            representative = next(
                (key for key in current_keys if key not in representative_keys),
                current_keys[0] if current_keys else None,
            )
        if representative is not None:
            representative_keys.append(representative)

    candidate_keys = list(chunks_by_key)
    scores_by_key: dict[str, float] = {}
    semantic_ranks: dict[str, int] = {}
    ranking_source = "model"
    ranking_method = "original_question_rerank"
    fallback_reason = None
    if candidate_keys:
        try:
            scores, _ = rerank_similarity(
                original_question,
                [
                    str(ranking_text(chunks_by_key[key]))
                    for key in candidate_keys
                ],
            )
            scores_by_key = {
                key: float(score) for key, score in zip(candidate_keys, scores, strict=True)
            }
            semantic_ranks = {
                key: rank for rank, key in enumerate(
                    sorted(candidate_keys, key=lambda key: (-scores_by_key[key], key)),
                    start=1,
                )
            }
        except Exception as error:
            # Keep both hops if the additional rerank request fails. Only local
            # ranks, not raw scores from different questions, are comparable.
            ranking_source = "fallback"
            ranking_method = "hop_rank_rrf"
            fallback_reason = f"原问题统一重排失败（{type(error).__name__}）"
            logger.warning("Sequential final reranking failed: %s", type(error).__name__)
            rankings = {str(i): keys for i, keys in enumerate(hop_keys)}
            scores_by_key = reciprocal_rank_fusion(
                rankings,
                {name: 1.0 / len(rankings) for name in rankings},
                k=options.rrf_k,
            ).scores
    else:
        ranking_source = "skipped"

    for key, chunk in chunks_by_key.items():
        chunk["constraint_compatibility"] = evaluate_constraint_compatibility(
            original_question,
            str(chunk.get("content_with_weight") or chunk.get("content_ltks") or ""),
            document_name=str(chunk.get("docnm_kwd") or ""),
        ).to_dict()
        # Per-hop scores and ranks remain in sequential_subqueries. The public
        # rerank score now refers to the original question, or is unknown on failure.
        chunk["rerank_score"] = scores_by_key[key] if ranking_source == "model" else None
        chunk["similarity"] = chunk["rerank_score"]
        chunk["final_ranking"] = {
            "method": ranking_method,
            "semantic_reranker_rank": semantic_ranks.get(key),
            "final_rerank_score": chunk["rerank_score"],
            "fallback_rrf_score": scores_by_key[key] if ranking_source == "fallback" else None,
        }

    ranked_keys = sorted(
        candidate_keys,
        key=lambda key: (
            chunks_by_key[key]["constraint_compatibility"]["conflict_count"],
            -scores_by_key[key],
            key,
        ),
    )
    # Coverage determines membership of the first page, never fixed positions.
    # Sort the selected evidence by the common query's ranking; append the rest
    # once so later pages cannot repeat a reserved first-page result.
    reserved_keys = set(representative_keys)
    if len(reserved_keys) > options.page_size:
        reserved_keys = set()  # A one-slot page cannot guarantee two-hop coverage.
    selected_keys = set(reserved_keys)
    for key in ranked_keys:
        if len(selected_keys) >= options.page_size:
            break
        selected_keys.add(key)
    ordered_keys = [key for key in ranked_keys if key in selected_keys]
    ordered_keys.extend(key for key in ranked_keys if key not in selected_keys)
    for key, chunk in chunks_by_key.items():
        chunk["final_ranking"]["coverage_reserved"] = key in reserved_keys

    start = (options.page - 1) * options.page_size
    end = options.page * options.page_size
    paged_chunks = [chunks_by_key[key] for key in ordered_keys[start:end]]
    hop_result_values = [
        result for _step, _query, result in hop_results
    ]
    initial_retrieval_ms = _sum_stage_latency(
        hop_result_values,
        "initial_retrieval",
    )
    rerank_ms = _sum_stage_latency(
        hop_result_values,
        "rerank",
    ) + (time.perf_counter() - merge_started) * 1000

    return {
        "total": sum(
            int(result.get("total") or 0)
            for _step, _query, result in hop_results
        ),
        "chunks": paged_chunks,
        "doc_aggs": _document_aggregations(paged_chunks),
        "stage_latency_ms": {
            "initial_retrieval": initial_retrieval_ms,
            "rerank": rerank_ms,
        },
        "retrieval_fusion": {
            "method": "sequential_original_question_rerank",
            "coverage_first": False,
            "final_rerank": {
                "query": original_question,
                "source": ranking_source,
                "method": ranking_method,
                "candidate_count": len(candidate_keys),
                "fallback_reason": fallback_reason,
                "coverage_window_size": options.page_size,
                "coverage_chunk_ids": [
                    str(chunks_by_key[key].get("chunk_id") or "")
                    for key in ordered_keys if key in reserved_keys
                ],
            },
            "hops": [
                {
                    "subquery_id": step.id,
                    "hop": hop_number,
                    "query": query,
                    "retrieval_fusion": result.get("retrieval_fusion") or {},
                    "returned_chunk_count": len(result.get("chunks") or []),
                }
                for hop_number, (step, query, result) in enumerate(
                    hop_results,
                    start=1,
                )
            ],
        },
        "sequential_retrieval": {
            "applied": True,
            "bridge": bridge.to_dict(),
            "steps": [
                {
                    "id": step.id,
                    "hop": hop_number,
                    "query": query,
                    "depends_on": list(step.depends_on),
                    "inherit_document_scope": step.inherit_document_scope,
                }
                for hop_number, (step, query, _result) in enumerate(
                    hop_results,
                    start=1,
                )
            ],
        },
    }


def _execute_sequential_retrieval(
    *,
    dealer: Dealer,
    index_names: str | list[str],
    original_question: str,
    first_keyword_question: str,
    retrieval_plan: RetrievalPlan,
    options: Dealer.RetrievalOptions,
) -> tuple[dict[str, Any] | None, str | None, dict[str, float]]:
    execution_started = time.perf_counter()
    completed_hop_results: list[dict[str, Any]] = []

    def finish(
        result: dict[str, Any] | None,
        reason: str | None,
    ) -> tuple[dict[str, Any] | None, str | None, dict[str, float]]:
        source_results = [result] if result is not None else completed_hop_results
        initial_retrieval_ms = _sum_stage_latency(
            source_results,
            "initial_retrieval",
        )
        rerank_ms = _sum_stage_latency(source_results, "rerank")
        total_ms = (time.perf_counter() - execution_started) * 1000
        stage_latency_ms = {
            "initial_retrieval": initial_retrieval_ms,
            "rerank": rerank_ms,
            "multi_hop": max(
                0.0,
                total_ms - initial_retrieval_ms - rerank_ms,
            ),
        }
        if result is not None:
            result["stage_latency_ms"] = stage_latency_ms
        return result, reason, stage_latency_ms

    if retrieval_plan.mode != "sequential":
        return finish(None, "查询计划不是递进式检索")
    if len(retrieval_plan.subqueries) != 2:
        return finish(None, "递进式查询计划必须包含两步")

    first_step, second_step = retrieval_plan.subqueries
    hop_page_size = min(
        options.rerank_candidate_size,
        max(options.page * options.page_size, 2),
    )
    hop_options = replace(options, page=1, page_size=hop_page_size)

    try:
        with observation_context(node_id=first_step.id, goal_ids=first_step.goal_ids):
            first_result = dealer.retrieval(
                question=first_step.query,
                keyword_question=(first_keyword_question if first_step.query == original_question else first_step.query),
                tenant_ids=index_names, options=hop_options,
            )
    except Exception as error:
        record_execution_node(first_step.id, "budget_exhausted" if isinstance(error, QueryBudgetExceeded) else "failed",
                              error=error, goal_ids=first_step.goal_ids, candidate_count=0, end_reason="retrieval_failed")
        record_execution_node(second_step.id, "blocked", goal_ids=second_step.goal_ids,
                              end_reason="dependency_failed", depends_on=second_step.depends_on)
        raise
    record_execution_node(first_step.id, "succeeded" if first_result.get("chunks") else "empty",
                          goal_ids=first_step.goal_ids, candidate_count=len(first_result.get("chunks") or []),
                          end_reason="retrieved" if first_result.get("chunks") else "empty_results")
    completed_hop_results.append(first_result)
    def retain_first(reason, *, rule, error=None):
        record_execution_node(second_step.id, "blocked", error=error, goal_ids=second_step.goal_ids,
                              end_reason=rule, depends_on=second_step.depends_on, detail=reason[:500])
        first_result["sequential_retrieval"] = {
            "applied": False, "fallback_reason": reason,
            "completed_nodes": [first_step.id], "blocked_nodes": [second_step.id],
        }
        return finish(first_result, reason)

    with observation_context(node_id=first_step.id, goal_ids=first_step.goal_ids):
        bridge = extract_bridge_value(
            slot=str(first_step.output_slot or ""), original_question=original_question,
            first_query=first_step.query, chunks=first_result.get("chunks") or [],
        )
    if bridge is None:
        return retain_first("第一跳证据中没有找到可验证的桥接值", rule="dependency_has_no_grounded_value")

    try:
        second_query = resolve_query_template(
            second_step.query_template,
            {bridge.slot: bridge.value},
        )
        missing = missing_constraints(second_step.required_constraints, second_query)
        if missing:
            raise QueryValidationError("resolved_constraint_dropped", "第二跳解析后的查询丢失目标约束: " + ", ".join(missing),
                                       field="resolved_query", node_id=second_step.id)
    except ValueError as error:
        return retain_first(str(error), rule=getattr(error, "rule", "invalid_query_template"), error=error)

    second_options = hop_options
    if second_step.inherit_document_scope:
        if not bridge.document_id:
            return retain_first("第二跳要求继承文档范围，但第一跳证据缺少文档 ID", rule="missing_document_scope")
        second_options = replace(hop_options, doc_ids=[bridge.document_id])

    try:
        with observation_context(node_id=second_step.id, goal_ids=second_step.goal_ids):
            second_result = dealer.retrieval(
                question=second_query, keyword_question=second_query,
                tenant_ids=index_names, options=second_options,
            )
    except Exception as error:
        record_execution_node(second_step.id, "budget_exhausted" if isinstance(error, QueryBudgetExceeded) else "failed",
                              error=error, goal_ids=second_step.goal_ids, end_reason="retrieval_failed", candidate_count=0)
        second_result = {"chunks": [], "total": 0, "execution_error": type(error).__name__}
    else:
        record_execution_node(second_step.id, "succeeded" if second_result.get("chunks") else "empty",
                              goal_ids=second_step.goal_ids, candidate_count=len(second_result.get("chunks") or []),
                              end_reason="retrieved" if second_result.get("chunks") else "empty_results")
    completed_hop_results.append(second_result)
    result = _merge_sequential_hops(
        original_question=original_question,
        hop_results=[
            (first_step, first_step.query, first_result),
            (second_step, second_query, second_result),
        ],
        bridge=bridge,
        options=options,
    )
    return finish(result, None)


def _clarification_result(request: QueryRequest, message: str) -> dict[str, Any]:
    # A clarification is an execution skip, not a failed retrieval attempt.
    record_observation("plan_execution", "execution", "skipped", mode="clarify",
                       execution_success=None, end_reason="clarification_required")
    return {
        "total": 0, "chunks": [], "chunks_before_sufficiency": [], "doc_aggs": [],
        "response_mode": "clarify", "clarification": message,
        "query_contract": request.to_dict(),
        "evidence_sufficiency": {
            "sufficient": False, "source": "rule", "reason": "查询需要澄清",
            "missing_requirements": [message], "supporting_chunk_indices": [],
            "evaluated_chunk_count": 0, "evaluated_chunk_ids": [],
        },
        "query_runtime": current_runtime().snapshot() if current_runtime() else {},
    }


def _build_goal_repair_queries(goals, chunks, question):
    """把缺失目标转成补查词，同时记录每条查询属于哪个目标。"""
    queries = {}
    query_owners = {}
    # 已找到的图表不重复查；未知对应关系只补齐尚未找到的候选图表。
    present_locators = None
    for goal in goals:
        if goal.candidate_locators:
            if present_locators is None:
                present_locators = set().union(*(
                    extract_locators(str(chunk.get("locator_kwd") or ""))
                    for chunk in chunks
                ))
            for index, locator in enumerate(goal.candidate_locators):
                if extract_locators(locator) <= present_locators:
                    continue
                query_id = f"{goal.id}_{index + 1}"
                queries[query_id] = " ".join([
                    locator, "完整内容 文字及视觉事实", *shared_scope_terms(question),
                ])
                query_owners[query_id] = goal.id
        else:
            # 普通目标直接使用已有描述和限定条件，不再调用模型编写查询。
            queries[goal.id] = " ".join(dict.fromkeys([goal.text, *goal.constraints]))
            query_owners[goal.id] = goal.id
    return queries, query_owners


def _repair_missing_goals(result, decision, *, question, analysis, dealer, index_names, options):
    """证据不足时只补查一轮，返回合并后的结果和再次核验的判定。

    result 是已取回的片段，decision 是首次充分性判定，analysis.goals
    是用户需求清单。此函数更新 result；不负责最终拒答或清空片段。
    """
    # 1. 确认确实需要补查。核验器故障不能解释为“资料里缺少事实”。
    if decision.sufficient or not analysis.goals:
        return result, decision
    if not _environment_flag("RAG_QUERY_REPAIR_ENABLED", True):
        return result, decision
    if decision.source in {"fallback", "disabled"}:
        return result, decision

    # 2. 找出没有被支持的目标。零片段时，所有目标都视为缺失。
    missing_goal_ids = {
        item["goal_id"]
        for item in decision.goal_coverage
        if item["status"] != "supported"
    }
    existing_chunks = result.get("chunks") or []
    if not existing_chunks:
        missing_goal_ids = {goal.id for goal in analysis.goals}
    # 依赖目标需要先取得桥接值，不能直接当作独立查询补查。
    repairable_goals = [
        goal for goal in analysis.goals
        if goal.id in missing_goal_ids and not goal.depends_on
    ]
    if not repairable_goals:
        return result, decision

    # 3. 为每个缺口准备搜索词；一个图表目标可能对应多条候选查询。
    queries, query_owners = _build_goal_repair_queries(
        repairable_goals, existing_chunks, question,
    )
    if not queries:
        return result, decision
    existing_chunk_ids = {_chunk_key(chunk) for chunk in existing_chunks}

    # 4. 并发补查。候选数可大于初次 top-k，但受重排候选上限约束。
    repair_page_size = min(options.rerank_candidate_size, max(options.page_size, 10))

    def retrieve_repair_query(query_id, query):
        with observation_context(
            node_id="repair_" + query_id,
            goal_ids=[query_owners[query_id]],
            source="repair",
        ):
            return dealer.retrieval(
                question=query,
                keyword_question=query,
                tenant_ids=index_names,
                options=replace(options, page=1, page_size=repair_page_size),
            )

    # 默认参数绑定当前查询，避免循环中的 lambda 全部使用最后一条查询。
    repair_tasks = {
        query_id: lambda query_id=query_id, query=query: retrieve_repair_query(query_id, query)
        for query_id, query in queries.items()
    }
    repair_results = run_independent(repair_tasks)

    # 5. 失败分支只记录诊断，成功分支仍可合并；按片段 ID 统计新增证据。
    successful_repairs = []
    failed_query_ids = []
    for query_id, repair_result in repair_results.items():
        if isinstance(repair_result, Exception):
            failed_query_ids.append(query_id)
            continue
        repair_step = QuerySubqueryPlan(
            id=query_id, query=queries[query_id], goal_ids=[query_owners[query_id]],
        )
        successful_repairs.append((repair_step, repair_result))

    repaired_chunk_ids = {
        _chunk_key(chunk)
        for _, repair_result in successful_repairs
        for chunk in repair_result.get("chunks") or []
    }
    new_chunk_ids = repaired_chunk_ids - existing_chunk_ids
    repair_diagnostics = {
        "attempted_goal_ids": list(dict.fromkeys(query_owners.values())),
        "queries": queries,
        "new_candidate_count": len(new_chunk_ids),
        "failed_goal_ids": list(dict.fromkeys(query_owners[key] for key in failed_query_ids)),
        "failed_query_ids": failed_query_ids,
        "stop_reason": "no_new_evidence" if not new_chunk_ids else "repair_limit",
    }
    result["query_repair"] = repair_diagnostics
    if not new_chunk_ids:
        return result, decision

    # 6. 保留原有片段，并合并、去重、重排补查结果。合并容量最多 20 条；
    # 后面的表格上下文展开仍可能调整片段内容或数量。
    merged_page_size = min(20, max(
        options.page_size, len(existing_chunk_ids) + len(new_chunk_ids),
    ))
    retained_step = QuerySubqueryPlan(id="retained", query=question)
    merged_result = _merge_parallel_subqueries(
        original_question=question,
        subquery_results=[(retained_step, result), *successful_repairs],
        options=replace(options, page=1, page_size=merged_page_size),
    )
    merged_chunks = expand_table_contexts(merged_result["chunks"], question)

    # 7. 用完整问题重新核验全部目标，不只核验本轮补查词；不再递归补查。
    final_decision = check_evidence_sufficiency(
        question, merged_chunks, goals=analysis.goals,
    )
    result["chunks"] = merged_chunks
    result["chunks_before_sufficiency"] = list(merged_chunks)
    result["total"] = merged_result["total"]
    result["total_before_evidence_sufficiency"] = merged_result["total"]
    result["doc_aggs"] = merged_result["doc_aggs"]
    repair_diagnostics["stop_reason"] = (
        "goals_satisfied" if final_decision.sufficient else "repair_limit"
    )
    return result, final_decision


@dataclass(slots=True)
class _PlanResult:
    result: dict[str, Any]
    prior_initial_retrieval_ms: float = 0.0
    prior_rerank_ms: float = 0.0
    multi_hop_ms: float | None = None


def _resolve_cached_request(question: str, request: QueryRequest | None) -> QueryRequest:
    """Resolve conversation context once per request-local runtime."""
    request = request or QueryRequest.from_user(question)
    runtime = current_runtime()
    key = ("resolved_request", request)
    if key not in runtime.cache:
        runtime.cache[key] = resolve_request(request)
    return runtime.cache[key]


def _prepare_cached_request(question: str):
    """Reuse query analysis and its executable plan within this request."""
    runtime = current_runtime()
    key = ("prepared_request", question)
    if key not in runtime.cache:
        runtime.cache[key] = prepare_retrieval_request(question)
    return runtime.cache[key]


def _run_graph_plan(dealer, index_names, question, plan, options):
    with observe_plan_execution(
        "graph", [step.id for step in plan.subqueries], source=plan.source,
    ) as execution:
        nodes = execute_graph(
            plan.subqueries, original_question=question,
            retrieve=lambda query, doc_ids: dealer.retrieval(
                question=query, keyword_question=query, tenant_ids=index_names,
                options=replace(
                    options, page=1,
                    page_size=min(options.rerank_candidate_size, options.page_size * 2),
                    doc_ids=doc_ids,
                ),
            ),
        )
        result = _merge_parallel_subqueries(
            original_question=question,
            subquery_results=[
                (replace(node.step, query=node.query), node.result)
                for node in nodes
            ],
            options=options,
        )
        result["retrieval_graph"] = {
            "applied": True,
            "nodes": [node.diagnostic() for node in nodes],
        }
    return result, execution.summary


def _run_parallel_plan(dealer, index_names, question, plan, options):
    try:
        with observe_plan_execution(
            "parallel", [step.id for step in plan.subqueries], source=plan.source,
        ) as execution:
            result, fallback_reason = _execute_parallel_retrieval(
                dealer=dealer, index_names=index_names, original_question=question,
                retrieval_plan=plan, options=options,
            )
    except Exception as error:
        logger.warning(
            "Parallel retrieval failed; falling back to single query: %s",
            type(error).__name__,
        )
        return None, f"并行检索执行失败（{type(error).__name__}）", execution.summary
    return result, fallback_reason, execution.summary


def _run_sequential_plan(dealer, index_names, question, keyword_question, plan, options):
    try:
        with observe_plan_execution(
            "sequential", [step.id for step in plan.subqueries], source=plan.source,
        ) as execution:
            result, fallback_reason, stage_latency = _execute_sequential_retrieval(
                dealer=dealer, index_names=index_names, original_question=question,
                first_keyword_question=keyword_question, retrieval_plan=plan, options=options,
            )
    except Exception as error:
        logger.warning(
            "Sequential retrieval failed; falling back to single query: %s",
            type(error).__name__,
        )
        return None, f"递进式检索执行失败（{type(error).__name__}）", None, execution.summary
    return result, fallback_reason, stage_latency, execution.summary


def _run_single_plan(dealer, index_names, question, keyword_question, options, *, source):
    try:
        with observe_plan_execution("single", ["single"], source=source) as execution:
            try:
                with observation_context(node_id="single"):
                    result = dealer.retrieval(
                        question=question, keyword_question=keyword_question,
                        tenant_ids=index_names, options=options,
                    )
            except Exception as error:
                record_execution_node(
                    "single", "budget_exhausted" if isinstance(error, QueryBudgetExceeded) else "failed",
                    error=error, end_reason="retrieval_failed", candidate_count=0,
                )
                raise
            record_execution_node(
                "single", "succeeded" if result.get("chunks") else "empty",
                candidate_count=len(result.get("chunks") or []),
                end_reason="retrieved" if result.get("chunks") else "empty_results",
            )
    except QueryBudgetExceeded:
        result = {"total": 0, "chunks": [], "doc_aggs": [], "budget_exhausted": True}
    return result, execution.summary


def _execute_retrieval_plan(
    dealer, index_names, question, keyword_question, route, plan, options,
) -> _PlanResult:
    """Run the requested mode, then use single retrieval if it cannot produce a result."""
    result = None
    attempts = []
    prior_initial_retrieval_ms = 0.0
    prior_rerank_ms = 0.0
    multi_hop_ms = None
    parallel_fallback_reason = None
    sequential_fallback_reason = None
    parallel_enabled = _environment_flag("RAG_PARALLEL_RETRIEVAL_ENABLED", True)
    sequential_enabled = _environment_flag("RAG_SEQUENTIAL_RETRIEVAL_ENABLED", True)
    graph_enabled = (
        _environment_flag("RAG_QUERY_GRAPH_ENABLED", True)
        and sequential_enabled
        and parallel_enabled
    )

    if plan.mode == "graph" and graph_enabled:
        result, summary = _run_graph_plan(dealer, index_names, question, plan, options)
        attempts.append(summary)
    elif plan.mode == "parallel":
        if parallel_enabled:
            result, parallel_fallback_reason, summary = _run_parallel_plan(
                dealer, index_names, question, plan, options,
            )
            attempts.append(summary)
        else:
            parallel_fallback_reason = "并行检索已关闭"
    elif plan.mode == "sequential":
        if sequential_enabled:
            result, sequential_fallback_reason, stage_latency, summary = _run_sequential_plan(
                dealer, index_names, question, keyword_question, plan, options,
            )
            attempts.append(summary)
            if stage_latency is not None:
                multi_hop_ms = stage_latency["multi_hop"]
                if result is None:
                    # Preserve work already spent on the first hop before falling back.
                    prior_initial_retrieval_ms = stage_latency["initial_retrieval"]
                    prior_rerank_ms = stage_latency["rerank"]
        else:
            sequential_fallback_reason = "递进式检索已关闭"

    if result is None:
        if plan.mode != "single" and not attempts:
            skipped = {
                "mode": plan.mode, "status": "skipped", "execution_success": None,
                "end_reason": "execution_disabled", "source": plan.source,
            }
            attempts.append(skipped)
            record_observation("plan_execution", "execution", **skipped)
        # 执行单查询调用
        result, summary = _run_single_plan(
            dealer, index_names, question, keyword_question, options,
            source="fallback" if attempts else plan.source,
        )
        attempts.append(summary)
        if plan.mode == "graph":
            result["retrieval_graph"] = {
                "applied": False,
                "fallback_reason": "依赖图或其所需的并行/递进执行已关闭",
            }
        elif plan.mode == "sequential":
            result["sequential_retrieval"] = {
                "applied": False, "fallback_reason": sequential_fallback_reason,
            }
        elif plan.mode == "parallel":
            result["parallel_retrieval"] = {
                "applied": False, "fallback_reason": parallel_fallback_reason,
            }

    result["plan_execution"] = {
        "requested_mode": route.mode,
        "planned_mode": plan.mode,
        "executed_mode": attempts[-1]["mode"],
        "fallback_applied": plan.source == "fallback" or len(attempts) > 1,
        "requested_plan_execution_success": (
            False if route.mode != plan.mode else attempts[0].get("execution_success")
        ),
        "plan_execution_success": attempts[0].get("execution_success"),
        "final_execution_success": attempts[-1].get("execution_success"),
        "attempts": attempts,
    }
    return _PlanResult(result, prior_initial_retrieval_ms, prior_rerank_ms, multi_hop_ms)


def _verify_retrieved_evidence(result, *, question, analysis, dealer, index_names, options):
    """统一执行证据门控：首次核验 → 按需补查 → 根据最终判定放行或拒答。

    返回 (result, gate_ms, repair_ms)。gate_ms 只记录首次核验耗时；
    repair_ms 包含补查及再次核验。门控前快照用于评测和失败排查。
    """
    # 1. 补齐表格的表头等上下文，再保存快照。拒答时只清空回答用的
    # chunks，仍保留 chunks_before_sufficiency，便于检查检索到了什么。
    initial_chunks = list(result.get("chunks") or [])
    verification_chunks = expand_table_contexts(initial_chunks, question)
    result["chunks"] = verification_chunks
    result["chunks_before_sufficiency"] = list(verification_chunks)
    result["total_before_evidence_sufficiency"] = result.get("total", 0)

    # 2. 所有检索分支已经合并完毕，此时才检查能否回答完整用户问题。
    gate_started = time.perf_counter()
    gate_options = {"goals": analysis.goals} if analysis.goals else {}
    initial_decision = check_evidence_sufficiency(
        question, verification_chunks, **gate_options,
    )
    record_observation(
        "evidence_gate",
        "decision",
        "sufficient" if initial_decision.sufficient else "insufficient",
        source=initial_decision.source,
        end_reason="initial_evidence_check",
        detail=initial_decision.reason[:500],
    )
    gate_ms = (time.perf_counter() - gate_started) * 1000

    # 3. 补查函数自行判断是否需要执行；已充分或无法安全补查时直接返回。
    # 若实际补查，快照会更新为合并后的证据，并返回再次核验的判定。
    repair_started = time.perf_counter()
    result, final_decision = _repair_missing_goals(
        result,
        initial_decision,
        question=question,
        analysis=analysis,
        dealer=dealer,
        index_names=index_names,
        options=options,
    )
    repair_ms = (time.perf_counter() - repair_started) * 1000

    # 4. 对外只保存最终判定。仍不足时不把片段交给回答阶段；
    # 保留核验原因和门控前快照，让上层能解释拒答并进行评测。
    result["evidence_sufficiency"] = final_decision.to_dict()
    result["evidence_sufficiency"]["enabled"] = _environment_flag(
        "RAG_EVIDENCE_SUFFICIENCY_ENABLED", True,
    )
    if not final_decision.sufficient:
        result["total"] = 0
        result["chunks"] = []
        result["doc_aggs"] = []
    return result, gate_ms, repair_ms


def _record_retrieval_latency(
    result, *, pipeline_started, query_transform_ms, execution, gate_ms, repair_ms,
):
    """Keep wall time separate from child work that can overlap under concurrency."""
    initial_retrieval_ms = (
        execution.prior_initial_retrieval_ms
        + _stage_latency(result, "initial_retrieval")
    )
    rerank_ms = execution.prior_rerank_ms + _stage_latency(result, "rerank")
    total_ms = (time.perf_counter() - pipeline_started) * 1000
    accounted_ms = (
        query_transform_ms + initial_retrieval_ms + rerank_ms
        + (execution.multi_hop_ms or 0.0) + gate_ms + repair_ms
    )
    result["stage_latency_ms"] = {
        "query_transform": round(query_transform_ms, 3),
        "initial_retrieval": round(initial_retrieval_ms, 3),
        "rerank": round(rerank_ms, 3),
        "multi_hop": round(execution.multi_hop_ms, 3) if execution.multi_hop_ms is not None else None,
        "gate": round(gate_ms, 3),
        "repair": round(repair_ms, 3),
        "other": round(max(0.0, total_ms - accounted_ms), 3),
        "total": round(total_ms, 3),
    }
    result["stage_latency_semantics"] = {
        "total": "wall_time", "initial_retrieval": "summed_child_work",
        "rerank": "summed_child_work_plus_merge", "repair": "wall_time_including_recheck",
        "additive": False,
    }


@bounded_query
def retrieve_raw_results(
    index_names: str | list[str],
    question: str,
    *,
    page: int = 1,
    page_size: int = DEFAULT_PAGE_SIZE,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    vector_similarity_weight: float = DEFAULT_VECTOR_SIMILARITY_WEIGHT,
    candidate_size: int = DEFAULT_RETRIEVAL_CANDIDATE_SIZE,
    rerank_candidate_size: int = DEFAULT_RERANK_CANDIDATE_SIZE,
    rrf_k: int = DEFAULT_RRF_K,
    final_reranker_weight: float = DEFAULT_FINAL_RERANKER_WEIGHT,
    final_rrf_k: int = DEFAULT_FINAL_RRF_K,
    request: QueryRequest | None = None,
) -> dict[str, Any]:
    """Run the KB query pipeline and return full evidence and diagnostics.

    Chat consumes compact chunks, while offline evaluation also needs the raw
    rankings, evidence snapshots, plan trace, and stage timings.
    """
    pipeline_started = time.perf_counter()
    options = Dealer.RetrievalOptions(
        kb_ids=None,
        vector_similarity_weight=vector_similarity_weight,
        similarity_threshold=similarity_threshold,
        candidate_size=candidate_size,
        rerank_candidate_size=rerank_candidate_size,
        rrf_k=rrf_k,
        final_reranker_weight=final_reranker_weight,
        final_rrf_k=final_rrf_k,
        page=page,
        page_size=page_size,
    )

    # 1. 解析会话指代并生成计划；需要澄清的问题不进入检索。
    request = _resolve_cached_request(question, request)
    # 进行了指代消解的用户查询，可能和原始问题不同。后续检索和证据充分性检查都使用解析后的查询。
    original_question = request.resolved_question
    if request.clarification:
        return _clarification_result(request, request.clarification)

    query_transform_started = time.perf_counter()
    prepared_request = _prepare_cached_request(original_question)
    query_analysis = prepared_request.analysis
    retrieval_route = prepared_request.route
    retrieval_plan = prepared_request.plan
    if retrieval_plan.mode == "clarify":
        return _clarification_result(request, retrieval_route.reason)

    if query_analysis.goals:
        options = replace(options, page_size=min(20, max(page_size, len(query_analysis.goals))))
    if retrieval_plan.mode == "single":
        rewritten_question, query_transform = _prepare_query_transform(original_question, query_analysis)
    else:
        # 多查询计划已有自己的子查询，无需再次执行单查询改写。
        rewritten_question = original_question
        query_transform = QueryExpansionResult.noop(
            original_question, reason="多查询计划已生成，跳过单查询改写",
        )
    query_transform_ms = (time.perf_counter() - query_transform_started) * 1000

    # 2. 执行计划。失败或功能关闭时，执行器负责记录并回退到单查询。
    dealer = _BudgetedDealer(get_retrieval_dealer(), transform_nodes=retrieval_plan.mode != "single")
    execution = _execute_retrieval_plan(
        dealer, index_names, original_question, query_transform.effective_query,
        retrieval_route, retrieval_plan, options,
    )

    # 3. 保存门控前证据，检查充分性，并在必要时补查缺口。
    result, gate_ms, repair_ms = _verify_retrieved_evidence(
        execution.result, question=original_question, analysis=query_analysis,
        dealer=dealer, index_names=index_names, options=options,
    )

    # 4. 附加查询决策与运行诊断，供 Agent 和离线评测复查。
    result["query_rewrite"] = {
        "original": original_question,
        "rewritten": rewritten_question,
        "changed": rewritten_question != original_question,
    }
    result["query_intent"] = query_analysis.to_dict()
    result["retrieval_route"] = retrieval_route.to_dict()
    result["retrieval_plan"] = retrieval_plan.to_dict()
    result["query_transform"] = query_transform.to_dict()
    result["query_contract"] = request.to_dict()
    result["query_runtime"] = current_runtime().snapshot() if current_runtime() else {}
    _record_retrieval_latency(
        result, pipeline_started=pipeline_started,
        query_transform_ms=query_transform_ms, execution=execution,
        gate_ms=gate_ms, repair_ms=repair_ms,
    )
    return result


def _optional_float(value: Any) -> float | None:
    """Convert NumPy and Python numeric values into JSON-safe floats."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_retrieved_chunk(chunk: dict[str, Any], rank: int) -> dict[str, Any]:
    """Build the compact, JSON-safe chunk representation used by callers."""
    document_name = str(chunk.get("docnm_kwd", "N/A"))
    document_name = document_name.replace("\\", "/").split("/")[-1]

    return {
        # ``id`` is retained for the existing citation prompt and frontend.
        "id": rank,
        **evidence_metadata(chunk),
        "rank": rank,
        "chunk_id": chunk.get("chunk_id", ""),
        "document_id": chunk.get("doc_id", "N/A"),
        "document_name": document_name,
        "content_with_weight": chunk.get("content_with_weight", "N/A"),
        "similarity": _optional_float(chunk.get("similarity")),
        "rerank_score": _optional_float(chunk.get("rerank_score")),
        "rrf_score": _optional_float(chunk.get("rrf_score")),
        "vector_similarity": _optional_float(
            chunk.get("vector_similarity")
        ),
        "term_similarity": _optional_float(chunk.get("term_similarity")),
        "retrieval_ranks": chunk.get("retrieval_ranks") or {},
        "retrieval_scores": chunk.get("retrieval_scores") or {},
        "rrf_contributions": chunk.get("rrf_contributions") or {},
        "constraint_compatibility": (
            chunk.get("constraint_compatibility") or {}
        ),
        "final_ranking": chunk.get("final_ranking") or {},
        "sequential_subqueries": chunk.get("sequential_subqueries") or [],
        "parallel_subqueries": chunk.get("parallel_subqueries") or [],
        "positions": chunk.get("positions") or [],
        "kb_id": chunk.get("kb_id", ""),
        "image_id": chunk.get("image_id", ""),
    }


def retrieve_content(
    indexNames: str,
    question: str,
    *,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> list[dict[str, Any]]:
    """Retrieve the top chunks for chat while retaining ranking metadata."""
    chunks, _diagnostics = retrieve_content_with_diagnostics(
        indexNames,
        question,
        page_size=page_size,
    )
    return chunks


def retrieve_content_with_diagnostics(
    indexNames: str,
    question: str,
    *,
    page_size: int = DEFAULT_PAGE_SIZE,
    request: QueryRequest | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return chat chunks plus bounded policy/retrieval diagnostics."""
    kwargs = {"request": request} if request is not None else {}
    results = retrieve_raw_results(indexNames, question, page_size=page_size, **kwargs)
    return format_retrieval_results(results)


def format_retrieval_results(results: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Shared result envelope for production tools and complete Agent evals."""
    chunks = [
        format_retrieved_chunk(chunk, rank)
        for rank, chunk in enumerate(results.get("chunks", []), start=1)
    ]
    diagnostics = {
        "query_contract": dict(results.get("query_contract") or {}),
        "query_transform": dict(results.get("query_transform") or {}),
        "query_runtime": dict(results.get("query_runtime") or {}),
        "plan_execution": dict(results.get("plan_execution") or {}),
        "retrieval_graph": dict(results.get("retrieval_graph") or {}),
        "query_repair": dict(results.get("query_repair") or {}),
        "response_mode": results.get("response_mode", "answer" if chunks else "insufficient"),
        "clarification": results.get("clarification"),
        "candidate_chunks": [format_retrieved_chunk(chunk, rank) for rank, chunk in enumerate(
            results.get("chunks_before_sufficiency") or [], start=1,
        )][:20],
        "evidence_sufficiency": dict(
            results.get("evidence_sufficiency") or {}
        ),
        "query_intent": dict(results.get("query_intent") or {}),
        "retrieval_route": dict(results.get("retrieval_route") or {}),
        "retrieval_plan": dict(results.get("retrieval_plan") or {}),
        "sequential_retrieval": dict(
            results.get("sequential_retrieval") or {}
        ),
        "parallel_retrieval": dict(
            results.get("parallel_retrieval") or {}
        ),
        "stage_latency_ms": dict(results.get("stage_latency_ms") or {}),
        "stage_latency_semantics": dict(results.get("stage_latency_semantics") or {}),
        "total_before_evidence_sufficiency": int(
            results.get("total_before_evidence_sufficiency") or 0
        ),
    }
    return chunks, diagnostics


if __name__ == "__main__":
    result = retrieve_content(
        question="世运电路成长性如何",
        indexNames="test01",
    )
    print(result)
