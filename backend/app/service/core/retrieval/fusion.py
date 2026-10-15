"""多批检索结果的确定性融合与重排，不生成查询、不执行工具。"""
import logging
import os
import re
import time
from typing import Any
from service.core.retrieval.constraints import evaluate_constraint_compatibility, extract_locators
from service.core.evidence.bindings import BridgeExtraction
from service.core.rag.nlp.model import rerank_similarity
from service.core.rag.nlp.search_v2 import Dealer, reciprocal_rank_fusion
from service.core.evidence.metadata import evidence_metadata, ranking_text
from typing import Protocol

class QuerySubqueryPlan(Protocol):
    id: str
    query: str

DEFAULT_PARALLEL_COVERAGE_PER_SUBQUERY = 2
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

def _approximate_figure_penalty(
    question: str,
    chunk: dict[str, Any],
) -> int:
    """优先使用正文或表格的精确值，而非图像估算值。"""
    if str(chunk.get("evidence_type_kwd") or "") != "figure":
        return 0
    if not re.search(r"多少|具体数值|精确|分别是", question):
        return 0
    if re.search(r"图\s*\d+|图表|柱|曲线|折线", question):
        return 0
    content = str(chunk.get("content_with_weight") or "")
    return int(bool(re.search(r"约|左右|未经人工核验", content)))

def _primary_table_row_signature(chunk: dict[str, Any]) -> str | None:
    """返回片段所代表的表格主行标识。"""

    # 优先使用表格id和行号来标识表格行，避免使用表格内容来标识，因为表格内容可能会有重复或相似的情况
    table_id = str(chunk.get("table_id_kwd") or "").strip()
    row_index = chunk.get("table_row_int")
    if row_index is None or str(row_index).strip() == "":
        row_index = chunk.get("table_row_start_int")
    normalized_row_index = (
        "" if row_index is None else str(row_index).strip()
    )

    # 同一已解析表格中的同一行，是最可靠的证据身份标识。
    if table_id and normalized_row_index:
        return f"table-row:{table_id}:{normalized_row_index}"

    # 旧表格片段可能没有稳定的表号和行号；优先使用主行单元格，
    # 避免扩展后的 Markdown 中共享表头和邻近行干扰去重。
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

    # 为各独立目标的最佳结果保留入选资格，而非固定排序位置。
    # 共享片段可同时覆盖它出现过的各子查询，最终顺序仍由共同问题的重排决定。
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
                    "company_constraint": result.get("company_constraint") or {},
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
        # 保留已经核验的桥接证据，避免仅保留无关的第一跳最高分结果。
        # 后续各跳保留自己的最佳独立结果，防止背景证据因对完整问题得分较低而丢失。
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
            # 额外重排请求失败时，保留两跳证据；
            # 不同问题的原始分数不能直接比较，只比较各自局部排名。
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
        # 各跳分数和排名保留在 sequential_subqueries 中；
        # 对外重排分数对应原始问题，重排失败时为未知。
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
    # 目标覆盖决定首页入选集合，不决定固定位置。
    # 入选证据按共同问题的排名排序，剩余结果只追加一次，
    # 防止后续页面重复出现首页预留结果。
    reserved_keys = set(representative_keys)
    if len(reserved_keys) > options.page_size:
        reserved_keys = set()  # 只有一个结果位置的页面无法保证同时覆盖两跳证据。
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
                    "company_constraint": result.get("company_constraint") or {},
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
