"""Knowledge-base retrieval entry points used by chat and offline evaluation."""

import logging
import os
from dataclasses import replace
from functools import lru_cache
from typing import Any
from service.core.evidence_metadata import evidence_metadata

from service.core.constraint_compatibility import evaluate_constraint_compatibility
from service.core.evidence_sufficiency import check_evidence_sufficiency
from service.core.query_expansion import QueryExpansionResult, expand_query
from service.core.query_intent import (
    QueryIntentDecision,
    QuerySubqueryPlan,
    analyze_query_intent,
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


DEFAULT_PAGE_SIZE = 5
DEFAULT_SIMILARITY_THRESHOLD = 0.1
DEFAULT_VECTOR_SIMILARITY_WEIGHT = 0.6
DEFAULT_RETRIEVAL_CANDIDATE_SIZE = 100
DEFAULT_RERANK_CANDIDATE_SIZE = 20
DEFAULT_RRF_K = 60
DEFAULT_FINAL_RERANKER_WEIGHT = 0.4
DEFAULT_FINAL_RRF_K = 10


logger = logging.getLogger(__name__)


def _environment_flag(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().casefold() not in {"0", "false", "no", "off"}


@lru_cache(maxsize=1)
def get_retrieval_dealer() -> Dealer:
    """Create the Elasticsearch-backed dealer only when retrieval is used."""
    return Dealer(dataStore=ESConnection())


def _prepare_query_transform(
    original_question: str,
    query_intent: QueryIntentDecision,
) -> tuple[str, QueryExpansionResult]:
    if query_intent.need_rewrite:
        rewritten_question = rewrite_query(original_question)
        query_transform = expand_query(
            original_question,
            rewritten_question,
        )
    else:
        rewritten_question = original_question
        query_transform = QueryExpansionResult.noop(
            original_question,
            reason="RAG 意图识别判定无需改写",
        )
    return rewritten_question, query_transform


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


def _merge_sequential_hops(
    *,
    original_question: str,
    hop_results: list[tuple[QuerySubqueryPlan, str, dict[str, Any]]],
    bridge: BridgeExtraction,
    options: Dealer.RetrievalOptions,
) -> dict[str, Any]:
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
                    str(chunks_by_key[key].get("content_with_weight")
                        or chunks_by_key[key].get("content_ltks") or "")
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

    return {
        "total": sum(
            int(result.get("total") or 0)
            for _step, _query, result in hop_results
        ),
        "chunks": paged_chunks,
        "doc_aggs": _document_aggregations(paged_chunks),
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
    query_intent: QueryIntentDecision,
    options: Dealer.RetrievalOptions,
) -> tuple[dict[str, Any] | None, str | None]:
    if query_intent.retrieval_mode != "sequential":
        return None, "查询计划不是递进式检索"
    if len(query_intent.subqueries) != 2:
        return None, "递进式查询计划必须包含两步"

    first_step, second_step = query_intent.subqueries
    hop_page_size = min(
        options.rerank_candidate_size,
        max(options.page * options.page_size, 2),
    )
    hop_options = replace(options, page=1, page_size=hop_page_size)

    first_result = dealer.retrieval(
        question=first_step.query,
        keyword_question=(
            first_keyword_question
            if first_step.query == original_question
            else first_step.query
        ),
        tenant_ids=index_names,
        options=hop_options,
    )
    bridge = extract_bridge_value(
        slot=str(first_step.output_slot or ""),
        original_question=original_question,
        first_query=first_step.query,
        chunks=first_result.get("chunks") or [],
    )
    if bridge is None:
        return None, "第一跳证据中没有找到可验证的桥接值"

    try:
        second_query = resolve_query_template(
            second_step.query_template,
            {bridge.slot: bridge.value},
        )
    except ValueError as error:
        return None, str(error)

    second_options = hop_options
    if second_step.inherit_document_scope:
        if not bridge.document_id:
            return None, "第二跳要求继承文档范围，但第一跳证据缺少文档 ID"
        second_options = replace(hop_options, doc_ids=[bridge.document_id])

    second_result = dealer.retrieval(
        question=second_query,
        keyword_question=second_query,
        tenant_ids=index_names,
        options=second_options,
    )
    return (
        _merge_sequential_hops(
            original_question=original_question,
            hop_results=[
                (first_step, first_step.query, first_result),
                (second_step, second_query, second_result),
            ],
            bridge=bridge,
            options=options,
        ),
        None,
    )


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
) -> dict[str, Any]:
    """Return the complete result produced by ``Dealer.retrieval``.

    Chat only needs a compact view of each chunk, while offline evaluation also
    needs stable chunk IDs and ranking scores. Keeping this raw entry point
    prevents the evaluator from duplicating the production retrieval config.
    """
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
    # 去掉首尾空白
    original_question = question.strip()
    query_intent = analyze_query_intent(original_question)
    rewritten_question, query_transform = _prepare_query_transform(
        original_question,
        query_intent,
    )
    keyword_question = query_transform.effective_query
    dealer = get_retrieval_dealer()

    result = None
    sequential_fallback_reason = None
    sequential_enabled = _environment_flag(
        "RAG_SEQUENTIAL_RETRIEVAL_ENABLED",
        True,
    )
    if query_intent.retrieval_mode == "sequential" and sequential_enabled:
        try:
            result, sequential_fallback_reason = _execute_sequential_retrieval(
                dealer=dealer,
                index_names=index_names,
                original_question=original_question,
                first_keyword_question=keyword_question,
                query_intent=query_intent,
                options=options,
            )
        except Exception as error:
            logger.warning(
                "Sequential retrieval failed; falling back to single query: %s",
                type(error).__name__,
            )
            sequential_fallback_reason = (
                f"递进式检索执行失败（{type(error).__name__}）"
            )
    elif query_intent.retrieval_mode == "sequential":
        sequential_fallback_reason = "递进式检索已关闭"

    if result is None:
        # 普通查询以及无有效桥接值的递进查询都安全回退到原有单跳链路。
        result = dealer.retrieval(
            question=original_question,
            keyword_question=keyword_question,
            tenant_ids=index_names,
            options=options,
        )
        if query_intent.retrieval_mode == "sequential":
            result["sequential_retrieval"] = {
                "applied": False,
                "fallback_reason": sequential_fallback_reason,
            }

    # 保留检查前的结果，离线评测分别统计检索质量和拒答表现。
    result["chunks_before_sufficiency"] = list(result.get("chunks") or [])
    result["total_before_evidence_sufficiency"] = result.get("total", 0)
    # 只在所有检索跳完成并合并后检查证据完整性，避免第一跳因暂时缺少
    # 依赖证据而被提前清空。
    sufficiency = check_evidence_sufficiency(
        original_question,
        result.get("chunks", []),
    )
    result["evidence_sufficiency"] = sufficiency.to_dict()
    result["evidence_sufficiency"]["enabled"] = _environment_flag(
        "RAG_EVIDENCE_SUFFICIENCY_ENABLED", True,
    )
    # 如果证据不充分，将检索结果清空，避免返回不可靠的证据给用户
    if not sufficiency.sufficient:
        result["total"] = 0
        result["chunks"] = []
        result["doc_aggs"] = []
    result["query_rewrite"] = {
        "original": original_question,
        "rewritten": rewritten_question,
        "changed": rewritten_question != original_question,
    }
    result["query_intent"] = query_intent.to_dict()
    result["query_transform"] = query_transform.to_dict()
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
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return chat chunks plus bounded policy/retrieval diagnostics."""
    results = retrieve_raw_results(indexNames, question, page_size=page_size)
    chunks = [
        format_retrieved_chunk(chunk, rank)
        for rank, chunk in enumerate(results.get("chunks", []), start=1)
    ]
    diagnostics = {
        "evidence_sufficiency": dict(
            results.get("evidence_sufficiency") or {}
        ),
        "query_intent": dict(results.get("query_intent") or {}),
        "sequential_retrieval": dict(
            results.get("sequential_retrieval") or {}
        ),
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
