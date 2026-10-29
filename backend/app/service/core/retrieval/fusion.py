"""多批检索结果的确定性融合与重排，不生成查询、不执行工具。"""
import logging
import time
from typing import Any
from service.core.retrieval.constraints import evaluate_constraint_compatibility
from service.core.evidence.bindings import BridgeExtraction
from service.core.rag.nlp.model import rerank_similarity
from service.core.rag.nlp.search_v2 import Dealer, reciprocal_rank_fusion
from service.core.evidence.metadata import rerank_text
from typing import Protocol

class QuerySubqueryPlan(Protocol):
    id: str
    query: str

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
                    rerank_text(chunks_by_key[key])
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
                    "company_hints": result.get("company_hints") or {},
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
