"""知识库搜索入口：执行给定查询，返回候选证据；不理解需求、不规划或补查。"""

import time
import os
from functools import lru_cache
from typing import Any

from domain.evidence import EvidenceCollection
from service.core.evidence.adapter import collection_from_records, evidence_from_record
from service.core.evidence.metadata import evidence_metadata
from service.core.retrieval.result import RetrievalResult
from service.core.retrieval.entities import resolve_company_hints
from service.core.retrieval.table_context import expand_table_contexts
from service.core.query.semantics import alias_expansions
from service.core.query.contract import RetrievalTask
from service.core.rag.nlp.search_v2 import Dealer
from service.core.rag.utils.es_conn import ESConnection
from service.core.runtime.execution import bounded_query
from service.core.runtime.context import current_runtime

DEFAULT_PAGE_SIZE = 5
DEFAULT_SIMILARITY_THRESHOLD = 0.1
DEFAULT_VECTOR_SIMILARITY_WEIGHT = 0.6
DEFAULT_RETRIEVAL_CANDIDATE_SIZE = 100
DEFAULT_RERANK_CANDIDATE_SIZE = 20
DEFAULT_RRF_K = 60
DEFAULT_FINAL_RERANKER_WEIGHT = 0.7
DEFAULT_FINAL_RRF_K = 10

@lru_cache(maxsize=1)
def get_retrieval_dealer() -> Dealer:
    return Dealer(dataStore=ESConnection())


@bounded_query
def retrieve_evidence(index_names: str | list[str], question: str, *,
                      task: RetrievalTask | None = None,
                      page: int = 1, page_size: int = DEFAULT_PAGE_SIZE,
                      similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
                      vector_similarity_weight: float = DEFAULT_VECTOR_SIMILARITY_WEIGHT,
                      candidate_size: int = DEFAULT_RETRIEVAL_CANDIDATE_SIZE,
                      rerank_candidate_size: int = DEFAULT_RERANK_CANDIDATE_SIZE,
                      rrf_k: int = DEFAULT_RRF_K,
                      final_reranker_weight: float = DEFAULT_FINAL_RERANKER_WEIGHT,
                      final_rrf_k: int = DEFAULT_FINAL_RRF_K) -> RetrievalResult:
    """一次请求只执行一次混合检索。索引身份由服务端调用方绑定。"""
    query = task.query if task else question
    if not isinstance(query, str) or not query.strip() or len(query) > 12000:
        raise ValueError("搜索查询必须为1到12000个字符")
    if type(page) is not int or page < 1 or type(page_size) is not int or not 1 <= page_size <= 20:
        raise ValueError("分页参数无效")
    query = query.strip()
    started = time.perf_counter()
    scope = task.company_hints if task and task.company_hints is not None else resolve_company_hints(query)
    options = Dealer.RetrievalOptions(
        kb_ids=None, page=page, page_size=page_size,
        similarity_threshold=similarity_threshold, vector_similarity_weight=vector_similarity_weight,
        candidate_size=candidate_size, rerank_candidate_size=rerank_candidate_size,
        rrf_k=rrf_k, final_reranker_weight=final_reranker_weight, final_rrf_k=final_rrf_k,
        company_hints=scope, doc_ids=list(task.document_ids) if task and task.document_ids else None,
    )
    # 这里只增加词表里明确等价的别名；模型生成的新搜索方向由 Agent 负责。
    aliases = (alias_expansions(query, 8) if os.getenv("RETRIEVAL_ALIAS_EXPANSION_ENABLED", "true").lower()
               not in {"0", "false", "no", "off"} else [])
    keyword_query = " ".join([query, *aliases])
    runtime = current_runtime()
    key = ("search", repr(index_names), query, keyword_query, repr(options))
    with runtime._lock:
        cached = runtime.cache.get(key)
    if cached is not None:
        result = cached.copy()
        result.task = task
        return result
    runtime.reserve("retrieval")
    # raw是经过检索并重排后的结果字典
    raw = get_retrieval_dealer().retrieval(
        question=query, keyword_question=keyword_query, tenant_ids=index_names, options=options,
    )
    chunks = [record if record.get("chunk_id") else {**record, "chunk_id": evidence_from_record(record).id}
              for record in raw.get("chunks", [])]
    chunks = expand_table_contexts(chunks, query, candidates=raw.get("table_candidates"))
    evidence = collection_from_records(chunks, query=query) # 将结果字典转换成evidenceCollection
    diagnostics = retrieval_diagnostics(raw) # 用于排查问题和评测分析
    diagnostics["keyword_query"] = keyword_query
    diagnostics["alias_terms"] = aliases
    diagnostics["stage_latency_ms"]["total"] = round((time.perf_counter() - started) * 1000, 3)
    # 搜索命中不等于答案充分，所有返回片段均待 Agent 合并后核验。
    result = RetrievalResult(evidence=evidence, diagnostics=diagnostics,
                             total_candidates=int(raw.get("total") or 0), task=task,
                             recoverable_evidence=tuple(collection_from_records(
                                 [row for chunk in chunks for row in chunk.get('_table_recovery_records', [])], query=query)))
    with runtime._lock:
        runtime.cache[key] = result.copy()
    return result

def _optional_float(value: Any) -> float | None:
    """将 NumPy 和 Python 数值转换为可进行 JSON 序列化的浮点数。"""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def format_retrieved_chunk(chunk: dict[str, Any], rank: int) -> dict[str, Any]:
    """构建调用方使用的紧凑、可进行 JSON 序列化的片段表示。"""
    # JSON 格式化与领域适配使用相同缺省值，避免格式化本身改变报告绑定的来源身份。
    document_name = str(chunk.get("document_name") or chunk.get("docnm_kwd") or "未知文档")
    document_name = document_name.replace("\\", "/").split("/")[-1]

    return {
        # 保留 id 字段，供现有引用提示与前端使用。
        "id": rank,
        **evidence_metadata(chunk),
        "rank": rank,
        "chunk_id": chunk.get("chunk_id", ""),
        "document_id": chunk.get("document_id") or chunk.get("doc_id") or "",
        "document_name": document_name,
        "content_with_weight": chunk.get("content_with_weight", chunk.get("content", "")),
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

def format_retrieval_results(results: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """外部检索器和历史 JSON 的适配准备，不执行检索。"""
    chunks = [
        format_retrieved_chunk(chunk, rank)
        for rank, chunk in enumerate(results.get("chunks", []), start=1)
    ]
    diagnostics = retrieval_diagnostics(results)
    diagnostics.update(
        candidate_chunks=[format_retrieved_chunk(chunk, rank) for rank, chunk in enumerate(
            results.get("chunks_before_sufficiency") or [], start=1)],
        evidence_sufficiency=dict(results.get("evidence_sufficiency") or {}),
        query_contract=dict(results.get("query_contract") or {}),
        query_intent=dict(results.get("query_intent") or {}),
        response_mode=results.get("response_mode", "answer" if chunks else "insufficient"),
        clarification=results.get("clarification"),
        total=results.get("total"),
    )

    return chunks, diagnostics

def retrieval_diagnostics(results: dict[str, Any]) -> dict[str, Any]:
    """诊断只描述搜索执行；不携带需求理解或回答判定。"""
    return {
        "retrieval_fusion": dict(results.get("retrieval_fusion") or {}),
        "doc_aggs": list(results.get("doc_aggs") or []),
        "company_hints": dict(results.get("company_hints") or {}),
        "stage_latency_ms": dict(results.get("stage_latency_ms") or {}),
        "query_runtime": dict(results.get("query_runtime") or {}),
    }
