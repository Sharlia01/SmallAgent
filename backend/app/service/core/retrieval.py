"""Knowledge-base retrieval entry points used by chat and offline evaluation."""

from functools import lru_cache
from typing import Any

from service.core.evidence_sufficiency import check_evidence_sufficiency
from service.core.query_expansion import QueryExpansionResult, expand_query
from service.core.query_intent import analyze_query_intent
from service.core.query_rewrite import rewrite_query
from service.core.rag.nlp.search_v2 import Dealer
from service.core.rag.utils.es_conn import ESConnection


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
    """Create the Elasticsearch-backed dealer only when retrieval is used."""
    return Dealer(dataStore=ESConnection())


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
    keyword_question = query_transform.effective_query
    # 将两个问题一起交给检索器
    result = get_retrieval_dealer().retrieval(
        question=original_question,
        keyword_question=keyword_question,
        tenant_ids=index_names,
        options=options,
    )
    sufficiency = check_evidence_sufficiency(
        original_question,
        result.get("chunks", []),
    )
    result["evidence_sufficiency"] = sufficiency.to_dict()
    # 如果证据不充分，将检索结果清空，避免返回不可靠的证据给用户
    if not sufficiency.sufficient:
        result["total_before_evidence_sufficiency"] = result.get("total", 0)
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
    results = retrieve_raw_results(indexNames, question, page_size=page_size)
    return [
        format_retrieved_chunk(chunk, rank)
        for rank, chunk in enumerate(results.get("chunks", []), start=1)
    ]


if __name__ == "__main__":
    result = retrieve_content(
        question="世运电路成长性如何",
        indexNames="test01",
    )
    print(result)
