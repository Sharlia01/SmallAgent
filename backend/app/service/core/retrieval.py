"""Knowledge-base retrieval entry points used by chat and offline evaluation."""

from functools import lru_cache
from typing import Any

from service.core.rag.nlp.search_v2 import Dealer
from service.core.rag.utils.es_conn import ESConnection


DEFAULT_PAGE_SIZE = 5
DEFAULT_SIMILARITY_THRESHOLD = 0.1
DEFAULT_VECTOR_SIMILARITY_WEIGHT = 0.6


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
        page=page,
        page_size=page_size,
    )
    return get_retrieval_dealer().retrieval(
        question=question,
        tenant_ids=index_names,
        options=options,
    )


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
        "vector_similarity": _optional_float(
            chunk.get("vector_similarity")
        ),
        "term_similarity": _optional_float(chunk.get("term_similarity")),
        "positions": chunk.get("positions") or [],
        "kb_id": chunk.get("kb_id", ""),
        "image_id": chunk.get("image_id", ""),
    }


def retrieve_content(indexNames: str, question: str):
    """Retrieve the top chunks for chat while retaining ranking metadata."""
    results = retrieve_raw_results(indexNames, question)
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
