"""Pure helpers for evaluating ranked RAG retrieval results."""

from __future__ import annotations

import math
import os
import unicodedata
from collections import Counter, defaultdict
from typing import Any, Iterable


def normalize_text(text: Any) -> str:
    """Normalize OCR and layout differences before comparing evidence text."""
    normalized = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return "".join(character for character in normalized if character.isalnum())


def normalize_document_name(name: Any) -> str:
    """Normalize a document path to the filename stored in the gold set."""
    normalized = unicodedata.normalize("NFKC", str(name or ""))
    return os.path.basename(normalized.replace("\\", "/")).casefold()


def evidence_text_match_score(reference: Any, candidate: Any) -> float:
    """Return how much of a gold evidence string is present in a chunk.

    Exact normalized containment receives 1.0. The character trigram fallback
    tolerates small OCR differences while remaining asymmetric: a candidate
    must cover most of the gold evidence, rather than merely sharing a short
    phrase with it.
    """
    reference_text = normalize_text(reference)
    candidate_text = normalize_text(candidate)
    if not reference_text or not candidate_text:
        return 0.0
    if reference_text in candidate_text:
        return 1.0
    if candidate_text in reference_text:
        return len(candidate_text) / len(reference_text)

    ngram_size = min(3, len(reference_text))
    reference_ngrams = {
        reference_text[index : index + ngram_size]
        for index in range(len(reference_text) - ngram_size + 1)
    }
    candidate_ngrams = {
        candidate_text[index : index + ngram_size]
        for index in range(len(candidate_text) - ngram_size + 1)
    }
    if not reference_ngrams:
        return 0.0
    return len(reference_ngrams & candidate_ngrams) / len(reference_ngrams)


def _chunk_document_name(chunk: dict[str, Any]) -> Any:
    return chunk.get("document_name") or chunk.get("docnm_kwd")


def _chunk_content(chunk: dict[str, Any]) -> Any:
    return chunk.get("content_with_weight") or chunk.get("content")


def evidence_chunk_match_score(
    evidence: dict[str, Any],
    chunk: dict[str, Any],
) -> float:
    """Score a chunk against evidence after enforcing document identity."""
    evidence_document = normalize_document_name(evidence.get("document_name"))
    chunk_document = normalize_document_name(_chunk_document_name(chunk))
    if not evidence_document or evidence_document != chunk_document:
        return 0.0
    return evidence_text_match_score(
        evidence.get("text"),
        _chunk_content(chunk),
    )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _json_safe(value: Any) -> Any:
    """Recursively convert NumPy-like values without importing NumPy."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return _json_safe(item_method())
        except (TypeError, ValueError):
            pass
    return str(value)


def serialize_retrieved_chunk(
    chunk: dict[str, Any],
    rank: int,
) -> dict[str, Any]:
    """Keep ranking diagnostics while omitting the large embedding vector."""
    return {
        "rank": rank,
        "chunk_id": str(chunk.get("chunk_id", "")),
        "document_id": str(chunk.get("doc_id", "")),
        "document_name": normalize_document_name(_chunk_document_name(chunk)),
        "content_with_weight": str(_chunk_content(chunk) or ""),
        "similarity": _optional_float(chunk.get("similarity")),
        "vector_similarity": _optional_float(
            chunk.get("vector_similarity")
        ),
        "term_similarity": _optional_float(chunk.get("term_similarity")),
        "positions": _json_safe(chunk.get("positions") or []),
        "kb_id": str(chunk.get("kb_id", "")),
        "image_id": str(chunk.get("image_id", "")),
    }


def get_source_modality(sample: dict[str, Any]) -> str:
    metadata = sample.get("metadata") or {}
    if metadata.get("source_modality"):
        return str(metadata["source_modality"])
    evidence_types = {
        evidence.get("evidence_type")
        for evidence in sample.get("relevant_evidence", [])
        if evidence.get("evidence_type")
    }
    if evidence_types:
        return "+".join(sorted(evidence_types))
    return "text" if sample.get("answerable") else "none"


def evaluate_retrieval_case(
    sample: dict[str, Any],
    raw_result: dict[str, Any],
    *,
    latency_ms: float,
    top_k: int,
    match_threshold: float,
) -> dict[str, Any]:
    """Evaluate one ranked retrieval result against all gold evidence."""
    raw_chunks = list(raw_result.get("chunks") or [])[:top_k]
    serialized_chunks = [
        serialize_retrieved_chunk(chunk, rank)
        for rank, chunk in enumerate(raw_chunks, start=1)
    ]
    evidence_matches = []
    relevant_ranks = set()

    for evidence_index, evidence in enumerate(
        sample.get("relevant_evidence") or [],
        start=1,
    ):
        scored_chunks = [
            (
                rank,
                evidence_chunk_match_score(evidence, chunk),
                str(chunk.get("chunk_id", "")),
            )
            for rank, chunk in enumerate(raw_chunks, start=1)
        ]
        passing_matches = [
            item for item in scored_chunks if item[1] >= match_threshold
        ]
        first_match = min(passing_matches, default=None, key=lambda item: item[0])
        best_score = max((item[1] for item in scored_chunks), default=0.0)
        matched_rank = first_match[0] if first_match else None
        if matched_rank is not None:
            relevant_ranks.add(matched_rank)

        evidence_matches.append(
            {
                "evidence_index": evidence_index,
                "document_name": evidence.get("document_name"),
                "page": evidence.get("page"),
                "evidence_type": evidence.get("evidence_type", "text"),
                "matched": first_match is not None,
                "matched_rank": matched_rank,
                "matched_chunk_id": first_match[2] if first_match else None,
                "best_match_score": round(best_score, 6),
            }
        )

    answerable = bool(sample.get("answerable"))
    gold_evidence_count = len(evidence_matches)
    matched_evidence_count = sum(
        1 for evidence in evidence_matches if evidence["matched"]
    )
    first_relevant_rank = min(relevant_ranks) if relevant_ranks else None
    metric_suffix = str(top_k)

    metrics = {
        f"hit_at_{metric_suffix}": (
            bool(first_relevant_rank) if answerable else None
        ),
        f"recall_at_{metric_suffix}": (
            matched_evidence_count / gold_evidence_count
            if answerable and gold_evidence_count
            else None
        ),
        f"mrr_at_{metric_suffix}": (
            1.0 / first_relevant_rank
            if answerable and first_relevant_rank
            else (0.0 if answerable else None)
        ),
    }

    return {
        "id": sample.get("id"),
        "question": sample.get("question"),
        "reference_answer": sample.get("reference_answer"),
        "question_type": sample.get("question_type"),
        "source_modality": get_source_modality(sample),
        "answerable": answerable,
        "error": None,
        "latency_ms": round(float(latency_ms), 3),
        "retrieval": {
            "total_candidates": int(raw_result.get("total") or 0),
            "retrieved_count": len(serialized_chunks),
            "empty": not serialized_chunks,
            "chunks": serialized_chunks,
        },
        "gold_evidence_count": gold_evidence_count,
        "matched_evidence_count": matched_evidence_count,
        "first_relevant_rank": first_relevant_rank,
        "evidence_matches": evidence_matches,
        "metrics": metrics,
    }


def build_error_case(
    sample: dict[str, Any],
    *,
    error: Exception,
    latency_ms: float,
    top_k: int,
) -> dict[str, Any]:
    """Record one failed query without losing the rest of the evaluation run."""
    return {
        "id": sample.get("id"),
        "question": sample.get("question"),
        "reference_answer": sample.get("reference_answer"),
        "question_type": sample.get("question_type"),
        "source_modality": get_source_modality(sample),
        "answerable": bool(sample.get("answerable")),
        "error": f"{type(error).__name__}: {error}",
        "latency_ms": round(float(latency_ms), 3),
        "retrieval": {
            "total_candidates": 0,
            "retrieved_count": 0,
            "empty": True,
            "chunks": [],
        },
        "gold_evidence_count": len(sample.get("relevant_evidence") or []),
        "matched_evidence_count": 0,
        "first_relevant_rank": None,
        "evidence_matches": [],
        "metrics": {
            f"hit_at_{top_k}": None,
            f"recall_at_{top_k}": None,
            f"mrr_at_{top_k}": None,
        },
    }


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def aggregate_results(
    results: list[dict[str, Any]],
    *,
    top_k: int,
) -> dict[str, Any]:
    """Compute macro retrieval metrics for one result slice."""
    successful = [result for result in results if not result.get("error")]
    answerable = [result for result in successful if result.get("answerable")]
    unanswerable = [
        result for result in successful if not result.get("answerable")
    ]
    hit_key = f"hit_at_{top_k}"
    recall_key = f"recall_at_{top_k}"
    mrr_key = f"mrr_at_{top_k}"

    return {
        "query_count": len(results),
        "successful_query_count": len(successful),
        "error_count": len(results) - len(successful),
        "answerable_query_count": len(answerable),
        "unanswerable_query_count": len(unanswerable),
        "empty_rate": _mean(
            float(result["retrieval"]["empty"]) for result in successful
        ),
        "unanswerable_empty_rate": _mean(
            float(result["retrieval"]["empty"])
            for result in unanswerable
        ),
        "average_latency_ms": _mean(
            float(result["latency_ms"]) for result in successful
        ),
        hit_key: _mean(
            float(result["metrics"][hit_key]) for result in answerable
        ),
        recall_key: _mean(
            float(result["metrics"][recall_key]) for result in answerable
        ),
        mrr_key: _mean(
            float(result["metrics"][mrr_key]) for result in answerable
        ),
    }


def build_summary(
    results: list[dict[str, Any]],
    *,
    top_k: int,
) -> dict[str, Any]:
    """Build overall and per-slice summaries for a completed run."""
    by_question_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_source_modality: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_question_type[str(result.get("question_type") or "unknown")].append(
            result
        )
        by_source_modality[str(result.get("source_modality") or "unknown")].append(
            result
        )

    return {
        "question_type_counts": dict(
            sorted(Counter(result["question_type"] for result in results).items())
        ),
        "source_modality_counts": dict(
            sorted(Counter(result["source_modality"] for result in results).items())
        ),
        "overall": aggregate_results(results, top_k=top_k),
        "by_question_type": {
            name: aggregate_results(items, top_k=top_k)
            for name, items in sorted(by_question_type.items())
        },
        "by_source_modality": {
            name: aggregate_results(items, top_k=top_k)
            for name, items in sorted(by_source_modality.items())
        },
    }
