"""Pure helpers for evaluating ranked RAG retrieval results."""

from __future__ import annotations

import math
import os
import unicodedata
from html.parser import HTMLParser
from collections import Counter, defaultdict
from typing import Any, Iterable

from service.core.evidence_metadata import evidence_metadata


RESULT_SCHEMA_VERSION = "2.1"
DEFAULT_METRIC_KS = (1, 3, 5, 10)


def resolve_metric_ks(
    top_k: int,
    metric_ks: Iterable[int] | None = None,
) -> tuple[int, ...]:
    """Return useful metric cutoffs that can be scored from Top K results."""
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    requested = DEFAULT_METRIC_KS if metric_ks is None else tuple(metric_ks)
    cutoffs = {
        cutoff
        for cutoff in requested
        if isinstance(cutoff, int)
        and not isinstance(cutoff, bool)
        and 0 < cutoff <= top_k
    }
    cutoffs.add(top_k)
    return tuple(sorted(cutoffs))


def normalize_text(text: Any) -> str:
    """Normalize OCR and layout differences before comparing evidence text."""
    normalized = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return "".join(character for character in normalized if character.isalnum())


def normalize_document_name(name: Any) -> str:
    """Normalize a document path to the filename stored in the gold set."""
    normalized = unicodedata.normalize("NFKC", str(name or ""))
    return os.path.basename(normalized.replace("\\", "/")).casefold()

def normalize_table_cell(value: Any) -> str:
    """Normalize table cells while preserving numeric signs and decimals."""
    normalized = unicodedata.normalize(
        "NFKC",
        str(value or ""),
    ).casefold().strip()

    # 财务表格通常使用括号表示负数，例如 (23181)。
    if (
        len(normalized) >= 2
        and normalized.startswith("(")
        and normalized.endswith(")")
    ):
        normalized = f"-{normalized[1:-1]}"

    # 千位分隔符不影响数值含义。
    normalized = normalized.replace(",", "")

    return "".join(
        character
        for character in normalized
        if character.isalnum() or character in ".%+-"
    )


class _TableHTMLParser(HTMLParser):
    """Extract captions and rows from the simple HTML tables stored in chunks."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables = []
        self._table = None
        self._caption_parts = None
        self._row = None
        self._cell_parts = None

    def handle_starttag(self, tag, attrs):
        tag = tag.casefold()

        if tag == "table":
            self._table = {
                "caption": "",
                "rows": [],
            }
        elif self._table is None:
            return
        elif tag == "caption":
            self._caption_parts = []
        elif tag == "tr":
            self._row = []
        elif tag in {"td", "th"} and self._row is not None:
            self._cell_parts = []

    def handle_data(self, data):
        if self._cell_parts is not None:
            self._cell_parts.append(data)
        elif self._caption_parts is not None:
            self._caption_parts.append(data)

    def handle_endtag(self, tag):
        tag = tag.casefold()

        if tag in {"td", "th"} and self._cell_parts is not None:
            if self._row is not None:
                self._row.append("".join(self._cell_parts).strip())
            self._cell_parts = None

        elif tag == "tr" and self._row is not None:
            if self._table is not None and self._row:
                self._table["rows"].append(self._row)
            self._row = None

        elif tag == "caption" and self._caption_parts is not None:
            if self._table is not None:
                self._table["caption"] = "".join(
                    self._caption_parts
                ).strip()
            self._caption_parts = None

        elif tag == "table" and self._table is not None:
            self.tables.append(self._table)
            self._table = None


def parse_html_tables(content: Any) -> list[dict[str, Any]]:
    parser = _TableHTMLParser()
    parser.feed(str(content or ""))
    parser.close()
    return parser.tables

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

def table_row_chunk_match_score(evidence, chunk):
    # 1. 文档必须一致
    evidence_document = normalize_document_name(
        evidence.get("document_name")
    )
    chunk_document = normalize_document_name(
        _chunk_document_name(chunk)
    )

    if not evidence_document or evidence_document != chunk_document:
        return 0.0

    expected_cells = [
        normalize_table_cell(cell)
        for cell in evidence.get("row_cells") or []
    ]
    expected_text = evidence.get("text")
    best_score = 0.0
    content = _chunk_content(chunk)
    tables = parse_html_tables(content)

    # Some PDF tables are stored as flattened plain text instead of HTML.
    # Without row tags, use normalized evidence coverage as a fallback while
    # retaining the document identity check above.
    if not tables:
        if not expected_text:
            return 0.0
        return evidence_text_match_score(expected_text, content)

    # 2. 解析召回chunk里的HTML表格
    for table in tables:
        for row in table.get("rows") or []:

            # 兼容原来的row_cells精确匹配
            actual_cells = [
                normalize_table_cell(cell)
                for cell in row
            ]

            if expected_cells:
                if actual_cells == expected_cells:
                    return 1.0

                expected_length = len(expected_cells)
                for start in range(
                    len(actual_cells) - expected_length + 1
                ):
                    if (
                        actual_cells[start:start + expected_length]
                        == expected_cells
                    ):
                        return 1.0

            # 支持现有relevant_evidence.text
            if expected_text:
                actual_row_text = " ".join(row)
                row_score = evidence_text_match_score(
                    expected_text,
                    actual_row_text,
                )
                best_score = max(best_score, row_score)

    return best_score


def figure_chunk_match_score(
    evidence: dict[str, Any],
    chunk: dict[str, Any],
) -> float:
    """Match a retrieved figure by its caption/locator in the same document."""
    evidence_document = normalize_document_name(evidence.get("document_name"))
    chunk_document = normalize_document_name(_chunk_document_name(chunk))
    if not evidence_document or evidence_document != chunk_document:
        return 0.0

    # Figure gold text often describes a visual fact that OCR cannot recover.
    # The locator identifies the figure that must be retrieved, so use it as
    # the primary retrieval-relevance reference and retain text as a fallback
    # for older datasets without a locator.
    reference = evidence.get("locator") or evidence.get("text")
    return evidence_text_match_score(reference, _chunk_content(chunk))


def relevant_evidence_chunk_match_score(
    evidence: dict[str, Any],
    chunk: dict[str, Any],
) -> float:
    evidence_type = str(
        evidence.get("evidence_type") or "text"
    ).casefold()

    if evidence_type == "table":
        return table_row_chunk_match_score(evidence, chunk)

    if evidence_type == "figure":
        return figure_chunk_match_score(evidence, chunk)

    if evidence_type == "text":
        return evidence_chunk_match_score(evidence, chunk)

    raise ValueError(
        f"Unsupported evidence_type: {evidence_type}"
    )

def evaluate_evidence_requirement(
    requirement: dict[str, Any],
    chunks: list[dict[str, Any]],
    match_threshold: float,
) -> dict[str, Any]:
    """Evaluate OR alternatives whose evidence atoms may match different chunks."""
    successful_alternatives = []
    best_coverage = 0.0

    for alternative_index, alternative in enumerate(
        requirement.get("alternatives") or [],
        start=1,
    ):
        if not alternative:
            continue

        atom_matches = []

        for atom_index, atom in enumerate(alternative, start=1):
            match_type = atom.get("type", "text")

            if match_type == "text":
                score_function = evidence_chunk_match_score
            elif match_type == "table_row":
                score_function = table_row_chunk_match_score
            else:
                raise ValueError(
                    f"Unsupported evidence match type: {match_type}"
                )

            scored_chunks = [
                (
                    rank,
                    score_function(atom, chunk),
                    str(chunk.get("chunk_id", "")),
                )
                for rank, chunk in enumerate(chunks, start=1)
            ]

            passing_matches = [
                item
                for item in scored_chunks
                if item[1] >= match_threshold
            ]
            first_match = min(
                passing_matches,
                default=None,
                key=lambda item: item[0],
            )
            best_score = max(
                (item[1] for item in scored_chunks),
                default=0.0,
            )

            atom_matches.append(
                {
                    "atom_index": atom_index,
                    "matched": first_match is not None,
                    "matched_rank": (
                        first_match[0] if first_match else None
                    ),
                    "matched_chunk_id": (
                        first_match[2] if first_match else None
                    ),
                    "best_match_score": round(best_score, 6),
                }
            )

        matched_atom_count = sum(
            1 for atom_match in atom_matches
            if atom_match["matched"]
        )
        coverage = matched_atom_count / len(atom_matches)
        best_coverage = max(best_coverage, coverage)

        if matched_atom_count == len(atom_matches):
            completion_rank = max(
                atom_match["matched_rank"]
                for atom_match in atom_matches
            )
            successful_alternatives.append(
                {
                    "alternative_index": alternative_index,
                    "completion_rank": completion_rank,
                    "atom_matches": atom_matches,
                    "matched_chunk_ids": [
                        atom_match["matched_chunk_id"]
                        for atom_match in atom_matches
                    ],
                }
            )

    if not successful_alternatives:
        return {
            "requirement_id": requirement.get("id"),
            "matched": False,
            "matched_rank": None,
            "matched_chunk_id": None,
            "matched_chunk_ids": [],
            "best_match_score": round(best_coverage, 6),
            "alternative_index": None,
            "atom_matches": [],
        }

    # 如果多个 OR 分支都成功，选择最早形成完整证据的分支。
    best_alternative = min(
        successful_alternatives,
        key=lambda item: item["completion_rank"],
    )

    return {
        "requirement_id": requirement.get("id"),
        "matched": True,
        "matched_rank": best_alternative["completion_rank"],
        "matched_chunk_id": None,
        "matched_chunk_ids": best_alternative["matched_chunk_ids"],
        "best_match_score": 1.0,
        "alternative_index": best_alternative["alternative_index"],
        "atom_matches": best_alternative["atom_matches"],
    }

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
        **evidence_metadata(chunk),
        "chunk_id": str(chunk.get("chunk_id", "")),
        "document_id": str(chunk.get("doc_id", "")),
        "document_name": normalize_document_name(_chunk_document_name(chunk)),
        "content_with_weight": str(_chunk_content(chunk) or ""),
        "similarity": _optional_float(chunk.get("similarity")),
        "rerank_score": _optional_float(chunk.get("rerank_score")),
        "rrf_score": _optional_float(chunk.get("rrf_score")),
        "vector_similarity": _optional_float(
            chunk.get("vector_similarity")
        ),
        "term_similarity": _optional_float(chunk.get("term_similarity")),
        "retrieval_ranks": _json_safe(
            chunk.get("retrieval_ranks") or {}
        ),
        "retrieval_scores": _json_safe(
            chunk.get("retrieval_scores") or {}
        ),
        "rrf_contributions": _json_safe(
            chunk.get("rrf_contributions") or {}
        ),
        "constraint_compatibility": _json_safe(
            chunk.get("constraint_compatibility") or {}
        ),
        "final_ranking": _json_safe(chunk.get("final_ranking") or {}),
        "sequential_subqueries": _json_safe(
            chunk.get("sequential_subqueries") or []
        ),
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


def _iter_gold_matchers(sample: dict[str, Any]):
    requirements = sample.get("evidence_requirements") or []
    if requirements:
        for requirement in requirements:
            for alternative in requirement.get("alternatives") or []:
                for atom in alternative:
                    match_type = atom.get("type", "text")
                    if match_type == "text":
                        yield evidence_chunk_match_score, atom
                    elif match_type == "table_row":
                        yield table_row_chunk_match_score, atom
                    else:
                        raise ValueError(
                            f"Unsupported evidence match type: {match_type}"
                        )
        return

    for evidence in sample.get("relevant_evidence") or []:
        yield relevant_evidence_chunk_match_score, evidence


def _relevant_chunk_count(
    sample: dict[str, Any],
    chunks: list[dict[str, Any]],
    match_threshold: float,
) -> int:
    matchers = list(_iter_gold_matchers(sample))
    return sum(
        any(
            score_function(evidence, chunk) >= match_threshold
            for score_function, evidence in matchers
        )
        for chunk in chunks
    )


def _chunk_fingerprint(chunk: dict[str, Any], rank: int) -> tuple[str, ...]:
    document_name = normalize_document_name(_chunk_document_name(chunk))
    content = normalize_text(_chunk_content(chunk))
    if content:
        return ("content", document_name, content)

    chunk_id = str(chunk.get("chunk_id") or "").strip()
    if chunk_id:
        return ("chunk_id", chunk_id)
    return ("rank", str(rank))


def _exact_duplicate_rate(chunks: list[dict[str, Any]]) -> float | None:
    """Return exact normalized duplicate share; empty result sets are unknown."""
    if not chunks:
        return None
    fingerprints = {
        _chunk_fingerprint(chunk, rank)
        for rank, chunk in enumerate(chunks, start=1)
    }
    return round((len(chunks) - len(fingerprints)) / len(chunks), 6)


def _evaluate_chunks(
    sample: dict[str, Any],
    raw_chunks: list[dict[str, Any]],
    *,
    top_k: int,
    match_threshold: float,
) -> dict[str, Any]:
    """Score either retrieval stage using exactly the same evidence rules."""
    evidence_requirements = sample.get("evidence_requirements") or []

    # 先确定“标准答案”长什么样子，然后再去看检索结果里有没有匹配的。
    # 多证据题的覆盖要求；所有 requirement 都满足，Hit 才算成功
    if evidence_requirements:
        evidence_matches = [
            evaluate_evidence_requirement(
                requirement,
                raw_chunks,
                match_threshold,
            )
            for requirement in evidence_requirements
        ]
    else:
        # 对于每一条relevant_evidence，检查检索结果里有没有匹配的chunk。
        evidence_matches = []
        for evidence_index, evidence in enumerate(
            sample.get("relevant_evidence") or [],
            start=1,
        ):
            scored_chunks = [
                (
                    rank,
                    relevant_evidence_chunk_match_score(evidence, chunk),
                    str(chunk.get("chunk_id", "")),
                )
                for rank, chunk in enumerate(raw_chunks, start=1)
            ]
            # 检索片段里得分超过阈值的片段
            passing_matches = [
                item for item in scored_chunks
                if item[1] >= match_threshold
            ]
            # 排名最前的匹配片段（如果有的话）
            first_match = min(
                passing_matches,
                default=None,
                key=lambda item: item[0],
            )
            # 所有检索片段里得分最高的片段的分数
            best_score = max(
                (item[1] for item in scored_chunks),
                default=0.0,
            )
            matched_rank = first_match[0] if first_match else None

            # 记录每条evidence的匹配结果，包括是否匹配、匹配的排名、匹配的chunk_id，以及最佳匹配分数。
            evidence_matches.append(
                {
                    "evidence_index": evidence_index,
                    "document_name": evidence.get("document_name"),
                    "page": evidence.get("page"),
                    "evidence_type": evidence.get(
                        "evidence_type",
                        "text",
                    ),
                    "matched": first_match is not None,
                    "matched_rank": matched_rank,
                    "matched_chunk_id": (
                        first_match[2] if first_match else None
                    ),
                    "best_match_score": round(best_score, 6),
                }
            )

    answerable = bool(sample.get("answerable"))
    gold_evidence_count = len(evidence_matches)
    matched_evidence_count = sum(
        1 for evidence in evidence_matches if evidence["matched"]
    )
    matched_ranks = [
        evidence["matched_rank"]
        for evidence in evidence_matches
        if evidence["matched_rank"] is not None
    ]

    if evidence_requirements:
        # 对于requirements，只有当所有requirements都匹配时，才算满足要求。
        all_requirements_matched = (
            bool(evidence_matches)
            and all(evidence["matched"] for evidence in evidence_matches)
        )
        first_relevant_rank = (
            max(matched_ranks)
            if all_requirements_matched
            else None
        )
    else:
        first_relevant_rank = (
            min(matched_ranks)
            if matched_ranks
            else None
        )

    metric_suffix = str(top_k)
    relevant_chunk_count = _relevant_chunk_count(
        sample,
        raw_chunks,
        match_threshold,
    )

    metrics = {
        f"hit_at_{metric_suffix}": (
            bool(first_relevant_rank) if answerable else None
        ),
        f"recall_at_{metric_suffix}": (
            matched_evidence_count / gold_evidence_count
            if answerable and gold_evidence_count
            else None
        ),
        f"precision_at_{metric_suffix}": (
            relevant_chunk_count / top_k if answerable else None
        ),
        f"mrr_at_{metric_suffix}": (
            1.0 / first_relevant_rank
            if answerable and first_relevant_rank
            else (0.0 if answerable else None)
        ),
        f"all_requirements_hit_at_{metric_suffix}": (
            bool(first_relevant_rank)
            if answerable and evidence_requirements
            else None
        ),
        f"duplicate_rate_at_{metric_suffix}": _exact_duplicate_rate(raw_chunks),
    }

    return {
        "gold_evidence_count": gold_evidence_count,
        "matched_evidence_count": matched_evidence_count,
        "first_relevant_rank": first_relevant_rank,
        "evidence_matches": evidence_matches,
        "metrics": metrics,
    }


def _evaluate_ranked_chunks(
    sample: dict[str, Any],
    raw_chunks: list[dict[str, Any]],
    *,
    top_k: int,
    metric_ks: Iterable[int],
    match_threshold: float,
) -> dict[str, Any]:
    evaluations = {
        cutoff: _evaluate_chunks(
            sample,
            raw_chunks[:cutoff],
            top_k=cutoff,
            match_threshold=match_threshold,
        )
        for cutoff in metric_ks
    }
    top_k_result = evaluations[top_k]
    metrics = {}
    for cutoff in metric_ks:
        metrics.update(evaluations[cutoff]["metrics"])
    return {**top_k_result, "metrics": metrics}


def _unavailable_metrics(
    top_k: int,
    metric_ks: Iterable[int] | None = None,
) -> dict[str, None]:
    metrics = {}
    for cutoff in resolve_metric_ks(top_k, metric_ks):
        metrics.update({
            f"hit_at_{cutoff}": None,
            f"recall_at_{cutoff}": None,
            f"precision_at_{cutoff}": None,
            f"mrr_at_{cutoff}": None,
            f"all_requirements_hit_at_{cutoff}": None,
            f"duplicate_rate_at_{cutoff}": None,
        })
    return metrics


def _gate_metrics(decision: dict[str, Any]) -> dict[str, Any]:
    """Use the recorded decision, never empty chunks, to identify rejection."""
    source = decision.get("source")
    sufficient = decision.get("sufficient")
    disabled = decision.get("enabled") is False or source == "disabled"
    available = (
        not disabled
        and source in {"model", "rule", "fallback"}
        and isinstance(sufficient, bool)
    )
    return {
        "rejected": not sufficient if available else None,
        "source": source,
        "fallback": source == "fallback" if available else None,
        "enabled": False if disabled else (True if available else None),
    }


def evaluate_retrieval_case(
    sample: dict[str, Any],
    raw_result: dict[str, Any],
    *,
    latency_ms: float,
    top_k: int,
    match_threshold: float,
    metric_ks: Iterable[int] | None = None,
) -> dict[str, Any]:
    """Evaluate retrieval, gate decisions and delivered evidence separately."""
    resolved_metric_ks = resolve_metric_ks(top_k, metric_ks)
    raw_chunks = list(raw_result.get("chunks") or [])[:top_k]
    serialized_chunks = [
        serialize_retrieved_chunk(chunk, rank)
        for rank, chunk in enumerate(raw_chunks, start=1)
    ]
    post_gate = _evaluate_ranked_chunks(
        sample,
        raw_chunks,
        top_k=top_k,
        metric_ks=resolved_metric_ks,
        match_threshold=match_threshold,
    )
    # Missing historical snapshots are unknown, not an empty retrieval.
    before_chunks = raw_result.get("chunks_before_sufficiency")
    before = None
    serialized_before = None
    if before_chunks is not None:
        before_chunks = list(before_chunks)[:top_k]
        before = _evaluate_ranked_chunks(
            sample,
            before_chunks,
            top_k=top_k,
            metric_ks=resolved_metric_ks,
            match_threshold=match_threshold,
        )
        serialized_before = [
            serialize_retrieved_chunk(chunk, rank)
            for rank, chunk in enumerate(before_chunks, start=1)
        ]

    return {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "id": sample.get("id"),
        "question": sample.get("question"),
        "reference_answer": sample.get("reference_answer"),
        "question_type": sample.get("question_type"),
        "split": str(
            (sample.get("metadata") or {}).get("split") or "unknown"
        ),
        "source_modality": get_source_modality(sample),
        "answerable": bool(sample.get("answerable")),
        "error": None,
        "latency_ms": round(float(latency_ms), 3),
        "retrieval": {
            "metric_ks": list(resolved_metric_ks),
            "total_candidates": int(raw_result.get("total") or 0),
            "retrieved_count": len(serialized_chunks),
            "empty": not serialized_chunks,
            "chunks_before_sufficiency": serialized_before,
            "retrieved_count_before_sufficiency": (
                len(serialized_before) if serialized_before is not None else None
            ),
            "empty_before_sufficiency": (
                not serialized_before if serialized_before is not None else None
            ),
            "query_rewrite": _json_safe(
                raw_result.get("query_rewrite") or {}
            ),
            "query_intent": _json_safe(
                raw_result.get("query_intent") or {}
            ),
            "query_transform": _json_safe(
                raw_result.get("query_transform") or {}
            ),
            "retrieval_fusion": _json_safe(
                raw_result.get("retrieval_fusion") or {}
            ),
            "sequential_retrieval": _json_safe(
                raw_result.get("sequential_retrieval") or {}
            ),
            "evidence_sufficiency": _json_safe(
                raw_result.get("evidence_sufficiency") or {}
            ),
            "total_before_evidence_sufficiency": int(
                raw_result.get("total_before_evidence_sufficiency") or 0
            ),
            "chunks": serialized_chunks,
        },
        # Existing fields retain their post-gate meaning for old consumers.
        **post_gate,
        "retrieval_metrics": (
            before["metrics"]
            if before is not None
            else _unavailable_metrics(top_k, resolved_metric_ks)
        ),
        "evidence_matches_before_sufficiency": (
            before["evidence_matches"] if before is not None else None
        ),
        "post_gate_metrics": dict(post_gate["metrics"]),
        "gate_metrics": _gate_metrics(raw_result.get("evidence_sufficiency") or {}),
    }


def build_error_case(
    sample: dict[str, Any],
    *,
    error: Exception,
    latency_ms: float,
    top_k: int,
    metric_ks: Iterable[int] | None = None,
) -> dict[str, Any]:
    """Record one failed query without losing the rest of the evaluation run."""
    resolved_metric_ks = resolve_metric_ks(top_k, metric_ks)
    return {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "id": sample.get("id"),
        "question": sample.get("question"),
        "reference_answer": sample.get("reference_answer"),
        "question_type": sample.get("question_type"),
        "split": str(
            (sample.get("metadata") or {}).get("split") or "unknown"
        ),
        "source_modality": get_source_modality(sample),
        "answerable": bool(sample.get("answerable")),
        "error": f"{type(error).__name__}: {error}",
        "latency_ms": round(float(latency_ms), 3),
        "retrieval": {
            "metric_ks": list(resolved_metric_ks),
            "total_candidates": 0,
            "retrieved_count": 0,
            "empty": True,
            "chunks": [],
            "chunks_before_sufficiency": None,
            "retrieved_count_before_sufficiency": None,
            "empty_before_sufficiency": None,
        },
        "gold_evidence_count": len(sample.get("relevant_evidence") or []),
        "matched_evidence_count": 0,
        "first_relevant_rank": None,
        "evidence_matches": [],
        "metrics": _unavailable_metrics(top_k, resolved_metric_ks),
        "retrieval_metrics": _unavailable_metrics(top_k, resolved_metric_ks),
        "post_gate_metrics": _unavailable_metrics(top_k, resolved_metric_ks),
        "gate_metrics": _gate_metrics({}),
        "evidence_matches_before_sufficiency": None,
    }


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _rate(numerator: int, denominator: int) -> float | None:
    if not denominator:
        return None
    return round(numerator / denominator, 6)


def _percentile(values: Iterable[float], quantile: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * quantile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return round(ordered[lower_index], 6)
    fraction = position - lower_index
    return round(
        ordered[lower_index]
        + (ordered[upper_index] - ordered[lower_index]) * fraction,
        6,
    )


def _aggregate_stage_metrics(
    results: list[dict[str, Any]],
    *,
    top_k: int,
    before_sufficiency: bool,
    metric_ks: Iterable[int] | None = None,
) -> dict[str, Any]:
    resolved_metric_ks = resolve_metric_ks(top_k, metric_ks)
    successful = [result for result in results if not result.get("error")]
    empty_key = "empty_before_sufficiency" if before_sufficiency else "empty"
    metrics_key = "retrieval_metrics" if before_sufficiency else "metrics"
    available = [
        result for result in successful
        if isinstance(result["retrieval"].get(empty_key), bool)
        and isinstance(result.get(metrics_key), dict)
    ]
    answerable = [result for result in available if result.get("answerable")]
    unanswerable = [result for result in available if not result.get("answerable")]
    error_count = len(results) - len(successful)
    aggregated = {
        "query_count": len(results),
        "evaluated_query_count": len(available),
        "unavailable_query_count": len(successful) - len(available),
        "error_count": error_count,
        "error_rate": _rate(error_count, len(results)),
        "answerable_query_count": len(answerable),
        "unanswerable_query_count": len(unanswerable),
        "empty_rate": _mean(float(r["retrieval"][empty_key]) for r in available),
        "answerable_empty_rate": _mean(
            float(r["retrieval"][empty_key]) for r in answerable
        ),
        "unanswerable_empty_rate": _mean(
            float(r["retrieval"][empty_key]) for r in unanswerable
        ),
    }
    for cutoff in resolved_metric_ks:
        for metric_name in (
            "hit",
            "recall",
            "precision",
            "mrr",
            "all_requirements_hit",
        ):
            key = f"{metric_name}_at_{cutoff}"
            aggregated[key] = _mean(
                float(result[metrics_key][key])
                for result in answerable
                if result[metrics_key].get(key) is not None
            )

        duplicate_key = f"duplicate_rate_at_{cutoff}"
        aggregated[duplicate_key] = _mean(
            float(result[metrics_key][duplicate_key])
            for result in available
            if result[metrics_key].get(duplicate_key) is not None
        )
    return aggregated


def _aggregate_gate_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [result for result in results if not result.get("error")]
    decisions = [
        (
            result,
            _gate_metrics(result["retrieval"].get("evidence_sufficiency") or {}),
        )
        for result in successful
    ]
    available = [(r, gate) for r, gate in decisions if gate["rejected"] is not None]
    disabled_count = sum(gate["enabled"] is False for _, gate in decisions)
    answerable = [gate for r, gate in available if r.get("answerable")]
    unanswerable = [gate for r, gate in available if not r.get("answerable")]
    error_count = len(results) - len(successful)
    return {
        "query_count": len(results),
        "decision_query_count": len(available),
        "disabled_query_count": disabled_count,
        "unavailable_query_count": len(successful) - len(available) - disabled_count,
        "error_count": error_count,
        "error_rate": _rate(error_count, len(results)),
        "answerable_decision_count": len(answerable),
        "unanswerable_decision_count": len(unanswerable),
        "rejected_query_count": sum(gate["rejected"] for _, gate in available),
        "answerable_rejected_count": sum(gate["rejected"] for gate in answerable),
        "unanswerable_rejected_count": sum(gate["rejected"] for gate in unanswerable),
        "rejection_rate": _mean(float(gate["rejected"]) for _, gate in available),
        "answerable_rejection_rate": _mean(
            float(gate["rejected"]) for gate in answerable
        ),
        "unanswerable_rejection_rate": _mean(
            float(gate["rejected"]) for gate in unanswerable
        ),
        "fallback_count": sum(gate["fallback"] for _, gate in available),
        "fallback_rate": _mean(float(gate["fallback"]) for _, gate in available),
    }


def aggregate_results(
    results: list[dict[str, Any]],
    *,
    top_k: int,
    metric_ks: Iterable[int] | None = None,
) -> dict[str, Any]:
    """Keep legacy metrics and add separate stage metrics for one slice."""
    resolved_metric_ks = resolve_metric_ks(top_k, metric_ks)
    successful = [result for result in results if not result.get("error")]
    answerable = [result for result in successful if result.get("answerable")]
    unanswerable = [
        result for result in successful if not result.get("answerable")
    ]
    sufficiency_checked = [
        result
        for result in successful
        if result["retrieval"].get("evidence_sufficiency")
    ]
    hit_key = f"hit_at_{top_k}"
    recall_key = f"recall_at_{top_k}"
    mrr_key = f"mrr_at_{top_k}"
    error_count = len(results) - len(successful)
    latencies = [float(result["latency_ms"]) for result in successful]

    return {
        "query_count": len(results),
        "successful_query_count": len(successful),
        "error_count": error_count,
        "error_rate": _rate(error_count, len(results)),
        "answerable_query_count": len(answerable),
        "unanswerable_query_count": len(unanswerable),
        "empty_rate": _mean(
            float(result["retrieval"]["empty"]) for result in successful
        ),
        "answerable_empty_rate": _mean(
            float(result["retrieval"]["empty"])
            for result in answerable
        ),
        "unanswerable_empty_rate": _mean(
            float(result["retrieval"]["empty"])
            for result in unanswerable
        ),
        "unanswerable_rejection_rate": _mean(
            float(result["retrieval"]["empty"])
            for result in unanswerable
        ),
        "evidence_sufficiency_fallback_rate": _mean(
            float(
                result["retrieval"]["evidence_sufficiency"].get("source")
                == "fallback"
            )
            for result in sufficiency_checked
        ),
        "latency_sample_count": len(latencies),
        "average_latency_ms": _mean(latencies),
        "latency_p50_ms": _percentile(latencies, 0.5),
        "latency_p95_ms": _percentile(latencies, 0.95),
        hit_key: _mean(
            float(result["metrics"][hit_key]) for result in answerable
        ),
        recall_key: _mean(
            float(result["metrics"][recall_key]) for result in answerable
        ),
        mrr_key: _mean(
            float(result["metrics"][mrr_key]) for result in answerable
        ),
        "retrieval_metrics": _aggregate_stage_metrics(
            results,
            top_k=top_k,
            before_sufficiency=True,
            metric_ks=resolved_metric_ks,
        ),
        "gate_metrics": _aggregate_gate_metrics(results),
        "post_gate_metrics": _aggregate_stage_metrics(
            results,
            top_k=top_k,
            before_sufficiency=False,
            metric_ks=resolved_metric_ks,
        ),
    }


def build_summary(
    results: list[dict[str, Any]],
    *,
    top_k: int,
    metric_ks: Iterable[int] | None = None,
) -> dict[str, Any]:
    """Build overall and per-slice summaries for a completed run."""
    resolved_metric_ks = resolve_metric_ks(top_k, metric_ks)
    by_question_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_source_modality: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_question_type[str(result.get("question_type") or "unknown")].append(
            result
        )
        by_source_modality[str(result.get("source_modality") or "unknown")].append(
            result
        )
        by_split[str(result.get("split") or "unknown")].append(result)

    return {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "question_type_counts": dict(
            sorted(Counter(result["question_type"] for result in results).items())
        ),
        "source_modality_counts": dict(
            sorted(Counter(result["source_modality"] for result in results).items())
        ),
        "split_counts": dict(
            sorted(
                Counter(
                    str(result.get("split") or "unknown")
                    for result in results
                ).items()
            )
        ),
        "metric_ks": list(resolved_metric_ks),
        "overall": aggregate_results(
            results,
            top_k=top_k,
            metric_ks=resolved_metric_ks,
        ),
        "by_question_type": {
            name: aggregate_results(
                items,
                top_k=top_k,
                metric_ks=resolved_metric_ks,
            )
            for name, items in sorted(by_question_type.items())
        },
        "by_source_modality": {
            name: aggregate_results(
                items,
                top_k=top_k,
                metric_ks=resolved_metric_ks,
            )
            for name, items in sorted(by_source_modality.items())
        },
        "by_split": {
            name: aggregate_results(
                items,
                top_k=top_k,
                metric_ks=resolved_metric_ks,
            )
            for name, items in sorted(by_split.items())
        },
    }
