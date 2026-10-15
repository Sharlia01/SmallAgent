"""评估检索增强排名结果的纯辅助函数。"""

from __future__ import annotations

import math
import os
import re
import unicodedata
from decimal import Decimal, InvalidOperation
from html.parser import HTMLParser
from collections import Counter, defaultdict
from typing import Any, Iterable

from service.core.evidence.metadata import evidence_metadata


RESULT_SCHEMA_VERSION = "3.0"
RETRIEVAL_LATENCY_STAGES = (
    "query_transform",
    "initial_retrieval",
    "rerank",
    "multi_hop",
    "gate",
    "repair",
    "other",
    "total",
)


def resolve_metric_ks(
    top_k: int,
    metric_ks: Iterable[int] | None = None,
) -> tuple[int, ...]:
    """返回指定评测截断位置，默认使用实际检索结果数量。"""
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    requested = (top_k,) if metric_ks is None else tuple(metric_ks)
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
    """比较证据文本前，先消除 OCR 和排版差异。"""
    normalized = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return "".join(character for character in normalized if character.isalnum())


def normalize_document_name(name: Any) -> str:
    """将文档路径规范化为标准答案集合中保存的文件名。"""
    normalized = unicodedata.normalize("NFKC", str(name or ""))
    return os.path.basename(normalized.replace("\\", "/")).casefold()


# 此映射仅包含等价财务标签；过宽的同义词会
# 错误合并市盈率与市净率等不同指标。
_TABLE_FIELD_ALIASES = {
    "pe": {"pe", "市盈率", "市盈率pe"},
    "pb": {"pb", "市净率", "市净率pb"},
    "eps": {
        "eps",
        "eps元",
        "eps元股",
        "每股收益",
        "每股收益元",
        "每股收益元股",
    },
    "dps": {
        "dps",
        "dps元",
        "dps元股",
        "每股红利",
        "每股红利元",
        "每股红利元股",
        "每股股利",
        "每股股利元",
        "每股股利元股",
    },
    "roe": {"roe", "净资产收益率", "净资产收益率roe"},
}
_TABLE_ALIAS_LOOKUP = {
    alias: canonical
    for canonical, aliases in _TABLE_FIELD_ALIASES.items()
    for alias in aliases
}
_TABLE_NUMBER_RE = re.compile(
    r"^(?P<number>[+-]?(?:\d+(?:\.\d*)?|\.\d+))"
    r"(?P<unit>人民币元|百万元|千万元|亿元|万元|千元|元|%|倍|x)?$",
    re.IGNORECASE,
)
_CURRENCY_MULTIPLIERS = {
    "人民币元": Decimal("1"),
    "元": Decimal("1"),
    "千元": Decimal("1000"),
    "万元": Decimal("10000"),
    "百万元": Decimal("1000000"),
    "千万元": Decimal("10000000"),
    "亿元": Decimal("100000000"),
}


def _normalize_decimal(value: Decimal) -> str:
    """将等值十进制数统一表示，避免使用科学计数法。"""
    if value == 0:
        return "0"
    rendered = format(value.normalize(), "f")
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


def _table_field_alias(value: str) -> str | None:
    """返回等价财务字段标签的规范编号。"""
    key = normalize_text(value)
    canonical = _TABLE_ALIAS_LOOKUP.get(key)
    if canonical:
        return canonical

    # 复合表头中保留实际与预测期间后缀，
    # 例如 P/E 25E 和“市盈率（PE）25E”。
    period_match = re.fullmatch(r"(.+?)((?:19|20)?\d{2}[ae])", key)
    if period_match:
        canonical = _TABLE_ALIAS_LOOKUP.get(period_match.group(1))
        if canonical:
            return f"{canonical}:{period_match.group(2)}"
    return None


def _match_table_number(value: Any) -> tuple[Decimal, str | None] | None:
    """解析独立数值及可选的显示单位。"""
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    normalized = re.sub(r"\s+", "", normalized).replace(",", "")
    if (
        len(normalized) >= 2
        and normalized.startswith("(")
        and normalized.endswith(")")
    ):
        normalized = f"-{normalized[1:-1]}"

    match = _TABLE_NUMBER_RE.fullmatch(normalized)
    if not match:
        return None
    try:
        number = Decimal(match.group("number"))
    except InvalidOperation:
        return None
    return number, (match.group("unit") or "").casefold() or None


def _parse_table_number(value: Any) -> tuple[Decimal, str | None] | None:
    """解析表格数值，并转换已明确声明的货币数量级。"""
    matched = _match_table_number(value)
    if not matched:
        return None
    number, unit = matched
    if unit in _CURRENCY_MULTIPLIERS:
        return number * _CURRENCY_MULTIPLIERS[unit], "currency"
    if unit == "%":
        return number, "percent"
    if unit in {"倍", "x"}:
        return number, "multiple"
    return number, None


def normalize_table_cell(value: Any) -> str:
    """规范化表格单元格，包含等价财务字段标签。"""
    normalized = unicodedata.normalize(
        "NFKC",
        str(value or ""),
    ).casefold().strip()

    field_alias = _table_field_alias(normalized)
    if field_alias:
        return f"field:{field_alias}"

    # 财务表格通常使用括号表示负数，例如 (23181)。
    if (
        len(normalized) >= 2
        and normalized.startswith("(")
        and normalized.endswith(")")
    ):
        normalized = f"-{normalized[1:-1]}"

    # 千位分隔符不影响数值含义。
    normalized = normalized.replace(",", "")

    parsed_number = _parse_table_number(normalized)
    if parsed_number:
        number, dimension = parsed_number
        suffix = f":{dimension}" if dimension else ""
        return f"number:{_normalize_decimal(number)}{suffix}"

    return "".join(
        character
        for character in normalized
        if character.isalnum() or character in ".%+-"
    )


def _table_cells_equivalent(expected: Any, actual: Any) -> bool:
    """通过受控别名和考虑单位的数值规则比较单元格。"""
    if normalize_table_cell(expected) == normalize_table_cell(actual):
        return True

    expected_number = _parse_table_number(expected)
    actual_number = _parse_table_number(actual)
    if not expected_number or not actual_number:
        return False

    expected_value, expected_dimension = expected_number
    actual_value, actual_dimension = actual_number
    if expected_dimension == actual_dimension:
        return expected_value == actual_value

    # 标准答案可能省略表头已声明的单位，解析单元格却重复单位，
    # 如 0.40 与 0.40元。只允许一侧使用隐含单位；
    # 两侧明确但不兼容的计量维度不能匹配。
    if expected_dimension is None or actual_dimension is None:
        expected_raw = _match_table_number(expected)
        actual_raw = _match_table_number(actual)
        return bool(
            expected_raw
            and actual_raw
            and expected_raw[0] == actual_raw[0]
        )
    return False


def _table_cell_sequences_equal(
    expected_cells: list[Any],
    actual_cells: list[Any],
) -> bool:
    return len(expected_cells) == len(actual_cells) and all(
        _table_cells_equivalent(expected, actual)
        for expected, actual in zip(expected_cells, actual_cells)
    )


class _TableHTMLParser(HTMLParser):
    """从片段中的简单 HTML 表格提取题注和行。"""

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
    """计算片段对标准证据文本的覆盖程度。

    规范化后完全包含时记为 1.0；字符三元组回退允许少量 OCR 差异。
    覆盖判断具有方向性：候选须包含大部分标准证据，
    仅共享一小段词语不能视为覆盖。
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
    """确认文档身份一致后，计算片段对标准证据的得分。"""
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

    expected_cells = list(evidence.get("row_cells") or [])

    def cells_match(values):
        actual_cells = list(values or [])
        if _table_cell_sequences_equal(expected_cells, actual_cells):
            return True
        expected_length = len(expected_cells)
        return bool(expected_length) and any(
            _table_cell_sequences_equal(
                expected_cells,
                actual_cells[start:start + expected_length],
            )
            for start in range(len(actual_cells) - expected_length + 1)
        )

    structured_cells = chunk.get("table_cells_kwd")
    if expected_cells and isinstance(structured_cells, (list, tuple)):
        if cells_match(structured_cells):
            return 1.0

        # 查询时扩展父表可为回答模型提供可靠邻近行，
        # 即使最初命中的单行片段不是目标行。
        context = chunk.get("table_context_obj") or {}
        if context.get("reliable") is True and any(
            cells_match(row.get("cells"))
            for row in context.get("rows") or []
            if isinstance(row, dict)
        ):
            return 1.0

        # 新解析表格以结构化单元格为准；
        # 重复正文或 HTML 不能覆盖明确的单元格不匹配。
        return 0.0

    expected_text = evidence.get("text")
    best_score = 0.0
    content = _chunk_content(chunk)
    tables = parse_html_tables(content)

    # 部分 PDF 表格只保存为展开后的纯文本，缺少 HTML 行标签。
    # 此时用规范化证据覆盖作为回退，同时继续检查文档身份。
    if not tables:
        if not expected_text:
            return 0.0
        return evidence_text_match_score(expected_text, content)

    # 2. 解析召回chunk里的HTML表格
    for table in tables:
        for row in table.get("rows") or []:

            # 兼容原来的row_cells精确匹配
            actual_cells = list(row)

            if expected_cells:
                if _table_cell_sequences_equal(expected_cells, actual_cells):
                    return 1.0

                expected_length = len(expected_cells)
                for start in range(
                    len(actual_cells) - expected_length + 1
                ):
                    if _table_cell_sequences_equal(
                        expected_cells,
                        actual_cells[start:start + expected_length],
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
    """按同一文档中的题注或定位信息匹配检索图表。"""
    evidence_document = normalize_document_name(evidence.get("document_name"))
    chunk_document = normalize_document_name(_chunk_document_name(chunk))
    if not evidence_document or evidence_document != chunk_document:
        return 0.0

    # 图表标准答案常描述 OCR 无法恢复的视觉事实。
    # 优先用定位信息判断应召回哪张图；
    # 旧数据集缺少定位信息时，再用文本作为回退。
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
    """评估可选证据组合，允许各证据单元分别匹配不同片段。"""
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


def _stage_latency_snapshot(raw_result: dict[str, Any]) -> dict[str, float | None]:
    timings = raw_result.get("stage_latency_ms") or {}
    return {
        stage: (
            round(value, 3)
            if (value := _optional_float(timings.get(stage))) is not None
            and value >= 0
            else None
        )
        for stage in RETRIEVAL_LATENCY_STAGES
    }


def _json_safe(value: Any) -> Any:
    """递归转换类似 NumPy 的数值，不需要直接导入 NumPy。"""
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
    """保留排名诊断信息，省略体积较大的向量字段。"""
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
        "parallel_subqueries": _json_safe(
            chunk.get("parallel_subqueries") or []
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
    """计算规范化后的精确重复比例；结果集为空时视为未知。"""
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
    """使用完全相同的证据规则评估不同检索阶段。"""
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


def _with_gate_attribution(
    gate: dict[str, Any],
    *,
    answerable: bool | None = None,
    pre_gate_evidence_sufficient: bool | None = None,
) -> dict[str, Any]:
    """将拒绝原因归于上游检索或门控本身。"""
    attributed = dict(gate)
    evidence_sufficient = (
        pre_gate_evidence_sufficient
        if answerable is True
        and isinstance(pre_gate_evidence_sufficient, bool)
        else None
    )
    rejection_attribution = None
    if attributed.get("rejected") is True:
        if answerable is False:
            rejection_attribution = "correct_unanswerable_rejection"
        elif evidence_sufficient is True:
            rejection_attribution = "false_rejection_with_sufficient_evidence"
        elif evidence_sufficient is False:
            rejection_attribution = (
                "correct_rejection_after_insufficient_evidence"
            )

    attributed.update({
        "pre_gate_evidence_sufficient": evidence_sufficient,
        "rejection_attribution": rejection_attribution,
    })
    return attributed


def _gate_metrics(
    decision: dict[str, Any],
    *,
    answerable: bool | None = None,
    pre_gate_evidence_sufficient: bool | None = None,
) -> dict[str, Any]:
    """根据已记录的判定识别拒绝原因，不能仅依据片段列表为空推断。"""
    source = decision.get("source")
    sufficient = decision.get("sufficient")
    disabled = decision.get("enabled") is False or source == "disabled"
    available = (
        not disabled
        and source in {"model", "rule", "fallback"}
        and isinstance(sufficient, bool)
    )
    return _with_gate_attribution(
        {
            "rejected": not sufficient if available else None,
            "source": source,
            "fallback": source == "fallback" if available else None,
            "enabled": False if disabled else (True if available else None),
        },
        answerable=answerable,
        pre_gate_evidence_sufficient=pre_gate_evidence_sufficient,
    )


def evaluate_retrieval_case(
    sample: dict[str, Any],
    raw_result: dict[str, Any],
    *,
    latency_ms: float,
    top_k: int,
    match_threshold: float,
    metric_ks: Iterable[int] | None = None,
) -> dict[str, Any]:
    """分别评估检索结果、门控决策和最终交付证据。"""
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
    # 缺少历史快照时视为未知，不能当成检索结果为空。
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

    answerable = bool(sample.get("answerable"))
    recall_key = f"recall_at_{top_k}"
    pre_gate_recall = (
        before["metrics"].get(recall_key) if before is not None else None
    )
    pre_gate_evidence_sufficient = (
        pre_gate_recall == 1.0 if pre_gate_recall is not None else None
    )

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
        "answerable": answerable,
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
            "stage_latency_ms": _stage_latency_snapshot(raw_result),
            "stage_latency_semantics": _json_safe(raw_result.get("stage_latency_semantics") or {}),
            "expected_retrieval": _json_safe(sample.get("expected_retrieval") or {}),
            "query_rewrite": _json_safe(
                raw_result.get("query_rewrite") or {}
            ),
            "query_intent": _json_safe(
                raw_result.get("query_intent") or {}
            ),
            "query_understanding": _json_safe(raw_result.get("query_understanding") or {}),
            "financial_requirements": _json_safe(raw_result.get("financial_requirements")),
            "financial_bindings": _json_safe(raw_result.get("financial_bindings") or []),
            "retrieval_task": _json_safe(raw_result.get("retrieval_task")),
            "query_contract": _json_safe(raw_result.get("query_contract") or {}),
            "query_runtime": _json_safe(raw_result.get("query_runtime") or {}),
            "plan_execution": _json_safe(raw_result.get("plan_execution") or {}),
            "retrieval_graph": _json_safe(raw_result.get("retrieval_graph") or {}),
            "query_repair": _json_safe(raw_result.get("query_repair") or {}),
            "agent_execution": _json_safe(raw_result.get("agent_execution") or {}),
            "retrieval_route": _json_safe(
                raw_result.get("retrieval_route") or {}
            ),
            "retrieval_plan": _json_safe(
                raw_result.get("retrieval_plan") or {}
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
            "parallel_retrieval": _json_safe(
                raw_result.get("parallel_retrieval") or {}
            ),
            "evidence_sufficiency": _json_safe(
                raw_result.get("evidence_sufficiency") or {}
            ),
            "total_before_evidence_sufficiency": int(
                raw_result.get("total_before_evidence_sufficiency") or 0
            ),
            "chunks": serialized_chunks,
        },
        "gold_evidence_count": post_gate["gold_evidence_count"],
        "matched_evidence_count": post_gate["matched_evidence_count"],
        "first_relevant_rank": post_gate["first_relevant_rank"],
        "evidence_matches": post_gate["evidence_matches"],
        "retrieval_metrics": (
            before["metrics"]
            if before is not None
            else _unavailable_metrics(top_k, resolved_metric_ks)
        ),
        "evidence_matches_before_sufficiency": (
            before["evidence_matches"] if before is not None else None
        ),
        "post_gate_metrics": dict(post_gate["metrics"]),
        "gate_metrics": _gate_metrics(
            raw_result.get("evidence_sufficiency") or {},
            answerable=answerable,
            pre_gate_evidence_sufficient=pre_gate_evidence_sufficient,
        ),
    }


def build_error_case(
    sample: dict[str, Any],
    *,
    error: Exception,
    latency_ms: float,
    top_k: int,
    metric_ks: Iterable[int] | None = None,
) -> dict[str, Any]:
    """记录单条查询失败，同时保留本轮评测的其他结果。"""
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
            "query_runtime": _json_safe(getattr(error, "query_runtime", {}) or {}),
            "metric_ks": list(resolved_metric_ks),
            "total_candidates": 0,
            "retrieved_count": 0,
            "empty": True,
            "chunks": [],
            "chunks_before_sufficiency": None,
            "retrieved_count_before_sufficiency": None,
            "empty_before_sufficiency": None,
            "stage_latency_ms": {
                stage: None for stage in RETRIEVAL_LATENCY_STAGES
            },
        },
        "gold_evidence_count": len(sample.get("relevant_evidence") or []),
        "matched_evidence_count": 0,
        "first_relevant_rank": None,
        "evidence_matches": [],
        "retrieval_metrics": _unavailable_metrics(top_k, resolved_metric_ks),
        "post_gate_metrics": _unavailable_metrics(top_k, resolved_metric_ks),
        "gate_metrics": _gate_metrics(
            {},
            answerable=bool(sample.get("answerable")),
        ),
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


def _sample_ids(
    results: Iterable[dict[str, Any]],
    predicate,
) -> list[str]:
    """返回满足 predicate 条件的稳定、去重样本编号。"""
    sample_ids = []
    seen = set()
    for result in results:
        sample_id = str(result.get("id") or "").strip()
        if not sample_id or sample_id in seen or not predicate(result):
            continue
        seen.add(sample_id)
        sample_ids.append(sample_id)
    return sample_ids


def _result_gate_metrics(
    result: dict[str, Any],
    *,
    top_k: int,
) -> dict[str, Any]:
    """读取门控判定，并在证据允许时推导拒绝原因归属。"""
    stored = result.get("gate_metrics")
    if isinstance(stored, dict) and (
        isinstance(stored.get("rejected"), bool)
        or stored.get("enabled") is False
    ):
        gate = stored
    else:
        gate = _gate_metrics(
            result.get("retrieval", {}).get("evidence_sufficiency") or {}
        )

    retrieval_metrics = result.get("retrieval_metrics")
    pre_gate_recall = (
        retrieval_metrics.get(f"recall_at_{top_k}")
        if isinstance(retrieval_metrics, dict)
        else None
    )
    return _with_gate_attribution(
        gate,
        answerable=result.get("answerable"),
        pre_gate_evidence_sufficient=(
            pre_gate_recall == 1.0
            if pre_gate_recall is not None
            else None
        ),
    )


def _stage_non_perfect_sample_ids(
    results: list[dict[str, Any]],
    *,
    top_k: int,
    before_sufficiency: bool,
    metric_ks: Iterable[int] | None = None,
) -> dict[str, list[str]]:
    """将未达到理想值的阶段指标映射到对应样本。"""
    resolved_metric_ks = resolve_metric_ks(top_k, metric_ks)
    empty_key = "empty_before_sufficiency" if before_sufficiency else "empty"

    def stage_metrics(result: dict[str, Any]) -> dict[str, Any] | None:
        if before_sufficiency:
            metrics = result.get("retrieval_metrics")
        else:
            # 继续兼容历史 2.1 版本记录，但 2.2 输出不再生成旧别名。
            metrics = result.get("post_gate_metrics") or result.get("metrics")
        return metrics if isinstance(metrics, dict) else None

    successful = [result for result in results if not result.get("error")]
    available = [
        result
        for result in successful
        if isinstance(result.get("retrieval", {}).get(empty_key), bool)
        and stage_metrics(result) is not None
    ]
    answerable = [result for result in available if result.get("answerable")]
    failures: dict[str, list[str]] = {
        "error_rate": _sample_ids(results, lambda result: bool(result.get("error"))),
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
            failures[key] = _sample_ids(
                answerable,
                lambda result, key=key: (
                    stage_metrics(result).get(key) is not None
                    and float(stage_metrics(result)[key]) < 1.0
                ),
            )

        duplicate_key = f"duplicate_rate_at_{cutoff}"
        failures[duplicate_key] = _sample_ids(
            available,
            lambda result, key=duplicate_key: (
                stage_metrics(result).get(key) is not None
                and float(stage_metrics(result)[key]) > 0.0
            ),
        )
    return failures


def _gate_non_perfect_sample_ids(
    results: list[dict[str, Any]],
    *,
    top_k: int,
) -> dict[str, list[str]]:
    """返回具有明确理想方向的门控指标所对应的样本。"""
    successful = [result for result in results if not result.get("error")]
    decisions = [
        (result, _result_gate_metrics(result, top_k=top_k))
        for result in successful
    ]
    available = [
        (result, gate)
        for result, gate in decisions
        if gate["rejected"] is not None
    ]
    answerable = [result for result, _gate in available if result.get("answerable")]
    unanswerable = [
        result for result, _gate in available if not result.get("answerable")
    ]
    gates_by_id = {
        str(result.get("id") or ""): gate
        for result, gate in available
    }
    return {
        "error_rate": _sample_ids(results, lambda result: bool(result.get("error"))),
        "answerable_rejection_rate": _sample_ids(
            answerable,
            lambda result: gates_by_id[str(result.get("id") or "")]["rejected"]
            is True,
        ),
        "correct_rejection_after_insufficient_evidence_rate": _sample_ids(
            answerable,
            lambda result: (
                gates_by_id[str(result.get("id") or "")][
                    "pre_gate_evidence_sufficient"
                ] is False
                and gates_by_id[str(result.get("id") or "")]["rejected"]
                is not True
            ),
        ),
        "false_rejection_with_sufficient_evidence_rate": _sample_ids(
            answerable,
            lambda result: (
                gates_by_id[str(result.get("id") or "")][
                    "rejection_attribution"
                ] == "false_rejection_with_sufficient_evidence"
            ),
        ),
        "unanswerable_rejection_rate": _sample_ids(
            unanswerable,
            lambda result: gates_by_id[str(result.get("id") or "")]["rejected"]
            is not True,
        ),
        "fallback_rate": _sample_ids(
            (result for result, _gate in available),
            lambda result: gates_by_id[str(result.get("id") or "")]["fallback"]
            is True,
        ),
    }


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


def _aggregate_stage_latencies(
    results: list[dict[str, Any]],
) -> dict[str, dict[str, float | int | None]]:
    """聚合各自可用且不重叠的阶段耗时。"""
    successful = [result for result in results if not result.get("error")]
    aggregated = {}
    for stage in RETRIEVAL_LATENCY_STAGES:
        values = []
        for result in successful:
            value = _optional_float(
                result.get("retrieval", {})
                .get("stage_latency_ms", {})
                .get(stage)
            )
            if value is not None and value >= 0:
                values.append(value)
        aggregated[stage] = {
            "sample_count": len(values),
            "average_ms": _mean(values),
            "p50_ms": _percentile(values, 0.5),
            "p95_ms": _percentile(values, 0.95),
        }
    return aggregated


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

    def stage_metrics(result: dict[str, Any]) -> dict[str, Any] | None:
        if before_sufficiency:
            metrics = result.get("retrieval_metrics")
        else:
            metrics = result.get("post_gate_metrics") or result.get("metrics")
        return metrics if isinstance(metrics, dict) else None

    available = [
        result for result in successful
        if isinstance(result.get("retrieval", {}).get(empty_key), bool)
        and stage_metrics(result) is not None
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
                float(stage_metrics(result)[key])
                for result in answerable
                if stage_metrics(result).get(key) is not None
            )

        duplicate_key = f"duplicate_rate_at_{cutoff}"
        aggregated[duplicate_key] = _mean(
            float(stage_metrics(result)[duplicate_key])
            for result in available
            if stage_metrics(result).get(duplicate_key) is not None
        )
    return aggregated


def _aggregate_gate_metrics(
    results: list[dict[str, Any]],
    *,
    top_k: int,
) -> dict[str, Any]:
    successful = [result for result in results if not result.get("error")]
    decisions = [
        (
            result,
            _result_gate_metrics(result, top_k=top_k),
        )
        for result in successful
    ]
    available = [
        (result, gate)
        for result, gate in decisions
        if gate["rejected"] is not None
    ]
    disabled_count = sum(gate["enabled"] is False for _, gate in decisions)
    answerable_pairs = [
        (result, gate)
        for result, gate in available
        if result.get("answerable")
    ]
    answerable = [gate for _, gate in answerable_pairs]
    unanswerable = [gate for r, gate in available if not r.get("answerable")]
    evidence_sufficient = [
        gate for _, gate in answerable_pairs
        if gate["pre_gate_evidence_sufficient"] is True
    ]
    evidence_insufficient = [
        gate for _, gate in answerable_pairs
        if gate["pre_gate_evidence_sufficient"] is False
    ]
    attribution_unavailable = [
        gate for _, gate in answerable_pairs
        if gate["pre_gate_evidence_sufficient"] is None
    ]
    correct_insufficient_rejections = sum(
        gate["rejection_attribution"]
        == "correct_rejection_after_insufficient_evidence"
        for gate in evidence_insufficient
    )
    false_sufficient_rejections = sum(
        gate["rejection_attribution"]
        == "false_rejection_with_sufficient_evidence"
        for gate in evidence_sufficient
    )
    attributed_answerable_rejections = (
        correct_insufficient_rejections + false_sufficient_rejections
    )
    unattributed_answerable_rejections = sum(
        gate["rejected"] is True for gate in attribution_unavailable
    )
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
        "answerable_rejected_evidence_attribution_count": (
            attributed_answerable_rejections
        ),
        "answerable_rejected_evidence_attribution_unavailable_count": (
            unattributed_answerable_rejections
        ),
        "rejection_rate": _mean(float(gate["rejected"]) for _, gate in available),
        "answerable_rejection_rate": _mean(
            float(gate["rejected"]) for gate in answerable
        ),
        "evidence_sufficient_answerable_decision_count": len(
            evidence_sufficient
        ),
        "evidence_insufficient_answerable_decision_count": len(
            evidence_insufficient
        ),
        "evidence_attribution_unavailable_answerable_decision_count": len(
            attribution_unavailable
        ),
        "correct_rejection_after_insufficient_evidence_count": (
            correct_insufficient_rejections
        ),
        "correct_rejection_after_insufficient_evidence_rate": _rate(
            correct_insufficient_rejections,
            len(evidence_insufficient),
        ),
        "false_rejection_with_sufficient_evidence_count": (
            false_sufficient_rejections
        ),
        "false_rejection_with_sufficient_evidence_rate": _rate(
            false_sufficient_rejections,
            len(evidence_sufficient),
        ),
        "unanswerable_rejection_rate": _mean(
            float(gate["rejected"]) for gate in unanswerable
        ),
        "fallback_count": sum(gate["fallback"] for _, gate in available),
        "fallback_rate": _mean(float(gate["fallback"]) for _, gate in available),
    }


def aggregate_query_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """使用明确分母计算运行指标，不推测调用费用。

    路由一致性仅用于与标注结果对照，不代表回答质量。
    调用次数读取预算计数；不再从逐步事件中统计词元或流程状态。
    """
    retrievals = [item.get("retrieval") or {} for item in results]
    runtimes = [r["query_runtime"] for r in retrievals if r.get("query_runtime")]
    plans = [r["retrieval_plan"] for r in retrievals if r.get("retrieval_plan")]
    analyses = [r["query_intent"] for r in retrievals if r.get("query_intent")]
    executions = [r["plan_execution"] for r in retrievals if r.get("plan_execution")]
    executed_requests = [item for item in executions if type(item.get("requested_plan_execution_success")) is bool]
    repair_cohorts = {}
    for answerable in (True, False):
        cohort = [(item.get("retrieval") or {}).get("query_repair") or {} for item in results
                  if item.get("answerable") is answerable]
        attempts = [item for item in cohort if item.get("attempted_goal_ids")]
        repair_cohorts["answerable" if answerable else "unanswerable"] = {
            "sample_count": len(cohort), "attempt_count": len(attempts),
            "goals_satisfied_count": sum(item.get("stop_reason") == "goals_satisfied" for item in attempts),
            "success_rate": _rate(sum(item.get("stop_reason") == "goals_satisfied" for item in attempts), len(attempts)),
        }
    annotated = [r for r in retrievals if (r.get("expected_retrieval") or {}).get("mode")
                 and (r.get("retrieval_route") or {}).get("mode")]
    repairs = [r["query_repair"] for r in retrievals if (r.get("query_repair") or {}).get("attempted_goal_ids")]
    coverage = [goal for r in retrievals for goal in (r.get("evidence_sufficiency") or {}).get("goal_coverage", [])]
    financial = [(r.get("evidence_sufficiency") or {}).get("financial_fact_bundle") or {} for r in retrievals]
    return {
        "runtime_sample_count": len(runtimes),
        "runtime_unavailable_count": len(results) - len(runtimes),
        "average_model_calls": _mean(float(r.get("counts", {}).get("model", 0)) for r in runtimes),
        "average_retrieval_calls": _mean(float(r.get("counts", {}).get("retrieval", 0)) for r in runtimes),
        "plan_sample_count": len(plans),
        "analysis_sample_count": len(analyses),
        "analysis_sources": dict(Counter(item.get("source", "unknown") for item in analyses)),
        "analysis_fallback_rate": _rate(sum(item.get("source") == "fallback" for item in analyses), len(analyses)),
        "requested_plan_sample_count": len(executed_requests),
        "requested_plan_execution_unavailable_count": len(retrievals) - len(executed_requests),
        "requested_plan_execution_success_rate": _rate(
            sum(item.get("requested_plan_execution_success") is True for item in executed_requests), len(executed_requests)),
        "plan_fallback_rate": _rate(sum(p.get("source") == "fallback" for p in plans), len(plans)),
        "rule_plan_fallback_count": sum(p.get("source") == "rule" and bool(p.get("fallback_reason")) for p in plans),
        "annotated_route_count": len(annotated),
        "route_agreement": _mean(float(r["retrieval_route"]["mode"] == r["expected_retrieval"]["mode"]) for r in annotated),
        "evaluated_goal_count": len(coverage),
        "supported_goal_rate": _rate(sum(g.get("status") == "supported" for g in coverage), len(coverage)),
        "repair_count": len(repairs),
        "repair_success_rate": _rate(sum(r.get("stop_reason") == "goals_satisfied" for r in repairs), len(repairs)),
        "repair_cohorts": repair_cohorts,
        "verification_unavailable_count": sum((r.get("evidence_sufficiency") or {}).get("verification_status") == "unavailable" for r in retrievals),
        "financial_requirement_statuses": dict(Counter(item["status"] for bundle in financial
                                                       for item in bundle.get("requirement_results", []))),
        "financial_relationship_statuses": dict(Counter(item["status"] for bundle in financial
                                                        for item in bundle.get("relationship_results", []))),
        "financial_verification_unavailable_count": sum(bundle.get("verification_status") == "unavailable" for bundle in financial),
    }


def aggregate_retrieval_stages(
    results: list[dict[str, Any]],
    *,
    top_k: int,
    metric_ks: Iterable[int] | None = None,
) -> dict[str, Any]:
    """在检索评测与端到端记录之间共享检索和门控汇总。

    读取已经保存的单查询得分，不依赖执行器的延迟格式，也不重新检索。
    缺失的阶段数据仍标记为不可用。
    """
    resolved_metric_ks = resolve_metric_ks(top_k, metric_ks)
    aggregated = {
        "retrieval_metrics": _aggregate_stage_metrics(
            results,
            top_k=top_k,
            before_sufficiency=True,
            metric_ks=resolved_metric_ks,
        ),
        "gate_metrics": _aggregate_gate_metrics(results, top_k=top_k),
        "post_gate_metrics": _aggregate_stage_metrics(
            results,
            top_k=top_k,
            before_sufficiency=False,
            metric_ks=resolved_metric_ks,
        ),
    }
    post_gate_failures = _stage_non_perfect_sample_ids(
        results,
        top_k=top_k,
        before_sufficiency=False,
        metric_ks=resolved_metric_ks,
    )
    aggregated["non_perfect_sample_ids"] = {
        "error_rate": post_gate_failures["error_rate"],
        "retrieval_metrics": _stage_non_perfect_sample_ids(
            results,
            top_k=top_k,
            before_sufficiency=True,
            metric_ks=resolved_metric_ks,
        ),
        "gate_metrics": _gate_non_perfect_sample_ids(
            results,
            top_k=top_k,
        ),
        "post_gate_metrics": post_gate_failures,
    }
    return aggregated


def aggregate_results(
    results: list[dict[str, Any]],
    *,
    top_k: int,
    metric_ks: Iterable[int] | None = None,
) -> dict[str, Any]:
    """聚合运行统计及三个独立指标阶段的结果。"""
    successful = [result for result in results if not result.get("error")]
    answerable = [result for result in successful if result.get("answerable")]
    unanswerable = [
        result for result in successful if not result.get("answerable")
    ]
    error_count = len(results) - len(successful)
    latencies = [float(result["latency_ms"]) for result in successful]
    return {
        "query_count": len(results),
        "successful_query_count": len(successful),
        "error_count": error_count,
        "error_rate": _rate(error_count, len(results)),
        "answerable_query_count": len(answerable),
        "unanswerable_query_count": len(unanswerable),
        "latency_sample_count": len(latencies),
        "average_latency_ms": _mean(latencies),
        "latency_p50_ms": _percentile(latencies, 0.5),
        "latency_p95_ms": _percentile(latencies, 0.95),
        "latency_p99_ms": _percentile(latencies, 0.99),
        "stage_latency_ms": _aggregate_stage_latencies(results),
        "query_metrics": aggregate_query_metrics(results),
        **aggregate_retrieval_stages(
            results,
            top_k=top_k,
            metric_ks=metric_ks,
        ),
    }


def build_summary(
    results: list[dict[str, Any]],
    *,
    top_k: int,
    metric_ks: Iterable[int] | None = None,
) -> dict[str, Any]:
    """为已完成评测生成整体汇总和分组汇总。"""
    resolved_metric_ks = resolve_metric_ks(top_k, metric_ks)
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
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "question_type_counts": dict(
            sorted(Counter(result["question_type"] for result in results).items())
        ),
        "source_modality_counts": dict(
            sorted(Counter(result["source_modality"] for result in results).items())
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
    }
