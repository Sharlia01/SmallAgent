"""为财务模型提供有限、可回溯的正文和原始表格片段。"""

from dataclasses import dataclass, field
import re

from domain.evidence import fingerprint
from service.core.finance.rules import document_subject, VALUE_CELL, UNIT


@dataclass
class ModelSources:
    records: list[dict] = field(default_factory=list)
    by_id: dict[str, dict] = field(default_factory=dict)
    chunks: dict = field(default_factory=dict)
    incomplete: bool = False
    unavailable_tables: list[str] = field(default_factory=list)

    def add(self, chunk, kind, text, **location):
        if not text:
            return
        identifier = fingerprint((chunk.id, kind, location))[:24]
        record = dict(source_id=identifier, chunk_id=chunk.id, kind=kind, text=str(text), **location)
        self.records.append(record)
        self.by_id[identifier] = record
        self.chunks[chunk.id] = chunk


def build_sources(chunks, *, max_chunks=3, max_chars=6000, registry=None):
    """表格只暴露可靠原始网格；模型重建结果不能给自己提供独立证明。"""
    result = ModelSources()
    for chunk in chunks:
        if len(result.chunks) >= max_chunks:
            result.incomplete = True
            break
        if chunk.structure.kind == "figure":
            continue
        table = chunk.structure.table
        is_table = chunk.structure.kind == "table" or bool(table.rows)
        if not is_table:
            result.add(chunk, "text", chunk.text[:max_chars])
            result.incomplete |= len(chunk.text) > max_chars
            continue
        if (table.binding != "bound" or not table.reliable or table.binding_source != "deterministic"
                or not any(row.grounded for row in table.rows)):
            result.unavailable_tables.append(chunk.id)
            continue
        if registry is not None:
            subject, proof = document_subject(chunk, registry)
            if subject:
                for index, item in enumerate(proof):
                    result.add(chunk, "document_subject", item["quote"], proof_index=index)
        result.add(chunk, "table_caption", table.caption)
        remaining = max_chars - len(table.caption)
        seen = set()
        for row in table.rows:
            if not row.grounded or row.index in seen:
                continue
            seen.add(row.index)
            original = row.original()
            cost = sum(len(str(value)) for value in (*original.headers, *original.cells)) + len(row.context) + len(row.preamble)
            if cost > remaining:
                result.incomplete = True
                break
            remaining -= cost
            result.add(chunk, "table_context", row.context, row=row.index)
            result.add(chunk, "table_preamble", row.preamble, row=row.index)
            for column, (header, cell) in enumerate(zip(original.headers, original.cells)):
                result.add(chunk, "table_header", header, row=row.index, column=column)
                result.add(chunk, "table_cell", cell, row=row.index, column=column)
    return result


def checked_reference(raw, sources):
    """只接受服务器提供的片段及精确字符区间，模型不能指定任意外部来源。"""
    if not isinstance(raw, dict):
        raise ValueError("invalid_reference")
    identifier, start, end, quote = (raw.get(key) for key in ("source_id", "start", "end", "quote"))
    source = sources.by_id.get(identifier) if isinstance(identifier, str) else None
    if (source is None or type(start) is not int or type(end) is not int or not isinstance(quote, str)
            or not 0 <= start < end <= len(source["text"]) or source["text"][start:end] != quote):
        raise ValueError("reference_not_grounded")
    return dict(source_id=identifier, chunk_id=source["chunk_id"], kind=source["kind"],
                start=start, end=end, quote=quote,
                **{key: source[key] for key in ("row", "column") if key in source})


def check_field_evidence(raw, sources):
    if not isinstance(raw, dict) or len(raw) > 12:
        raise ValueError("invalid_field_evidence")
    fields = {}
    for field_name, references in raw.items():
        if not isinstance(references, list) or not 1 <= len(references) <= 4:
            raise ValueError("invalid_field_references")
        fields[field_name] = [checked_reference(ref, sources) for ref in references]
    return fields


def validate_value_reference(value, reference, sources):
    """数值必须逐字对应一个完整数值，不能把 120 中的 20 截取出来。"""
    text = sources.by_id[reference["source_id"]]["text"]
    start, end = reference["start"], reference["end"]
    if reference["quote"] != value or VALUE_CELL.fullmatch(value) is None:
        raise ValueError("numeric_value_not_grounded")
    if (re.search(r"\d[.,，]?$|[+\-−]$", text[:start])
            or re.match(r"\d|[.,，]\d", text[end:])):
        raise ValueError("partial_numeric_value")


def validate_unit_reference(unit, references, value_ref, sources):
    """单位必须完整存在，不能从“百万元”中截取“万元”。"""
    grounded = any(match.group() == unit and ref["start"] <= match.start() and match.end() <= ref["end"]
                   for ref in references for match in re.finditer(UNIT, sources.by_id[ref["source_id"]]["text"]))
    if not grounded:
        raise ValueError("unit_not_grounded")
    text = sources.by_id[value_ref["source_id"]]["text"]
    adjacent = re.match(r"\s*(" + UNIT + ")", text[value_ref["end"]:])
    if adjacent and adjacent[1] != unit:
        raise ValueError("explicit_unit_conflict")


def legacy_evidence(candidate, sources):
    """兼容旧候选协议，转换后仍执行相同的原文与数值检查。"""
    source = next((item for item in sources.records if item["chunk_id"] == candidate.get("chunk_id")
                   and item["kind"] == "text"), None)
    if source is None:
        raise ValueError("missing_text_source")
    raw = dict(source_id=source["source_id"], **{key: candidate.get(key) for key in ("start", "end", "quote")})
    ref = checked_reference(raw, sources)
    value = candidate.get("raw_value", "")
    positions = list(re.finditer(r"(?<![\d.])" + re.escape(value) + r"(?![\d.])", ref["quote"])) if value else []
    if len(positions) != 1:
        raise ValueError("ambiguous_numeric_location")
    start = ref["start"] + positions[0].start()
    value_ref = checked_reference(dict(source_id=ref["source_id"], start=start, end=start+len(value), quote=value), sources)
    return {**{key: [ref] for key in ("entity", "metric", "period", "unit")}, "value": [value_ref]}
