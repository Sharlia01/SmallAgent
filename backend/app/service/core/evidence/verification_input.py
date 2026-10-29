"""核验专用输入白名单；完整来源仍保留在 Evidence 中，不展开内部元数据。"""

from collections.abc import Iterable
from typing import Any, NotRequired, TypedDict

from domain.evidence import Evidence, fingerprint, plain
from service.core.evidence.table_views import table_source_notes


class VerificationTableRow(TypedDict):
    row_index: int | None
    headers: list[str]
    cells: list[str]
    column_indices: list[int]
    context: NotRequired[str]
    preamble: NotRequired[str]
    source_row: NotRequired[dict[str, list[str]]]


class VerificationTable(TypedDict):
    caption: str
    binding: str
    reliable: bool
    rows: list[VerificationTableRow]
    source_notes: list[str]


class VerificationRecord(TypedDict):
    index: int
    chunk_id: str
    document_id: str
    document_name: str
    content: NotRequired[str]
    evidence_type_kwd: str
    page_num_int: NotRequired[Any]
    locator_kwd: NotRequired[str]
    table_binding_kwd: NotRequired[str]
    visual_status_kwd: NotRequired[str]
    table: NotRequired[VerificationTable]


def build_verification_record(evidence: Evidence, *, text: str, index: int) -> VerificationRecord:
    """表格仅发送结构，其他证据保留正文；不读取旧候选行。"""
    structure = evidence.structure
    result: VerificationRecord = {
        "index": index, "chunk_id": evidence.id,
        "document_id": evidence.source.document_id, "document_name": evidence.source.name,
        "evidence_type_kwd": structure.kind,
    }
    if evidence.location.page_number is not None:
        result["page_num_int"] = plain(evidence.location.page_number)
    if structure.locator:
        result["locator_kwd"] = structure.locator
    if structure.kind == "figure":
        result["visual_status_kwd"] = structure.figure.status
    if structure.kind != "table":
        result["content"] = text
        return result

    table = structure.table
    rows: list[VerificationTableRow] = []
    for row in table.rows:
        # 分别保存数组，不用 zip 截短损坏结构；可靠性状态明确告知模型。
        item: VerificationTableRow = {
            "row_index": row.index, "headers": list(row.headers), "cells": list(row.cells),
            "column_indices": list(row.columns),
        }
        if row.context:
            item["context"] = row.context
        if row.preamble:
            item["preamble"] = row.preamble
        if (not row.grounded or row.headers != row.original_headers
                or row.cells != row.original_cells):
            item["source_row"] = {"headers": list(row.original_headers), "cells": list(row.original_cells)}
        rows.append(item)
    notes = table_source_notes(table)
    # 兼容显式脚注；未知内部字段不会自动进入模型输入。
    footnotes = table.metadata.get("footnotes")
    if isinstance(footnotes, str) and footnotes.strip():
        notes.append(footnotes)
    elif isinstance(footnotes, (list, tuple)):
        notes.extend(note for note in footnotes if isinstance(note, str) and note.strip())
    result["table_binding_kwd"] = table.binding
    result["table"] = {
        "caption": table.caption, "binding": table.binding,
        "reliable": bool(table.binding == "bound" and table.reliable
                         and table.rows and all(row.grounded for row in table.rows)),
        "rows": rows, "source_notes": list(dict.fromkeys(notes)),
    }
    return result


def _table_source_key(evidence: Evidence) -> tuple[str, ...] | None:
    """只有来源完整的同版父表才允许跨候选共享行和表头。"""
    values = (evidence.source.knowledge_base_id, evidence.source.document_id,
              evidence.location.table_id,
              evidence.structure.table.metadata.get("table_object_ref_kwd"))
    if not all(isinstance(value, str) and value.strip() for value in values):
        return None
    return values


def build_verification_evidence(items: Iterable[tuple[Evidence, str]]) -> dict[str, Any]:
    """仅对本次送审集合建立公共行与表头，保留每条证据的原索引。"""
    candidates, table_rows, table_headers = [], {}, {}
    row_refs, header_refs = {}, {}

    def register_headers(scope, headers, columns):
        # 列窗口也属于表头身份；同名表头不能掩盖不同的原始列位置。
        value = {"labels": headers, "column_indices": columns}
        key = (scope, fingerprint(value))
        if key not in header_refs:
            reference = f"h{len(table_headers) + 1}"
            header_refs[key] = reference
            table_headers[reference] = value
        return header_refs[key]

    for index, (evidence, text) in enumerate(items, 1):
        record: dict[str, Any] = dict(build_verification_record(evidence, text=text, index=index))
        candidates.append(record)
        if "table" not in record:
            continue
        table = record["table"]
        source_key = _table_source_key(evidence) if table["reliable"] else None
        # 来源不完整或结构不可靠时，将共享范围限制在当前候选内。
        scope = ("source", source_key) if source_key else ("candidate", index)
        references = []
        for position, row in enumerate(table["rows"]):
            row_index = row["row_index"]
            # 比较完整送审行，包括原始列、上下文和前置说明；不合并差异内容。
            row_scope = scope if source_key and type(row_index) is int else ("row", index, position)
            key = (row_scope, fingerprint(row))
            reference = row_refs.get(key)
            if reference is None:
                reference = f"r{len(table_rows) + 1}"
                row_refs[key] = reference
                shared_row = {name: value for name, value in row.items()
                              if name not in {"headers", "column_indices", "source_row"}}
                shared_row["header_ref"] = register_headers(scope, row["headers"], row["column_indices"])
                if "source_row" in row:
                    original = row["source_row"]
                    shared_row["source_row"] = {
                        "header_ref": register_headers(scope, original["headers"],
                                                       list(range(len(original["headers"])))),
                        "cells": original["cells"],
                    }
                table_rows[reference] = shared_row
            if reference not in references:
                references.append(reference)
        # 使用新字典替换发送视图，不修改 Evidence 或中间行对象。
        record["table"] = {name: value for name, value in table.items() if name != "rows"} | {
            "row_refs": references,
        }
    payload = {"candidate_evidence": candidates}
    if table_rows:
        payload.update(table_rows=table_rows, table_headers=table_headers)
    return payload
