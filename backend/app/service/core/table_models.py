"""Domain models for auditable tables and their searchable child chunks.

The complete table is the source of truth.  Searchable chunks deliberately keep
only a row (or a small row range) so large tables do not dilute retrieval.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any


TABLE_OBJECT_SCHEMA_VERSION = "1.0"
TABLE_PARSER_VERSION = "table-object-v5"


@dataclass(frozen=True)
class TableObject:
    """A complete, reconstructable table produced by a parser."""

    table_id: str
    parser_html: str
    cells_json: dict[str, Any]
    markdown: str
    semantic_summary: str | None
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = TABLE_OBJECT_SCHEMA_VERSION
        return payload


@dataclass(frozen=True)
class TableChunk:
    """A searchable child of a :class:`TableObject`."""

    table_id: str
    row_start: int
    row_end: int
    semantic_text: str
    markdown_excerpt: str
    cells_slice: dict[str, Any]
    chunk_kind: str = "row"

    def to_record(self) -> dict[str, Any]:
        """Map the domain names to fields that follow the ES suffix mapping."""
        return {
            "table_id_kwd": self.table_id,
            "table_chunk_kind_kwd": self.chunk_kind,
            "table_row_start_int": self.row_start,
            "table_row_end_int": self.row_end,
            "table_semantic_text_with_weight": self.semantic_text,
            "table_markdown_excerpt_with_weight": self.markdown_excerpt,
            "table_cells_slice_obj": self.cells_slice,
            "table_schema_version_kwd": TABLE_OBJECT_SCHEMA_VERSION,
        }


def make_table_id(
    document_name: str,
    table_ordinal: int,
    positions: list[Any],
    parser_html: str,
    cells_json: dict[str, Any] | None = None,
) -> str:
    """Return a content-addressed ID stable for an unchanged parsed table."""
    identity = {
        "schema_version": TABLE_OBJECT_SCHEMA_VERSION,
        "document_name": document_name,
        "table_ordinal": table_ordinal,
        "positions": positions,
        "parser_html": parser_html,
        "cells_json": cells_json,
    }
    canonical = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _markdown_cell(value: Any) -> str:
    text = "（空白）" if value in (None, "") else str(value)
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


def _row_markdown(caption: str, context: str, headers: list[str], cells: list[str]) -> str:
    parts = []
    title = "；".join(part for part in (caption, context) if part)
    if title:
        parts.append(f"### {title}")
    parts.extend([
        "| " + " | ".join(_markdown_cell(value) for value in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        "| " + " | ".join(_markdown_cell(value) for value in cells) + " |",
    ])
    return "\n".join(parts)


def build_table_markdown(cells_json: dict[str, Any]) -> str:
    """Deterministically render Markdown from normalized cells, never from HTML."""
    caption = str(cells_json.get("caption") or "")
    data_rows = cells_json.get("data_rows") or []
    parts = [f"### {caption}"] if caption else []

    if data_rows:
        current_headers: tuple[str, ...] | None = None
        current_context: str | None = None
        for row in data_rows:
            headers = tuple(str(value) for value in row.get("headers") or [])
            cells = [str(value) for value in row.get("cells") or []]
            if not headers or len(headers) != len(cells):
                continue
            context = str(row.get("context") or "")
            if headers != current_headers or context != current_context:
                if current_headers is not None:
                    parts.append("")
                if context:
                    parts.append(f"#### {context}")
                parts.extend([
                    "| " + " | ".join(_markdown_cell(value) for value in headers) + " |",
                    "| " + " | ".join("---" for _ in headers) + " |",
                ])
                current_headers = headers
                current_context = context
            parts.append("| " + " | ".join(_markdown_cell(value) for value in cells) + " |")
        return "\n".join(parts)

    # An unbound table remains readable without promoting its first row to a
    # semantic header.  Row labels are structural only.
    parts.append("> 表格结构未可靠绑定；以下内容保留原始行列顺序。")
    raw_grid = cells_json.get("raw_grid") or []
    for index, row in enumerate(raw_grid, start=1):
        parts.append(
            f"- 第{index}行：" + " | ".join(_markdown_cell(value) for value in row)
        )
    if not raw_grid and cells_json.get("raw_text"):
        parts.append("- 原始文本：" + _markdown_cell(cells_json["raw_text"]))
    return "\n".join(parts)


def build_semantic_summary(cells_json: dict[str, Any]) -> str:
    """Build a factual table-level summary without an LLM."""
    caption = str(cells_json.get("caption") or "未命名表格")
    data_rows = cells_json.get("data_rows") or []
    headers: list[str] = []
    for row in data_rows:
        for header in row.get("headers") or []:
            header = str(header)
            if header and header not in headers:
                headers.append(header)
    raw_grid = cells_json.get("raw_grid") or []
    row_count = len(data_rows) if data_rows else len(raw_grid)
    width = max((len(row) for row in raw_grid), default=0)
    parts = [f"表格：{caption}", f"共{row_count}个数据行"]
    if width:
        parts.append(f"原始网格{width}列")
    if headers:
        parts.append("字段：" + "、".join(headers[:40]))
    if cells_json.get("binding_status") != "bound":
        parts.append("表头与数据尚未可靠绑定")
    return "；".join(parts)


def build_table_chunks(table: TableObject) -> list[TableChunk]:
    """Create self-contained row chunks from one complete table object."""
    cells_json = table.cells_json
    caption = str(cells_json.get("caption") or "")
    chunks: list[TableChunk] = []
    for sequence, row in enumerate(cells_json.get("data_rows") or [], start=1):
        headers = [str(value) for value in row.get("headers") or []]
        cells = [str(value) for value in row.get("cells") or []]
        if not headers or len(headers) != len(cells):
            continue
        context = str(row.get("context") or "")
        preamble = str(row.get("preamble") or "")
        title = "；".join(part for part in (caption, context) if part)
        pairs = "；".join(
            f"{header}：{cell or '（空白）'}"
            for header, cell in zip(headers, cells)
        )
        semantic_text = "\n".join(
            part for part in (title, preamble, pairs) if part
        )
        row_index = int(row.get("row_index") or sequence)
        cells_slice = {
            "caption": caption,
            "binding_status": cells_json.get("binding_status"),
            "context": context,
            "rows": [{
                "row_index": row_index,
                "headers": headers,
                "cells": cells,
            }],
        }
        chunks.append(TableChunk(
            table_id=table.table_id,
            row_start=row_index,
            row_end=row_index,
            semantic_text=semantic_text,
            markdown_excerpt=_row_markdown(caption, context, headers, cells),
            cells_slice=cells_slice,
        ))

    if chunks:
        return chunks

    raw_grid = cells_json.get("raw_grid") or []
    semantic_rows = [
        "；".join(str(value) for value in row if value not in (None, ""))
        for row in raw_grid
    ]
    if not semantic_rows and cells_json.get("raw_text"):
        semantic_rows.append(str(cells_json["raw_text"]))
    semantic_text = "\n".join(
        part for part in (
            caption,
            "表格结构未可靠绑定，不得猜测表头与数值的对应关系。",
            *semantic_rows,
        ) if part
    )
    return [TableChunk(
        table_id=table.table_id,
        row_start=1,
        row_end=max(1, len(raw_grid)),
        semantic_text=semantic_text,
        markdown_excerpt=table.markdown,
        cells_slice={
            "caption": caption,
            "binding_status": cells_json.get("binding_status"),
            "raw_grid": raw_grid,
        },
        chunk_kind="unbound_table",
    )]
