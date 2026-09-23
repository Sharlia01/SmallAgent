"""Expand retrieved table rows into bounded LLM-facing table context."""

from __future__ import annotations

import logging
import os
import re
import unicodedata
from typing import Any, Callable

from service.core.rag.utils import num_tokens_from_string, truncate
from service.core.table_models import build_table_markdown


logger = logging.getLogger(__name__)

DEFAULT_SMALL_TABLE_ROWS = 5
DEFAULT_NEIGHBOR_RADIUS = 2
DEFAULT_MAX_CONTEXT_TOKENS = 512
DEFAULT_MAX_CONTEXT_COLUMNS = 8


def _positive_int_environment(name: str, default: int, *, minimum: int = 1) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return value if value >= minimum else default


def _enabled() -> bool:
    return os.getenv(
        "RAG_TABLE_CONTEXT_EXPANSION_ENABLED",
        "true",
    ).strip().casefold() not in {"0", "false", "no", "off"}


def _default_loader(reference: str) -> dict[str, Any]:
    # Imported lazily so ordinary retrieval does not load the ingestion stack
    # unless a new table chunk actually needs its parent object.
    from service.core.file_parse import load_table_object

    return load_table_object(reference)


def _normalized(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return "".join(character for character in text if character.isalnum())


def _header_matches_question(header: str, question: str) -> bool:
    normalized_header = _normalized(header)
    normalized_question = _normalized(question)
    if normalized_header and normalized_header in normalized_question:
        return True

    for part in re.split(r"[/｜|\s]+", unicodedata.normalize("NFKC", header)):
        normalized_part = _normalized(part)
        if len(normalized_part) >= 2 and normalized_part in normalized_question:
            return True
    return False


def _matched_row_position(
    data_rows: list[dict[str, Any]],
    chunk: dict[str, Any],
) -> int:
    row_start = chunk.get("table_row_start_int")
    try:
        row_start = int(row_start)
    except (TypeError, ValueError):
        row_start = None
    if row_start is not None:
        for position, row in enumerate(data_rows):
            if int(row.get("row_index") or -1) == row_start:
                return position

    expected_cells = [
        _normalized(value) for value in chunk.get("table_cells_kwd") or []
    ]
    if expected_cells:
        for position, row in enumerate(data_rows):
            actual_cells = [
                _normalized(value) for value in row.get("cells") or []
            ]
            if actual_cells == expected_cells:
                return position
    return 0


def _row_positions(
    data_rows: list[dict[str, Any]],
    matched_position: int,
    *,
    small_table_rows: int,
    neighbor_radius: int,
) -> tuple[list[int], str]:
    if len(data_rows) <= small_table_rows:
        return list(range(len(data_rows))), "small_table"

    matched_context = str(data_rows[matched_position].get("context") or "")
    if matched_context:
        same_section = [
            index
            for index, row in enumerate(data_rows)
            if str(row.get("context") or "") == matched_context
        ]
        if same_section:
            return same_section, "same_section"

    start = max(0, matched_position - neighbor_radius)
    end = min(len(data_rows), matched_position + neighbor_radius + 1)
    return list(range(start, end)), "neighbor_window"


def _column_indices(
    matched_row: dict[str, Any],
    question: str,
    max_columns: int,
) -> tuple[list[int], bool]:
    headers = [str(value) for value in matched_row.get("headers") or []]
    column_count = len(headers)
    if column_count <= max_columns:
        return list(range(column_count)), False

    # Column zero is normally the row label.  Repeat it in every wide-table
    # window and center the remaining columns around a query-matching header.
    value_slots = max(1, max_columns - 1)
    matching = [
        index
        for index, header in enumerate(headers[1:], start=1)
        if _header_matches_question(header, question)
    ]
    center = matching[0] if matching else 1
    start = max(1, center - value_slots // 2)
    end = min(column_count, start + value_slots)
    start = max(1, end - value_slots)
    return [0, *range(start, end)], True


def _slice_rows(
    data_rows: list[dict[str, Any]],
    row_positions: list[int],
    column_indices: list[int],
) -> list[dict[str, Any]]:
    sliced = []
    for position in row_positions:
        row = data_rows[position]
        headers = [str(value) for value in row.get("headers") or []]
        cells = [str(value) for value in row.get("cells") or []]
        available = [
            index
            for index in column_indices
            if index < len(headers) and index < len(cells)
        ]
        if not available:
            continue
        sliced.append({
            "row_index": row.get("row_index"),
            "context": str(row.get("context") or ""),
            "preamble": str(row.get("preamble") or ""),
            "headers": [headers[index] for index in available],
            "cells": [cells[index] for index in available],
        })
    return sliced


def _render_context(
    cells_json: dict[str, Any],
    data_rows: list[dict[str, Any]],
    row_positions: list[int],
    column_indices: list[int],
) -> str:
    return build_table_markdown({
        "caption": cells_json.get("caption") or "",
        "binding_status": cells_json.get("binding_status"),
        "data_rows": _slice_rows(data_rows, row_positions, column_indices),
        "raw_grid": [],
    })


def _fit_token_budget(
    cells_json: dict[str, Any],
    data_rows: list[dict[str, Any]],
    row_positions: list[int],
    matched_position: int,
    column_indices: list[int],
    max_tokens: int,
) -> tuple[str, list[int], bool]:
    selected = list(row_positions)
    truncated_rows = False
    content = _render_context(cells_json, data_rows, selected, column_indices)
    while len(selected) > 1 and num_tokens_from_string(content) > max_tokens:
        removable = max(
            (position for position in selected if position != matched_position),
            key=lambda position: (abs(position - matched_position), position),
        )
        selected.remove(removable)
        truncated_rows = True
        content = _render_context(cells_json, data_rows, selected, column_indices)

    if num_tokens_from_string(content) > max_tokens:
        content = truncate(content, max_tokens).rstrip()
        content += "\n\n> 表格上下文已按 token 限制截断。"
        truncated_rows = True
    return content, sorted(selected), truncated_rows


def expand_table_chunk_context(
    chunk: dict[str, Any],
    question: str,
    *,
    table_object: dict[str, Any] | None = None,
    loader: Callable[[str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Return a copy with bounded parent-table context for the LLM."""
    if chunk.get("evidence_type_kwd") != "table" or not _enabled():
        return chunk

    reference = str(chunk.get("table_object_ref_kwd") or "")
    if table_object is None:
        if not reference:
            return chunk
        table_object = (loader or _default_loader)(reference)

    if str(table_object.get("table_id") or "") != str(
        chunk.get("table_id_kwd") or ""
    ):
        raise ValueError("Retrieved table chunk references a different parent")

    expanded = dict(chunk)
    cells_json = table_object.get("cells_json") or {}
    binding_status = str(cells_json.get("binding_status") or "unbound")
    if binding_status != "bound":
        expanded["content_with_weight"] = str(
            table_object.get("markdown")
            or chunk.get("content_with_weight")
            or ""
        )
        expanded["table_context_obj"] = {
            "policy": "unbound_full_table",
            "reliable": False,
            "row_indices": [],
            "column_indices": [],
            "truncated": False,
        }
        return expanded

    data_rows = cells_json.get("data_rows") or []
    if not data_rows:
        return chunk

    small_table_rows = _positive_int_environment(
        "RAG_TABLE_SMALL_TABLE_ROWS",
        DEFAULT_SMALL_TABLE_ROWS,
    )
    neighbor_radius = _positive_int_environment(
        "RAG_TABLE_NEIGHBOR_RADIUS",
        DEFAULT_NEIGHBOR_RADIUS,
        minimum=0,
    )
    max_tokens = _positive_int_environment(
        "RAG_TABLE_CONTEXT_MAX_TOKENS",
        DEFAULT_MAX_CONTEXT_TOKENS,
    )
    max_columns = _positive_int_environment(
        "RAG_TABLE_CONTEXT_MAX_COLUMNS",
        DEFAULT_MAX_CONTEXT_COLUMNS,
        minimum=2,
    )
    matched_position = _matched_row_position(data_rows, chunk)
    row_positions, policy = _row_positions(
        data_rows,
        matched_position,
        small_table_rows=small_table_rows,
        neighbor_radius=neighbor_radius,
    )
    column_indices, column_window = _column_indices(
        data_rows[matched_position],
        question,
        max_columns,
    )
    content, selected_positions, truncated_rows = _fit_token_budget(
        cells_json,
        data_rows,
        row_positions,
        matched_position,
        column_indices,
        max_tokens,
    )
    selected_row_indices = [
        int(data_rows[position].get("row_index") or position + 1)
        for position in selected_positions
    ]
    context_rows = _slice_rows(
        data_rows,
        selected_positions,
        column_indices,
    )
    expanded["content_with_weight"] = content
    expanded["table_context_obj"] = {
        "policy": policy,
        "reliable": True,
        "matched_row_index": int(
            data_rows[matched_position].get("row_index") or matched_position + 1
        ),
        "row_indices": selected_row_indices,
        "column_indices": column_indices,
        "rows": context_rows,
        "column_window": column_window,
        "truncated": truncated_rows or column_window,
        "token_count": num_tokens_from_string(content),
    }
    return expanded


def expand_table_contexts(
    chunks: list[dict[str, Any]],
    question: str,
    *,
    loader: Callable[[str], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Expand table chunks while loading each parent object at most once."""
    if not _enabled():
        return chunks

    resolved_loader = loader or _default_loader
    cache: dict[str, dict[str, Any]] = {}
    expanded_chunks = []
    for chunk in chunks:
        # reference表示指向父表对象的引用字符串
        reference = str(chunk.get("table_object_ref_kwd") or "")
        if chunk.get("evidence_type_kwd") != "table" or not reference:
            expanded_chunks.append(chunk)
            continue
        try:
            #cache表示父表对象临时缓存
            if reference not in cache:
                cache[reference] = resolved_loader(reference)
            expanded_chunks.append(expand_table_chunk_context(
                chunk,
                question,
                table_object=cache[reference],
            ))
        except Exception as error:
            logger.warning(
                "Table context expansion failed for %s: %s",
                chunk.get("table_id_kwd") or "unknown",
                type(error).__name__,
            )
            expanded_chunks.append(chunk)
    return expanded_chunks
