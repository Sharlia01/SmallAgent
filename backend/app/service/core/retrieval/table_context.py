"""将检索命中的表格行扩展为长度受限、供模型使用的表格上下文。"""

from __future__ import annotations

import logging
import os
import re
import unicodedata
from typing import Any, Callable

from service.core.evidence.metadata import primary_table_row, without_entity_annotations
from service.core.rag.utils import num_tokens_from_string, truncate
from domain.tables import build_table_markdown
from domain.table_retrieval import IndexedTableRow, TableCandidate, TableContext
from service.core.evidence.table_retrieval_adapter import context_record


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
    # 读取父表只依赖存储层，不加载文档解析器和嵌入模型。
    from service.core.persistence.evidence_store import load_table_object

    return load_table_object(reference)


def _normalized(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return "".join(character for character in text if character.isalnum())


def _header_matches_question(header: str, question: str) -> bool:
    # 表头中的单位描述数值，不属于查询的期间或指标定位词。
    header = re.sub(r"[（(][^（）()]*[）)]", "", header)
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
        _normalized(value) for value in primary_table_row(chunk).get("cells") or []
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

    # 第零列通常是行标签，应在宽表的每个列窗口中保留；
    # 其余列以匹配查询的表头为中心选择。
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
            "column_indices": available,
            # 旧父表可能含有规则归属标签，扩展上下文只继承原始行数据。
            "source_row": without_entity_annotations(row),
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
    matched_positions: list[int] | None = None,
) -> tuple[str, list[int], bool]:
    selected = list(row_positions)
    priorities = {position: rank for rank, position in enumerate(matched_positions or [matched_position])}
    truncated_rows = False
    content = _render_context(cells_json, data_rows, selected, column_indices)
    while len(selected) > 1 and num_tokens_from_string(content) > max_tokens:
        removable = max(
            (position for position in selected if position != matched_position),
            # 先移除补充邻行，再按召回顺序移除次要命中；最佳命中始终保留。
            key=lambda position: (position not in priorities, priorities.get(position, 0),
                                  abs(position - matched_position), position),
        )
        selected.remove(removable)
        truncated_rows = True
        content = _render_context(cells_json, data_rows, selected, column_indices)

    if num_tokens_from_string(content) > max_tokens:
        content = truncate(content, max_tokens).rstrip()
        content += "\n\n> 表格上下文已按 token 限制截断。"
        truncated_rows = True
    return content, sorted(selected), truncated_rows


def build_table_context(
    candidate: TableCandidate | IndexedTableRow,
    question: str,
    *,
    table_object: dict[str, Any] | None = None,
    loader: Callable[[str], dict[str, Any]] | None = None,
) -> TableContext | None:
    """扩展只创建独立上下文，不改写来源快照或候选命中集合。"""
    chunk = candidate.source
    if chunk.get("evidence_type_kwd") != "table" or not _enabled():
        return None

    reference = str(chunk.get("table_object_ref_kwd") or "")
    if table_object is None:
        if not reference:
            return None
        table_object = (loader or _default_loader)(reference)

    if str(table_object.get("table_id") or "") != str(
        chunk.get("table_id_kwd") or ""
    ):
        raise ValueError("Retrieved table chunk references a different parent")

    cells_json = table_object.get("cells_json") or {}
    binding_status = str(cells_json.get("binding_status") or "unbound")
    if binding_status != "bound":
        content = str(
            table_object.get("markdown")
            or chunk.get("content_with_weight")
            or ""
        )
        diagnostics = {
            "policy": "unbound_full_table",
            "reliable": False,
            "row_indices": [],
            "column_indices": [],
            "truncated": False,
        }
        return TableContext(candidate, content, (), diagnostics)

    data_rows = cells_json.get("data_rows") or []
    if not data_rows:
        return None

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
    if isinstance(candidate, TableCandidate):
        primary = candidate.matched_row_indices[0]
        # 父表版本不一致时不能把未找到的主行悄悄替换为第一行。
        if not all(index in {int(row['row_index']) for row in data_rows}
                   for index in candidate.matched_row_indices):
            raise ValueError("Candidate rows are missing from parent table")
        matched_position = next(i for i, row in enumerate(data_rows) if int(row['row_index']) == primary)
    else:
        matched_position = _matched_row_position(data_rows, chunk)
    row_positions, policy = _row_positions(
        data_rows,
        matched_position,
        small_table_rows=small_table_rows,
        neighbor_radius=neighbor_radius,
    )
    legacy = chunk.get("table_candidate_obj") or {}
    requested_rows = (candidate.matched_row_indices if isinstance(candidate, TableCandidate)
                      else legacy.get("matched_row_indices") or [])
    ranking_source = candidate.ranking_source if isinstance(candidate, TableCandidate) else legacy.get("row_ranking_source")
    scored_indices = candidate.scored_row_indices if isinstance(candidate, TableCandidate) else legacy.get("scored_row_indices", ())
    by_row = {int(row["row_index"]): position for position, row in enumerate(data_rows)}
    matched_positions = list(dict.fromkeys(
        by_row[index] for index in requested_rows if index in by_row
    )) or [matched_position]
    if requested_rows:
        # 多个主行只展开一次；对邻行/分组取并集，不重复输出同一数据行。
        row_positions = sorted({position for match in matched_positions for position in _row_positions(
            data_rows, match, small_table_rows=small_table_rows, neighbor_radius=neighbor_radius,
        )[0]})
        policy = "grouped_matches"
    if ranking_source == "model":
        # 已评分但未入选的行不能通过邻域扩展重新占用上下文预算。
        excluded = set(scored_indices) - set(requested_rows)
        row_positions = [position for position in row_positions
                         if int(data_rows[position]["row_index"]) not in excluded]
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
        matched_positions,
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
    # 保留公司列和表头原文，由核验模型判断归属，不给邻近行附加推断标签。
    diagnostics = {
        "policy": policy,
        "reliable": True,
        "matched_row_index": int(
            data_rows[matched_position].get("row_index") or matched_position + 1
        ),
        "row_indices": selected_row_indices,
        "matched_row_indices": [int(data_rows[position]["row_index"]) for position in matched_positions],
        "omitted_matched_row_indices": [int(data_rows[position]["row_index"])
                                        for position in matched_positions if position not in selected_positions],
        "column_indices": column_indices,
        "caption": str(cells_json.get("caption") or ""),
        "binding_source": cells_json.get("binding_source") or "deterministic",
        "column_window": column_window,
        "truncated": truncated_rows or column_window,
        "token_count": num_tokens_from_string(content),
    }
    # 保存容量裁剪掉的命中行原始快照；不含低分淘汰行或辅助邻行。
    omitted = [position for position in matched_positions if position not in selected_positions]
    recovery_rows = tuple(without_entity_annotations(data_rows[position]) for position in omitted)
    return TableContext(candidate, content, tuple(context_rows), diagnostics, recovery_rows)


def expand_table_chunk_context(
    chunk: dict[str, Any], question: str, *, table_object=None, loader=None,
) -> dict[str, Any]:
    """旧字典调用入口：仅在边界适配，核心扩展返回不可变上下文。"""
    snapshot = IndexedTableRow(str(chunk.get("chunk_id") or ""), chunk)
    context = build_table_context(snapshot, question, table_object=table_object, loader=loader)
    return context_record(context) if context is not None else chunk


def expand_table_contexts(
    chunks: list[dict[str, Any]],
    question: str,
    *,
    loader: Callable[[str], dict[str, Any]] | None = None,
    candidates: dict[str, TableCandidate] | None = None,
) -> list[dict[str, Any]]:
    """扩展表格片段，每个父表对象最多加载一次。"""
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
            candidate = (candidates or {}).get(chunk.get("chunk_id"))
            if candidate is None:
                candidate = IndexedTableRow(str(chunk.get("chunk_id") or ""), chunk)
            context = build_table_context(candidate, question, table_object=cache[reference])
            expanded_chunks.append(context_record(context, chunk) if context is not None else chunk)
        except Exception as error:
            logger.warning(
                "Table context expansion failed for %s: %s",
                chunk.get("table_id_kwd") or "unknown",
                type(error).__name__,
            )
            expanded_chunks.append(chunk)
    return expanded_chunks
