import pytest

from service.core.table_context import (
    expand_table_chunk_context,
    expand_table_contexts,
)


pytestmark = pytest.mark.unit


def _rows(count, *, contexts=None, width=3):
    headers = ["指标", *[f"202{index}E" for index in range(1, width)]]
    result = []
    for index in range(1, count + 1):
        result.append({
            "row_index": index,
            "context": contexts[index - 1] if contexts else "",
            "preamble": "",
            "headers": headers,
            "cells": [f"指标{index}", *[f"{index}{column}" for column in range(1, width)]],
        })
    return result


def _parent(rows, *, binding_status="bound"):
    return {
        "table_id": "table-1",
        "markdown": "### 原始整表",
        "cells_json": {
            "caption": "财务预测",
            "binding_status": binding_status,
            "data_rows": rows if binding_status == "bound" else [],
        },
    }


def _chunk(row_index=1):
    return {
        "chunk_id": f"row-{row_index}",
        "evidence_type_kwd": "table",
        "table_id_kwd": "table-1",
        "table_object_ref_kwd": "storage/table_objects/table-1.json",
        "table_row_start_int": row_index,
        "table_row_end_int": row_index,
        "content_with_weight": f"原子行{row_index}",
    }


def test_small_table_expands_to_the_complete_table(monkeypatch):
    monkeypatch.setenv("RAG_TABLE_CONTEXT_EXPANSION_ENABLED", "true")
    expanded = expand_table_chunk_context(
        _chunk(2),
        "指标2是多少",
        table_object=_parent(_rows(3)),
    )

    assert expanded["table_context_obj"]["policy"] == "small_table"
    assert expanded["table_context_obj"]["row_indices"] == [1, 2, 3]
    assert all(f"指标{index}" in expanded["content_with_weight"] for index in range(1, 4))


def test_regular_table_expands_two_neighbor_rows_on_each_side(monkeypatch):
    monkeypatch.setenv("RAG_TABLE_CONTEXT_EXPANSION_ENABLED", "true")
    expanded = expand_table_chunk_context(
        _chunk(6),
        "指标6是多少",
        table_object=_parent(_rows(10)),
    )

    assert expanded["table_context_obj"]["policy"] == "neighbor_window"
    assert expanded["table_context_obj"]["row_indices"] == [4, 5, 6, 7, 8]
    assert "指标3" not in expanded["content_with_weight"]
    assert "指标8" in expanded["content_with_weight"]


def test_section_table_prefers_all_rows_in_the_same_section(monkeypatch):
    monkeypatch.setenv("RAG_TABLE_CONTEXT_EXPANSION_ENABLED", "true")
    contexts = ["利润表"] * 4 + ["现金流量表"] * 4
    expanded = expand_table_chunk_context(
        _chunk(6),
        "经营现金流",
        table_object=_parent(_rows(8, contexts=contexts)),
    )

    assert expanded["table_context_obj"]["policy"] == "same_section"
    assert expanded["table_context_obj"]["row_indices"] == [5, 6, 7, 8]
    assert "#### 现金流量表" in expanded["content_with_weight"]
    assert "#### 利润表" not in expanded["content_with_weight"]


def test_section_expansion_is_bounded_by_token_budget(monkeypatch):
    monkeypatch.setenv("RAG_TABLE_CONTEXT_EXPANSION_ENABLED", "true")
    monkeypatch.setenv("RAG_TABLE_CONTEXT_MAX_TOKENS", "80")
    rows = _rows(10, contexts=["利润表"] * 10)
    for row in rows:
        row["cells"][0] += "很长的指标说明" * 12

    expanded = expand_table_chunk_context(
        _chunk(6),
        "利润指标",
        table_object=_parent(rows),
    )

    context = expanded["table_context_obj"]
    assert context["policy"] == "same_section"
    assert context["truncated"] is True
    assert context["matched_row_index"] == 6
    assert 6 in context["row_indices"]
    assert len(context["row_indices"]) < 10


def test_wide_table_uses_query_centered_columns_and_repeats_row_label(monkeypatch):
    monkeypatch.setenv("RAG_TABLE_CONTEXT_EXPANSION_ENABLED", "true")
    monkeypatch.setenv("RAG_TABLE_CONTEXT_MAX_COLUMNS", "4")
    headers = ["指标", "2023A", "2024A", "2025E", "2026E", "2027E", "2028E", "2029E"]
    rows = [{
        "row_index": 1,
        "context": "",
        "preamble": "",
        "headers": headers,
        "cells": ["营业收入", "1", "2", "3", "4", "5", "6", "7"],
    }]
    expanded = expand_table_chunk_context(
        _chunk(1),
        "2027E营业收入是多少",
        table_object=_parent(rows),
    )

    assert expanded["table_context_obj"]["column_indices"] == [0, 4, 5, 6]
    assert expanded["table_context_obj"]["column_window"] is True
    assert "| 指标 | 2026E | 2027E | 2028E |" in expanded["content_with_weight"]
    assert "| 营业收入 | 4 | 5 | 6 |" in expanded["content_with_weight"]


def test_unbound_table_returns_full_parent_markdown_as_unreliable(monkeypatch):
    monkeypatch.setenv("RAG_TABLE_CONTEXT_EXPANSION_ENABLED", "true")
    expanded = expand_table_chunk_context(
        _chunk(1),
        "收入是多少",
        table_object=_parent([], binding_status="unbound"),
    )

    assert expanded["content_with_weight"] == "### 原始整表"
    assert expanded["table_context_obj"] == {
        "policy": "unbound_full_table",
        "reliable": False,
        "row_indices": [],
        "column_indices": [],
        "truncated": False,
    }


def test_parent_object_is_loaded_once_for_multiple_hits(monkeypatch):
    monkeypatch.setenv("RAG_TABLE_CONTEXT_EXPANSION_ENABLED", "true")
    calls = []

    expanded = expand_table_contexts(
        [_chunk(1), _chunk(2)],
        "指标",
        loader=lambda reference: calls.append(reference) or _parent(_rows(3)),
    )

    assert calls == ["storage/table_objects/table-1.json"]
    assert len(expanded) == 2
    assert all(chunk["table_context_obj"]["policy"] == "small_table" for chunk in expanded)
