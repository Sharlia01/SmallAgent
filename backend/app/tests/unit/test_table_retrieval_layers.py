"""检索分层契约：原始行不变、候选不扩展、上下文不冒充命中。"""
from copy import deepcopy

import pytest

from domain.evidence import plain
from service.core.retrieval.table_candidates import group_table_candidates
from service.core.retrieval.table_context import build_table_context, expand_table_contexts
from service.core.evidence.table_retrieval_adapter import candidate_record, context_record
from tests.unit.test_table_candidates import row

pytestmark = pytest.mark.unit


def make_candidate():
    records = [row(2), row(9)]
    fields = {record['chunk_id']: record for record in records}
    grouped = group_table_candidates(list(fields), fields)
    return fields, grouped.tables['t1-2']


def parent():
    return {'table_id': 't1', 'cells_json': {'binding_status': 'bound', 'data_rows': [
        {'row_index': i, 'headers': ['指标', '2025E'], 'cells': [f'指标{i}', str(i * 10)]}
        for i in range(1, 12)
    ]}}


def test_snapshots_and_candidates_are_independent_of_expansion(monkeypatch):
    monkeypatch.setenv('RAG_TABLE_CONTEXT_MAX_TOKENS', '2000')
    fields, candidate = make_candidate()
    original = deepcopy(fields)
    snapshot = plain(candidate.source)
    context = build_table_context(candidate, '指标2与指标9', table_object=parent())
    assert candidate.matched_row_indices == (2, 9)
    assert [r['row_index'] for r in context.matched_rows] == [2, 9]
    assert context.auxiliary_rows
    assert all(r['row_index'] not in (2, 9) for r in context.auxiliary_rows)
    assert fields == original
    fields['t1-2']['table_cells_slice_obj']['rows'][0]['cells'][0] = '污染'
    assert plain(candidate.source) == snapshot
    with pytest.raises(TypeError):
        candidate.matched_rows[0]['cells'][0] = '污染'
    assert candidate_record(candidate)['chunk_id'] == candidate.candidate_id
    assert candidate.source_rows[0].chunk_id != candidate.candidate_id
    exported = context_record(context)
    assert exported['table_context_obj']['matched_row_indices'] == [2, 9]
    assert candidate_record(candidate)['content_with_weight'] != exported['content_with_weight']


def test_missing_parent_row_does_not_silently_select_first_row():
    _, candidate = make_candidate()
    broken = parent()
    broken['cells_json']['data_rows'] = broken['cells_json']['data_rows'][:3]
    with pytest.raises(ValueError, match='missing'):
        build_table_context(candidate, '指标9', table_object=broken)
    record = candidate_record(candidate)
    result = expand_table_contexts([record], '指标9', candidates={candidate.candidate_id: candidate},
                                   loader=lambda _: broken)
    assert result == [record]
    assert 'table_context_obj' not in result[0]


def test_context_budget_does_not_change_candidate_matches(monkeypatch):
    monkeypatch.setenv('RAG_TABLE_CONTEXT_MAX_TOKENS', '30')
    _, candidate = make_candidate()
    context = build_table_context(candidate, '指标2与指标9', table_object=parent())
    assert candidate.matched_row_indices == (2, 9)
    omitted = context.diagnostics['omitted_matched_row_indices']
    assert omitted
    assert all(row['row_index'] not in omitted for row in context.rows)
    exported = context_record(context)
    for index in omitted:
        assert f'指标{index}；' not in exported['table_semantic_text_with_weight']
