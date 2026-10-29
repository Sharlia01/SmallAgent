"""验证精简索引保留事实，并能沿生产读取路径形成候选和回答资料。"""
from copy import deepcopy

import pytest

from domain.tables import TableObject, build_table_chunks
from service.core.evidence.metadata import index_evidence_metadata, primary_table_row
from service.core.evidence.adapter import evidence_from_record
from service.core.retrieval.table_candidates import group_table_candidates
from service.core.answering.answer_context import _table_reference
from evaluation.retrieval import table_row_chunk_match_score

pytestmark = pytest.mark.unit


def test_compact_row_preserves_context_through_grouping_and_answer():
    table = TableObject('t1', '', {
        'caption': '预测表', 'binding_status': 'bound', 'data_rows': [{
            'row_index': 4, 'headers': ['指标', '2025E'], 'cells': ['预付账款', '14'],
            'context': '流动资产', 'preamble': '单位：百万元',
        }],
    }, '', None, {})
    chunk = build_table_chunks(table)[0]
    record = {
        **chunk.to_record(), 'chunk_id': 'row4', 'doc_id': 'doc', 'kb_id': '1',
        'docnm_kwd': '报告.pdf', 'evidence_type_kwd': 'table', 'table_binding_kwd': 'bound',
        'table_object_ref_kwd': 't1.json', 'content_with_weight': chunk.markdown_excerpt,
        'table_headers_kwd': ['旧字段'], 'table_cells_kwd': ['错误值'],
    }
    original = deepcopy(record)
    compact = {key: value for key, value in record.items() if not key.startswith('table_')}
    compact.update(index_evidence_metadata(record))
    assert record == original
    assert not {'table_headers_kwd', 'table_cells_kwd', 'table_markdown_excerpt_with_weight'} & compact.keys()
    assert primary_table_row(compact)['cells'] == ['预付账款', '14']
    assert table_row_chunk_match_score({'document_name': '报告.pdf', 'row_cells': ['预付账款', '14']}, compact) == 1.0
    grouped = group_table_candidates(['row4'], {'row4': compact}).fields['row4']
    view = _table_reference(evidence_from_record(grouped).structure.table)
    row = view['matched_rows'][0]
    assert row['preamble'] == '单位：百万元'
    assert row['context'] == '流动资产'
    assert [cell['value'] for cell in row['values']] == ['预付账款', '14']
    assert [cell['column_index'] for cell in row['values']] == [0, 1]


def test_retired_fields_are_ignored_but_distinct_excerpt_is_preserved():
    legacy = {'evidence_type_kwd': 'table', 'table_row_int': '4',
              'table_headers_kwd': ['指标', '2025E'], 'table_cells_kwd': ['预付账款', '14'],
              'content_with_weight': '扩展内容', 'table_markdown_excerpt_with_weight': '命中内容'}
    assert 'table_cells_kwd' not in index_evidence_metadata(legacy)
    assert 'table_headers_kwd' not in index_evidence_metadata(legacy)
    assert index_evidence_metadata(legacy)['table_markdown_excerpt_with_weight'] == '命中内容'
    assert primary_table_row(legacy) == {}


def test_structured_primary_row_takes_precedence_over_legacy_copy():
    source = {'table_row_int': '4', 'table_cells_kwd': ['错误值'],
              'table_cells_slice_obj': {'rows': [
                  {'row_index': 3, 'cells': ['应收账款', '15']},
                  {'row_index': 4, 'cells': ['预付账款', '14']},
              ]}}
    assert primary_table_row(source)['cells'] == ['预付账款', '14']
