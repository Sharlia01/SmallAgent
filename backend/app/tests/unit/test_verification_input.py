"""核验输入保留事实、来源及裁剪边界。"""
from dataclasses import replace
import json
import pytest
from domain.evidence import EvidenceStructure, EvidenceTable, EvidenceTableRow
from service.core.evidence.adapter import evidence_from_record
from service.core.evidence.verification_input import build_verification_record

pytestmark = pytest.mark.unit


def table_evidence(row, **metadata):
    base = evidence_from_record({'chunk_id': 'c1', 'docnm_kwd': '年度报告',
                                 'content_with_weight': '原始正文及脚注', 'page_num_int': [2]})
    return replace(base, structure=EvidenceStructure(kind='table', table=EvidenceTable(
        binding='bound', caption='经营情况', rows=(row,), metadata=metadata)))


def test_preserves_projected_columns_notes_and_identity_without_internal_mirrors():
    row = EvidenceTableRow(('2026',), ('120',), 3, (2,),
                           ('公司', '单位', '2026'), ('甲公司', '万元', '120'), '水电业务', '不含税')
    evidence = table_evidence(row, table_candidate_obj={'rows': [{'cells': ['不应送审的行']}]},
                             table_semantic_text_with_weight='口径：合并报表', footnotes='数据未经审计')
    before = repr(evidence)
    record = build_verification_record(evidence, text=evidence.text, index=7)
    assert record['index'] == 7 and record['chunk_id'] == 'c1'
    assert 'content' not in record and record['page_num_int'] == [2]
    table = record['table']
    assert table['reliable']
    assert table['rows'][0]['source_row']['cells'] == ['甲公司', '万元', '120']
    assert table['rows'][0]['context'] == '水电业务'
    assert table['rows'][0]['preamble'] == '不含税'
    assert table['source_notes'] == ['口径：合并报表', '数据未经审计']
    assert '不应送审的行' not in json.dumps(record, ensure_ascii=False)
    assert 'table_candidate_obj' not in json.dumps(record)
    assert before == repr(evidence)


def test_identical_original_row_is_not_sent_twice():
    row = EvidenceTableRow(('指标', '2026'), ('收入', '120'), 0, (0, 1),
                           ('指标', '2026'), ('收入', '120'))
    record = build_verification_record(table_evidence(row), text='正文', index=1)
    assert 'source_row' not in record['table']['rows'][0]


def test_broken_structure_preserves_unpaired_values_without_text():
    row = EvidenceTableRow(('2026',), ('120', '单位不明'), 0)
    record = build_verification_record(table_evidence(row), text='保留完整原文', index=1)
    assert not record['table']['reliable']
    assert record['table']['rows'][0]['cells'] == ['120', '单位不明']
    assert 'content' not in record


@pytest.mark.parametrize('kind', ['text', 'figure'])
def test_non_table_content_and_visual_status_are_preserved(kind):
    evidence = evidence_from_record({'chunk_id': 'c1', 'content_with_weight': '图中文字',
                                     'evidence_type_kwd': kind, 'visual_status_kwd': 'unextracted'})
    record = build_verification_record(evidence, text=evidence.text, index=1)
    assert record['content'] == '图中文字'
    assert 'table' not in record
    if kind == 'figure':
        assert record['visual_status_kwd'] == 'unextracted'


def test_unbound_table_without_rows_does_not_send_content():
    evidence = evidence_from_record({'chunk_id': 'unbound', 'content_with_weight': '未绑定表格原文',
                                     'evidence_type_kwd': 'table', 'table_binding_kwd': 'unbound'})
    record = build_verification_record(evidence, text=evidence.text, index=1)
    assert 'content' not in record
    assert record['table']['binding'] == 'unbound'
    assert not record['table']['reliable']


def shared_evidence(identifier, indices=(3, 4), *, document='doc', parent='parent-v1'):
    rows = [EvidenceTableRow(('指标', '2026'), (f'指标{i}', str(i)), i, (0, 1),
                            ('指标', '2026'), (f'指标{i}', str(i))) for i in indices]
    base = evidence_from_record({'chunk_id': identifier, 'doc_id': document, 'kb_id': 'kb',
                                 'table_id_kwd': 't1', 'content_with_weight': '表格正文'})
    return replace(base, structure=EvidenceStructure(kind='table', table=EvidenceTable(
        binding='bound', rows=tuple(rows), metadata={'table_object_ref_kwd': parent})))


def batch(*items):
    from service.core.evidence.verification_input import build_verification_evidence
    return build_verification_evidence((item, item.text) for item in items)


def test_overlapping_candidates_share_rows_and_headers_without_changing_indices():
    a, b = shared_evidence('a'), shared_evidence('b', (4, 5))
    before = (a.version, b.version)
    payload = batch(a, b)
    assert len(payload['table_rows']) == 3
    assert len(payload['table_headers']) == 1
    candidates = payload['candidate_evidence']
    assert [(item['index'], item['chunk_id']) for item in candidates] == [(1, 'a'), (2, 'b')]
    assert candidates[0]['table']['row_refs'] == ['r1', 'r2']
    assert candidates[1]['table']['row_refs'] == ['r2', 'r3']
    assert all('headers' not in row for row in payload['table_rows'].values())
    assert all('content' not in item for item in candidates)
    assert before == (a.version, b.version)


@pytest.mark.parametrize('changes', [{'document': 'another'}, {'parent': 'parent-v2'}, {'parent': ''}])
def test_different_or_missing_table_identity_does_not_share_rows(changes):
    payload = batch(shared_evidence('a'), shared_evidence('b', **changes))
    assert len(payload['table_rows']) == 4
    assert len(payload['table_headers']) == 2


@pytest.mark.parametrize('changes', [
    {'cells': ('指标3', '999')}, {'context': '另一个分组'}, {'preamble': '含税'},
    {'headers': ('指标', '2027')}, {'columns': (1, 0)},
])
def test_row_differences_are_not_removed(changes):
    a = shared_evidence('a', (3,))
    row = replace(a.structure.table.rows[0], **changes)
    b = replace(a, id='b', structure=replace(a.structure, table=replace(a.structure.table, rows=(row,))))
    assert len(batch(a, b)['table_rows']) == 2


def test_different_valid_column_windows_and_original_headers_are_preserved():
    original = ('指标', '2025', '2026')
    values = ('收入', '100', '120')
    a = shared_evidence('a', (3,))
    def window(identifier, columns):
        row = EvidenceTableRow(tuple(original[i] for i in columns), tuple(values[i] for i in columns),
                               3, columns, original, values)
        return replace(a, id=identifier, structure=replace(a.structure, table=replace(a.structure.table, rows=(row,))))
    payload = batch(window('a', (0, 1)), window('b', (0, 2)))
    assert len(payload['table_rows']) == 2
    assert len(payload['table_headers']) == 3  # 两个列窗口，共享一份原始表头。
    for row in payload['table_rows'].values():
        assert row['source_row']['cells'] == list(values)
        assert payload['table_headers'][row['source_row']['header_ref']]['labels'] == list(original)


def test_selected_batch_has_no_dangling_or_unselected_rows():
    payload = batch(shared_evidence('b', (4, 5)))
    refs = payload['candidate_evidence'][0]['table']['row_refs']
    assert set(refs) == set(payload['table_rows'])
    assert {row['row_index'] for row in payload['table_rows'].values()} == {4, 5}
    assert all(row['header_ref'] in payload['table_headers'] for row in payload['table_rows'].values())


def test_budget_and_sent_request_use_the_same_shared_rows(monkeypatch):
    from domain.evidence import EvidenceCollection
    from service.core.evidence import sufficiency as s
    a, b = shared_evidence('a'), shared_evidence('b')
    sources = EvidenceCollection([a, b])
    items = tuple(s.VerificationItem(item, item.text) for item in sources)
    budget = s._request_token_estimate('收入是多少？', items, ())
    monkeypatch.setenv('RAG_EVIDENCE_SUFFICIENCY_CONTEXT_TOKENS', str(budget))
    selected = s._prepare_evidence(items, sources, question='收入是多少？')
    assert selected == items
    request = s._build_model_request('收入是多少？', selected, ())
    payload = json.loads(request['messages'][1]['content'])
    assert len(payload['candidate_evidence']) == 2
    assert len(payload['table_rows']) == 2
    assert len(payload['table_headers']) == 1
    assert payload['candidate_evidence'][0]['table']['row_refs'] == payload['candidate_evidence'][1]['table']['row_refs']
    monkeypatch.setenv('RAG_EVIDENCE_SUFFICIENCY_CONTEXT_TOKENS', str(budget - 1))
    selected = s._prepare_evidence(items, sources, question='收入是多少？')
    assert len(selected) == 1
    payload = json.loads(s._build_model_request('收入是多少？', selected, ())['messages'][1]['content'])
    references = {ref for item in payload['candidate_evidence'] for ref in item['table']['row_refs']}
    assert references == set(payload['table_rows'])
