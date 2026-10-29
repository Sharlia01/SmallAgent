"""身份冲突只比较来源事实，完整快照版本仍保留检索元数据差异。"""
from dataclasses import replace
import pytest
from domain.assessment import EvidenceAssessment
from domain.evidence import EvidenceCollection, RetrievalTrace
from service.core.evidence.adapter import evidence_from_record
from service.core.evidence import verification
from tests.evidence_helpers import make_query

pytestmark = pytest.mark.unit


def source(**changes):
    row = {'row_index': 3, 'headers': ['指标', '2026'], 'cells': ['收入', '120'],
           'context': '单位：万元', 'preamble': '合并口径'}
    record = {
        'chunk_id': 'same-table-group', 'kb_id': 'kb', 'doc_id': 'doc', 'docnm_kwd': '年报',
        'content_with_weight': '收入120万元', 'evidence_type_kwd': 'table',
        'page_num_int': [2], 'table_id_kwd': 't1', 'table_object_ref_kwd': 'parent-v1',
        'table_binding_kwd': 'bound',
        'table_context_obj': {'caption': '经营情况', 'reliable': True, 'rows': [row],
                              'matched_row_indices': [3, 4]},
        'table_candidate_obj': {'scored_row_indices': [3, 4], 'member_chunk_ids': ['a', 'b']},
    }
    record.update(changes)
    return evidence_from_record(record)


def test_ranking_changes_merge_traces_and_preserve_first_snapshot(monkeypatch):
    first = source()
    metadata = dict(first.structure.table.metadata)
    metadata['table_candidate_obj'] = {'scored_row_indices': [4, 3], 'member_chunk_ids': ['b', 'a']}
    metadata['table_context_obj'] = {**metadata['table_context_obj'], 'matched_row_indices': [4, 3]}
    second = replace(first, structure=replace(first.structure, table=replace(first.structure.table, metadata=metadata)))
    assert first.version != second.version
    assert first.source_content_fingerprint == second.source_content_fingerprint
    evidence = EvidenceCollection()
    evidence.add(first, RetrievalTrace('第一次查询'))
    before = evidence.version
    evidence.add(second, RetrievalTrace('补查'))
    assert not evidence.conflicts
    assert evidence.get(first.id) is first
    assert len(evidence.traces(first.id)) == 2
    assert evidence.version == before
    calls = []
    def assess(question, sources, **kwargs):
        calls.append(sources)
        return EvidenceAssessment(False, '缺少年度数据')
    monkeypatch.setattr(verification, 'assess_evidence_sufficiency', assess)
    verification.verify_evidence(make_query('收入是多少？'), evidence)
    assert len(calls) == 1


@pytest.mark.parametrize('changes', [
    {'content_with_weight': '收入130万元'}, {'doc_id': 'another'}, {'kb_id': 'another'},
    {'page_num_int': [3]}, {'table_id_kwd': 't2'}, {'table_object_ref_kwd': 'parent-v2'},
    {'table_semantic_text_with_weight': '单位：亿元'}, {'table_binding_kwd': 'unbound'},
])
def test_actual_source_changes_still_conflict(changes):
    first, second = source(), source(**changes)
    evidence = EvidenceCollection([first, second])
    assert len(evidence.conflicts) == 1
    assert evidence.get(first.id) is first
    assert evidence.conflicts[0].incoming_version == second.version


@pytest.mark.parametrize('changes', [
    {'cells': ('收入', '130')}, {'headers': ('指标', '2027')},
    {'columns': (1, 0)}, {'original_cells': ('收入', '999')},
    {'context': '单位：亿元'}, {'preamble': '不含税'}, {'index': 4},
])
def test_structured_row_changes_still_conflict(changes):
    first = source()
    row = replace(first.structure.table.rows[0], **changes)
    second = replace(first, structure=replace(first.structure, table=replace(first.structure.table, rows=(row,))))
    assert EvidenceCollection([first, second]).conflicts


@pytest.mark.parametrize('changes', [{'footnotes': '未经审计'}, {'table_semantic_text_with_weight': '特殊口径说明'}])
def test_notes_only_in_metadata_are_not_ignored(changes):
    first = source()
    second = replace(first, structure=replace(first.structure, table=replace(
        first.structure.table, metadata={**first.structure.table.metadata, **changes})))
    assert EvidenceCollection([first, second]).conflicts


def test_reconstructed_semantic_rows_do_not_change_source_identity():
    first = source()
    second = source(table_semantic_text_with_weight='经营情况\n单位：万元\n合并口径\n指标：收入；2026：120')
    assert first.version != second.version
    assert not EvidenceCollection([first, second]).conflicts
