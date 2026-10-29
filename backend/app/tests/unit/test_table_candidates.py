from copy import deepcopy

import pytest

from service.core.retrieval.table_candidates import group_table_candidates, select_candidate_ids
from service.core.retrieval.table_context import expand_table_chunk_context
from service.core.evidence.adapter import evidence_from_record, source_record
from service.core.rag.nlp.search_v2 import Dealer


pytestmark = pytest.mark.unit


def row(index, *, table="t1", kb="1"):
    headers, cells = ["指标", "2025E"], [f"指标{index}", str(index * 10)]
    return {
        "chunk_id": f"{table}-{index}", "kb_id": kb, "doc_id": "doc", "docnm_kwd": "报告.pdf",
        "evidence_type_kwd": "table", "table_binding_kwd": "bound",
        "table_id_kwd": table, "table_object_ref_kwd": f"{table}.json",
        "table_row_int": index, "table_row_start_int": index,
        "table_semantic_text_with_weight": f"指标：指标{index}；2025E：{index * 10}",
        "table_cells_slice_obj": {"caption": "预测表", "context": "", "rows": [
            {"row_index": index, "headers": headers, "cells": cells}]},
        "content_ltks": f"指标{index}", "content_with_weight": f"指标{index}原文",
    }


def candidates(records, limit=None):
    return group_table_candidates([r["chunk_id"] for r in records],
                                  {r["chunk_id"]: r for r in records}, limit=limit)


@pytest.mark.parametrize("grouped", [False, True])
def test_reranker_receives_source_context_without_rewriting_company_rows(monkeypatch, grouped):
    from service.core.rag.nlp import search_v2
    records = [row(1, table="a"), row(1, table="b")]
    for record, name in zip(records, ["润本股份.pdf", "国电电力.pdf"]):
        record["docnm_kwd"] = name
        record["table_cells_slice_obj"]["caption"] = "可比公司估值"
        record["table_semantic_text_with_weight"] = "公司：华能国际；2025E PE：10"
    before = deepcopy(records)
    fields = {r["chunk_id"]: r for r in records}
    groups = candidates(records) if grouped else None
    result = Dealer.SearchResult(total=2, candidate_ids=list(fields), candidate_fields=fields,
                                 table_candidates=groups.tables if groups else {})
    calls = []
    def scorer(question, texts):
        calls.extend(texts)
        return [0.8, 0.7], None
    monkeypatch.setattr(search_v2, "rerank_similarity", scorer)
    Dealer.__new__(Dealer).rerank_by_model(result, "华能国际2025E PE")
    assert '"文档名": "润本股份.pdf"' in calls[0]
    assert '"文档名": "国电电力.pdf"' in calls[1]
    for text in calls:
        assert '"表题": "可比公司估值"' in text
        assert "不代表正文中每个数值的主体" in text
        assert text.split("\n正文：\n", 1)[1] == "公司：华能国际；2025E PE：10"
    assert records == before


def test_table_flood_leaves_slots_for_later_text_and_other_tables():
    records = [row(i) for i in range(1, 15)] + [
        {"chunk_id": "body", "content_with_weight": "收入增长原因"}, row(1, table="t2")]
    original = deepcopy(records)
    grouped = candidates(records, limit=3)
    assert grouped.ids == ["t1-1", "body", "t2-1"]
    table = grouped.fields["t1-1"]
    assert len(table["table_candidate_obj"]["rows"]) == 14
    assert len(table["table_candidate_obj"]["member_chunk_ids"]) == 14
    assert len(table["table_candidate_obj"]["omitted_chunk_ids"]) == 0
    assert records == original


def test_no_cross_tenant_or_parent_version_grouping():
    first, other_kb, other_version = row(1), row(2, kb="2"), row(3)
    other_version["table_object_ref_kwd"] = "old.json"
    assert len(candidates([first, other_kb, other_version]).ids) == 3


def test_invalid_or_unbound_rows_remain_individual_candidates():
    first, second = row(1), row(2)
    first["table_binding_kwd"] = "unbound"
    second["table_cells_slice_obj"]["rows"][0]["headers"] = []
    assert len(candidates([first, second]).ids) == 2


def test_grouped_rows_survive_domain_roundtrip_without_parent_loading():
    grouped = candidates([row(1), row(7)]).fields["t1-1"]
    restored = source_record(evidence_from_record(grouped))
    assert restored["table_candidate_obj"]["matched_row_indices"] == [1, 7]
    assert "指标7" in restored["content_with_weight"]
    assert len(evidence_from_record(grouped).structure.table.rows) == 2


def test_expansion_preserves_distant_matches_and_unions_neighbor_windows(monkeypatch):
    monkeypatch.setenv("RAG_TABLE_CONTEXT_MAX_TOKENS", "2000")
    source = candidates([row(2), row(9)]).fields["t1-2"]
    parent = {"table_id": "t1", "cells_json": {"binding_status": "bound", "data_rows": [
        {"row_index": i, "headers": ["指标", "2025E"], "cells": [f"指标{i}", str(i * 10)]}
        for i in range(1, 12)
    ]}}
    expanded = expand_table_chunk_context(source, "指标2和指标9是多少", table_object=parent)
    context = expanded["table_context_obj"]
    assert context["matched_row_indices"] == [2, 9]
    assert context["row_indices"] == [1, 2, 3, 4, 7, 8, 9, 10, 11]
    assert context["omitted_matched_row_indices"] == []
    assert expanded["content_with_weight"].count("| 指标 | 2025E |") == 1


def test_grouping_happens_before_dealer_rerank_cutoff():
    dealer = Dealer.__new__(Dealer)
    records = [row(i) for i in range(1, 10)] + [{
        "chunk_id": "body", "content_ltks": "增长原因", "content_with_weight": "增长原因", "kb_id": "1"}]
    from types import SimpleNamespace
    branch = SimpleNamespace(name="keyword_original", ids=[r["chunk_id"] for r in records],
                             fields={r["chunk_id"]: r for r in records}, scores={}, highlight={}, total=10)
    result = dealer._merge_search_branches([branch], {"keyword_original": 1}, rrf_k=60, rerank_candidate_size=2)
    assert result.candidate_ids == [r["chunk_id"] for r in records]
    assert result.table_candidates == {}
    assert result.recall_fusion_scores["t1-9"] == pytest.approx(1 / 69)
    assert result.recall_fusion_scores["t1-1"] == pytest.approx(1 / 61)
    assert result.total == 10


def test_regrouping_subquery_results_unions_rows_and_sources():
    first = candidates([row(1), row(2)]).fields["t1-1"]
    second = candidates([row(2), row(9)]).fields["t1-2"]
    regrouped = candidates([first, second])
    grouped = regrouped.fields[regrouped.ids[0]]
    assert grouped["table_candidate_obj"]["matched_row_indices"] == [1, 2, 9]
    assert grouped["table_candidate_obj"]["member_chunk_ids"] == ["t1-1", "t1-2", "t1-9"]


def test_distinct_row_sets_have_distinct_evidence_ids():
    first = candidates([row(1), row(2)]).fields["t1-1"]
    second = candidates([row(1), row(9)]).fields["t1-1"]
    assert evidence_from_record(first).id != evidence_from_record(second).id
    reverse = candidates([row(2), row(1)]).fields["t1-2"]
    assert evidence_from_record(first).id == evidence_from_record(reverse).id


def test_context_budget_keeps_matches_before_neighbors(monkeypatch):
    monkeypatch.setenv("RAG_TABLE_CONTEXT_MAX_TOKENS", "90")
    source = candidates([row(2), row(9)]).fields["t1-2"]
    parent = {"table_id": "t1", "cells_json": {"binding_status": "bound", "data_rows": [
        {"row_index": i, "headers": ["指标", "2025E"], "cells": [f"指标{i}", str(i * 10)]}
        for i in range(1, 12)
    ]}}
    result = expand_table_chunk_context(source, "比较指标", table_object=parent)
    assert {2, 9} <= set(result["table_context_obj"]["row_indices"])
    assert result["table_context_obj"]["truncated"]


def test_oversized_row_bounds_reranking_without_corrupting_cells():
    from service.core.rag.utils import num_tokens_from_string
    from service.core.retrieval.table_candidates import MAX_CANDIDATE_TOKENS
    source = row(1)
    source["table_semantic_text_with_weight"] = "长表格字段与数据 " * 3000
    grouped = candidates([source]).fields["t1-1"]
    assert num_tokens_from_string(grouped["table_semantic_text_with_weight"]) <= MAX_CANDIDATE_TOKENS
    assert grouped["table_candidate_obj"]["rerank_text_truncated"]
    assert grouped["table_candidate_obj"]["rows"][0]["cells"] == source["table_cells_slice_obj"]["rows"][0]["cells"]


def test_evaluation_counts_selected_rows_but_not_rows_removed_from_context():
    from evaluation.retrieval import table_row_chunk_match_score
    source = candidates([row(1), row(7)]).fields["t1-1"]
    expected = {"document_name": "报告.pdf", "row_cells": ["指标7", "70"]}
    assert table_row_chunk_match_score(expected, source) == 1.0
    source["table_context_obj"] = {"reliable": True, "rows": [{"cells": ["指标1", "10"]}]}
    assert table_row_chunk_match_score(expected, source) == 0.0


def test_expansion_does_not_restore_discarded_scored_rows():
    source = candidates([row(2)]).fields['t1-2']
    source['table_candidate_obj'].update(row_ranking_source='model', scored_row_indices=[2, 3])
    parent = {'table_id': 't1', 'cells_json': {'binding_status': 'bound', 'data_rows': [
        {'row_index': i, 'headers': ['指标', '2025E'], 'cells': [f'指标{i}', str(i)]}
        for i in range(1, 5)
    ]}}
    result = expand_table_chunk_context(source, '指标2', table_object=parent)
    assert 2 in result['table_context_obj']['row_indices']
    assert 3 not in result['table_context_obj']['row_indices']


def test_rerank_scores_rows_once_then_groups_only_selected_prefix(monkeypatch):
    from types import SimpleNamespace
    from service.core.rag.nlp import search_v2
    calls = []
    records = [row(1), row(2), row(3), {
        'chunk_id': 'body', 'content_ltks': '正文', 'content_with_weight': '正文', 'kb_id': '1',
    }]
    def scorer(question, texts):
        calls.append(texts)
        return [0.8, 0.9, 0.01, 0.7], None
    monkeypatch.setattr(search_v2, 'rerank_similarity', scorer)
    branch = SimpleNamespace(name='keyword_original', ids=[r['chunk_id'] for r in records],
                             fields={r['chunk_id']: r for r in records}, scores={}, highlight={}, total=4)
    dealer = Dealer.__new__(Dealer)
    result = dealer._merge_search_branches(
        [branch], {'keyword_original': 1}, rrf_k=60, rerank_candidate_size=2, question='查询',
    )
    assert calls == []
    assert result.candidate_ids == ['t1-1', 't1-2', 't1-3', 'body']
    assert result.table_candidates == {}
    options = Dealer.RetrievalOptions(page_size=2, final_reranker_weight=1.0)
    ranked = dealer._rerank_results(result, '查询', options)
    assert ranked[0] == [1, 0, 3]
    assert len(calls) == 1 and len(calls[0]) == 4
    output = dealer._build_retrieval_result(result, ranked, options)
    assert len(calls) == 1
    assert len(output['chunks']) == 2
    table = next(iter(output['table_candidates'].values()))
    assert table.matched_row_indices == (2, 1)
    assert [r['rerank_score'] for r in table.matched_rows] == [0.9, 0.8]
    assert table.scored_row_indices == (2, 1, 3)
    assert table.member_chunk_ids == ('t1-2', 't1-1')
    assert output['chunks'][0]['rrf_score'] == result.recall_fusion_scores['t1-2']
    assert output['chunks'][1]['chunk_id'] == 'body'


def test_group_count_cutoff_stops_exactly_when_fifth_identity_is_seen():
    records = [row(1), row(2), row(3), row(4)]
    records.insert(1, {'chunk_id': 'body1'})
    records.extend([{'chunk_id': 'body2'}, {'chunk_id': 'body3'}, row(5),
                    {'chunk_id': 'body4'}, row(6), {'chunk_id': 'body5'}])
    fields = {r['chunk_id']: r for r in records}
    ids = list(fields)
    selected = select_candidate_ids(ids, fields, limit=5)
    assert selected == ids[:9]
    assert 't1-6' not in selected
    assert group_table_candidates(selected, fields).ids == ['t1-1', 'body1', 'body2', 'body3', 'body4']
    assert select_candidate_ids(ids, fields, limit=5, offset=5) == ['body5']


def test_top_k_keeps_low_scores_without_extra_threshold(monkeypatch):
    from service.core.rag.nlp import search_v2
    fields = {'a': {'content_with_weight': '正文甲'}, 'b': {'content_with_weight': '正文乙'}}
    result = Dealer.SearchResult(total=2, candidate_ids=list(fields), candidate_fields=fields)
    monkeypatch.setattr(search_v2, 'rerank_similarity', lambda *_: ([0.02, 0.01], None))
    indexes, _ = Dealer.__new__(Dealer)._rerank_results(
        result, '查询', Dealer.RetrievalOptions(page_size=2, similarity_threshold=0.5),
    )
    assert indexes == [0, 1]


def test_pre_rerank_cutoff_does_not_collect_late_rows_of_admitted_table():
    records = [row(1), row(2), {'chunk_id': 'body'}, row(3)]
    fields = {r['chunk_id']: r for r in records}
    assert select_candidate_ids(list(fields), fields, limit=2) == ['t1-1', 't1-2', 'body']
    assert select_candidate_ids(list(fields), fields, limit=1) == ['t1-1']
