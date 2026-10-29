"""覆盖证据身份、核验快照及模型输入与引用之间的架构约束。"""

from dataclasses import FrozenInstanceError, replace
import json
import re
from types import SimpleNamespace

import pytest

from agent.schemas import ToolResult
from domain.assessment import EvidenceAssessment
from domain.evidence import Evidence, EvidenceLocation, EvidenceSource, RetrievalTrace
from service.core.answering import answer_context as pipeline
from service.core.evidence import verification as verification, sufficiency as evidence_sufficiency
from service.core.evidence.adapter import collection_from_records, evidence_from_record, source_record
from service.core.entities.registry import EntityRegistry
from service.core.query.contract import QueryGoal
from tests.evidence_helpers import make_query, bind_assessment
from service.core.evidence.adapter import assessment_from_record
from service.core.evidence import assessment as evidence_assessment


pytestmark = pytest.mark.unit


def record(identifier="one", text="差旅申请需要审批。", **attributes):
    return {"chunk_id": identifier, "document_id": "report", "document_name": "制度.pdf",
            "content_with_weight": text, **attributes}


def decision(evidence, *, supporting=(1,)):
    return assessment_from_record({
        "sufficient": True, "source": "model", "reason": "证据完整",
        "evaluated_chunk_count": len(evidence), "evaluated_chunk_ids": [item.id for item in evidence],
        "supporting_chunk_indices": list(supporting),
    })


def answer_references(messages):
    """读取实际模型消息中的编号和 JSON，检查上下文与前端引用的一致性。"""
    lines = messages[-1]["content"].splitlines()
    return [(int(match.group(1)), json.loads(lines[index + 1]))
            for index, line in enumerate(lines)
            if (match := re.fullmatch(r"\[(\d+)\]", line))]


@pytest.fixture
def companies(monkeypatch):
    registry = EntityRegistry({"version": "architecture-test", "entities": [
        {"entity_id": "A", "canonical_name": "A公司", "kind": "company"},
        {"entity_id": "B", "canonical_name": "B公司", "kind": "company"},
    ]})
    monkeypatch.setattr(evidence_assessment, "get_entity_registry", lambda: registry)
    monkeypatch.setattr("service.core.entities.registry.get_entity_registry", lambda: registry)
    return registry


def test_original_evidence_is_an_independent_readonly_snapshot():
    raw = record(positions=[[2, 10]], table_cells_slice_obj={"rows": [{"cells": ["B公司", "80"]}]})
    evidence = evidence_from_record(raw)
    version = evidence.version
    raw["positions"][0][0] = 9
    raw["table_cells_slice_obj"]["rows"][0]["cells"][1] = "120"
    assert evidence.location.positions == ((2, 10),)
    assert evidence.structure.table.metadata["table_cells_slice_obj"]["rows"][0]["cells"][1] == "80"
    assert evidence.version == version
    with pytest.raises(FrozenInstanceError):
        evidence.text = "修改原文"
    with pytest.raises(TypeError):
        evidence.structure.table.metadata["table_cells_slice_obj"]["rows"][0]["cells"][1] = "120"


def test_assessment_filters_empty_sources_before_basic_checks():
    evidence = collection_from_records([record(str(index), "") for index in range(25)] + [record("valid")])
    assessment = evidence_sufficiency.assess_evidence_sufficiency(
        "差旅如何审批？", evidence, require_semantic=False)
    assert assessment.evaluated_chunk_ids == ["valid"]


def test_domain_location_can_be_exported_without_parser_specific_fields():
    evidence = Evidence("cell", "80", EvidenceSource("knowledge_base", "report", "报表.pdf"),
                        EvidenceLocation(positions=((2, 10),), page_number=(2, 3), table_id="t1",
                                         row=7, row_start=6, row_end=8))
    raw = source_record(evidence)
    assert raw["page_num_int"] == [2, 3]
    assert raw["positions"] == [[2, 10]]
    assert (raw["table_id_kwd"], raw["table_row_int"], raw["table_row_start_int"], raw["table_row_end_int"]) == ("t1", 7, 6, 8)
    assert evidence_from_record(raw).location == evidence.location


def test_ranking_changes_preserve_identity_and_keep_both_retrieval_traces():
    first = collection_from_records([record(similarity=0.9)], query="差旅流程")
    version = first.version
    original = first.get("one")
    first.merge(collection_from_records([record(similarity=0.6)], query="审批要求"))
    assert len(first) == 1 and first.get("one") is original
    assert first.version == version
    assert [(item.query, item.scores["similarity"]) for item in first.traces("one")] == [("差旅流程", 0.9), ("审批要求", 0.6)]
    copied = first.copy()
    copied.add(evidence_from_record(record("two")))
    assert len(first) == 1 and len(copied) == 2


@pytest.mark.parametrize("changed", [
    {"content_with_weight": "冲突原文"}, {"document_id": "another-report"}, {"positions": [[3, 10]]},
])
def test_same_id_with_different_source_refuses_before_semantic_model(monkeypatch, changed):
    evidence = collection_from_records([record()])
    original = evidence.get("one")
    evidence.merge(collection_from_records([record(**changed)]))
    monkeypatch.setattr(verification, "assess_evidence_sufficiency", lambda *_args, **_kwargs: pytest.fail("冲突来源不应进入模型核验"))
    assessment = verification.verify_evidence(make_query("差旅如何审批？"), evidence)
    assert len(evidence.conflicts) == 1
    assert evidence.get("one") is original
    assert not assessment.sufficient
    assert "身份冲突" in assessment.missing_requirements[0]


def test_assessment_reuse_requires_same_query_evidence_and_policy(monkeypatch):
    query = make_query("差旅如何审批？")
    evidence = collection_from_records([record()])
    calls = []

    def check(question, records, **kwargs):
        calls.append((question, records, kwargs))
        return decision(records)

    monkeypatch.setattr(verification, "assess_evidence_sufficiency", check)
    assessment = verification.verify_evidence(query, evidence)
    evidence.add(evidence.get("one"), RetrievalTrace("第二次查询", {"similarity": 0.3}))
    assert verification.verify_evidence(query, evidence, previous=assessment) is assessment
    assert len(calls) == 1
    extended = evidence.copy()
    extended.add(evidence_from_record(record("two")))
    assert verification.verify_evidence(query, extended, previous=assessment) is not assessment
    new_query = make_query("差旅如何审批？", (QueryGoal("g1", "审批要求", source_spans=("审批",)),))
    assert verification.verify_evidence(new_query, evidence, previous=assessment) is not assessment
    monkeypatch.setenv("RAG_EVIDENCE_SUFFICIENCY_MODEL", "new-check-model")
    assert verification.verify_evidence(query, evidence, previous=assessment) is not assessment
    assert len(calls) == 4
    with pytest.raises(ValueError, match="已失效"):
        pipeline.build_answer_context(query, evidence, assessment)


def test_report_changes_cannot_reuse_a_verified_snapshot(monkeypatch):
    query = make_query("差旅如何审批？")
    evidence = collection_from_records([record()])
    verdict = decision(evidence)
    assessment = bind_assessment(query, evidence, verdict)
    verdict.supporting_chunk_indices.clear()
    assert assessment.supporting_chunk_indices == [1]
    assessment.supporting_chunk_indices.clear()
    with pytest.raises(ValueError, match="已失效"):
        pipeline.build_answer_context(query, evidence, assessment)
    monkeypatch.setattr(verification, "assess_evidence_sufficiency", lambda *_args, **_kwargs: decision(evidence))
    assert verification.verify_evidence(query, evidence, previous=assessment) is not assessment


def test_query_context_preserves_all_goal_provenance():
    goal = QueryGoal("g1", "图2和图3中的利润关系", constraints=("2021年",),
                     source_spans=("图2和图3",), candidate_locators=("图2", "图3"))
    query = make_query("图2和图3中的2021年利润关系？", (goal,))
    assert query.goals[0] is goal
    version = query.version
    query.understanding = replace(query.understanding,
                                  goals=(replace(goal, candidate_locators=("图2",)),))
    assert query.version != version


def test_answer_selection_and_citations_share_one_mapping_without_reverification(monkeypatch):
    query = make_query("差旅如何审批？")
    evidence = collection_from_records([record("irrelevant", "无关材料"), record("supported")])
    assessment = bind_assessment(query, evidence, decision(evidence, supporting=(2,)))
    monkeypatch.setattr(verification, "assess_evidence_sufficiency", lambda *_args, **_kwargs: pytest.fail("格式化不能再次抽取事实"))
    context = pipeline.build_answer_context(query, evidence, assessment)
    messages = pipeline.build_answer_messages(context, [])
    documents = pipeline.answer_documents(context, evidence, assessment)
    assert answer_references(messages)[0][1]["content"] == "差旅申请需要审批。"
    assert "无关材料" not in messages[-1]["content"]
    assert [(item["id"], item["chunk_id"]) for item in documents] == [(1, "supported")]
    assert len(evidence) == 2


def test_implicit_subjects_keep_their_distinct_documents_and_locations():
    query = make_query("国电电力和华能国际2025年营业收入分别是多少？")
    evidence = collection_from_records([
        record("gd", "报告期内，公司实现营业收入100亿元。", document_id="gd-report",
               document_name="国电电力2025年年度报告.pdf", page_num_int=[12], locator_kwd="经营情况",
               similarity=0.9, subject_entity_ids=["legacy-wrong-company"]),
        record("hn", "报告期内，公司实现营业收入200亿元。", document_id="hn-report",
               document_name="华能国际2025年年度报告.pdf", page_num_int=15),
    ])
    assessment = bind_assessment(query, evidence, decision(evidence, supporting=(1, 2)))

    context = pipeline.build_answer_context(query, evidence, assessment)
    references = answer_references(pipeline.build_answer_messages(context, []))
    documents = pipeline.answer_documents(context, evidence, assessment)

    assert [item["document_name"] for _, item in references] == [
        "国电电力2025年年度报告.pdf", "华能国际2025年年度报告.pdf",
    ]
    assert references[0][1]["location"] == {"page": [12], "locator": "经营情况"}
    assert references[1][1]["location"] == {"page": 15}
    for (number, reference), document, block in zip(references, documents, context.blocks):
        assert number == document["id"]
        assert reference["document_id"] == document["document_id"]
        assert reference["content"] == document["content_with_weight"] == block.text
        assert "similarity" not in reference and "subject_entity_ids" not in reference


def test_table_keeps_company_column_and_units_even_outside_column_window():
    query = make_query("华能国际2025年收入是多少？")
    evidence = collection_from_records([record(
        "peer-table", "2025年收入：200", document_name="国电电力投资价值分析.pdf",
        evidence_type_kwd="table", table_binding_kwd="bound", table_id_kwd="t1",
        table_row_start_int=3, table_row_end_int=3,
        table_context_obj={"reliable": True, "caption": "可比公司营业收入", "rows": [{
            "row_index": 3, "column_indices": [1], "headers": ["2025年收入"], "cells": ["200"],
            "context": "电力企业对比", "preamble": "单位：亿元",
            "source_row": {"headers": ["公司", "2025年收入"], "cells": ["华能国际", "200"],
                           "entity_attribution": {"subject_entity_ids": ["wrong-company"]}},
        }]},
    )])
    assessment = bind_assessment(query, evidence, decision(evidence))

    context = pipeline.build_answer_context(query, evidence, assessment)
    _, reference = answer_references(pipeline.build_answer_messages(context, []))[0]

    assert reference["document_name"] == "国电电力投资价值分析.pdf"
    assert "content" not in reference
    table = reference["table"]
    assert table["caption"] == "可比公司营业收入"
    assert table["binding"] == "bound" and table["reliable"] is True
    row = table["matched_rows"][0]
    assert row["context"] == "电力企业对比" and row["preamble"] == "单位：亿元"
    assert row["row_index"] == 3
    assert [cell["in_selected_window"] for cell in row["values"]] == [False, True]
    assert [(cell["header"], cell["value"]) for cell in row["values"]] == [("公司", "华能国际"), ("2025年收入", "200")]
    assert "entity_attribution" not in str(reference)


def test_table_without_parent_expansion_keeps_original_caption():
    query = make_query("报告中的收入是多少？")
    evidence = collection_from_records([record(
        text="收入：200",
        evidence_type_kwd="table", table_binding_kwd="bound",
        table_semantic_text_with_weight="公司经营数据\n单位：亿元\n公司：华能国际；收入：200",
        table_cells_slice_obj={"caption": "公司经营数据", "rows": [
            {"headers": ["公司", "收入"], "cells": ["华能国际", "200"]},
        ]},
    )])
    assessment = bind_assessment(query, evidence, decision(evidence))
    context = pipeline.build_answer_context(query, evidence, assessment)
    reference = answer_references(pipeline.build_answer_messages(context, []))[0][1]
    assert reference["table"]["caption"] == "公司经营数据"
    assert "content" not in reference
    assert "单位：亿元" in reference["table"]["source_notes"]


def test_goal_support_keeps_context_without_adding_unreviewed_or_unrelated_chunks():
    query = make_query("该公司的收入是多少？", (QueryGoal("g1", "公司收入"),))
    evidence = collection_from_records([
        record("subject-context", "本节分析国电电力，以下金额单位均为亿元。"),
        record("revenue", "报告期内，公司实现营业收入100。"),
        record("unrelated", "同一报告中的其他内容。"),
        record("not-reviewed", "预算不足，未送审的内容。"),
    ])
    verdict = replace(decision(evidence, supporting=(2,)),
        evaluated_chunk_ids=["subject-context", "revenue", "unrelated"], evaluated_chunk_count=3,
        goal_coverage=[{"goal_id": "g1", "status": "supported",
                        "supporting_chunk_ids": ["subject-context", "revenue", "not-reviewed"]}])
    assessment = bind_assessment(query, evidence, verdict)

    context = pipeline.build_answer_context(query, evidence, assessment)
    references = answer_references(pipeline.build_answer_messages(context, []))
    assert [block.evidence_id for block in context.blocks] == ["subject-context", "revenue"]
    assert context.goal_support[0].supporting_reference_ids == (1, 2)
    assert [item["content"] for _, item in references] == [
        "本节分析国电电力，以下金额单位均为亿元。", "报告期内，公司实现营业收入100。",
    ]
    assert [item["chunk_id"] for item in pipeline.answer_documents(context, evidence, assessment)] == [
        "subject-context", "revenue",
    ]


def test_goal_navigation_uses_final_references_and_preserves_shared_sources():
    question = "比较甲公司和乙公司的收入，并说明统计口径"
    goals = (QueryGoal("a", "甲公司收入", constraints=("2025年",)),
             QueryGoal("b", "乙公司收入"))
    query = make_query(question, goals)
    evidence = collection_from_records([
        record("irrelevant", "无关内容"), record("a", "甲公司收入100"),
        record("shared", "两家公司使用相同口径"), record("b", "乙公司收入200"),
    ])
    # 核验顺序、原始证据顺序和筛选后的引用顺序均可能不同。
    verdict = replace(decision(evidence, supporting=(1,)),
        evaluated_chunk_ids=["b", "irrelevant", "shared", "a"],
        goal_coverage=[
            {"goal_id": "b", "status": "supported", "supporting_chunk_ids": ["b", "shared"]},
            {"goal_id": "a", "status": "supported", "supporting_chunk_ids": ["shared", "a", "a"]},
        ])
    context = pipeline.build_answer_context(query, evidence, bind_assessment(query, evidence, verdict))
    messages = pipeline.build_answer_messages(context, [])
    navigation = json.loads(messages[-1]["content"].splitlines()[1])
    assert navigation == [
        {"goal_id": "a", "text": "甲公司收入", "constraints": ["2025年"], "supporting_reference_ids": [1, 2]},
        {"goal_id": "b", "text": "乙公司收入", "constraints": [], "supporting_reference_ids": [2, 3]},
    ]
    assert [(number, ref["content"]) for number, ref in answer_references(messages)] == [
        (1, "甲公司收入100"), (2, "两家公司使用相同口径"), (3, "乙公司收入200"),
    ]
    assert question in messages[-1]["content"]
    assert "无关内容" not in messages[-1]["content"]


def test_missing_goal_navigation_keeps_original_question_and_references():
    query = make_query("甲公司的收入是多少？", (QueryGoal("a", "甲公司收入"),))
    evidence = collection_from_records([record("a", "甲公司收入100")])
    context = pipeline.build_answer_context(query, evidence,
        bind_assessment(query, evidence, decision(evidence)))
    messages = pipeline.build_answer_messages(context, [])
    assert context.goal_support == ()
    assert "目标与支持来源" not in messages[-1]["content"]
    assert query.question in messages[-1]["content"]
    assert answer_references(messages)[0][1]["content"] == "甲公司收入100"


def test_reference_json_preserves_newlines_and_embedded_reference_markers():
    text = '  公司说明：\n[99]\n{"content":"保持原文"}\n'
    query = make_query("公司的说明是什么？")
    evidence = collection_from_records([record(text=text, document_name='报告\n[88]".pdf')])
    assessment = bind_assessment(query, evidence, decision(evidence))
    context = pipeline.build_answer_context(query, evidence, assessment)

    references = answer_references(pipeline.build_answer_messages(context, []))
    assert len(references) == 1 and references[0][0] == 1
    assert references[0][1]["content"] == text
    assert references[0][1]["document_name"] == '报告\n[88]".pdf'


def test_mixed_sources_keep_original_citation_numbers_and_web_links():
    query = make_query("综合这些资料说明")
    evidence = collection_from_records([
        record("web", "网页搜索摘要", source_type="web", url="https://example.com/report",
               web_sources=[{"title": "报告", "url": "https://example.com/report"}]),
        record("kb", "内部资料"),
        record("session", "会话资料", source_type="session_document"),
    ])
    assessment = bind_assessment(query, evidence, decision(evidence, supporting=(1, 2, 3)))
    context = pipeline.build_answer_context(query, evidence, assessment)

    references = dict(answer_references(pipeline.build_answer_messages(context, [])))
    assert references[1]["content"] == "网页搜索摘要"
    assert references[1]["url"] == "https://example.com/report"
    assert references[1]["web_sources"][0]["url"] == "https://example.com/report"
    assert references[2]["content"] == "内部资料" and references[3]["content"] == "会话资料"
    assert [(item["id"], item["chunk_id"]) for item in pipeline.answer_documents(context, evidence, assessment)] == [
        (1, "web"), (2, "kb"), (3, "session"),
    ]


@pytest.mark.parametrize("attributes,kind,status", [
    ({"evidence_type_kwd": "table", "table_binding_kwd": "unbound"}, "table", "unbound"),
    ({"evidence_type_kwd": "figure", "visual_status_kwd": "disabled"}, "figure", "disabled"),
])
def test_answer_keeps_unreliable_table_or_unread_figure_status(attributes, kind, status):
    query = make_query("资料说明了什么？")
    evidence = collection_from_records([record(text="原始识别内容", **attributes)])
    assessment = bind_assessment(query, evidence, decision(evidence))
    context = pipeline.build_answer_context(query, evidence, assessment)

    reference = answer_references(pipeline.build_answer_messages(context, []))[0][1]
    assert reference[kind]["binding" if kind == "table" else "status"] == status
    assert reference["content"] == "原始识别内容"


def test_tool_diagnostics_do_not_restore_or_replace_query_context():
    query = make_query("差旅如何审批？")
    evidence = collection_from_records([record()])
    result = ToolResult("search_knowledge_base", "另一个问题", "工具输出", evidence=evidence,
                        diagnostics={"query_contract": {"resolved_question": "错误问题"},
                                     "query_intent": {"goals": [{"id": "fake", "text": "错误目标"}]}})
    assessment = verification.verify_evidence(query, evidence, require_semantic=False)
    context = pipeline.build_answer_context(query, evidence, assessment)
    assert context.question == "差旅如何审批？" and query.goals == ()


def test_session_evidence_preserves_short_original_and_respects_existing_limits():
    text = "  第一段\n\n第二段  "
    first = pipeline.session_evidence("session", text)
    assert next(iter(first)).text == text
    assert next(iter(first)).source.kind == "session_document"
    first.merge(pipeline.session_evidence("session", text))
    assert len(first) == 1
    long = pipeline.session_evidence("session", "长" * 4500)
    assert [len(item.text) for item in long] == [2000, 2000]


@pytest.mark.parametrize("route", ["empty", "disabled", "model", "fallback"])
def test_sufficiency_routes_return_a_report_bound_to_the_actual_query(monkeypatch, route):
    query = make_query("差旅如何审批？")
    records = [] if route == "empty" else [record()]
    monkeypatch.setenv("RAG_EVIDENCE_SUFFICIENCY_ENABLED", "false" if route == "disabled" else "true")
    monkeypatch.setenv("RAG_EVIDENCE_SUFFICIENCY_FAIL_OPEN", "false")
    monkeypatch.setenv("RAG_EVIDENCE_SUFFICIENCY_TRANSIENT_RETRY", "false")
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if route == "fallback":
            raise TimeoutError("模拟服务不可用")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=
            '{"sufficient":true,"reason":"审批流程明确","missing_requirements":[],"supporting_chunk_indices":[1]}'))])

    assessment = verification.verify_evidence(query,
        collection_from_records(records),
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    evidence = collection_from_records(records)
    assert isinstance(assessment, EvidenceAssessment)
    assert assessment.matches(query.version, evidence.version, pipeline.verification_policy_version())
    assert assessment.sufficient is (route in {"disabled", "model"})
    assert len(calls) == (1 if route in {"model", "fallback"} else 0)
    if route == "fallback":
        assert assessment.verification_status == "unavailable"
        assert assessment.diagnostics["attempts"] == 1
        assert assessment.to_dict()["failure_kind"] == assessment.diagnostics["failure_kind"]


@pytest.mark.parametrize("change", ["query", "evidence", "policy", "report"])
def test_bound_report_cannot_be_relabelled_as_a_new_verification(monkeypatch, change):
    query = make_query("差旅如何审批？")
    evidence = collection_from_records([record()])
    assessment = verification.verify_evidence(query, evidence, require_semantic=False)
    if change == "query":
        query = make_query("差旅报销需要什么材料？")
    elif change == "evidence":
        evidence.add(evidence_from_record(record("two", "新增审批规定")))
    elif change == "policy":
        monkeypatch.setenv("RAG_EVIDENCE_SUFFICIENCY_MODEL", "another-verifier")
    else:
        assessment.missing_requirements.append("修改报告")
    with pytest.raises(ValueError, match="已失效"):
        bind_assessment(query, evidence, assessment)


@pytest.mark.parametrize(('markers', 'expected'), [
    ({'table_candidate_obj': {'matched_row_indices': [4, 6]}}, ['context', 'matched', 'matched']),
    ({'table_context_obj': {'matched_row_indices': ['4', '6'], 'matched_row_index': 4}},
     ['context', 'matched', 'matched']),
    ({'table_context_obj': {'matched_row_index': '4'}}, ['context', 'matched', 'context']),
    ({'table_row_int': '4'}, ['context', 'matched', 'context']),
    ({}, ['unknown', 'unknown', 'unknown']),
    ({'table_candidate_obj': {'matched_row_indices': []}, 'table_row_int': 4},
     ['unknown', 'unknown', 'unknown']),
    ({'table_candidate_obj': {'matched_row_indices': [True, 4.5, 'broken']}},
     ['unknown', 'unknown', 'unknown']),
    ({'table_binding_kwd': 'unbound', 'table_row_int': 4}, ['unknown', 'unknown', 'unknown']),
])
def test_answer_messages_preserve_table_retrieval_roles(markers, expected):
    """检查最终模型消息，覆盖多主行、旧索引与无可靠角色的证据。"""
    rows = [
        {'row_index': index, 'headers': ['指标', '2025E'], 'cells': [label, value]}
        for index, label, value in [(3, '应收账款', '15'), (4, '预付账款', '14'), (6, '存货', '140')]
    ]
    attributes = {'evidence_type_kwd': 'table', 'table_binding_kwd': 'bound', **markers}
    attributes['table_context_obj'] = {
        'reliable': True, 'rows': rows, **attributes.get('table_context_obj', {}),
    }
    evidence = collection_from_records([record(text='原始表格', **attributes)])
    query = make_query('2025E预付账款是多少？')
    assessment = bind_assessment(query, evidence, decision(evidence))
    context = pipeline.build_answer_context(query, evidence, assessment)
    messages = pipeline.build_answer_messages(context, [])
    reference = answer_references(messages)[0][1]
    table = reference['table']
    if attributes['table_binding_kwd'] == 'unbound':
        assert table['reliable'] is False
        assert reference['content'] == '原始表格'
        assert 'matched_rows' not in table
        return
    actual = {}
    for role in ('matched', 'context', 'unknown'):
        for output in table[f'{role}_rows']:
            actual[output['row_index']] = role
            original = next(row for row in rows if row['row_index'] == output['row_index'])
            assert [cell['value'] for cell in output['values']] == original['cells']
    assert [actual[row['row_index']] for row in rows] == expected
    assert '角色仅说明检索来源' in messages[0]['content']
    assert 'content' not in reference
    assert pipeline.answer_documents(context, evidence, assessment)[0]['content_with_weight'] == '原始表格'


def test_paired_table_preserves_duplicate_headers_empty_and_zero_without_markdown():
    raw_rows = [{'row_index': 4, 'headers': ['指标', '2025E', '2025E'],
                 'cells': ['预付账款', '0', ''], 'preamble': '单位：百万元'}]
    evidence = collection_from_records([record(
        text='| 指标 | 2025E | 2025E |', evidence_type_kwd='table', table_binding_kwd='bound',
        table_row_int=4, table_cells_slice_obj={'caption': '预测表', 'rows': raw_rows},
        table_semantic_text_with_weight='预测表\n单位：百万元\n指标：预付账款；2025E：0；2025E：（空白）',
    )])
    reference = pipeline._reference_context(evidence.get('one'), evidence.get('one').text)
    assert 'content' not in reference
    table = reference['table']
    assert 'semantic_text' not in table and 'source_notes' not in table
    values = table['matched_rows'][0]['values']
    assert [(cell['header'], cell['value']) for cell in values] == [
        ('指标', '预付账款'), ('2025E', '0'), ('2025E', ''),
    ]
    assert table['matched_rows'][0]['preamble'] == '单位：百万元'


def test_malformed_bound_table_falls_back_without_truncating_cells():
    item = evidence_from_record(record(
        text='保留原始表格', evidence_type_kwd='table', table_binding_kwd='bound',
        table_cells_slice_obj={'rows': [{'headers': ['指标', '2025E'], 'cells': ['预付账款']}]},
    ))
    reference = pipeline._reference_context(item, item.text)
    assert reference['content'] == '保留原始表格'
    assert reference['table']['reliable'] is False
    assert 'matched_rows' not in reference['table']
