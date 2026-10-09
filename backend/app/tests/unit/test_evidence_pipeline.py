"""覆盖证据身份、核验快照及模型输入与引用之间的架构约束。"""

import json
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest

from agent.orchestrator import AgentOrchestrator, AgentRun
from agent.schemas import ToolResult
from domain.assessment import EvidenceAssessment
from domain.evidence import (Evidence, EvidenceCollection, EvidenceLocation, EvidenceSource, EvidenceStructure,
                             EvidenceTable, EvidenceTableRow, RetrievalTrace)
from service.core import evidence_pipeline as pipeline, evidence_verification as verification, evidence_sufficiency, financial_facts, retrieval
from service.core.evidence_adapter import collection_from_records, evidence_from_record, source_record
from service.core.entity_attribution import EntityRegistry
from service.core.grounded_answer import execute_grounded_turn
from service.core.query_contract import QueryGoal
from service.core.retrieval_result import retrieval_result, adapt_retrieval_records, retrieval_record
from tests.evidence_helpers import make_query, bind_assessment
from service.core.evidence_adapter import assessment_from_record
from service.core import evidence_assessment
from service.core import evidence_adapter, grounded_answer


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


@pytest.fixture
def companies(monkeypatch):
    registry = EntityRegistry({"version": "architecture-test", "entities": [
        {"entity_id": "A", "canonical_name": "A公司", "kind": "company"},
        {"entity_id": "B", "canonical_name": "B公司", "kind": "company"},
    ]})
    monkeypatch.setattr(financial_facts, "get_entity_registry", lambda: registry)
    monkeypatch.setattr(evidence_assessment, "get_entity_registry", lambda: registry)
    monkeypatch.setattr("service.core.entity_attribution.get_entity_registry", lambda: registry)
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


def test_typed_table_is_verified_once_without_source_dictionary_roundtrips(companies, monkeypatch):
    headers, cells = ("公司", "2023年营业收入（亿元）"), ("B公司", "80")
    row = EvidenceTableRow(headers, cells, 0, (0, 1), headers, cells)
    item = Evidence("table", "原始表格", EvidenceSource("knowledge_base", "report", "报表.pdf"),
                    structure=EvidenceStructure("table", EvidenceTable(binding="bound", rows=(row,))))
    evidence = EvidenceCollection([item])
    evidence.add(item, RetrievalTrace("营业收入", {"similarity": 0.8}))
    query = make_query("B公司2023年营业收入是多少？")
    real_assess = verification.assess_financial_facts
    calls = []

    def facts(question, sources, *args, **kwargs):
        assert sources is evidence
        calls.append("facts")
        return real_assess(question, sources, *args, **kwargs)

    def semantic(question, sources, *, facts, **kwargs):
        assert sources is evidence and sources.get("table") is item
        assert facts.sufficient and facts.facts[0].value == "80"
        calls.append("semantic")
        return replace(decision(sources), facts=facts)

    def no_conversion(*args, **kwargs):
        pytest.fail("核验阶段不能导出或重建来源字典")

    monkeypatch.setattr(verification, "assess_financial_facts", facts)
    monkeypatch.setattr(verification, "assess_evidence_sufficiency", semantic)
    monkeypatch.setattr(evidence_adapter, "source_record", no_conversion)
    monkeypatch.setattr(evidence_adapter, "source_records", no_conversion)
    monkeypatch.setattr(evidence_adapter, "collection_from_records", no_conversion)
    assessment = verification.verify_evidence(query,
        evidence)
    assert verification.verify_evidence(query, evidence, previous=assessment) is assessment
    answer = pipeline.build_answer_context(query, evidence, assessment)
    assert calls == ["facts", "semantic"]
    assert answer.citations[0].evidence is item
    assert "80亿元" in answer.blocks[0].text
    assert evidence.traces("table")[0].scores["similarity"] == 0.8


@pytest.mark.parametrize("malformed", [{"column_indices": []}, {"source_row": "bad"}, {"headers": "bad"}])
def test_malformed_table_mapping_cannot_supply_verified_facts(companies, malformed):
    row = {"headers": ["公司", "2023年营业收入（亿元）"], "cells": ["B公司", "80"], **malformed}
    evidence = collection_from_records([record(evidence_type_kwd="table", table_binding_kwd="bound",
                                               table_cells_slice_obj={"rows": [row]})])
    table = next(iter(evidence)).structure.table
    assert not table.rows[0].grounded
    assert not financial_facts.assess_financial_facts("B公司2023年营业收入是多少？", evidence).sufficient


def test_model_projection_skips_empty_sources_before_applying_input_limit():
    evidence = collection_from_records([record(str(index), "") for index in range(25)] + [record("valid")])
    inputs = evidence_sufficiency._prepare_evidence(evidence)
    assert [item.evidence.id for item in inputs] == ["valid"]


def test_calculation_indices_follow_actual_model_inputs_and_keep_stable_source_ids(monkeypatch):
    monkeypatch.setenv("RAG_EVIDENCE_SUFFICIENCY_ENABLED", "true")
    table = record("values", "每股收益和每股红利", evidence_type_kwd="table", table_binding_kwd="bound",
                   table_id_kwd="t1", table_cells_slice_obj={"rows": [
                       {"headers": ["指标", "2023年"], "cells": ["每股红利", "0.24"], "row_index": 0},
                       {"headers": ["指标", "2023年"], "cells": ["每股收益", "0.40"], "row_index": 1},
                   ]})
    evidence = collection_from_records([record("blank", ""), table])
    requests = []

    def create(**kwargs):
        requests.append(json.loads(kwargs["messages"][1]["content"]))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
            "sufficient": True, "reason": "已核验输入足以计算", "missing_requirements": [],
            "supporting_chunk_indices": [1], "derived_evidence_ids": ["payout_ratio"],
        })))])

    query = make_query("2023年派息率是多少？")
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    assessment = verification.verify_evidence(query, evidence, client=client)
    assert assessment.sufficient and assessment.evaluated_chunk_ids == ["values"]
    calculation = requests[0]["derived_evidence"][0]
    assert calculation["supporting_chunk_ids"] == ["values"]
    assert calculation["supporting_chunk_indices"] == [1]
    assert [item["chunk_index"] for item in calculation["inputs"]] == [1, 1]
    assert assessment.facts.derived_evidence[0]["supporting_chunk_indices"] == [2]
    context = pipeline.build_answer_context(query, evidence, assessment)
    assert context.citations[0].evidence is evidence.get("values")


def test_retrieval_without_source_id_preserves_identity_in_financial_json(companies, monkeypatch):
    monkeypatch.setenv("RAG_EVIDENCE_SUFFICIENCY_ENABLED", "false")
    raw = {"total": 1, "chunks": [{"content_with_weight": "B公司2023年营业收入为80亿元。"}]}
    query = make_query("B公司2023年营业收入是多少？")
    result, _, _, retrieved = retrieval._verify_retrieved_evidence(
        raw, dealer=None,
        index_names="user-index", options=None, context=query,
    )
    exported = retrieval_record(retrieved)
    identifier = next(iter(retrieved.evidence)).id
    assert exported["chunks"][0]["chunk_id"] == identifier
    assert "80亿元" in exported["chunks"][0]["answer_context"]
    assert retrieved.assessment.matches(query.version, retrieved.evidence.version, pipeline.verification_policy_version())


def test_grounded_default_uses_typed_result_and_reuses_its_report(monkeypatch):
    monkeypatch.setenv("RAG_EVIDENCE_SUFFICIENCY_ENABLED", "false")
    query = make_query("差旅如何审批？")
    evidence = collection_from_records([record()])
    assessment = verification.verify_evidence(query, evidence)
    retrieved = retrieval_result(query, evidence, evidence, assessment, {})
    monkeypatch.setattr(grounded_answer, "query_context_for", lambda request: query)
    monkeypatch.setattr(retrieval, "_retrieve_evidence_result", lambda *args, **kwargs: retrieved)
    monkeypatch.setattr(grounded_answer, "adapt_retrieval_records",
                        lambda *args: pytest.fail("类型化的默认检索不能走历史 JSON 入口"))
    monkeypatch.setattr(verification, "assess_financial_facts",
                        lambda *args, **kwargs: pytest.fail("相同报告不能重复核验"))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: [])))
    result = grounded_answer.execute_grounded_turn(index_names="user-index", question=query.question, answer_client=client)
    assert result.response_mode == "answer"
    assert result.raw_retrieval["query_contract"] == query.request.to_dict()
    assert result.raw_retrieval["evidence_sufficiency"] == assessment.to_dict()
    assert result.documents[0]["chunk_id"] == "one"


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
    goal = QueryGoal("g1", "图2和图3中的利润关系", ("2021年",), (), ("图2和图3",),
                     ({"kind": "time", "value": "2021年", "source_quote": "2021年", "start": 8, "end": 13},),
                     ("图2", "图3"))
    query = make_query("图2和图3中的2021年利润关系？", (goal,))
    assert query.goals[0] is goal
    version = query.version
    query.prepared = replace(query.prepared, analysis=replace(query.prepared.analysis,
                             goals=(replace(goal, candidate_locators=("图2",)),)))
    assert query.version != version


def test_answer_selection_and_citations_share_one_mapping_without_reverification(monkeypatch):
    query = make_query("差旅如何审批？")
    evidence = collection_from_records([record("irrelevant", "无关材料"), record("supported")])
    assessment = bind_assessment(query, evidence, decision(evidence, supporting=(2,)))
    monkeypatch.setattr(verification, "assess_financial_facts", lambda *_args, **_kwargs: pytest.fail("格式化不能再次抽取事实"))
    context = pipeline.build_answer_context(query, evidence, assessment)
    messages = pipeline.build_answer_messages(context, [])
    documents = pipeline.answer_documents(context, evidence, assessment)
    assert "[1] 差旅申请需要审批。" in messages[-1]["content"]
    assert "无关材料" not in messages[-1]["content"]
    assert [(item["id"], item["chunk_id"]) for item in documents] == [(1, "supported")]
    assert len(evidence) == 2


def test_financial_answer_uses_b_fact_while_citation_preserves_full_original(companies):
    raw = record("mixed", "A公司2023年营业收入为120亿元，B公司2023年营业收入为80亿元。",
                 answer_context="B公司2023年营业收入为120亿元。",
                 financial_facts=[{"entity_id": "B", "value": "120", "attribution_status": "verified"}])
    evidence = collection_from_records([record("only-a", "A公司2023年营业收入为120亿元。"), raw])
    query = make_query("B公司2023年营业收入是多少？")
    assessment = verification.verify_evidence(query, evidence, require_semantic=False)
    assert not hasattr(assessment, "financial_fact_bundle")
    assert assessment.to_dict()["supporting_fact_ids"] == assessment.facts.selected_fact_ids
    context = pipeline.build_answer_context(query, evidence, assessment)
    messages = pipeline.build_answer_messages(context, [{"role": "assistant", "content": "旧答案120亿元。"}])
    documents = pipeline.answer_documents(context, evidence, assessment)
    assert "80亿元" in messages[-1]["content"] and "120亿元" not in messages[-1]["content"]
    assert len(messages) == 2 and len(documents) == 1
    assert context.blocks[0].fact_ids == tuple(assessment.facts.selected_fact_ids)
    assert documents[0]["id"] == 1 and documents[0]["chunk_id"] == "mixed"
    assert documents[0]["content_with_weight"] == raw["content_with_weight"]
    assert "financial_facts" not in evidence.get("mixed").structure.attribution.metadata
    documents[0]["financial_facts"][0]["source"]["quote"] = "修改展示数据"
    assert pipeline.build_answer_context(query, evidence, assessment) == context


def test_tool_diagnostics_do_not_restore_or_replace_query_context():
    query = make_query("差旅如何审批？")
    evidence = collection_from_records([record()])
    result = ToolResult("search_knowledge_base", "另一个问题", "工具输出", evidence=evidence,
                        diagnostics={"query_contract": {"resolved_question": "错误问题"},
                                     "query_intent": {"goals": [{"id": "fake", "text": "错误目标"}]}})
    assessment = AgentOrchestrator._verify_evidence(query, evidence, AgentRun([result]))
    context = pipeline.build_answer_context(query, evidence, assessment)
    assert context.question == "差旅如何审批？" and query.goals == ()


def test_typed_retrieval_returns_the_same_query_context_and_server_assessment(monkeypatch):
    query = make_query("差旅如何审批？", (QueryGoal("g1", "差旅审批要求", source_spans=("差旅",)),))
    raw = {"chunks": [{"chunk_id": "one", "doc_id": "report", "docnm_kwd": "制度.pdf",
                       "content_with_weight": "差旅申请需要审批。"}]}
    raw["chunks_before_sufficiency"] = raw["chunks"]
    verdict = decision(collection_from_records(raw["chunks"]))
    seen = []

    def retrieve(index, question, *, context, **options):
        seen.append(context)
        evidence = collection_from_records(raw["chunks"])
        candidates = collection_from_records(raw["chunks_before_sufficiency"])
        report = bind_assessment(context, candidates, verdict)
        return retrieval_result(context, evidence, candidates, report, {})

    monkeypatch.setattr(retrieval, "_retrieve_evidence_result", retrieve)
    result = retrieval.retrieve_evidence("user-index", "差旅", context=query)
    assert seen == [query]
    assert result.assessment.matches(query.version, result.evidence.version, pipeline.verification_policy_version())
    assert "query_contract" not in result.diagnostics and "candidate_chunks" not in result.diagnostics
    events = result.diagnostics["query_runtime"]["events"]
    assert events[-1]["end_reason"] == "evidence_returned"


def test_chat_and_offline_use_the_same_fact_context_and_citations(companies, monkeypatch):
    monkeypatch.setenv("RAG_EVIDENCE_SUFFICIENCY_ENABLED", "false")
    question = "B公司2023年营业收入是多少？"
    raw = {"chunks": [{"chunk_id": "mixed", "doc_id": "report", "docnm_kwd": "报表.pdf",
                       "content_with_weight": "A公司2023年营业收入为120亿元，B公司2023年营业收入为80亿元。"}],
           "evidence_sufficiency": {"sufficient": True, "source": "disabled"}}
    query = make_query(question)
    records, diagnostics = retrieval.format_retrieval_results(raw)
    retrieved = adapt_retrieval_records(query, records, diagnostics)
    tool = ToolResult("search_knowledge_base", question, "", evidence=retrieved.evidence, assessment=retrieved.assessment)
    monkeypatch.setattr(verification, "assess_evidence_sufficiency", lambda *_args, **_kwargs: pytest.fail("相同报告不能重复核验"))
    assessment = AgentOrchestrator._verify_evidence(query, retrieved.evidence, AgentRun([tool]))
    chat = AgentOrchestrator._prepare_answer(query, retrieved.evidence, assessment, AgentRun([tool]), [])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: [])))
    offline = execute_grounded_turn(index_names="user-index", question=question, answer_client=client,
                                    retrieval_function=lambda *_args, **_kwargs: retrieved)
    assert offline.answer_messages == chat.answer_messages
    assert offline.documents == chat.response_documents


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


@pytest.mark.parametrize("source", [
    {"chunk_id": "one", "doc_id": "report", "docnm_kwd": "制度.pdf", "content_with_weight": "差旅申请需要审批。"},
    {"chunk_id": "one", "content_with_weight": "差旅申请需要审批。"},
])
def test_typed_retrieval_preserves_a_real_scoped_report_through_json_formatting(monkeypatch, source):
    monkeypatch.setenv("RAG_EVIDENCE_SUFFICIENCY_ENABLED", "false")
    query = make_query("差旅如何审批？")
    assessment = verification.verify_evidence(query,
        collection_from_records([source]))
    raw = {"chunks": [source], "chunks_before_sufficiency": [source]}
    monkeypatch.setattr(retrieval, "_retrieve_evidence_result", lambda *_args, **_kwargs: retrieval_result(query, collection_from_records([source]),
            collection_from_records([source]), assessment, {}))
    result = retrieval.retrieve_evidence("user-index", query.question, context=query)
    assert result.assessment is assessment
    assert assessment.matches(query.version, result.evidence.version, pipeline.verification_policy_version())


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


def test_partial_financial_report_keeps_supported_goal_proof_for_repair(companies):
    goals = (QueryGoal("a", "A公司2023年营业收入"), QueryGoal("b", "B公司2023年营业收入"))
    query = make_query("比较A公司和B公司2023年营业收入", goals)
    assessment = verification.verify_evidence(query,
        collection_from_records([record("a", "A公司2023年营业收入为120亿元。")]))
    assert not assessment.sufficient
    assert assessment.supporting_chunk_indices == [1]
    assert [entry["status"] for entry in assessment.goal_coverage] == ["supported", "missing"]
    assert assessment.goal_coverage[0]["supporting_fact_ids"] == assessment.supporting_fact_ids


def test_calculation_and_fact_proof_survive_legacy_json_without_polluting_the_report(companies):
    from tests.unit.test_financial_facts import expand

    query = make_query("B公司2022年和2023年营业收入相差多少？")
    raw = expand([{"row_index": 3, "context": "B公司合并利润表；单位：亿元",
                   "headers": ["指标", "B公司 / 2022A", "B公司 / 2023A"], "cells": ["营业收入", "0.1", "0.3"]}])
    evidence = collection_from_records([raw])
    original = verification.verify_evidence(query, evidence, require_semantic=False)
    serialized = original.to_dict()
    restored = bind_assessment(query, evidence, assessment_from_record(serialized))
    serialized["derived_evidence"][0]["result"] = "999"
    serialized["financial_fact_bundle"]["facts"][0]["value"] = "999"
    assert restored.derived_evidence[0]["result"] == "0.2"
    assert original.derived_evidence[0]["result"] == "0.2"
    assert len(restored.derived_evidence[0]["input_fact_ids"]) == 2
    assert restored.matches(query.version, evidence.version, pipeline.verification_policy_version())
    assert pipeline.build_answer_context(query, evidence, restored) == pipeline.build_answer_context(query, evidence, original)
