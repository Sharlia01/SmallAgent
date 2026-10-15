from service.core.evidence.verification import verify_evidence
from dataclasses import asdict, replace
from service.core.evidence.adapter import collection_from_records
from tests.evidence_helpers import make_evidence_record, make_tool_result, make_query
import json
from types import SimpleNamespace

import pytest

from agent.orchestrator import AgentOrchestrator, AgentRun
from tests.evidence_helpers import messages_for_records as build_answer_messages
from agent.schemas import ToolResult
from service.core.finance import facts as financial_facts
from service.core.evidence import sufficiency as evidence_sufficiency
from service.core.entities.attribution import EntityRegistry, annotate_document
from service.core.finance.facts import (
    project_fact_context,
)
from tests.financial_helpers import assess_for_question as assess_financial_facts, financial_spec
from service.core.query.contract import QueryGoal
from service.core.runtime.context import query_runtime
from service.core.retrieval.table_context import expand_table_chunk_context


pytestmark = pytest.mark.unit


@pytest.fixture
def registry(monkeypatch):
    registry = EntityRegistry({"version": "test", "entities": [
        {"entity_id": "A", "canonical_name": "A公司", "kind": "company", "stock_codes": ["000001.SH"]},
        {"entity_id": "B", "canonical_name": "B公司", "kind": "company", "stock_codes": ["000002.SH"]},
    ]})
    monkeypatch.setattr(financial_facts, "get_entity_registry", lambda: registry)
    monkeypatch.setattr("service.core.entities.attribution.get_entity_registry", lambda: registry)
    return registry


def chunk(text, identifier="mixed"):
    return {"chunk_id": identifier, "document_id": "report", "content_with_weight": text}


def table(rows, *, caption="单位：亿元", document=None):
    parent = {"table_id": "t1", "cells_json": {
        "binding_status": "bound", "binding_source": "deterministic", "caption": caption, "data_rows": rows,
    }}
    return {"chunk_id": "table-hit", "document_id": "report", "document_name": "A公司.pdf",
            "evidence_type_kwd": "table", "table_binding_kwd": "bound", "table_id_kwd": "t1",
            "table_row_start_int": rows[0]["row_index"], "content_with_weight": "原始表格",
            **(document or {})}, parent


def expand(rows, *, caption="单位：亿元", document=None):
    hit, parent = table(rows, caption=caption, document=document)
    return expand_table_chunk_context(hit, "B公司2023年营业收入", table_object=parent)


def test_mixed_chunk_only_supplies_the_target_fact_and_keeps_original(registry):
    raw = chunk("A公司2023年营业收入为120亿元，B公司2023年营业收入为80亿元。")
    bundle = assess_financial_facts("B公司2023年营业收入是多少？", collection_from_records([raw]))
    assert bundle.sufficient
    assert {(fact.entity_id, fact.value) for fact in bundle.facts} == {("A", "120"), ("B", "80")}
    assert {entry["status"] for entry in bundle.matches} == {"usable", "not_applicable"}
    projected = project_fact_context(collection_from_records([raw]), bundle)[0]
    assert "80亿元" in projected.text
    assert "120亿元" not in projected.text
    assert collection_from_records([raw]).get(raw["chunk_id"]).text == raw["content_with_weight"]
    fact = next(fact for fact in bundle.facts if fact.entity_id == "B")
    assert raw["content_with_weight"][fact.source["start"]:fact.source["end"]] == fact.source["quote"]
    assert fact.value_kind == fact.statement_scope == "unknown"


def test_forged_extraction_status_and_name_number_cooccurrence_are_not_proof(registry):
    raw = chunk("A公司2023年营业收入为120亿元，B公司2023年营业收入为80亿元。")
    raw["financial_facts"] = [{"entity_id": "B", "value": "120", "attribution_status": "verified"}]
    raw["answer_context"] = "B公司2023年营业收入为120亿元。"
    messages = build_answer_messages("B公司2023年营业收入是多少？", [raw], [])
    assert "80亿元" in messages[-1]["content"]
    assert "120亿元" not in messages[-1]["content"]


def test_ordered_respectively_quote_is_verified_but_not_reinserted_whole(registry):
    raw = chunk("A公司、B公司2023年营业收入分别为120亿元、80亿元。")
    bundle = assess_financial_facts("B公司2023年营业收入是多少？", collection_from_records([raw]))
    assert bundle.sufficient
    assert all(fact.verification_method == "ordered_respectively" for fact in bundle.facts)
    assert "120亿元" not in project_fact_context(collection_from_records([raw]), bundle)[0].text


@pytest.mark.parametrize("text", [
    "A公司与B公司相比，2023年营业收入为120亿元。",
    "A公司旗下B公司2023年营业收入为120亿元。",
    "本公司2023年营业收入为120亿元。",
    "A公司2023年营业收入为120亿元，B公司也增长了。",
])
def test_ambiguous_or_missing_company_relations_cannot_support_b(registry, text):
    assert not assess_financial_facts("B公司2023年营业收入是多少？", collection_from_records([chunk(text)])).sufficient


def test_comparison_has_separate_requirements_and_does_not_use_company_set_membership(registry):
    goals = (QueryGoal("a", "A公司2023年营业收入"), QueryGoal("b", "B公司2023年营业收入"))
    raw = chunk("A公司2023年营业收入为120亿元，B公司2022年营业收入为80亿元。")
    bundle = assess_financial_facts("比较A公司与B公司2023年营业收入", collection_from_records([raw]), goals)
    assert not bundle.sufficient
    assert [entry["status"] for entry in bundle.goal_coverage] == ["supported", "missing"]
    valid = chunk("A公司2023年营业收入为120亿元，B公司2023年营业收入为80亿元。")
    assert assess_financial_facts("比较A公司与B公司2023年营业收入", collection_from_records([valid]), goals).sufficient


def test_period_binding_is_local_to_each_company(registry):
    question = "A公司2023年营业收入与B公司2024年营业收入分别是多少？"
    raw = chunk("A公司2023年营业收入为120亿元，B公司2024年营业收入为80亿元。")
    bundle = assess_financial_facts(question, collection_from_records([raw]))
    assert bundle.sufficient
    assert [(r.entity_id, r.period["year"]) for r in bundle.requirements] == [("A", 2023), ("B", 2024)]


@pytest.mark.parametrize(("question", "text"), [
    ("B公司2023年实际营业收入是多少？", "B公司2023E营业收入为80亿元。"),
    ("B公司2023年合并口径营业收入是多少？", "B公司2023年营业收入为80亿元。"),
    ("B公司2023年全年营业收入是多少？", "B公司2023年上半年营业收入为80亿元。"),
])
def test_period_and_explicit_scopes_are_not_guessed(registry, question, text):
    assert not assess_financial_facts(question, collection_from_records([chunk(text)])).sufficient


def test_correct_fact_can_answer_despite_other_unknown_evidence(registry):
    raw = [chunk("B公司旗下项目2023年营业收入为120亿元。", "unknown"),
           chunk("B公司2023年营业收入为80亿元。", "good")]
    assert assess_financial_facts("B公司2023年营业收入是多少？", collection_from_records(raw)).sufficient


def test_conflicting_values_and_unknown_scope_are_distinct(registry):
    known = [chunk("B公司2023年实际合并口径营业收入为80亿元。", "one"),
             chunk("B公司2023年实际合并口径营业收入为81亿元。", "two")]
    bundle = assess_financial_facts("B公司2023年营业收入是多少？", collection_from_records(known))
    assert bundle.goal_coverage[0]["status"] == "conflict"
    unknown = [chunk("B公司2023年营业收入为80亿元。", "one"),
               {**chunk("B公司2023年营业收入为81亿元。", "two"), "document_id": "other-report"}]
    assert assess_financial_facts("B公司2023年营业收入是多少？", collection_from_records(unknown)).goal_coverage[0]["status"] == "ambiguous"


def test_unit_normalization_deduplicates_equivalent_values(registry):
    raw = [chunk("B公司2023年营业收入为80亿元。", "one"),
           chunk("B公司2023年营业收入为8,000百万元。", "two")]
    assert assess_financial_facts("B公司2023年营业收入是多少？", collection_from_records(raw)).sufficient


@pytest.mark.parametrize(("raw_value", "value"), [("８０．００", "80.00"), ("（８０．００）", "-80.00"), ("⑩", "10")])
def test_table_value_preserves_original_numeric_text_alongside_decimal(registry, raw_value, value):
    raw = expand([{"row_index": 2, "headers": ["公司", "2023年营业收入（亿元）"], "cells": ["B公司", raw_value]}])
    bundle = assess_financial_facts("B公司2023年营业收入是多少？", collection_from_records([raw]))
    assert bundle.sufficient
    assert bundle.facts[0].raw_value == raw_value and bundle.facts[0].value == value


def test_expanded_peer_rows_use_their_own_company_not_report_filename(registry):
    rows = [{"row_index": 3, "headers": ["公司", "2023年营业收入（亿元）"], "cells": ["A公司", "120"]},
            {"row_index": 4, "headers": ["公司", "2023年营业收入（亿元）"], "cells": ["B公司", "80"]}]
    raw = expand(rows)
    bundle = assess_financial_facts("B公司2023年营业收入是多少？", collection_from_records([raw]))
    assert bundle.sufficient
    fact = next(fact for fact in bundle.facts if fact.entity_id == "B")
    assert fact.source["row"] == 4 and fact.source["column"] == 1
    assert fact.source["binding_evidence"][0]["company_quotes"] == ["B公司"]
    assert "120" not in project_fact_context(collection_from_records([raw]), bundle)[0].text.split("事实ID：")[0]


def test_multi_company_columns_are_bound_individually(registry):
    raw = expand([{"row_index": 2, "headers": ["指标", "A公司 / 2023年", "B公司 / 2023年"],
                   "cells": ["营业收入", "120", "80"]}])
    bundle = assess_financial_facts("B公司2023年营业收入是多少？", collection_from_records([raw]))
    assert bundle.sufficient
    assert next(fact for fact in bundle.facts if fact.entity_id == "B").source["column"] == 2


def test_clipped_columns_preserve_original_coordinates_and_full_company_code_proof(registry, monkeypatch):
    monkeypatch.setenv("RAG_TABLE_CONTEXT_MAX_COLUMNS", "3")
    rows = [{"row_index": 5, "headers": ["公司", "股票代码", "2021年营业收入", "2022年营业收入", "2023年营业收入（亿元）"],
             "cells": ["B公司", "000002.SH", "60", "70", "80"]}]
    raw = expand(rows)
    bundle = assess_financial_facts("B公司2023年营业收入是多少？", collection_from_records([raw]))
    assert bundle.sufficient
    assert next(fact for fact in bundle.facts if fact.value == "80").source["column"] == 4
    rows[0]["cells"][1] = "000001.SH"
    assert not assess_financial_facts("B公司2023年营业收入是多少？", collection_from_records([expand(rows)])).sufficient


def test_statement_requires_report_subject_and_section_not_filename(registry):
    rows = [{"row_index": 3, "context": "合并利润表；单位：亿元", "headers": ["指标", "2023年"],
             "cells": ["营业收入", "80"]}]
    assert not assess_financial_facts("B公司2023年营业收入是多少？", collection_from_records([expand(rows)])).sufficient
    document = annotate_document([{"content_with_weight": "报告主体：B公司"}], "B公司.pdf", registry)
    raw = expand(rows, document=document)
    assert assess_financial_facts("B公司2023年营业收入是多少？", collection_from_records([raw])).sufficient
    next(item for item in document["entity_attribution_obj"]["document_subject_evidence"]
         if item["source"] == "explicit_label")["quote"] = "报告主体：A公司"
    assert not assess_financial_facts("B公司2023年营业收入是多少？", collection_from_records([expand(rows, document=document)])).sufficient


def test_model_reconstructed_table_is_not_verified_by_its_bound_flag(registry):
    raw = expand([{"row_index": 2, "headers": ["公司", "2023年营业收入"], "cells": ["B公司", "80"]}])
    raw["table_context_obj"]["binding_source"] = "vision"
    assert not assess_financial_facts("B公司2023年营业收入是多少？", collection_from_records([raw])).sufficient


def test_final_messages_exclude_history_numbers_and_keep_frontend_citations(registry):
    documents = [chunk("A公司2023年营业收入为120亿元。", "a"), chunk("B公司2023年营业收入为80亿元。", "b")]
    messages = build_answer_messages("B公司2023年营业收入是多少？", documents,
                                    [{"role": "assistant", "content": "B公司营业收入为120亿元。"}])
    assert "120亿元" not in json.dumps(messages, ensure_ascii=False)
    assert "[1]" in messages[-1]["content"] and "80亿元" in messages[-1]["content"]
    assert "120亿元" in documents[0]["content_with_weight"]


def test_final_fact_gate_rejects_wrong_company_despite_verified_tool_result(registry, monkeypatch):
    calls = []
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: calls.append(kw))))
    orchestrator = AgentOrchestrator(planner_client=client, planner_model="planner", answer_client=client,
                                     answer_model="answer", tools=[])
    question = "B公司2023年营业收入是多少？"
    tool_result = make_tool_result(
        tool_name="search_knowledge_base", query=question, content="A公司2023年营业收入为120亿元。",
        sources=[make_evidence_record(
            source_type="knowledge_base", source_id="a", title="report.pdf",
            content="A公司2023年营业收入为120亿元。",
        )],
        metadata={"query_contract": {"resolved_question": question}, "evidence_sufficiency": {"sufficient": True}},
    )
    monkeypatch.setattr(orchestrator, "_run_agent", lambda **kwargs: verified_agent_run(results=[tool_result], **kwargs))
    result = orchestrator.run(session_id="s", question=question)
    assert result.response_mode == "refuse" and result.answer_stream is None
    assert calls == []


def test_session_documents_also_use_fact_selection(registry, monkeypatch):
    # 本例只验证财务事实选择；共享需求启用的语义核验另有覆盖。
    monkeypatch.setenv("RAG_EVIDENCE_SUFFICIENCY_ENABLED", "false")
    calls = []
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: calls.append(kw) or [])))
    orchestrator = AgentOrchestrator(planner_client=client, planner_model="planner", answer_client=client,
                                     answer_model="answer", tools=[])
    monkeypatch.setattr(orchestrator, "_run_agent", lambda **kwargs: verified_agent_run(**kwargs))
    result = orchestrator.run(session_id="s", question="B公司2023年营业收入是多少？",
                              session_context="A公司2023年营业收入为120亿元，B公司2023年营业收入为80亿元。")
    assert result.response_mode == "answer"
    assert result.retrieved_content == []
    assert len(calls) == 1
    assert "120亿元" not in json.dumps(calls[0]["messages"], ensure_ascii=False)


def test_fact_gate_survives_disabled_or_fail_open_sufficiency(registry, monkeypatch):
    monkeypatch.setenv("RAG_EVIDENCE_SUFFICIENCY_ENABLED", "false")
    monkeypatch.setenv("RAG_EVIDENCE_SUFFICIENCY_FAIL_OPEN", "true")
    decision = verify_evidence(make_query("B公司2023年营业收入是多少？"),
        collection_from_records([chunk("A公司2023年营业收入为120亿元。")]))
    assert decision.sufficient is False and decision.source == "rule"


def test_rule_coverage_keeps_fact_ids_local_to_each_goal_even_in_shared_chunks(registry):
    raw = chunk("A公司2023年营业收入为120亿元，B公司2023年营业收入为80亿元。")
    goals = (QueryGoal("a", "A公司2023年营业收入"), QueryGoal("b", "B公司2023年营业收入"),
             QueryGoal("missing", "B公司2024年营业收入"))
    decision = verify_evidence(make_query("比较A公司和B公司2023年营业收入，再查询B公司2024年营业收入", goals),
        collection_from_records([raw]))
    assert not decision.sufficient
    facts = {fact.fact_id: fact for fact in decision.facts.facts}
    assert {facts[identifier].entity_id for identifier in decision.goal_coverage[0]["supporting_fact_ids"]} == {"A"}
    assert {facts[identifier].entity_id for identifier in decision.goal_coverage[1]["supporting_fact_ids"]} == {"B"}


def test_disabled_semantic_verifier_preserves_verified_fact_audit(registry, monkeypatch):
    monkeypatch.setenv("RAG_EVIDENCE_SUFFICIENCY_ENABLED", "false")
    decision = verify_evidence(make_query("B公司2023年营业收入是多少？"),
        collection_from_records([chunk("B公司2023年营业收入为80亿元。")]))
    assert decision.sufficient
    assert decision.supporting_fact_ids == decision.facts.selected_fact_ids


@pytest.mark.parametrize(("question", "goals"), [
    ("B公司去年营业收入是多少？", (QueryGoal("b", "B公司2023年营业收入"),)),
    ("B公司2023至2025年营业收入分别是多少？", (QueryGoal("b", "B公司2023年和2025年营业收入"),)),
])
def test_planned_years_do_not_override_unresolved_relative_periods_or_ranges(registry, question, goals):
    raw = chunk("B公司2023年营业收入为80亿元，B公司2025年营业收入为90亿元。")
    assert not assess_financial_facts(question, collection_from_records([raw]), goals).sufficient


def test_sufficiency_model_only_sees_projected_facts_not_audit_rows(registry, monkeypatch):
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
            "sufficient": True, "reason": "事实完整", "missing_requirements": [], "supporting_chunk_indices": [1],
        })))])
    monkeypatch.setenv("RAG_EVIDENCE_SUFFICIENCY_ENABLED", "true")
    raw = expand([{"row_index": 2, "headers": ["公司", "2023年营业收入（亿元）"], "cells": ["A公司", "120"]},
                  {"row_index": 3, "headers": ["公司", "2023年营业收入（亿元）"], "cells": ["B公司", "80"]}])
    decision = verify_evidence(make_query("B公司2023年营业收入是多少？"),
        collection_from_records([raw]),
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    assert decision.sufficient
    payload = json.loads(calls[0]["messages"][1]["content"])
    assert "table_context_obj" not in payload["candidate_evidence"][0]
    assert "数值：120" not in calls[0]["messages"][1]["content"]
    assert decision.facts.selected_fact_ids


def test_derived_result_uses_decimal_and_records_input_fact_ids(registry):
    raw = expand([{"row_index": 3, "context": "B公司合并利润表；单位：亿元",
                   "headers": ["指标", "B公司 / 2022A", "B公司 / 2023A"], "cells": ["营业收入", "0.1", "0.3"]}])
    bundle = assess_financial_facts("B公司2022年和2023年营业收入相差多少？", collection_from_records([raw]))
    assert bundle.sufficient
    calculation = bundle.derived_evidence[0]
    assert calculation["result"] == "0.2" and len(calculation["input_fact_ids"]) == 2
    assert {item["scope"] for item in calculation["inputs"]} == {"B"}


def test_cross_company_calculation_binds_each_operand(registry):
    raw = expand([{"row_index": 2, "headers": ["指标", "A公司 / 2023A", "B公司 / 2023A"],
                   "cells": ["营业收入", "120", "80"]}])
    bundle = assess_financial_facts("A公司和B公司2023年营业收入相差多少？", collection_from_records([raw]))
    assert bundle.sufficient
    assert bundle.derived_evidence[0]["result"] == "40"
    assert [item["scope"] for item in bundle.derived_evidence[0]["inputs"]] == ["A", "B"]


def test_unknown_independent_calculation_scopes_are_not_treated_as_equal(registry):
    raw = [chunk("B公司2022年营业收入为60亿元。", "one"), chunk("B公司2023年营业收入为80亿元。", "two")]
    bundle = assess_financial_facts("B公司2022年和2023年营业收入相差多少？", collection_from_records(raw))
    assert not bundle.sufficient
    with pytest.raises(ValueError):
        build_answer_messages("B公司2022年和2023年营业收入相差多少？", raw, [])


def test_agent_union_keeps_goals_and_cannot_bypass_the_final_fact_gate(registry, monkeypatch):
    question = "比较A公司和B公司2023年营业收入"
    goals = [QueryGoal("a", "A公司2023年营业收入"), QueryGoal("b", "B公司2023年营业收入")]
    results = []
    for identifier, content in [("a", "A公司2023年营业收入为120亿元。"), ("b", "B公司2022年营业收入为80亿元。")]:
        results.append(make_tool_result(tool_name="search_knowledge_base", query=question, content=content,
            sources=[make_evidence_record(source_type="knowledge_base", source_id=identifier, title="report.pdf", content=content,
                                metadata={"chunk_id": identifier, "document_id": "report"})],
            metadata={"query_contract": {"resolved_question": question}, "query_intent": {"goals": [g.to_dict() for g in goals]},
                      "evidence_sufficiency": {"sufficient": True}}))
    context = make_query(question, goals)
    monkeypatch.setattr("agent.orchestrator.prepare_query_context", lambda *args, **kwargs: context)
    received = []
    def check(question, evidence, **kwargs):
        received.append(kwargs["goals"])
        return evidence_sufficiency.EvidenceAssessment(True, "模拟语义放行", [], [1, 2], "model", 2, ["a", "b"])
    monkeypatch.setattr("service.core.evidence.verification.assess_evidence_sufficiency", check)
    calls = []
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: calls.append(kw))))
    orchestrator = AgentOrchestrator(planner_client=client, planner_model="planner", answer_client=client,
                                     answer_model="answer", tools=[])
    monkeypatch.setattr(orchestrator, "_run_agent", lambda **kwargs: verified_agent_run(results=results, **kwargs))
    execution = orchestrator.run(session_id="s", question=question)
    assert execution.response_mode == "refuse" and calls == []
    assert received == []  # 事实缺口直接拒绝，不进入语义模型。
    assert {item.goal_id for item in execution.assessment.facts.requirements} == {"a", "b"}


def test_successful_single_kb_result_does_not_skip_session_fact_check(registry, monkeypatch):
    question = "B公司2023年营业收入是多少？"
    result = make_tool_result(tool_name="search_knowledge_base", query=question, content="证据",
        sources=[make_evidence_record(source_type="knowledge_base", source_id="b", title="report.pdf",
                            content="B公司2023年营业收入为80亿元。", metadata={"chunk_id": "b", "document_id": "report"})],
        metadata={"query_contract": {"resolved_question": question}, "evidence_sufficiency": {"sufficient": True}})
    calls = []
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: calls.append(kw))))
    orchestrator = AgentOrchestrator(planner_client=client, planner_model="planner", answer_client=client,
                                     answer_model="answer", tools=[])
    monkeypatch.setattr(orchestrator, "_run_agent", lambda **kwargs: verified_agent_run(results=[result], **kwargs))
    execution = orchestrator.run(session_id="s", question=question, session_context="B公司2023年营业收入为81亿元。")
    assert execution.response_mode == "refuse" and calls == []


def test_goal_omission_does_not_disable_financial_grounding(registry):
    goal = QueryGoal("reason", "解释经营情况")
    bundle = assess_financial_facts("B公司2023年营业收入是多少？", collection_from_records([chunk("A公司2023年营业收入为120亿元。")]), (goal,))
    assert bundle.applicable and not bundle.sufficient


def test_approximate_value_is_preserved_and_not_declared_an_exact_conflict(registry):
    raw = chunk("B公司2023年实际合并口径营业收入约为80亿元。")
    bundle = assess_financial_facts("B公司2023年营业收入是多少？", collection_from_records([raw]))
    assert bundle.sufficient and bundle.facts[0].value_qualifier == "approximate"
    assert "数值：约80亿元" in project_fact_context(collection_from_records([raw]), bundle)[0].text
    exact = chunk("B公司2023年实际合并口径营业收入为80.03亿元。", "exact")
    assert assess_financial_facts("B公司2023年营业收入是多少？", collection_from_records([raw, exact])).goal_coverage[0]["status"] == "ambiguous"


def test_financial_explanation_can_keep_non_numeric_context(registry):
    raw = chunk("B公司2023年营业收入为80亿元。营业收入下降主要受电价下调影响。")
    messages = build_answer_messages("B公司2023年营业收入是多少，下降原因是什么？", [raw], [])
    assert "营业收入下降主要受电价下调影响" in messages[-1]["content"]


def candidate_client(text, *, value="80", verdict="supported"):
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        payload = json.loads(kwargs["messages"][1]["content"])
        if "entities" in payload:
            response = {"candidates": [{"chunk_id": "mixed", "entity_id": "B", "metric": "营业收入",
                "period_text": "2023年", "raw_value": value, "unit": "亿元", "start": 0, "end": len(text),
                "quote": text, "attribution_status": "verified"}]}
        elif "candidates" in payload:
            response = {"verdicts": [{"fact_id": item["fact_id"], "status": verdict,
                                     "quote": item["source"]["quote"]} for item in payload["candidates"]]}
        else:
            response = {"sufficient": True, "reason": "事实齐全", "missing_requirements": [], "supporting_chunk_indices": [1]}
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(response, ensure_ascii=False)))])
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))), calls


def test_optional_model_extraction_requires_separate_verification_and_reuses_server_cache(registry, monkeypatch):
    monkeypatch.setenv("RAG_FINANCIAL_FACT_MODEL_ENABLED", "true")
    monkeypatch.setenv("RAG_EVIDENCE_SUFFICIENCY_ENABLED", "true")
    text = "A公司2023年营业收入为120亿元。B公司披露2023年业绩，其营业收入为80亿元。"
    client, calls = candidate_client(text)
    with query_runtime():
        decision = verify_evidence(make_query("B公司2023年营业收入是多少？"),
        collection_from_records([chunk(text)]),
        client=client)
        assert decision.sufficient and len(calls) == 3
        messages = build_answer_messages("B公司2023年营业收入是多少？", [chunk(text)], [])
        assert "数值：80亿元" in messages[-1]["content"]
        assert "120亿元" not in messages[-1]["content"]
    # 仅携带元数据不能在另一请求中重放已核验的模型事实。
    untrusted = {**chunk(text), "financial_facts": [asdict(fact) for fact in decision.facts.facts]}
    with pytest.raises(ValueError):
        build_answer_messages("B公司2023年营业收入是多少？", [untrusted], [])


def test_model_candidate_claim_and_cooccurrence_cannot_override_a_rule_relationship(registry, monkeypatch):
    monkeypatch.setenv("RAG_FINANCIAL_FACT_MODEL_ENABLED", "true")
    text = "A公司2023年营业收入为120亿元，B公司2023年营业收入为80亿元，其经营稳健。"
    client, calls = candidate_client(text, value="120")
    # 即使模型错误地同意候选，也不能覆盖明确的 B 公司对应 80 的规则绑定。
    from service.core.finance.model_extraction import extract_and_verify_candidates
    requirements = list(financial_spec("B公司2023年营业收入是多少？").facts)
    with query_runtime():
        facts = extract_and_verify_candidates(collection_from_records([chunk(text)]), requirements, client, registry=registry)
        assert facts[0].attribution_status == "unknown" and len(calls) == 2


def test_model_unknown_verdict_does_not_accept_extraction_verified_claim(registry, monkeypatch):
    monkeypatch.setenv("RAG_FINANCIAL_FACT_MODEL_ENABLED", "true")
    text = "B公司披露2023年业绩，其营业收入为80亿元。"
    client, _ = candidate_client(text, verdict="unknown")
    bundle = assess_financial_facts("B公司2023年营业收入是多少？", collection_from_records([chunk(text)]), allow_model=True, client=client)
    assert not bundle.sufficient
    assert bundle.facts[0].attribution_status == "unknown"


def test_explicit_currency_is_required_and_not_inherited_from_another_company(registry):
    raw = chunk("A公司2023年营业收入为120人民币亿元，B公司2023年营业收入为80美元。")
    bundle = assess_financial_facts("B公司2023年美元营业收入是多少？", collection_from_records([raw]))
    assert bundle.sufficient
    assert next(fact for fact in bundle.facts if fact.entity_id == "B").currency == "USD"
    assert not assess_financial_facts("B公司2023年人民币营业收入是多少？", collection_from_records([raw])).sufficient


def test_unnamed_target_in_mixed_company_sources_requires_confirmation(registry):
    raw = chunk("A公司2023年营业收入为120亿元，B公司2023年营业收入为80亿元。")
    bundle = assess_financial_facts("2023年营业收入是多少？", collection_from_records([raw]))
    assert bundle.applicable and not bundle.sufficient


def test_unnamed_target_does_not_inherit_report_without_declared_binding(registry):
    raw = chunk("B公司2023年营业收入为80亿元。")
    assert not assess_financial_facts("2023年营业收入是多少？", collection_from_records([raw])).sufficient
    document = annotate_document([{"content_with_weight": "报告主体：B公司"}], "B公司.pdf", registry)
    assert not assess_financial_facts("2023年营业收入是多少？", collection_from_records([{**raw, **document}])).sufficient
    assert not assess_financial_facts("它2023年营业收入是多少？", collection_from_records([{**raw, **document}])).sufficient


def test_real_peer_company_in_another_company_report_is_bound_to_its_row():
    from service.core.resources.paths import config_file
    registry = EntityRegistry(json.loads(config_file("entity_registry.json").read_text()))
    rows = [{"row_index": 5, "headers": ["公司", "2023年营业收入（亿元）"], "cells": ["龙源电力", "80"]}]
    hit, parent = table(rows)
    hit["document_name"] = "国电电力.pdf"
    raw = expand_table_chunk_context(hit, "龙源电力2023年营业收入", table_object=parent)
    assert assess_financial_facts("龙源电力2023年营业收入是多少？", collection_from_records([raw]), registry=registry).sufficient
    assert not assess_financial_facts("国电电力2023年营业收入是多少？", collection_from_records([raw]), registry=registry).sufficient


def test_revenue_unit_is_not_invented_for_a_bound_number(registry):
    raw = expand([{"row_index": 2, "headers": ["公司", "2023年营业收入"], "cells": ["B公司", "80"]}], caption="财务数据")
    bundle = assess_financial_facts("B公司2023年营业收入是多少？", collection_from_records([raw]))
    assert not bundle.sufficient
    assert bundle.facts[0].unit is None and bundle.matches[0]["status"] == "unknown"


def test_qualitative_goal_does_not_require_an_unrequested_profit_number(registry):
    question = "B公司2023年营业收入是多少，净利润下降原因是什么？"
    goals = (QueryGoal("amount", "B公司2023年营业收入是多少"), QueryGoal("reason", "B公司净利润下降原因是什么"))
    raw = chunk("B公司2023年营业收入为80亿元。净利润下降主要受燃料成本上升影响。")
    bundle = assess_financial_facts(question, collection_from_records([raw]), goals)
    assert bundle.sufficient
    assert {requirement.metric for requirement in bundle.requirements} == {"营业收入"}


def test_numeric_why_question_still_requires_attributed_facts(registry):
    bundle = assess_financial_facts("为什么B公司2023年营业收入为80亿元？", collection_from_records([chunk("A公司2023年营业收入为80亿元。")]))
    assert bundle.applicable and not bundle.sufficient


def test_revenue_growth_percentage_is_not_a_revenue_amount(registry):
    raw = expand([{"row_index": 2, "headers": ["公司", "2023年营业收入同比增速（%）"], "cells": ["B公司", "20"]}])
    bundle = assess_financial_facts("B公司2023年营业收入是多少？", collection_from_records([raw]))
    assert not bundle.sufficient
    assert bundle.facts[0].metric == "营业收入"
    assert bundle.facts[0].measure_kind == "growth_rate"
    assert bundle.facts[0].comparison_basis == "year_over_year"


def test_explicit_snapshot_date_is_not_collapsed_into_an_annual_period(registry):
    raw = chunk("B公司截至2023年12月31日总资产为80亿元。")
    assert assess_financial_facts("B公司截至2023年12月31日总资产是多少？", collection_from_records([raw])).sufficient
    assert not assess_financial_facts("B公司2023年度总资产是多少？", collection_from_records([raw])).sufficient


def test_unresolved_relative_period_cannot_take_any_retrieved_year(registry):
    raw = chunk("B公司2023年营业收入为80亿元。")
    assert not assess_financial_facts("B公司去年营业收入是多少？", collection_from_records([raw])).sufficient


def test_opening_company_and_matching_ticker_can_ground_a_forecast_statement(registry):
    document = annotate_document([{"content_with_weight": "B公司（000002.SH）\n证券研究报告"}], "B公司.pdf", registry)
    rows = [{"row_index": 3, "context": "关键财务与估值指标；单位：亿元", "headers": ["指标", "2023E"],
             "cells": ["营业收入", "80"]}]
    assert assess_financial_facts("B公司2023年预测营业收入是多少？", collection_from_records([expand(rows, document=document)])).sufficient
    rows[0]["context"] = "子公司财务预测；单位：亿元"
    assert not assess_financial_facts("B公司2023年预测营业收入是多少？", collection_from_records([expand(rows, document=document)])).sufficient


def test_forecast_caption_does_not_assign_type_to_unmarked_historical_columns(registry):
    raw = expand([{"row_index": 2, "headers": ["指标", "B公司 / 2022", "B公司 / 2023E"],
                   "cells": ["营业收入", "60", "80"]}], caption="财务预测；单位：亿元")
    bundle = assess_financial_facts("B公司2022年营业收入是多少？", collection_from_records([raw]))
    assert bundle.sufficient
    assert next(fact for fact in bundle.facts if fact.period["year"] == 2022).value_kind == "unknown"


def test_company_name_is_not_a_business_scope():
    registry = EntityRegistry({"version": "test", "entities": [{
        "entity_id": "wind", "canonical_name": "风电股份", "kind": "company"}]})
    bundle = assess_financial_facts("风电股份2023年营业收入是多少？",
                              collection_from_records([chunk("风电股份2023年营业收入为80亿元。")]), registry=registry)
    assert bundle.sufficient and bundle.facts[0].business_scope == "unknown"


@pytest.mark.parametrize(("metric", "value"), [("存货", "119"), ("预付账款", "14"), ("固定资产", "265")])
def test_supported_balance_metrics_use_their_own_row(registry, metric, value):
    rows = [
        {"row_index": 2, "headers": ["指标", "B公司 / 2025E"], "cells": ["应收票据及账款", "17"]},
        {"row_index": 3, "headers": ["指标", "B公司 / 2025E"], "cells": [metric, value]},
    ]
    bundle = assess_financial_facts(f"B公司2025E{metric}是多少？", collection_from_records([expand(rows)]))
    assert bundle.sufficient
    assert [f.value for f in bundle.facts if f.fact_id in bundle.selected_fact_ids] == [value]


def test_growth_rate_lookup_covers_every_year_without_calculating(registry):
    rows = [{"row_index": 2, "headers": ["指标", "B公司 / 2025E", "B公司 / 2026E", "B公司 / 2027E"],
             "cells": ["营业收入增长率（%）", "24.81", "20.96", "15.68"]}]
    question = "B公司2025E-2027E营业收入增长率分别是多少？"
    bundle = assess_financial_facts(question, collection_from_records([expand(rows)]))
    assert bundle.sufficient and not bundle.derived_evidence
    assert {r.period["year"] for r in bundle.requirements} == {2025, 2026, 2027}
    assert {f.value for f in bundle.facts if f.fact_id in bundle.selected_fact_ids} == {"24.81", "20.96", "15.68"}
    rows[0]["cells"][2] = "—"
    missing = assess_financial_facts(question, collection_from_records([expand(rows)]))
    assert not missing.sufficient
    assert any("2026" in reason for reason in missing.missing_requirements)


@pytest.mark.parametrize("text", ["B公司2025年营业收入为100亿元。", "B公司2025年营业收入环比增长率为20%。"])
def test_yoy_rate_requires_its_own_measure_and_basis(registry, text):
    bundle = assess_financial_facts("B公司2025年营业收入同比增长率是多少？", collection_from_records([chunk(text)]))
    assert bundle.applicable and not bundle.sufficient


def test_positive_decline_magnitude_is_not_a_positive_growth_rate(registry):
    text = "B公司2025年营业收入同比下降20%。"
    bundle = assess_financial_facts("B公司2025年营业收入同比增长率是多少？", collection_from_records([chunk(text)]))
    assert bundle.applicable and not bundle.sufficient


def test_operating_cashflow_label_does_not_imply_net_cashflow(registry):
    text = "B公司2025年经营现金流增长率为20.47%。"
    evidence = collection_from_records([chunk(text)])
    assert assess_financial_facts("B公司2025年经营现金流增长率是多少？", evidence).sufficient
    assert not assess_financial_facts("B公司2025年经营活动现金流量净额增长率是多少？", evidence).sufficient


def test_text_growth_rate_and_amount_remain_distinct_in_answer_context(registry):
    raw = chunk("B公司2025年营业收入为100亿元。B公司2025年营业收入同比增长率为20%。")
    evidence = collection_from_records([raw])
    bundle = assess_financial_facts("B公司2025年营业收入同比增长率是多少？", evidence)
    assert bundle.sufficient
    projected = project_fact_context(evidence, bundle)[0].text
    assert "数值：20%" in projected and "growth_rate" in projected
    assert "100亿元" not in projected


def test_generic_rate_with_conflicting_comparison_bases_is_ambiguous(registry):
    text = "B公司2025年营业收入同比增长率为20%。B公司2025年营业收入环比增长率为20%。"
    assert not assess_financial_facts("B公司2025年营业收入增长率是多少？", collection_from_records([chunk(text)])).sufficient


def test_rate_without_percent_unit_does_not_borrow_table_money_unit(registry):
    rows = [{"row_index": 2, "headers": ["公司", "2025年营业收入增长率"], "cells": ["B公司", "20"]}]
    assert not assess_financial_facts("B公司2025年营业收入增长率是多少？", collection_from_records([expand(rows)])).sufficient


def test_unknown_metric_cannot_be_silently_removed_by_query_planning(registry):
    raw = chunk("B公司2025年营业收入为100亿元。")
    goals = (QueryGoal("revenue", "B公司2025年营业收入是多少"),)
    bundle = assess_financial_facts("B公司2025年营业收入和净资产收益率分别是多少？", collection_from_records([raw]), goals)
    assert bundle.applicable and not bundle.sufficient
    assert bundle.unresolved_targets and bundle.missing_requirements


def test_unknown_only_financial_target_requires_verification(registry):
    bundle = assess_financial_facts("B公司2025年资本充足率是多少？", collection_from_records([]))
    assert bundle.applicable and not bundle.sufficient and not bundle.requirements
    assert bundle.unresolved_targets[0]["reason"] == "unsupported_metric"


def test_qualitative_question_does_not_request_an_unknown_numeric_fact(registry):
    bundle = assess_financial_facts("B公司资本充足率下降的原因是什么？", collection_from_records([]))
    assert not bundle.applicable


@pytest.mark.parametrize("question", ["计算B公司2024年和2025年营业收入增长率", "计算B公司2024-2025年营业收入增长率"])
def test_calculated_growth_uses_amounts_not_disclosed_percentages(registry, question):
    rows = [
        {"row_index": 2, "headers": ["指标", "B公司 / 2024A", "B公司 / 2025A"], "cells": ["营业收入", "100", "120"]},
        {"row_index": 3, "headers": ["指标", "B公司 / 2024A", "B公司 / 2025A"], "cells": ["营业收入增长率（%）", "10", "20"]},
    ]
    bundle = assess_financial_facts(question, collection_from_records([expand(rows)]))
    assert bundle.sufficient
    assert bundle.derived_evidence[0]["result"] == "20.0"
    assert {i["measure_kind"] for i in bundle.derived_evidence[0]["inputs"]} == {"level"}


def test_rate_difference_is_in_percentage_points(registry):
    rows = [{"row_index": 2, "headers": ["指标", "B公司 / 2024A", "B公司 / 2025A"],
             "cells": ["营业收入同比增长率（%）", "10", "20"]}]
    bundle = assess_financial_facts("B公司2024年和2025年营业收入同比增长率相差多少？", collection_from_records([expand(rows)]))
    assert bundle.sufficient
    assert bundle.derived_evidence[0]["result"] == "10"
    assert bundle.derived_evidence[0]["unit"] == "个百分点"


def test_historical_compound_metric_is_read_without_restoring_verification_scope(registry):
    from service.core.evidence.adapter import assessment_from_record
    bundle = assess_financial_facts("B公司2025年营业收入是多少？", collection_from_records([chunk("B公司2025年营业收入为100亿元。")]))
    record = bundle.to_dict()
    fact = record["facts"][0]
    fact.pop("measure_kind")
    fact.pop("comparison_basis")
    fact["metric"] = "营业收入/同比"
    assessment = assessment_from_record({"financial_fact_bundle": record})
    assert assessment.facts.facts[0].metric == "营业收入"
    assert assessment.facts.facts[0].measure_kind == "growth_rate"
    assert not assessment.query_version


def test_unknown_financial_target_cannot_bypass_the_shared_verification_entry(registry):
    query = make_query("B公司2025年资本充足率是多少？")
    assessment = verify_evidence(query, collection_from_records([chunk("B公司2025年资本充足率是20%。")]), require_semantic=False)
    assert not assessment.sufficient
    assert assessment.facts.unresolved_targets


def verified_agent_run(context, results=(), session_id="s", session_context=None):
    """财务核验现由 Agent 循环持有；夹具沿用真实核验入口，不伪造通过报告。"""
    from agent.loop import AgentLoop, collect_tool_evidence
    from service.core.answering.answer_context import session_evidence
    from service.core.query.contract import QueryUnderstanding, QueryGoal
    if context.understanding is None:
        context.understanding = QueryUnderstanding((QueryGoal("whole", context.question),))
    context.understanding = replace(context.understanding, financial=financial_spec(context.question, context.goals))
    evidence = collect_tool_evidence(list(results))
    evidence.merge(session_evidence(session_id, session_context))
    run = AgentRun(tool_results=list(results), evidence=evidence, query=context)
    run.assessment = AgentLoop._assess(context, run)
    return run
