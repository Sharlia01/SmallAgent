"""数值问题共用证据核验；不生成财务契约，原文完整进入回答阶段。"""

import json
from types import SimpleNamespace

import pytest

from agent.loop import AgentLoop
from agent.executor import ToolExecutor
from agent.plans import AgentPlan, AgentTask, validate_plan
from domain.assessment import EvidenceAssessment
from evaluation.retrieval import aggregate_query_metrics
from service.core.answering.turn import execute_agent_turn
from service.core.answering.answer_context import build_answer_context, build_answer_messages
from service.core.evidence import sufficiency
from service.core.evidence.adapter import assessment_from_record, collection_from_records
from service.core.evidence.verification import verify_evidence
from service.core.evidence.bindings import BridgeExtraction
from service.core.query.context import QueryContext
from service.core.query.contract import QueryGoal, QueryRequest, QueryUnderstanding
from tests.agent_helpers import ModelClient, SearchTool, initial, records, task


pytestmark = pytest.mark.unit


def verdict(supported=True, indices=(1,)):
    return {
        "sufficient": supported,
        "reason": "输入与对应关系明确" if supported else "缺少基期数据",
        "missing_requirements": [] if supported else ["2022年营业收入"],
        "supporting_chunk_indices": list(indices),
        "goal_coverage": [{"goal_id": "g1", "status": "supported" if supported else "missing",
                           "supporting_chunk_indices": list(indices) if supported else []}],
    }


def plan(question):
    return initial(goals=[dict(id="g1", text=question, source_spans=[question])],
                   tasks=[task(query=question)])


@pytest.fixture(autouse=True)
def enable_verification(monkeypatch):
    monkeypatch.setenv("RAG_EVIDENCE_SUFFICIENCY_ENABLED", "true")
    monkeypatch.setenv("RAG_EVIDENCE_SUFFICIENCY_FAIL_OPEN", "false")
    monkeypatch.setenv("RAG_EVIDENCE_SUFFICIENCY_TRANSIENT_RETRY", "false")


@pytest.mark.parametrize("question,text", [
    ("2023年营业收入是多少，下降原因是什么？",
     "2023年营业收入为80亿元。电价下降10%导致收入减少。"),
    ("2023年营业收入同比增长多少？",
     "2023年营业收入120亿元，2022年为100亿元，两年均采用合并口径。"),
    ("2023年ROIC是多少？", "2023年ROIC为12%。"),
])
def test_numeric_question_uses_one_verifier_and_preserves_original_context(monkeypatch, question, text):
    planner = ModelClient([plan(question)])
    checker = ModelClient([verdict()])
    monkeypatch.setattr(sufficiency, "get_verification_client", lambda: checker)
    tool = SearchTool([records(text=text)])
    query = QueryContext(QueryRequest.from_user(question), _resolved=True)
    run = AgentLoop(client=planner, model="test", tools=[tool]).run(context=query)

    assert run.stop_reason == "evidence_ready"
    assert len(planner.calls) == len(checker.calls) == len(tool.calls) == 1
    assert not hasattr(query.understanding, "financial")
    payload = json.loads(checker.calls[0]["messages"][1]["content"])
    assert payload["question"] == question
    assert payload["candidate_evidence"][0]["content"] == text
    assert set(payload) == {"question", "candidate_evidence", "goals"}

    context = build_answer_context(query, run.evidence, run.assessment)
    assert context.blocks[0].text == text
    assert context.citations[0].evidence is run.evidence.get("c1")
    messages = build_answer_messages(context, [])
    assert text in messages[-1]["content"]
    assert "需要计算时自行" in messages[0]["content"]
    assert verify_evidence(query, run.evidence, previous=run.assessment) is run.assessment
    assert len(checker.calls) == 1


def test_missing_input_replans_and_rechecks_accumulated_originals(monkeypatch):
    question = "2023年营业收入同比增长多少？"
    planner = ModelClient([plan(question), {"tasks": [task(id="s2", query="2022年营业收入")]}])
    partial = verdict(False, ())
    partial["goal_coverage"][0]["supporting_chunk_indices"] = [1]
    # 新协议只输出逐目标引用；缺基期不应丢掉已取得的本期输入。
    partial.pop("supporting_chunk_indices")
    complete = verdict(True, (1, 2))
    complete.pop("supporting_chunk_indices")
    checker = ModelClient([partial, complete])
    monkeypatch.setattr(sufficiency, "get_verification_client", lambda: checker)
    tool = SearchTool([records(text="2023年营业收入120亿元。"),
                       records("c2", text="2022年营业收入100亿元，与2023年口径一致。")])
    run = AgentLoop(client=planner, model="test", tools=[tool]).run(
        context=QueryContext(QueryRequest.from_user(question), _resolved=True))
    assert run.stop_reason == "evidence_ready" and run.steps == 2
    assert len(planner.calls) == len(checker.calls) == 2
    second = json.loads(checker.calls[1]["messages"][1]["content"])
    assert [item["chunk_id"] for item in second["candidate_evidence"]] == ["c1", "c2"]
    feedback = json.loads(planner.calls[1]["messages"][1]["content"])["assessment"]
    assert feedback["goal_coverage"][0]["status"] == "missing"
    assert "financial_fact_bundle" not in feedback


def test_verification_failure_stops_numeric_question_without_supplementary_search(monkeypatch):
    question = "2023年营业收入是多少？"
    planner, checker = ModelClient([plan(question)]), ModelClient([TimeoutError("provider unavailable")])
    monkeypatch.setattr(sufficiency, "get_verification_client", lambda: checker)
    tool = SearchTool([records(text="2023年营业收入80亿元。")])
    run = AgentLoop(client=planner, model="test", tools=[tool]).run(
        context=QueryContext(QueryRequest.from_user(question), _resolved=True))
    assert run.stop_reason == "verification_failed" and run.outcome == "execution_error"
    assert len(tool.calls) == len(planner.calls) == len(checker.calls) == 1


def test_unbound_table_reaches_verifier_as_structure_without_financial_grid_gate():
    query = QueryContext(QueryRequest.from_user("报告年度销售收入是多少？"), _resolved=True)
    original = "|指标|报告年度|\n|销售收入|80亿元|\n报告年度指2023年。"
    evidence = collection_from_records([dict(chunk_id="table", content_with_weight=original,
                                              evidence_type_kwd="table", table_binding_kwd="unbound")])
    checker = ModelClient([dict(sufficient=False, reason="结构不足以对应", missing_requirements=["完整表格结构"],
                               supporting_chunk_indices=[])])
    report = verify_evidence(query, evidence, client=checker)
    assert not report.sufficient and len(checker.calls) == 1
    payload = json.loads(checker.calls[0]["messages"][1]["content"])
    assert "content" not in payload["candidate_evidence"][0]
    assert not payload["candidate_evidence"][0]["table"]["reliable"]
    assert evidence.get("table").text == original
    assert payload["candidate_evidence"][0]["table_binding_kwd"] == "unbound"


def test_old_financial_payload_is_ignored_without_restoring_verification_authority():
    record = dict(sufficient=True, reason="历史报告", supporting_chunk_indices=[1],
                  financial_fact_bundle={"facts": [{"unexpected": "legacy schema"}]},
                  derived_evidence=[{"status": "supported", "result": "999"}],
                  supporting_fact_ids=["old"], query_version="forged", evidence_version="forged")
    restored = assessment_from_record(record)
    assert isinstance(restored, EvidenceAssessment)
    assert not restored.matches("forged", "forged", "forged")
    assert not {"financial_fact_bundle", "derived_evidence", "supporting_fact_ids"} & restored.to_dict().keys()


def test_retired_financial_statistics_are_not_reported_as_zero_success():
    metrics = aggregate_query_metrics([dict(retrieval={"evidence_sufficiency": {
        "financial_fact_bundle": {"requirement_results": [{"status": "supported"}]}}})])
    assert not any(key.startswith("financial_") for key in metrics)


def test_repaired_subject_uses_generic_goal_binding():
    query = QueryContext(QueryRequest.from_user("先查推荐公司，再查它的收入"),
                         QueryUnderstanding((QueryGoal("g1", "推荐公司的收入", subject_slot="company"),),
                                            entity_names=()))
    previous = (AgentTask("first", query="推荐公司", output_slot="company", goal_ids=("g1",)),)
    retry = AgentTask("retry", query="推荐公司 完整报告", output_slot="retry_company",
                      binding_key="company", goal_ids=("g1",))
    consumer = AgentTask("income", query_template="{retry_company}收入", depends_on=("retry",),
                         required_slots=("retry_company",), goal_ids=("g1",))
    planned = AgentPlan((retry, consumer))
    assert validate_plan(planned, query, {"search_knowledge_base"}, previous=previous) is planned
    tool = SearchTool([records(text="推荐国电电力"), records("income", text="营业收入80亿元")])
    executor = ToolExecutor([tool], bridge_extractor=lambda **kwargs: BridgeExtraction(
        kwargs["slot"], "国电电力", 1, "c1", "doc1", "model", "推荐国电电力"))
    discovery = executor.execute_task(retry, query, {}, {})
    result = executor.execute_task(consumer, query, {"retry": discovery}, {})
    assert result.status == "succeeded"
    assert tool.calls[-1].query == "国电电力收入"
    assert tool.calls[-1].company_hints.entity_names == ("国电电力",)


def test_turn_delivers_inputs_to_answer_model_without_server_calculation(monkeypatch):
    question = "2023年营业收入同比增长多少？"
    original = "2023年营业收入120亿元，2022年营业收入100亿元，两年口径相同。"
    planner, checker = ModelClient([plan(question)]), ModelClient([verdict()])
    monkeypatch.setattr(sufficiency, "get_verification_client", lambda: checker)
    answer_calls = []

    def answer(**kwargs):
        answer_calls.append(kwargs)
        assert original in kwargs["messages"][-1]["content"]
        assert "需要计算时自行" in kwargs["messages"][0]["content"]
        return [SimpleNamespace(choices=[SimpleNamespace(
            delta=SimpleNamespace(content="同比增长20%（120÷100－1） ##1$$。", reasoning_content=None),
            finish_reason="stop")])]

    result = execute_agent_turn(
        index_names="test", question=question, planner_client=planner,
        answer_client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=answer))),
        retrieval_function=lambda *args, **kwargs: {"chunks": records(text=original)},
    )
    assert result.response_mode == "answer" and "20%" in result.answer
    assert len(planner.calls) == len(checker.calls) == len(answer_calls) == 1
    assert result.documents[0]["content_with_weight"] == original
    assert "financial_requirements" not in result.raw_retrieval
    assert "financial_bindings" not in result.raw_retrieval
    assert "derived_evidence" not in result.raw_retrieval["evidence_sufficiency"]
