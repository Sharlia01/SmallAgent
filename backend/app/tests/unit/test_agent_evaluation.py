import json
from types import SimpleNamespace

import pytest

from agent.orchestrator import MinimalAgent
from agent.tools.rag_search import RagSearchTool
from service.core.agent_evaluation import execute_agent_turn
from domain.assessment import EvidenceAssessment
from service.core.end_to_end_evaluation import evaluate_refusal
from service.core import agent_evaluation, evidence_verification, retrieval
from service.core.evidence_adapter import collection_from_records
from service.core.evidence_assessment import finalize_assessment
from service.core.retrieval_result import retrieval_result


pytestmark = pytest.mark.unit


class Client:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return next(self.responses)


def call(query, **extra):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content="", tool_calls=[SimpleNamespace(id="call-1", function=SimpleNamespace(
            name="search_knowledge_base", arguments=json.dumps({"query": query, **extra}),
        ))],
    ))])


def answer_stream():
    return [SimpleNamespace(choices=[SimpleNamespace(
        finish_reason="stop", delta=SimpleNamespace(content="答案。##1$$", reasoning_content=None),
    )])]


def chunk(identifier="e1"):
    return {"chunk_id": identifier, "doc_id": "d1", "docnm_kwd": "报告.pdf",
            "content_with_weight": "2025年全年实际利润为10亿元。"}


def test_agent_eval_keeps_user_contract_when_planner_drops_conditions(monkeypatch):
    monkeypatch.setenv("RAG_EVIDENCE_SUFFICIENCY_ENABLED", "false")
    observed = []
    question = "2025年全年实际利润是多少，不含预测值？"
    history = [{"role": "user", "content": "请看报告"}]
    def retrieve(index, query, *, request, **options):
        observed.append((index, query, request))
        return {"total": 1, "chunks": [chunk()], "query_contract": request.to_dict(),
                "evidence_sufficiency": {"sufficient": True, "source": "model"}}
    planner, answer = Client([call("利润")]), Client([answer_stream()])
    result = execute_agent_turn(index_names="authorized-index", question=question,
                                conversation_history=history, planner_client=planner,
                                answer_client=answer, retrieval_function=retrieve)
    assert observed[0][0:2] == ("authorized-index", "利润")
    assert observed[0][2].user_question == question
    assert observed[0][2].history == (("user", "请看报告"),)
    assert result.answer == "答案。##1$$"
    assert len(observed) == 1
    assert len(planner.calls) == 1
    assert result.raw_retrieval["agent_execution"]["stop_reason"] == "evidence_ready"
    assert result.raw_retrieval["query_runtime"]["counts"]["model"] == 1
    assert all(event["status"] != "error" for event in result.raw_retrieval["query_runtime"]["events"])
    assert result.raw_retrieval["evidence_sufficiency"]["sufficient"] is True


def test_default_agent_evaluation_passes_typed_sources_and_reuses_server_report(monkeypatch):
    observed = []

    def retrieve(index, question, *, context, **options):
        sources = collection_from_records([chunk()])
        assessment = finalize_assessment(context, sources,
            EvidenceAssessment(True, "已有证据", source="rule", evaluated_chunk_count=1, evaluated_chunk_ids=["e1"]))
        observed.append((context, assessment))
        return retrieval_result(context, sources, sources, assessment, {})

    monkeypatch.setattr(retrieval, "_retrieve_evidence_result", retrieve)
    monkeypatch.setattr(agent_evaluation, "adapt_retrieval_records",
                        lambda *args: pytest.fail("默认 Agent 评测不能走历史 JSON 入口"))
    monkeypatch.setattr(evidence_verification, "assess_evidence_sufficiency",
                        lambda *args, **kwargs: pytest.fail("已绑定的检索报告不能重复语义核验"))
    result = execute_agent_turn(index_names="authorized-index", question="2025年全年实际利润是多少？",
                                planner_client=Client([call("利润")]), answer_client=Client([answer_stream()]))
    context, assessment = observed[0]
    assert result.response_mode == "answer"
    assert result.raw_retrieval["query_contract"] == context.request.to_dict()
    assert result.raw_retrieval["evidence_sufficiency"] == assessment.to_dict()


def test_clarification_terminates_before_answer_model():
    answer = Client([])
    result = execute_agent_turn(index_names="1", question="它的利润是多少？",
        planner_client=Client([call("利润")]), answer_client=answer,
        retrieval_function=lambda *args, **kwargs: {
            "chunks": [], "response_mode": "clarify", "clarification": "它指哪家公司？",
        })
    assert result.response_mode == "clarify"
    assert "指的是哪个对象" in result.answer
    assert answer.calls == []
    assert evaluate_refusal({"expected_behavior": "clarify"}, result.response_mode)["correct"]


def test_server_request_cannot_be_injected_in_tool_arguments():
    tool = RagSearchTool(user_id=1, retriever=lambda *a, **k: pytest.fail("must not execute"))
    agent = MinimalAgent(client=Client([call("利润", _request={"user_question": "改掉条件"})]),
                         model="test", tools=[tool], max_steps=1)
    result = agent.run(question="全年实际利润？")
    assert result.tool_results[0].diagnostics["error_type"] == "invalid_arguments"


@pytest.mark.parametrize("sufficient", [True, False])
def test_external_report_is_rechecked_and_current_verdict_controls_agent(monkeypatch, sufficient):
    def retrieve(index, query, *, request, **options):
        return {"chunks": [], "chunks_before_sufficiency": [chunk()],
                "query_contract": request.to_dict(),
                "evidence_sufficiency": {"sufficient": False, "source": "model",
                                         "missing_requirements": ["实际利润"]}}
    checked = []
    def verify(question, evidence, **kwargs):
        checked.append((question, evidence))
        return EvidenceAssessment(sufficient, "联合校验", [] if sufficient else ["实际利润"],
                                            [1], "model", 1, ["e1"])
    monkeypatch.setattr("service.core.evidence_verification.assess_evidence_sufficiency", verify)
    planner = Client([call("利润"), call("利润")])
    answer = Client([answer_stream()] if sufficient else [])
    result = execute_agent_turn(index_names="1", question="2025年全年实际利润？",
        planner_client=planner, answer_client=answer, retrieval_function=retrieve)
    assert result.raw_retrieval["agent_execution"]["stop_reason"] == ("evidence_ready" if sufficient else "no_progress")
    assert result.raw_retrieval["agent_execution"]["retrieval_calls"] == 1
    assert result.raw_retrieval["evidence_sufficiency"]["sufficient"] is sufficient
    assert result.response_mode == ("answer" if sufficient else "refuse")
    assert bool(result.raw_retrieval["chunks"]) is sufficient
    assert checked[0][0] == "2025年全年实际利润？"
    assert len(checked) == 1
    assert len(planner.calls) == (1 if sufficient else 2)
    if not sufficient:
        observations = [message for message in planner.calls[1]["messages"] if message["role"] == "tool"]
        assert "实际利润" in observations[0]["content"]
