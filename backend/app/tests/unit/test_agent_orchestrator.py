"""回答入口只消费循环内已核验的证据，并保留引用与原文的一致性。"""
from types import SimpleNamespace
import pytest
from agent import loop
from agent.orchestrator import AgentOrchestrator
from tests.agent_helpers import ModelClient, SearchTool, initial, goal, records, coverage_verifier

pytestmark = pytest.mark.unit


class AnswerClient:
    def __init__(self, error=None):
        self.calls = []
        self.error = error
        self.chat = SimpleNamespace(completions=self)
    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return [SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="每年培训一次", reasoning_content=None), finish_reason="stop")])]


def orchestrator(planner, answer, tools):
    return AgentOrchestrator(planner_client=planner, planner_model="planner", answer_client=answer, answer_model="answer", tools=tools)


def test_final_answer_reuses_loop_assessment_without_reverification(monkeypatch):
    checked = []
    monkeypatch.setattr(loop, "verify_evidence", coverage_verifier(calls=checked))
    planner, answer = ModelClient([initial()]), AnswerClient()
    result = orchestrator(planner, answer, [SearchTool([records()])]).run(session_id="s1", question="员工培训要求")
    assert result.response_mode == "answer"
    assert result.evidence is result.agent_run.evidence
    assert result.assessment is result.agent_run.assessment
    assert len(checked) == 1 and len(answer.calls) == 1 and len(planner.calls) == 1
    assert result.response_documents
    assert "每年培训一次" in result.answer_messages[-1]["content"]
    assert "events" not in result.query_runtime


def test_missing_evidence_refuses_without_answer_generation(monkeypatch):
    monkeypatch.setattr(loop, "verify_evidence", coverage_verifier())
    answer = AnswerClient()
    result = orchestrator(ModelClient([initial(), {"tasks": []}]), answer, [SearchTool([[]])]).run(session_id="s", question="员工培训要求")
    assert result.response_mode == "refuse" and not answer.calls
    assert result.evidence is result.agent_run.evidence
    assert result.assessment is result.agent_run.assessment
    assert result.evidence_sufficiency == result.agent_run.assessment.to_dict()
    assert not result.response_documents


def test_clarification_returns_before_tools_and_answer(monkeypatch):
    answer = AnswerClient()
    tool = SearchTool([])
    result = orchestrator(ModelClient([initial(goals=[], tasks=[], clarification="需要哪份制度？")]), answer, [tool]).run(
        session_id="s", question="员工培训要求", session_context="每年培训一次")
    assert result.response_mode == "clarify"
    assert not tool.calls and not answer.calls
    assert result.evidence is result.agent_run.evidence
    assert len(result.evidence) == 1
    assert result.assessment is None and result.evidence_sufficiency is None
    assert result.context is None and not result.response_documents


def test_replan_clarification_preserves_evidence_and_assessment(monkeypatch):
    monkeypatch.setattr(loop, "verify_evidence", coverage_verifier(required_count=2))
    planner = ModelClient([initial(), {"tasks": [], "clarification": "还需要哪份制度？"}])
    answer = AnswerClient()
    result = orchestrator(planner, answer, [SearchTool([records()])]).run(
        session_id="s", question="员工培训要求")
    assert result.response_mode == "clarify"
    assert result.direct_answer == "还需要哪份制度？"
    assert result.evidence is result.agent_run.evidence
    assert len(result.evidence) == 1 and len(result.retrieved_content) == 1
    assert result.assessment is result.agent_run.assessment
    assert result.evidence_sufficiency == result.agent_run.assessment.to_dict()
    assert not result.assessment.sufficient
    assert result.context is None and not result.response_documents
    assert not result.answer_messages and not answer.calls


def test_unresolved_pronoun_is_clarified_before_any_tool():
    planner, answer = ModelClient([initial(goals=[], tasks=[], clarification="它指哪家公司？")]), AnswerClient()
    result = orchestrator(planner, answer, []).run(session_id="s", question="它的收入是多少？")
    assert result.response_mode == "clarify" and len(planner.calls) == 1 and not answer.calls


def test_session_document_can_satisfy_request_without_tool(monkeypatch):
    monkeypatch.setattr(loop, "verify_evidence", coverage_verifier())
    tool = SearchTool([])
    result = orchestrator(ModelClient([initial(tasks=[])]), AnswerClient(), [tool]).run(
        session_id="s", question="员工培训要求", session_context="每年培训一次")
    assert result.response_mode == "answer" and not tool.calls
    assert len(result.response_documents) == 1
    assert not result.retrieved_content


def test_answer_transport_error_is_not_retried_as_planning_failure(monkeypatch):
    monkeypatch.setattr(loop, "verify_evidence", coverage_verifier())
    planner, answer = ModelClient([initial()]), AnswerClient(RuntimeError("answer offline"))
    with pytest.raises(RuntimeError, match="answer offline"):
        orchestrator(planner, answer, [SearchTool([records()])]).run(session_id="s", question="员工培训要求")
    assert len(planner.calls) == len(answer.calls) == 1
