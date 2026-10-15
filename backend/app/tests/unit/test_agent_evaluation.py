"""独立问答与端到端评测共用 Agent；外部报告不能替代服务端核验。"""
import pytest
from agent import loop
from service.core.answering.grounded_answer import execute_grounded_turn
from evaluation.agent import execute_agent_turn
from tests.agent_helpers import ModelClient, initial, records, coverage_verifier, task
from tests.unit.test_agent_orchestrator import AnswerClient

pytestmark = pytest.mark.unit


def test_evaluation_preserves_contract_and_returns_full_evidence(monkeypatch):
    monkeypatch.setattr(loop, "verify_evidence", coverage_verifier())
    planner, answer = ModelClient([initial()]), AnswerClient()
    calls = []
    def retrieve(index, query, **options):
        calls.append(query)
        return {"chunks": records(), "evidence_sufficiency": {"sufficient": False}}
    result = execute_agent_turn(index_names="test", question="员工培训要求", planner_client=planner,
                                answer_client=answer, retrieval_function=retrieve)
    assert result.response_mode == "answer"
    assert result.raw_retrieval["query_contract"]["user_question"] == "员工培训要求"
    assert result.raw_retrieval["query_understanding"]["goals"][0]["id"] == "g1"
    assert len(result.documents) == 1 and len(calls) == 1
    assert result.raw_retrieval["evidence_sufficiency"]["sufficient"]


def test_independent_entry_delegates_to_the_same_application_runner(monkeypatch):
    from service.core.answering import turn
    calls = []
    sentinel = object()
    monkeypatch.setattr(turn, "execute_agent_turn", lambda **kwargs: calls.append(kwargs) or sentinel)
    result = execute_grounded_turn(index_names="test", question="员工培训要求")
    assert result is sentinel and calls[0]["question"] == "员工培训要求"


def test_external_positive_verdict_cannot_force_answer(monkeypatch):
    monkeypatch.setattr(loop, "verify_evidence", coverage_verifier(2))
    answer = AnswerClient()
    result = execute_agent_turn(index_names="test", question="员工培训要求",
        planner_client=ModelClient([initial(), {"tasks": []}]), answer_client=answer,
        retrieval_function=lambda *args, **kwargs: {"chunks": records(), "evidence_sufficiency": {"sufficient": True}})
    assert result.response_mode == "refuse" and not answer.calls
    assert len(result.raw_retrieval["chunks_before_sufficiency"]) == 1
