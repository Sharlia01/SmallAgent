"""Agent 根据合并证据补查，不以一次工具成功作为完成条件。"""
import pytest
from agent import loop
from agent.loop import AgentLoop
from service.core.query.context import prepare_query_context
from service.core.runtime.context import QueryBudgetExceeded
from tests.agent_helpers import ModelClient, SearchTool, initial, task, records, coverage_verifier

pytestmark = pytest.mark.unit


def test_partial_evidence_survives_replanning_and_merges(monkeypatch):
    calls = []
    monkeypatch.setattr(loop, "verify_evidence", coverage_verifier(2, calls))
    tool = SearchTool([records("c1"), records("c2")])
    client = ModelClient([initial(), {"tasks": [task(id="s2", query="员工培训要求 完整制度")]}])
    agent = AgentLoop(client=client, model="test", tools=[tool])
    context = prepare_query_context("员工培训要求")
    result = agent.run(context=context)
    assert result.stop_reason == "evidence_ready"
    assert len(result.evidence) == 2 and len(calls) == 2
    assert len(client.calls) == 2  # 首轮理解规划一次，按缺口补查一次。
    assert [len(evidence) for _, evidence in calls] == [1, 2]


def test_ready_without_evidence_cannot_bypass_verification(monkeypatch):
    calls = []
    monkeypatch.setattr(loop, "verify_evidence", coverage_verifier(calls=calls))
    client = ModelClient([initial(tasks=[]), {"tasks": []}])
    result = AgentLoop(client=client, model="test", tools=[]).run(context=prepare_query_context("员工培训要求"))
    assert not result.assessment.sufficient and result.stop_reason == "no_plan"
    assert len(calls) == 1


def test_empty_result_can_change_query_but_identical_search_stops(monkeypatch):
    monkeypatch.setattr(loop, "verify_evidence", coverage_verifier())
    tool = SearchTool([[]])
    client = ModelClient([initial(), {"tasks": [task(id="s2")]}])
    result = AgentLoop(client=client, model="test", tools=[tool]).run(context=prepare_query_context("员工培训要求"))
    assert result.stop_reason == "no_progress"
    assert len(tool.calls) == 1


def test_later_planner_failure_keeps_existing_evidence(monkeypatch):
    monkeypatch.setattr(loop, "verify_evidence", coverage_verifier(2))
    tool = SearchTool([records()])
    client = ModelClient([initial(), RuntimeError("provider failed")])
    result = AgentLoop(client=client, model="test", tools=[tool]).run(context=prepare_query_context("员工培训要求"))
    assert result.stop_reason == "planning_failed"
    assert len(result.evidence) == 1


def test_initial_failure_does_not_fall_back_to_rag_analysis():
    tool = SearchTool([])
    result = AgentLoop(client=ModelClient([QueryBudgetExceeded("limit")]), model="test", tools=[tool]).run(
        context=prepare_query_context("员工培训要求"))
    assert result.stop_reason == "budget_exhausted" and not tool.calls
    assert not result.assessment.sufficient


def test_verification_failure_stops_instead_of_searching_missing_facts(monkeypatch):
    def failed(*args, **kwargs):
        raise RuntimeError("verification offline")
    monkeypatch.setattr(loop, "verify_evidence", failed)
    client = ModelClient([initial()])
    result = AgentLoop(client=client, model="test", tools=[SearchTool([records()])]).run(context=prepare_query_context("员工培训要求"))
    assert result.stop_reason == "verification_failed"
    assert len(client.calls) == 1 and len(result.evidence) == 1


def test_greeting_requires_no_tool_execution(monkeypatch):
    monkeypatch.setattr(loop, "verify_evidence", coverage_verifier())
    client = ModelClient([initial(goals=[dict(id="g1", text="你好", constraints=[], source_spans=["你好"])], tasks=[], requires_evidence=False)])
    result = AgentLoop(client=client, model="test", tools=[]).run(context=prepare_query_context("你好"))
    assert result.assessment.sufficient and not result.tool_results
