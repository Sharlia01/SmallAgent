"""Agent 根据合并证据补查，不以一次工具成功作为完成条件。"""
import json
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
    checks = []
    monkeypatch.setattr(loop, "verify_evidence", coverage_verifier(calls=checks))
    tool = SearchTool([[]])
    client = ModelClient([initial(), {"tasks": [task(id="s2")]}])
    result = AgentLoop(client=client, model="test", tools=[tool]).run(context=prepare_query_context("员工培训要求"))
    assert result.stop_reason == "no_progress"
    assert len(tool.calls) == 1
    assert result.steps == 1 and len(checks) == 1
    assert result.replan_decisions[-1]["execution_skipped"] == "repeated_direct_plan"


def test_replan_uses_final_opportunity_prompt_and_records_stop_reason(monkeypatch):
    monkeypatch.setattr(loop, "verify_evidence", coverage_verifier(2))
    client = ModelClient([initial(), {"tasks": [], "reason": "没有可行的新取证方向"}])
    result = AgentLoop(client=client, model="test", tools=[SearchTool([records()])]).run(
        context=prepare_query_context("员工培训要求"))
    payload = json.loads(client.calls[1]["messages"][1]["content"])
    assert "progress" not in payload
    assert "这是最后一次补查机会" in client.calls[1]["messages"][0]["content"]
    assert payload["observations"][0]["retrieval_status"] == "succeeded"
    assert payload["observations"][0]["binding_status"] == "not_requested"
    assert result.stop_reason == "no_plan"
    assert result.replan_decisions[0]["reason"] == "没有可行的新取证方向"


@pytest.mark.parametrize("failed,changed_top_k", [(True, False), (False, True)])
def test_retry_or_changed_retrieval_is_not_intercepted(monkeypatch, failed, changed_top_k):
    monkeypatch.setattr(loop, "verify_evidence", coverage_verifier(2))
    tool = SearchTool([RuntimeError("temporary") if failed else records(), records("c2")])
    client = ModelClient([initial(), {"tasks": [task(id="s2", top_k=10 if changed_top_k else 5)]}])
    result = AgentLoop(client=client, model="test", tools=[tool]).run(context=prepare_query_context("员工培训要求"))
    assert len(tool.calls) == 2 and result.steps == 2
    assert "execution_skipped" not in result.replan_decisions[0]


def test_repeated_plan_guard_preserves_binding_and_scope_changes():
    from dataclasses import replace
    from agent.executor import ToolExecutor
    from agent.plans import AgentTask, AgentPlan
    from service.core.query.contract import QueryGoal, QueryUnderstanding

    context = prepare_query_context("员工培训要求")
    context.understanding = QueryUnderstanding((QueryGoal("g1", "员工培训要求"),))
    executor = ToolExecutor([SearchTool([records()])])
    first = AgentTask("s1", query="员工培训要求", goal_ids=("g1",))
    completed, cache = {}, {}
    executor.execute_plan(AgentPlan((first,)), context, completed, cache)
    repeated = replace(first, id="s2")
    assert executor.is_repeated_direct_plan(AgentPlan((repeated,)), context, cache)
    for changed in (replace(repeated, output_slot="company"),
                    replace(repeated, query="", query_template="{company}要求", depends_on=("s1",), required_slots=("company",)),
                    replace(repeated, inherit_document_scope=True),
                    replace(repeated, query="培训周期")):
        assert not executor.is_repeated_direct_plan(AgentPlan((changed,)), context, cache)


def test_later_planner_failure_keeps_existing_evidence(monkeypatch):
    monkeypatch.setattr(loop, "verify_evidence", coverage_verifier(2))
    tool = SearchTool([records()])
    client = ModelClient([initial(), RuntimeError("provider failed"), RuntimeError("fallback failed")])
    result = AgentLoop(client=client, model="test", tools=[tool]).run(context=prepare_query_context("员工培训要求"))
    assert result.stop_reason == "planning_failed"
    assert len(result.evidence) == 1
    assert len(client.calls) == 3


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


@pytest.mark.parametrize(('failure', 'outcome'), [
    (RuntimeError('provider offline'), 'execution_error'),
    (QueryBudgetExceeded('limit'), 'execution_error'),
])
def test_planning_technical_failure_has_explicit_outcome(failure, outcome):
    result = AgentLoop(client=ModelClient([failure]), model='test', tools=[]).run(
        context=prepare_query_context('员工培训要求'))
    assert result.outcome == outcome


def test_invalid_plan_after_finite_repair_is_not_evidence_refusal():
    client = ModelClient([{'understanding': {}}, {'understanding': {}}])
    result = AgentLoop(client=client, model='test', tools=[]).run(context=prepare_query_context('员工培训要求'))
    assert result.outcome == 'invalid_plan' and len(client.calls) == 2


def test_tool_failure_and_empty_search_have_different_outcomes(monkeypatch):
    from agent.schemas import ToolResult
    from agent.executor import TaskExecution
    from domain.evidence import EvidenceCollection
    from domain.assessment import EvidenceAssessment
    from agent.plans import AgentTask
    monkeypatch.setattr(loop, 'verify_evidence', coverage_verifier())
    for failed, expected in ((True, 'execution_error'), (False, 'insufficient_evidence')):
        client = ModelClient([initial()])
        agent = AgentLoop(client=client, model='test', tools=[SearchTool([])], max_steps=1)
        tool_result = ToolResult(tool_name='search_knowledge_base', query='员工培训要求', content='', success=not failed,
                                 evidence=EvidenceCollection(), error='network' if failed else None)
        state = TaskExecution(AgentTask('s1', query='员工培训要求'), 'failed' if failed else 'succeeded', tool_result)
        monkeypatch.setattr(agent.executor, 'execute_plan', lambda *args: [state])
        result = agent.run(context=prepare_query_context('员工培训要求'))
        assert result.outcome == expected


@pytest.mark.parametrize('outcome', ['unsupported_capability', 'calculation_error'])
def test_post_search_capability_failure_stops_redundant_replanning(monkeypatch, outcome):
    from domain.assessment import EvidenceAssessment
    monkeypatch.setattr(loop, 'verify_evidence', lambda *args, **kwargs: EvidenceAssessment(
        sufficient=False, reason='不能可靠计算', diagnostics={'outcome': outcome}))
    tool, client = SearchTool([records()]), ModelClient([initial()])
    result = AgentLoop(client=client, model='test', tools=[tool]).run(context=prepare_query_context('员工培训要求'))
    assert result.outcome == outcome and result.stop_reason == outcome
    assert len(tool.calls) == len(client.calls) == 1


def test_input_limit_stops_replanning_and_explains_incomplete_review(monkeypatch):
    from service.core.answering.policy import render_evidence_refusal
    monkeypatch.setenv("RAG_EVIDENCE_SUFFICIENCY_CONTEXT_TOKENS", "1")
    client = ModelClient([initial()])
    tool = SearchTool([records()])
    result = AgentLoop(client=client, model="test", tools=[tool]).run(
        context=prepare_query_context("员工培训要求"))
    assert result.stop_reason == "verification_failed"
    assert len(client.calls) == len(tool.calls) == 1
    decision = result.assessment.to_dict()
    assert decision["input_limited"] and decision["omitted_chunk_ids"]
    assert decision["failure_kind"] == "input_limit"
    assert "尚未完成全部材料核验" in render_evidence_refusal(decision)


def test_unused_repair_slot_does_not_bypass_repeated_search_guard(monkeypatch):
    monkeypatch.setattr(loop, "verify_evidence", coverage_verifier(2))
    tool = SearchTool([records()])
    client = ModelClient([initial(), {"tasks": [task(id="s2", output_slot="growth_rate_v2")]}])
    agent = AgentLoop(client=client, model="test", tools=[tool])
    def unexpected_extraction(**kwargs):
        pytest.fail("无消费者的变量不应触发绑定提取")
    agent.executor._bridge_extractor = unexpected_extraction
    result = agent.run(context=prepare_query_context("员工培训要求"))
    assert result.stop_reason == "no_progress"
    assert len(tool.calls) == 1
    assert result.replan_decisions[-1]["execution_skipped"] == "repeated_direct_plan"
