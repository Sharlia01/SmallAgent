"""指代消解与规划的 Flash 首轮、Max 兜底；不访问模型服务。"""
import json
from types import SimpleNamespace

import pytest

from agent.planner import AgentPlanner
from service.core.query.context import prepare_query_context
from service.core.query.contract import QueryRequest, resolve_request
from service.core.query.model_validation import validated_completion
from service.core.query.validation import QueryValidationError
from service.core.runtime.clients import agent_model, agent_fallback_model, context_model, context_fallback_model
from service.core.runtime.context import QueryBudgetExceeded, QueryRuntime, query_runtime
from tests.agent_helpers import ModelClient, initial, task

pytestmark = pytest.mark.unit
PRIMARY = "qwen3.7-flash"
FALLBACK = "qwen3.8-max"


@pytest.fixture(autouse=True)
def model_settings(monkeypatch):
    for key, value in {"AGENT_MODEL": PRIMARY, "AGENT_FALLBACK_MODEL": FALLBACK,
                       "AGENT_CONTEXT_MODEL": PRIMARY, "AGENT_CONTEXT_FALLBACK_MODEL": FALLBACK,
                       "AGENT_CONTEXT_TIMEOUT_SECONDS": "6", "AGENT_CONTEXT_FALLBACK_TIMEOUT_SECONDS": "20"}.items():
        monkeypatch.setenv(key, value)


def planner(client):
    return AgentPlanner(client=client, model=agent_model(),
                        tool_definitions=[{"function": {"name": "search_knowledge_base"}}])


def context_request():
    return QueryRequest.from_user("它的项目进展？", [{"role": "user", "content": "国电电力"}])


def bindings(value="国电电力"):
    return {"bindings": [{"mention": "它", "value": value, "source_turn": 0, "source_quote": "国电电力"}]}


def test_model_defaults_and_independent_overrides(monkeypatch):
    for key in ("AGENT_MODEL", "AGENT_FALLBACK_MODEL", "AGENT_CONTEXT_MODEL", "AGENT_CONTEXT_FALLBACK_MODEL"):
        monkeypatch.delenv(key)
    assert agent_model() == context_model() == PRIMARY
    assert agent_fallback_model() == context_fallback_model() == FALLBACK
    monkeypatch.setenv("AGENT_MODEL", "custom-primary")
    monkeypatch.setenv("AGENT_FALLBACK_MODEL", "custom-fallback")
    assert context_model() == "custom-primary"
    assert context_fallback_model() == "custom-fallback"
    monkeypatch.setenv("AGENT_CONTEXT_MODEL", "context-primary")
    monkeypatch.setenv("AGENT_CONTEXT_FALLBACK_MODEL", "context-fallback")
    assert context_model() == "context-primary" and context_fallback_model() == "context-fallback"
    assert agent_model() == "custom-primary" and agent_fallback_model() == "custom-fallback"


def test_success_uses_flash_only():
    context_client = ModelClient([bindings()])
    assert resolve_request(context_request(), client=context_client).bindings
    plan_client = ModelClient([initial()])
    assert planner(plan_client).analyze_and_plan(prepare_query_context("员工培训要求")).plan.tasks
    assert [call["model"] for call in context_client.calls] == [PRIMARY]
    assert [call["model"] for call in plan_client.calls] == [PRIMARY]


@pytest.mark.parametrize("failure", ["not json", bindings("华能国际"), {"bindings": [1]}, TimeoutError("timeout")])
def test_context_failure_switches_to_max_with_original_provenance(failure):
    request = context_request()
    client = ModelClient([failure, bindings()])
    result = resolve_request(request, client=client)
    assert result.resolved_question == "国电电力的项目进展？"
    assert result.user_question == request.user_question and result.bindings[0].source_quote == "国电电力"
    assert [call["model"] for call in client.calls] == [PRIMARY, FALLBACK]
    assert [call["timeout"] for call in client.calls] == [6, 20]
    assert client.calls[0]["messages"][:2] == client.calls[1]["messages"][:2]


@pytest.mark.parametrize("failure", ["not json", initial(tasks=[task(tool="missing")]), TimeoutError("timeout")])
def test_initial_plan_failure_switches_to_max(failure):
    client = ModelClient([failure, initial()])
    context = prepare_query_context("员工培训要求")
    decision = planner(client).analyze_and_plan(context)
    assert context.understanding is decision.understanding
    assert [call["model"] for call in client.calls] == [PRIMARY, FALLBACK]
    assert client.calls[0]["messages"][:2] == client.calls[1]["messages"][:2]


def test_replanning_uses_same_fallback_without_replacing_understanding():
    client = ModelClient([initial(), {"tasks": [task(id="s2", tool="missing")]},
                          {"tasks": [task(id="s2", query="培训频次")]}])
    context = prepare_query_context("员工培训要求")
    instance = planner(client)
    first = instance.analyze_and_plan(context)
    result = instance.replan(context, previous=first.plan.tasks, observations=[], assessment={})
    assert result.tasks[0].id == "s2" and context.understanding is first.understanding
    assert [call["model"] for call in client.calls] == [PRIMARY, PRIMARY, FALLBACK]


def test_valid_clarification_does_not_upgrade():
    client = ModelClient([{"bindings": []}])
    assert resolve_request(context_request(), client=client).clarification
    assert len(client.calls) == 1
    client = ModelClient([initial(goals=[], tasks=[], clarification="请明确对象。")])
    assert planner(client).analyze_and_plan(prepare_query_context("查询资料")).understanding.clarification
    assert len(client.calls) == 1


def test_both_attempts_fail_without_committing_invalid_understanding():
    client = ModelClient([bindings("华能国际"), bindings("华能国际")])
    request = context_request()
    result = resolve_request(request, client=client)
    assert result.clarification and not result.bindings and result.resolved_question == request.user_question
    assert [call["model"] for call in client.calls] == [PRIMARY, FALLBACK]
    invalid = initial(tasks=[task(tool="missing")])
    client = ModelClient([invalid, invalid])
    context = prepare_query_context("员工培训要求")
    with pytest.raises(QueryValidationError):
        planner(client).analyze_and_plan(context)
    assert context.understanding is None
    assert [call["model"] for call in client.calls] == [PRIMARY, FALLBACK]


def test_second_request_failure_is_not_retried_again():
    failure = TimeoutError("second timeout")
    client = ModelClient([TimeoutError("first timeout"), failure])
    with pytest.raises(TimeoutError) as caught:
        planner(client).analyze_and_plan(prepare_query_context("员工培训要求"))
    assert caught.value is failure and len(client.calls) == 2


def test_budget_exhaustion_does_not_send_fallback_request():
    client = ModelClient(["not json", initial()])
    with query_runtime(QueryRuntime(max_model_calls=1)), pytest.raises(QueryBudgetExceeded):
        planner(client).analyze_and_plan(prepare_query_context("员工培训要求"))
    assert len(client.calls) == 1


def test_fallback_shares_budget_and_respects_remaining_timeout(monkeypatch):
    runtime = QueryRuntime(max_model_calls=2)
    monkeypatch.setattr(runtime, "remaining", lambda: 3)
    client = ModelClient([TimeoutError("timeout"), bindings()])
    with query_runtime(runtime):
        assert resolve_request(context_request(), client=client).bindings
    assert runtime.counts["model"] == 2
    assert [call["timeout"] for call in client.calls] == [3, 3]


def test_truncated_response_uses_max_and_larger_output_budget():
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'),
            finish_reason="length" if len(calls) == 1 else "stop")])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    result = validated_completion(client, "test", parse=json.loads, validate=lambda value: value,
                                  model=PRIMARY, fallback_model=FALLBACK, max_tokens=500,
                                  messages=[{"role": "system", "content": "Return JSON"}])
    assert result == {"ok": True}
    assert [call["model"] for call in calls] == [PRIMARY, FALLBACK]
    assert calls[1]["max_tokens"] > calls[0]["max_tokens"]
