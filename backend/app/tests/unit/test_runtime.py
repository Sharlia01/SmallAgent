"""请求预算、缓存及跨模块执行上下文的行为约束。"""

import json
from types import SimpleNamespace

import pytest

from service.core.query.model_validation import validated_completion
from service.core.runtime.context import (
    QueryRuntime, QueryBudgetExceeded, current_runtime, query_runtime,
)
from service.core.runtime.execution import run_independent
from service.core.runtime.model_calls import model_completion

pytestmark = pytest.mark.unit


def test_budget_is_shared_by_parallel_workers():
    budget = QueryRuntime(max_retrievals=1)
    def reserve():
        assert current_runtime() is budget
        budget.reserve("retrieval")
        return "ok"
    with query_runtime(budget):
        results = run_independent({"a": reserve, "b": reserve})
    assert sum(value == "ok" for value in results.values()) == 1
    assert sum(isinstance(value, QueryBudgetExceeded) for value in results.values()) == 1
    assert current_runtime() is None


def test_model_budget_prevents_another_provider_call():
    calls = []
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs: calls.append(kwargs) or SimpleNamespace(usage=None),
    )))
    with query_runtime(QueryRuntime(max_model_calls=1)):
        model_completion(client, "analysis", model="test", timeout=120)
        with pytest.raises(QueryBudgetExceeded):
            model_completion(client, "rewrite", model="test")
    assert len(calls) == 1
    assert 0 < calls[0]["timeout"] <= 45


def test_deadline_does_not_start_more_work():
    budget = QueryRuntime(timeout_seconds=0)
    with query_runtime(budget):
        values = run_independent({"a": lambda: pytest.fail("must not start")})
    assert isinstance(values["a"], QueryBudgetExceeded)


def test_request_cache_is_not_shared_with_another_request():
    with query_runtime() as first:
        first.cache[("tenant-a", "query")] = "private evidence"
    with query_runtime() as second:
        assert not second.cache


def test_model_repair_in_worker_shares_request_budget_and_logs_failure(caplog):
    """模型调用、查询修复和并发执行拆开后，仍须使用同一份请求状态。"""
    budget = QueryRuntime(max_model_calls=2)
    responses = iter(['{"value": "invalid"}', '{"value": 42}'])

    def create(**kwargs):
        assert current_runtime() is budget
        budget.cache[("worker",)] = "shared"
        return SimpleNamespace(usage=None, choices=[SimpleNamespace(
            finish_reason="stop", message=SimpleNamespace(content=next(responses)),
        )])

    def validate(value):
        if type(value["value"]) is not int:
            raise ValueError("value must be an integer")
        return value["value"]

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    with query_runtime(budget):
        results = run_independent({"worker": lambda: validated_completion(
            client, "query_analysis", parse=json.loads, validate=validate,
            messages=[{"role": "user", "content": "返回一个整数"}], model="test",
        )})
        snapshot = budget.snapshot()
        assert budget.cache[("worker",)] == "shared"

    assert results == {"worker": 42}
    assert snapshot["counts"] == {"model": 2}
    assert "query_analysis validation failed (attempt 1/2)" in caplog.text
    assert "value must be an integer" in caplog.text
    assert "events" not in snapshot
    assert current_runtime() is None
    with query_runtime() as next_request:
        assert not next_request.counts and not next_request.cache


def test_repeated_invalid_model_output_logs_both_attempts_and_raises(caplog):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content="not JSON"), finish_reason="stop",
        )])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    with query_runtime() as runtime, pytest.raises(json.JSONDecodeError):
        validated_completion(client, "query_understanding", parse=json.loads, validate=lambda value: value,
                             messages=[{"role": "user", "content": "分析问题"}], model="test")

    assert runtime.counts["model"] == len(calls) == 2
    assert "attempt 1/2" in caplog.text and "attempt 2/2" in caplog.text
    assert "invalid_json" in calls[1]["messages"][-1]["content"]


def test_transport_failure_logs_stage_and_preserves_original_exception(caplog):
    failure = TimeoutError("service timed out")
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        raise failure

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    with pytest.raises(TimeoutError) as raised:
        validated_completion(client, "query_understanding", parse=json.loads, validate=lambda value: value,
                             messages=[{"role": "user", "content": "分析问题"}], model="test")

    assert raised.value is failure and len(calls) == 1
    assert "query_understanding model request failed: TimeoutError: service timed out" in caplog.text
