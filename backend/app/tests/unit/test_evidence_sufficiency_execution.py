"""充分性评估的规则顺序、重试预算和模型边界行为。"""

import json
from types import SimpleNamespace

import pytest

from service.core.evidence import sufficiency as sufficiency
from service.core.evidence.adapter import collection_from_records
from service.core.runtime.context import QueryRuntime, query_runtime


pytestmark = pytest.mark.unit


class ScriptedCompletions:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def completion(content=None, *, finish_reason="stop"):
    if content is None:
        content = json.dumps({"sufficient": True, "reason": "审批要求明确", "missing_requirements": [],
                              "supporting_chunk_indices": [1]}, ensure_ascii=False)
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=content), finish_reason=finish_reason,
    )])


def client(completions):
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def sources():
    return collection_from_records([
        {"chunk_id": "first", "content_with_weight": "差旅申请需要审批。"},
        {"chunk_id": "second", "content_with_weight": "审批后提交报销材料。"},
    ])


@pytest.fixture(autouse=True)
def model_policy(monkeypatch):
    monkeypatch.setenv("RAG_EVIDENCE_SUFFICIENCY_ENABLED", "true")
    monkeypatch.setenv("RAG_EVIDENCE_SUFFICIENCY_FAIL_OPEN", "false")
    monkeypatch.setenv("RAG_EVIDENCE_SUFFICIENCY_TRANSIENT_RETRY", "true")
    monkeypatch.setenv("RAG_EVIDENCE_SUFFICIENCY_TIMEOUT_SECONDS", "8")


@pytest.mark.parametrize("recovered", [True, False])
def test_transport_retry_preserves_attempts_and_logs_failure(recovered, caplog):
    completions = ScriptedCompletions(TimeoutError("first request failed"),
                                     completion() if recovered else TimeoutError("second request failed"))
    with query_runtime(QueryRuntime(max_model_calls=2)) as runtime:
        result = sufficiency.assess_evidence_sufficiency(
            "差旅如何审批？", sources(), client=client(completions),
        )
    assert len(completions.calls) == runtime.counts["model"] == 2
    assert result.diagnostics["attempts"] == 2
    assert result.sufficient is recovered
    assert result.source == ("model" if recovered else "fallback")
    assert "Evidence verification request failed; retrying once" in caplog.text
    if not recovered:
        assert "Evidence verification failed (attempts=2, kind=service_unavailable)" in caplog.text
    assert "events" not in runtime.snapshot()


@pytest.mark.parametrize("constraint", ["call_budget", "deadline", "retry_disabled"])
def test_request_retry_respects_runtime_and_policy(monkeypatch, constraint, caplog):
    runtime = QueryRuntime(max_model_calls=1 if constraint == "call_budget" else 2,
                           timeout_seconds=1 if constraint == "deadline" else 45)
    if constraint == "retry_disabled":
        monkeypatch.setenv("RAG_EVIDENCE_SUFFICIENCY_TRANSIENT_RETRY", "false")
    completions = ScriptedCompletions(TimeoutError("request failed"), completion())
    with query_runtime(runtime):
        result = sufficiency.assess_evidence_sufficiency(
            "差旅如何审批？", sources(), client=client(completions),
        )
    assert len(completions.calls) == runtime.counts["model"] == 1
    assert result.diagnostics["attempts"] == 1
    assert result.diagnostics["failure_kind"] == "service_unavailable"
    assert "retrying once" not in caplog.text
    assert "Evidence verification failed" in caplog.text


def test_exhausted_budget_does_not_start_provider_request(caplog):
    completions = ScriptedCompletions(completion())
    with query_runtime(QueryRuntime(max_model_calls=0)) as runtime:
        result = sufficiency.assess_evidence_sufficiency(
            "差旅如何审批？", sources(), client=client(completions),
        )
    assert completions.calls == []
    assert result.diagnostics["failure_kind"] == "budget_exhausted"
    assert result.diagnostics["attempts"] == 1
    assert runtime.counts.get("model", 0) == 0
    assert "budget_exhausted" in caplog.text


@pytest.mark.parametrize("response, rejected_phase", [
    (completion("not JSON"), "parse"),
    (completion('{"sufficient":true,"reason":"证据完整","missing_requirements":[],"supporting_chunk_indices":[true]}'),
     "semantic_validation"),
    (completion(finish_reason="length"), "semantic_validation"),
    (SimpleNamespace(choices=[]), "parse"),
])
def test_invalid_responses_fail_without_transport_retry(response, rejected_phase, caplog):
    completions = ScriptedCompletions(response, completion())
    with query_runtime(QueryRuntime(max_model_calls=2)) as runtime:
        result = sufficiency.assess_evidence_sufficiency(
            "差旅如何审批？", sources(), client=client(completions),
        )
    assert len(completions.calls) == 1
    assert result.diagnostics["attempts"] == 1
    assert result.diagnostics["failure_kind"] == "contract_rejected"
    assert "kind=contract_rejected" in caplog.text
    assert "retrying once" not in caplog.text


def test_client_configuration_failure_has_no_request_attempt(monkeypatch):
    def unavailable_client():
        raise RuntimeError("missing provider configuration")

    monkeypatch.setattr(sufficiency, "get_verification_client", unavailable_client)
    with query_runtime() as runtime:
        result = sufficiency.assess_evidence_sufficiency("差旅如何审批？", sources())
    assert result.diagnostics["attempts"] == 0
    assert result.diagnostics["failure_kind"] == "configuration_or_request_error"
    assert runtime.counts.get("model", 0) == 0


@pytest.mark.parametrize("require_semantic, enabled", [(False, "true"), (True, "false"), (True, "true")])
def test_visual_rule_precedes_semantic_execution_policy(monkeypatch, require_semantic, enabled):
    monkeypatch.setenv("RAG_EVIDENCE_SUFFICIENCY_ENABLED", enabled)
    completions = ScriptedCompletions(AssertionError("must not call provider"))
    evidence = collection_from_records([{
        "chunk_id": "figure", "content_with_weight": "图中利润柱状图的题注",
        "evidence_type_kwd": "figure", "visual_status_kwd": "disabled",
    }])
    result = sufficiency.assess_evidence_sufficiency(
        "图中哪一年的柱为负？", evidence, client=client(completions), require_semantic=require_semantic,
    )
    assert result.sufficient is False and result.source == "rule"
    assert result.missing_requirements == ["图表视觉事实或明确描述该事实的正文证据"]
    assert result.evaluated_chunk_ids == ["figure"]
    assert completions.calls == []


def test_question_limit_does_not_start_model(monkeypatch):
    completions = ScriptedCompletions(AssertionError("must not call provider"))
    result = sufficiency.assess_evidence_sufficiency(
        "问题" * sufficiency.MAX_EVIDENCE_QUESTION_CHARS, sources(),
        client=client(completions),
    )
    assert result.diagnostics["failure_kind"] == "input_limit"
    assert result.diagnostics["attempts"] == 0
    assert completions.calls == []
