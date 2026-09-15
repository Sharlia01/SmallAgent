from types import SimpleNamespace

import pytest

from service.core import query_intent


class FakeCompletions:
    def __init__(self, content=None, error=None):
        self.content = content
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.content),
                )
            ]
        )


def fake_client(completions):
    return SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
    )


@pytest.mark.unit
def test_precise_document_locator_skips_rewrite_without_calling_model(
    monkeypatch,
):
    monkeypatch.setenv("RAG_QUERY_INTENT_ENABLED", "true")
    completions = FakeCompletions(error=AssertionError("must not be called"))

    decision = query_intent.analyze_query_intent(
        "请列出表5中2025E至2027E的归母净利润。",
        client=fake_client(completions),
    )

    assert decision.intent == "precise_lookup"
    assert decision.need_rewrite is False
    assert decision.need_decompose is False
    assert decision.source == "rule"
    assert completions.calls == []


@pytest.mark.unit
def test_colloquial_finance_query_requests_rewrite_without_model(monkeypatch):
    monkeypatch.setenv("RAG_QUERY_INTENT_ENABLED", "true")
    completions = FakeCompletions(error=AssertionError("must not be called"))

    decision = query_intent.analyze_query_intent(
        "研报觉得国电电力未来三年能赚多少钱，对应估值贵不贵？",
        client=fake_client(completions),
    )

    assert decision.intent == "colloquial_lookup"
    assert decision.need_rewrite is True
    assert decision.need_decompose is False
    assert decision.source == "rule"
    assert completions.calls == []


@pytest.mark.unit
def test_unmatched_query_uses_model_structured_decision(monkeypatch):
    monkeypatch.setenv("RAG_QUERY_INTENT_ENABLED", "true")
    completions = FakeCompletions(
        '{"intent":"complex_lookup","need_rewrite":false,'
        '"need_decompose":true,"reason":"包含两个独立检索目标"}'
    )

    decision = query_intent.analyze_query_intent(
        "比较火电和水电的装机容量，并分析各自利润变化的原因。",
        client=fake_client(completions),
    )

    assert decision.intent == "complex_lookup"
    assert decision.need_rewrite is False
    assert decision.need_decompose is True
    assert decision.reason == "包含两个独立检索目标"
    assert decision.source == "model"
    call = completions.calls[0]
    assert call["temperature"] == 0
    assert call["response_format"] == {"type": "json_object"}
    assert call["extra_body"] == {"enable_thinking": False}


@pytest.mark.unit
def test_ambiguous_query_never_allows_model_to_request_rewrite(monkeypatch):
    monkeypatch.setenv("RAG_QUERY_INTENT_ENABLED", "true")
    completions = FakeCompletions(
        '{"intent":"ambiguous_query","need_rewrite":true,'
        '"need_decompose":false,"reason":"缺少查询主体"}'
    )

    decision = query_intent.analyze_query_intent(
        "请介绍相关情况",
        client=fake_client(completions),
    )

    assert decision.intent == "ambiguous_query"
    assert decision.need_rewrite is False


@pytest.mark.unit
def test_provider_failure_conservatively_skips_rewrite(monkeypatch):
    monkeypatch.setenv("RAG_QUERY_INTENT_ENABLED", "true")
    completions = FakeCompletions(error=TimeoutError("provider timeout"))

    decision = query_intent.analyze_query_intent(
        "国电电力的主营业务有哪些？",
        client=fake_client(completions),
    )

    assert decision.intent == "general_lookup"
    assert decision.need_rewrite is False
    assert decision.source == "fallback"


@pytest.mark.unit
def test_disabled_intent_analysis_preserves_legacy_rewrite_policy(monkeypatch):
    monkeypatch.setenv("RAG_QUERY_INTENT_ENABLED", "false")
    completions = FakeCompletions(error=AssertionError("must not be called"))

    decision = query_intent.analyze_query_intent(
        "国电电力的主营业务有哪些？",
        client=fake_client(completions),
    )

    assert decision.need_rewrite is True
    assert decision.source == "disabled"
    assert completions.calls == []
