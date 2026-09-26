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
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


@pytest.mark.unit
def test_precise_document_locator_is_analyzed_without_model(monkeypatch):
    monkeypatch.setenv("RAG_QUERY_INTENT_ENABLED", "true")
    completions = FakeCompletions(error=AssertionError("must not be called"))

    analysis = query_intent.analyze_query_intent(
        "请列出表5中2025E至2027E的归母净利润。",
        client=fake_client(completions),
    )

    assert analysis.intent == "precise_lookup"
    assert analysis.need_rewrite is False
    assert analysis.target_count == 1
    assert analysis.single_retrieval_likely_covers_all is True
    assert not hasattr(analysis, "need_decompose")
    assert completions.calls == []


@pytest.mark.unit
def test_colloquial_finance_query_requests_rewrite_without_model(monkeypatch):
    monkeypatch.setenv("RAG_QUERY_INTENT_ENABLED", "true")
    completions = FakeCompletions(error=AssertionError("must not be called"))

    analysis = query_intent.analyze_query_intent(
        "上半年卖电均价掉了多少？",
        client=fake_client(completions),
    )

    assert analysis.intent == "colloquial_lookup"
    assert analysis.need_rewrite is True
    assert analysis.target_count == 1
    assert analysis.has_dependency is False
    assert completions.calls == []


@pytest.mark.unit
def test_model_returns_analysis_signals_without_a_plan(monkeypatch):
    monkeypatch.setenv("RAG_QUERY_INTENT_ENABLED", "true")
    completions = FakeCompletions(
        '{"intent":"complex_lookup","need_rewrite":false,'
        '"target_count":2,"targets_independent":true,'
        '"has_dependency":false,'
        '"single_retrieval_likely_covers_all":false,'
        '"reason":"两个指标通常位于不同表格行"}'
    )

    analysis = query_intent.analyze_query_intent(
        "经营活动现金流和企业自由现金流分别是多少？",
        client=fake_client(completions),
    )

    assert analysis.intent == "complex_lookup"
    assert analysis.target_count == 2
    assert analysis.targets_independent is True
    assert analysis.has_dependency is False
    assert analysis.single_retrieval_likely_covers_all is False
    call = completions.calls[0]
    assert call["temperature"] == 0
    assert call["response_format"] == {"type": "json_object"}
    system_prompt = call["messages"][0]["content"]
    assert "不生成子查询" in system_prompt
    assert "不得包含 retrieval_mode、need_decompose、subqueries" in system_prompt


@pytest.mark.unit
def test_planning_fields_in_analysis_response_use_safe_fallback(monkeypatch):
    monkeypatch.setenv("RAG_QUERY_INTENT_ENABLED", "true")
    completions = FakeCompletions(
        '{"intent":"complex_lookup","need_rewrite":false,'
        '"target_count":2,"targets_independent":true,'
        '"has_dependency":false,'
        '"single_retrieval_likely_covers_all":false,'
        '"retrieval_mode":"parallel","reason":"两个目标"}'
    )

    analysis = query_intent.analyze_query_intent(
        "指标甲和指标乙分别是多少？",
        client=fake_client(completions),
    )

    assert analysis.source == "fallback"
    assert analysis.target_count == 1
    assert analysis.single_retrieval_likely_covers_all is True


@pytest.mark.unit
def test_ambiguous_query_never_allows_model_to_request_rewrite(monkeypatch):
    monkeypatch.setenv("RAG_QUERY_INTENT_ENABLED", "true")
    completions = FakeCompletions(
        '{"intent":"ambiguous_query","need_rewrite":true,'
        '"target_count":1,"targets_independent":false,'
        '"has_dependency":false,'
        '"single_retrieval_likely_covers_all":true,'
        '"reason":"缺少查询主体"}'
    )

    analysis = query_intent.analyze_query_intent(
        "请介绍相关情况",
        client=fake_client(completions),
    )

    assert analysis.intent == "ambiguous_query"
    assert analysis.need_rewrite is False


@pytest.mark.unit
def test_rating_meaning_query_exposes_dependency_without_building_plan(
    monkeypatch,
):
    monkeypatch.setenv("RAG_QUERY_INTENT_ENABLED", "true")
    completions = FakeCompletions(error=AssertionError("must not be called"))

    analysis = query_intent.analyze_query_intent(
        "这份研报对国电电力到底是看多还是看空？给了什么评级？",
        client=fake_client(completions),
    )

    assert analysis.intent == "complex_lookup"
    assert analysis.target_count == 2
    assert analysis.targets_independent is False
    assert analysis.has_dependency is True
    assert completions.calls == []


@pytest.mark.unit
def test_provider_failure_conservatively_uses_one_target(monkeypatch):
    monkeypatch.setenv("RAG_QUERY_INTENT_ENABLED", "true")
    completions = FakeCompletions(error=TimeoutError("provider timeout"))

    analysis = query_intent.analyze_query_intent(
        "国电电力的主营业务有哪些？",
        client=fake_client(completions),
    )

    assert analysis.intent == "general_lookup"
    assert analysis.need_rewrite is False
    assert analysis.target_count == 1
    assert analysis.source == "fallback"


@pytest.mark.unit
def test_disabled_analysis_preserves_legacy_rewrite_policy(monkeypatch):
    monkeypatch.setenv("RAG_QUERY_INTENT_ENABLED", "false")
    completions = FakeCompletions(error=AssertionError("must not be called"))

    analysis = query_intent.analyze_query_intent(
        "国电电力的主营业务有哪些？",
        client=fake_client(completions),
    )

    assert analysis.need_rewrite is True
    assert analysis.source == "disabled"
    assert completions.calls == []
