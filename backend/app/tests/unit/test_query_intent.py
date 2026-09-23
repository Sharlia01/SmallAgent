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
        "上半年卖电均价掉了多少？",
        client=fake_client(completions),
    )

    assert decision.intent == "colloquial_lookup"
    assert decision.need_rewrite is True
    assert decision.need_decompose is False
    assert decision.source == "rule"
    assert completions.calls == []


@pytest.mark.unit
def test_multi_facet_colloquial_query_can_build_three_parallel_steps(
    monkeypatch,
):
    monkeypatch.setenv("RAG_QUERY_INTENT_ENABLED", "true")
    completions = FakeCompletions(
        '{"intent":"complex_lookup","need_rewrite":true,'
        '"need_decompose":true,"retrieval_mode":"parallel",'
        '"reason":"盈利、每股收益和市盈率可分别检索",'
        '"subqueries":['
        '{"id":"profit","query":"国电电力2025至2027年归母净利润预测是多少？",'
        '"query_template":"","depends_on":[],"output_slot":null,'
        '"required_slots":[],"inherit_document_scope":false},'
        '{"id":"eps","query":"国电电力2025至2027年EPS预测是多少？",'
        '"query_template":"","depends_on":[],"output_slot":null,'
        '"required_slots":[],"inherit_document_scope":false},'
        '{"id":"pe","query":"国电电力2025至2027年PE估值是多少？",'
        '"query_template":"","depends_on":[],"output_slot":null,'
        '"required_slots":[],"inherit_document_scope":false}'
        ']}'
    )

    decision = query_intent.analyze_query_intent(
        "研报觉得国电电力未来三年能赚多少钱，对应估值贵不贵？",
        client=fake_client(completions),
    )

    assert decision.need_rewrite is True
    assert decision.retrieval_mode == "parallel"
    assert [step.id for step in decision.subqueries] == [
        "profit",
        "eps",
        "pe",
    ]


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
    assert decision.retrieval_mode == "parallel"
    assert decision.subqueries == []
    assert decision.reason == "包含两个独立检索目标"
    assert decision.source == "model"
    call = completions.calls[0]
    assert call["temperature"] == 0
    assert call["response_format"] == {"type": "json_object"}
    assert call["extra_body"] == {"enable_thinking": False}


@pytest.mark.unit
def test_rating_meaning_query_builds_sequential_plan_without_model(
    monkeypatch,
):
    monkeypatch.setenv("RAG_QUERY_INTENT_ENABLED", "true")
    completions = FakeCompletions(error=AssertionError("must not be called"))
    question = "这份研报对国电电力到底是看多还是看空？给了什么评级？"

    decision = query_intent.analyze_query_intent(
        question,
        client=fake_client(completions),
    )

    assert decision.intent == "complex_lookup"
    assert decision.retrieval_mode == "sequential"
    assert decision.need_decompose is True
    assert len(decision.subqueries) == 2
    first, second = decision.subqueries
    assert first.query == question
    assert first.output_slot == "rating"
    assert second.depends_on == [first.id]
    assert second.required_slots == ["rating"]
    assert "{rating}" in second.query_template
    assert second.inherit_document_scope is True
    assert completions.calls == []


@pytest.mark.unit
def test_model_sequential_plan_is_parsed_and_validated(monkeypatch):
    monkeypatch.setenv("RAG_QUERY_INTENT_ENABLED", "true")
    completions = FakeCompletions(
        '{"intent":"complex_lookup","need_rewrite":false,'
        '"need_decompose":true,"retrieval_mode":"sequential",'
        '"reason":"第二步依赖第一步实体",'
        '"subqueries":['
        '{"id":"policy_lookup","query":"公司采用了什么分红政策？",'
        '"query_template":"","depends_on":[],"output_slot":"policy",'
        '"required_slots":[],"inherit_document_scope":false},'
        '{"id":"policy_effect","query":"",'
        '"query_template":"{policy}对股息率有什么影响？",'
        '"depends_on":["policy_lookup"],"output_slot":null,'
        '"required_slots":["policy"],"inherit_document_scope":true}'
        ']}'
    )

    decision = query_intent.analyze_query_intent(
        "公司采用了什么分红政策，这项政策对股息率有什么影响？",
        client=fake_client(completions),
    )

    assert decision.retrieval_mode == "sequential"
    assert decision.subqueries[0].output_slot == "policy"
    assert decision.subqueries[1].required_slots == ["policy"]


@pytest.mark.unit
def test_multiple_figure_locators_are_planned_as_parallel_queries(monkeypatch):
    monkeypatch.setenv("RAG_QUERY_INTENT_ENABLED", "true")
    completions = FakeCompletions(
        '{"intent":"complex_lookup","need_rewrite":false,'
        '"need_decompose":true,"retrieval_mode":"parallel",'
        '"reason":"需要分别查询图2和图3",'
        '"subqueries":['
        '{"id":"figure_3","query":"图3中2021年归母净利润是否为负？",'
        '"query_template":"","depends_on":[],"output_slot":null,'
        '"required_slots":[],"inherit_document_scope":false},'
        '{"id":"figure_2","query":"图2中2025年显示了哪些季度的营业收入柱？",'
        '"query_template":"","depends_on":[],"output_slot":null,'
        '"required_slots":[],"inherit_document_scope":false}'
        ']}'
    )

    decision = query_intent.analyze_query_intent(
        "不要混淆图2和图3：哪张图显示2021年归母净利润为负，"
        "哪张图显示2025年只有第一、第二季度营业收入柱？",
        client=fake_client(completions),
    )

    assert decision.retrieval_mode == "parallel"
    assert [step.id for step in decision.subqueries] == [
        "figure_3",
        "figure_2",
    ]
    assert len(completions.calls) == 1


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
