from types import SimpleNamespace

import pytest

from service.core.query_intent import QueryAnalysis
from service.core import query_planning


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
            choices=[SimpleNamespace(
                message=SimpleNamespace(content=self.content),
            )]
        )


def fake_client(completions):
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def analysis(**overrides):
    values = {
        "intent": "complex_lookup",
        "need_rewrite": False,
        "target_count": 2,
        "targets_independent": True,
        "has_dependency": False,
        "single_retrieval_likely_covers_all": False,
        "reason": "测试分析",
        "source": "model",
    }
    values.update(overrides)
    return QueryAnalysis(**values)


@pytest.mark.unit
def test_router_uses_single_when_one_result_set_likely_covers_all():
    route = query_planning.select_retrieval_route(
        analysis(single_retrieval_likely_covers_all=True)
    )

    assert route.mode == "single"


@pytest.mark.unit
def test_router_uses_parallel_only_for_independent_uncovered_targets():
    route = query_planning.select_retrieval_route(analysis())

    assert route.mode == "parallel"


@pytest.mark.unit
def test_dependency_takes_precedence_over_parallel_signals():
    route = query_planning.select_retrieval_route(
        analysis(has_dependency=True)
    )

    assert route.mode == "sequential"


@pytest.mark.unit
def test_single_route_does_not_call_query_planner():
    completions = FakeCompletions(error=AssertionError("must not be called"))
    route = query_planning.RetrievalRoute(
        mode="single",
        reason="同一行",
        source="rule",
    )

    plan = query_planning.build_retrieval_plan(
        "2026E和2027E财务费用分别是多少？",
        route,
        client=fake_client(completions),
    )

    assert plan.mode == "single"
    assert plan.subqueries == []
    assert completions.calls == []


@pytest.mark.unit
def test_parallel_planner_preserves_relative_time():
    completions = FakeCompletions(
        '{"subqueries":['
        '{"id":"earnings_forecast",'
        '"query":"国电电力未来三年归母净利润和每股收益预测是多少？",'
        '"query_template":"","depends_on":[],"output_slot":null,'
        '"required_slots":[],"inherit_document_scope":false},'
        '{"id":"valuation",'
        '"query":"国电电力未来三年市盈率和PE估值是多少？",'
        '"query_template":"","depends_on":[],"output_slot":null,'
        '"required_slots":[],"inherit_document_scope":false}'
        ']}'
    )
    route = query_planning.RetrievalRoute(
        mode="parallel",
        reason="两个独立目标",
        source="analysis",
    )

    plan = query_planning.build_retrieval_plan(
        "研报觉得国电电力未来三年能赚多少钱，对应估值贵不贵？",
        route,
        client=fake_client(completions),
    )

    assert plan.mode == "parallel"
    assert plan.source == "model"
    assert [step.id for step in plan.subqueries] == [
        "earnings_forecast",
        "valuation",
    ]
    assert all("未来三年" in step.query for step in plan.subqueries)
    assert "已经确定该问题必须使用 parallel" in (
        completions.calls[0]["messages"][0]["content"]
    )


@pytest.mark.unit
def test_parallel_plan_that_invents_years_falls_back_to_single():
    completions = FakeCompletions(
        '{"subqueries":['
        '{"id":"earnings","query":"国电电力2024至2026年利润预测？",'
        '"query_template":"","depends_on":[],"output_slot":null,'
        '"required_slots":[],"inherit_document_scope":false},'
        '{"id":"valuation","query":"国电电力2024至2026年PE估值？",'
        '"query_template":"","depends_on":[],"output_slot":null,'
        '"required_slots":[],"inherit_document_scope":false}'
        ']}'
    )
    route = query_planning.RetrievalRoute(
        mode="parallel",
        reason="两个独立目标",
        source="analysis",
    )

    plan = query_planning.build_retrieval_plan(
        "国电电力未来三年利润和估值如何？",
        route,
        client=fake_client(completions),
    )

    assert plan.mode == "single"
    assert plan.source == "fallback"
    assert plan.subqueries == []


@pytest.mark.unit
def test_rating_dependency_builds_rule_plan_without_model():
    completions = FakeCompletions(error=AssertionError("must not be called"))
    question = "这份研报对国电电力是看多还是看空？给了什么评级？"
    route = query_planning.RetrievalRoute(
        mode="sequential",
        reason="评级含义依赖评级名称",
        source="analysis",
    )

    plan = query_planning.build_retrieval_plan(
        question,
        route,
        client=fake_client(completions),
    )

    assert plan.mode == "sequential"
    assert plan.source == "rule"
    first, second = plan.subqueries
    assert first.query == question
    assert first.output_slot == "rating"
    assert second.depends_on == [first.id]
    assert second.required_slots == ["rating"]
    assert second.inherit_document_scope is True
    assert completions.calls == []


@pytest.mark.unit
def test_model_sequential_plan_is_parsed_and_validated():
    completions = FakeCompletions(
        '{"subqueries":['
        '{"id":"policy_lookup","query":"公司采用了什么分红政策？",'
        '"query_template":"","depends_on":[],"output_slot":"policy",'
        '"required_slots":[],"inherit_document_scope":false},'
        '{"id":"policy_effect","query":"",'
        '"query_template":"{policy}对股息率有什么影响？",'
        '"depends_on":["policy_lookup"],"output_slot":null,'
        '"required_slots":["policy"],"inherit_document_scope":true}'
        ']}'
    )
    route = query_planning.RetrievalRoute(
        mode="sequential",
        reason="后一步依赖政策名称",
        source="analysis",
    )

    plan = query_planning.build_retrieval_plan(
        "公司采用了什么分红政策，这项政策对股息率有什么影响？",
        route,
        client=fake_client(completions),
    )

    assert plan.mode == "sequential"
    assert plan.subqueries[0].output_slot == "policy"
    assert plan.subqueries[1].required_slots == ["policy"]
