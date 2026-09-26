import json
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from service.core.query_contract import QueryRequest, parse_goals, resolve_request
from service.core.query_expansion import validate_query_preservation, expand_query
from service.core.query_intent import _rule_based_analysis, _parse_query_analysis
from service.core.query_planning import (
    QuerySubqueryPlan, _validate_parallel_subquery_drift, _validate_goal_coverage,
    validate_dependency_graph,
)
from service.core.sequential_retrieval import extract_bridge_value


pytestmark = pytest.mark.unit


def client_for(payload):
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs: SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=json.dumps(payload, ensure_ascii=False)),
        )]),
    )))


@pytest.mark.parametrize(("original", "candidate"), [
    ("张三在北京负责什么？", "李四在上海负责什么？"),
    ("利润大于10亿元", "利润小于10亿元"),
    ("毛利率为-3%", "毛利率为3%"),
    ("毛利率为-3%", "毛利率为+3%"),
    ("2025年收入比2024年高多少？", "2024年收入比2025年高多少？"),
    ("2025年扣非归母净利润", "2025年归母净利润"),
    ("实际新增装机容量", "预测累计装机容量"),
    ("不包括2025年预测值的净利润", "2025年净利润"),
    ("2025年基本每股收益", "2025年每股收益"),
    ("企业自由现金流20808百万元", "企业自由现金流20808万元"),
])
def test_semantic_drift_is_rejected(original, candidate):
    assert not validate_query_preservation(original, candidate).valid


@pytest.mark.parametrize(("original", "candidate"), [
    ("今年归母净利润", "本年度归属于母公司股东的净利润"),
    ("2025年每股收益", "2025年EPS"),
    ("毛利率为-3%", "毛利率为－3％"),
])
def test_registered_equivalence_preserves_semantics(original, candidate):
    assert validate_query_preservation(original, candidate).valid


def test_formal_query_expansion_is_independent_of_rewriting():
    question = "国电电力2025年归母净利润"
    result = expand_query(question, question, allow_model=False)
    assert result.expansion_applied
    assert result.source == "rule"
    assert question in result.effective_query
    assert result.expansion_validation.valid


@pytest.mark.parametrize("question", [
    "请解释表5中的毛利率，并查询另一份报告的现金流。",
    "国电电力2025年能赚多少？华能国际2024年装机容量是多少？",
    "解释国电电力评级的含义，同时查询2025年营业收入。",
])
def test_local_cues_do_not_override_multiple_goals(question):
    assert _rule_based_analysis(question) is None


def test_known_rating_does_not_trigger_dependency_rule():
    analysis = _rule_based_analysis("‘买入’评级是什么意思？")
    assert analysis is None or not analysis.has_dependency


def test_user_question_is_immutable_and_resolution_has_provenance():
    request = QueryRequest.from_user("它今年的现金流呢？", [
        {"role": "user", "content": "请介绍国电电力"},
    ])
    resolved = resolve_request(request, client=client_for({"bindings": [{
        "mention": "它", "value": "国电电力", "source_turn": 0, "source_quote": "国电电力",
    }]}))
    assert resolved.user_question == "它今年的现金流呢？"
    assert resolved.resolved_question == "国电电力今年的现金流呢？"
    assert resolved.bindings[0].source_turn == 0
    assert resolved.clarification is None
    with pytest.raises(FrozenInstanceError):
        resolved.user_question = "改掉用户问题"
    assert "history" not in resolved.to_dict()


@pytest.mark.parametrize("binding", [
    {"mention": "它", "value": "华能国际", "source_turn": 0, "source_quote": "国电电力"},
    {"mention": "它", "value": "国电电力", "source_turn": True, "source_quote": "国电电力"},
    {"mention": "今年", "value": "2028", "source_turn": 0, "source_quote": "国电电力"},
])
def test_ungrounded_context_requires_clarification(binding):
    request = QueryRequest.from_user("它今年的现金流呢？", [{"role": "user", "content": "国电电力"}])
    result = resolve_request(request, client=client_for({"bindings": [binding]}))
    assert result.clarification
    assert result.resolved_question == request.user_question


def test_missing_context_never_guesses_an_entity():
    assert resolve_request(QueryRequest.from_user("它的利润呢？")).clarification


def test_assistant_only_entity_is_not_promoted_to_user_fact():
    request = QueryRequest.from_user("它的利润呢？", [{"role": "assistant", "content": "虚构公司"}])
    result = resolve_request(request, client=client_for({"bindings": [{
        "mention": "它", "value": "虚构公司", "source_turn": 0, "source_quote": "虚构公司",
    }]}))
    assert result.clarification


def test_goal_constraints_must_come_from_user_text():
    with pytest.raises(ValueError, match="goal_source_not_grounded"):
        parse_goals([{"id": "profit", "text": "利润", "constraints": ["2029年"]}], "2025年利润")


def test_goal_graph_rejects_cycles():
    with pytest.raises(ValueError, match="cyclic"):
        parse_goals([
            {"id": "a", "text": "甲", "depends_on": ["b"]},
            {"id": "b", "text": "乙", "depends_on": ["a"]},
        ], "甲和乙")


def test_parallel_plan_cannot_drop_shared_entity_or_year():
    with pytest.raises(ValueError, match="shared constraint"):
        _validate_parallel_subquery_drift("国电电力2025年营业收入和净利润分别是多少？", [
            QuerySubqueryPlan("revenue", query="营业收入是多少？"),
            QuerySubqueryPlan("profit", query="净利润是多少？"),
        ])


def test_sequential_plan_cannot_invent_a_company_or_year():
    with pytest.raises(ValueError, match="invented"):
        _validate_goal_coverage("国电电力采用了什么政策，该政策有何影响？", [
            QuerySubqueryPlan("first", query="华能国际2029年采用什么政策？", output_slot="policy"),
            QuerySubqueryPlan("next", query_template="{policy}的影响？", depends_on=["first"], required_slots=["policy"]),
        ], ())


def test_plan_preserves_goal_local_entity_year_binding():
    question = "国电电力2024年利润和华能国际2025年利润"
    goals = parse_goals([
        {"id": "a", "text": "国电电力2024年利润", "constraints": ["国电电力", "2024年"]},
        {"id": "b", "text": "华能国际2025年利润", "constraints": ["华能国际", "2025年"]},
    ], question)
    with pytest.raises(ValueError, match="goal-local"):
        _validate_goal_coverage(question, [
            QuerySubqueryPlan("qa", query="国电电力2025年利润", goal_ids=["a"]),
            QuerySubqueryPlan("qb", query="华能国际2024年利润", goal_ids=["b"]),
        ], goals)


@pytest.mark.parametrize("steps", [
    [QuerySubqueryPlan("a", query="问题", depends_on=["missing"])],
    [QuerySubqueryPlan("a", query="问题"), QuerySubqueryPlan("b", query="问题")],
    [QuerySubqueryPlan("a", query="主体", output_slot="x"),
     QuerySubqueryPlan("b", query_template="{x.__class__}", depends_on=["a"], required_slots=["x"])],
    [QuerySubqueryPlan("a", query="主体", output_slot="x"),
     QuerySubqueryPlan("b", query_template="{x}", depends_on=[], required_slots=["x"])],
])
def test_graph_is_validated_before_execution(steps):
    with pytest.raises(ValueError):
        validate_dependency_graph(steps)


def test_bridge_binds_rating_to_requested_company():
    result = extract_bridge_value(
        slot="rating", original_question="国电电力的评级？", first_query="国电电力的评级？",
        chunks=[{"chunk_id": "ratings", "doc_id": "report", "content_with_weight":
                 "本报告给予华能国际‘买入’评级，国电电力维持‘中性’评级。"}],
        client=client_for({"found": False}),
    )
    assert result is not None
    assert result.value == "中性"
    assert result.subject == "国电电力"
    assert "国电电力" in result.source_quote


def test_conflicting_ratings_abstain_instead_of_choosing_first():
    result = extract_bridge_value(
        slot="rating", original_question="国电电力的评级？", first_query="国电电力的评级？",
        chunks=[{"content_with_weight": "国电电力‘买入’评级。国电电力‘中性’评级。"}],
        client=client_for({"found": False}),
    )
    assert result is None
