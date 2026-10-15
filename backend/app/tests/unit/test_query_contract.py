import json
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from service.core.query.contract import QueryRequest, parse_goals, resolve_request
from service.core.query.preservation import validate_query_preservation
from agent.plans import AgentTask as QuerySubqueryPlan, validate_dependency_graph
from service.core.evidence.bindings import extract_bridge_value


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








@pytest.mark.parametrize("steps", [
    [QuerySubqueryPlan("a", query="问题", depends_on=["missing"])],
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
