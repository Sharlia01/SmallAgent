"""基于 baselineV2_v4 的验收回归，模型输出使用测试夹具。

历史运行未保存被拒绝的模型响应；这些测试通过解析和执行
覆盖有效等价表达及无效反例，不声称重放缺失的提供方输出。
"""

from service.core.evidence.adapter import reference_records

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from service.core.query.contract import QueryGoal, parse_goals
from service.core.query.preservation import _extract_organizations, validate_query_preservation
from service.core.runtime.context import QueryRuntime, query_runtime
from service.core.query.validation import QueryValidationError, validate_goals
from service.core.evidence.bindings import BridgeExtraction


pytestmark = pytest.mark.unit


class ScriptedClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        value = next(self.responses)
        if isinstance(value, Exception):
            raise value
        content = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=content), finish_reason="stop",
        )], usage=SimpleNamespace(total_tokens=10))


def analysis(goals, *, dependent=False):
    return dict(intent="complex_lookup", need_rewrite=False, target_count=len(goals),
                targets_independent=not dependent and len(goals) > 1,
                has_dependency=dependent, single_retrieval_likely_covers_all=False,
                reason="独立事实或依赖事实需要分别检索", goals=goals)


DIVIDENDS = "国电电力2025—2027年的现金分红底线是什么，2025年上半年拟如何分红？"
DIVIDEND_GOALS = [
    dict(id="floor", text="国电电力2025—2027年的现金分红底线", constraints=["国电电力", "2025—2027年", "现金分红底线"]),
    dict(id="interim", text="国电电力2025年上半年的分红安排", constraints=["国电电力", "2025年上半年", "分红"]),
]
VALUATION = "研报觉得国电电力未来三年能赚多少钱，对应估值贵不贵？"
VALUATION_GOALS = [
    dict(id="profit", text="国电电力未来三年的盈利预测", constraints=["国电电力", "未来三年", "赚多少钱"]),
    dict(id="value", text="国电电力未来三年的估值水平", constraints=["国电电力", "未来三年", "估值贵不贵"]),
]
FIGURES = "不要混淆图2和图3：哪张图显示2021年归母净利润为负，哪张图显示2025年只有第一、第二季度营业收入柱？"
FIGURE_GOALS = [
    dict(id="profit", text="在图2和图3中，哪张图显示2021年归母净利润为负？", constraints=["图2", "图3", "2021年", "归母净利润为负"]),
    dict(id="revenue", text="在图2和图3中，哪张图显示2025年只有第一、第二季度营业收入柱？", constraints=["图2", "图3", "2025年", "只有第一、第二季度营业收入柱"]),
]


@pytest.mark.parametrize("filename", [
    "guodian_power_eval_v2.jsonl",
    "power_reports_eval_v2_dev.jsonl",
    "power_reports_eval_v2_test.jsonl",
])
def test_every_benchmark_question_preserves_itself(filename):
    dataset = Path(__file__).resolve().parents[3] / "evals" / "data" / filename
    rows = [json.loads(line) for line in dataset.read_text().splitlines()]
    for row in rows:
        result = validate_query_preservation(row["question"], row["question"])
        assert result.valid, (row["id"], result)


@pytest.mark.parametrize("text", [
    "2025年上半年国电电力各电源板块归母净利润", "查询国电电力2025年的收入",
    "查找国电电力的财务费用", "国电电力和华能能源的营业收入",
])
def test_entity_extraction_only_returns_contiguous_names(text):
    entities = _extract_organizations(text)
    assert "国电电力" in entities
    assert all(entity in text and not entity.startswith(("查询", "查找", "2025")) for entity in entities)


@pytest.mark.parametrize(("original", "candidate"), [
    ("2025年上半年国电电力归母净利润", "查询国电电力2025年上半年归母净利润"),
    ("国电电力2026E固定资产是多少", "国电电力2026年固定资产预测值"),
    ("国电电力2025年每股收益", "国电电力2025 EPS"),
    ("2025年上半年营业收入", "2025H1营收"),
])
def test_equivalent_queries_are_accepted(original, candidate):
    assert validate_query_preservation(original, candidate).valid






def test_goal_constraint_cannot_add_a_year_absent_from_the_question():
    with pytest.raises(QueryValidationError, match="goal_source_not_grounded"):
        parse_goals([dict(VALUATION_GOALS[1], constraints=["国电电力", "2028年"])], VALUATION)


def test_paraphrased_constraint_does_not_claim_to_quote_the_question():
    goal = parse_goals([dict(VALUATION_GOALS[1], constraints=["是否昂贵"])], VALUATION)[0]
    assert goal.constraint_provenance[0]["kind"] == "paraphrase"
    assert goal.constraint_provenance[0]["source_spans"] == ()


def test_constraint_cannot_move_a_year_to_a_different_metric():
    question = "2025年营业收入和2026年净利润"
    with pytest.raises(QueryValidationError, match="metric_time_rebound"):
        parse_goals([dict(id="revenue", text="2025年营业收入", constraints=["2026年营业收入"])], question)


def test_price_unit_basis_remains_compatible_with_guodian_015():
    question = "上半年卖电均价掉了多少，现在一兆瓦时大概多少钱？"
    goal = parse_goals([dict(id="price", text="现在每兆瓦时电价是多少",
                             constraints=["现在", "一兆瓦时", "多少钱"])], question)[0]
    assert goal.constraints == ("现在", "每兆瓦时", "多少钱")






















@pytest.mark.parametrize(("question", "texts", "rule"), [
    ("国电电力2024年利润和华能国际2025年利润", ["国电电力2025年利润", "华能国际2024年利润"], "subject_time_rebound"),
    ("国电电力2025年营业收入和2026年净利润", ["国电电力2026年营业收入", "国电电力2025年净利润"], "metric_time_rebound"),
    ("图2营业收入，图3归母净利润", ["图3营业收入", "图2归母净利润"], "metric_locator_rebound"),
    ("国电电力2025年上半年光伏发电量和上网电量同比增幅", ["国电电力2025年上半年光伏发电量同比增幅", "国电电力2025年上半年上网电量同比增幅"], "metric_scope_dropped"),
])
def test_wrong_local_bindings_are_rejected(question, texts, rule):
    with pytest.raises(QueryValidationError, match=rule):
        validate_goals(question, tuple(QueryGoal(f"g{i}", text) for i, text in enumerate(texts)))


@pytest.mark.parametrize(("original", "candidate"), [
    ("2026E营业收入", "2026年实际营业收入"),
    ("2025年上半年营业收入", "2025年全年营业收入"),
    ("国电电力2025年营收", "华能能源2025年营收"),
    ("先查归母净利润最高的电源类型", "查询煤电归母净利润"),
])
def test_acceptance_fixes_do_not_allow_invented_values_or_scopes(original, candidate):
    assert not validate_query_preservation(original, candidate).valid




def test_reporting_period_cannot_move_to_another_metric_in_same_year():
    question = "2025年上半年营业收入和2025年下半年净利润"
    with pytest.raises(QueryValidationError, match="metric_period_rebound"):
        validate_goals(question, (
            QueryGoal("a", "2025年下半年营业收入"), QueryGoal("b", "2025年上半年净利润"),
        ))
