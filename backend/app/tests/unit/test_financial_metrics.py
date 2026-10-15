"""查询、正文、表格必须共享指标及度量语义。"""

import pytest

from service.core.finance.metrics import (
    metric_dimension, parse_metric_mentions,
    table_metric_mentions,
)
from agent.financial_coverage import requirement_period_labels, requirement_metric_mentions, requires_calculation
from service.core.finance.model_extraction import _candidate_measure

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(("text", "metric", "kind", "basis"), [
    ("营业收入", "营业收入", "level", "not_applicable"),
    ("营收同比增长率", "营业收入", "growth_rate", "year_over_year"),
    ("营业收入的同比增长率", "营业收入", "growth_rate", "year_over_year"),
    ("营业收入（同比增长率）", "营业收入", "growth_rate", "year_over_year"),
    ("营业收入环比增速", "营业收入", "growth_rate", "period_over_period"),
    ("营业收入增长率", "营业收入", "growth_rate", "unknown"),
    ("营业收入同比增加3亿元", "营业收入", "absolute_change", "year_over_year"),
    ("营业收入占比", "营业收入", "share", "not_applicable"),
    ("毛利率同比增加3个百分点", "毛利率", "absolute_change", "year_over_year"),
    ("经营现金流增长率", "经营现金流", "growth_rate", "unknown"),
    ("经营现金流净额", "经营活动现金流量净额", "level", "not_applicable"),
    ("预付账款", "预付账款", "level", "not_applicable"),
])
def test_metric_semantics(text, metric, kind, basis):
    mention, = parse_metric_mentions(text)
    assert mention.identity == (metric, kind, basis)
    assert text[mention.start:mention.end] == mention.raw_label


def test_alias_offsets_and_longest_metric_match():
    text = "预测 ＥＰＳ 和存货账面余额"
    mentions = parse_metric_mentions(text)
    assert [m.metric for m in mentions] == ["每股收益", "存货账面余额"]
    assert all(text[m.start:m.end] == m.raw_label for m in mentions)
    assert not parse_metric_mentions("PEOPLE")
    assert metric_dimension("市盈率") == "multiple"


def test_table_axes_share_the_same_qualifier_rules():
    assert table_metric_mentions("2025E同比增长率（%）", "营业收入")[0].identity == (
        "营业收入", "growth_rate", "year_over_year")
    assert table_metric_mentions("营业收入同比增长率（%）", "环比增长率") == []


def test_disclosed_rates_do_not_implicitly_request_a_calculation():
    for text in ("2025年营业收入增长率是多少", "2025年和2026年营业收入增长率分别是多少",
                 "2025E至2027E营业收入增长率", "2025E-2027E营业收入增长率"):
        assert not requires_calculation(text)
        assert requirement_metric_mentions(text)[0].measure_kind == "growth_rate"
    assert requirement_metric_mentions("计算2024年和2025年营业收入增长率")[0].measure_kind == "level"
    assert requirement_metric_mentions("2024年和2025年营业收入增长率相差多少")[0].measure_kind == "growth_rate"


def test_contiguous_forecast_years_include_the_middle_year():
    assert requirement_period_labels("2025E-2027E") == {"2025预测", "2026预测", "2027预测"}


def test_model_cannot_relabel_a_growth_rate_as_an_amount():
    quote = "B公司2025年营业收入同比增长率为20%"
    assert _candidate_measure({"measure_kind": "level"}, quote, "营业收入") is None
    assert _candidate_measure({}, quote, "营业收入") == ("growth_rate", "year_over_year")
    assert _candidate_measure({}, quote + "，营业收入为100亿元", "营业收入") is None
