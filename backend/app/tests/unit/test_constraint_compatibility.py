import pytest

from service.core.constraint_compatibility import (
    evaluate_constraint_compatibility,
    extract_measurement_dimensions,
    extract_organizations,
    extract_time_scopes,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2025年第二季度", {"2025-Q2"}),
        ("2025 Q2", {"2025-Q2"}),
        ("25Q2", {"2025-Q2"}),
        ("Q2 2025", {"2025-Q2"}),
        ("2025年上半年", {"2025-H1"}),
        ("第二季度", {"ANY-Q2"}),
        ("2025年", {"2025-YEAR"}),
        ("2026E", {"2026-FY"}),
        ("截至2025年6月", {"2025-M06"}),
        (
            "图2：单季营业收入 2023 2024 2025 600 500 100 Q1 Q2 Q3 Q4",
            set(),
        ),
    ],
)
def test_extract_time_scopes_normalizes_common_financial_periods(
    value,
    expected,
):
    assert extract_time_scopes(value) == expected


@pytest.mark.unit
def test_extract_organizations_removes_query_lead_in_text():
    assert extract_organizations(
        "关键财务与估值指标表预计国电电力2025E每股红利是多少？"
    ) == {"国电电力"}


@pytest.mark.unit
def test_constraint_compatibility_distinguishes_match_unknown_and_conflict():
    question = "国电电力2025年第二季度的收入表现如何？"

    compatible = evaluate_constraint_compatibility(
        question,
        "2025年第二季度，公司实现收入378.42亿元。",
        document_name="国电电力.pdf",
    )
    unknown = evaluate_constraint_compatibility(
        question,
        "公司实现收入378.42亿元。",
        document_name="国电电力.pdf",
    )
    conflict = evaluate_constraint_compatibility(
        question,
        "2025年上半年，公司实现收入776.55亿元。",
        document_name="国电电力.pdf",
    )

    assert compatible.categories["time"].level == "compatible"
    assert compatible.conflict_count == 0
    assert unknown.categories["time"].level == "unknown"
    assert unknown.conflict_count == 0
    assert conflict.categories["time"].level == "conflict"
    assert conflict.conflict_count == 1


@pytest.mark.unit
def test_bare_year_is_compatible_with_a_more_specific_same_year_scope():
    decision = evaluate_constraint_compatibility(
        "国电电力2025年的营业收入是多少？",
        "2025年上半年营业收入为776.55亿元。",
        document_name="国电电力.pdf",
    )

    assert decision.categories["time"].level == "compatible"


@pytest.mark.unit
def test_candidate_bare_year_stays_unknown_for_quarter_query():
    decision = evaluate_constraint_compatibility(
        "国电电力2025年第二季度收入是多少？",
        "2025年营业收入情况。",
        document_name="国电电力.pdf",
    )

    assert decision.categories["time"].level == "unknown"


@pytest.mark.unit
def test_constraint_compatibility_detects_actual_forecast_conflict():
    decision = evaluate_constraint_compatibility(
        "国电电力2025年全年实际实现的营业收入是多少？",
        "预计2025E营业收入为1769.61亿元。",
        document_name="国电电力.pdf",
    )

    assert decision.categories["fact_status"].level == "conflict"
    assert decision.categories["time"].level == "compatible"


@pytest.mark.unit
def test_achievement_wording_does_not_imply_actual_value_constraint():
    decision = evaluate_constraint_compatibility(
        "2025年上半年是否实现了量利双增？",
        "2025年上半年光伏发电量增长122.55%。",
    )

    assert "fact_status" not in decision.categories


@pytest.mark.unit
def test_constraint_compatibility_detects_locator_conflict():
    decision = evaluate_constraint_compatibility(
        "根据图3，哪一年的归母净利润为负？",
        "图2：国电电力单季营业收入。",
        document_name="国电电力.pdf",
    )

    assert decision.categories["locator"].level == "conflict"


@pytest.mark.unit
def test_measurement_dimension_extraction_is_business_agnostic():
    assert extract_measurement_dimensions("财务费用率为3.71%") == {"rate"}
    assert extract_measurement_dimensions("毛利率为16.27%") == {"rate"}
    assert extract_measurement_dimensions("财务费用为10049百万元") == {
        "currency_amount"
    }
    assert extract_measurement_dimensions("项目成本为500元") == {
        "currency_amount"
    }
    assert extract_measurement_dimensions("每股收益为0.38元/股") == {
        "per_share"
    }
    assert extract_measurement_dimensions("装机规模达到100万千瓦") == {
        "power"
    }
    assert extract_measurement_dimensions("发电量为200亿千瓦时") == {
        "energy"
    }


@pytest.mark.unit
def test_measurement_compatibility_detects_dimension_not_business_identity():
    rate_question = "国电电力2025年上半年财务费用率是多少？"

    exact_rate = evaluate_constraint_compatibility(
        rate_question,
        "2025年上半年财务费用率为3.71%。",
        document_name="国电电力.pdf",
    )
    expense_amount = evaluate_constraint_compatibility(
        rate_question,
        "2026E财务费用为10049百万元。",
        document_name="国电电力.pdf",
    )
    unrelated_margin = evaluate_constraint_compatibility(
        rate_question,
        "2025年上半年毛利率为16.27%。",
        document_name="国电电力.pdf",
    )

    assert exact_rate.categories["measurement_dimension"].level == (
        "compatible"
    )
    assert expense_amount.categories["measurement_dimension"].level == (
        "conflict"
    )
    # Both are rates.  Topic identity is intentionally left to the semantic
    # reranker instead of a corpus-specific metric-name dictionary.
    assert unrelated_margin.categories["measurement_dimension"].level == (
        "compatible"
    )
    assert exact_rate.categories["measurement_subject"].level == "compatible"
    assert expense_amount.categories["measurement_subject"].level == (
        "compatible"
    )
    assert unrelated_margin.categories["measurement_subject"].level == (
        "unknown"
    )
