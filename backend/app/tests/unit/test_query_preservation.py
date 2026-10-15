import pytest
from service.core.query import preservation as query_expansion

@pytest.mark.unit
def test_expansion_must_preserve_explicit_negation_and_full_date():
    original = "统计截至2025年8月20日的数据，不包括预测值。"
    candidate = "统计截至2025年8月20日的数据和预测值。"

    validation = query_expansion.validate_query_preservation(
        original,
        candidate,
    )

    assert validation.valid is False
    assert "不包括" in validation.missing_anchors
    assert "2025年8月20日" in validation.required_anchors


@pytest.mark.unit
def test_query_anchor_extraction_preserves_explicit_chinese_quarter():
    validation = query_expansion.validate_query_preservation(
        "国电电力2025年第二季度的收入是多少？",
        "国电电力2025年收入是多少？",
    )

    assert "2025年第二季度" in validation.required_anchors
    assert "2025年第二季度" in validation.missing_anchors
    assert validation.valid is False


@pytest.mark.unit
def test_query_anchor_extraction_does_not_treat_large_value_as_year():
    anchors = query_expansion.extract_query_anchors(
        "企业自由现金流为20808百万元"
    )

    assert "2080" not in anchors
    assert "2080年" not in anchors
    assert "20808" in anchors


@pytest.mark.unit
def test_report_context_is_not_part_of_organization_anchor():
    original = "这份研报对国电电力到底是看多还是看空？给了什么评级？"
    rewritten = "国电电力 研报 投资评级 看多 看空"

    anchors = query_expansion.extract_query_anchors(original)
    validation = query_expansion.validate_query_preservation(
        original,
        rewritten,
    )

    assert "国电电力" in anchors
    assert "这份研报对国电电力" not in anchors
    assert validation.valid is True
    assert validation.missing_anchors == []
    assert validation.added_risky_terms == []
