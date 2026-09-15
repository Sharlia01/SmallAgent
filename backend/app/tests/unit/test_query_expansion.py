from types import SimpleNamespace

import pytest

from service.core import query_expansion


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
def test_expands_colloquial_finance_query_and_preserves_anchors(monkeypatch):
    monkeypatch.setenv("QUERY_EXPANSION_ENABLED", "true")
    completions = FakeCompletions(
        '{"expanded_query":"国电电力 未来三年 盈利预测 归母净利润 '
        '每股收益 EPS 市盈率 PE 估值","added_terms":'
        '["归母净利润","每股收益","EPS","市盈率","PE"]}'
    )
    original = "研报觉得国电电力未来三年能赚多少钱，对应估值贵不贵？"
    rewritten = "国电电力未来三年盈利预测与估值"

    result = query_expansion.expand_query(
        original,
        rewritten,
        client=fake_client(completions),
    )

    assert result.expansion_applied is True
    assert result.effective_query == (
        "国电电力 未来三年 盈利预测 归母净利润 "
        "每股收益 EPS 市盈率 PE 估值"
    )
    assert result.added_terms == [
        "归母净利润",
        "每股收益",
        "EPS",
        "市盈率",
        "PE",
    ]
    assert result.expansion_validation.valid is True
    assert "国电电力" in result.expansion_validation.required_anchors
    assert "未来三年" in result.expansion_validation.required_anchors
    assert "估值" in result.expansion_validation.required_anchors
    call = completions.calls[0]
    assert call["temperature"] == 0
    assert call["response_format"] == {"type": "json_object"}
    assert call["extra_body"] == {"enable_thinking": False}


@pytest.mark.unit
def test_missing_original_entity_falls_back_to_valid_rewrite(monkeypatch):
    monkeypatch.setenv("QUERY_EXPANSION_ENABLED", "true")
    completions = FakeCompletions(
        '{"expanded_query":"未来三年 盈利预测 归母净利润 EPS PE 估值",'
        '"added_terms":["归母净利润","EPS","PE"]}'
    )
    original = "国电电力未来三年能赚多少钱，估值贵不贵？"
    rewritten = "国电电力未来三年盈利预测与估值"

    result = query_expansion.expand_query(
        original,
        rewritten,
        client=fake_client(completions),
    )

    assert result.expansion_applied is False
    assert result.effective_query == rewritten
    assert result.expansion_validation.valid is False
    assert "国电电力" in result.expansion_validation.missing_anchors
    assert "保留校验失败" in result.fallback_reason


@pytest.mark.unit
def test_invented_year_is_rejected(monkeypatch):
    monkeypatch.setenv("QUERY_EXPANSION_ENABLED", "true")
    completions = FakeCompletions(
        '{"expanded_query":"国电电力 2025-2027年 盈利预测 归母净利润 '
        'EPS PE 估值","added_terms":["归母净利润","EPS","PE"]}'
    )
    original = "国电电力未来三年能赚多少钱，估值贵不贵？"
    rewritten = "国电电力未来三年盈利预测与估值"

    result = query_expansion.expand_query(
        original,
        rewritten,
        client=fake_client(completions),
    )

    assert result.expansion_applied is False
    assert result.effective_query == rewritten
    assert result.expansion_validation.valid is False
    assert "未来三年" in result.expansion_validation.missing_anchors
    assert any(
        "2025" in term for term in result.expansion_validation.added_risky_terms
    )


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
def test_invalid_rewrite_and_expansion_fall_back_to_original(monkeypatch):
    monkeypatch.setenv("QUERY_EXPANSION_ENABLED", "true")
    completions = FakeCompletions(
        '{"expanded_query":"归母净利润 EPS PE",'
        '"added_terms":["EPS","PE"]}'
    )
    original = "根据图3，国电电力2024年归母净利润是多少？"
    rewritten = "国电电力归母净利润"

    result = query_expansion.expand_query(
        original,
        rewritten,
        client=fake_client(completions),
    )

    assert result.effective_query == original
    assert result.rewrite_validation.valid is False
    assert result.expansion_validation.valid is False
    assert "图3" in result.rewrite_validation.missing_anchors
    assert "2024年" in result.rewrite_validation.missing_anchors


@pytest.mark.unit
def test_provider_failure_uses_valid_rewritten_query(monkeypatch):
    monkeypatch.setenv("QUERY_EXPANSION_ENABLED", "true")
    completions = FakeCompletions(error=TimeoutError("provider timeout"))
    original = "国电电力未来三年能赚多少钱？"
    rewritten = "国电电力未来三年盈利预测与归母净利润"

    result = query_expansion.expand_query(
        original,
        rewritten,
        client=fake_client(completions),
    )

    assert result.effective_query == rewritten
    assert result.expansion_applied is False
    assert result.source == "fallback"
    assert result.expansion_validation.valid is False


@pytest.mark.unit
def test_expansion_can_be_disabled_without_calling_model(monkeypatch):
    monkeypatch.setenv("QUERY_EXPANSION_ENABLED", "false")
    completions = FakeCompletions(error=AssertionError("must not be called"))
    original = "国电电力未来三年能赚多少钱？"
    rewritten = "国电电力未来三年盈利预测与归母净利润"

    result = query_expansion.expand_query(
        original,
        rewritten,
        client=fake_client(completions),
    )

    assert result.effective_query == rewritten
    assert result.source == "disabled"
    assert completions.calls == []


@pytest.mark.unit
def test_unchanged_rewrite_skips_expansion_without_calling_model(monkeypatch):
    monkeypatch.setenv("QUERY_EXPANSION_ENABLED", "true")
    completions = FakeCompletions(error=AssertionError("must not be called"))
    original = "国电电力归母净利润"

    result = query_expansion.expand_query(
        original,
        original,
        client=fake_client(completions),
    )

    assert result.effective_query == original
    assert result.source == "skipped"
    assert result.expansion_applied is False
    assert completions.calls == []


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

    assert not any(anchor.startswith("2080") for anchor in anchors)
