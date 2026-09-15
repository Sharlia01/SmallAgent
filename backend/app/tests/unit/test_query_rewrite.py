from types import SimpleNamespace

import pytest

from service.core import query_rewrite


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
def test_rewrite_query_normalizes_colloquial_finance_question(monkeypatch):
    monkeypatch.setenv("QUERY_REWRITE_ENABLED", "true")
    completions = FakeCompletions(
        '{"query":"国电电力未来三年盈利预测、归母净利润、每股收益 EPS 和市盈率 PE 估值"}'
    )

    result = query_rewrite.rewrite_query(
        "研报觉得国电电力未来三年能赚多少钱，对应估值贵不贵？",
        client=fake_client(completions),
    )

    assert result == (
        "国电电力未来三年盈利预测、归母净利润、每股收益 EPS 和市盈率 PE 估值"
    )
    call = completions.calls[0]
    assert call["temperature"] == 0
    assert call["response_format"] == {"type": "json_object"}
    assert call["extra_body"] == {"enable_thinking": False}
    assert call["messages"][-1] == {
        "role": "user",
        "content": "研报觉得国电电力未来三年能赚多少钱，对应估值贵不贵？",
    }


@pytest.mark.unit
def test_rewrite_query_falls_back_when_provider_fails(monkeypatch):
    monkeypatch.setenv("QUERY_REWRITE_ENABLED", "true")
    completions = FakeCompletions(error=TimeoutError("provider timeout"))

    result = query_rewrite.rewrite_query(
        "口语问题",
        client=fake_client(completions),
    )

    assert result == "口语问题"


@pytest.mark.unit
def test_rewrite_query_can_be_disabled_without_calling_model(monkeypatch):
    monkeypatch.setenv("QUERY_REWRITE_ENABLED", "false")
    completions = FakeCompletions(error=AssertionError("must not be called"))

    result = query_rewrite.rewrite_query(
        "保持原问题",
        client=fake_client(completions),
    )

    assert result == "保持原问题"
    assert completions.calls == []


@pytest.mark.unit
def test_rewrite_query_rejects_malformed_output(monkeypatch):
    monkeypatch.setenv("QUERY_REWRITE_ENABLED", "true")
    completions = FakeCompletions('{"answer":"错误字段"}')

    result = query_rewrite.rewrite_query(
        "原始问题",
        client=fake_client(completions),
    )

    assert result == "原始问题"
