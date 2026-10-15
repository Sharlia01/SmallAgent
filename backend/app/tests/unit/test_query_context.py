"""查询入口完成消解，并在同一轮工具调用间复用上下文。"""

from dataclasses import FrozenInstanceError, replace
from unittest.mock import Mock

import pytest

from service.core.query import context as query_context
from service.core.query.contract import ContextBinding, QueryRequest, QueryUnderstanding
from service.core.runtime.context import query_runtime


pytestmark = pytest.mark.unit


def test_prepare_resolves_once_and_preserves_original_request(monkeypatch):
    question = "它今年的现金流呢？"
    history = [{"role": "user", "content": "请介绍国电电力"}]
    original = QueryRequest.from_user(question, history)
    resolved = replace(
        original, resolved_question="国电电力今年的现金流呢？",
        bindings=(ContextBinding("它", "国电电力", 0, "国电电力"),),
    )
    resolve = Mock(return_value=resolved)
    monkeypatch.setattr(query_context, "resolve_request", resolve)

    with query_runtime():
        first = query_context.prepare_query_context(question, history)
        plan = QueryUnderstanding(source="rule")
        first.understanding = plan
        second = query_context.prepare_query_context(question, history)
        reused = query_context.prepare_query_context("模型提炼的检索词", context=first)

    assert first is second is reused
    assert first.request is resolved
    assert first.question == "国电电力今年的现金流呢？"
    assert first.request.user_question == question
    assert first.request.history == original.history
    assert first.understanding is plan
    resolve.assert_called_once_with(original)
    with pytest.raises(FrozenInstanceError):
        first.request.user_question = "改掉原问题"


def test_prepare_respects_explicit_request_and_context(monkeypatch):
    request = QueryRequest.from_user("它的利润呢？")
    context = query_context.QueryContext(request)
    resolve = Mock(return_value=replace(request, clarification="请明确对象。"))
    monkeypatch.setattr(query_context, "resolve_request", resolve)

    prepared = query_context.prepare_query_context(
        "其他问题", request=QueryRequest.from_user("其他请求"), context=context,
    )

    assert prepared is context
    assert prepared.request.user_question == request.user_question
    assert prepared.clarification == "请明确对象。"
    resolve.assert_called_once_with(request)


def test_prepare_explicit_request_keeps_history_for_resolution(monkeypatch):
    request = QueryRequest.from_user("它的利润呢？", [{"role": "user", "content": "国电电力"}])
    resolve = Mock(return_value=replace(request, resolved_question="国电电力的利润呢？"))
    monkeypatch.setattr(query_context, "resolve_request", resolve)

    prepared = query_context.prepare_query_context("模型检索词", request=request)

    assert prepared.question == "国电电力的利润呢？"
    assert prepared.request.history == request.history
    resolve.assert_called_once_with(request)


def test_prepare_without_runtime_exposes_clarification():
    prepared = query_context.prepare_query_context("它的利润呢？")

    assert prepared.request.user_question == "它的利润呢？"
    assert prepared.clarification == "请明确“它”指的是哪个对象。"


def test_prepared_context_is_scoped_to_runtime():
    with query_runtime():
        first = query_context.prepare_query_context("现金流是多少？")
        assert query_context.prepare_query_context("现金流是多少？") is first
    with query_runtime():
        second = query_context.prepare_query_context("现金流是多少？")

    assert second is not first
    assert second.understanding is None
