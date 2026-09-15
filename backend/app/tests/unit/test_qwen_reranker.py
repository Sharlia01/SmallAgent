from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from service.core.rag.nlp import model as reranker


pytestmark = pytest.mark.unit


def _response(results, *, status_code=200):
    return SimpleNamespace(
        status_code=status_code,
        output=SimpleNamespace(results=results),
        code=None,
        message=None,
    )


def test_rerank_uses_qwen_and_restores_input_order(monkeypatch):
    call = Mock(
        return_value=_response(
            [
                SimpleNamespace(index=1, relevance_score=0.91),
                SimpleNamespace(index=0, relevance_score=0.42),
            ]
        )
    )
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setattr(reranker.dashscope.TextReRank, "call", call)

    scores, metadata = reranker.rerank_similarity(
        "原始问题",
        iter(["候选片段一", "候选片段二"]),
    )

    np.testing.assert_allclose(scores, [0.42, 0.91])
    assert metadata is None
    call.assert_called_once_with(
        model="qwen3.7-text-rerank",
        top_n=2,
        query="原始问题",
        documents=["候选片段一", "候选片段二"],
        api_key="test-key",
    )


def test_empty_candidates_do_not_call_dashscope(monkeypatch):
    call = Mock(side_effect=AssertionError("must not call DashScope"))
    monkeypatch.setattr(reranker.dashscope.TextReRank, "call", call)

    scores, metadata = reranker.rerank_similarity("问题", [])

    assert scores.shape == (0,)
    assert metadata is None
    call.assert_not_called()


@pytest.mark.parametrize(
    "query,texts",
    [(None, ["片段"]), ("问题", "片段"), ("问题", [None])],
)
def test_invalid_inputs_are_rejected(query, texts):
    with pytest.raises(TypeError):
        reranker.rerank_similarity(query, texts)


def test_missing_api_key_is_reported(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="DASHSCOPE_API_KEY"):
        reranker.rerank_similarity("问题", ["片段"])


def test_provider_error_is_reported(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setattr(
        reranker.dashscope.TextReRank,
        "call",
        Mock(side_effect=TimeoutError("provider timeout")),
    )

    with pytest.raises(RuntimeError, match="qwen3.7-text-rerank request failed"):
        reranker.rerank_similarity("问题", ["片段"])


def test_unsuccessful_response_is_reported(monkeypatch):
    response = _response([], status_code=500)
    response.code = "InternalError"
    response.message = "failed"
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setattr(
        reranker.dashscope.TextReRank,
        "call",
        Mock(return_value=response),
    )

    with pytest.raises(RuntimeError, match="returned 500"):
        reranker.rerank_similarity("问题", ["片段"])


@pytest.mark.parametrize(
    "results,error",
    [
        ([SimpleNamespace(index=2, relevance_score=0.5)], "invalid result index"),
        (
            [
                SimpleNamespace(index=0, relevance_score=0.5),
                SimpleNamespace(index=0, relevance_score=0.4),
            ],
            "duplicate result index",
        ),
        ([SimpleNamespace(index=0, relevance_score="bad")], "invalid relevance score"),
        ([SimpleNamespace(index=0, relevance_score=np.nan)], "did not score every"),
        ([], "did not score every"),
    ],
)
def test_invalid_rerank_results_are_rejected(monkeypatch, results, error):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setattr(
        reranker.dashscope.TextReRank,
        "call",
        Mock(return_value=_response(results)),
    )

    with pytest.raises(RuntimeError, match=error):
        reranker.rerank_similarity("问题", ["片段"])
