from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from service.core.rag.nlp import model as reranker


pytestmark = pytest.mark.unit


def test_rerank_uses_local_bge_in_batches_and_preserves_input_order(monkeypatch):
    tokenizer = Mock(side_effect=[
        {"input_ids": torch.tensor([[1], [2]])},
        {"input_ids": torch.tensor([[3]])},
    ])
    model = Mock(side_effect=[
        SimpleNamespace(logits=torch.tensor([[0.42], [0.91]])),
        SimpleNamespace(logits=torch.tensor([[-0.25]])),
    ])
    monkeypatch.setattr(
        reranker,
        "get_reranker_model",
        lambda: (tokenizer, model, "cpu"),
    )
    monkeypatch.setenv("RERANKER_MAX_LENGTH", "512")

    scores, metadata = reranker.rerank_similarity(
        "原始问题",
        iter(["候选片段一", "候选片段二", "候选片段三"]),
        batch_size=2,
    )

    np.testing.assert_allclose(scores, [0.42, 0.91, -0.25])
    assert metadata is None
    assert tokenizer.call_args_list[0].args[0] == [
        ["原始问题", "候选片段一"],
        ["原始问题", "候选片段二"],
    ]
    assert tokenizer.call_args_list[0].kwargs == {
        "padding": True,
        "truncation": True,
        "return_tensors": "pt",
        "max_length": 512,
    }
    assert tokenizer.call_args_list[1].args[0] == [
        ["原始问题", "候选片段三"],
    ]
    assert model.call_count == 2


def test_empty_candidates_do_not_load_local_model(monkeypatch):
    load_model = Mock(side_effect=AssertionError("must not load model"))
    monkeypatch.setattr(reranker, "get_reranker_model", load_model)

    scores, metadata = reranker.rerank_similarity("问题", [])

    assert scores.shape == (0,)
    assert metadata is None
    load_model.assert_not_called()


@pytest.mark.parametrize(
    "query,texts",
    [(None, ["片段"]), ("问题", "片段"), ("问题", [None])],
)
def test_invalid_inputs_are_rejected(query, texts):
    with pytest.raises(TypeError):
        reranker.rerank_similarity(query, texts)


def test_invalid_batch_size_is_rejected(monkeypatch):
    monkeypatch.setattr(
        reranker,
        "get_reranker_model",
        Mock(side_effect=AssertionError("must validate first")),
    )

    with pytest.raises(ValueError, match="batch_size"):
        reranker.rerank_similarity("问题", ["片段"], batch_size=0)


def test_missing_local_model_is_reported(monkeypatch, tmp_path):
    missing = tmp_path / "missing-reranker"
    monkeypatch.setenv("RERANKER_MODEL_PATH", str(missing))
    reranker.get_reranker_model.cache_clear()

    with pytest.raises(RuntimeError, match="does not exist"):
        reranker.get_reranker_model()

    reranker.get_reranker_model.cache_clear()


def test_local_inference_error_is_reported(monkeypatch):
    tokenizer = Mock(side_effect=RuntimeError("tokenization failed"))
    monkeypatch.setattr(
        reranker,
        "get_reranker_model",
        lambda: (tokenizer, Mock(), "cpu"),
    )

    with pytest.raises(RuntimeError, match="Local BAAI/bge-reranker-v2-m3"):
        reranker.rerank_similarity("问题", ["片段"])


@pytest.mark.parametrize(
    "logits,error",
    [
        (torch.tensor([[float("nan")], [0.1]]), "non-finite"),
        (torch.tensor([[0.2]]), "1 scores for 2 candidates"),
    ],
)
def test_invalid_local_scores_are_rejected(monkeypatch, logits, error):
    tokenizer = Mock(return_value={"input_ids": torch.tensor([[1], [2]])})
    model = Mock(return_value=SimpleNamespace(logits=logits))
    monkeypatch.setattr(
        reranker,
        "get_reranker_model",
        lambda: (tokenizer, model, "cpu"),
    )

    with pytest.raises(RuntimeError, match=error):
        reranker.rerank_similarity("问题", ["片段一", "片段二"])
