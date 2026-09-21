import json

import httpx
import numpy as np
import pytest

from service.core.rag.nlp import model as inference


pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def clear_provider_caches():
    inference.clear_inference_caches()
    yield
    inference.clear_inference_caches()


def test_remote_embedding_batches_requests_and_restores_input_order():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url.path == "/v1/embeddings"
        assert request.headers["Authorization"] == "Bearer secret"
        payload = json.loads(request.content)
        assert payload["model"] == "BAAI/bge-small-zh-v1.5"
        assert payload["normalize"] is True
        data = [
            {
                "index": index,
                "embedding": [float(1 + index)] * inference.EMBEDDING_DIMENSION,
            }
            for index in reversed(range(len(payload["input"])))
        ]
        return httpx.Response(
            200,
            json={"dimension": inference.EMBEDDING_DIMENSION, "data": data},
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = inference.RemoteEmbeddingProvider(
        base_url="https://gpu.example.com",
        model_id="BAAI/bge-small-zh-v1.5",
        api_key="secret",
        max_retries=0,
        client=client,
    )

    vectors = provider.embed(["a", "b", "c"], batch_size=2)

    assert len(requests) == 2
    assert [vector[0] for vector in vectors] == [1.0, 2.0, 1.0]
    client.close()


def test_remote_embedding_rejects_a_different_dimension():
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                json={"dimension": 1024, "data": []},
            )
        )
    )
    provider = inference.RemoteEmbeddingProvider(
        base_url="https://gpu.example.com",
        model_id="embedding-model",
        max_retries=0,
        client=client,
    )

    with pytest.raises(RuntimeError, match="dimension 1024"):
        provider.embed(["text"], batch_size=1)
    client.close()


def test_remote_embedding_rejects_a_different_model():
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                json={"model": "another-model", "data": []},
            )
        )
    )
    provider = inference.RemoteEmbeddingProvider(
        base_url="https://gpu.example.com",
        model_id="expected-model",
        max_retries=0,
        client=client,
    )

    with pytest.raises(RuntimeError, match="another-model"):
        provider.embed(["text"], batch_size=1)
    client.close()


def test_remote_reranker_restores_order_and_converts_probability_scores():
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert request.url.path == "/v1/rerank"
        assert payload["query"] == "问题"
        assert payload["top_n"] == len(payload["documents"])
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 1, "relevance_score": 0.8},
                    {"index": 0, "relevance_score": 0.2},
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = inference.RemoteRerankerProvider(
        base_url="https://gpu.example.com",
        model_id="BAAI/bge-reranker-v2-m3",
        max_retries=0,
        response_score_mode="probability",
        output_score_mode="raw_logits",
        client=client,
    )

    scores = provider.rerank(
        "问题",
        ["片段一", "片段二"],
        batch_size=2,
        max_length=512,
    )

    np.testing.assert_allclose(scores, [np.log(0.2 / 0.8), np.log(0.8 / 0.2)])
    client.close()


def test_remote_reranker_rejects_a_mismatched_declared_score_mode():
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                json={
                    "score_mode": "probability",
                    "results": [{"index": 0, "relevance_score": 0.8}],
                },
            )
        )
    )
    provider = inference.RemoteRerankerProvider(
        base_url="https://gpu.example.com",
        model_id="reranker",
        max_retries=0,
        response_score_mode="raw_logits",
        client=client,
    )

    with pytest.raises(RuntimeError, match="score mode"):
        provider.rerank("问题", ["片段"], batch_size=1, max_length=512)
    client.close()


def test_embedding_and_reranker_backends_are_selected_independently(monkeypatch):
    monkeypatch.setenv("EMBEDDING_BACKEND", "remote")
    monkeypatch.setenv("EMBEDDING_REMOTE_BASE_URL", "https://gpu.example.com")
    monkeypatch.setenv("RERANKER_BACKEND", "local")

    assert isinstance(
        inference.get_embedding_provider(),
        inference.RemoteEmbeddingProvider,
    )
    assert isinstance(
        inference.get_reranker_provider(),
        inference.LocalRerankerProvider,
    )


def test_remote_backend_requires_its_base_url(monkeypatch):
    monkeypatch.setenv("EMBEDDING_BACKEND", "remote")
    monkeypatch.delenv("EMBEDDING_REMOTE_BASE_URL", raising=False)

    with pytest.raises(RuntimeError, match="EMBEDDING_REMOTE_BASE_URL"):
        inference.get_embedding_provider()


def test_reranker_defaults_to_probability_scores_for_retrieval(monkeypatch):
    monkeypatch.setenv("RERANKER_BACKEND", "remote")
    monkeypatch.setenv("RERANKER_REMOTE_BASE_URL", "https://gpu.example.com")
    monkeypatch.delenv("RERANKER_SCORE_MODE", raising=False)
    monkeypatch.delenv("RERANKER_REMOTE_SCORE_MODE", raising=False)

    provider = inference.get_reranker_provider()

    assert provider.output_score_mode == "probability"
    assert provider.response_score_mode == "raw_logits"


def test_unknown_backend_is_rejected(monkeypatch):
    monkeypatch.setenv("RERANKER_BACKEND", "somewhere")

    with pytest.raises(RuntimeError, match="RERANKER_BACKEND"):
        inference.get_reranker_provider()
