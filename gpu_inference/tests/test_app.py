from fastapi.testclient import TestClient

import app as inference_app


def test_embedding_endpoint_authenticates_and_returns_indexed_vectors(monkeypatch):
    class FakeEmbeddingEngine:
        def embed(self, texts, *, batch_size):
            assert texts == ["第一条", "第二条"]
            assert batch_size == 32
            return [
                [float(index)] * inference_app.EMBEDDING_DIMENSION
                for index in range(len(texts))
            ]

    monkeypatch.setenv("INFERENCE_SERVER_API_KEY", "shared-secret")
    monkeypatch.setattr(
        inference_app,
        "get_embedding_engine",
        lambda: FakeEmbeddingEngine(),
    )

    with TestClient(inference_app.app) as client:
        unauthorized = client.post(
            "/v1/embeddings",
            json={
                "model": "BAAI/bge-small-zh-v1.5",
                "input": ["第一条", "第二条"],
            },
        )
        response = client.post(
            "/v1/embeddings",
            headers={"Authorization": "Bearer shared-secret"},
            json={
                "model": "BAAI/bge-small-zh-v1.5",
                "input": ["第一条", "第二条"],
            },
        )

    assert unauthorized.status_code == 401
    assert response.status_code == 200
    assert response.json()["dimension"] == 512
    assert [item["index"] for item in response.json()["data"]] == [0, 1]


def test_reranker_endpoint_keeps_original_indexes_in_ranked_response(monkeypatch):
    class FakeRerankerEngine:
        def rerank(self, query, documents, *, batch_size, max_length):
            assert query == "问题"
            assert documents == ["片段一", "片段二"]
            assert batch_size == 8
            assert max_length == 512
            return [0.2, 0.9]

    monkeypatch.delenv("INFERENCE_SERVER_API_KEY", raising=False)
    monkeypatch.setattr(
        inference_app,
        "get_reranker_engine",
        lambda: FakeRerankerEngine(),
    )

    with TestClient(inference_app.app) as client:
        response = client.post(
            "/v1/rerank",
            json={
                "model": "BAAI/bge-reranker-v2-m3",
                "query": "问题",
                "documents": ["片段一", "片段二"],
                "top_n": 2,
                "max_length": 512,
            },
        )

    assert response.status_code == 200
    assert response.json()["score_mode"] == "raw_logits"
    assert response.json()["results"] == [
        {"index": 1, "relevance_score": 0.9},
        {"index": 0, "relevance_score": 0.2},
    ]


def test_request_item_limit_is_enforced(monkeypatch):
    monkeypatch.delenv("INFERENCE_SERVER_API_KEY", raising=False)
    monkeypatch.setenv("INFERENCE_MAX_REQUEST_ITEMS", "1")

    with TestClient(inference_app.app) as client:
        response = client.post(
            "/v1/embeddings",
            json={
                "model": "BAAI/bge-small-zh-v1.5",
                "input": ["第一条", "第二条"],
            },
        )

    assert response.status_code == 413

