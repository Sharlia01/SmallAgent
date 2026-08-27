import numpy as np
import pytest

from service.core.rag.nlp import model as embedding_model


class FakeSentenceTransformer:
    def __init__(self, dimension=512):
        self.dimension = dimension
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((texts, kwargs))
        return np.ones((len(texts), self.dimension), dtype=np.float32)


@pytest.mark.unit
def test_bge_embedding_contract_is_fixed_to_512_dimensions():
    assert embedding_model.EMBEDDING_DIMENSION == 512
    assert embedding_model.EMBEDDING_VECTOR_FIELD == "q_512_vec"


@pytest.mark.unit
def test_generate_embedding_uses_local_model_for_single_and_batch(monkeypatch):
    fake_model = FakeSentenceTransformer()
    monkeypatch.setattr(embedding_model, "get_embedding_model", lambda: fake_model)
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "7")

    single_embedding = embedding_model.generate_embedding("单条问题")
    batch_embeddings = embedding_model.generate_embedding(
        ["第一个片段", "第二个片段"],
        batch_size=3,
    )

    assert len(single_embedding) == 512
    assert len(batch_embeddings) == 2
    assert all(len(vector) == 512 for vector in batch_embeddings)
    assert fake_model.calls[0][0] == ["单条问题"]
    assert fake_model.calls[0][1] == {
        "batch_size": 7,
        "convert_to_numpy": True,
        "normalize_embeddings": True,
        "show_progress_bar": False,
    }
    assert fake_model.calls[1][1]["batch_size"] == 3


@pytest.mark.unit
def test_generate_embedding_rejects_unexpected_dimension(monkeypatch):
    monkeypatch.setattr(
        embedding_model,
        "get_embedding_model",
        lambda: FakeSentenceTransformer(dimension=1024),
    )

    with pytest.raises(RuntimeError, match="unexpected embedding shape"):
        embedding_model.generate_embedding("测试")


@pytest.mark.unit
def test_get_embedding_model_reports_missing_volume(monkeypatch, tmp_path):
    embedding_model.get_embedding_model.cache_clear()
    monkeypatch.setenv("EMBEDDING_MODEL_PATH", str(tmp_path / "missing-model"))

    with pytest.raises(RuntimeError, match="model directory does not exist"):
        embedding_model.get_embedding_model()

    embedding_model.get_embedding_model.cache_clear()
