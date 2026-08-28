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


# 用例功能：验证本地 BGE Embedding 的维度和 ES 向量字段契约固定为 512 维。
# 执行步骤：
# 1. 读取 Embedding 维度常量。
# 2. 读取 Elasticsearch 向量字段常量。
# 3. 验证维度为 512，字段名为 q_512_vec。
@pytest.mark.unit
def test_bge_embedding_contract_is_fixed_to_512_dimensions():
    assert embedding_model.EMBEDDING_DIMENSION == 512
    assert embedding_model.EMBEDDING_VECTOR_FIELD == "q_512_vec"


# 用例功能：验证本地 BGE 能按约定处理单条和批量文本，并传递正确的推理参数。
# 执行步骤：
# 1. 用可记录调用的伪 SentenceTransformer 替换真实模型。
# 2. 通过环境变量设置默认批大小，并生成一条文本的向量。
# 3. 使用显式批大小生成两条文本的向量。
# 4. 验证返回数量、向量维度、批大小和归一化等参数。
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


# 用例功能：验证 Embedding 推理结果不是 512 维时会立即报错。
# 执行步骤：
# 1. 用返回 1024 维向量的伪模型替换本地 BGE。
# 2. 调用 Embedding 生成函数处理测试文本。
# 3. 验证函数抛出 RuntimeError，且错误信息指出向量形状异常。
@pytest.mark.unit
def test_generate_embedding_rejects_unexpected_dimension(monkeypatch):
    monkeypatch.setattr(
        embedding_model,
        "get_embedding_model",
        lambda: FakeSentenceTransformer(dimension=1024),
    )

    with pytest.raises(RuntimeError, match="unexpected embedding shape"):
        embedding_model.generate_embedding("测试")


# 用例功能：验证本地 BGE 模型目录缺失时会返回明确的配置错误。
# 执行步骤：
# 1. 清空模型加载缓存，并将模型路径指向不存在的临时目录。
# 2. 调用本地模型加载函数。
# 3. 验证函数抛出 RuntimeError，且错误信息提示模型目录不存在。
# 4. 再次清空缓存，避免影响其他测试。
@pytest.mark.unit
def test_get_embedding_model_reports_missing_volume(monkeypatch, tmp_path):
    embedding_model.get_embedding_model.cache_clear()
    monkeypatch.setenv("EMBEDDING_MODEL_PATH", str(tmp_path / "missing-model"))

    with pytest.raises(RuntimeError, match="model directory does not exist"):
        embedding_model.get_embedding_model()

    embedding_model.get_embedding_model.cache_clear()
