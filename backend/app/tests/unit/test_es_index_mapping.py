from unittest.mock import Mock

import pytest

from service.core.rag.utils.es_conn import ESConnection


# 用例功能：验证创建新 ES 索引时会应用仅包含 512 维 BGE 向量的 mapping。
# 执行步骤：
# 1. 配置测试密码，并用 Mock 替换 Elasticsearch 客户端。
# 2. 设置目标索引不存在，调用索引初始化函数。
# 3. 验证创建请求使用项目配置的 settings 和 mappings。
# 4. 验证 dense vector 模板只匹配 q_512_vec，维度为 512 且使用 cosine。
@pytest.mark.unit
def test_ensure_index_applies_dense_vector_templates(monkeypatch):
    monkeypatch.setenv("ELASTIC_PASSWORD", "test-only-password")
    connection = ESConnection()
    fake_es = Mock()
    fake_es.indices.exists.return_value = False
    connection.es = fake_es

    connection._ensure_index("1")

    fake_es.indices.create.assert_called_once_with(
        index="1",
        settings=connection.mapping["settings"],
        mappings=connection.mapping["mappings"],
    )
    dynamic_templates = connection.mapping["mappings"]["dynamic_templates"]
    vector_templates = [
        template["dense_vector"]
        for template in dynamic_templates
        if "dense_vector" in template
    ]
    assert vector_templates == [
        {
            "match": "*_512_vec",
            "mapping": {
                "type": "dense_vector",
                "index": True,
                "similarity": "cosine",
                "dims": 512,
            },
        }
    ]


# 用例功能：验证目标 ES 索引已存在时不会重复创建。
# 执行步骤：
# 1. 配置测试密码，并用 Mock 替换 Elasticsearch 客户端。
# 2. 设置目标索引已存在，调用索引初始化函数。
# 3. 验证 Elasticsearch 的 create 方法没有被调用。
@pytest.mark.unit
def test_ensure_index_keeps_existing_index(monkeypatch):
    monkeypatch.setenv("ELASTIC_PASSWORD", "test-only-password")
    connection = ESConnection()
    fake_es = Mock()
    fake_es.indices.exists.return_value = True
    connection.es = fake_es

    connection._ensure_index("1")

    fake_es.indices.create.assert_not_called()
