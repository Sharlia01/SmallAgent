from unittest.mock import Mock

import pytest

from service.core.rag.utils.es_conn import ESConnection


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


@pytest.mark.unit
def test_ensure_index_keeps_existing_index(monkeypatch):
    monkeypatch.setenv("ELASTIC_PASSWORD", "test-only-password")
    connection = ESConnection()
    fake_es = Mock()
    fake_es.indices.exists.return_value = True
    connection.es = fake_es

    connection._ensure_index("1")

    fake_es.indices.create.assert_not_called()
