from types import SimpleNamespace

import pytest

from service.core import sequential_retrieval


class FakeCompletions:
    def __init__(self, content):
        self.content = content

    def create(self, **_kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.content),
                )
            ]
        )


def fake_client(content):
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=FakeCompletions(content),
        )
    )


@pytest.mark.unit
def test_rating_bridge_is_extracted_from_the_exact_source_chunk():
    chunks = [
        {
            "chunk_id": "rating",
            "doc_id": "doc-1",
            "content_with_weight": "投资建议：维持“优于大市”评级。",
        }
    ]

    result = sequential_retrieval.extract_bridge_value(
        slot="rating",
        original_question="研报给了什么评级？",
        first_query="研报给了什么评级？",
        chunks=chunks,
        client=fake_client("must not be used"),
    )

    assert result is not None
    assert result.value == "优于大市"
    assert result.source_chunk_id == "rating"
    assert result.document_id == "doc-1"
    assert result.source == "rule"


@pytest.mark.unit
def test_model_bridge_value_must_appear_in_the_claimed_chunk():
    result = sequential_retrieval.extract_bridge_value(
        slot="policy",
        original_question="公司采用什么政策，这项政策有什么影响？",
        first_query="公司采用什么政策？",
        chunks=[
            {
                "chunk_id": "policy",
                "doc_id": "doc-1",
                "content_with_weight": "公司发布未来三年现金分红规划。",
            }
        ],
        client=fake_client(
            '{"found":true,"value":"高股息政策",'
            '"source_chunk_index":1}'
        ),
    )

    assert result is None


@pytest.mark.unit
def test_query_template_rejects_unknown_dependency_slot():
    with pytest.raises(ValueError):
        sequential_retrieval.resolve_query_template(
            "{unknown}的定义是什么？",
            {"rating": "优于大市"},
        )
