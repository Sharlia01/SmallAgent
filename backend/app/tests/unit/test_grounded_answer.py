from types import SimpleNamespace

import pytest

from service.core.grounded_answer import execute_grounded_turn
from service.core import evidence_sufficiency


class FakeCompletions:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def stream_chunk(content=None, finish_reason=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                delta=SimpleNamespace(content=content, reasoning_content=None),
            )
        ]
    )


@pytest.mark.unit
def test_grounded_turn_refuses_without_calling_answer_model():
    completions = FakeCompletions([])
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    result = execute_grounded_turn(
        index_names="1",
        question="未知指标是什么？",
        answer_client=client,
        retrieval_function=lambda *_args, **_kwargs: {
            "chunks": [],
            "evidence_sufficiency": {
                "sufficient": False,
                "source": "model",
                "missing_requirements": ["未知指标的数值"],
            },
        },
    )

    assert result.response_mode == "refuse"
    assert "可直接支持答案的知识库证据" in result.answer
    assert result.raw_retrieval["evidence_sufficiency"]["source"] == "rule"
    assert result.documents == []
    assert completions.calls == []


@pytest.mark.unit
def test_grounded_turn_uses_production_prompt_and_collects_stream(monkeypatch):
    monkeypatch.setenv("RAG_EVIDENCE_SUFFICIENCY_ENABLED", "true")
    sufficiency = FakeCompletions(SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=
        '{"sufficient":true,"reason":"证据完整","missing_requirements":[],"supporting_chunk_indices":[1]}'))]))
    monkeypatch.setattr(evidence_sufficiency, "get_evidence_sufficiency_client",
                        lambda: SimpleNamespace(chat=SimpleNamespace(completions=sufficiency)))
    stream = [
        stream_chunk("答案。##1$$"),
        stream_chunk(finish_reason="stop"),
    ]
    completions = FakeCompletions(stream)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    result = execute_grounded_turn(
        index_names="1",
        question="问题？",
        answer_client=client,
        answer_model="answer-model",
        retrieval_function=lambda *_args, **_kwargs: {
            "chunks": [
                {
                    "chunk_id": "chunk-1",
                    "doc_id": "doc-1",
                    "docnm_kwd": "制度.pdf",
                    "content_with_weight": "答案证据。",
                }
            ],
            "evidence_sufficiency": {
                "sufficient": True,
                "source": "model",
            },
        },
    )

    assert result.response_mode == "answer"
    assert result.answer == "答案。##1$$"
    assert result.documents[0]["source_type"] == "knowledge_base"
    assert "[1] 答案证据。" in result.answer_messages[-1]["content"]
    assert completions.calls[0]["stream"] is True
    assert completions.calls[0]["temperature"] == 0
    assert len(sufficiency.calls) == 1
