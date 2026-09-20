import json
from types import SimpleNamespace

import pytest

from service.core.faithfulness_evaluation import evaluate_faithfulness


class FakeCompletions:
    def __init__(self, content: str):
        self.content = content
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.content),
                )
            ]
        )


def _client_with_response(content: str):
    completions = FakeCompletions(content)
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
    )
    return client, completions


@pytest.mark.unit
def test_faithfulness_scores_all_claims_against_all_documents():
    client, completions = _client_with_response(
        '{"assessments":['
        '{"claim_index":1,"requires_evidence":true,"supported":true,'
        '"supporting_document_indices":[1],"reason":"收入被直接支持"},'
        '{"claim_index":2,"requires_evidence":true,"supported":false,'
        '"supporting_document_indices":[],"reason":"证据没有利润数据"}]}'
    )

    result = evaluate_faithfulness(
        question="收入和利润是多少？",
        answer="收入为10亿元。利润为2亿元。",
        documents=[
            {
                "document_name": "报告.pdf",
                "content_with_weight": "公司收入为10亿元。",
            }
        ],
        response_mode="answer",
        judge_client=client,
        judge_model="judge",
    )

    assert result["faithfulness_score"] == 0.5
    assert result["factual_claim_count"] == 2
    assert result["supported_claim_count"] == 1
    assert result["unsupported_claim_count"] == 1
    assert result["unsupported_claims"] == [
        {
            "claim_index": 2,
            "claim": "利润为2亿元。",
            "reason": "证据没有利润数据",
        }
    ]
    request_payload = json.loads(completions.kwargs["messages"][1]["content"])
    assert request_payload["claims"][0] == {
        "claim_index": 1,
        "claim": "收入为10亿元。",
    }
    assert request_payload["retrieved_evidence"][0]["content"] == (
        "公司收入为10亿元。"
    )
    assert "citation_ids" not in request_payload["claims"][0]


@pytest.mark.unit
def test_faithfulness_does_not_require_inline_citations():
    client, _completions = _client_with_response(
        '{"assessments":['
        '{"claim_index":1,"requires_evidence":true,"supported":true,'
        '"supporting_document_indices":[1],"reason":"完整支持"}]}'
    )

    result = evaluate_faithfulness(
        question="收入是多少？",
        answer="收入为10亿元。",
        documents=[{"content_with_weight": "收入为10亿元。"}],
        response_mode="answer",
        judge_client=client,
        judge_model="judge",
    )

    assert result["faithfulness_score"] == 1.0
    assert result["claims"][0]["citation_ids"] == []


@pytest.mark.unit
def test_faithfulness_is_null_when_answer_has_no_factual_claims():
    client, _completions = _client_with_response(
        '{"assessments":['
        '{"claim_index":1,"requires_evidence":false,"supported":false,'
        '"supporting_document_indices":[],"reason":"这是建议"}]}'
    )

    result = evaluate_faithfulness(
        question="我该怎么办？",
        answer="建议咨询专业人员。",
        documents=[],
        response_mode="answer",
        judge_client=client,
        judge_model="judge",
    )

    assert result["faithfulness_score"] is None
    assert result["factual_claim_count"] == 0
    assert result["judge_source"] == "model"


@pytest.mark.unit
def test_faithfulness_is_null_for_refusal_and_disabled_judge():
    class FailingCompletions:
        def create(self, **_kwargs):
            raise AssertionError("refusal must not call the judge")

    client = SimpleNamespace(
        chat=SimpleNamespace(completions=FailingCompletions())
    )
    refusal = evaluate_faithfulness(
        question="未知问题",
        answer="当前资料不足。",
        documents=[],
        response_mode="refuse",
        judge_client=client,
        judge_model="judge",
    )
    disabled = evaluate_faithfulness(
        question="收入是多少？",
        answer="收入为10亿元。",
        documents=[{"content_with_weight": "收入为10亿元。"}],
        response_mode="answer",
    )

    assert refusal["faithfulness_score"] is None
    assert refusal["judge_source"] == "skipped_refusal"
    assert disabled["faithfulness_score"] is None
    assert disabled["judge_source"] == "disabled"
    assert disabled["claim_count"] == 1
    assert disabled["claims"][0]["supported"] is None


@pytest.mark.unit
def test_faithfulness_returns_null_when_judge_response_is_invalid():
    client, _completions = _client_with_response(
        '{"assessments":['
        '{"claim_index":1,"requires_evidence":true,"supported":true,'
        '"supporting_document_indices":[9],"reason":"索引越界"}]}'
    )

    result = evaluate_faithfulness(
        question="收入是多少？",
        answer="收入为10亿元。",
        documents=[{"content_with_weight": "收入为10亿元。"}],
        response_mode="answer",
        judge_client=client,
        judge_model="judge",
    )

    assert result["faithfulness_score"] is None
    assert result["judge_source"] == "fallback"
    assert "indices are invalid" in result["judge_error"]
