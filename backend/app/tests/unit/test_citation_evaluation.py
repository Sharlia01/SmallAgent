from types import SimpleNamespace
import json

import pytest

from evaluation.citations import evaluate_citations, parse_answer_claims


class FakeCompletions:
    def create(self, **_kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            '{"assessments":['
                            '{"claim_index":1,"requires_citation":true,'
                            '"supported":true,"reason":"直接支持"},'
                            '{"claim_index":2,"requires_citation":true,'
                            '"supported":false,"reason":"没有引用"}]}'
                        )
                    )
                )
            ]
        )


@pytest.mark.unit
def test_citation_evaluation_checks_ids_and_claim_support():
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions())
    )
    result = evaluate_citations(
        "收入为10亿元。##1$$利润为2亿元。##3$$",
        [
            {
                "document_name": "报告.pdf",
                "content_with_weight": "收入为10亿元。",
            }
        ],
        judge_client=client,
        judge_model="judge",
    )

    assert result["citation_ids"] == [1, 3]
    assert result["invalid_citation_ids"] == [3]
    assert result["citation_validity"] == 0.5
    assert result["citation_precision"] == 1.0
    assert result["citation_recall"] == 0.5
    assert result["judge_source"] == "model"


@pytest.mark.unit
def test_shared_classification_skips_intro_and_keeps_missing_citation_penalty():
    answer = "财务数据如下：\n收入为10亿元。\n利润为2亿元##1$$。"
    classified = parse_answer_claims(answer)
    for claim in classified:
        claim["requires_evidence"] = claim["claim_index"] != 1
    requests = []

    def create(**kwargs):
        payload = json.loads(kwargs["messages"][1]["content"])
        requests.append(payload)
        # 即使引用模型自行给出不同分类，也沿用共享分类。
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
            "assessments": [
                {"claim_index": 2, "requires_citation": False, "supported": False, "reason": "缺少引用"},
                {"claim_index": 3, "requires_citation": True, "supported": True, "reason": "直接支持"},
            ],
        })))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    result = evaluate_citations(answer, [{"content_with_weight": "收入10亿元，利润2亿元。"}],
                                judge_client=client, judge_model="judge", classified_claims=classified)
    assert [item["claim_index"] for item in requests[0]["items"]] == [2, 3]
    assert all(item["requires_citation"] for item in requests[0]["items"])
    assert result["claims"][0]["requires_citation"] is False
    assert result["factual_claim_count"] == 2
    assert result["citation_recall"] == 0.5
    assert result["classification_source"] == "faithfulness"


@pytest.mark.unit
def test_failed_shared_classification_does_not_trigger_independent_judgement():
    def create(**kwargs):
        raise AssertionError("分类不可用时不应再次询问引用模型")
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    classified = parse_answer_claims("收入10亿元。")
    classified[0]["requires_evidence"] = None
    result = evaluate_citations("收入10亿元。", [], judge_client=client,
                                judge_model="judge", classified_claims=classified)
    assert result["judge_source"] == "fallback"
    assert "classification is unavailable" in result["judge_error"]
    assert result["citation_recall"] is None


@pytest.mark.unit
def test_all_non_factual_claims_skip_citation_model():
    def create(**kwargs):
        raise AssertionError("只有开场白时无需调用引用模型")
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    classified = parse_answer_claims("数据如下：")
    classified[0]["requires_evidence"] = False
    result = evaluate_citations("数据如下：", [], judge_client=client,
                                judge_model="judge", classified_claims=classified)
    assert result["factual_claim_count"] == 0
    assert result["citation_recall"] is None
    assert result["judge_error"] is None
