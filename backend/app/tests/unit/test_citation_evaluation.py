from types import SimpleNamespace

import pytest

from service.core.citation_evaluation import evaluate_citations


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
