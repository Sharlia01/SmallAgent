import pytest

from service.core.end_to_end_evaluation import (
    aggregate_e2e_results,
    build_e2e_summary,
    evaluate_refusal,
    evaluate_sufficiency,
)


@pytest.mark.unit
def test_refusal_and_sufficiency_use_explicit_execution_state():
    sample = {"answerable": False, "expected_behavior": "refuse"}
    refusal = evaluate_refusal(sample, "answer")
    sufficiency = evaluate_sufficiency(
        sample,
        {
            "evidence_matches_before_sufficiency": [],
            "retrieval": {
                "evidence_sufficiency": {"sufficient": True},
            },
        },
    )

    assert refusal["unsafe_answer"] is True
    assert sufficiency["unsafe_accept"] is True


@pytest.mark.unit
def test_e2e_aggregation_reports_unsafe_answers_and_over_refusals():
    base = {
        "error": None,
        "question_type": "fact",
        "latency_ms": {"total": 10.0},
        "answer_metrics": {
            "correctness_score": None,
            "completeness_score": None,
        },
        "citation_metrics": {
            "citation_validity": None,
            "citation_precision": None,
            "citation_recall": None,
        },
    }
    results = [
        {
            **base,
            "response_mode": "answer",
            "faithfulness_metrics": {
                "faithfulness_score": 0.5,
                "factual_claim_count": 2,
                "supported_claim_count": 1,
                "unsupported_claim_count": 1,
                "judge_source": "model",
            },
            "refusal_metrics": {
                "expected_behavior": "refuse",
                "actual_behavior": "answer",
                "unsafe_answer": True,
                "over_refusal": False,
            },
            "sufficiency_metrics": {
                "expected_sufficient": False,
                "actual_sufficient": True,
                "correct": False,
                "unsafe_accept": True,
                "false_rejection": False,
            },
        },
        {
            **base,
            "response_mode": "refuse",
            "faithfulness_metrics": {
                "faithfulness_score": None,
                "factual_claim_count": None,
                "supported_claim_count": None,
                "unsupported_claim_count": None,
                "judge_source": "skipped_refusal",
            },
            "refusal_metrics": {
                "expected_behavior": "answer",
                "actual_behavior": "refuse",
                "unsafe_answer": False,
                "over_refusal": True,
            },
            "sufficiency_metrics": {
                "expected_sufficient": True,
                "actual_sufficient": False,
                "correct": False,
                "unsafe_accept": False,
                "false_rejection": True,
            },
        },
    ]

    summary = aggregate_e2e_results(results)

    assert summary["refusal_metrics"]["unsafe_answer_rate"] == 1.0
    assert summary["refusal_metrics"]["over_refusal_rate"] == 1.0
    assert summary["faithfulness"] == 0.5
    assert summary["faithfulness_answer_count"] == 1
    assert summary["faithfulness_evaluated_count"] == 1
    assert summary["faithfulness_coverage"] == 1.0
    assert summary["faithfulness_judge_failure_count"] == 0
    assert summary["faithfulness_micro"] == 0.5
    assert summary["faithfulness_unsupported_claim_count"] == 1
    assert summary["sufficiency_metrics"]["unsafe_accept_count"] == 1
    assert summary["sufficiency_metrics"]["false_rejection_count"] == 1

    results[0]["split"] = "test"
    results[1]["split"] = "challenge"
    sliced = build_e2e_summary(results)
    assert sliced["split_counts"] == {"challenge": 1, "test": 1}
    assert sliced["by_split"]["test"]["refusal_metrics"][
        "unsafe_answer_count"
    ] == 1
    assert sliced["by_split"]["challenge"]["refusal_metrics"][
        "over_refusal_count"
    ] == 1


@pytest.mark.unit
def test_unread_figure_match_is_relevant_but_not_sufficient():
    result = evaluate_sufficiency(
        {"answerable": True},
        {
            "evidence_matches_before_sufficiency": [
                {
                    "matched": True,
                    "matched_chunk_id": "figure-1",
                    "evidence_type": "figure",
                }
            ],
            "retrieval": {
                "chunks_before_sufficiency": [
                    {
                        "chunk_id": "figure-1",
                        "evidence_type_kwd": "figure",
                        "visual_status_kwd": "disabled",
                    }
                ],
                "evidence_sufficiency": {"sufficient": False},
            },
        },
    )

    assert result["expected_sufficient"] is False
    assert result["correct"] is True
