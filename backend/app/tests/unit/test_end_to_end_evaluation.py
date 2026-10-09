import pytest

from service.core.end_to_end_evaluation import (
    aggregate_e2e_results,
    build_e2e_summary,
    evaluate_refusal,
    evaluate_sufficiency,
)
from service.core.retrieval_evaluation import (
    build_error_case,
    build_summary,
    evaluate_retrieval_case,
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
            "id": "unsafe-answer",
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
            "id": "over-refusal",
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
    sample_ids = summary["non_perfect_sample_ids"]
    assert sample_ids["faithfulness"] == ["unsafe-answer"]
    assert sample_ids["faithfulness_micro"] == ["unsafe-answer"]
    assert sample_ids["refusal_metrics"]["refusal_precision"] == [
        "over-refusal",
    ]
    assert sample_ids["refusal_metrics"]["refusal_recall"] == [
        "unsafe-answer",
    ]
    assert sample_ids["sufficiency_metrics"]["accuracy"] == [
        "unsafe-answer",
        "over-refusal",
    ]

    results[0]["split"] = "test"
    results[1]["split"] = "challenge"
    sliced = build_e2e_summary(results)
    assert "split_counts" not in sliced
    assert "by_split" not in sliced


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


def _sample(identifier, *, evidence_texts=("收入100亿元",), split="dev"):
    return {
        "id": identifier,
        "question": "报告中的业绩是多少？",
        "answerable": bool(evidence_texts),
        "question_type": "fact" if evidence_texts else "unanswerable",
        "relevant_evidence": [
            {"document_name": "报告.pdf", "text": text}
            for text in evidence_texts
        ],
        "metadata": {"split": split},
    }


def _chunk(identifier="gold", text="收入100亿元"):
    return {
        "chunk_id": identifier,
        "docnm_kwd": "报告.pdf",
        "content_with_weight": text,
    }


def _score(sample, raw_result, *, top_k=5):
    return evaluate_retrieval_case(
        sample, raw_result, latency_ms=10.0, top_k=top_k, match_threshold=1.0,
    )


def _as_e2e_result(case, response_mode="answer"):
    return {
        **case,
        "latency_ms": {"retrieval": case["latency_ms"], "total": 20.0},
        "response_mode": response_mode,
        "answer_metrics": {},
        "faithfulness_metrics": {},
        "citation_metrics": {},
        "refusal_metrics": evaluate_refusal(case, response_mode),
        "sufficiency_metrics": evaluate_sufficiency(case, case),
    }


@pytest.mark.unit
@pytest.mark.parametrize("top_k", [5, 10])
def test_e2e_summary_aggregates_retrieval_scores_and_slices(top_k):
    partial = _score(
        _sample("partial", evidence_texts=("收入100亿元", "净利润20亿元")),
        {
            "chunks_before_sufficiency": [_chunk()],
            "chunks": [_chunk()],
            "evidence_sufficiency": {"source": "model", "sufficient": True},
        },
        top_k=top_k,
    )
    rejected = _score(
        {**_sample("rejected", split="test"), "question_type": "paraphrase"},
        {
            "chunks_before_sufficiency": [_chunk("noise", "无关信息"), _chunk()],
            "chunks": [],
            "evidence_sufficiency": {"source": "model", "sufficient": False},
        },
        top_k=top_k,
    )
    unanswerable = _score(
        _sample("unanswerable", evidence_texts=()),
        {
            "chunks_before_sufficiency": [],
            "chunks": [],
            "evidence_sufficiency": {"source": "rule", "sufficient": False},
        },
        top_k=top_k,
    )
    failed = build_error_case(
        _sample("failed", split="test"), error=RuntimeError("retrieval failed"),
        latency_ms=5.0, top_k=top_k,
    )
    cases = [partial, rejected, unanswerable, failed]
    results = [
        _as_e2e_result(partial),
        _as_e2e_result(rejected, "refuse"),
        _as_e2e_result(unanswerable, "refuse"),
        _as_e2e_result(failed, "error"),
    ]

    summary = build_e2e_summary(results, top_k=top_k)
    overall = summary["overall"]
    before = overall["retrieval_metrics"]
    after = overall["post_gate_metrics"]
    assert summary["result_schema_version"] == "1.2"
    assert summary["metric_ks"] == [top_k]
    assert before["query_count"] == 4
    assert before["evaluated_query_count"] == 3
    assert before["answerable_query_count"] == 2
    assert before["unanswerable_query_count"] == 1
    assert before["error_count"] == 1
    assert before["error_rate"] == 0.25
    assert before[f"hit_at_{top_k}"] == 1.0
    assert before[f"recall_at_{top_k}"] == 0.75
    assert before[f"mrr_at_{top_k}"] == 0.75
    assert before[f"precision_at_{top_k}"] == pytest.approx(1 / top_k)
    assert before[f"duplicate_rate_at_{top_k}"] == 0.0
    assert before[f"all_requirements_hit_at_{top_k}"] is None
    assert after[f"hit_at_{top_k}"] == 0.5
    assert after[f"recall_at_{top_k}"] == 0.25
    assert after[f"mrr_at_{top_k}"] == 0.5
    assert after[f"precision_at_{top_k}"] == pytest.approx(1 / (2 * top_k))
    assert overall["gate_metrics"]["false_rejection_with_sufficient_evidence_count"] == 1
    assert overall["refusal_metrics"]["over_refusal_count"] == 1
    assert "split_counts" not in summary
    assert "by_split" not in summary
    assert summary["by_question_type"]["paraphrase"]["retrieval_metrics"][f"mrr_at_{top_k}"] == 0.5
    assert summary["by_question_type"]["unanswerable"]["retrieval_metrics"][f"recall_at_{top_k}"] is None
    if top_k == 10:
        assert "recall_at_5" not in before
        assert "mrr_at_5" not in summary["by_question_type"]["paraphrase"]["retrieval_metrics"]

    failures = overall["non_perfect_sample_ids"]
    assert failures["error_rate"] == ["failed"]
    assert failures["retrieval_metrics"][f"recall_at_{top_k}"] == ["partial"]
    assert failures["retrieval_metrics"][f"mrr_at_{top_k}"] == ["rejected"]
    assert failures["post_gate_metrics"][f"mrr_at_{top_k}"] == ["rejected"]
    assert failures["refusal_metrics"]["over_refusal_rate"] == ["rejected"]

    # 两个评测执行器必须使用相同的分母、得分和诊断规则。
    retrieval_summary = build_summary(cases, top_k=top_k)
    pairs = [(overall, retrieval_summary["overall"])]
    pairs.extend(
        (values, retrieval_summary["by_question_type"][name])
        for name, values in summary["by_question_type"].items()
    )
    for e2e, retrieval in pairs:
        for group in ("retrieval_metrics", "post_gate_metrics", "gate_metrics"):
            assert e2e[group] == retrieval[group]
            assert e2e["non_perfect_sample_ids"][group] == retrieval["non_perfect_sample_ids"][group]


@pytest.mark.unit
def test_e2e_summary_keeps_missing_stage_data_unavailable():
    legacy = _as_e2e_result(_score(_sample("legacy"), {"chunks": [_chunk()]}))
    no_retrieval = _as_e2e_result(_score(_sample("no-retrieval"), {}))
    for key in ("retrieval", "retrieval_metrics", "post_gate_metrics", "gate_metrics"):
        no_retrieval.pop(key)

    overall = aggregate_e2e_results([legacy, no_retrieval])
    before = overall["retrieval_metrics"]
    after = overall["post_gate_metrics"]
    assert before["evaluated_query_count"] == 0
    assert before["unavailable_query_count"] == 2
    assert before["recall_at_5"] is None
    assert before["mrr_at_5"] is None
    assert after["evaluated_query_count"] == 1
    assert after["unavailable_query_count"] == 1
    assert after["recall_at_5"] == 1.0
    assert overall["non_perfect_sample_ids"]["retrieval_metrics"]["recall_at_5"] == []


@pytest.mark.unit
@pytest.mark.parametrize("cohort", ["empty", "unanswerable", "error"])
def test_e2e_summary_without_scored_answerable_queries_has_null_metrics(cohort):
    results = []
    if cohort == "unanswerable":
        results.append(_as_e2e_result(_score(
            _sample("unanswerable", evidence_texts=()),
            {"chunks_before_sufficiency": [], "chunks": []},
        ), "refuse"))
    elif cohort == "error":
        results.append(_as_e2e_result(build_error_case(
            _sample("failed"), error=RuntimeError("failed"), latency_ms=10, top_k=5,
        ), "error"))
    overall = build_e2e_summary(results)["overall"]
    for group in ("retrieval_metrics", "post_gate_metrics"):
        assert overall[group]["answerable_query_count"] == 0
        assert overall[group]["error_count"] == int(cohort == "error")
        for metric in ("recall", "mrr", "hit", "precision", "all_requirements_hit"):
            assert overall[group][f"{metric}_at_5"] is None


@pytest.mark.unit
def test_e2e_summary_preserves_multi_evidence_completion_rank():
    sample = _sample("multi", evidence_texts=("收入100亿元", "净利润20亿元"))
    sample["question_type"] = "parallel_multi_evidence"
    sample["evidence_requirements"] = [
        {"id": f"r{i}", "alternatives": [[{"type": "text", **evidence}]]}
        for i, evidence in enumerate(sample["relevant_evidence"])
    ]
    chunks = [_chunk(), _chunk("noise", "无关信息"), _chunk("profit", "净利润20亿元")]
    result = _as_e2e_result(_score(sample, {
        "chunks_before_sufficiency": chunks,
        "chunks": chunks,
        "evidence_sufficiency": {"source": "model", "sufficient": True},
    }))
    summary = build_e2e_summary([result])
    for group in ("retrieval_metrics", "post_gate_metrics"):
        metrics = summary["overall"][group]
        assert metrics["recall_at_5"] == 1.0
        assert metrics["hit_at_5"] == 1.0
        assert metrics["all_requirements_hit_at_5"] == 1.0
        assert metrics["mrr_at_5"] == pytest.approx(1 / 3, abs=1e-6)
