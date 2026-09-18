"""Regression tests for independent retrieval and rejection metrics."""

import json

import pytest

from service.core.retrieval_evaluation import (
    build_error_case,
    build_summary,
    evaluate_retrieval_case,
)


pytestmark = pytest.mark.unit


def sample(*, answerable=True):
    return {
        "id": "answerable" if answerable else "unanswerable",
        "question": "2026E固定资产是多少？",
        "reference_answer": "467602百万元" if answerable else "未提供",
        "answerable": answerable,
        "question_type": "fact" if answerable else "unanswerable",
        "relevant_evidence": [
            {"document_name": "report.pdf", "text": "固定资产467602百万元"}
        ] if answerable else [],
        "metadata": {},
    }


def chunk(identifier="gold", content="固定资产467602百万元"):
    return {
        "chunk_id": identifier,
        "docnm_kwd": "report.pdf",
        "content_with_weight": content,
        "q_512_vec": [0.1] * 512,
        "retrieval_ranks": {"vector_original": 2},
        "final_ranking": {"semantic_reranker_rank": 1},
    }


def evaluate(raw_result, *, answerable=True, top_k=5):
    return evaluate_retrieval_case(
        sample(answerable=answerable), raw_result,
        latency_ms=12, top_k=top_k, match_threshold=0.8,
    )


def test_rejected_hit_keeps_pre_gate_ranking_and_serializable_evidence():
    before = [chunk("irrelevant", "其他业务信息"), chunk()]
    result = evaluate({
        "total": 0,
        "total_before_evidence_sufficiency": 20,
        "chunks_before_sufficiency": before,
        "chunks": [],
        "evidence_sufficiency": {"source": "model", "sufficient": False},
    })

    assert result["result_schema_version"] == "2.0"
    assert result["retrieval_metrics"] == {
        "hit_at_5": True, "recall_at_5": 1.0, "mrr_at_5": 0.5,
    }
    assert result["post_gate_metrics"] == result["metrics"] == {
        "hit_at_5": False, "recall_at_5": 0.0, "mrr_at_5": 0.0,
    }
    assert result["gate_metrics"]["rejected"] is True
    saved = result["retrieval"]
    assert saved["retrieved_count_before_sufficiency"] == 2
    assert saved["empty_before_sufficiency"] is False
    assert saved["retrieved_count"] == 0
    assert saved["total_before_evidence_sufficiency"] == 20
    assert [c["rank"] for c in saved["chunks_before_sufficiency"]] == [1, 2]
    assert saved["chunks_before_sufficiency"][1]["retrieval_ranks"] == {
        "vector_original": 2,
    }
    assert "q_512_vec" not in json.dumps(result)
    assert "q_512_vec" in before[1]  # Serialization does not mutate retrieval data.
    assert result["evidence_matches"][0]["matched"] is False
    assert result["evidence_matches_before_sufficiency"][0]["matched_rank"] == 2


@pytest.mark.parametrize("sufficient", [True, False])
def test_top_k_is_applied_before_scoring_both_stages(sufficient):
    chunks = [chunk("irrelevant", "其他业务信息"), chunk()]
    result = evaluate({
        "chunks_before_sufficiency": chunks,
        "chunks": chunks if sufficient else [],
        "evidence_sufficiency": {"source": "model", "sufficient": sufficient},
    }, top_k=1)
    assert result["retrieval_metrics"]["hit_at_1"] is False
    assert result["post_gate_metrics"]["hit_at_1"] is False
    assert len(result["retrieval"]["chunks_before_sufficiency"]) == 1


@pytest.mark.parametrize("sufficient", [True, False])
def test_fallback_outcome_and_failure_are_both_recorded(sufficient):
    chunks = [chunk()]
    result = evaluate({
        "chunks_before_sufficiency": chunks,
        "chunks": chunks if sufficient else [],
        "evidence_sufficiency": {"source": "fallback", "sufficient": sufficient},
    })
    overall = build_summary([result], top_k=5)["overall"]
    assert overall["retrieval_metrics"]["hit_at_5"] == 1.0
    assert overall["post_gate_metrics"]["hit_at_5"] == float(sufficient)
    assert overall["gate_metrics"]["answerable_rejection_rate"] == float(not sufficient)
    assert overall["gate_metrics"]["fallback_rate"] == 1.0


def test_empty_retrieval_without_a_decision_is_not_counted_as_rejection():
    result = evaluate({
        "chunks_before_sufficiency": [], "chunks": [],
    }, answerable=False)
    overall = build_summary([result], top_k=5)["overall"]
    assert result["retrieval_metrics"]["hit_at_5"] is None
    assert result["gate_metrics"]["rejected"] is None
    assert overall["retrieval_metrics"]["empty_rate"] == 1.0
    assert overall["gate_metrics"]["unavailable_query_count"] == 1
    assert overall["gate_metrics"]["unanswerable_rejection_rate"] is None
    # The legacy field intentionally retains its old empty-result definition.
    assert overall["unanswerable_rejection_rate"] == 1.0


def test_no_evidence_rule_is_an_explicit_rejection():
    result = evaluate({
        "chunks_before_sufficiency": [], "chunks": [],
        "evidence_sufficiency": {"source": "rule", "sufficient": False},
    }, answerable=False)
    overall = build_summary([result], top_k=5)["overall"]
    assert overall["gate_metrics"]["unanswerable_rejection_rate"] == 1.0
    assert overall["gate_metrics"]["fallback_rate"] == 0.0


@pytest.mark.parametrize("chunks", [[], [chunk()]])
def test_missing_historical_snapshot_is_unknown_even_when_output_is_nonempty(chunks):
    result = evaluate({"chunks": chunks})
    overall = build_summary([result], top_k=5)["overall"]
    assert result["retrieval"]["chunks_before_sufficiency"] is None
    assert result["retrieval_metrics"]["hit_at_5"] is None
    assert overall["retrieval_metrics"]["evaluated_query_count"] == 0
    assert overall["retrieval_metrics"]["unavailable_query_count"] == 1
    assert overall["retrieval_metrics"]["empty_rate"] is None
    assert overall["post_gate_metrics"]["hit_at_5"] == float(bool(chunks))


@pytest.mark.parametrize("decision", [
    {"source": "disabled", "sufficient": True},
    # The checker can return its no-evidence rule before reading its enable flag.
    {"source": "rule", "sufficient": False, "enabled": False},
])
def test_disabled_checks_are_not_counted_as_successful_gate_decisions(decision):
    chunks = [] if decision["source"] == "rule" else [chunk()]
    result = evaluate({
        "chunks_before_sufficiency": chunks, "chunks": chunks,
        "evidence_sufficiency": decision,
    })
    overall = build_summary([result], top_k=5)["overall"]
    assert result["retrieval_metrics"] == result["post_gate_metrics"]
    assert result["gate_metrics"]["rejected"] is None
    assert overall["gate_metrics"]["disabled_query_count"] == 1
    assert overall["gate_metrics"]["decision_query_count"] == 0
    assert overall["gate_metrics"]["answerable_rejection_rate"] is None
    assert overall["gate_metrics"]["fallback_rate"] is None


def test_summary_keeps_correct_denominators_and_all_slices():
    chunks = [chunk()]
    accepted = evaluate({
        "chunks_before_sufficiency": chunks, "chunks": chunks,
        "evidence_sufficiency": {"source": "model", "sufficient": True},
    })
    rejected = evaluate({
        "chunks_before_sufficiency": chunks, "chunks": [],
        "evidence_sufficiency": {"source": "model", "sufficient": False},
    })
    unanswerable = evaluate({
        "chunks_before_sufficiency": chunks, "chunks": [],
        "evidence_sufficiency": {"source": "model", "sufficient": False},
    }, answerable=False)
    failed = build_error_case(
        sample(), error=RuntimeError("retrieval unavailable"), latency_ms=10, top_k=5,
    )
    summary = build_summary([accepted, rejected, unanswerable, failed], top_k=5)
    overall = summary["overall"]
    assert summary["result_schema_version"] == "2.0"
    assert overall["retrieval_metrics"]["hit_at_5"] == 1.0
    assert overall["post_gate_metrics"]["hit_at_5"] == overall["hit_at_5"] == 0.5
    assert overall["retrieval_metrics"]["empty_rate"] == 0.0
    assert overall["post_gate_metrics"]["empty_rate"] == pytest.approx(2 / 3, abs=1e-6)
    gate = overall["gate_metrics"]
    assert gate["decision_query_count"] == 3
    assert gate["answerable_decision_count"] == 2
    assert gate["unanswerable_decision_count"] == 1
    assert gate["answerable_rejection_rate"] == 0.5
    assert gate["unanswerable_rejection_rate"] == 1.0
    assert gate["rejected_query_count"] == 2
    for group in ["retrieval_metrics", "post_gate_metrics", "gate_metrics"]:
        assert overall[group]["error_count"] == 1
        assert group in summary["by_question_type"]["fact"]
        assert group in summary["by_source_modality"]["text"]
    assert summary["by_question_type"]["unanswerable"]["retrieval_metrics"]["hit_at_5"] is None
    assert failed["retrieval_metrics"]["hit_at_5"] is None
    assert failed["gate_metrics"]["rejected"] is None


def test_legacy_result_rows_can_still_be_aggregated():
    legacy = evaluate({"chunks": [chunk()]})
    for key in ["result_schema_version", "retrieval_metrics", "post_gate_metrics", "gate_metrics"]:
        legacy.pop(key)
    legacy["retrieval"].pop("empty_before_sufficiency")
    overall = build_summary([legacy], top_k=5)["overall"]
    assert overall["hit_at_5"] == overall["post_gate_metrics"]["hit_at_5"] == 1.0
    assert overall["retrieval_metrics"]["hit_at_5"] is None
    assert overall["gate_metrics"]["rejection_rate"] is None
