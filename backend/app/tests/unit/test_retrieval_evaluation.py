"""独立检索指标与拒绝指标的回归测试。"""

import json

import pytest

from evaluation.retrieval import (
    build_error_case,
    build_summary,
    evaluate_retrieval_case,
    resolve_metric_ks,
    table_row_chunk_match_score,
)


pytestmark = pytest.mark.unit


def test_metric_cutoffs_default_to_top_k_and_allow_explicit_diagnostics():
    assert resolve_metric_ks(5) == (5,)
    assert resolve_metric_ks(10) == (10,)
    assert resolve_metric_ks(7) == (7,)
    assert resolve_metric_ks(5, (1, 3, 5, 10)) == (1, 3, 5)


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


def evaluate(raw_result, *, answerable=True, top_k=5, metric_ks=None):
    return evaluate_retrieval_case(
        sample(answerable=answerable), raw_result,
        latency_ms=12, top_k=top_k, match_threshold=0.8,
        metric_ks=metric_ks,
    )


def test_structured_table_cells_match_before_rendered_content():
    evidence = {
        "document_name": "国电电力.pdf",
        "evidence_type": "table",
        "row_cells": [
            "001289.SZ", "龙源电力", "16.65", "1,392", "19.6",
        ],
        "text": "001289.SZ 龙源电力 16.65 1,392 19.6",
    }
    retrieved = {
        "docnm_kwd": "国电电力.pdf",
        "table_cells_kwd": [
            "001289.sz", "龙源电力", "16.65", "1392", "19.6",
        ],
        "content_with_weight": "文本措辞可以与gold不同",
    }

    assert table_row_chunk_match_score(evidence, retrieved) == 1.0


@pytest.mark.parametrize(("gold_label", "extracted_label"), [
    ("P/E", "市盈率（PE）"),
    ("PE 25E", "市盈率（PE）25E"),
    ("每股收益", "EPS（元）"),
    ("每股收益", "每股收益（元）"),
    ("每股红利", "DPS"),
    ("每股红利", "每股股利（元/股）"),
    ("ROE", "净资产收益率（ROE）"),
])
def test_table_field_aliases_match_chinese_english_and_unit_labels(
    gold_label,
    extracted_label,
):
    evidence = {
        "document_name": "国电电力.pdf",
        "row_cells": [gold_label, "14.30", "8.2"],
    }
    retrieved = {
        "docnm_kwd": "国电电力.pdf",
        "table_cells_kwd": [extracted_label, "14.3", "8.20"],
    }

    assert table_row_chunk_match_score(evidence, retrieved) == 1.0


def test_table_numeric_units_and_currency_scales_are_normalized():
    evidence = {
        "document_name": "国电电力.pdf",
        "row_cells": ["归母净利润", "70.50亿元", "0.40"],
    }
    retrieved = {
        "docnm_kwd": "国电电力.pdf",
        "table_cells_kwd": ["归母净利润", "7,050百万元", "0.400元"],
    }

    assert table_row_chunk_match_score(evidence, retrieved) == 1.0


def test_table_numeric_normalization_does_not_merge_incompatible_units():
    evidence = {
        "document_name": "国电电力.pdf",
        "row_cells": ["指标", "10亿元"],
    }
    retrieved = {
        "docnm_kwd": "国电电力.pdf",
        "table_cells_kwd": ["指标", "10%"],
    }

    assert table_row_chunk_match_score(evidence, retrieved) == 0.0


def test_table_field_aliases_do_not_merge_distinct_financial_metrics():
    evidence = {
        "document_name": "国电电力.pdf",
        "row_cells": ["P/E", "14.3"],
    }
    retrieved = {
        "docnm_kwd": "国电电力.pdf",
        "table_cells_kwd": ["市净率（PB）", "14.3"],
    }

    assert table_row_chunk_match_score(evidence, retrieved) == 0.0


def test_structured_table_cell_mismatch_cannot_be_hidden_by_duplicate_text():
    evidence = {
        "document_name": "国电电力.pdf",
        "evidence_type": "table",
        "row_cells": ["龙源电力", "19.6"],
        "text": "龙源电力 19.6",
    }
    retrieved = {
        "docnm_kwd": "国电电力.pdf",
        "table_cells_kwd": ["龙源电力", "18.6"],
        "content_with_weight": "龙源电力 19.6",
    }

    assert table_row_chunk_match_score(evidence, retrieved) == 0.0


def test_reliable_parent_expansion_matches_a_neighboring_table_row():
    evidence = {
        "document_name": "国电电力.pdf",
        "evidence_type": "table",
        "row_cells": ["每股收益", "0.40", "0.44", "0.49"],
    }
    retrieved = {
        "docnm_kwd": "国电电力.pdf",
        "table_cells_kwd": ["归母净利润", "7050", "7895", "8717"],
        "table_context_obj": {
            "reliable": True,
            "rows": [{
                "row_index": 8,
                "headers": ["指标", "2025E", "2026E", "2027E"],
                "cells": ["每股收益", "0.40", "0.44", "0.49"],
            }],
        },
        "content_with_weight": "扩展后的 Markdown",
    }

    assert table_row_chunk_match_score(evidence, retrieved) == 1.0


def test_unreliable_parent_expansion_cannot_override_atomic_cell_mismatch():
    evidence = {
        "document_name": "国电电力.pdf",
        "evidence_type": "table",
        "row_cells": ["每股收益", "0.40"],
    }
    retrieved = {
        "docnm_kwd": "国电电力.pdf",
        "table_cells_kwd": ["每股收益", "0.41"],
        "table_context_obj": {
            "reliable": False,
            "rows": [{"cells": ["每股收益", "0.40"]}],
        },
        "content_with_weight": "每股收益 0.40",
    }

    assert table_row_chunk_match_score(evidence, retrieved) == 0.0


def test_canonical_html_table_remains_compatible_with_legacy_gold_text():
    evidence = {
        "document_name": "国电电力.pdf",
        "evidence_type": "table",
        "text": "001289.SZ 龙源电力 16.65 1,392 19.6",
    }
    retrieved = {
        "docnm_kwd": "国电电力.pdf",
        "content_with_weight": (
            "<table><tr><th>代码</th><th>公司</th><th>股价</th>"
            "<th>总市值</th><th>PE / 25E</th></tr>"
            "<tr><td>001289.SZ</td><td>龙源电力</td><td>16.65</td>"
            "<td>1,392</td><td>19.6</td></tr></table>"
        ),
    }

    assert table_row_chunk_match_score(evidence, retrieved) == 1.0


def test_rejected_hit_keeps_pre_gate_ranking_and_serializable_evidence():
    before = [chunk("irrelevant", "其他业务信息"), chunk()]
    result = evaluate({
        "total": 0,
        "total_before_evidence_sufficiency": 20,
        "chunks_before_sufficiency": before,
        "chunks": [],
        "evidence_sufficiency": {"source": "model", "sufficient": False},
    }, metric_ks=(1, 3, 5))

    assert result["result_schema_version"] == "3.2"
    assert result["retrieval"]["metric_ks"] == [1, 3, 5]
    assert result["retrieval_metrics"]["hit_at_1"] is False
    assert result["retrieval_metrics"]["hit_at_3"] is True
    assert result["retrieval_metrics"]["hit_at_5"] is True
    assert result["retrieval_metrics"]["recall_at_5"] == 1.0
    assert result["retrieval_metrics"]["precision_at_5"] == 0.2
    assert result["retrieval_metrics"]["mrr_at_5"] == 0.5
    assert "metrics" not in result
    assert result["post_gate_metrics"]["hit_at_5"] is False
    assert result["post_gate_metrics"]["recall_at_5"] == 0.0
    assert result["post_gate_metrics"]["precision_at_5"] == 0.0
    assert result["post_gate_metrics"]["mrr_at_5"] == 0.0
    assert result["gate_metrics"]["rejected"] is True
    assert result["gate_metrics"]["pre_gate_evidence_sufficient"] is True
    assert result["gate_metrics"]["rejection_attribution"] == (
        "false_rejection_with_sufficient_evidence"
    )
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
    assert "q_512_vec" in before[1]  # 序列化过程不修改原始检索数据。
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


def test_precision_and_exact_duplicate_rate_are_reported_at_multiple_cutoffs():
    duplicate = chunk("duplicate")
    chunks = [chunk(), duplicate, chunk("noise", "其他业务信息")]
    result = evaluate({
        "chunks_before_sufficiency": chunks,
        "chunks": chunks,
        "evidence_sufficiency": {"source": "model", "sufficient": True},
    }, metric_ks=(1, 3, 5))

    metrics = result["retrieval_metrics"]
    assert metrics["precision_at_1"] == 1.0
    assert metrics["precision_at_3"] == pytest.approx(2 / 3, abs=1e-6)
    assert metrics["precision_at_5"] == 0.4
    assert metrics["duplicate_rate_at_1"] == 0.0
    assert metrics["duplicate_rate_at_3"] == pytest.approx(1 / 3, abs=1e-6)
    assert metrics["duplicate_rate_at_5"] == pytest.approx(1 / 3, abs=1e-6)


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
    assert "unanswerable_rejection_rate" not in overall


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
    # 核验器可能在读取启用开关前，先返回无证据规则判定。
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


def test_gate_summary_separates_retrieval_limited_and_false_rejections():
    sufficient_rejected = evaluate({
        "chunks_before_sufficiency": [chunk()],
        "chunks": [],
        "evidence_sufficiency": {"source": "model", "sufficient": False},
    })
    sufficient_rejected["id"] = "sufficient-rejected"

    insufficient_rejected = evaluate({
        "chunks_before_sufficiency": [chunk("noise", "其他业务信息")],
        "chunks": [],
        "evidence_sufficiency": {"source": "model", "sufficient": False},
    })
    insufficient_rejected["id"] = "insufficient-rejected"

    sufficient_accepted = evaluate({
        "chunks_before_sufficiency": [chunk()],
        "chunks": [chunk()],
        "evidence_sufficiency": {"source": "model", "sufficient": True},
    })
    sufficient_accepted["id"] = "sufficient-accepted"

    insufficient_accepted = evaluate({
        "chunks_before_sufficiency": [chunk("noise", "其他业务信息")],
        "chunks": [chunk("noise", "其他业务信息")],
        "evidence_sufficiency": {"source": "model", "sufficient": True},
    })
    insufficient_accepted["id"] = "insufficient-accepted"

    summary = build_summary([
        sufficient_rejected,
        insufficient_rejected,
        sufficient_accepted,
        insufficient_accepted,
    ], top_k=5)
    gate = summary["overall"]["gate_metrics"]

    assert sufficient_rejected["gate_metrics"]["rejection_attribution"] == (
        "false_rejection_with_sufficient_evidence"
    )
    assert insufficient_rejected["gate_metrics"]["rejection_attribution"] == (
        "correct_rejection_after_insufficient_evidence"
    )
    assert gate["evidence_sufficient_answerable_decision_count"] == 2
    assert gate["evidence_insufficient_answerable_decision_count"] == 2
    assert gate["answerable_rejected_evidence_attribution_count"] == 2
    assert (
        gate["answerable_rejected_evidence_attribution_unavailable_count"]
        == 0
    )
    assert (
        gate["evidence_attribution_unavailable_answerable_decision_count"]
        == 0
    )
    assert gate["correct_rejection_after_insufficient_evidence_count"] == 1
    assert gate["correct_rejection_after_insufficient_evidence_rate"] == 0.5
    assert gate["false_rejection_with_sufficient_evidence_count"] == 1
    assert gate["false_rejection_with_sufficient_evidence_rate"] == 0.5

    gate_ids = summary["overall"]["non_perfect_sample_ids"]["gate_metrics"]
    assert gate_ids["answerable_rejection_rate"] == [
        "sufficient-rejected",
        "insufficient-rejected",
    ]
    assert gate_ids["false_rejection_with_sufficient_evidence_rate"] == [
        "sufficient-rejected",
    ]
    assert gate_ids["correct_rejection_after_insufficient_evidence_rate"] == [
        "insufficient-accepted",
    ]


def test_stage_latencies_are_preserved_and_aggregated_independently():
    first = evaluate({
        "chunks_before_sufficiency": [chunk()],
        "chunks": [chunk()],
        "stage_latency_ms": {
            "query_transform": 10,
            "initial_retrieval": 20,
            "rerank": 30,
            "multi_hop": None,
            "gate": 40,
            "other": 5,
            "total": 105,
        },
        "evidence_sufficiency": {"source": "model", "sufficient": True},
    })
    second = evaluate({
        "chunks_before_sufficiency": [chunk()],
        "chunks": [chunk()],
        "stage_latency_ms": {
            "query_transform": 20,
            "initial_retrieval": 40,
            "rerank": 60,
            "multi_hop": 80,
            "gate": 100,
            "other": 15,
            "total": 315,
        },
        "evidence_sufficiency": {"source": "model", "sufficient": True},
    })

    assert first["retrieval"]["stage_latency_ms"]["rerank"] == 30.0
    assert first["retrieval"]["stage_latency_ms"]["multi_hop"] is None
    summary = build_summary([first, second], top_k=5)["overall"]
    stages = summary["stage_latency_ms"]
    assert stages["query_transform"] == {
        "sample_count": 2,
        "average_ms": 15.0,
        "p50_ms": 15.0,
        "p95_ms": 19.5,
    }
    assert stages["multi_hop"] == {
        "sample_count": 1,
        "average_ms": 80.0,
        "p50_ms": 80.0,
        "p95_ms": 80.0,
    }


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
    assert summary["result_schema_version"] == "3.2"
    assert summary["metric_ks"] == [5]
    assert "hit_at_1" not in overall["retrieval_metrics"]
    assert "hit_at_3" not in overall["retrieval_metrics"]
    assert overall["error_rate"] == 0.25
    assert overall["latency_p50_ms"] == 12.0
    assert overall["latency_p95_ms"] == 12.0
    assert overall["retrieval_metrics"]["hit_at_5"] == 1.0
    assert overall["post_gate_metrics"]["hit_at_5"] == 0.5
    assert "hit_at_5" not in overall
    assert overall["retrieval_metrics"]["empty_rate"] == 0.0
    assert overall["post_gate_metrics"]["empty_rate"] == pytest.approx(2 / 3, abs=1e-6)
    gate = overall["gate_metrics"]
    assert gate["decision_query_count"] == 3
    assert gate["answerable_decision_count"] == 2
    assert gate["unanswerable_decision_count"] == 1
    assert gate["answerable_rejection_rate"] == 0.5
    assert gate["unanswerable_rejection_rate"] == 1.0
    assert gate["rejected_query_count"] == 2
    assert gate["evidence_sufficient_answerable_decision_count"] == 2
    assert gate["evidence_insufficient_answerable_decision_count"] == 0
    assert gate["false_rejection_with_sufficient_evidence_count"] == 1
    assert gate["false_rejection_with_sufficient_evidence_rate"] == 0.5
    assert gate["correct_rejection_after_insufficient_evidence_count"] == 0
    assert gate["correct_rejection_after_insufficient_evidence_rate"] is None
    for group in ["retrieval_metrics", "post_gate_metrics", "gate_metrics"]:
        assert overall[group]["error_count"] == 1
        assert group in summary["by_question_type"]["fact"]
        assert group in summary["by_source_modality"]["text"]
    assert "split_counts" not in summary
    assert "by_split" not in summary
    assert summary["by_question_type"]["unanswerable"]["retrieval_metrics"]["hit_at_5"] is None
    assert failed["retrieval_metrics"]["hit_at_5"] is None
    assert failed["gate_metrics"]["rejected"] is None


def test_summary_lists_non_perfect_sample_ids_for_each_directional_metric():
    perfect = evaluate({
        "chunks_before_sufficiency": [chunk()],
        "chunks": [chunk()],
        "evidence_sufficiency": {"source": "model", "sufficient": True},
    })
    perfect["id"] = "perfect"

    missed = evaluate({
        "chunks_before_sufficiency": [chunk("noise", "其他业务信息")],
        "chunks": [chunk("noise", "其他业务信息")],
        "evidence_sufficiency": {"source": "model", "sufficient": True},
    })
    missed["id"] = "missed"

    duplicate = evaluate({
        "chunks_before_sufficiency": [chunk(), chunk("duplicate")],
        "chunks": [chunk(), chunk("duplicate")],
        "evidence_sufficiency": {"source": "model", "sufficient": True},
    })
    duplicate["id"] = "duplicate"

    unsafe_accept = evaluate({
        "chunks_before_sufficiency": [chunk("noise", "其他业务信息")],
        "chunks": [chunk("noise", "其他业务信息")],
        "evidence_sufficiency": {"source": "model", "sufficient": True},
    }, answerable=False)
    unsafe_accept["id"] = "unsafe-accept"

    summary = build_summary(
        [perfect, missed, duplicate, unsafe_accept],
        top_k=5,
    )
    sample_ids = summary["overall"]["non_perfect_sample_ids"]

    assert sample_ids["retrieval_metrics"]["hit_at_5"] == ["missed"]
    assert sample_ids["post_gate_metrics"]["hit_at_5"] == ["missed"]
    assert sample_ids["retrieval_metrics"]["duplicate_rate_at_5"] == [
        "duplicate",
    ]
    assert sample_ids["gate_metrics"]["unanswerable_rejection_rate"] == [
        "unsafe-accept",
    ]
    assert summary["by_question_type"]["fact"]["non_perfect_sample_ids"][
        "retrieval_metrics"
    ]["hit_at_5"] == ["missed"]


def test_legacy_result_rows_can_still_be_aggregated():
    legacy = evaluate({"chunks": [chunk()]})
    legacy["metrics"] = legacy["post_gate_metrics"]
    for key in ["result_schema_version", "retrieval_metrics", "post_gate_metrics", "gate_metrics"]:
        legacy.pop(key)
    legacy["retrieval"].pop("empty_before_sufficiency")
    overall = build_summary([legacy], top_k=5)["overall"]
    assert overall["post_gate_metrics"]["hit_at_5"] == 1.0
    assert "hit_at_5" not in overall
    assert overall["retrieval_metrics"]["hit_at_5"] is None
    assert overall["gate_metrics"]["rejection_rate"] is None
