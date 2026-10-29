"""Agent 全量集合、每次检索排名和真实回答正文不能混用。"""
import pytest

from evaluation.retrieval import evaluate_retrieval_case, aggregate_agent_evidence
from evaluation.end_to_end import evaluate_sufficiency

pytestmark = pytest.mark.unit


def chunk(identifier, text):
    return {"chunk_id": identifier, "document_name": "报告.pdf", "content_with_weight": text,
            "q_512_vec": [0.1, 0.2]}


def sample():
    return {"id": "complete", "answerable": True,
            "relevant_evidence": [{"document_name": "报告.pdf", "text": "培训每年一次"}]}


def score(raw):
    return evaluate_retrieval_case(sample(), raw, latency_ms=1, top_k=5, match_threshold=1)


def test_full_answer_set_is_not_truncated_and_has_no_ranking_metrics():
    chunks = [chunk(str(i), f"无关信息{i}") for i in range(6)] + [chunk("gold", "培训每年一次")]
    case = score({"evaluation_scope": "agent_evidence_sets", "chunks": chunks,
                  "chunks_before_sufficiency": chunks, "retrieval_calls": [],
                  "evidence_sufficiency": {"source": "rule", "sufficient": True}})
    assert len(case["retrieval"]["chunks"]) == 7
    assert len(case["retrieval"]["chunks_before_sufficiency"]) == 7
    assert case["answer_evidence_metrics"]["recall"] == 1
    assert case["answer_evidence_metrics"]["precision"] == pytest.approx(1 / 7)
    assert not any("mrr" in key or "_at_" in key for key in case["answer_evidence_metrics"])
    assert all("q_512_vec" not in item for item in case["retrieval"]["chunks"])
    assert evaluate_sufficiency(sample(), case)["unsafe_accept"] is False

    # 单次纯检索仍然按 K 截断，避免改变已有搜索基线的口径。
    ranked = score({"chunks": chunks, "chunks_before_sufficiency": chunks})
    assert len(ranked["retrieval"]["chunks"]) == 5
    assert ranked["retrieval_metrics"]["recall_at_5"] == 0


def test_missing_verification_snapshot_is_unknown_not_empty():
    case = score({"evaluation_scope": "agent_evidence_sets", "chunks": [], "retrieval_calls": []})
    assert case["verification_evidence_metrics"] is None
    assert case["evidence_matches_before_sufficiency"] is None
    assert evaluate_sufficiency(sample(), case)["expected_sufficient"] is None
    summary = aggregate_agent_evidence([case])
    assert summary["verification_evidence_metrics"]["unavailable_query_count"] == 1
    assert summary["answer_evidence_metrics"]["recall"] == 0
    assert summary["retrieval_call_metrics"]["recorded_query_count"] == 1
    assert summary["retrieval_call_metrics"]["call_count"] == 0


def test_call_rankings_keep_independent_cutoffs_errors_and_no_vectors():
    gold = chunk("gold", "培训每年一次")
    case = score({"evaluation_scope": "agent_evidence_sets", "chunks": [gold],
                  "chunks_before_sufficiency": [gold], "retrieval_calls": [
                      {"call_id": 1, "query": "首次查询", "top_k": 1, "latency_ms": 1,
                       "result": {"chunks": [chunk("noise", "无关"), gold]}},
                      {"call_id": 2, "query": "补查", "top_k": 5, "latency_ms": 2,
                       "result": {"chunks": [gold]}},
                      {"call_id": 3, "query": "失败查询", "top_k": 5, "latency_ms": 3,
                       "error": "TimeoutError", "result": None},
                  ]})
    calls = case["retrieval"]["calls"]
    assert calls[0]["retrieval_metrics"]["hit_at_1"] is False
    assert calls[1]["retrieval_metrics"]["mrr_at_5"] == 1
    assert "q_512_vec" not in calls[1]["retrieval"]["chunks"][0]
    summary = aggregate_agent_evidence([case])["retrieval_call_metrics"]
    assert summary["call_count"] == 3
    assert summary["by_top_k"]["5"]["error_count"] == 1
    assert summary["by_top_k"]["5"]["evaluated_query_count"] == 1


def test_projected_answer_cannot_match_hidden_source_table_cells():
    table_sample = {"id": "table", "answerable": True, "relevant_evidence": [
        {"document_name": "报告.pdf", "evidence_type": "table", "row_cells": ["收入", "100"], "text": "收入100"}]}
    source = {**chunk("table", "收入100"), "table_cells_slice_obj": {"rows": [{"cells": ["收入", "100"]}]}}
    projected = {**source, "content_with_weight": "仅允许回答培训要求", "source_content_with_weight": "收入100",
                 "evidence_view": "answer_context"}
    case = evaluate_retrieval_case(table_sample, {
        "evaluation_scope": "agent_evidence_sets", "chunks_before_sufficiency": [source], "chunks": [projected],
    }, latency_ms=1, top_k=5, match_threshold=1)
    assert case["verification_evidence_metrics"]["recall"] == 1
    assert case["answer_evidence_metrics"]["recall"] == 0
    assert case["retrieval"]["chunks"][0]["source_content_with_weight"] == "收入100"
