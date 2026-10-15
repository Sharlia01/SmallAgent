import pytest

from evaluation.retrieval import aggregate_query_metrics


pytestmark = pytest.mark.unit


def test_runtime_metrics_use_budget_counts_and_include_failed_samples():
    result = aggregate_query_metrics([
        {"retrieval": {
            "query_runtime": {"counts": {"model": 2, "retrieval": 3}},
            "expected_retrieval": {"mode": "parallel"}, "retrieval_route": {"mode": "parallel"},
            "retrieval_plan": {"mode": "single", "source": "fallback"},
            "query_repair": {"attempted_goal_ids": ["profit"], "stop_reason": "goals_satisfied"},
            "evidence_sufficiency": {"goal_coverage": [{"goal_id": "profit", "status": "supported"}]},
        }},
        {"error": "network unavailable"},
    ])
    assert result["runtime_sample_count"] == result["runtime_unavailable_count"] == 1
    assert result["average_model_calls"] == 2
    assert result["average_retrieval_calls"] == 3
    assert "model_event_count" not in result
    assert "reported_total_tokens" not in result
    assert result["plan_fallback_rate"] == 1
    assert result["repair_success_rate"] == result["route_agreement"] == 1
    assert "phase_metrics" not in result
    assert "stage_metrics" not in result


def test_missing_runtime_metrics_are_not_reported_as_zero_cost():
    result = aggregate_query_metrics([{"retrieval": {}}])
    assert result["average_model_calls"] is None
    assert "reported_total_tokens" not in result
    assert result["repair_success_rate"] is None
