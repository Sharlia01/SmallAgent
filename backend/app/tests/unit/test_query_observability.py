import pytest

from service.core.retrieval_evaluation import aggregate_query_metrics


pytestmark = pytest.mark.unit


def test_runtime_costs_keep_unknown_usage_and_failed_samples_visible():
    result = aggregate_query_metrics([
        {"retrieval": {
            "query_runtime": {"counts": {"model": 2, "retrieval": 3}, "events": [
                {"model": "test", "total_tokens": 120}, {"model": "test", "total_tokens": None},
            ]},
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
    assert result["model_event_count"] == 2
    assert result["token_usage_event_count"] == 1
    assert result["reported_total_tokens"] == 120
    assert result["plan_fallback_rate"] == 1
    assert result["repair_success_rate"] == result["route_agreement"] == 1
    assert "phase_metrics" not in result
    assert "stage_metrics" not in result


def test_missing_runtime_metrics_are_not_reported_as_zero_cost():
    result = aggregate_query_metrics([{"retrieval": {}}])
    assert result["average_model_calls"] is None
    assert result["reported_total_tokens"] is None
    assert result["repair_success_rate"] is None
