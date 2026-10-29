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


def test_agent_repairs_count_executed_samples_and_evidence_success():
    def record(answerable, steps, stop_reason, **extra):
        return {"answerable": answerable, "retrieval": {"agent_execution": {
            "steps": steps, "stop_reason": stop_reason, **extra,
        }}}

    records = [
        record(True, 1, "evidence_ready"),
        record(True, 2, "evidence_ready"),
        record(False, 3, "max_steps"),
        record(False, 2, "verification_failed"),
        record(True, 1, "no_progress", replan_decisions=[{
            "decision": "repair", "execution_skipped": "repeated_direct_plan",
        }]),
        record(True, 1, "planning_failed"),
    ]
    # 即使后续回答失败，已经补齐证据仍算补查成功。
    records[1]["error"] = "answer failed"
    # Agent 记录优先，不能被旧字段覆盖或重复计数。
    records[0]["retrieval"]["query_repair"] = {
        "attempted_goal_ids": ["old"], "stop_reason": "goals_satisfied",
    }
    result = aggregate_query_metrics(records)
    assert result["repair_count"] == 3
    assert result["repair_success_rate"] == pytest.approx(1 / 3)
    assert result["repair_unavailable_count"] == 0
    assert result["repair_cohorts"]["answerable"] == {
        "sample_count": 4, "unavailable_count": 0, "attempt_count": 1,
        "goals_satisfied_count": 1, "success_rate": 1,
    }
    assert result["repair_cohorts"]["unanswerable"] == {
        "sample_count": 2, "unavailable_count": 0, "attempt_count": 2,
        "goals_satisfied_count": 0, "success_rate": 0,
    }


@pytest.mark.parametrize("steps", [None, True, -1, "2"])
def test_invalid_agent_steps_are_unavailable_and_do_not_use_legacy(steps):
    result = aggregate_query_metrics([{"answerable": True, "retrieval": {
        "agent_execution": {"steps": steps, "stop_reason": "evidence_ready"},
        "query_repair": {"attempted_goal_ids": ["old"], "stop_reason": "goals_satisfied"},
    }}])
    assert result["repair_count"] == 0
    assert result["repair_success_rate"] is None
    assert result["repair_unavailable_count"] == 1
    assert result["repair_cohorts"]["answerable"]["unavailable_count"] == 1


def test_missing_repair_data_is_distinct_from_initial_execution_only():
    result = aggregate_query_metrics([
        {"retrieval": {}},
        {"retrieval": {"agent_execution": {"steps": 0, "stop_reason": "planning_failed"}}},
        {"retrieval": {"agent_execution": {"steps": 1, "stop_reason": "evidence_ready"}}},
    ])
    assert result["repair_count"] == 0
    assert result["repair_success_rate"] is None
    assert result["repair_unavailable_count"] == 1
