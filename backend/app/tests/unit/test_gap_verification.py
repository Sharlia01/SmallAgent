"""补查仅核验缺口，复用结论仍有来源上下文及完整报告约束。"""
import json

import pytest

from service.core.evidence import sufficiency
from service.core.evidence.verification import verify_evidence
from service.core.evidence.adapter import collection_from_records
from service.core.evidence.assessment import verification_policy_version
from service.core.query.contract import QueryGoal
from service.core.answering.answer_context import build_answer_context
from tests.agent_helpers import ModelClient
from tests.evidence_helpers import make_query

pytestmark = pytest.mark.unit


def verdict(coverage, *, sufficient=False, recheck=None):
    result = {"sufficient": sufficient, "reason": "核验完成", "missing_requirements": [] if sufficient else ["原因依据不足"],
              "goal_coverage": [{"goal_id": goal, "status": status, "supporting_chunk_indices": indices}
                                for goal, status, indices in coverage]}
    if recheck is not None:
        result["recheck_goal_ids"] = recheck
    return result


def setup_case(monkeypatch, *, conflict=False):
    monkeypatch.setenv("RAG_EVIDENCE_SUFFICIENCY_ENABLED", "true")
    monkeypatch.setenv("RAG_EVIDENCE_SUFFICIENCY_CONTEXT_TOKENS", "16384")
    query = make_query("甲公司2025年收入及增长原因？", (QueryGoal("income", "2025年收入"), QueryGoal("reason", "增长原因")))
    old = collection_from_records([
        {"chunk_id": "unused", "content_with_weight": "无关背景"},
        {"chunk_id": "income", "content_with_weight": "甲公司2025年收入100万元"},
        {"chunk_id": "partial", "content_with_weight": "甲公司2025年收入增长，尚缺具体原因"},
    ])
    first_client = ModelClient([verdict([("income", "supported", [2]),
                                       ("reason", "conflict" if conflict else "missing", [3])])])
    previous = verify_evidence(query, old, client=first_client)
    assert previous.source == "model"
    current = old.copy()
    current.merge(collection_from_records([{"chunk_id": "new", "content_with_weight": "收入增长原因是销量增加"}]))
    return query, current, previous, first_client


def test_first_round_unchanged_gap_keeps_context_and_merges_reindexed_report(monkeypatch):
    query, current, previous, first_client = setup_case(monkeypatch)
    first = json.loads(first_client.calls[0]["messages"][1]["content"])
    assert [g["id"] for g in first["goals"]] == ["income", "reason"]
    assert "retained_goals" not in first
    assert "补查缺口核验" not in first_client.calls[0]["messages"][0]["content"]
    client = ModelClient([verdict([("reason", "supported", [2, 3])], sufficient=True, recheck=[])])
    report = verify_evidence(query, current, previous=previous, client=client)
    payload = json.loads(client.calls[0]["messages"][1]["content"])
    assert payload["question"] == query.question
    assert [g["id"] for g in payload["goals"]] == ["reason"]
    assert payload["retained_goals"][0]["supporting_chunk_indices"] == [1]
    assert len(payload["candidate_evidence"]) == 3
    assert report.sufficient
    assert [item["goal_id"] for item in report.goal_coverage] == ["income", "reason"]
    assert report.supporting_chunk_indices == [2, 3, 4]
    assert report.evaluated_chunk_ids == ["unused", "income", "partial", "new"]
    assert report.diagnostics["current_evaluated_chunk_ids"] == ["income", "partial", "new"]
    assert report.diagnostics["reused_goal_ids"] == ["income"]
    assert report.matches(query.version, current.version, verification_policy_version(), require_semantic=True)
    assert [b.evidence_id for b in build_answer_context(query, current, report).blocks] == ["income", "partial", "new"]
    assert verify_evidence(query, current, previous=report, client=client) is report
    assert len(client.calls) == 1


@pytest.mark.parametrize("change", ["content", "query", "policy", "tampered"])
def test_invalid_previous_scope_uses_full_goals(monkeypatch, change):
    query, current, previous, _ = setup_case(monkeypatch)
    if change == "content":
        current = collection_from_records([{"chunk_id": item.id, "content_with_weight": item.text + "修订"} for item in current])
    elif change == "query":
        query = make_query("乙公司收入及原因？", query.goals)
    elif change == "policy":
        monkeypatch.setenv("RAG_EVIDENCE_SUFFICIENCY_CONTEXT_TOKENS", "16385")
    else:
        previous.goal_coverage[0]["status"] = "missing"
    client = ModelClient([verdict([("income", "supported", [2]), ("reason", "supported", [4])], sufficient=True)])
    report = verify_evidence(query, current, previous=previous, client=client)
    payload = json.loads(client.calls[0]["messages"][1]["content"])
    assert len(payload["goals"]) == 2 and "retained_goals" not in payload
    assert "gap_verification" not in report.diagnostics


@pytest.mark.parametrize("signal", ["old_goal", "new_conflict"])
def test_new_conflicts_restore_full_verification(monkeypatch, signal):
    query, current, previous, _ = setup_case(monkeypatch)
    partial = verdict([("reason", "supported" if signal == "old_goal" else "conflict", [3])],
                      sufficient=signal == "old_goal", recheck=["income"] if signal == "old_goal" else [])
    client = ModelClient([partial, verdict([("income", "conflict", [2, 4]), ("reason", "supported", [4])])])
    report = verify_evidence(query, current, previous=previous, client=client)
    assert len(client.calls) == 2
    full = json.loads(client.calls[1]["messages"][1]["content"])
    assert len(full["goals"]) == 2 and len(full["candidate_evidence"]) == 4
    assert "retained_goals" not in full
    assert report.diagnostics["gap_restored_full_verification"]
    assert report.diagnostics["reused_goal_ids"] == []
    assert not report.sufficient


def test_existing_conflict_is_pending_and_keeps_all_context(monkeypatch):
    query, current, previous, _ = setup_case(monkeypatch, conflict=True)
    client = ModelClient([verdict([("reason", "supported", [4])], sufficient=True, recheck=[])])
    report = verify_evidence(query, current, previous=previous, client=client)
    payload = json.loads(client.calls[0]["messages"][1]["content"])
    assert len(payload["candidate_evidence"]) == 4
    assert report.sufficient and report.diagnostics["gap_goal_ids"] == ["reason"]


def test_context_budget_omission_restores_full_goals_without_partial_model_call(monkeypatch):
    query, current, previous, _ = setup_case(monkeypatch)
    original = sufficiency._prepare_evidence
    def prepare(candidates, sources, **kwargs):
        if kwargs.get("gap_context"):
            return tuple(item for item in candidates if item.evidence.id != "income")
        return original(candidates, sources, **kwargs)
    monkeypatch.setattr(sufficiency, "_prepare_evidence", prepare)
    client = ModelClient([verdict([("income", "supported", [2]), ("reason", "supported", [4])], sufficient=True)])
    report = verify_evidence(query, current, previous=previous, client=client)
    assert len(client.calls) == 1
    assert len(json.loads(client.calls[0]["messages"][1]["content"])["goals"]) == 2
    assert report.diagnostics["gap_restored_full_verification"]


def test_missing_conflict_signal_is_rejected_not_silently_inherited(monkeypatch):
    query, current, previous, _ = setup_case(monkeypatch)
    client = ModelClient([verdict([("reason", "supported", [3])], sufficient=True)])
    report = verify_evidence(query, current, previous=previous, client=client)
    assert not report.sufficient and report.verification_status == "unavailable"
    assert report.diagnostics["failure_kind"] == "contract_rejected"
    assert report.diagnostics["reused_goal_ids"] == []
    assert len(client.calls) == 1
