from tests.evidence_helpers import make_query
from service.core.evidence_verification import verify_evidence
from service.core.evidence_adapter import collection_from_records
import json
from types import SimpleNamespace

import pytest

from service.core import evidence_sufficiency, retrieval
from service.core.query_contract import QueryGoal
from service.core.query_intent import QueryAnalysis


pytestmark = pytest.mark.unit


def verdict(*, sufficient=False, source="model"):
    return evidence_sufficiency.EvidenceAssessment(
        sufficient, "按目标校验", [] if sufficient else ["归母净利润"], [1], source, 1, ["revenue"],
        [{"goal_id": "revenue", "status": "supported", "supporting_chunk_ids": ["revenue"]},
         {"goal_id": "profit", "status": "supported" if sufficient else "missing", "supporting_chunk_ids": []}],
    )


@pytest.mark.parametrize("failure", ["missing_goal", "contradiction", "bad_index", "no_support"])
def test_goal_verifier_rejects_incomplete_or_inconsistent_proof(monkeypatch, failure):
    monkeypatch.setenv("RAG_EVIDENCE_SUFFICIENCY_ENABLED", "true")
    monkeypatch.setenv("RAG_EVIDENCE_SUFFICIENCY_FAIL_OPEN", "false")
    coverage = [{"goal_id": "revenue", "status": "supported", "supporting_chunk_indices": [1]},
                {"goal_id": "profit", "status": "supported", "supporting_chunk_indices": [1]}]
    if failure == "missing_goal":
        coverage.pop()
    elif failure == "contradiction":
        coverage[1]["status"] = "missing"
    elif failure == "bad_index":
        coverage[1]["supporting_chunk_indices"] = [2]
    payload = {"sufficient": True, "reason": "充分", "missing_requirements": [],
               "supporting_chunk_indices": [1], "goal_coverage": coverage}
    if failure == "no_support":
        payload["supporting_chunk_indices"] = []
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: response)))
    decision = verify_evidence(make_query("营业收入和归母净利润？", (QueryGoal("revenue", "营业收入"), QueryGoal("profit", "归母净利润"))),
        collection_from_records([{"chunk_id": "revenue", "content_with_weight": "营业收入10亿元。"}]),
        client=client)
    assert not decision.sufficient
    assert decision.source == "fallback"


@pytest.mark.parametrize("new_evidence", [True, False])
def test_repair_only_queries_missing_goal_and_retains_supported_evidence(monkeypatch, new_evidence):
    monkeypatch.setenv("RAG_QUERY_REPAIR_ENABLED", "true")
    calls = []
    first = {"chunk_id": "revenue", "content_with_weight": "营业收入10亿元。"}
    second = {"chunk_id": "profit", "content_with_weight": "归母净利润1亿元。"}
    def retrieve(**kwargs):
        calls.append(kwargs["question"])
        return {"total": 1, "chunks": [second if new_evidence else first]}
    monkeypatch.setattr(retrieval, "_merge_parallel_subqueries", lambda **kw: {
        "chunks": [c for _, r in kw["subquery_results"] for c in r["chunks"]], "total": 2, "doc_aggs": [],
    })
    monkeypatch.setattr(retrieval, "expand_table_contexts", lambda chunks, question: chunks)
    checked = []
    def verify(context, chunks):
        assert context.goals == analysis.goals
        checked.extend(chunk.id for chunk in chunks)
        return verdict(sufficient=True)
    monkeypatch.setattr(retrieval, "verify_evidence", verify)
    analysis = QueryAnalysis("multi_question", False, 2, True, False, False, "两个目标", "model",
        goals=(QueryGoal("revenue", "营业收入", ("2025年",)), QueryGoal("profit", "归母净利润", ("2025年",))))
    result, evidence, decision = retrieval._repair_missing_goals({"total": 1, "chunks": [first]}, collection_from_records([first]), verdict(),
        dealer=SimpleNamespace(retrieval=retrieve), index_names="1", options=retrieval.Dealer.RetrievalOptions(),
        context=make_query("2025年营业收入和归母净利润？", analysis.goals))
    assert calls == ["归母净利润 2025年"]
    assert result["chunks"][0]["chunk_id"] == "revenue"
    assert decision.sufficient is new_evidence
    assert checked == (["revenue", "profit"] if new_evidence else [])
    assert result["query_repair"]["stop_reason"] == ("goals_satisfied" if new_evidence else "no_new_evidence")
