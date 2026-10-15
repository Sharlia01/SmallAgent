"""检索适配不解释问题、不信任旧报告，也不清空部分证据。"""
from service.core.retrieval.result import RetrievalResult, adapt_retrieval_records, retrieval_record
from service.core.evidence.adapter import collection_from_records
from service.core.query.contract import RetrievalTask
from tests.agent_helpers import records
import pytest


pytestmark = pytest.mark.unit


def test_external_verdict_does_not_gate_or_verify_search_results():
    result = adapt_retrieval_records(records(), {"evidence_sufficiency": {"sufficient": False}, "query_intent": {"goals": []}})
    assert len(result.evidence) == 1
    assert not hasattr(result, "assessment") and not hasattr(result, "understanding")
    assert "evidence_sufficiency" not in result.diagnostics


def test_external_candidates_merge_without_duplicates_and_preserve_order_and_task():
    task = RetrievalTask("员工培训要求", ("g1",), ("doc1",))
    result = adapt_retrieval_records(records(), {
        "candidate_chunks": records("c2") + records(), "total": 100,
    }, task=task)
    exported = retrieval_record(result)
    assert [item.id for item in result.evidence] == ["c2", "c1"]
    assert len(result.evidence.traces("c1")) == 1
    assert result.evidence.traces("c1")[0].query == task.query
    assert exported["chunks"] == exported["chunks_before_sufficiency"]
    assert [chunk["chunk_id"] for chunk in exported["chunks"]] == ["c2", "c1"]
    assert "candidate_chunks" not in result.diagnostics
    assert not hasattr(result, "candidates")
    assert exported["total"] == 100
    assert exported["retrieval_task"]["document_ids"] == ("doc1",)
    assert "response_mode" not in exported


def test_external_candidates_survive_empty_primary_results_and_record_conflicts():
    result = adapt_retrieval_records([], {"candidate_chunks": records(text="候选原文")})
    assert result.evidence.get("c1").text == "候选原文"

    conflicting = adapt_retrieval_records(records(text="不同原文"), {
        "candidate_chunks": records(text="候选原文"),
    })
    assert len(conflicting.evidence) == 1
    assert conflicting.evidence.get("c1").text == "候选原文"
    assert len(conflicting.evidence.conflicts) == 1


def test_copy_preserves_sources_but_isolates_collection_and_diagnostics():
    result = RetrievalResult(
        evidence=collection_from_records(records()),
        diagnostics={"stage_latency_ms": {"total": 10}},
    )
    copied = result.copy()
    copied.evidence.merge(collection_from_records(records("c2")))
    copied.diagnostics["stage_latency_ms"]["total"] = 20
    assert [item.id for item in result.evidence] == ["c1"]
    assert copied.evidence.get("c1") is result.evidence.get("c1")
    assert result.diagnostics["stage_latency_ms"]["total"] == 10
