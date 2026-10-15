"""检索适配不解释问题、不信任旧报告，也不清空部分证据。"""
from service.core.retrieval.result import RetrievalResult, adapt_retrieval_records, retrieval_record
from service.core.evidence.adapter import collection_from_records
from service.core.query.contract import RetrievalTask
from tests.agent_helpers import records


def test_external_verdict_does_not_gate_or_verify_search_results():
    result = adapt_retrieval_records(records(), {"evidence_sufficiency": {"sufficient": False}, "query_intent": {"goals": []}})
    assert len(result.evidence) == 1
    assert not hasattr(result, "assessment") and not hasattr(result, "understanding")
    assert "evidence_sufficiency" not in result.diagnostics


def test_result_keeps_ranked_and_candidate_sources_and_task():
    task = RetrievalTask("员工培训要求", ("g1",), ("doc1",))
    result = adapt_retrieval_records(records(), {"candidate_chunks": records("c2"), "total": 100}, task=task)
    exported = retrieval_record(result)
    assert len(exported["chunks"]) == 1 and len(exported["chunks_before_sufficiency"]) == 2
    assert exported["total"] == 100
    assert exported["retrieval_task"]["document_ids"] == ("doc1",)
    assert "response_mode" not in exported
