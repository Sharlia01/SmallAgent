"""搜索入口只执行指定查询；别名扩展和过滤均为确定性适配。"""
import pytest
from service.core.retrieval import pipeline
from service.core.query.contract import RetrievalTask
from service.core.runtime.context import query_runtime, QueryRuntime
from tests.agent_helpers import records

pytestmark = pytest.mark.unit


class Dealer:
    def __init__(self):
        self.calls = []
    def retrieval(self, **kwargs):
        self.calls.append(kwargs)
        return {"chunks": records(), "total": 1, "stage_latency_ms": {"rerank": 1}}


@pytest.fixture
def dealer(monkeypatch):
    dealer = Dealer()
    monkeypatch.setattr(pipeline, "get_retrieval_dealer", lambda: dealer)
    monkeypatch.setattr(pipeline, "expand_table_contexts", lambda chunks, question: chunks)
    return dealer


def test_search_does_not_call_semantic_models_and_keeps_partial_evidence(dealer, monkeypatch):
    from service.core.runtime import model_calls
    monkeypatch.setattr(model_calls, "model_completion", lambda *args, **kwargs: pytest.fail("检索不能触发语义分析"))
    result = pipeline.retrieve_evidence("tenant", "员工培训要求和差旅要求")
    assert len(result.evidence) == 1 and len(dealer.calls) == 1
    assert not hasattr(result, "assessment")


def test_server_document_scope_and_tenant_reach_dealer(dealer):
    pipeline.retrieve_evidence("tenant42", "员工培训要求", task=RetrievalTask("员工培训要求", ("g1",), ("doc1",)))
    call = dealer.calls[0]
    assert call["tenant_ids"] == "tenant42"
    assert call["options"].doc_ids == ["doc1"]


def test_same_query_different_document_scope_does_not_share_cache(dealer):
    with query_runtime():
        for doc in ("a", "a", "b"):
            pipeline.retrieve_evidence("tenant", "员工培训要求", task=RetrievalTask("员工培训要求", ("g1",), (doc,)))
    assert len(dealer.calls) == 2


def test_search_budget_counts_only_cache_misses(dealer):
    runtime = QueryRuntime(max_retrievals=1)
    with query_runtime(runtime):
        pipeline.retrieve_evidence("tenant", "员工培训要求")
        pipeline.retrieve_evidence("tenant", "员工培训要求")
    assert runtime.counts["retrieval"] == 1


def test_only_dictionary_aliases_are_added_to_keyword_query(dealer):
    pipeline.retrieve_evidence("tenant", "每股收益")
    assert dealer.calls[0]["question"] == "每股收益"
    assert "EPS" in dealer.calls[0]["keyword_question"]
