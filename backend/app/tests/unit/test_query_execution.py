from threading import Barrier
from types import SimpleNamespace

import pytest

from service.core.query_execution import execute_graph
from service.core.query_planning import QuerySubqueryPlan
from service.core.query_runtime import (
    QueryRuntime, QueryBudgetExceeded, current_runtime, model_completion,
    query_runtime, run_independent,
)
from service.core.sequential_retrieval import BridgeExtraction


pytestmark = pytest.mark.unit


def test_mixed_graph_executes_independent_nodes_then_scoped_dependency():
    barrier = Barrier(2)
    calls = []
    def retrieve(query, doc_ids):
        calls.append((query, doc_ids))
        if query in {"评级", "利润"}:
            barrier.wait(timeout=2)
        return {"chunks": [{"chunk_id": query, "content_with_weight": "买入", "doc_id": "doc-1"}]}
    def bridge(**kwargs):
        return BridgeExtraction(kwargs["slot"], "买入", 1, "评级", "doc-1", "rule")
    steps = [
        QuerySubqueryPlan("rating", query="评级", output_slot="rating"),
        QuerySubqueryPlan("profit", query="利润"),
        QuerySubqueryPlan("definition", query_template="{rating}定义", depends_on=["rating"], required_slots=["rating"], inherit_document_scope=True),
    ]
    with query_runtime():
        results = execute_graph(steps, original_question="利润、评级及定义", retrieve=retrieve, bridge_extractor=bridge)
    assert all(result.status == "succeeded" for result in results)
    assert calls[-1] == ("买入定义", ["doc-1"])


def test_failed_node_only_blocks_its_descendants():
    def retrieve(query, doc_ids):
        if query == "主体":
            raise ConnectionError("unavailable")
        return {"chunks": [{"chunk_id": "survivor"}]}
    results = execute_graph([
        QuerySubqueryPlan("a", query="主体", output_slot="entity"),
        QuerySubqueryPlan("b", query="另一个事实"),
        QuerySubqueryPlan("c", query_template="{entity}定义", depends_on=["a"], required_slots=["entity"]),
    ], original_question="多目标", retrieve=retrieve)
    assert [result.status for result in results] == ["failed", "succeeded", "blocked"]
    assert results[1].result["chunks"][0]["chunk_id"] == "survivor"


def test_bridge_failure_retains_first_hop_evidence():
    result = execute_graph([
        QuerySubqueryPlan("a", query="主体", output_slot="entity"),
        QuerySubqueryPlan("b", query_template="{entity}定义", depends_on=["a"], required_slots=["entity"]),
    ], original_question="定义", retrieve=lambda *_: {"chunks": [{"chunk_id": "retained"}]},
        bridge_extractor=lambda **_: None)
    assert result[0].status == "unresolved_bridge"
    assert result[0].result["chunks"]
    assert result[1].status == "blocked"


def test_budget_is_shared_by_parallel_workers():
    budget = QueryRuntime(max_retrievals=1)
    def reserve():
        assert current_runtime() is budget
        budget.reserve("retrieval")
        return "ok"
    with query_runtime(budget):
        results = run_independent({"a": reserve, "b": reserve})
    assert sum(value == "ok" for value in results.values()) == 1
    assert sum(isinstance(value, QueryBudgetExceeded) for value in results.values()) == 1
    assert current_runtime() is None


def test_model_budget_prevents_another_provider_call():
    calls = []
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs: calls.append(kwargs) or SimpleNamespace(usage=None),
    )))
    with query_runtime(QueryRuntime(max_model_calls=1)):
        model_completion(client, "analysis", model="test", timeout=120)
        with pytest.raises(QueryBudgetExceeded):
            model_completion(client, "rewrite", model="test")
    assert len(calls) == 1
    assert 0 < calls[0]["timeout"] <= 45


def test_deadline_does_not_start_more_work():
    budget = QueryRuntime(timeout_seconds=0)
    with query_runtime(budget):
        values = run_independent({"a": lambda: pytest.fail("must not start")})
    assert isinstance(values["a"], QueryBudgetExceeded)


def test_request_cache_is_not_shared_with_another_request():
    with query_runtime() as first:
        first.cache[("tenant-a", "query")] = "private evidence"
    with query_runtime() as second:
        assert not second.cache


@pytest.mark.parametrize("doc_id", [None, "", " "])
def test_missing_bridge_document_id_never_expands_scope(doc_id):
    calls = []
    def retrieve(query, doc_ids):
        calls.append(query)
        return {"chunks": [{"chunk_id": "c1", "content_with_weight": "买入"}]}
    results = execute_graph([
        QuerySubqueryPlan("a", query="评级", output_slot="rating"),
        QuerySubqueryPlan("b", query_template="{rating}定义", depends_on=["a"],
                          required_slots=["rating"], inherit_document_scope=True),
    ], original_question="评级定义", retrieve=retrieve, bridge_extractor=lambda **kwargs:
        BridgeExtraction("rating", "买入", 1, "评级", doc_id, "rule"))
    assert calls == ["评级"]
    assert results[1].status == "blocked"
