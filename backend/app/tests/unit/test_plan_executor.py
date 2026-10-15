"""任务依赖、来源范围、故障隔离和跨工具执行由 Agent 统一管理。"""
from threading import Barrier
import pytest
from agent.executor import ToolExecutor
from agent.plans import AgentTask, AgentPlan
from agent.tools.rag_search import RagSearchTool
from service.core.evidence.bindings import BridgeExtraction
from service.core.query.context import prepare_query_context
from service.core.runtime.context import QueryBudgetExceeded, query_runtime
from tests.agent_helpers import SearchTool, records

pytestmark = pytest.mark.unit


def dependent_plan(tool="search_knowledge_base", scoped=False):
    return AgentPlan((
        AgentTask("first", "search_knowledge_base", query="研报推荐公司", goal_ids=("g1",), output_slot="company"),
        AgentTask("second", tool, query_template="{company}公告", goal_ids=("g2",),
                  depends_on=("first",), required_slots=("company",), inherit_document_scope=scoped),
    ))


def binding(**kwargs):
    return BridgeExtraction(kwargs["slot"], "甲公司", 1, "c1", "doc1", "rule", source_quote="甲公司")


def test_kb_binding_can_drive_web_query():
    kb, web = SearchTool([records(text="推荐甲公司")]), SearchTool([records("web")])
    web.name = "search_web"
    executor = ToolExecutor([kb, web], bridge_extractor=binding)
    with query_runtime():
        states = executor.execute_plan(dependent_plan("search_web"), prepare_query_context("推荐公司及公告"), {}, {})
    assert all(state.status == "succeeded" for state in states)
    assert web.calls[0].query == "甲公司公告"
    assert states[0].binding.source_chunk_id == "c1"


def test_external_candidate_only_result_supports_binding_and_observation():
    def retrieve(*args, **kwargs):
        return [], {"candidate_chunks": records(text="推荐甲公司")}

    def extract(**kwargs):
        assert [chunk["chunk_id"] for chunk in kwargs["chunks"]] == ["c1"]
        return binding(**kwargs)

    kb = RagSearchTool(user_id=42, retriever=retrieve)
    web = SearchTool([records("web")])
    web.name = "search_web"
    with query_runtime():
        states = ToolExecutor([kb, web], bridge_extractor=extract).execute_plan(
            dependent_plan("search_web"), prepare_query_context("推荐公司及公告"), {}, {})
    assert all(state.status == "succeeded" for state in states)
    assert web.calls[0].query == "甲公司公告"
    observation = states[0].observation()
    assert observation["result_count"] == 1
    assert observation["snippets"][0]["id"] == "c1"


def test_same_report_scope_reaches_search_request():
    kb = SearchTool([records(text="推荐甲公司"), records("c2")])
    with query_runtime():
        ToolExecutor([kb], bridge_extractor=binding).execute_plan(dependent_plan(scoped=True), prepare_query_context("研报推荐公司及公告"), {}, {})
    assert kb.calls[1].document_ids == ("doc1",)


def test_unresolved_binding_blocks_only_its_descendants():
    kb = SearchTool([records(), records("independent")])
    plan = AgentPlan((*dependent_plan().tasks, AgentTask("other", kb.name, query="其他制度", goal_ids=("g3",))))
    with query_runtime():
        states = ToolExecutor([kb], bridge_extractor=lambda **kwargs: None).execute_plan(plan, prepare_query_context("制度"), {}, {})
    by_id = {state.task.id: state for state in states}
    assert by_id["second"].status == "blocked"
    assert by_id["other"].result.evidence
    assert len(kb.calls) == 2


def test_failure_keeps_successful_sibling_evidence():
    class Tool(SearchTool):
        def run_for_task(self, context, *, task, **kwargs):
            if task.query == "失败查询":
                raise RuntimeError("offline")
            return super().run_for_task(context, task=task, **kwargs)
    tool = Tool([records()])
    plan = AgentPlan((AgentTask("a", tool.name, query="失败查询", goal_ids=("g1",)), AgentTask("b", tool.name, query="成功查询", goal_ids=("g2",))))
    with query_runtime():
        states = ToolExecutor([tool]).execute_plan(plan, prepare_query_context("查询"), {}, {})
    assert states[0].status == "failed"
    assert states[1].result.evidence


def test_independent_tools_execute_concurrently():
    barrier = Barrier(2)
    class Tool(SearchTool):
        def run_for_task(self, context, *, task, **kwargs):
            barrier.wait(timeout=3)
            return super().run_for_task(context, task=task, **kwargs)
    tool = Tool([records("c1"), records("c2")])
    plan = AgentPlan(tuple(AgentTask(f"s{i}", tool.name, query=f"查询{i}", goal_ids=("g1",)) for i in range(2)))
    with query_runtime():
        states = ToolExecutor([tool]).execute_plan(plan, prepare_query_context("查询"), {}, {})
    assert all(state.status == "succeeded" for state in states)


def test_budget_error_is_not_converted_into_empty_search():
    tool = SearchTool([QueryBudgetExceeded("limit")])
    with query_runtime():
        states = ToolExecutor([tool]).execute_plan(AgentPlan((AgentTask("s1", tool.name, query="查询", goal_ids=("g1",)),)), prepare_query_context("查询"), {}, {})
    assert states[0].status == "budget_exhausted"


def test_bridge_budget_error_keeps_retrieved_evidence_and_blocks_child():
    def exhausted(**kwargs):
        raise QueryBudgetExceeded("bridge limit")
    tool = SearchTool([records(text="推荐甲公司")])
    with query_runtime():
        states = ToolExecutor([tool], bridge_extractor=exhausted).execute_plan(
            dependent_plan(), prepare_query_context("推荐公司及公告"), {}, {})
    assert states[0].status == "budget_exhausted"
    assert states[0].result.evidence
    assert states[1].status == "blocked"
    assert len(tool.calls) == 1
