"""知识库限定的 Agent 问答入口；线上独立调用和离线评测共用。"""

from __future__ import annotations

import os
import time
from threading import Lock
from dataclasses import asdict, dataclass
from openai import OpenAI
from typing import Any, Callable

from agent.orchestrator import AgentOrchestrator
from agent.tools.rag_search import RagSearchTool
from service.core.retrieval.pipeline import retrieve_evidence, format_retrieval_results
from service.core.retrieval.result import RetrievalResult, adapt_retrieval_records, retrieval_record
from service.core.answering.streaming import collect_answer_stream
from service.core.query.contract import RetrievalTask


DEFAULT_ANSWER_MODEL = "deepseek-v4-pro"


@dataclass(frozen=True)
class GroundedTurnResult:
    """记录检索、回答策略和生成过程的可观测结果。"""

    question: str
    answer: str
    thinking: str
    response_mode: str
    documents: list[dict[str, Any]]
    raw_retrieval: dict[str, Any]
    answer_messages: list[dict[str, str]]
    latency_ms: dict[str, float]

    def metadata(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("raw_retrieval", None)
        data.pop("answer_messages", None)
        return data


def _default_answer_client() -> OpenAI:
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    base_url = os.getenv("DASHSCOPE_BASE_URL", "").strip()
    if not api_key or not base_url:
        raise RuntimeError(
            "DASHSCOPE_API_KEY and DASHSCOPE_BASE_URL are required for answers"
        )
    return OpenAI(api_key=api_key, base_url=base_url)


def execute_agent_turn(
    *, index_names: str | list[str], question: str,
    conversation_history: list[dict[str, str]] | None = None,
    answer_client: Any | None = None, answer_model: str | None = None,
    planner_client: Any | None = None, planner_model: str | None = None,
    retrieval_function: Callable[..., RetrievalResult | dict[str, Any]] = retrieve_evidence,
    retrieval_options: dict[str, Any] | None = None,
) -> GroundedTurnResult:
    """执行查询理解的全部层次，不创建会话或写入数据库。

    该语料仅用于知识库评测，网页搜索不能补全刻意设置的知识库不可回答问题。
    索引访问权限由命令行操作人员负责确认。
    """
    client = answer_client or _default_answer_client()
    retrieval_runs = []
    retrieval_intervals = []
    retrieval_lock = Lock()
    started = time.perf_counter()

    def retrieve(_bound_user, query, *, page_size, task: RetrievalTask | None = None):
        options = dict(retrieval_options or {})
        options.setdefault("page_size", page_size)
        before = time.perf_counter()
        if retrieval_function is retrieve_evidence or (task and task.document_ids):
            # 继承文档范围时，自定义检索器也必须实现任务协议；不能静默扩大搜索范围。
            options["task"] = task
        output = retrieval_function(index_names, query, **options)
        with retrieval_lock:
            retrieval_intervals.append((before, time.perf_counter()))
            retrieval_runs.append(output)
        if isinstance(output, RetrievalResult):
            return output
        records, diagnostics = format_retrieval_results(output)
        return adapt_retrieval_records(records, diagnostics, task=task)

    tool = RagSearchTool(user_id=0, retriever=retrieve)
    orchestrator = AgentOrchestrator(
        planner_client=planner_client or client,
        planner_model=planner_model or os.getenv("AGENT_MODEL", "deepseek-v4-pro"),
        answer_client=client, answer_model=answer_model or os.getenv("CHAT_MODEL", DEFAULT_ANSWER_MODEL),
        tools=[tool],
    )
    execution = orchestrator.run(
        session_id="offline-agent-eval", question=question,
        conversation_history=conversation_history,
    )
    orchestration_ms = (time.perf_counter() - started) * 1000
    with retrieval_lock:
        raw_runs = list(retrieval_runs)
        intervals = sorted(retrieval_intervals)
    # 并行检索按时间区间的并集计时，否则相加会夸大检索耗时、挤掉规划耗时。
    retrieval_ms, previous_end = 0.0, started
    for begin, end in intervals:
        retrieval_ms += max(0, end - max(begin, previous_end)) * 1000
        previous_end = max(previous_end, end)
    answer_started = time.perf_counter()
    if execution.direct_answer is not None:
        answer, thinking = execution.direct_answer, ""
    else:
        collected = collect_answer_stream(execution.answer_stream)
        answer, thinking = collected.answer, collected.thinking
    raw_runs = [retrieval_record(run) if isinstance(run, RetrievalResult) else run for run in raw_runs]
    raw = dict(raw_runs[-1]) if raw_runs else {"total": 0, "chunks": []}
    raw["query_runtime"] = execution.query_runtime
    query = execution.agent_run.query
    raw["query_contract"] = query.request.to_dict() if query else {}
    raw["query_understanding"] = query.understanding.to_dict() if query and query.understanding else {}
    raw["financial_requirements"] = query.financial_requirements.to_dict() if query and query.financial_requirements else None
    raw["financial_bindings"] = [asdict(binding) for binding in query.financial_bindings] if query else []
    candidates = {}
    for run in raw_runs:
        for chunk in run.get("chunks_before_sufficiency", run.get("chunks", [])):
            key = str(chunk.get("chunk_id") or chunk.get("content_with_weight"))
            candidates.setdefault(key, chunk)
    raw["chunks_before_sufficiency"] = list(candidates.values())
    if execution.evidence_sufficiency is not None:
        raw["evidence_sufficiency"] = execution.evidence_sufficiency
    if execution.response_mode != "answer":
        raw["chunks"] = []
        raw["total"] = 0
    else:
        raw["chunks"] = execution.retrieved_content
        raw["total"] = len(execution.retrieved_content)
    raw["agent_execution"] = {
        "steps": execution.agent_run.steps, "stop_reason": execution.agent_run.stop_reason,
        "tool_queries": [{"tool": result.tool_name, "query": result.query,
                          "goal_ids": list(result.task.goal_ids) if result.task else []}
                         for result in execution.agent_run.tool_results],
        "retrieval_calls": len(raw_runs),
        "tasks": [{**state.task.to_dict(), "status": state.status,
                   "binding": state.binding.to_dict() if state.binding else None, "error": state.error}
                  for state in execution.agent_run.executions.values()],
    }
    return GroundedTurnResult(
        question=question, answer=answer, thinking=thinking,
        response_mode=execution.response_mode, documents=execution.response_documents,
        raw_retrieval=raw, answer_messages=execution.answer_messages,
        latency_ms={
            "retrieval": round(retrieval_ms, 3),
            "planning_and_answer_request": round(max(0, orchestration_ms - retrieval_ms), 3),
            "answer": round((time.perf_counter() - answer_started) * 1000, 3),
            "total": round((time.perf_counter() - started) * 1000, 3),
        },
    )
