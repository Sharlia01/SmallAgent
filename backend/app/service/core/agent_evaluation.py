"""通过真实的 Agent、知识库工具和回答路径执行只读评测。"""

from __future__ import annotations

import os
import time
from typing import Any, Callable

from agent.orchestrator import AgentOrchestrator
from agent.tools.rag_search import RagSearchTool
from service.core.grounded_answer import GroundedTurnResult, _default_answer_client, DEFAULT_ANSWER_MODEL
from service.core.retrieval import retrieve_evidence, format_retrieval_results
from service.core.retrieval_result import RetrievalResult, adapt_retrieval_records, retrieval_record
from service.core.streaming_response import collect_answer_stream


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
    raw_runs = []
    retrieval_ms = 0.0
    started = time.perf_counter()

    def retrieve(_bound_user, query, *, page_size, context):
        nonlocal retrieval_ms
        options = dict(retrieval_options or {})
        options.setdefault("page_size", page_size)
        before = time.perf_counter()
        if retrieval_function is retrieve_evidence:
            options["context"] = context
        output = retrieval_function(index_names, query, request=context.request, **options)
        retrieval_ms += (time.perf_counter() - before) * 1000
        raw_runs.append(output)
        if isinstance(output, RetrievalResult):
            return output
        records, diagnostics = format_retrieval_results(output)
        return adapt_retrieval_records(context, records, diagnostics)

    tool = RagSearchTool(user_id=0, retriever=retrieve)
    orchestrator = AgentOrchestrator(
        planner_client=planner_client or client,
        planner_model=planner_model or os.getenv("AGENT_MODEL", "deepseek-v4-pro"),
        answer_client=client, answer_model=answer_model or os.getenv("CHAT_MODEL", DEFAULT_ANSWER_MODEL),
        tools=[tool], fallback_tool=tool,
    )
    execution = orchestrator.run(
        session_id="offline-agent-eval", question=question,
        conversation_history=conversation_history,
    )
    orchestration_ms = (time.perf_counter() - started) * 1000
    answer_started = time.perf_counter()
    if execution.direct_answer is not None:
        answer, thinking = execution.direct_answer, ""
    else:
        collected = collect_answer_stream(execution.answer_stream)
        answer, thinking = collected.answer, collected.thinking
    raw_runs = [retrieval_record(run) if isinstance(run, RetrievalResult) else run for run in raw_runs]
    raw = dict(raw_runs[-1]) if raw_runs else {"total": 0, "chunks": []}
    raw["query_runtime"] = execution.query_runtime
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
        "tool_queries": [{"tool": result.tool_name, "query": result.query}
                         for result in execution.agent_run.tool_results],
        "retrieval_calls": len(raw_runs),
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
