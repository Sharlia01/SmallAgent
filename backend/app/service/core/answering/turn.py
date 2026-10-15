"""知识库限定的 Agent 问答入口；线上独立调用和离线评测共用。"""

from __future__ import annotations

import os
import time
from threading import Lock
from dataclasses import asdict, dataclass
from copy import deepcopy
from openai import OpenAI
from typing import Any, Callable

from agent.orchestrator import AgentOrchestrator
from agent.tools.rag_search import RagSearchTool
from service.core.retrieval.pipeline import retrieve_evidence, format_retrieval_results
from service.core.retrieval.result import RetrievalResult, adapt_retrieval_records, retrieval_record
from service.core.evidence.adapter import reference_records
from service.core.answering.streaming import collect_answer_stream
from service.core.query.contract import RetrievalTask


DEFAULT_ANSWER_MODEL = "deepseek-v4-pro"


@dataclass(frozen=True)
class GroundedTurnResult:
    """记录检索、回答策略和生成过程的可观测结果。"""

    question: str  # 用户本次传进来的原始问题，不是Agent拆分出来的子查询
    answer: str  #最终返回给用户的正文。可能是正常答案、拒答说明，也可能是要求用户补充信息的问题
    thinking: str  #回答模型通过接口的 reasoning_content 字段返回的推理文本。没有返回时为空字符串；它不包含整个 Agent 的执行记录
    response_mode: str #流程选择了哪种回答方式：answer 正常回答、refuse 拒答、clarify 请求澄清。它记录流程决定，不是分析最终正文后得出的分类
    documents: list[dict[str, Any]]  #最终回答上下文选中的来源片段，带引用编号、原文、文档名等信息。不是所有搜到的资料，也不保证模型在答案中实际引用了每一条
    raw_retrieval: dict[str, Any]  #供评测和排查使用的详细记录，包括每次搜索、累计核验证据、最终回答证据、核验结果以及 Agent 执行情况。
    answer_messages: list[dict[str, str]] #实际构造给回答模型的完整消息列表，包括系统提示、适用的历史消息、参考资料和当前问题。拒答或澄清不调用回答模型时，通常为空列表
    latency_ms: dict[str, float] #本次问答各部分的耗时，单位是毫秒。


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
    retrieval_runs = []  #表示本轮搜索记录
    retrieval_intervals = []  #本轮搜索时间
    retrieval_lock = Lock()  #加锁，并行搜索时保护记录
    started = time.perf_counter()

    def retrieve(_bound_user, query, *, page_size, task: RetrievalTask | None = None):
        '''这层包装需要记录每一次搜索，保存查询词、结果、错误和耗时'''
        # 应用评测参数，比如返回多少条、候选数量、重排参数，并传递任务中的文档范围限制
        options = dict(retrieval_options or {})
        options.setdefault("page_size", page_size)
        before = time.perf_counter()
        with retrieval_lock:
            call = {"call_id": len(retrieval_runs) + 1, "query": query,
                    "top_k": options["page_size"], "result": None, "error": None}
            retrieval_runs.append(call)
        if retrieval_function is retrieve_evidence or (task and task.document_ids):
            # 继承文档范围时，自定义检索器也必须实现任务协议；不能静默扩大搜索范围。
            options["task"] = task
        try:
            output = retrieval_function(index_names, query, **options)
            call["result"] = deepcopy(retrieval_record(output) if isinstance(output, RetrievalResult) else output)
            if isinstance(output, RetrievalResult):
                return output
            records, diagnostics = format_retrieval_results(output)
            return adapt_retrieval_records(records, diagnostics, task=task)
        except Exception as error:
            call["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            ended = time.perf_counter()
            with retrieval_lock:
                call["latency_ms"] = round((ended - before) * 1000, 3)
                retrieval_intervals.append((before, ended))

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
    # 多次检索没有一个共同的排名；不能用最后一次调用冒充整个 Agent 的检索结果。
    # retrieval_calls表示每次搜索记录
    raw = {"evaluation_scope": "agent_evidence_sets", "retrieval_calls": raw_runs}
    raw["query_runtime"] = execution.query_runtime
    query = execution.agent_run.query
    raw["query_contract"] = query.request.to_dict() if query else {}
    raw["query_understanding"] = query.understanding.to_dict() if query and query.understanding else {}
    raw["financial_requirements"] = query.financial_requirements.to_dict() if query and query.financial_requirements else None
    raw["financial_bindings"] = [asdict(binding) for binding in query.financial_bindings] if query else []
    # 记录实际交给核验器的累计资料，沿用 Agent 的去重顺序和检索轨迹。
    # 即使最终拒答、没有回答引用，也保留此快照供评测定位原因。
    raw["chunks_before_sufficiency"] = (
        reference_records(execution.agent_run.evidence)
        if execution.evidence_sufficiency is not None else None
    )
    if execution.evidence_sufficiency is not None:
        raw["evidence_sufficiency"] = execution.evidence_sufficiency
    documents = execution.response_documents
    # 引用保留原始来源；评分使用回答模型真正看到的筛选正文，不能混入未采用的资料。
    context_texts = {block.evidence_id: block.text for block in execution.context.blocks} if execution.context else {}
    raw["chunks"] = [
        {**document, "source_content_with_weight": document.get("content_with_weight", ""),
         "evidence_view": "answer_context",
         "content_with_weight": context_texts[document["chunk_id"]]}
        for document in documents
    ]
    raw["total"] = len(raw["chunks"])
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
        response_mode=execution.response_mode, documents=documents,
        raw_retrieval=raw, answer_messages=execution.answer_messages,
        latency_ms={
            "retrieval": round(retrieval_ms, 3),
            "planning_and_answer_request": round(max(0, orchestration_ms - retrieval_ms), 3),
            "answer": round((time.perf_counter() - answer_started) * 1000, 3),
            "total": round((time.perf_counter() - started) * 1000, 3),
        },
    )
