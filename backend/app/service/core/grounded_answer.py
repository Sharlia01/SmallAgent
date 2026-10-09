"""供离线评测使用的知识库单轮执行，不写入会话等外部状态。"""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

from openai import OpenAI

from service.core.evidence_pipeline import build_answer_context, build_answer_messages, answer_documents
from service.core.evidence_verification import verify_evidence
from service.core.query_context import query_context_for
from service.core.retrieval_result import RetrievalResult, adapt_retrieval_records, retrieval_record
from service.core.answer_policy import render_evidence_refusal
from service.core.retrieval import format_retrieval_results, retrieve_evidence
from service.core.streaming_response import collect_answer_stream
from service.core.query_contract import QueryRequest
from service.core.query_runtime import bounded_query


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


@bounded_query
def execute_grounded_turn(
    *,
    index_names: str | list[str],
    question: str,
    conversation_history: list[dict[str, str]] | None = None,
    answer_client: Any | None = None,
    answer_model: str | None = None,
    retrieval_function: Callable[..., RetrievalResult | dict[str, Any]] = retrieve_evidence,
    retrieval_options: dict[str, Any] | None = None,
) -> GroundedTurnResult:
    """使用生产检索与回答提示执行评测，不写入数据库或 Redis。

    此入口仅使用知识库。Agent 工具路由有独立语义，
    不能通过网页搜索回答知识库基准中刻意设置的不可回答问题。
    """
    normalized_question = question.strip() if isinstance(question, str) else ""
    if not normalized_question:
        raise ValueError("question must be a non-empty string")

    query = query_context_for(QueryRequest.from_user(normalized_question, conversation_history))
    request = query.resolve()
    total_started = time.perf_counter()
    retrieval_started = time.perf_counter()
    options = dict(retrieval_options or {})
    if conversation_history:
        options["request"] = request
    if retrieval_function is retrieve_evidence:
        options["context"] = query
    output = retrieval_function(
        index_names,
        normalized_question,
        **options,
    )
    retrieval_latency = (time.perf_counter() - retrieval_started) * 1000

    if isinstance(output, RetrievalResult):
        retrieved = output
        raw = retrieval_record(retrieved)
    else:
        raw = output
        records, diagnostics = format_retrieval_results(raw)
        retrieved = adapt_retrieval_records(query, records, diagnostics)
    clarification = query.clarification or retrieved.clarification
    if clarification:
        total_latency = (time.perf_counter() - total_started) * 1000
        return GroundedTurnResult(
            question=normalized_question,
            answer=clarification,
            thinking="",
            response_mode="clarify",
            documents=[],
            raw_retrieval=raw,
            answer_messages=[],
            latency_ms={
                "retrieval": round(retrieval_latency, 3),
                "answer": 0.0,
                "total": round(total_latency, 3),
            },
        )

    # 候选池与选中来源共同组成快照，不能把旧报告重新绑定到较小的来源集合。
    evidence = retrieved.candidates.copy()
    evidence.merge(retrieved.evidence)
    assessment = verify_evidence(query, evidence, previous=retrieved.assessment)
    raw = {**raw, "evidence_sufficiency": assessment.to_dict()}
    if not assessment.sufficient:
        return GroundedTurnResult(
            question=normalized_question, answer=render_evidence_refusal(assessment.to_dict()), thinking="",
            response_mode="refuse", documents=[], raw_retrieval=raw, answer_messages=[],
            latency_ms={"retrieval": round(retrieval_latency, 3), "answer": 0.0,
                        "total": round((time.perf_counter() - total_started) * 1000, 3)},
        )
    context = build_answer_context(query, evidence, assessment)
    answer_messages = build_answer_messages(context, list(conversation_history or []))
    documents = answer_documents(context, evidence, assessment)

    resolved_client = answer_client or _default_answer_client()
    resolved_model = answer_model or os.getenv(
        "CHAT_MODEL",
        DEFAULT_ANSWER_MODEL,
    )
    answer_started = time.perf_counter()
    stream = resolved_client.chat.completions.create(
        model=resolved_model,
        messages=answer_messages,
        stream=True,
        temperature=0,
    )
    collected = collect_answer_stream(stream)
    answer_latency = (time.perf_counter() - answer_started) * 1000
    total_latency = (time.perf_counter() - total_started) * 1000

    return GroundedTurnResult(
        question=normalized_question,
        answer=collected.answer,
        thinking=collected.thinking,
        response_mode="answer",
        documents=documents,
        raw_retrieval=raw,
        answer_messages=answer_messages,
        latency_ms={
            "retrieval": round(retrieval_latency, 3),
            "answer": round(answer_latency, 3),
            "total": round(total_latency, 3),
        },
    )
