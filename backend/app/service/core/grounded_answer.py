"""Side-effect-free knowledge-base turn execution for offline evaluation."""

from __future__ import annotations

import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

from openai import OpenAI

from agent.orchestrator import build_answer_messages
from service.core.answer_policy import render_refusal
from service.core.retrieval import format_retrieved_chunk, retrieve_raw_results
from service.core.streaming_response import collect_answer_stream


DEFAULT_ANSWER_MODEL = "deepseek-v4-pro"


@dataclass(frozen=True)
class GroundedTurnResult:
    """Observable retrieval -> policy -> answer execution result."""

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


def execute_grounded_turn(
    *,
    index_names: str | list[str],
    question: str,
    conversation_history: list[dict[str, str]] | None = None,
    answer_client: Any | None = None,
    answer_model: str | None = None,
    retrieval_function: Callable[..., dict[str, Any]] = retrieve_raw_results,
    retrieval_options: dict[str, Any] | None = None,
) -> GroundedTurnResult:
    """Run production retrieval and answer prompting without DB/Redis writes.

    This is intentionally knowledge-base-only. Agent tool routing has separate
    semantics and must not let web search answer a KB-unanswerable benchmark.
    """
    normalized_question = question.strip() if isinstance(question, str) else ""
    if not normalized_question:
        raise ValueError("question must be a non-empty string")

    total_started = time.perf_counter()
    retrieval_started = time.perf_counter()
    raw = retrieval_function(
        index_names,
        normalized_question,
        **(retrieval_options or {}),
    )
    retrieval_latency = (time.perf_counter() - retrieval_started) * 1000

    sufficiency = raw.get("evidence_sufficiency") or {}
    if sufficiency.get("sufficient") is False:
        answer = render_refusal(sufficiency.get("missing_requirements"))
        total_latency = (time.perf_counter() - total_started) * 1000
        return GroundedTurnResult(
            question=normalized_question,
            answer=answer,
            thinking="",
            response_mode="refuse",
            documents=[],
            raw_retrieval=raw,
            answer_messages=[],
            latency_ms={
                "retrieval": round(retrieval_latency, 3),
                "answer": 0.0,
                "total": round(total_latency, 3),
            },
        )

    documents = [
        {
            **format_retrieved_chunk(chunk, rank),
            "source_type": "knowledge_base",
        }
        for rank, chunk in enumerate(raw.get("chunks") or [], start=1)
    ]
    answer_messages = build_answer_messages(
        normalized_question,
        documents,
        list(conversation_history or []),
    )

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
