"""Provider-neutral helpers for consuming OpenAI-compatible answer streams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class CollectedAnswer:
    """Complete text reconstructed from a streamed model response."""

    answer: str
    thinking: str
    chunk_count: int


def read_stream_chunk(chunk: Any) -> tuple[Any, str, str]:
    """Extract finish state, answer text and reasoning text from one chunk."""
    choices = getattr(chunk, "choices", None) or []
    if not choices:
        return None, "", ""

    choice = choices[0]
    delta = getattr(choice, "delta", None)
    if delta is None:
        return getattr(choice, "finish_reason", None), "", ""

    answer_text = getattr(delta, "content", None) or ""
    thinking_text = getattr(delta, "reasoning_content", None) or ""
    return getattr(choice, "finish_reason", None), answer_text, thinking_text


def collect_answer_stream(stream: Iterable[Any]) -> CollectedAnswer:
    """Consume the same streaming response used by chat without SSE or storage."""
    answer_parts: list[str] = []
    thinking_parts: list[str] = []
    chunk_count = 0

    for chunk in stream:
        chunk_count += 1
        finish_reason, answer_text, thinking_text = read_stream_chunk(chunk)
        if answer_text:
            answer_parts.append(answer_text)
        if thinking_text:
            thinking_parts.append(thinking_text)
        if finish_reason is not None:
            break

    return CollectedAnswer(
        answer="".join(answer_parts),
        thinking="".join(thinking_parts),
        chunk_count=chunk_count,
    )
