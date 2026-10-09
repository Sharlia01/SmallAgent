"""读取 OpenAI 兼容回答流的辅助函数，不依赖具体提供方。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class CollectedAnswer:
    """从模型响应流重建的完整文本。"""

    answer: str
    thinking: str
    chunk_count: int


def read_stream_chunk(chunk: Any) -> tuple[Any, str, str]:
    """从单个响应片段提取结束状态、回答文本及推理文本。"""
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
    """读取与聊天相同的回答流，不生成 SSE 事件或写入存储。"""
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
