"""读取 OpenAI 兼容回答流的辅助函数，不依赖具体提供方。"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Iterable


logger = logging.getLogger(__name__)


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


def close_answer_stream(stream: Any) -> None:
    """释放模型响应连接，不关闭后续规划、评分仍需复用的客户端。"""
    close = getattr(stream, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            # 清理失败要留日志，但不能覆盖读取阶段的原异常或阻止前端断开清理。
            logger.warning("关闭回答模型响应流失败", exc_info=True)


def collect_answer_stream(stream: Iterable[Any]) -> CollectedAnswer:
    """读取与聊天相同的回答流，不生成 SSE 事件或写入存储。"""
    answer_parts: list[str] = []
    thinking_parts: list[str] = []
    chunk_count = 0

    try:
        for chunk in stream:
            chunk_count += 1
            finish_reason, answer_text, thinking_text = read_stream_chunk(chunk)
            if answer_text:
                answer_parts.append(answer_text)
            if thinking_text:
                thinking_parts.append(thinking_text)
            if finish_reason is not None and not getattr(stream, "consume_usage_tail", False):
                break
    finally:
        # finish_reason 后可能还有 usage/[DONE] 事件；提前退出不等于底层响应已关闭。
        # 不能把连接释放推迟到 GC，否则后续评分请求可能在连接池持锁时回收旧流并死锁。
        close_answer_stream(stream)

    return CollectedAnswer(
        answer="".join(answer_parts),
        thinking="".join(thinking_parts),
        chunk_count=chunk_count,
    )


class ObservedAnswerStream:
    """聊天和离线评测共用的流观测；仅记录时间、长度及用量，不记录正文。"""

    consume_usage_tail = True

    def __init__(self, *, model, session_id, trace_id, reference_count, messages, sdk_max_retries=None):
        import time
        from datetime import datetime, timezone
        self._clock = time.perf_counter
        self._started = self._clock()
        self._stream = self._iterator = None
        self._closed = False
        self._last_event = None
        self.metrics = dict(
            stage='answer_generation', model=model, session_id=session_id, trace_id=trace_id,
            request_started_at=datetime.now(timezone.utc).isoformat(), references=reference_count,
            input_chars=sum(len(str(message.get('content') or '')) for message in messages),
            message_count=len(messages), application_retry_count=0, sdk_retry_count=None,
            sdk_max_retries=sdk_max_retries, request_open_ms=None, first_event_ms=None,
            first_reasoning_ms=None, first_answer_ms=None, last_answer_ms=None,
            max_event_gap_ms=0.0, max_read_wait_ms=0.0, chunks=0, answer_chars=0,
            reasoning_chars=0, input_tokens=None, output_tokens=None, reasoning_tokens=None,
            finish_reason=None, status='closed_early', error_kind=None,
        )

    def attach(self, stream):
        self._stream, self._iterator = stream, iter(stream)
        self.metrics['request_open_ms'] = self._elapsed()
        return self

    def _elapsed(self):
        return round((self._clock() - self._started) * 1000, 3)

    def __iter__(self):
        return self

    def __next__(self):
        if self._closed:
            raise StopIteration
        started = self._clock()
        try:
            chunk = next(self._iterator)
        except StopIteration:
            self.metrics['status'] = 'ok' if self.metrics['finish_reason'] is not None else 'ended_without_finish'
            self.close()
            raise
        except Exception as error:
            # 已收到完整正文后的用量尾包失败不丢弃答案，单独记录异常。
            tail = self.metrics['finish_reason'] is not None
            self.metrics.update(status='usage_tail_error' if tail else 'stream_error', error_kind=type(error).__name__)
            self.close()
            if tail:
                raise StopIteration from None
            raise
        now = self._clock()
        elapsed = round((now - self._started) * 1000, 3)
        m = self.metrics
        m['max_read_wait_ms'] = max(m['max_read_wait_ms'], round((now - started) * 1000, 3))
        if self._last_event is not None:
            m['max_event_gap_ms'] = max(m['max_event_gap_ms'], round((now - self._last_event) * 1000, 3))
        self._last_event = now
        m['chunks'] += 1
        if m['first_event_ms'] is None:
            m['first_event_ms'] = elapsed
        finish, answer, reasoning = read_stream_chunk(chunk)
        if answer:
            if m['first_answer_ms'] is None:
                m['first_answer_ms'] = elapsed
            m['last_answer_ms'] = elapsed
            m['answer_chars'] += len(answer)
        if reasoning:
            if m['first_reasoning_ms'] is None:
                m['first_reasoning_ms'] = elapsed
            m['reasoning_chars'] += len(reasoning)
        if finish is not None:
            m['finish_reason'] = finish
        usage = getattr(chunk, 'usage', None)
        if usage is not None:
            def value(obj, key):
                return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)
            for field, source in [('input_tokens', 'prompt_tokens'), ('output_tokens', 'completion_tokens')]:
                count = value(usage, source)
                if count is not None:
                    m[field] = count
            count = value(value(usage, 'completion_tokens_details'), 'reasoning_tokens')
            if count is not None:
                m['reasoning_tokens'] = count
        return chunk

    def request_failed(self, error):
        self.metrics.update(status='request_error', error_kind=type(error).__name__)
        self.close()

    def close(self):
        import json
        if self._closed:
            return
        self._closed = True
        m = self.metrics
        m['total_ms'] = self._elapsed()
        m['answer_span_ms'] = (round(m['last_answer_ms'] - m['first_answer_ms'], 3)
                               if m['first_answer_ms'] is not None else None)
        m['after_last_answer_ms'] = (round(m['total_ms'] - m['last_answer_ms'], 3)
                                     if m['last_answer_ms'] is not None else None)
        close_answer_stream(self._stream)
        logger.info('PERF answer_generation %s', json.dumps(m, ensure_ascii=False))
