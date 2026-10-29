"""当前请求的预算、缓存与上下文状态，不依赖查询业务或模型客户端。"""

from __future__ import annotations

import math
import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import Lock
from typing import Any
from uuid import uuid4

POLICY_VERSION = "query-v7"

def positive_setting(name: str, default: float, *, maximum: float = 300) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (ValueError, TypeError):
        return default
    return min(value, maximum) if math.isfinite(value) and value > 0 else default


class QueryBudgetExceeded(RuntimeError):
    pass


@dataclass
class QueryRuntime:
    trace_id: str = field(default_factory=lambda: uuid4().hex, init=False)
    timeout_seconds: float = field(default_factory=lambda: positive_setting(
        "RAG_QUERY_DEADLINE_SECONDS", 120,
    ))
    max_model_calls: int = field(default_factory=lambda: int(positive_setting(
        "RAG_QUERY_MAX_MODEL_CALLS", 16, maximum=64,
    )))
    max_retrievals: int = field(default_factory=lambda: int(positive_setting(
        "RAG_QUERY_MAX_RETRIEVALS", 12, maximum=32,
    )))
    started: float = field(default_factory=time.monotonic)
    counts: dict[str, int] = field(default_factory=dict)
    cache: dict[tuple, Any] = field(default_factory=dict, repr=False)
    _lock: Lock = field(default_factory=Lock, repr=False)

    def remaining(self) -> float:
        return max(0.0, self.timeout_seconds - (time.monotonic() - self.started))

    def reserve(self, kind: str) -> None:
        # 并发分支共用请求预算，检查剩余额度与扣减计数必须在同一临界区完成。
        with self._lock:
            limit = self.max_model_calls if kind == "model" else self.max_retrievals
            if self.remaining() <= 0 or self.counts.get(kind, 0) >= limit:
                raise QueryBudgetExceeded(f"query {kind} budget exhausted")
            self.counts[kind] = self.counts.get(kind, 0) + 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "policy_version": POLICY_VERSION,
                "trace_id": self.trace_id,
                "counts": dict(self.counts),
                "limits": {"deadline_seconds": self.timeout_seconds,
                           "model_calls": self.max_model_calls, "retrievals": self.max_retrievals},
                "remaining_ms": round(self.remaining() * 1000, 3),
            }


_runtime: ContextVar[QueryRuntime | None] = ContextVar("query_runtime", default=None)


def current_runtime() -> QueryRuntime | None:
    return _runtime.get()


@contextmanager
def query_runtime(runtime: QueryRuntime | None = None):
    existing = current_runtime()
    if existing is not None:
        yield existing
        return
    active = runtime or QueryRuntime()
    token = _runtime.set(active)
    try:
        yield active
    finally:
        _runtime.reset(token)
