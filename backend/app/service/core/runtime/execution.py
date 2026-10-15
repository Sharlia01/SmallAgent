"""请求执行边界与有界并发；截止时间不能强制中断已经运行的本地计算。"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, wait
from contextvars import copy_context
from functools import lru_cache, wraps
from threading import Lock, BoundedSemaphore
from typing import Any, Callable

from service.core.runtime.context import (
    QueryBudgetExceeded, current_runtime, positive_setting, query_runtime,
)

logger = logging.getLogger(__name__)
_local_inference_lock = Lock()


def bounded_query(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with query_runtime() as runtime:
            try:
                result = function(*args, **kwargs)
            except Exception as error:
                logger.exception("Query pipeline %s failed", function.__name__)
                # 抛出原异常，附上预算快照，便于评测确认是否因额度耗尽而失败。
                error.query_runtime = runtime.snapshot()
                raise
            if isinstance(result, dict):
                result["query_runtime"] = runtime.snapshot()
            elif hasattr(result, "query_runtime"):
                result.query_runtime = runtime.snapshot()
            elif hasattr(result, "diagnostics"):
                result.diagnostics["query_runtime"] = runtime.snapshot()
            elif hasattr(result, "raw_retrieval"):
                result.raw_retrieval["query_runtime"] = runtime.snapshot()
            return result
    return wrapped


def serialized_local_inference(function):
    """避免模型并发冷启动和重复占用 GPU 内存。"""
    @wraps(function)
    def wrapped(*args, **kwargs):
        runtime = current_runtime()
        if runtime is not None and runtime.remaining() <= 0:
            raise QueryBudgetExceeded("local inference deadline exceeded")
        acquired = (_local_inference_lock.acquire(timeout=runtime.remaining())
                    if runtime is not None else _local_inference_lock.acquire())
        if not acquired:
            raise QueryBudgetExceeded("local inference queue deadline exceeded")
        try:
            return function(*args, **kwargs)
        finally:
            _local_inference_lock.release()
    return wrapped


@lru_cache(maxsize=1)
def _executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(
        max_workers=int(positive_setting("RAG_QUERY_WORKERS", 8, maximum=32)),
        thread_name_prefix="rag-query",
    )


@lru_cache(maxsize=1)
def _executor_slots() -> BoundedSemaphore:
    return BoundedSemaphore(int(positive_setting("RAG_QUERY_WORKERS", 8, maximum=32)) * 2)


def run_independent(tasks: dict[str, Callable[[], Any]]) -> dict[str, Any | Exception]:
    """分轮执行有界并发任务，发生错误时保留已完成的同级任务结果。"""
    concurrency = int(positive_setting("RAG_QUERY_CONCURRENCY", 3, maximum=8))
    results: dict[str, Any | Exception] = {}
    items = list(tasks.items())
    for offset in range(0, len(items), concurrency):
        runtime = current_runtime()
        if runtime is not None and runtime.remaining() <= 0:
            results.update({key: QueryBudgetExceeded("query deadline exceeded")
                            for key, _ in items[offset:]})
            break
        pending = {}
        for key, task in items[offset:offset + concurrency]:
            slots = _executor_slots()
            if not slots.acquire(timeout=runtime.remaining() if runtime else 30):
                results[key] = QueryBudgetExceeded("query worker queue is full")
                continue
            try:
                future = _executor().submit(copy_context().run, task)
            except Exception:
                slots.release()
                raise
            future.add_done_callback(lambda _future, slots=slots: slots.release())
            pending[future] = key
        done, unfinished = wait(pending, timeout=runtime.remaining() if runtime else None)
        for future in done:
            try:
                results[pending[future]] = future.result()
            except Exception as error:
                results[pending[future]] = error
        for future in unfinished:
            future.cancel()
            results[pending[future]] = QueryBudgetExceeded("query deadline exceeded")
    return {key: results[key] for key in tasks}
