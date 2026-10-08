"""Request-local budgets, observations and bounded execution for query work.

No user data is cached across requests. Context is explicitly copied to workers;
deadlines prevent new work and bound network calls, not arbitrary local CPU work.
"""

from __future__ import annotations

import math
import json
import os
import time
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from dataclasses import dataclass, field
from functools import lru_cache, wraps
from threading import Lock, BoundedSemaphore
from typing import Any, Callable
from uuid import uuid4


POLICY_VERSION = "query-v5"
OBSERVABILITY_VERSION = "1.0"
RESPONSE_EXCERPT_CHARS = 2000


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
        "RAG_QUERY_DEADLINE_SECONDS", 45,
    ))
    max_model_calls: int = field(default_factory=lambda: int(positive_setting(
        "RAG_QUERY_MAX_MODEL_CALLS", 16, maximum=64,
    )))
    max_retrievals: int = field(default_factory=lambda: int(positive_setting(
        "RAG_QUERY_MAX_RETRIEVALS", 12, maximum=32,
    )))
    started: float = field(default_factory=time.monotonic)
    counts: dict[str, int] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    cache: dict[tuple, Any] = field(default_factory=dict, repr=False)
    _lock: Lock = field(default_factory=Lock, repr=False)

    def remaining(self) -> float:
        return max(0.0, self.timeout_seconds - (time.monotonic() - self.started))

    def reserve(self, kind: str) -> None:
        with self._lock:
            limit = self.max_model_calls if kind == "model" else self.max_retrievals
            if self.remaining() <= 0 or self.counts.get(kind, 0) >= limit:
                raise QueryBudgetExceeded(f"query {kind} budget exhausted")
            self.counts[kind] = self.counts.get(kind, 0) + 1

    def record(self, **event: Any) -> None:
        with self._lock:
            self.events.append({
                **_observation.get(), **event,
                "event_id": len(self.events) + 1,
                "elapsed_ms": round((time.monotonic() - self.started) * 1000, 3),
            })

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "policy_version": POLICY_VERSION,
                "observability_version": OBSERVABILITY_VERSION,
                "trace_id": self.trace_id,
                "counts": dict(self.counts),
                "limits": {"deadline_seconds": self.timeout_seconds,
                           "model_calls": self.max_model_calls, "retrievals": self.max_retrievals},
                "remaining_ms": round(self.remaining() * 1000, 3),
                "events": deepcopy(self.events),
            }


_runtime: ContextVar[QueryRuntime | None] = ContextVar("query_runtime", default=None)
_observation: ContextVar[dict[str, Any]] = ContextVar("query_observation", default={})
_local_inference_lock = Lock()


def current_runtime() -> QueryRuntime | None:
    return _runtime.get()


@contextmanager
def observation_context(**fields: Any):
    """Correlate attempts/nodes without passing diagnostic kwargs to providers."""
    token = _observation.set({**_observation.get(), **fields})
    try:
        yield
    finally:
        _observation.reset(token)


def record_observation(stage: str, phase: str, status: str, **fields: Any) -> None:
    runtime = current_runtime()
    if runtime is not None:
        runtime.record(stage=stage, phase=phase, status=status, **fields)


def failure_details(error: Exception, *, default_rule: str = "unexpected_error",
                    default_field: str | None = None) -> dict[str, Any]:
    """Bound exception text; never serialize provider request/response objects."""
    field_name = getattr(error, "field", None) or default_field
    detail = str(error)
    fields = {
        "error_type": type(error).__name__,
        "rule": getattr(error, "rule", None) or default_rule,
        "failure_fields": [field_name] if field_name else [],
        "detail": detail[:500], "detail_truncated": len(detail) > 500,
    }
    for name in ("node_id", "goal_id"):
        if getattr(error, name, None) is not None:
            fields[name] = getattr(error, name)
    context = getattr(error, "context", None)
    if context:
        # Keep provider objects and unbounded payloads out of diagnostics.
        fields["validation_context"] = {
            str(key): value if isinstance(value, (bool, int, float)) else str(value)[:1000]
            for key, value in context.items()
        }
    if isinstance(error, json.JSONDecodeError):
        fields.update(rule="invalid_json", failure_fields=["response.content"],
                      json_line=error.lineno, json_column=error.colno, json_offset=error.pos)
    if isinstance(error, QueryBudgetExceeded):
        fields["rule"] = "budget_exhausted"
    return fields


def response_diagnostics(content: str) -> dict[str, Any]:
    return {"response_excerpt": content[:RESPONSE_EXCERPT_CHARS],
            "response_chars": len(content),
            "response_excerpt_truncated": len(content) > RESPONSE_EXCERPT_CHARS}


@dataclass
class PlanObservation:
    mode: str
    expected_node_ids: list[str]
    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock, repr=False)


_plan_observation: ContextVar[PlanObservation | None] = ContextVar("plan_observation", default=None)


def record_execution_node(node_id: str, status: str, *, error: Exception | None = None,
                          **fields: Any) -> dict[str, Any]:
    node = {"node_id": node_id, "status": status, **fields}
    if error is not None:
        node.update(failure_details(error, default_rule="node_execution_failed"))
    active = _plan_observation.get()
    if active is not None:
        with active._lock:
            active.nodes[node_id] = node
    record_observation("plan_node_execution", "node_execution", **node)
    return node


@contextmanager
def observe_plan_execution(mode: str, node_ids: list[str], *, source: str):
    """One execution attempt; an empty successful retrieval is not a failure."""
    observation = PlanObservation(mode, node_ids)
    token = _plan_observation.set(observation)
    started = time.monotonic()
    with observation_context(execution_id=uuid4().hex, execution_mode=mode):
        try:
            yield observation
        except Exception as error:
            observation.summary.update(status="failed", execution_success=False,
                                       end_reason="budget_exhausted" if isinstance(error, QueryBudgetExceeded) else "execution_error",
                                       **failure_details(error, default_rule="plan_execution_failed"))
            raise
        finally:
            with observation._lock:
                nodes = [deepcopy(observation.nodes[node]) for node in node_ids if node in observation.nodes]
            completed = [n for n in nodes if n["status"] in {"succeeded", "empty"}]
            success = bool(node_ids) and len(completed) == len(node_ids)
            evidence = any(n.get("candidate_count", 0) for n in nodes)
            status = "succeeded" if success else "partial" if completed or evidence else "failed"
            reason = ("completed" if evidence else "empty_results") if success else "incomplete_nodes"
            observation.summary = {
                "mode": mode, "source": source, "status": status, "execution_success": success,
                "end_reason": reason, "expected_node_count": len(node_ids),
                "completed_node_count": len(completed), "nodes": nodes,
                "unreported_node_ids": [node for node in node_ids if node not in observation.nodes],
                "duration_ms": round((time.monotonic() - started) * 1000, 3),
                **observation.summary,
            }
            record_observation("plan_execution", "execution", **observation.summary)
            _plan_observation.reset(token)


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


def bounded_query(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with query_runtime() as runtime, observation_context(pipeline_id=uuid4().hex):
            started = time.monotonic()
            try:
                result = function(*args, **kwargs)
            except Exception as error:
                record_observation("query_pipeline", "pipeline", "error",
                                   end_reason="budget_exhausted" if isinstance(error, QueryBudgetExceeded) else "unhandled_error",
                                   duration_ms=round((time.monotonic() - started) * 1000, 3),
                                   **failure_details(error))
                # The caller/evaluator still receives the original exception type.
                # Preserve the trace before the request-local context is reset.
                error.query_runtime = runtime.snapshot()
                raise
            if isinstance(result, dict):
                response_mode = result.get("response_mode")
                budget_exhausted = result.get("budget_exhausted")
                sufficiency = result.get("evidence_sufficiency") or {}
                evidence = result.get("chunks")
            else:
                response_mode = getattr(result, "response_mode", None)
                if getattr(result, "stop_reason", None) == "clarification_required":
                    response_mode = "clarify"
                budget_exhausted = getattr(result, "budget_exhausted", False)
                sufficiency = getattr(result, "evidence_sufficiency", None) or {}
                evidence = getattr(result, "retrieved_content", None)
                if evidence is None:
                    evidence = any(
                        getattr(tool_result, "sources", None)
                        for tool_result in getattr(result, "tool_results", ())
                    )
            reason = ("clarification_required" if response_mode == "clarify" else
                      "budget_exhausted" if budget_exhausted else
                      "verification_unavailable" if sufficiency.get("verification_status") == "unavailable" else
                      "evidence_insufficient" if sufficiency.get("sufficient") is False else
                      "evidence_returned" if evidence else "empty_results")
            record_observation("query_pipeline", "pipeline", "completed", end_reason=reason,
                               duration_ms=round((time.monotonic() - started) * 1000, 3))
            if isinstance(result, dict):
                result["query_runtime"] = runtime.snapshot()
            elif hasattr(result, "query_runtime"):
                result.query_runtime = runtime.snapshot()
            return result
    return wrapped


def serialized_local_inference(function):
    """Avoid concurrent model cold starts and GPU memory multiplication."""
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


def model_completion(client: Any, stage: str, **kwargs: Any) -> Any:
    runtime = current_runtime()
    started = time.monotonic()
    status = "error"
    usage = None
    finish_reason = None
    sent = False
    failure = {}
    try:
        if runtime is not None:
            runtime.reserve("model")
            kwargs["timeout"] = min(float(kwargs.get("timeout", 20)), runtime.remaining())
        # with_options is supported by the configured SDK. Fake clients only
        # need chat.completions.create; no network retry may outrun the budget.
        if runtime is not None and callable(getattr(client, "with_options", None)):
            client = client.with_options(max_retries=0)
        sent = True
        result = client.chat.completions.create(**kwargs)
        status = "ok"
        usage = getattr(result, "usage", None)
        choices = getattr(result, "choices", None) or []
        finish_reason = getattr(choices[0], "finish_reason", None) if choices else None
        return result
    except Exception as error:
        status = "error" if sent else "blocked"
        failure = failure_details(error, default_rule="provider_request_failed" if sent else "request_not_started")
        raise
    finally:
        if runtime is not None:
            runtime.record(
                stage=stage, phase="request", status=status, request_sent=sent,
                duration_ms=round((time.monotonic() - started) * 1000, 3),
                total_tokens=getattr(usage, "total_tokens", None),
                model=kwargs.get("model"),
                finish_reason=finish_reason,
                end_reason="response_received" if status == "ok" else failure.get("rule"),
                **failure,
            )


def observe_contract(stage: str, content: str, *, parse: Callable[[str], Any],
                     validate: Callable[[Any], Any], finish_reason: str | None = None) -> Any:
    """Separate decoding/schema construction from semantic acceptance."""
    from service.core.query_validation import QueryValidationError
    started = time.monotonic()
    try:
        value = parse(content)
    except (ValueError, TypeError, KeyError) as error:
        record_observation(stage + "_parse", "parse", "rejected", finish_reason=finish_reason,
                           end_reason="parse_failed", **failure_details(error, default_rule="invalid_schema", default_field="response"),
                           **response_diagnostics(content), duration_ms=round((time.monotonic() - started) * 1000, 3))
        record_observation(stage + "_validation", "semantic_validation", "skipped",
                           end_reason="parse_failed", finish_reason=finish_reason)
        raise
    record_observation(stage + "_parse", "parse", "accepted", end_reason="parsed",
                       finish_reason=finish_reason, duration_ms=round((time.monotonic() - started) * 1000, 3))
    started = time.monotonic()
    try:
        if finish_reason == "length":
            raise QueryValidationError("response_truncated", "model output reached the token limit", field="response.finish_reason")
        accepted = validate(value)
    except (ValueError, TypeError, KeyError) as error:
        record_observation(stage + "_validation", "semantic_validation", "rejected",
                           finish_reason=finish_reason, end_reason="validation_failed",
                           **failure_details(error, default_rule="invalid_contract", default_field="response"),
                           **response_diagnostics(content), duration_ms=round((time.monotonic() - started) * 1000, 3))
        raise
    record_observation(stage + "_validation", "semantic_validation", "accepted",
                       finish_reason=finish_reason, end_reason="validated",
                       duration_ms=round((time.monotonic() - started) * 1000, 3))
    return accepted


def validated_completion(client: Any, stage: str, *, parse: Callable[[str], Any],
                         validate: Callable[[Any], Any], **kwargs: Any) -> Any:
    """One semantic repair attempt, charged to the same request budget.

    Transport errors are not retried. Provider success and contract acceptance
    are separate observations; a failed second validation propagates normally.
    """
    from service.core.query_validation import QueryValidationError
    messages = list(kwargs.pop("messages"))
    operation_id = uuid4().hex
    for attempt in (1, 2):
        with observation_context(operation_id=operation_id, attempt=attempt, source="model"):
            try:
                completion = model_completion(client, stage, messages=messages, **kwargs)
            except Exception as error:
                for suffix, phase in (("_parse", "parse"), ("_validation", "semantic_validation")):
                    record_observation(stage + suffix, phase, "skipped", end_reason="request_failed")
                record_observation(stage + "_completion", "completion", "failed",
                                   end_reason="budget_exhausted" if isinstance(error, QueryBudgetExceeded) else "request_failed",
                                   **failure_details(error))
                raise
            choices = getattr(completion, "choices", None) or []
            choice = choices[0] if choices else None
            finish_reason = getattr(choice, "finish_reason", None)
            raw_content = getattr(getattr(choice, "message", None), "content", None)
            content = raw_content.strip() if isinstance(raw_content, str) else ""
            def parse_response(text):
                if not choices:
                    raise QueryValidationError("missing_choices", "model returned no choices", field="response.choices")
                if not text:
                    raise QueryValidationError("empty_content", "model returned empty content", field="response.content")
                return parse(text)
            try:
                value = observe_contract(stage, content, parse=parse_response, validate=validate,
                                         finish_reason=finish_reason)
            except (ValueError, TypeError, KeyError) as error:
                record_observation(stage + "_completion", "completion", "retrying" if attempt == 1 else "failed",
                                   end_reason="repair_requested" if attempt == 1 else "repair_exhausted",
                                   finish_reason=finish_reason, **failure_details(error))
                if attempt == 2:
                    raise
                if finish_reason == "length":
                    kwargs["max_tokens"] = min(4096, int(kwargs.get("max_tokens", 1800)) * 2)
                messages = [*messages,
                            {"role": "assistant", "content": content[:6000] or "{}"},
                            {"role": "user", "content":
                             "上一份输出未通过校验：" + json.dumps(failure_details(error), ensure_ascii=False) +
                             "。请根据原始问题修正并返回完整JSON。保留未知关系，不要猜测答案；"
                             "分别绑定各目标的主体、时间和指标，允许等价专业术语。"}]
            else:
                record_observation(stage + "_completion", "completion", "accepted",
                                   end_reason="accepted_after_repair" if attempt > 1 else "accepted",
                                   finish_reason=finish_reason)
                return value


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
    """Execute bounded waves, retaining completed siblings after an error."""
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
