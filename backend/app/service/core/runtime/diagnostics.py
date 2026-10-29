"""整理错误信息和最终计划执行结果，不积累逐步事件历史。"""

from __future__ import annotations

import json
import logging
import time
from copy import deepcopy
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from service.core.runtime.context import QueryBudgetExceeded

logger = logging.getLogger(__name__)


def failure_details(error: Exception, *, default_rule: str = "unexpected_error",
                    default_field: str | None = None) -> dict[str, Any]:
    """限制异常文本长度，不序列化提供方的请求或响应对象。"""
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
    source_spans = getattr(error, "source_spans", None)
    if source_spans is not None:
        fields["source_spans"] = [span[:1000] for span in source_spans[:8]]
        fields["source_spans_truncated"] = len(source_spans) > 8 or any(len(span) > 1000 for span in source_spans)
    context = getattr(error, "context", None)
    if context:
        # 诊断信息不包含提供方对象或大小不受限的数据。
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


@dataclass
class PlanResult:
    mode: str
    expected_node_ids: list[str]
    nodes: dict[str, dict[str, Any]] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock, repr=False)


_plan_result: ContextVar[PlanResult | None] = ContextVar("plan_result", default=None)


def set_node_result(node_id: str, status: str, *, error: Exception | None = None,
                    **fields: Any) -> dict[str, Any]:
    node = {"node_id": node_id, "status": status, **fields}
    if error is not None:
        node.update(failure_details(error, default_rule="node_execution_failed"))
    active = _plan_result.get()
    if active is not None:
        with active._lock:
            active.nodes[node_id] = node
    return node


@contextmanager
def collect_plan_result(mode: str, node_ids: list[str], *, source: str):
    """汇总每个节点的最终结果；同一节点只保留一个状态，不记录过程事件。"""
    result = PlanResult(mode, node_ids)
    token = _plan_result.set(result)
    started = time.monotonic()
    try:
        yield result
    except Exception as error:
        logger.warning("%s retrieval plan failed: %s: %s", mode, type(error).__name__, str(error)[:500])
        result.summary.update(status="failed", execution_success=False,
                              end_reason="budget_exhausted" if isinstance(error, QueryBudgetExceeded) else "execution_error",
                              **failure_details(error, default_rule="plan_execution_failed"))
        raise
    finally:
        with result._lock:
            nodes = [deepcopy(result.nodes[node]) for node in node_ids if node in result.nodes]
        completed = [n for n in nodes if n["status"] in {"succeeded", "empty"}]
        success = bool(node_ids) and len(completed) == len(node_ids)
        evidence = any(n.get("candidate_count", 0) for n in nodes)
        status = "succeeded" if success else "partial" if completed or evidence else "failed"
        reason = ("completed" if evidence else "empty_results") if success else "incomplete_nodes"
        result.summary = {
            "mode": mode, "source": source, "status": status, "execution_success": success,
            "end_reason": reason, "expected_node_count": len(node_ids),
            "completed_node_count": len(completed), "nodes": nodes,
            "unreported_node_ids": [node for node in node_ids if node not in result.nodes],
            "duration_ms": round((time.monotonic() - started) * 1000, 3),
            **result.summary,
        }
        _plan_result.reset(token)
