"""执行已校验的检索依赖图，并保留失败路径中已经取得的证据。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from service.core.query_planning import QuerySubqueryPlan, validate_dependency_graph
from service.core.query_runtime import (
    QueryBudgetExceeded, run_independent, observation_context,
    record_execution_node, failure_details,
)
from service.core.query_validation import QueryValidationError, missing_constraints
from service.core.sequential_retrieval import BridgeExtraction, extract_bridge_value, resolve_query_template


@dataclass
class NodeResult:
    step: QuerySubqueryPlan
    status: str
    query: str = ""
    result: dict[str, Any] = field(default_factory=dict)
    bridge: BridgeExtraction | None = None
    error: str | None = None
    failure: dict[str, Any] = field(default_factory=dict)
    duration_ms: float | None = None

    def diagnostic(self) -> dict[str, Any]:
        return {
            "id": self.step.id, "goal_ids": self.step.goal_ids,
            "depends_on": self.step.depends_on, "query": self.query,
            "status": self.status, "error": self.error,
            "bridge": self.bridge.to_dict() if self.bridge else None,
            "candidate_count": len(self.result.get("chunks") or []),
            "end_reason": self.failure.get("rule") or (
                self.error if self.status == "blocked" else {
                    "succeeded": "retrieved", "empty": "empty_results",
                    "failed": "retrieval_failed", "budget_exhausted": "budget_exhausted",
                    "unresolved_bridge": "dependency_has_no_grounded_value",
                }.get(self.status, self.status)
            ),
            "duration_ms": self.duration_ms,
            **self.failure,
        }

    def record(self) -> None:
        diagnostic = self.diagnostic()
        diagnostic.pop("id")
        diagnostic.pop("error")
        diagnostic.pop("node_id", None)
        status = diagnostic.pop("status")
        record_execution_node(self.step.id, status, **diagnostic)


def execute_graph(
    steps: list[QuerySubqueryPlan], *, original_question: str,
    retrieve: Callable[[str, list[str] | None], dict[str, Any]],
    bridge_extractor: Callable[..., BridgeExtraction | None] = extract_bridge_value,
) -> list[NodeResult]:
    waves = validate_dependency_graph(steps)
    completed: dict[str, NodeResult] = {}
    for wave in waves:
        tasks = {}
        for step in wave:
            parents = [completed[parent] for parent in step.depends_on]
            if any(parent.bridge is None for parent in parents):
                completed[step.id] = NodeResult(step, "blocked", error="dependency_has_no_grounded_value")
                completed[step.id].record()
                continue
            slots = {parent.bridge.slot: parent.bridge.value for parent in parents if parent.bridge}
            try:
                query = resolve_query_template(step.query_template, slots) if step.query_template else step.query
                missing = missing_constraints(step.required_constraints, query)
                if missing:
                    raise QueryValidationError("resolved_constraint_dropped", "resolved query lost goal constraints: " + ", ".join(missing), field="resolved_query", node_id=step.id)
                doc_ids = None
                if step.inherit_document_scope:
                    scopes = {parent.bridge.document_id for parent in parents if parent.bridge}
                    if len(scopes) != 1 or any(not isinstance(scope, str) or not scope.strip() for scope in scopes):
                        raise QueryValidationError("invalid_document_scope", "inherited document scopes are missing or conflicting", field="doc_ids", node_id=step.id)
                    doc_ids = sorted(scopes)
            except ValueError as error:
                completed[step.id] = NodeResult(step, "blocked", error=str(error)[:500], failure=failure_details(error, default_rule="invalid_query_template", default_field="query_template"))
                completed[step.id].record()
                continue

            def execute(step=step, query=query, doc_ids=doc_ids):
                started = time.monotonic()
                with observation_context(node_id=step.id, goal_ids=step.goal_ids):
                    result = retrieve(query, doc_ids)
                    state = NodeResult(step, "succeeded" if result.get("chunks") else "empty", query, result)
                    if step.output_slot and result.get("chunks"):
                        try:
                            state.bridge = bridge_extractor(
                                slot=step.output_slot, original_question=original_question,
                                first_query=query, chunks=result["chunks"],
                            )
                        except Exception as error:
                            state.error = type(error).__name__
                            state.failure = failure_details(error, default_rule="bridge_extraction_failed", default_field="output_slot")
                        if state.bridge is None:
                            state.status = "unresolved_bridge"
                    state.duration_ms = round((time.monotonic() - started) * 1000, 3)
                    return state
            tasks[step.id] = execute
        for node_id, value in run_independent(tasks).items():
            if isinstance(value, Exception):
                step = next(item for item in wave if item.id == node_id)
                completed[node_id] = NodeResult(
                    step, "budget_exhausted" if isinstance(value, QueryBudgetExceeded) else "failed",
                    error=type(value).__name__,
                    failure=failure_details(value, default_rule="node_execution_failed"),
                )
            else:
                completed[node_id] = value
            completed[node_id].record()
    return [completed[step.id] for step in steps]
