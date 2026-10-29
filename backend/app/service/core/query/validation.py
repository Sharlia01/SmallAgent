"""查询理解与执行计划共用的结构化校验异常。"""

from __future__ import annotations


class QueryValidationError(ValueError):
    def __init__(self, rule: str, detail: str, *, field: str | None = None,
                 node_id: str | None = None, goal_id: str | None = None,
                 expected=None, actual=None, offending=None, source=None, source_spans=None):
        self.rule = rule
        self.field = field
        self.node_id = node_id
        self.goal_id = goal_id
        self.source_spans = source_spans
        self.context = {key: value for key, value in {
            "expected": expected, "actual": actual, "offending": offending, "source": source,
        }.items() if value is not None}
        super().__init__(f"{rule}: {detail}")
