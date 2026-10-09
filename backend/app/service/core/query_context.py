"""应用层持有本轮请求，工具共享已经消解和分析的查询上下文。"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from domain.evidence import fingerprint
from service.core.query_contract import QueryRequest, QueryGoal, resolve_request
from service.core.query_runtime import current_runtime

if TYPE_CHECKING:
    from service.core.query_planning import PreparedRetrievalRequest


@dataclass(slots=True)
class QueryContext:
    request: QueryRequest
    prepared: "PreparedRetrievalRequest | None" = None
    _resolved: bool = False

    def resolve(self) -> QueryRequest:
        if not self._resolved:
            self.request = resolve_request(self.request)
            self._resolved = True
        return self.request

    @property
    def question(self) -> str:
        return self.request.resolved_question

    @property
    def goals(self) -> tuple[QueryGoal, ...]:
        return self.prepared.analysis.goals if self.prepared else ()

    @property
    def clarification(self) -> str | None:
        if self.request.clarification:
            return self.request.clarification
        if self.prepared and self.prepared.route.mode == "clarify":
            return self.prepared.route.reason
        return None

    @property
    def version(self) -> str:
        return fingerprint((self.request.to_dict(), [goal.to_dict() for goal in self.goals]))


def query_context_for(request: QueryRequest) -> QueryContext:
    """同一请求中的多次检索共享对象，业务信息不经诊断字典回传。"""
    runtime = current_runtime()
    if runtime is None:
        return QueryContext(request)
    key = ("query_context", request)
    if key not in runtime.cache:
        runtime.cache[key] = QueryContext(request)
    return runtime.cache[key]
