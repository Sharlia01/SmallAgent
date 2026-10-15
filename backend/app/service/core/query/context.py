"""应用层持有本轮请求，工具共享已经消解和分析的查询上下文。"""

from dataclasses import asdict, dataclass
from domain.facts import FinancialRequirements, FinancialBinding

from domain.evidence import fingerprint
from service.core.query.contract import QueryRequest, QueryGoal, QueryUnderstanding, resolve_request
from service.core.runtime.context import current_runtime


@dataclass(slots=True)
class QueryContext:
    request: QueryRequest  #用户的请求资料包。保存原问题、消解后的问题、聊天历史等
    understanding: QueryUnderstanding | None = None  # 理解用户目标和财务事实
    _resolved: bool = False  #是否做过指代消解
    scope_goal_ids: tuple[str, ...] | None = None
    resolved_financial: FinancialRequirements | None = None # 根据已有证据补充后的财务需求
    financial_bindings: tuple[FinancialBinding, ...] = () #记录了slot和value的对应关系，以及记录了这种关系的原始片段

    @property
    def financial_requirements(self) -> FinancialRequirements | None:
        value = self.resolved_financial
        if value is None and self.understanding:
            value = self.understanding.financial
        return value.for_goals(set(self.scope_goal_ids)) if value and self.scope_goal_ids is not None else value

    def resolve(self) -> QueryRequest:
        if not self._resolved:
            self.request = resolve_request(self.request)
            self._resolved = True
        return self.request

    @property
    def question(self) -> str:
        """核验当前范围的问题；完整用户原文始终从 request 读取。"""
        if self.scope_goal_ids is not None:
            parts = []
            for goal in self.goals:
                constraints = "（" + "；".join(goal.constraints) + "）" if goal.constraints else ""
                parts.append(goal.text + constraints)
            return "；".join(parts)
        return self.request.resolved_question

    @property
    def goals(self) -> tuple[QueryGoal, ...]:
        goals = self.understanding.goals if self.understanding else ()
        if self.scope_goal_ids is None:
            return goals
        return tuple(goal for goal in goals if goal.id in self.scope_goal_ids)

    def for_goals(self, goal_ids: tuple[str, ...]) -> "QueryContext":
        """建立局部核验视图；原始请求和共享需求保持不变，局部报告不能冒充全局报告。"""
        known_ids = {goal.id for goal in self.goals}
        if not known_ids and not goal_ids and self.understanding and not self.understanding.requires_evidence:
            return self
        if not goal_ids or not set(goal_ids) <= known_ids:
            raise ValueError("工具任务必须引用已知目标")
        if set(goal_ids) == known_ids:
            return self
        return QueryContext(self.request, self.understanding, self._resolved, tuple(sorted(goal_ids)),
                            self.resolved_financial, self.financial_bindings)

    @property
    def clarification(self) -> str | None:
        if self.request.clarification:
            return self.request.clarification
        if self.understanding:
            return self.understanding.clarification
        return None

    @property
    def version(self) -> str:
        return fingerprint((self.request.to_dict(),
                            self.understanding.to_dict() if self.understanding else None,
                            self.scope_goal_ids,
                            self.financial_requirements.to_dict() if self.financial_requirements else None,
                            [asdict(binding) for binding in self.financial_bindings]))


def query_context_for(request: QueryRequest) -> QueryContext:
    """同一请求中的多次检索共享对象，业务信息不经诊断字典回传。"""
    runtime = current_runtime()
    if runtime is None:
        return QueryContext(request)
    key = ("query_context", request)
    if key not in runtime.cache:
        runtime.cache[key] = QueryContext(request)
    return runtime.cache[key]


def prepare_query_context(
    question: str,
    conversation_history: list[dict[str, str]] | None = None,
    *,
    request: QueryRequest | None = None,
    context: QueryContext | None = None,
) -> QueryContext:
    """返回已完成指代消解的上下文，调用方可直接读取 request。

    已有上下文优先于请求和文本输入，保留服务端绑定的用户问题及共享需求。
    相同运行时内复用请求缓存，重复准备不会再次执行指代消解。
    """
    if context is None:
        context = query_context_for(
            request if request is not None else QueryRequest.from_user(question, conversation_history)
        )
    context.resolve()
    return context
