"""Agent 的执行反馈循环：累积证据、核验缺口，再决定是否需要补查。"""

import logging
from dataclasses import dataclass, field
from typing import Any

from agent.executor import ToolExecutor, TaskExecution
from agent.planner import AgentPlanner
from agent.schemas import ToolResult
from agent.financial_bindings import bind_financial_requirements
from domain.assessment import EvidenceAssessment
from domain.evidence import EvidenceCollection
from service.core.evidence.assessment import finalize_assessment
from service.core.evidence.verification import verify_evidence
from service.core.query.context import QueryContext
from service.core.query.contract import QueryGoal, QueryUnderstanding
from service.core.runtime.context import QueryBudgetExceeded, positive_setting
from service.core.runtime.execution import bounded_query

logger = logging.getLogger(__name__)
DEFAULT_MAX_STEPS = 3


@dataclass
class AgentRun:
    tool_results: list[ToolResult] = field(default_factory=list)  # 当前轮次已执行的全部工具结果，包含成功、失败和缓存
    steps: int = 0
    stop_reason: str = "completed"
    evidence: EvidenceCollection = field(default_factory=EvidenceCollection)  #当前已取得的全部证据，包括会话文档和历次工具返回的证据
    assessment: EvidenceAssessment | None = None  #对当前累计证据能否满足完整用户需求的最终核验报告
    executions: dict[str, TaskExecution] = field(default_factory=dict)
    clarification: str | None = None
    query: QueryContext | None = None


def collect_tool_evidence(results: list[ToolResult]) -> EvidenceCollection:
    evidence = EvidenceCollection()
    for result in results:
        evidence.merge(result.candidates)
        evidence.merge(result.evidence)
    return evidence


class AgentLoop:
    def __init__(self, *, client: Any, model: str, tools: list, max_steps: int | None = None):
        if max_steps is None:
            max_steps = int(positive_setting("AGENT_MAX_STEPS", DEFAULT_MAX_STEPS, maximum=10))
        if type(max_steps) is not int or not 1 <= max_steps <= 10:
            raise ValueError("max_steps 必须为1到10的整数")
        self.executor = ToolExecutor(tools)
        self.planner = AgentPlanner(client=client, model=model, tool_definitions=self.executor.definitions())
        self.max_steps = max_steps

    @staticmethod
    def _assess(context: QueryContext, run: AgentRun) -> EvidenceAssessment:
        """核验故障不能被解释为资料缺失；保留故障报告并停止无效补查。"""
        semantic = bool(context.understanding.requires_evidence)
        try:
            # 需求保持不变；执行所得的主体/期间绑定形成新的核验版本。
            bind_financial_requirements(context, run.evidence, run.executions)
            return verify_evidence(context, run.evidence, previous=run.assessment, require_semantic=semantic)
        except Exception as error:
            return finalize_assessment(context, run.evidence, EvidenceAssessment(
                sufficient=False, reason="证据核验未能完成", source="fallback",
                verification_status="unavailable", diagnostics={"failure_kind": type(error).__name__},
            ), require_semantic=semantic)

    @bounded_query
    def run(self, *, context: QueryContext, session_context: str | None = None,
            initial_evidence: EvidenceCollection | None = None) -> AgentRun:
        run = AgentRun(evidence=initial_evidence.copy() if initial_evidence is not None else EvidenceCollection(), query=context)
        cache = {}
        try:
            decision = self.planner.analyze_and_plan(context, session_context=session_context)
        except Exception as error:
            logger.warning("Agent initial planning failed: %s", type(error).__name__)
            if context.request.clarification:
                run.clarification = context.request.clarification
                run.stop_reason = "clarification_required"
                return run
            # 不回退到另一套 RAG 意图识别。保留完整需求，明确报告规划失败。
            context.understanding = QueryUnderstanding(
                (QueryGoal("whole_question", context.question, source_spans=(context.question,)),),
                source="fallback", failure_reason=type(error).__name__,
            )
            run.stop_reason = "budget_exhausted" if isinstance(error, QueryBudgetExceeded) else "planning_failed"
            run.assessment = finalize_assessment(context, run.evidence, EvidenceAssessment(
                sufficient=False, reason="查询规划未能完成，请稍后重试", source="fallback",
                verification_status="unavailable", diagnostics={"failure_kind": type(error).__name__},
            ), require_semantic=True)
            return run
        # 是否需要用户澄清；如果需要，直接返回，不执行任何工具。
        if context.clarification or decision.plan.clarification:
            run.clarification = context.clarification or decision.plan.clarification
            run.stop_reason = "clarification_required"
            return run
        plan = decision.plan
        # 空计划也进入同一轮核验：会话证据可能已足够，无须重复核验或另设回答捷径。
        for step in range(1, self.max_steps + 1):
            run.steps = step
            before = run.evidence.version
            executions = self.executor.execute_plan(plan, context, run.executions, cache)
            for execution in executions:
                if execution.result is not None:
                    run.tool_results.append(execution.result)
                    run.evidence.merge(execution.result.candidates)
                    run.evidence.merge(execution.result.evidence)
            # 核验证据
            run.assessment = self._assess(context, run)
            if run.assessment.sufficient:
                run.stop_reason = "evidence_ready"
                return run
            
            #这一轮结束后，Agent 是否应该停止继续补查，以及记录什么停止原因
            if run.assessment.verification_status != "completed" or run.assessment.source == "fallback":
                exhausted = (run.assessment.diagnostics.get("failure_kind") == "QueryBudgetExceeded"
                             or any(execution.status == "budget_exhausted" for execution in executions))
                run.stop_reason = "budget_exhausted" if exhausted else "verification_failed"
                return run
            if any(execution.status == "budget_exhausted" for execution in executions):
                run.stop_reason = "budget_exhausted"
                return run
            if executions and all(execution.status in {"cached", "blocked"} for execution in executions) and before == run.evidence.version:
                run.stop_reason = "no_progress"  #没有取得进展，就停止重复尝试
                return run
            if step == self.max_steps:
                run.stop_reason = "max_steps"
                return run
            # 重新规划
            try:
                plan = self.planner.replan(
                    context, previous=tuple(state.task for state in run.executions.values()),
                    observations=[state.observation(include_content=index >= len(run.executions) - 8)
                                  for index, state in enumerate(run.executions.values())],
                    assessment=run.assessment.to_dict(),
                )
            except Exception as error:
                # 后续规划失败仍保留先前证据，不能重启整条管线丢失已完成工作。
                logger.warning("Agent replanning failed: %s", type(error).__name__)
                run.stop_reason = "budget_exhausted" if isinstance(error, QueryBudgetExceeded) else "planning_failed"
                return run
            if plan.clarification:
                run.clarification = plan.clarification
                run.stop_reason = "clarification_required"
                return run
            if not plan.tasks:
                run.stop_reason = "no_plan"
                return run
        return run
