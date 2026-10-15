"""应用编排边界：准备请求，运行 Agent，再用已核验的证据构建回答。"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any
from domain.evidence import EvidenceCollection, AnswerContext
from domain.assessment import EvidenceAssessment
from service.core.evidence.adapter import reference_records
from service.core.query.context import QueryContext, prepare_query_context
from service.core.answering.answer_context import build_answer_context, build_answer_messages, answer_documents, session_evidence
from service.core.runtime.execution import bounded_query
from service.core.runtime.context import current_runtime
from service.core.answering.policy import render_evidence_refusal

from agent.loop import AgentLoop, AgentRun
from agent.tools.base import AgentTool



logger = logging.getLogger(__name__)

@dataclass(slots=True)
class AgentExecution:
    """完整执行结果，证据只有一个主集合，前端数据在输出边界生成。"""

    agent_run: AgentRun
    evidence: EvidenceCollection = field(default_factory=EvidenceCollection)
    assessment: EvidenceAssessment | None = None
    context: AnswerContext | None = None
    answer_messages: list[dict[str, str]] = field(default_factory=list)
    answer_stream: Any | None = None
    direct_answer: str | None = None
    response_mode: str = "answer"
    query_runtime: dict[str, Any] = field(default_factory=dict)

    @property
    def retrieved_content(self) -> list[dict[str, Any]]:
        """评测及推荐问题所需的工具证据视图，不能作为回答模型输入。"""
        selected = EvidenceCollection(item for item in self.evidence if item.source.kind != "session_document")
        for item in selected:
            for trace in self.evidence.traces(item.id):
                selected.add(item, trace)
        return reference_records(selected)

    @property
    def response_documents(self) -> list[dict[str, Any]]:
        if self.context and self.assessment:
            return answer_documents(self.context, self.evidence, self.assessment)
        return []

    @property
    def evidence_sufficiency(self) -> dict[str, Any] | None:
        return self.assessment.to_dict() if self.assessment else None


class AgentOrchestrator:
    """协调规划模型、工具和回答模型。"""

    def __init__(
        self,
        *,
        planner_client: Any,
        planner_model: str,
        answer_client: Any,
        answer_model: str,
        tools: list[AgentTool],
        max_steps: int | None = None,
    ):
        self._agent = AgentLoop(
            client=planner_client,
            model=planner_model,
            tools=tools,
            max_steps=max_steps,
        )
        self._answer_client = answer_client
        self._answer_model = answer_model

    def _run_agent(self, *, session_id: str, session_context: str | None, context: QueryContext) -> AgentRun:
        return self._agent.run(
            context=context, session_context=session_context,
            initial_evidence=session_evidence(session_id, session_context),
        )

    def _run_agent_with_timing(
        self,
        *,
        session_id: str,
        session_context: str | None,
        context: QueryContext,
    ) -> AgentRun:
        """记录 Agent 理解、执行、核验和补查的整体耗时。"""
        started_at = time.perf_counter()
        agent_run = self._run_agent(
            session_id=session_id,
            session_context=session_context,
            context=context,
        )
        logger.info(
            "PERF session_id=%s stage=agent_run_total status=ok "
            "steps=%s stop_reason=%s duration_ms=%.1f",
            session_id,
            agent_run.steps,
            agent_run.stop_reason,
            (time.perf_counter() - started_at) * 1000,
        )
        return agent_run

    @staticmethod
    def _refuse(agent_run: AgentRun, evidence: EvidenceCollection, assessment: EvidenceAssessment) -> AgentExecution:
        return AgentExecution(agent_run=agent_run, evidence=evidence, assessment=assessment,
                              direct_answer=render_evidence_refusal(assessment.to_dict()), response_mode="refuse")

    @staticmethod
    def _prepare_answer(query: QueryContext, evidence: EvidenceCollection, assessment: EvidenceAssessment,
                        agent_run: AgentRun, history: list[dict[str, str]]) -> AgentExecution:
        """回答提示与引用视图只消费同一份已经通过的核验报告。"""
        context = build_answer_context(query, evidence, assessment)
        return AgentExecution(agent_run=agent_run, evidence=evidence, assessment=assessment, context=context,
                              answer_messages=build_answer_messages(context, history))

    def _start_answer_stream(
        self,
        *,
        session_id: str,
        messages: list[dict[str, str]],
        reference_count: int,
    ) -> Any:
        """启动回答流，记录请求耗时，并保持原始异常向上传递。"""
        answer_request_started_at = time.perf_counter()
        try:
            answer_stream = self._answer_client.chat.completions.create(
                model=self._answer_model,
                messages=messages,
                stream=True,
                temperature=0,
            )
        except Exception:
            logger.info(
                "PERF session_id=%s stage=answer_request model=%s "
                "status=error duration_ms=%.1f",
                session_id,
                self._answer_model,
                (time.perf_counter() - answer_request_started_at) * 1000,
            )
            raise
        logger.info(
            "PERF session_id=%s stage=answer_request model=%s status=ok "
            "references=%s duration_ms=%.1f",
            session_id,
            self._answer_model,
            reference_count,
            (time.perf_counter() - answer_request_started_at) * 1000,
        )
        return answer_stream

    @bounded_query
    def run(self, *, session_id: str, question: str, session_context: str | None = None,
            conversation_history: list[dict[str, str]] | None = None) -> AgentExecution:
        """按查询准备、工具执行、集合核验和回答构建的顺序协调流程。"""

        # 指代消解保留历史来源；需求理解和初始规划在 Agent 内一次完成。
        query = prepare_query_context(question, conversation_history)
        # 运行 Agent 决策和工具循环
        agent_run = self._run_agent_with_timing(
            session_id=session_id, session_context=session_context, context=query,
        )
        clarification = agent_run.clarification
        if clarification:  # 执行反馈可能揭示需要用户确认的新歧义
            return AgentExecution(agent_run, direct_answer=clarification, response_mode="clarify")
        evidence = agent_run.evidence
        assessment = agent_run.assessment
        if not assessment.sufficient:
            return self._refuse(agent_run, evidence, assessment)

        # 准备回答模型的输入，包含引用视图和提示消息
        history = [{"role": role, "content": content} for role, content in query.request.history]
        execution = self._prepare_answer(query, evidence, assessment, agent_run, history)
        execution.answer_stream = self._start_answer_stream(
            session_id=session_id, messages=execution.answer_messages,
            reference_count=len(execution.context.citations),
        )
        execution.query_runtime = current_runtime().snapshot()
        return execution
