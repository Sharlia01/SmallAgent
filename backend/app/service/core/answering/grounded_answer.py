"""供离线评测使用的知识库单轮执行，不写入会话等外部状态。"""

from __future__ import annotations

from typing import Any, Callable

from service.core.answering.turn import GroundedTurnResult, DEFAULT_ANSWER_MODEL

from service.core.retrieval.result import RetrievalResult
from service.core.retrieval.pipeline import retrieve_evidence


def execute_grounded_turn(
    *, index_names: str | list[str], question: str,
    conversation_history: list[dict[str, str]] | None = None,
    answer_client: Any | None = None, answer_model: str | None = None,
    retrieval_function: Callable[..., RetrievalResult | dict[str, Any]] = retrieve_evidence,
    retrieval_options: dict[str, Any] | None = None,
) -> GroundedTurnResult:
    """独立知识库问答也走同一 Agent；仅注册知识库工具，不另建语义流程。"""
    from service.core.answering.turn import execute_agent_turn
    return execute_agent_turn(
        index_names=index_names, question=question, conversation_history=conversation_history,
        answer_client=answer_client, answer_model=answer_model,
        retrieval_function=retrieval_function, retrieval_options=retrieval_options,
    )
