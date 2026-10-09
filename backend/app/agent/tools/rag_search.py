"""通过标准 Agent 工具协议提供知识库搜索。"""

from service.core.query_context import QueryContext, query_context_for
from service.core.retrieval_result import RetrievalResult, adapt_retrieval_records
from service.core.query_contract import QueryRequest
from service.core.query_expansion import validate_query_preservation

from agent.schemas import ToolResult
from service.core.retrieval import (
    DEFAULT_PAGE_SIZE,
    retrieve_evidence,
)
from utils import logger


MAX_RAG_RESULTS = 20
EMPTY_RESULT_MESSAGE = "未在当前用户的知识库中找到足以完整回答问题的证据。"
FAILED_RESULT_MESSAGE = "知识库检索暂时不可用。"


def _format_evidence(evidence) -> str:
    if not evidence:
        return EMPTY_RESULT_MESSAGE
    return "\n\n".join(f"[{index}] {item.source.name}\n{item.text}"
                        for index, item in enumerate(evidence, 1))


class RagSearchTool:
    """仅搜索已认证用户拥有的知识库。"""

    name = "search_knowledge_base"
    description = (
        "搜索当前用户的内部知识库，适合回答上传文档、内部资料和历史材料中的问题。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "根据用户问题提炼出的知识库检索词。",
            },
            "top_k": {
                "type": "integer",
                "description": "需要返回的结果数量。通常使用 5，复杂问题可适当增加。",
                "minimum": 1,
                "maximum": MAX_RAG_RESULTS,
                "default": DEFAULT_PAGE_SIZE,
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self, user_id: int, *, retriever=None):
        # 在服务端绑定租户上下文，不接受模型输出的租户选择。
        # 防止工具调用访问其他用户的知识库索引。
        self._user_id = user_id
        self._retriever = retriever

    def run(
        self,
        *,
        query: str,
        top_k: int = DEFAULT_PAGE_SIZE,
        _context: QueryContext | None = None,
    ) -> ToolResult:
        """搜索当前绑定用户的知识库。

        可预期的运行故障通过 ToolResult 返回，便于 Agent 观察失败并选择回退路径。
        """
        normalized_query = query.strip() if isinstance(query, str) else ""
        if not normalized_query or len(normalized_query) > 12000:
            return ToolResult.failure(
                tool_name=self.name,
                query=normalized_query,
                error="query 必须为 1 到 12000 个字符。",
                diagnostics={"error_type": "validation_error"},
            )

        if (
            not isinstance(top_k, int)
            or isinstance(top_k, bool)
            or not 1 <= top_k <= MAX_RAG_RESULTS
        ):
            return ToolResult.failure(
                tool_name=self.name,
                query=normalized_query,
                error=f"top_k 必须是 1 到 {MAX_RAG_RESULTS} 之间的整数。",
                diagnostics={"error_type": "validation_error"},
            )

        try:
            context = _context or query_context_for(QueryRequest.from_user(normalized_query))
            retrieve = self._retriever or retrieve_evidence
            arguments = {"context": context} if _context is not None or self._retriever is None else {}
            output = retrieve(str(self._user_id), normalized_query, page_size=top_k, **arguments)
            if isinstance(output, RetrievalResult):
                result = output
            else:
                chunks, diagnostics = output
                result = adapt_retrieval_records(context, chunks, diagnostics)
        except Exception as error:
            logger.exception(
                "Agent RAG 工具检索失败: user_id=%s, query=%s",
                self._user_id,
                normalized_query,
            )
            return ToolResult.failure(
                tool_name=self.name,
                query=normalized_query,
                error=FAILED_RESULT_MESSAGE,
                diagnostics={"error_type": type(error).__name__},
            )

        if _context is not None:
            result.diagnostics["agent_query"] = {
                "query": normalized_query,
                "validation": validate_query_preservation(context.question, normalized_query).__dict__,
                "used_as_user_question": False,
            }
        return ToolResult(
            tool_name=self.name, query=normalized_query,
            content=result.clarification or _format_evidence(result.evidence),
            evidence=result.evidence, candidates=result.candidates, assessment=result.assessment,
            clarification=result.clarification,
            diagnostics={"result_count": len(result.evidence), "top_k": top_k, **result.diagnostics},
        )

    def run_for_context(self, context: QueryContext, *, query: str, top_k: int = DEFAULT_PAGE_SIZE) -> ToolResult:
        """共享的业务上下文由服务端传入，不暴露给模型工具参数。"""
        return self.run(query=query, top_k=top_k, _context=context)


def search_knowledge_base(
    user_id: int,
    query: str,
    top_k: int = DEFAULT_PAGE_SIZE,
) -> ToolResult:
    """为不管理工具实例的调用方提供函数式入口。"""
    return RagSearchTool(user_id=user_id).run(query=query, top_k=top_k)
