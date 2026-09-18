"""Knowledge-base search exposed through the standard Agent tool protocol."""

from typing import Any
from service.core.evidence_metadata import evidence_metadata

from agent.schemas import ToolResult, ToolSource
from service.core.retrieval import DEFAULT_PAGE_SIZE, retrieve_content
from utils import logger


MAX_RAG_RESULTS = 20
EMPTY_RESULT_MESSAGE = "未在当前用户的知识库中找到足以完整回答问题的证据。"
FAILED_RESULT_MESSAGE = "知识库检索暂时不可用。"


def _source_id(chunk: dict[str, Any], rank: int) -> str:
    """Return a stable source ID, with a deterministic fallback."""
    chunk_id = chunk.get("chunk_id")
    if chunk_id:
        return str(chunk_id)
    return f"rag-{rank}"


def _chunk_to_source(chunk: dict[str, Any], rank: int) -> ToolSource:
    """Map the existing retrieval chunk into the provider-neutral schema."""
    source_rank = chunk.get("rank") or rank
    content = chunk.get("content_with_weight") or ""
    title = chunk.get("document_name") or "未知文档"

    return ToolSource(
        source_type="knowledge_base",
        source_id=_source_id(chunk, source_rank),
        title=str(title),
        content=str(content),
        score=chunk.get("similarity"),
        metadata={
            **evidence_metadata(chunk),
            "rank": source_rank,
            "document_id": chunk.get("document_id"),
            "chunk_id": chunk.get("chunk_id"),
            "rerank_score": chunk.get("rerank_score"),
            "rrf_score": chunk.get("rrf_score"),
            "vector_similarity": chunk.get("vector_similarity"),
            "term_similarity": chunk.get("term_similarity"),
            "retrieval_ranks": chunk.get("retrieval_ranks") or {},
            "retrieval_scores": chunk.get("retrieval_scores") or {},
            "rrf_contributions": chunk.get("rrf_contributions") or {},
            "positions": chunk.get("positions") or [],
            "kb_id": chunk.get("kb_id"),
            "image_id": chunk.get("image_id"),
        },
    )


def _format_sources_for_agent(sources: list[ToolSource]) -> str:
    """Build compact evidence text that can be sent back to an LLM."""
    if not sources:
        return EMPTY_RESULT_MESSAGE

    sections = []
    for index, source in enumerate(sources, start=1):
        sections.append(f"[{index}] {source.title}\n{source.content}")
    return "\n\n".join(sections)


class RagSearchTool:
    """Search only the knowledge base owned by one authenticated user."""

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

    def __init__(self, user_id: int):
        # Bind tenant context here instead of accepting it from model output.
        # This prevents an Agent tool call from selecting another user's index.
        self._user_id = user_id

    def run(
        self,
        *,
        query: str,
        top_k: int = DEFAULT_PAGE_SIZE,
    ) -> ToolResult:
        """Search the bound user's knowledge base.

        Expected operational failures are returned as ``ToolResult`` objects so
        an Agent can observe the failure and choose a fallback path.
        """
        normalized_query = query.strip() if isinstance(query, str) else ""
        if not normalized_query:
            return ToolResult.failure(
                tool_name=self.name,
                query=normalized_query,
                error="query 不能为空。",
                metadata={"error_type": "validation_error"},
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
                metadata={"error_type": "validation_error"},
            )

        try:
            chunks = retrieve_content(
                str(self._user_id),
                normalized_query,
                page_size=top_k,
            )
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
                metadata={"error_type": type(error).__name__},
            )

        sources = [
            _chunk_to_source(chunk, rank)
            for rank, chunk in enumerate(chunks, start=1)
        ]
        return ToolResult(
            tool_name=self.name,
            query=normalized_query,
            content=_format_sources_for_agent(sources),
            sources=sources,
            metadata={"result_count": len(sources), "top_k": top_k},
        )


def search_knowledge_base(
    user_id: int,
    query: str,
    top_k: int = DEFAULT_PAGE_SIZE,
) -> ToolResult:
    """Functional entry point for callers that do not manage tool instances."""
    return RagSearchTool(user_id=user_id).run(query=query, top_k=top_k)
