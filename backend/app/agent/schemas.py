"""Provider-neutral data structures shared by all Agent tools."""

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class ToolSource:
    """One piece of evidence returned by an Agent tool.

    RAG chunks and web pages use the same shape so the future orchestrator can
    rank, render, and cite evidence without depending on a specific backend.
    Backend-specific fields belong in ``metadata``.
    """

    source_type: str
    source_id: str
    title: str
    content: str
    score: float | None = None
    url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(slots=True)
class ToolResult:
    """Standard result envelope returned by every Agent tool."""

    tool_name: str
    query: str
    content: str
    sources: list[ToolSource] = field(default_factory=list)
    success: bool = True
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def failure(
        cls,
        *,
        tool_name: str,
        query: str,
        error: str,
        metadata: dict[str, Any] | None = None,
    ) -> "ToolResult":
        """Build a failed result without raising through the Agent loop."""
        return cls(
            tool_name=tool_name,
            query=query,
            content="",
            success=False,
            error=error,
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)
