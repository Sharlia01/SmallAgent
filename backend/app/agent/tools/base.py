"""Common interface implemented by Agent tools."""

from typing import Any, Protocol, runtime_checkable

from agent.schemas import ToolResult


@runtime_checkable
class AgentTool(Protocol):
    """Minimal contract used by the future Agent orchestrator."""

    name: str
    description: str
    parameters: dict[str, Any]

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the tool and return a standard result envelope."""
        ...
