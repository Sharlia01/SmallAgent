"""Agent 工具共同实现的接口。"""

from typing import Any, Protocol, runtime_checkable

from agent.schemas import ToolResult


@runtime_checkable
class AgentTool(Protocol):
    """Agent 编排器使用的最小工具契约。"""

    name: str
    description: str
    parameters: dict[str, Any]

    def run(self, **kwargs: Any) -> ToolResult:
        """执行工具并返回标准结果封装。"""
        ...
