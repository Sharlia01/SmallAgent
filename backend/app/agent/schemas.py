"""工具输出协议：返回执行状态和来源；需求理解、澄清与核验由 Agent 持有。"""

from dataclasses import dataclass, field
from typing import Any
from domain.evidence import Evidence, EvidenceCollection
from service.core.evidence.adapter import reference_records
from service.core.query.contract import RetrievalTask


@dataclass(slots=True)
class ToolResult:
    tool_name: str
    query: str
    content: str
    # 工具返回的完整来源资料；核验状态和回答引用不在此字段中区分。
    evidence: EvidenceCollection = field(default_factory=EvidenceCollection)
    success: bool = True
    error: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    task: RetrievalTask | None = None
    # 仅当前运行内可恢复的原始快照，不随 to_dict 作为可用证据导出。
    recoverable_evidence: tuple[Evidence, ...] = ()

    @classmethod
    def failure(cls, *, tool_name: str, query: str, error: str, diagnostics=None):
        return cls(tool_name, query, "", success=False, error=error, diagnostics=diagnostics or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name, "query": self.query, "content": self.content,
            "evidence": reference_records(self.evidence), "success": self.success,
            "error": self.error, "diagnostics": self.diagnostics,
            "task": self.task.to_dict() if self.task else None,
        }
