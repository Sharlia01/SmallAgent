"""工具输出协议：返回执行状态和来源；需求理解、澄清与核验由 Agent 持有。"""

from dataclasses import dataclass, field
from typing import Any
from domain.evidence import EvidenceCollection
from service.core.evidence.adapter import reference_records
from service.core.query.contract import RetrievalTask


@dataclass(slots=True)
class ToolResult:
    tool_name: str
    query: str
    content: str
    evidence: EvidenceCollection = field(default_factory=EvidenceCollection)
    candidates: EvidenceCollection = field(default_factory=EvidenceCollection)
    success: bool = True
    error: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    task: RetrievalTask | None = None

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
