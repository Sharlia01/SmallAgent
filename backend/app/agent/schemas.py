"""工具调用结果；证据、业务判定与运行诊断分别表达。"""

from dataclasses import dataclass, field
from typing import Any
from domain.evidence import EvidenceCollection
from domain.assessment import EvidenceAssessment
from service.core.evidence_adapter import reference_records


@dataclass(slots=True)
class ToolResult:
    """一次工具调用的状态及其业务输出。"""

    tool_name: str
    query: str
    content: str
    evidence: EvidenceCollection = field(default_factory=EvidenceCollection)
    candidates: EvidenceCollection = field(default_factory=EvidenceCollection)
    assessment: EvidenceAssessment | None = None
    clarification: str | None = None
    success: bool = True
    error: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def failure(cls, *, tool_name: str, query: str, error: str, diagnostics=None):
        return cls(tool_name=tool_name, query=query, content="", success=False, error=error,
                   diagnostics=diagnostics or {})

    def to_dict(self) -> dict[str, Any]:
        """提供工具协议的 JSON 表示，不序列化领域对象的内部索引。"""
        return {"tool_name": self.tool_name, "query": self.query, "content": self.content,
                "evidence": reference_records(self.evidence), "success": self.success, "error": self.error,
                "clarification": self.clarification, "diagnostics": self.diagnostics,
                "assessment": self.assessment.to_dict() if self.assessment else None}
