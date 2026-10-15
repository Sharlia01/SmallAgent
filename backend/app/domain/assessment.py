"""各核验阶段共用的证据报告，直接保存结论、事实与适用范围。"""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal

from domain.evidence import fingerprint
from domain.facts import FinancialFactAssessment

EvidenceAssessmentSource = Literal["model", "rule", "fallback", "disabled"]


@dataclass(frozen=True, slots=True)
class EvidenceAssessment:
    """一份核验报告；完成范围绑定后才能复用或构建回答上下文。"""

    sufficient: bool
    reason: str
    missing_requirements: list[str] = field(default_factory=list)
    supporting_chunk_indices: list[int] = field(default_factory=list)
    source: EvidenceAssessmentSource = "rule"
    evaluated_chunk_count: int = 0
    evaluated_chunk_ids: list[str] = field(default_factory=list)
    goal_coverage: list[dict[str, Any]] = field(default_factory=list)
    used_derived_evidence_ids: list[str] = field(default_factory=list)
    verification_status: str = "completed"
    facts: FinancialFactAssessment = field(default_factory=FinancialFactAssessment)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    query_version: str = ""
    evidence_version: str = ""
    policy_version: str = ""
    # 底层评估尚未绑定模式；完整核验报告记录实际是否要求语义检查。
    require_semantic: bool | None = None
    _report_version: str = field(init=False, repr=False)

    def __post_init__(self):
        # 报告保存独立快照；抽取结果、模型响应或调用方列表的后续修改不能污染报告。
        for name in ("missing_requirements", "supporting_chunk_indices", "evaluated_chunk_ids",
                     "goal_coverage", "used_derived_evidence_ids", "facts", "diagnostics"):
            object.__setattr__(self, name, deepcopy(getattr(self, name)))
        object.__setattr__(self, "_report_version", self._current_report_version())

    @property
    def supporting_fact_ids(self) -> list[str]:
        """事实选择只保存在 facts 中，避免维护第二份可变支持集合。"""
        return list(self.facts.selected_fact_ids)

    @property
    def derived_evidence(self) -> list[dict[str, Any]]:
        """计算结果与输入事实由同一个事实汇总对象管理。"""
        return self.facts.derived_evidence

    def _current_report_version(self) -> str:
        return fingerprint((self.to_dict(), self.facts.to_dict(), self.diagnostics,
                            self.query_version, self.evidence_version, self.policy_version, self.require_semantic))

    def matches(self, query_version: str, evidence_version: str, policy_version: str, *,
                require_semantic: bool | None = None) -> bool:
        # 底层解析产生的报告尚未绑定查询，不能冒充已经完成最终核验的报告。
        return (bool(self.query_version and self.evidence_version and self.policy_version)
                and self.require_semantic is not None
                and (require_semantic is None or self.require_semantic == require_semantic)
                and (self.query_version, self.evidence_version, self.policy_version)
                == (query_version, evidence_version, policy_version)
                and self._report_version == self._current_report_version())

    def to_dict(self) -> dict[str, Any]:
        """在 JSON 边界保留现有输出字段；业务流程始终使用结构化 facts。"""
        return deepcopy({
            "sufficient": self.sufficient,
            "reason": self.reason,
            "missing_requirements": self.missing_requirements,
            "supporting_chunk_indices": self.supporting_chunk_indices,
            "source": self.source,
            "evaluated_chunk_count": self.evaluated_chunk_count,
            "evaluated_chunk_ids": self.evaluated_chunk_ids,
            "goal_coverage": self.goal_coverage,
            "derived_evidence": self.derived_evidence,
            "used_derived_evidence_ids": self.used_derived_evidence_ids,
            "verification_status": self.verification_status,
            "failure_kind": self.diagnostics.get("failure_kind"),
            "failure": self.diagnostics.get("failure"),
            "attempts": self.diagnostics.get("attempts", 0),
            "financial_fact_bundle": self.facts.to_dict() if self.facts.applicable else {},
            "supporting_fact_ids": self.supporting_fact_ids,
        })
