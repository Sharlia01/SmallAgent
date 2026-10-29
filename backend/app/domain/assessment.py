"""统一证据核验报告，保存目标覆盖、来源引用和适用范围。"""

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal

from domain.evidence import fingerprint

EvidenceAssessmentSource = Literal["model", "rule", "fallback", "disabled"]


@dataclass(frozen=True, slots=True)
class EvidenceAssessment:
    """一份核验报告；完成范围绑定后才能复用或构建回答上下文。"""

    sufficient: bool  # 当前证据是否足以回答；关闭检查或配置失败放行时也可能为 True，需结合 source 和执行状态理解
    reason: str  # 充分、不充分或核验未完成的原因说明
    missing_requirements: list[str] = field(default_factory=list)  # 尚缺的回答依据；核验失败时也可描述尚未完成的核验要求
    supporting_chunk_indices: list[int] = field(default_factory=list)  # 支持回答的片段序号，从 1 开始，对应 evaluated_chunk_ids 的顺序
    source: EvidenceAssessmentSource = "rule"  # 判定来源：model 模型、rule 规则、fallback 失败降级、disabled 检查关闭
    evaluated_chunk_count: int = 0  # 本次报告涉及的片段数量；模型核验时为实际送审数量
    evaluated_chunk_ids: list[str] = field(default_factory=list)  # 本次评估片段的稳定 ID，模型核验时按送审顺序保存
    goal_coverage: list[dict[str, Any]] = field(default_factory=list)  # 逐目标覆盖结果，包含 goal_id、supported/missing/conflict 状态及支持片段 ID
    verification_status: str = "completed"  # 核验执行状态：completed 已完成、unavailable 未能完成；完成不代表证据充分
    diagnostics: dict[str, Any] = field(default_factory=dict)  # 诊断信息，如失败类型、尝试次数、输入预算限制及未送审片段 ID
    query_version: str = ""  # 绑定的查询及目标版本指纹；为空表示尚未绑定，变化后旧报告不能复用
    evidence_version: str = ""  # 绑定的完整证据集合版本指纹；证据内容、顺序或身份冲突变化后需重新核验
    policy_version: str = ""  # 绑定的核验规则与配置版本指纹；策略变化后旧报告不能复用
    require_semantic: bool | None = None  # 本次是否要求语义充分性检查；None 表示报告尚未绑定核验模式
    _report_version: str = field(init=False, repr=False)  # 创建时自动计算的报告内容指纹，用于发现后续修改，防止复用被改动的报告

    def __post_init__(self):
        # 报告保存独立快照；抽取结果、模型响应或调用方列表的后续修改不能污染报告。
        for name in ("missing_requirements", "supporting_chunk_indices", "evaluated_chunk_ids",
                     "goal_coverage", "diagnostics"):
            object.__setattr__(self, name, deepcopy(getattr(self, name)))
        object.__setattr__(self, "_report_version", self._current_report_version())

    def _current_report_version(self) -> str:
        return fingerprint((self.to_dict(), self.diagnostics,
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
        """导出通用核验结果；不再生成财务事实或服务端计算记录。"""
        return deepcopy({
            "sufficient": self.sufficient,
            "reason": self.reason,
            "missing_requirements": self.missing_requirements,
            "supporting_chunk_indices": self.supporting_chunk_indices,
            "source": self.source,
            "evaluated_chunk_count": self.evaluated_chunk_count,
            "evaluated_chunk_ids": self.evaluated_chunk_ids,
            "goal_coverage": self.goal_coverage,
            "verification_status": self.verification_status,
            "outcome": self.diagnostics.get("outcome"),
            "failure_kind": self.diagnostics.get("failure_kind"),
            "failure": self.diagnostics.get("failure"),
            "attempts": self.diagnostics.get("attempts", 0),
            "input_limited": self.diagnostics.get("input_limited", False),
            "omitted_chunk_ids": self.diagnostics.get("omitted_chunk_ids", []),
        })
