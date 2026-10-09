"""财务事实、查询要求和事实覆盖结果，不依赖抽取或核验实现。"""

from dataclasses import asdict, dataclass, field
from typing import Any

VERSION = "financial-facts-v2"


@dataclass(frozen=True)
class CalculationInput:
    """由可靠原表或已核验事实提供的计算操作数。"""

    metric: str
    period: str
    value: str
    unit: str | None
    scope: str
    chunk_id: str
    chunk_index: int
    table_id: str | None = None
    row: int | None = None
    column: int | None = None
    source_label: str = ""
    source_header: str = ""
    source_value: str = ""
    document_name: str = ""
    document_id: str = ""
    fact_id: str = ""
    value_kind: str = "unknown"
    statement_scope: str = "unknown"
    business_scope: str = "unknown"
    currency: str = "unknown"
    value_qualifier: str = "reported"

# 表示一个查询目标需要哪个公司的什么指标、哪个期间及口径
@dataclass(frozen=True)
class FactRequirement:
    requirement_id: str
    goal_id: str
    entity_id: str | None
    entity_name: str
    metric: str
    period: dict[str, Any] | None
    value_kind: str = "unknown"
    statement_scope: str = "unknown"
    business_scope: str = "unknown"
    binding_status: str = "resolved"
    currency: str = "unknown"

# 保存一条独立事实，包括数值、归属、口径和原文出处
@dataclass
class FinancialFact:
    """一条带出处的财务事实；公司归属与完整事实核验分别记录。

    attribution_status 只说明数值属于谁；verification_status 还要求
    数值可解析且期间有依据。口径缺失保留 unknown，不因归属确认而补齐。
    """
    fact_id: str
    entity_id: str | None
    entity_name: str
    metric: str
    period: dict[str, Any] | None
    raw_value: str
    value: str | None
    unit: str | None
    currency: str
    value_kind: str
    statement_scope: str
    business_scope: str
    attribution_status: str
    verification_status: str
    verification_method: str
    source: dict[str, Any]
    value_qualifier: str = "reported"
    decimal_places: int = 0

#汇总事实、目标匹配结果、缺口、冲突和计算结果
@dataclass
class FactEvidenceBundle:
    requirements: list[FactRequirement] = field(default_factory=list)
    facts: list[FinancialFact] = field(default_factory=list)
    matches: list[dict[str, Any]] = field(default_factory=list)
    goal_coverage: list[dict[str, Any]] = field(default_factory=list)
    selected_fact_ids: list[str] = field(default_factory=list)
    missing_requirements: list[str] = field(default_factory=list)
    derived_evidence: list[dict[str, Any]] = field(default_factory=list)

    @property
    def applicable(self) -> bool:
        return bool(self.requirements)

    @property
    def sufficient(self) -> bool:
        return self.applicable and not self.missing_requirements

    def to_dict(self) -> dict[str, Any]:
        return {"version": VERSION, "applicable": self.applicable,
                "sufficient": self.sufficient, **asdict(self)}

