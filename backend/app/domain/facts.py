"""财务事实、查询要求和事实覆盖结果，不依赖抽取或核验实现。"""

from dataclasses import asdict, dataclass, field
from typing import Any

VERSION = "financial-facts-v5"


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
    measure_kind: str = "level"  # 原值 / 增长率 / 增量 / 占比；不确定时保留 unknown
    comparison_basis: str = "not_applicable"  # 不适用 / 同比 / 环比 / 未知

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
    measure_kind: str = "level"
    comparison_basis: str = "not_applicable"
    source_spans: tuple[str, ...] = ()
    # 引用是理解阶段声明的未知条件；只有带来源的执行结果才能填充。
    entity_ref: str | None = None
    period_ref: str | None = None


@dataclass(frozen=True)
class CalculationRequirement:
    """操作数引用事实要求编号，顺序由 Agent 明确声明，不再从问句猜测。"""

    id: str
    goal_id: str
    operation: str
    operand_ids: tuple[str, ...]
    source_spans: tuple[str, ...] = ()


@dataclass(frozen=True)
class UnresolvedFinancialRequirement:
    id: str
    goal_id: str
    metric_label: str
    reason: str
    source_spans: tuple[str, ...] = ()


@dataclass(frozen=True)
class FinancialRequirements:
    """固定的财务需求契约；空对象不代表解析失败，失败必须显式记录。"""

    facts: tuple[FactRequirement, ...] = ()
    calculations: tuple[CalculationRequirement, ...] = ()
    unresolved: tuple[UnresolvedFinancialRequirement, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def for_goals(self, goal_ids: set[str]) -> "FinancialRequirements":
        calculations = tuple(item for item in self.calculations if item.goal_id in goal_ids)
        operands = {identifier for item in calculations for identifier in item.operand_ids}
        return FinancialRequirements(
            tuple(item for item in self.facts if item.goal_id in goal_ids or item.requirement_id in operands),
            calculations, tuple(item for item in self.unresolved if item.goal_id in goal_ids),
        )


@dataclass(frozen=True)
class FinancialBinding:
    """本轮核验实际采用的变量值与原文证明，参与报告版本计算。"""

    slot: str
    value: str
    source_chunk_id: str
    document_id: str
    source_quote: str

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
    measure_kind: str = "level"
    comparison_basis: str = "not_applicable"

@dataclass(frozen=True)
class RequirementVerification:
    """业务结论与执行故障分开；Agent 根据具体缺口决定补查或澄清。"""

    requirement_id: str
    goal_id: str
    status: str  # supported / missing / ambiguous / conflict / unsupported
    reason_code: str
    supporting_fact_ids: tuple[str, ...] = ()
    candidate_fact_ids: tuple[str, ...] = ()
    unresolved_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class RelationshipVerification:
    """候选关系的审计记录；模型意见不能直接覆盖来源定位或数值错误。"""

    fact_id: str
    status: str  # supported / unknown / rejected
    reason_code: str
    references: tuple[dict[str, Any], ...] = ()
    method: str = "model_relationship"


#汇总事实、目标匹配结果、缺口、冲突和计算结果
@dataclass
class FinancialFactAssessment:
    """财务事实核验结果，包含事实、目标匹配、缺口、冲突及计算结果。"""

    requirements: list[FactRequirement] = field(default_factory=list)
    facts: list[FinancialFact] = field(default_factory=list)
    matches: list[dict[str, Any]] = field(default_factory=list)
    goal_coverage: list[dict[str, Any]] = field(default_factory=list)
    selected_fact_ids: list[str] = field(default_factory=list)
    missing_requirements: list[str] = field(default_factory=list)
    derived_evidence: list[dict[str, Any]] = field(default_factory=list)
    verification_required: bool = False
    unresolved_targets: list[dict[str, str]] = field(default_factory=list)
    requirement_results: list[RequirementVerification] = field(default_factory=list)
    relationship_results: list[RelationshipVerification] = field(default_factory=list)
    verification_status: str = "completed"
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def applicable(self) -> bool:
        return self.verification_required or bool(self.requirements)

    @property
    def sufficient(self) -> bool:
        return (self.verification_status == "completed" and self.applicable and bool(self.requirements)
                and not self.unresolved_targets and not self.missing_requirements)

    def to_dict(self) -> dict[str, Any]:
        return {"version": VERSION, "applicable": self.applicable,
                "sufficient": self.sufficient, **asdict(self)}

