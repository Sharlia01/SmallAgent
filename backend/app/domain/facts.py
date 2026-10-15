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
    requirement_id: str  # 这条财务要求的唯一编号
    goal_id: str         # 这条要求服务于哪个用户目标
    entity_id: str | None  # 公司在实体目录中的标准ID,由服务端识别生成
    entity_name: str          #公司名称
    metric: str               # 要查询的标准财务指标
    period: dict[str, Any] | None   #规范化后的期间信息，例如“2024全年”
    value_kind: str = "unknown"     # 要求实际值还是预测值
    statement_scope: str = "unknown"  #报表口径，例如“母公司”，“合并”
    business_scope: str = "unknown"   #指标属于哪个业务范围，“光伏、火电”
    binding_status: str = "resolved"  #公司、期间等条件的绑定状态
    currency: str = "unknown"         #币种要求，例如“USD，CNY”
    measure_kind: str = "level"       #要查指标的原值、增长率、增量还是占比
    comparison_basis: str = "not_applicable"  #比较基准，同比/环比
    source_spans: tuple[str, ...] = ()  #这条要求依据用户问题中的哪些原文片段
    # 引用是理解阶段声明的未知条件；只有带来源的执行结果才能填充。
    entity_ref: str | None = None   #公司尚未知时，需要绑定的变量名称，slot
    period_ref: str | None = None   #期间尚未知时，需要绑定的变量名称

@dataclass(frozen=True)
class CalculationRequirement:
    """操作数引用事实要求编号，顺序由 Agent 明确声明，不再从问句猜测。"""

    id: str
    goal_id: str
    operation: str  #执行什么运算，只能使用已支持的操作
    operand_ids: tuple[str, ...]  # FactRequirement.requirement_id使用的事实id列表
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

    facts: tuple[FactRequirement, ...] = () #需要查什么
    calculations: tuple[CalculationRequirement, ...] = () #需要算什么
    unresolved: tuple[UnresolvedFinancialRequirement, ...] = () #还有什么需求尚未明确

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
    """一条要求综合所有事实后的结论"""

    requirement_id: str
    goal_id: str
    status: str  # supported / missing / ambiguous / conflict / unsupported
    reason_code: str
    supporting_fact_ids: tuple[str, ...] = () #这条要求支持的事实列表
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

    requirements: list[FactRequirement] = field(default_factory=list) #本次要核验的需求列表
    facts: list[FinancialFact] = field(default_factory=list) #从资料中提取出的 FinancialFact 列表，包含已确认、未确认、被拒绝或已被替代的事实记录
    matches: list[dict[str, Any]] = field(default_factory=list)  #每条要求与某条事实之间的匹配记录
    goal_coverage: list[dict[str, Any]] = field(default_factory=list) #对用户目标的覆盖记录，同一个目标可以有多个记录
    selected_fact_ids: list[str] = field(default_factory=list)   #已选中用于支持事实要求的事实ID
    derived_evidence: list[dict[str, Any]] = field(default_factory=list) #财务计算记录，包含操作、输入、结果
    unresolved_targets: list[dict[str, str]] = field(default_factory=list) #FinancialRequirements.unresolved
    missing_requirements: list[str] = field(default_factory=list) #尚未解决的问题的文字说明，可能是没搜到的
    requirement_results: list[RequirementVerification] = field(default_factory=list)  #一条要求综合所有事实后的结论
    relationship_results: list[RelationshipVerification] = field(default_factory=list) #模型辅助流程对候选事实关系的核验记录
    verification_status: str = "completed"  #核验过程是否正常完成
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def applicable(self) -> bool:
        # 是否适用由需求决定；合法计算必有操作数要求，未解决需求也不能被跳过。
        return bool(self.requirements or self.unresolved_targets)

    @property
    def sufficient(self) -> bool:
        return (self.verification_status == "completed" and self.applicable and bool(self.requirements)
                and not self.unresolved_targets and not self.missing_requirements)

    def to_dict(self) -> dict[str, Any]:
        return {"version": VERSION, "applicable": self.applicable,
                "sufficient": self.sufficient, **asdict(self)}

