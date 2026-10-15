"""解码并校验 Agent 提交的财务需求，不搜索、不抽取证据、不另行调用模型。"""

from dataclasses import replace
import re

from domain.facts import (
    CalculationRequirement, FactRequirement, FinancialRequirements,
    UnresolvedFinancialRequirement,
)
from service.core.entities.attribution import get_entity_registry
from service.core.finance.metrics import metric_aliases, parse_metric_mentions
from service.core.finance.rules import PERIOD, company_mentions, parse_currency, parse_scopes, period_order_key, period_value, scope_text
from service.core.query.semantics import canonical_text, domain_policy
from service.core.query.temporal import normalize_relative_years
from service.core.query.validation import QueryValidationError, source_clauses
from agent.financial_coverage import (
    is_qualitative_financial_question, requirement_period_labels, unresolved_financial_targets,
    requirement_metric_mentions, requires_calculation,
)

IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
MAX_REQUIREMENTS = 64
OPERATIONS = {"subtract", "absolute_difference", "change_percent", "decline_percent", "divide_percent"}
ENUMS = {
    "value_kind": {"unknown", "actual", "forecast"},
    "statement_scope": {"unknown", "consolidated", "parent"},
    "currency": {"unknown", "CNY", "USD", "HKD"},
    "measure_kind": {"level", "growth_rate", "absolute_change", "share"},
    "comparison_basis": {"unknown", "not_applicable", "year_over_year", "period_over_period"},
}


def _error(rule, detail):
    raise QueryValidationError(rule, detail, field="understanding.financial")


def _identifier(value):
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        _error("invalid_financial_id", "财务要求或引用编号无效")
    return value


def _text(value, *, empty=False):
    if not isinstance(value, str) or len(value) > 2000 or (not empty and not value.strip()):
        _error("invalid_financial_text", "财务要求文本无效")
    return value.strip()


def _spans(raw):
    if not isinstance(raw, list) or not 1 <= len(raw) <= 8:
        _error("missing_financial_provenance", "每项财务要求必须引用原文")
    return tuple(_text(item) for item in raw)


def _rows(payload, name):
    rows = payload.get(name)
    if not isinstance(rows, list) or len(rows) > MAX_REQUIREMENTS or not all(isinstance(row, dict) for row in rows):
        _error("invalid_financial_list", f"{name} 必须为最多64项的列表")
    return rows


def _fields(row, allowed):
    if set(row) - set(allowed.split()):
        _error("unexpected_financial_field", "模型不能提供实体ID、核验结果或未声明字段")


def decode_financial_requirements(payload) -> FinancialRequirements | None:
    """模型仅提供可追溯的名称和期间表达；实体ID及规范期间由服务端确认。"""
    if payload is None:
        return None
    if not isinstance(payload, dict) or set(payload) != {"facts", "calculations", "unresolved"}:
        _error("invalid_financial_schema", "financial 需要 facts、calculations、unresolved")
    facts = []
    for row in _rows(payload, "facts"):
        _fields(row, "id goal_id entity_name entity_ref period period_ref metric value_kind statement_scope business_scope currency measure_kind comparison_basis source_spans")
        values = {name: row.get(name, "level" if name == "measure_kind" else "not_applicable" if name == "comparison_basis" else "unknown") for name in ENUMS}
        if any(not isinstance(value, str) or value not in ENUMS[name] for name, value in values.items()):
            _error("invalid_financial_enum", "财务口径或度量类型无效")
        entity_ref = row.get("entity_ref")
        period_ref = row.get("period_ref")
        for ref in (entity_ref, period_ref):
            if ref is not None:
                _identifier(ref)
        entity_name = _text(row.get("entity_name", ""), empty=True)
        period = row.get("period")
        if period is not None:
            period = _text(period)
        if (entity_ref and entity_name) or (period_ref and period):
            _error("conflicting_financial_binding", "已知值与待绑定引用不能同时提供")
        # 暂存原始期间表达，validate 阶段统一规范化，避免模型伪造 year/label。
        facts.append(FactRequirement(
            _identifier(row.get("id")), _identifier(row.get("goal_id")), None,
            entity_name, _text(row.get("metric")), {"expression": period} if period else None,
            business_scope=_text(row.get("business_scope", "unknown")),
            source_spans=_spans(row.get("source_spans")), entity_ref=entity_ref, period_ref=period_ref, **values,
        ))
    calculations = []
    for row in _rows(payload, "calculations"):
        _fields(row, "id goal_id operation operand_ids source_spans")
        operands = row.get("operand_ids")
        if row.get("operation") not in OPERATIONS or not isinstance(operands, list) or len(operands) != 2:
            _error("invalid_calculation", "计算只能使用已登记的二元操作")
        calculations.append(CalculationRequirement(
            _identifier(row.get("id")), _identifier(row.get("goal_id")), row["operation"],
            tuple(_identifier(item) for item in operands), _spans(row.get("source_spans")),
        ))
    unresolved = []
    for row in _rows(payload, "unresolved"):
        _fields(row, "id goal_id metric_label reason source_spans")
        if row.get("reason") not in {"unsupported_metric", "ambiguous", "unsupported_calculation"}:
            _error("invalid_unresolved_reason", "未解决需求必须说明不支持或歧义原因")
        unresolved.append(UnresolvedFinancialRequirement(
            _identifier(row.get("id")), _identifier(row.get("goal_id")),
            _text(row.get("metric_label")), row["reason"], _spans(row.get("source_spans")),
        ))
    identifiers = [item.requirement_id for item in facts] + [item.id for item in (*calculations, *unresolved)]
    if len(set(identifiers)) != len(identifiers):
        _error("duplicate_financial_id", "财务要求编号必须唯一")
    if not identifiers:
        _error("empty_financial_contract", "无财务需求时使用 null；识别失败时显式填写 unresolved")
    return FinancialRequirements(tuple(facts), tuple(calculations), tuple(unresolved))


def _normalized_period(expression, source, reference_date):
    if not expression:
        return None
    normalized_source = normalize_relative_years(source, reference_date)
    expression = normalize_relative_years(expression, reference_date)
    period = period_value(expression)
    allowed = {str(period_value(label)) for label in requirement_period_labels(normalized_source)}
    exact = period_value(normalized_source)
    if period is None or (str(period) not in allowed and period != exact):
        _error("financial_period_not_grounded", "期间不在原文或明确参考日期的范围内")
    return period


def _normalize_fact(item, question, reference_date, registry):
    source = "；".join(item.source_spans)
    metric_catalog = domain_policy()["financial_metrics"]
    if item.metric not in metric_catalog:
        _error("unsupported_financial_metric", "不支持的指标必须写入 unresolved，不能伪装成已支持指标")
    payout = bool(re.search(r"派息率|股利支付率|分红支付率", source))
    if not any(canonical_text(alias) in canonical_text(source) for alias in metric_aliases(item.metric)) and not (
        payout and item.metric in {"每股红利", "每股收益"}
    ):
        _error("financial_metric_not_grounded", "指标没有原文依据")
    entity_id = None
    if item.entity_name:
        matches = company_mentions(item.entity_name, registry)
        ids = registry.resolved_ids(matches, "company")
        source_ids = registry.resolved_ids(company_mentions(source, registry), "company")
        if len(ids) == 1:
            entity_id = ids[0]
            if entity_id not in source_ids:
                _error("financial_entity_not_grounded", "公司未出现在要求引用的原文中")
        elif item.entity_name not in source:
            _error("financial_entity_not_grounded", "未知公司的名称必须引用原文")
    elif not item.entity_ref:
        _error("missing_financial_subject", "主体必须明确、声明依赖引用，或列为歧义")
    if item.entity_ref == "document_subject" and not re.search(r"报告|研报|文档|材料|附件", source):
        _error("undeclared_document_subject", "只有用户明确指向文档时才可从文档主体绑定公司")
    raw_period = (item.period or {}).get("expression")
    if item.period_ref == "document_subject":
        _error("invalid_period_reference", "文档主体引用不能用于期间")
    period = _normalized_period(raw_period, source, reference_date)
    if not raw_period and not item.period_ref and requirement_period_labels(source):
        _error("missing_financial_period", "原文已经指定期间，不能省略")
    if not raw_period and not item.period_ref and re.search(r"今年|去年|明年|未来|最近|近[一二三\d]年|本季度|上季度", source):
        _error("ambiguous_financial_period", "相对期间必须明确规范化或保留为待澄清要求")
    kind, scope, business = parse_scopes(scope_text(source, registry))
    period_kinds = {parse_scopes(match.group())[0] for match in PERIOD.finditer(source)
                    if period and period_value(match.group()) == period} - {"unknown"}
    if len(period_kinds) == 1:
        kind = next(iter(period_kinds))  # 混合 A/E 年份按当前要求对应的期间标记绑定。
    expected = {"value_kind": kind, "statement_scope": scope, "business_scope": business,
                "currency": parse_currency(source)}
    for name, actual in expected.items():
        value = getattr(item, name)
        if value != "unknown" and value != actual:
            _error("financial_scope_not_grounded", f"{name} 没有对应原文依据，请引用更准确的局部分句")
        if actual != "unknown" and value != actual:
            _error("financial_scope_dropped", f"遗漏明确的 {name}")
    if item.business_scope not in {"unknown", *domain_policy().get("business_scopes", [])}:
        _error("unsupported_business_scope", "业务范围不受支持")
    # 保留原文的主体—期间关系，不允许把另一家公司的年份借给当前公司。
    if item.entity_name and period:
        _validate_relation(normalize_relative_years(question, reference_date), entity_id, item.metric, period, registry)
    binding = "deferred" if item.entity_ref or item.period_ref else "resolved" if entity_id else "unknown"
    return replace(item, entity_id=entity_id, period=period, binding_status=binding)


def _validate_calculations(financial):
    facts = {item.requirement_id: item for item in financial.facts}
    for calculation in financial.calculations:
        if len(set(calculation.operand_ids)) != 2 or any(key not in facts for key in calculation.operand_ids):
            _error("invalid_calculation_operands", "计算必须引用两个不同且已声明的事实要求")
        left, right = (facts[key] for key in calculation.operand_ids)
        if calculation.operation == "divide_percent":
            if (left.metric, right.metric) != ("每股红利", "每股收益") or left.period != right.period:
                _error("invalid_payout_inputs", "派息率必须用同期间的每股红利除以每股收益")
            if (left.entity_id, left.entity_ref, left.entity_name) != (right.entity_id, right.entity_ref, right.entity_name):
                _error("invalid_payout_subject", "派息率操作数必须来自同一主体")
        elif left.metric != right.metric or left.measure_kind != right.measure_kind:
            _error("incompatible_calculation_inputs", "差值和变化率必须使用相同指标与度量")
        if calculation.operation in {"change_percent", "decline_percent", "divide_percent"} and (
            left.measure_kind != "level" or right.measure_kind != "level"
        ):
            _error("invalid_calculation_measure", "比例计算需要原值，不能重复计算已披露的增长率")
        source = "；".join(calculation.source_spans)
        patterns = {"subtract": r"差|增加|减少|下降|高多少|低多少", "absolute_difference": r"相差|差额|差值",
                    "change_percent": r"增长率|增幅", "decline_percent": r"降幅|下降率", "divide_percent": r"派息率|股利支付率|分红支付率"}
        if not re.search(patterns[calculation.operation], source):
            _error("calculation_not_grounded", "计算类型没有原文依据")
        # 年度变化率有明确的基期方向；跨公司差值则由原文和操作数显式顺序决定。
        if calculation.operation in {"change_percent", "decline_percent"}:
            if (left.entity_id, left.entity_ref, left.entity_name) != (right.entity_id, right.entity_ref, right.entity_name):
                _error("invalid_change_subject", "跨期变化率必须使用同一主体")
            current, base = (left, right) if calculation.operation == "change_percent" else (right, left)
            if current.period and base.period:
                if current.period["granularity"] != base.period["granularity"]:
                    _error("incompatible_calculation_periods", "跨期变化率必须使用相同粒度的期间")
                if period_order_key(current.period) <= period_order_key(base.period):
                    _error("calculation_direction_reversed", "变化率的当前期间必须晚于基期")


def validate_financial_requirements(financial, goals, question, *, reference_date=""):
    """返回规范化契约；发现遗漏只报错，交由同一次理解流程修正。"""
    registry = get_entity_registry()
    goal_ids = {goal.id for goal in goals}
    if financial:
        for item in (*financial.facts, *financial.calculations, *financial.unresolved):
            if item.goal_id not in goal_ids or any(span not in question for span in item.source_spans):
                _error("financial_provenance_invalid", "财务要求必须关联已知目标并逐字引用用户问题")
        for issue in financial.unresolved:
            if canonical_text(issue.metric_label) not in canonical_text("；".join(issue.source_spans)):
                _error("unresolved_label_not_grounded", "未解决指标必须保留原文名称")
        financial = replace(financial, facts=tuple(
            _normalize_fact(item, question, reference_date, registry) for item in financial.facts
        ))
        _validate_calculations(financial)
    _validate_coverage(financial, goals, question, registry, reference_date)
    return financial


def _validate_coverage(financial, goals, question, registry, reference_date):
    facts = financial.facts if financial else ()
    unresolved = financial.unresolved if financial else ()
    calculations = financial.calculations if financial else ()
    # 同时检查原文和逐目标表达；此处只识别明显遗漏，不制造新的事实要求。
    parts = [(goal.id, "；".join(goal.source_spans) or goal.text) for goal in goals]
    parts += [(None, part) for part in source_clauses(question)]
    for goal_id, source in parts:
        source = normalize_relative_years(source, reference_date)
        if is_qualitative_financial_question(source):
            continue
        relevant = financial.for_goals({goal_id}).facts if financial and goal_id else facts
        issues = [item for item in unresolved if goal_id is None or item.goal_id == goal_id]
        for mention in requirement_metric_mentions(source):
            metric_issues = [issue for issue in issues if mention.metric in issue.metric_label or mention.raw_label in issue.metric_label]
            matching = [item for item in relevant if item.metric == mention.metric and item.measure_kind == mention.measure_kind
                        and (mention.comparison_basis == "unknown" or item.comparison_basis == mention.comparison_basis)]
            if not matching and not metric_issues:
                _error("financial_requirement_missing", f"遗漏指标或度量：{mention.raw_label}")
            if matching and not metric_issues:
                _validate_scope_coverage(source, matching, registry)
        if requires_calculation(source) and not any(goal_id is None or item.goal_id == goal_id for item in calculations) and not issues:
            _error("calculation_requirement_missing", "明确计算要求必须声明操作数或未解决原因")
    for target in unresolved_financial_targets(question, goals, registry):
        if not any(target["metric_label"] in item.metric_label for item in unresolved):
            _error("unsupported_requirement_missing", f"未支持的指标必须显式保留：{target['metric_label']}")


def _validate_scope_coverage(source, matching, registry):
    """检查明确公司和期间是否被遗漏，不生成默认公司或默认年份。"""
    # 每个局部分句单独检查，避免把 A 的年份强加给 B。
    mentions = company_mentions(source, registry)
    company_list = len(mentions) > 1 and all(re.fullmatch(r"[、和与及\s]+", source[left["end"]:right["start"]])
                                           for left, right in zip(mentions, mentions[1:]))
    for clause in ([source] if company_list else source_clauses(source)):
        if not any(item.metric in {mention.metric for mention in parse_metric_mentions(clause)} for item in matching):
            continue
        entities = registry.resolved_ids(company_mentions(clause, registry), "company")
        periods = [period_value(label) for label in requirement_period_labels(clause)]
        for entity in entities or [None]:
            scoped = [item for item in matching if entity is None or item.entity_id == entity]
            if not scoped:
                _error("financial_subject_omitted", "财务要求遗漏明确的查询主体")
            for period in periods:
                if period and not any(item.period == period for item in scoped):
                    _error("financial_period_omitted", "财务要求遗漏明确的查询期间")


def _validate_relation(question, entity_id, metric, period, registry):
    """按原文局部分句校验主体—指标—期间关系，允许一次要求只覆盖比较的一侧。"""
    global_ids = registry.resolved_ids(company_mentions(question, registry), "company")
    allowed = []
    for clause in source_clauses(question):
        ids = registry.resolved_ids(company_mentions(clause, registry), "company")
        ids = ids or (global_ids if len(global_ids) == 1 else [])
        if entity_id in ids and metric in {mention.metric for mention in parse_metric_mentions(clause)}:
            allowed.extend(period_value(label) for label in requirement_period_labels(clause))
    if allowed and period not in allowed:
        _error("financial_relation_changed", "主体、指标与期间的原文关系被改变")
