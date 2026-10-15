"""固定要求与候选事实的匹配、可比性和冲突判定，不调用模型。"""

import json
from decimal import Decimal

from domain.facts import FactRequirement, FinancialFact, RequirementVerification
from service.core.finance.metrics import metric_unit_compatible
from service.core.finance.rules import normalized_value


def match_fact(requirement: FactRequirement, fact: FinancialFact) -> str:
    if not requirement.entity_id or requirement.binding_status != "resolved":
        return "unknown"
    if fact.entity_id and fact.entity_id != requirement.entity_id:
        return "not_applicable"
    if fact.verification_status in {"rejected", "superseded"}:
        return "not_applicable"
    if fact.metric != requirement.metric or (requirement.period and fact.period and fact.period != requirement.period):
        return "not_applicable"
    if requirement.period and not fact.period:
        return "unknown"
    if fact.attribution_status != "verified" or fact.verification_status != "verified":
        return "unknown"
    if fact.measure_kind != requirement.measure_kind:
        return "not_applicable"
    if requirement.comparison_basis not in {"unknown", "not_applicable"}:
        if fact.comparison_basis == "unknown":
            return "unknown"
        if fact.comparison_basis != requirement.comparison_basis:
            return "not_applicable"
    if not metric_unit_compatible(fact.metric, fact.measure_kind, fact.unit):
        return "unknown" if fact.unit is None else "not_applicable"
    for key in ("value_kind", "statement_scope", "business_scope", "currency"):
        expected, actual = getattr(requirement, key), getattr(fact, key)
        if expected != "unknown" and actual == "unknown":
            return "unknown"
        if expected != "unknown" and expected != actual:
            return "not_applicable"
    if requirement.business_scope == "unknown" and fact.business_scope != "unknown":
        return "not_applicable"
    return "usable"


def candidate_scopes_are_ambiguous(
    candidates: list[FinancialFact], values: set[tuple[Decimal, str, str]],
) -> bool:
    """判断候选是否缺少足以比较数值的共同期间和口径依据。"""
    scopes = {
        (json.dumps(fact.period, sort_keys=True), fact.value_kind, fact.statement_scope,
         fact.business_scope, fact.currency, fact.measure_kind, fact.comparison_basis) for fact in candidates
    }
    if len(scopes) > 1:
        return True
    if len(values) <= 1:
        return False

    # 近似值与精确值不同，不能直接宣布事实冲突。
    if any(fact.value_qualifier == "approximate" for fact in candidates):
        return True

    origins = {(fact.source.get("document_id"), fact.source.get("table_id"), fact.source.get("row")) for fact in candidates}
    shared_row = len(origins) == 1 and all(fact.source.get("table_id") for fact in candidates)
    # 独立来源都写 unknown，不代表口径相同；同一原表行则保留已有结构依据。
    return not shared_row and any(
        fact.statement_scope == "unknown" or fact.value_kind == "unknown" for fact in candidates
    )


def candidate_support_status(candidates: list[FinancialFact]) -> str:
    """将可用候选归为支持、缺失或冲突；先确认可比性，再比较数值。"""
    if not candidates:
        return "missing"
    # 使用精确十进制和统一单位比较，80 亿元与 8,000 百万元是相同数值。
    values = {normalized_value(fact) for fact in candidates}
    if candidate_scopes_are_ambiguous(candidates, values):
        return "missing"
    if len(values) > 1:
        return "conflict"
    return "supported"


def verify_requirement(requirement, facts):
    """保留相关但未确认的候选，避免把关系歧义误写成没有资料。"""
    matches = [{"requirement_id": requirement.requirement_id, "goal_id": requirement.goal_id,
                "fact_id": fact.fact_id, "status": match_fact(requirement, fact)} for fact in facts]
    usable = [fact for fact, match in zip(facts, matches) if match["status"] == "usable"]
    related = [fact for fact, match in zip(facts, matches) if match["status"] == "unknown"
               and fact.metric == requirement.metric
               and (not fact.entity_id or not requirement.entity_id or fact.entity_id == requirement.entity_id)
               and (not fact.period or not requirement.period or fact.period == requirement.period)
               and fact.verification_status != "rejected"]
    status = candidate_support_status(usable)
    fields = set()
    reason = "verified_facts" if status == "supported" else "comparable_values_conflict" if status == "conflict" else "no_matching_fact"
    if requirement.binding_status != "resolved":
        status, reason = "ambiguous", "unresolved_requirement_binding"
        fields.update(key for key, ref in (("entity", requirement.entity_ref), ("period", requirement.period_ref)) if ref)
        if not requirement.entity_id:
            fields.add("entity")
    elif status == "missing" and (usable or related):
        status, reason = "ambiguous", "candidate_scope_ambiguous" if usable else "relationship_unconfirmed"
        candidates = usable or related
        for name in ("period", "value_kind", "statement_scope", "business_scope", "currency"):
            values = {str(getattr(fact, name)) for fact in candidates}
            if len(values) > 1 or (getattr(requirement, name) not in (None, "unknown") and values <= {"None", "unknown"}):
                fields.add(name)
        if related:
            fields.add("relationship")
    selected = usable if status == "supported" else []
    result = RequirementVerification(requirement.requirement_id, requirement.goal_id, status, reason,
        tuple(fact.fact_id for fact in selected), tuple(fact.fact_id for fact in usable + related), tuple(sorted(fields)))
    if status != "supported":
        for match in matches:
            if match["status"] == "usable":
                match["status"] = "conflict" if status == "conflict" else "unknown"
    return result, matches, selected
