"""财务事实级核验与确定性的回答上下文构建。

入库标注和模型 JSON 都只是候选，不代表核验结果。
确定性来源检查与语义关系核验分工；未确认关系保留为歧义，
模型执行失败不能被解释为资料缺失。
"""

from __future__ import annotations

import re
from dataclasses import asdict, replace
from typing import Any

from domain.evidence import EvidenceCollection, AnswerBlock
from domain.assessment import EvidenceAssessment
from domain.facts import FinancialFact, FinancialFactAssessment, FinancialRequirements, RequirementVerification

from service.core.entities.attribution import EntityRegistry, get_entity_registry
from service.core.finance.matching import verify_requirement
from service.core.finance.model_policy import verification_mode
from service.core.finance.rules import CLAUSE, UNIT, financial_metrics, extract_table_facts, extract_text_facts


def extract_financial_facts(chunks: EvidenceCollection, *, registry: EntityRegistry | None = None) -> list[FinancialFact]:
    registry = registry or get_entity_registry()
    facts = []
    for chunk in chunks:
        if chunk.structure.kind == "figure" and chunk.structure.figure.status != "extracted":
            continue
        is_table = chunk.structure.kind == "table" or bool(chunk.structure.table.rows and chunk.structure.table.rows[0].cells)
        facts.extend(extract_table_facts(chunk, registry) if is_table else extract_text_facts(chunk, registry))
        from service.core.finance.model_extraction import cached_facts
        facts.extend(cached_facts(chunk, registry))
    return list({fact.fact_id: fact for fact in facts}.values())


def _review_uncertain_facts(bundle, chunks, *, registry, client):
    """无需关键词触发：缺口、歧义和启发式归属都可进入关系核验。"""
    if verification_mode() == "rules":
        return
    # 依靠报告主体与章节推断的表格归属需要语义复核；清晰原表映射仍走快速路径。
    bundle.facts = [replace(fact, attribution_status="unknown", verification_status="unknown")
                    if fact.verification_method == "report_subject_and_section" else fact for fact in bundle.facts]
    pending = [item for item in bundle.requirements if item.binding_status == "resolved"
               and verify_requirement(item, bundle.facts)[0].status != "supported"]
    if not pending or not chunks:
        return
    from service.core.finance.model_extraction import review_candidates
    try:
        result = review_candidates(chunks, pending, client, registry=registry)
        # 同一数值位置的语义补全替代原先未确认的候选；保留旧记录供审计，避免把补全误判为口径冲突。
        bundle.facts = _supersede_rule_candidates(bundle.facts, result.facts)
        bundle.facts = list({fact.fact_id: fact for fact in [*bundle.facts, *result.facts]}.values())
        bundle.relationship_results.extend(result.relationships)
        bundle.verification_status = result.verification_status
        bundle.diagnostics.update(result.diagnostics)
    except Exception as error:
        bundle.verification_status = "unavailable"
        bundle.diagnostics["failure_kind"] = type(error).__name__


def _supersede_rule_candidates(facts, reviewed):
    def position(fact):
        source = fact.source
        if "table_id" in source:
            return source["chunk_id"], source["table_id"], source["row"], source["column"]
        return source["chunk_id"], source.get("value_reference", {}).get("start", source.get("value_start"))

    confirmed = [fact for fact in reviewed if fact.verification_status == "verified"]
    output = []
    for fact in facts:
        replaced = fact.verification_method != "model_relationship" and any(
            position(fact) == position(new) and fact.metric == new.metric and fact.value == new.value
            and (not fact.entity_id or fact.entity_id == new.entity_id)
            and (not fact.period or fact.period == new.period) for new in confirmed)
        output.append(replace(fact, verification_status="superseded") if replaced else fact)
    return output


def _record_requirement_coverage(bundle, requirement, result, matches):
    bundle.requirement_results.append(result)
    bundle.matches.extend(matches)
    bundle.goal_coverage.append({"goal_id": result.goal_id, "requirement_id": result.requirement_id,
                                "status": result.status, "supporting_fact_ids": list(result.supporting_fact_ids),
                                "reason_code": result.reason_code, "unresolved_fields": list(result.unresolved_fields)})
    bundle.selected_fact_ids.extend(result.supporting_fact_ids)
    if result.status != "supported":
        labels = {"missing": "没有找到匹配事实", "ambiguous": "关系或口径尚未确认", "conflict": "可比事实数值冲突"}
        period = (requirement.period or {}).get("label", "期间未明确")
        bundle.missing_requirements.append(f"{requirement.entity_name} / {requirement.metric} / {period}：{labels[result.status]}")


def assess_financial_facts(
    requirements: FinancialRequirements | None, chunks: EvidenceCollection, *,
    registry: EntityRegistry | None = None, allow_model: bool = False, client: Any | None = None,
) -> FinancialFactAssessment:
    """只消费已准备的财务需求；不从问题或证据重新推断用户想查什么。"""
    if requirements is not None and not isinstance(requirements, FinancialRequirements):
        raise TypeError("财务核验必须接收结构化 FinancialRequirements")
    if requirements is None:
        return FinancialFactAssessment()
    registry = registry or get_entity_registry()
    unresolved = [asdict(item) for item in requirements.unresolved]
    bundle = FinancialFactAssessment(requirements=list(requirements.facts),
                                    verification_required=bool(unresolved or requirements.calculations),
                                    unresolved_targets=unresolved)
    for item in requirements.unresolved:
        status = "ambiguous" if item.reason == "ambiguous" else "unsupported"
        bundle.requirement_results.append(RequirementVerification(item.id, item.goal_id, status, item.reason))
        bundle.missing_requirements.append(f"{item.metric_label}：{item.reason}")
        bundle.goal_coverage.append({"goal_id": item.goal_id, "requirement_id": item.id,
                                     "status": status, "supporting_fact_ids": []})
    if not bundle.applicable:
        return bundle
    bundle.facts = extract_financial_facts(chunks, registry=registry)
    if allow_model:
        _review_uncertain_facts(bundle, chunks, registry=registry, client=client)
    supporting_facts = {}
    for requirement in bundle.requirements:
        result, matches, candidates = verify_requirement(requirement, bundle.facts)
        if result.status == "missing" and bundle.diagnostics.get("unavailable_tables"):
            result = replace(result, status="ambiguous", reason_code="original_table_required", unresolved_fields=("source_structure",))
        if result.status in {"missing", "ambiguous"} and bundle.diagnostics.get("source_limit_reached"):
            result = replace(result, status="ambiguous", reason_code="source_context_limit", unresolved_fields=("source_context",))
        if result.status == "missing" and bundle.verification_status != "completed":
            # 抽取还未成功执行，不能据此宣布原文没有事实。
            result = replace(result, status="ambiguous", reason_code="verification_unavailable",
                             unresolved_fields=("relationship",))
        _record_requirement_coverage(bundle, requirement, result, matches)
        if result.status == "supported":
            supporting_facts[requirement.requirement_id] = candidates
    bundle.selected_fact_ids = list(dict.fromkeys(bundle.selected_fact_ids))
    from service.core.finance.calculations import build_derived_evidence, calculation_status
    bundle.derived_evidence = build_derived_evidence(requirements, chunks, supporting_facts=supporting_facts)
    statuses = {item.requirement_id: item.status for item in bundle.requirement_results}
    inputs = {item.id: item.operand_ids for item in requirements.calculations}
    for calculation in bundle.derived_evidence:
        supported = calculation["status"] == "supported"
        status = calculation_status(calculation, [statuses.get(key, "missing") for key in inputs[calculation["id"]]])
        bundle.requirement_results.append(RequirementVerification(
            calculation["id"], calculation["goal_id"], status,
            calculation.get("reason", "verified_calculation"), tuple(calculation.get("input_fact_ids", []))))
        if supported:
            # 计算目标可能不同于输入目标，回答引用仍需能追溯到每个已核验操作数。
            for index, fact_id in enumerate(calculation["input_fact_ids"], 1):
                identifier = f"{calculation['id']}_input{index}"
                bundle.matches.append({"goal_id": calculation["goal_id"], "requirement_id": identifier,
                                       "fact_id": fact_id, "status": "usable"})
                bundle.goal_coverage.append({"goal_id": calculation["goal_id"], "requirement_id": identifier,
                                             "status": "supported", "supporting_fact_ids": [fact_id]})
        bundle.goal_coverage.append({
            "goal_id": calculation["goal_id"], "requirement_id": calculation["id"],
            "status": status,
            "supporting_fact_ids": calculation.get("input_fact_ids", []),
        })
        if not supported:
            bundle.missing_requirements.append(f"计算 {calculation['id']}：{calculation.get('reason', '输入未确认')}")
    return bundle


def financial_rejection(evidence: EvidenceCollection, bundle: FinancialFactAssessment, goals=()) -> EvidenceAssessment:
    """保留业务缺口与执行故障，供 Agent 决定补查、澄清或停止。"""
    identifiers = [item.id for item in evidence]
    coverage = []
    for goal in goals:
        local = [entry for entry in bundle.goal_coverage if entry["goal_id"] == goal.id]
        status = next((status for status in ("conflict", "ambiguous", "unsupported", "missing")
                       if any(entry["status"] == status for entry in local)), "supported" if local else "missing")
        fact_ids = {identifier for entry in local for identifier in entry["supporting_fact_ids"]}
        selected = [fact for fact in bundle.facts if fact.fact_id in fact_ids]
        coverage.append({"goal_id": goal.id, "status": status,
                         "supporting_chunk_ids": list(dict.fromkeys(fact.source["chunk_id"] for fact in selected)),
                         "supporting_fact_ids": [fact.fact_id for fact in selected]})
    supported = {fact.source["chunk_id"] for fact in bundle.facts if fact.fact_id in bundle.selected_fact_ids}
    return EvidenceAssessment(
        sufficient=False, reason="财务关系核验未能完成" if bundle.verification_status != "completed" else "财务事实存在未解决要求",
        source="fallback" if bundle.verification_status != "completed" else "rule",
        verification_status=bundle.verification_status, diagnostics=bundle.diagnostics,
        missing_requirements=bundle.missing_requirements,
        supporting_chunk_indices=[index for index, identifier in enumerate(identifiers, 1) if identifier in supported],
        evaluated_chunk_count=len(evidence), evaluated_chunk_ids=identifiers, goal_coverage=coverage, facts=bundle,
    )

def render_fact(fact: FinancialFact, goal_ids: list[str]) -> str:
    period = (fact.period or {}).get("label", "未知")
    lines = [f"目标：{', '.join(goal_ids)}；事实ID：{fact.fact_id}", f"公司：{fact.entity_name}",
             f"指标：{fact.metric}；度量：{fact.measure_kind}；比较基准：{fact.comparison_basis}；期间：{period}", f"数值：{'约' if fact.value_qualifier == 'approximate' else ''}{fact.value}{fact.unit or '（单位未说明）'}",
             f"实际/预测：{fact.value_kind}；报表口径：{fact.statement_scope}；币种：{fact.currency}"]
    source = fact.source
    if "table_id" in source:
        lines.append(f"原表：{source['table_id']}；原始行：{source['row']}；原始列：{source['column']}")
        lines.append(f"表头：{source['header']}；单元格原文：{source['cell']}")
    elif fact.verification_method == "direct_statement":
        lines.append("原文依据：" + source["quote"])
    # 有序列表的完整引文可能包含其他公司的数值，
    # 因此完整关系只保留在审计记录中，回答上下文仅接收当前事实。
    return "\n".join(lines)


def project_fact_context(evidence: EvidenceCollection, bundle: FinancialFactAssessment) -> tuple[AnswerBlock, ...]:
    """生成筛选正文，原文和查询判定分别保存在证据集合与事实报告中。"""
    if not bundle.applicable:
        return tuple(AnswerBlock(item.id, item.text) for item in evidence)
    selected = {fact.fact_id: fact for fact in bundle.facts if fact.fact_id in bundle.selected_fact_ids}
    output = []
    for item in evidence:
        local = [fact for fact in selected.values() if fact.source["chunk_id"] == item.id]
        blocks = []
        for fact in local:
            goal_ids = list(dict.fromkeys(entry["goal_id"] for entry in bundle.matches
                                         if entry["fact_id"] == fact.fact_id and entry["status"] == "usable"))
            blocks.append(render_fact(fact, goal_ids))
        local_ids = {fact.fact_id for fact in local}
        for derived in bundle.derived_evidence:
            if derived["status"] == "supported" and local_ids.intersection(derived.get("input_fact_ids") or []):
                blocks.append(f"服务端计算：{derived['requirement']}；操作：{derived['operation']}；"
                              f"结果：{derived['result']}{derived['unit']}；输入事实："
                              + ", ".join(derived["input_fact_ids"]))
        if item.structure.kind != "table":
            qualitative = [match.group().strip() for match in CLAUSE.finditer(item.text)
                           if not re.search(r"\d", match.group()) and not re.search(UNIT, match.group())
                           and not (financial_metrics(match.group()) and re.search(r"[零〇一二三四五六七八九十百千万亿两]", match.group()))]
            blocks.extend(qualitative)
        output.append(AnswerBlock(item.id, "\n\n".join(blocks), tuple(fact.fact_id for fact in local)))
    return tuple(output)
