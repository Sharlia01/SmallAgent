"""检索、Agent 和评测共用的完整证据核验入口。"""

from typing import Any
from dataclasses import replace

from domain.assessment import EvidenceAssessment
from domain.evidence import EvidenceCollection
from service.core.evidence.assessment import finalize_assessment, verification_policy_version
from service.core.evidence.sufficiency import assess_evidence_sufficiency
from service.core.query.context import QueryContext

def _incremental_sources(query, evidence, previous, require_semantic):
    """只裁掉已审且未引用的旧证据；即已经核验过且不支持目标的证据"""
    if (not require_semantic or previous is None or previous.source != "model"
            or previous.verification_status != "completed"
            or any(item.get("status") == "conflict" for item in previous.goal_coverage)):
        return evidence

    scope_ids = previous.diagnostics.get("verification_scope_ids") # 取出上一轮核验对应的完整证据范围
    current_ids = {item.id for item in evidence}
    if not scope_ids or not set(scope_ids) <= current_ids:
        return evidence
    # 用当前原文重建旧集合，检查删除、修改、报告篡改、目标和策略变化。
    old_scope = evidence.select(scope_ids)
    if not previous.matches(query.version, old_scope.version, verification_policy_version(),
                            require_semantic=require_semantic):
        return evidence
    evaluated = set(previous.evaluated_chunk_ids) # 取出上一轮实际送审核验的证据ID
    # 顶层支持索引已汇总全部目标引用，包括缺失目标的部分证据。
    supported = {
        previous.evaluated_chunk_ids[index - 1]
        for index in previous.supporting_chunk_indices
        if 1 <= index <= len(previous.evaluated_chunk_ids)
    }
    # 未送审过的旧材料也保留，不能把预算遗漏误认为已审无用。
    return evidence.select(item.id for item in evidence
                           if item.id not in evaluated or item.id in supported)

def _gap_context(query, evidence, previous, require_semantic):
    """仅对范围有效、逐目标引用完整的旧模型报告复用已支持目标。"""
    if (not require_semantic or not query.goals or previous is None
            or previous.source != "model" or previous.verification_status != "completed"):
        return None
    scope = previous.diagnostics.get("verification_scope_ids")
    current_ids = {item.id for item in evidence}
    if not scope or not set(scope) <= current_ids:
        return None
    if not previous.matches(query.version, evidence.select(scope).version, verification_policy_version(),
                            require_semantic=require_semantic):
        return None

    # 复用上一轮的已支持目标
    coverage = {item["goal_id"]: item for item in previous.goal_coverage} # 取出上一轮核验的目标覆盖
    if len(coverage) != len(previous.goal_coverage) or set(coverage) != {goal.id for goal in query.goals}:
        return None
    retained, pending = [], []
    for goal in query.goals:
        item = coverage[goal.id]
        if item.get("status") == "supported":
            ids = item.get("supporting_chunk_ids", [])
            if not ids or not set(ids) <= (set(previous.evaluated_chunk_ids) & current_ids):
                return None
            retained.append({**goal.to_dict(), "goal_id": goal.id, "status": "supported",
                             "supporting_chunk_ids": list(ids)})
        elif item.get("status") in {"missing", "conflict"}:
            pending.append(goal)
        else:
            return None
    # retained表示上一轮已经核验过且有了证据支持的目标
    return {"retained_goals": retained, "pending_goals": tuple(pending)} if retained and pending else None


def _merge_gap_assessment(query, evidence, previous, partial, context):
    """按稳定 ID 合并逐目标结果，再重建引用序号；保留本轮与继承范围的区别。"""
    retained = {item["goal_id"]: item for item in previous.goal_coverage if item["status"] == "supported"}
    checked = {item["goal_id"]: item for item in partial.goal_coverage}
    expected = {goal.id for goal in context["pending_goals"]}
    if set(checked) != expected or len(checked) != len(partial.goal_coverage):
        raise ValueError("缺口核验未覆盖全部待核验目标")
    coverage = [retained.get(goal.id) or checked[goal.id] for goal in query.goals]
    evaluated = set(previous.evaluated_chunk_ids) | set(partial.evaluated_chunk_ids)
    ids = [item.id for item in evidence if item.id in evaluated]
    supported = {identifier for item in coverage for identifier in item["supporting_chunk_ids"]}
    sufficient = partial.sufficient and all(item["status"] == "supported" for item in coverage)
    return replace(partial, sufficient=sufficient, goal_coverage=coverage,
                   evaluated_chunk_ids=ids, evaluated_chunk_count=len(ids),
                   supporting_chunk_indices=[index for index, identifier in enumerate(ids, 1) if identifier in supported],
                   diagnostics={**partial.diagnostics,
                       "current_evaluated_chunk_ids": list(partial.evaluated_chunk_ids),
                       "inherited_goal_evidence_versions": {
                           goal_id: {identifier: evidence.get(identifier).version
                                     for identifier in item["supporting_chunk_ids"]}
                           for goal_id, item in retained.items()},
                       "inherited_report_evidence_version": previous.evidence_version,
                   })


def _verify_with_previous(query: QueryContext, evidence: EvidenceCollection,
                          previous: EvidenceAssessment, *, require_semantic: bool,
                          client: Any | None = None) -> EvidenceAssessment:
    """有历史报告时尝试增量与缺口核验，必要时恢复完整核验。"""
    selected = _incremental_sources(query, evidence, previous, require_semantic)
    gap = _gap_context(query, evidence, previous, require_semantic)
    extra = {"gap_context": gap} if gap is not None else {}
    assessment = assess_evidence_sufficiency(
        query.question, selected, client=client,
        goals=gap["pending_goals"] if gap else query.goals,
        require_semantic=require_semantic, **extra,
    )
    reduced = len(selected) < len(evidence)
    # 局部判定必须是完整模型结果；失败时不得把局部放行当作全局充分。
    gap_completed = (gap is not None and assessment.source == "model"
                     and assessment.verification_status == "completed")
    gap_conflict = gap_completed and (
        bool(assessment.diagnostics.get("recheck_goal_ids"))
        or any(item.get("status") == "conflict" for item in assessment.goal_coverage)
    )
    restore_reduced = reduced and not assessment.sufficient and (
        assessment.verification_status == "completed"
        or assessment.diagnostics.get("failure_kind") == "input_limit"
    )
    missing_context = gap is not None and assessment.diagnostics.get("failure_kind") == "input_limit"
    restored = bool(gap_conflict or restore_reduced or missing_context)
    if restored: # 如果有冲突就回退，重新核验整体目标和证据
        assessment = assess_evidence_sufficiency(
            query.question, evidence, client=client, goals=query.goals,
            require_semantic=require_semantic,
        )
    elif gap_completed:
        assessment = _merge_gap_assessment(query, evidence, previous, assessment, gap)
    elif gap is not None:
        assessment = replace(assessment, sufficient=False)
    if gap is not None:
        assessment = replace(assessment, diagnostics={**assessment.diagnostics,
            "gap_verification": True,
            "gap_goal_ids": [goal.id for goal in gap["pending_goals"]],
            "reused_goal_ids": [] if restored or not gap_completed else [item["goal_id"] for item in gap["retained_goals"]],
            "gap_restored_full_verification": restored,
        })
    assessment = replace(assessment, diagnostics={
        **assessment.diagnostics,
        "incremental_input": reduced,
        "incremental_selected_count": len(selected),
        "incremental_restored_full_input": restored,
    })
    return assessment


# 这里是校验所有证据和用户原问题，核验是否充分
def verify_evidence(query: QueryContext, evidence: EvidenceCollection, *,
                    previous: EvidenceAssessment | None = None, require_semantic: bool = True,
                    client: Any | None = None) -> EvidenceAssessment:
    """核验完整来源；仅复用查询、证据、规则和核验模式一致的报告。"""
    if not isinstance(evidence, EvidenceCollection):
        raise TypeError("证据核验必须接收 EvidenceCollection；外部记录应先在入口适配")

    # 证据冲突检查；同一个证据 ID 对应不同的原文或定位，无法继续核验。
    if evidence.conflicts:
        assessment = EvidenceAssessment(
            sufficient=False, reason="相同证据标识对应不同原文或定位", source="rule",
            missing_requirements=["无身份冲突的来源证据"],
            evaluated_chunk_count=len(evidence), evaluated_chunk_ids=[item.id for item in evidence],
        )
    # 如果之前已经进行过证据核验，且查询、证据、规则和核验模式一致，则复用之前的报告。
    elif previous is not None and previous.matches(query.version, evidence.version, verification_policy_version(),
                                       require_semantic=require_semantic):
        return previous
    elif previous is None:
        # 首次核验：使用全部目标与原始证据，不进入增量或缺口处理。
        assessment = assess_evidence_sufficiency(
            query.question, evidence, client=client, goals=query.goals,
            require_semantic=require_semantic,
        )
        assessment = replace(assessment, diagnostics={
            **assessment.diagnostics,
            "incremental_input": False,
            "incremental_selected_count": len(evidence),
            "incremental_restored_full_input": False,
        })
    else:
        assessment = _verify_with_previous(
            query, evidence, previous,
            require_semantic=require_semantic, client=client,
        )
    # 记录这一轮的证据id, 不一定是都经过校验的证据
    assessment = replace(assessment, diagnostics={
        **assessment.diagnostics, "verification_scope_ids": [item.id for item in evidence],
    })
    # 最后记录这份报告对应的：哪个问题和回答目标；哪一组证据；哪个版本的核验规则；是否要求语义核验。
    return finalize_assessment(query, evidence, assessment, require_semantic=require_semantic)
