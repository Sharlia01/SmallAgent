"""检索、Agent 和评测共用的完整证据核验入口。"""

from dataclasses import replace
from typing import Any

from domain.assessment import EvidenceAssessment
from domain.evidence import EvidenceCollection
from service.core.evidence.assessment import finalize_assessment, verification_policy_version
from service.core.evidence.sufficiency import assess_evidence_sufficiency
from service.core.finance.facts import assess_financial_facts, financial_rejection
from service.core.query.context import QueryContext


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
    elif previous and previous.matches(query.version, evidence.version, verification_policy_version(),
                                       require_semantic=require_semantic):
        return previous
    else:
        fact_assessment = assess_financial_facts(
            query.financial_requirements, evidence, allow_model=require_semantic, client=client,
        )
        if fact_assessment.applicable and not fact_assessment.sufficient:
            assessment = financial_rejection(evidence, fact_assessment, query.goals)
        else:
            assessment = assess_evidence_sufficiency(
                query.question, evidence, facts=fact_assessment, client=client,
                goals=query.goals, require_semantic=require_semantic,
            )
            # 前面明明做过财务核验，但后面返回的报告没有带上适用的财务结果，那就把前面的结果补进去。
            if not assessment.facts.applicable and fact_assessment.applicable:
                assessment = replace(assessment, facts=fact_assessment)
    # 最后记录这份报告对应的：哪个问题和回答目标；哪一组证据；哪个版本的核验规则；是否要求语义核验。
    return finalize_assessment(query, evidence, assessment, require_semantic=require_semantic)
