"""检索、Agent 和评测共用的完整证据核验入口。"""

from dataclasses import replace
from typing import Any

from domain.assessment import EvidenceAssessment
from domain.evidence import EvidenceCollection
from service.core.evidence_assessment import finalize_assessment, verification_policy_version
from service.core.evidence_sufficiency import assess_evidence_sufficiency
from service.core.financial_facts import assess_financial_facts, financial_rejection
from service.core.query_context import QueryContext


def verify_evidence(query: QueryContext, evidence: EvidenceCollection, *,
                    previous: EvidenceAssessment | None = None, require_semantic: bool = True,
                    client: Any | None = None) -> EvidenceAssessment:
    """核验完整来源；仅复用查询、证据、规则和核验模式一致的报告。"""
    if not isinstance(evidence, EvidenceCollection):
        raise TypeError("证据核验必须接收 EvidenceCollection；外部记录应先在入口适配")
    if evidence.conflicts:
        assessment = EvidenceAssessment(
            sufficient=False, reason="相同证据标识对应不同原文或定位", source="rule",
            missing_requirements=["无身份冲突的来源证据"],
            evaluated_chunk_count=len(evidence), evaluated_chunk_ids=[item.id for item in evidence],
        )
    elif previous and previous.matches(query.version, evidence.version, verification_policy_version(),
                                       require_semantic=require_semantic):
        return previous
    else:
        facts = assess_financial_facts(query.question, evidence, query.goals,
                                       allow_model=require_semantic, client=client)
        if facts.applicable and not facts.sufficient:
            assessment = financial_rejection(evidence, facts, query.goals)
        else:
            assessment = assess_evidence_sufficiency(query.question, evidence, facts=facts, client=client,
                                                    goals=query.goals, require_semantic=require_semantic)
            # 充分性评估只选择支持事实，不能丢失已经完成的规则核验结果。
            if not assessment.facts.applicable and facts.applicable:
                assessment = replace(assessment, facts=facts)
    return finalize_assessment(query, evidence, assessment, require_semantic=require_semantic)
