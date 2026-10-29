"""核验报告的范围绑定，供检索与最终回答路径共用。"""

import os
from dataclasses import replace

from domain.assessment import EvidenceAssessment
from domain.evidence import EvidenceCollection, fingerprint
from service.core.entities.registry import get_entity_registry
from service.core.query.context import QueryContext
from service.core.runtime.context import POLICY_VERSION
from service.core.query.semantics import domain_policy


def verification_policy_version() -> str:
    settings = {name: os.getenv(name) for name in (
        "RAG_EVIDENCE_SUFFICIENCY_ENABLED", "RAG_EVIDENCE_SUFFICIENCY_FAIL_OPEN",
        "RAG_EVIDENCE_SUFFICIENCY_MODEL", "RAG_EVIDENCE_SUFFICIENCY_TIMEOUT_SECONDS",
        "RAG_EVIDENCE_SUFFICIENCY_TRANSIENT_RETRY",
        "RAG_EVIDENCE_SUFFICIENCY_CONTEXT_TOKENS",
    )}
    return fingerprint((POLICY_VERSION, "unified-evidence-v9-gap", domain_policy()["version"],
                        get_entity_registry().version, settings))


def finalize_assessment(query: QueryContext, evidence: EvidenceCollection, assessment: EvidenceAssessment, *,
                        require_semantic: bool = True) -> EvidenceAssessment:
    """绑定用户查询、回答目标、证据，核验规则；不执行新的业务判断。"""
    policy_version = verification_policy_version()
    if assessment.matches(query.version, evidence.version, policy_version, require_semantic=require_semantic):
        return assessment
    if assessment.query_version or assessment.evidence_version or assessment.policy_version:
        # 范围已绑定的旧报告不能被直接贴到新的查询或证据上，必须重新核验。
        raise ValueError("核验报告已失效，不能重新绑定到另一份查询、证据、规则或核验模式")

    return replace(assessment, query_version=query.version, evidence_version=evidence.version,
                   policy_version=policy_version, require_semantic=require_semantic)
