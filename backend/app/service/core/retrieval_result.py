"""检索服务返回的业务结果；诊断不承担业务状态的传递。"""

from dataclasses import asdict, dataclass, field
from typing import Any

from domain.evidence import EvidenceCollection
from domain.assessment import EvidenceAssessment
from service.core.evidence_adapter import collection_from_records, reference_records
from service.core.evidence_assessment import verification_policy_version
from service.core.evidence_verification import verify_evidence
from service.core.query_context import QueryContext
from service.core.query_contract import QueryRequest
from service.core.query_intent import QueryAnalysis
from service.core.financial_facts import project_fact_context


@dataclass(slots=True)
class RetrievalResult:
    evidence: EvidenceCollection
    candidates: EvidenceCollection
    assessment: EvidenceAssessment | None
    clarification: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    # 索引融合后的候选总数，不等于选中片段数量；外部检索器未提供时为 None。
    total_candidates: int | None = None
    # 检索发生时的类型化查询信息，仅用于输出快照，不替换调用方的 QueryContext。
    request: QueryRequest | None = None
    analysis: QueryAnalysis | None = None


def retrieval_result(query: QueryContext, evidence: EvidenceCollection, candidates: EvidenceCollection,
                     assessment: EvidenceAssessment | None, diagnostics: dict,
                     clarification: str | None = None, *, total_candidates: int | None = None) -> RetrievalResult:
    """组装业务结果；来源与报告已由同一次核验确定，不重新解析或核验。"""
    if assessment and not clarification:
        # 组装结果不能把未核验判定绑定成报告；候选顺序也是报告范围的一部分。
        sources = candidates.copy()
        sources.merge(evidence)
        if not assessment.matches(query.version, sources.version, verification_policy_version()):
            raise ValueError("核验报告未完成范围绑定或已失效，必须先核验当前查询与证据集合")
    business_fields = {"query_contract", "query_intent", "candidate_chunks", "evidence_sufficiency", "clarification", "response_mode", "total"}
    return RetrievalResult(evidence, candidates, assessment, clarification,
                           {key: value for key, value in diagnostics.items() if key not in business_fields},
                           total_candidates=total_candidates, request=query.request,
                           analysis=query.prepared.analysis if query.prepared else None)


def adapt_retrieval_records(query: QueryContext, records: list[dict], diagnostics: dict) -> RetrievalResult:
    """适配外部来源并委托统一核验；JSON 中的判定不能证明当前核验范围。"""
    evidence = collection_from_records(records, query=query.question)
    candidates = collection_from_records(diagnostics.get("candidate_chunks") or [], query=query.question)
    clarification = query.clarification or (diagnostics.get("clarification") if diagnostics.get("response_mode") == "clarify" else None)
    assessment = None
    if not clarification:
        candidates.merge(evidence)
        assessment = verify_evidence(query, candidates)
        evidence = (evidence or candidates.copy()) if assessment.sufficient else EvidenceCollection()
    return retrieval_result(query, evidence, candidates, assessment, diagnostics, clarification,
                            total_candidates=diagnostics.get("total"))


def retrieval_record(result: RetrievalResult) -> dict[str, Any]:
    """JSON 出口，筛选正文独立输出，不能重新作为原始来源进入核验。"""
    records = reference_records(result.evidence)
    if result.assessment and result.assessment.facts.applicable:
        blocks = {block.evidence_id: block for block in project_fact_context(result.evidence, result.assessment.facts)}
        for record in records:
            record["answer_context"] = blocks[record["chunk_id"]].text
            record["financial_facts"] = [asdict(fact) for fact in result.assessment.facts.facts
                                         if fact.source["chunk_id"] == record["chunk_id"]]
    total = result.total_candidates if result.total_candidates is not None else len(result.candidates or result.evidence)
    return {
        **result.diagnostics, "chunks": records, "total": total if records else 0,
        "doc_aggs": result.diagnostics.get("doc_aggs", []) if records else [],
        "query_contract": result.request.to_dict() if result.request else {},
        "query_intent": result.analysis.to_dict() if result.analysis else {},
        "chunks_before_sufficiency": reference_records(result.candidates),
        "evidence_sufficiency": result.assessment.to_dict() if result.assessment else {},
        "response_mode": "clarify" if result.clarification else "answer" if records else "insufficient",
        "clarification": result.clarification,
    }
