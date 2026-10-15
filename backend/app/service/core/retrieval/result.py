"""纯检索结果：保存来源和排名，不携带意图分析或答案充分性结论。"""

from dataclasses import dataclass, field, replace
from copy import deepcopy
from typing import Any

from domain.evidence import EvidenceCollection
from service.core.evidence.adapter import collection_from_records, reference_records
from service.core.query.contract import RetrievalTask


@dataclass(slots=True)
class RetrievalResult:
    evidence: EvidenceCollection
    candidates: EvidenceCollection = field(default_factory=EvidenceCollection)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    total_candidates: int | None = None
    task: RetrievalTask | None = None

    def copy(self) -> "RetrievalResult":
        # 来源对象内部包含只读映射，不能 deepcopy；集合提供自己的快照复制。
        return replace(self, evidence=self.evidence.copy(), candidates=self.candidates.copy(),
                       diagnostics=deepcopy(self.diagnostics))


def adapt_retrieval_records(records: list[dict], diagnostics: dict, *,
                            task: RetrievalTask | None = None) -> RetrievalResult:
    """外部检索结果只做来源适配，不在适配过程中触发模型核验。"""
    query = task.query if task else ""
    evidence = collection_from_records(records, query=query)
    candidates = collection_from_records(diagnostics.get("candidate_chunks") or [], query=query)
    candidates.merge(evidence)
    business_fields = {"candidate_chunks", "evidence_sufficiency", "query_contract", "query_intent",
                       "query_understanding", "clarification", "response_mode", "total"}
    return RetrievalResult(evidence, candidates,
                           {k: v for k, v in diagnostics.items() if k not in business_fields},
                           diagnostics.get("total"), task)


def retrieval_record(result: RetrievalResult) -> dict[str, Any]:
    """评测出口的 chunks 表示检索命中，不能据此推断是否允许回答。"""
    records = reference_records(result.evidence)
    return {
        **result.diagnostics, "chunks": records,
        "total": result.total_candidates if result.total_candidates is not None else len(records),
        "retrieval_task": result.task.to_dict() if result.task else None,
        "chunks_before_sufficiency": reference_records(result.candidates),
    }
