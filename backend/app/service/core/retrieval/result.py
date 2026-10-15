"""纯检索结果：保存来源和排名，不携带意图分析或答案充分性结论。"""

from dataclasses import dataclass, field, replace
from copy import deepcopy
from typing import Any

from domain.evidence import EvidenceCollection
from service.core.evidence.adapter import collection_from_records, reference_records
from service.core.query.contract import RetrievalTask


@dataclass(slots=True)
class RetrievalResult:
    evidence: EvidenceCollection  # 交给 Agent 的完整检索资料；是否足以回答由后续核验决定。
    diagnostics: dict[str, Any] = field(default_factory=dict) #检索过程的诊断信息，用于分析耗时
    total_candidates: int | None = None    # 底层检索报告的候选总数，可能总共有50条候选，但只返回了前5条
    task: RetrievalTask | None = None

    def copy(self) -> "RetrievalResult":
        # 来源对象内部包含只读映射，不能 deepcopy；集合提供自己的快照复制。
        return replace(self, evidence=self.evidence.copy(), diagnostics=deepcopy(self.diagnostics))


def adapt_retrieval_records(records: list[dict], diagnostics: dict, *,
                            task: RetrievalTask | None = None) -> RetrievalResult:
    """外部检索结果只做来源适配，不在适配过程中触发模型核验。"""
    query = task.query if task else ""
    # 外部旧格式的额外候选在入口合并，内部只传递一份待核验资料。
    # 保持原先“候选在前、主要结果补入”的顺序，避免改变引用及核验版本。
    evidence = collection_from_records(diagnostics.get("candidate_chunks") or [], query=query)
    evidence.merge(collection_from_records(records, query=query))
    business_fields = {"candidate_chunks", "evidence_sufficiency", "query_contract", "query_intent",
                       "query_understanding", "clarification", "response_mode", "total"}
    return RetrievalResult(
        evidence=evidence,
        diagnostics={k: v for k, v in diagnostics.items() if k not in business_fields},
        total_candidates=diagnostics.get("total"), task=task,
    )


def retrieval_record(result: RetrievalResult) -> dict[str, Any]:
    records = reference_records(result.evidence)
    return {
        **result.diagnostics, "chunks": records,
        "total": result.total_candidates if result.total_candidates is not None else len(records),
        "retrieval_task": result.task.to_dict() if result.task else None,
        # 保留评测协议中的核验前快照；最终回答引用在回答阶段另行生成。
        "chunks_before_sufficiency": reference_records(result.evidence),
    }
