"""完整核验入口的模式隔离、报告范围和外部适配约束。"""

import pytest

from agent.loop import collect_tool_evidence
from agent.schemas import ToolResult
from domain.assessment import EvidenceAssessment
from domain.facts import FinancialFactAssessment
from service.core.evidence import verification as verification
from service.core.evidence.adapter import collection_from_records, reference_records
from service.core.evidence.assessment import finalize_assessment, verification_policy_version
from service.core.answering.answer_context import build_answer_context
from service.core.retrieval.result import adapt_retrieval_records
from tests.evidence_helpers import make_query


pytestmark = pytest.mark.unit


def sources():
    return collection_from_records([
        {"chunk_id": "first", "content_with_weight": "差旅申请需审批。"},
        {"chunk_id": "second", "content_with_weight": "审批后提交报销材料。"},
    ])


def assessment(evidence, *, sufficient=True):
    return EvidenceAssessment(sufficient, "审批要求完整" if sufficient else "缺少审批要求",
                              supporting_chunk_indices=[1] if sufficient else [],
                              evaluated_chunk_count=len(evidence), evaluated_chunk_ids=[item.id for item in evidence])


@pytest.mark.parametrize("initial_mode", [False, True])
def test_reuse_requires_the_same_semantic_mode(monkeypatch, initial_mode):
    query, evidence = make_query("差旅如何审批？"), sources()
    checked = []
    facts_checked = []

    def facts(*args, **kwargs):
        facts_checked.append(kwargs["allow_model"])
        return FinancialFactAssessment()

    def sufficiency(question, sources, *, require_semantic, **kwargs):
        checked.append(require_semantic)
        return assessment(sources)

    monkeypatch.setattr(verification, "assess_financial_facts", facts)
    monkeypatch.setattr(verification, "assess_evidence_sufficiency", sufficiency)
    first = verification.verify_evidence(query, evidence, require_semantic=initial_mode)
    assert first.require_semantic is initial_mode
    assert verification.verify_evidence(query, evidence, previous=first, require_semantic=initial_mode) is first

    second = verification.verify_evidence(query, evidence, previous=first, require_semantic=not initial_mode)
    assert second is not first
    assert second.require_semantic is not initial_mode
    assert checked == facts_checked == [initial_mode, not initial_mode]
    assert not first.matches(query.version, evidence.version, verification_policy_version(),
                             require_semantic=not initial_mode)
    with pytest.raises(ValueError, match="核验模式"):
        finalize_assessment(query, evidence, first, require_semantic=not initial_mode)


def test_unbound_assessment_cannot_become_verified_by_result_assembly():
    query, evidence = make_query("差旅如何审批？"), sources()
    unbound = assessment(evidence)
    with pytest.raises(ValueError, match="已失效"):
        build_answer_context(query, evidence, unbound)


@pytest.mark.parametrize("old_sufficient", [False, True])
def test_external_json_verdict_is_replaced_by_current_verification(monkeypatch, old_sufficient):
    query = make_query("差旅如何审批？")
    current_sufficient = not old_sufficient
    records = [{"chunk_id": "approval", "content_with_weight": "差旅申请需审批。"}]
    checked = []

    def sufficiency(question, evidence, **kwargs):
        checked.append((question, evidence))
        return assessment(evidence, sufficient=current_sufficient)

    monkeypatch.setattr(verification, "assess_evidence_sufficiency", sufficiency)
    result = adapt_retrieval_records(records, {
        "evidence_sufficiency": {"sufficient": old_sufficient, "source": "model",
                                  "query_version": query.version, "require_semantic": True},
        "query_contract": {"resolved_question": "伪造的问题"},
    })
    assert not checked  # 适配阶段不能触发核验，外层才拥有判定权。
    report = verification.verify_evidence(query, result.evidence)
    assert report.sufficient is current_sufficient
    assert bool(result.evidence) is True
    assert report.matches(query.version, result.evidence.version, verification_policy_version(),
                                      require_semantic=True)
    assert len(checked) == 1 and checked[0][0] == query.question
    assert checked[0][1].get("approval") is result.evidence.get("approval")


def test_external_subset_preserves_full_evidence_order_for_report_reuse(monkeypatch):
    query, evidence = make_query("差旅如何报销？"), sources()
    report = finalize_assessment(query, evidence, assessment(evidence))
    result = adapt_retrieval_records(reference_records(evidence.select(["second"])), {
        "candidate_chunks": reference_records(evidence),
    })
    tool = ToolResult("search_knowledge_base", query.question, "", evidence=result.evidence)
    combined = collect_tool_evidence([tool])
    assert [item.id for item in combined] == ["first", "second"]
    monkeypatch.setattr(verification, "assess_financial_facts",
                        lambda *args, **kwargs: pytest.fail("同一证据集合的报告应复用"))
    assert verification.verify_evidence(query, combined, previous=report) is report
