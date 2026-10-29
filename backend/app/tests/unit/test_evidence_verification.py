"""完整核验入口的模式隔离、报告范围和外部适配约束。"""

import pytest

from agent.loop import collect_tool_evidence
from agent.schemas import ToolResult
from domain.assessment import EvidenceAssessment
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
    def sufficiency(question, sources, *, require_semantic, **kwargs):
        checked.append(require_semantic)
        return assessment(sources)

    monkeypatch.setattr(verification, "assess_evidence_sufficiency", sufficiency)
    first = verification.verify_evidence(query, evidence, require_semantic=initial_mode)
    assert first.require_semantic is initial_mode
    assert verification.verify_evidence(query, evidence, previous=first, require_semantic=initial_mode) is first

    second = verification.verify_evidence(query, evidence, previous=first, require_semantic=not initial_mode)
    assert second is not first
    assert second.require_semantic is not initial_mode
    assert checked == [initial_mode, not initial_mode]
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
    monkeypatch.setattr(verification, "assess_evidence_sufficiency",
                        lambda *args, **kwargs: pytest.fail("同一证据集合的报告应复用"))
    assert verification.verify_evidence(query, combined, previous=report) is report


def test_context_budget_change_invalidates_previous_report(monkeypatch):
    query, evidence = make_query("差旅如何报销？"), sources()
    monkeypatch.setenv("RAG_EVIDENCE_SUFFICIENCY_CONTEXT_TOKENS", "16384")
    report = finalize_assessment(query, evidence, assessment(evidence))
    monkeypatch.setenv("RAG_EVIDENCE_SUFFICIENCY_CONTEXT_TOKENS", "32768")
    assert not report.matches(query.version, evidence.version, verification_policy_version())


def incremental_fixture(monkeypatch, *, conflict=False):
    query = make_query("收入和原因？")
    old = collection_from_records([
        {"chunk_id": identifier, "content_with_weight": identifier + " 原文"}
        for identifier in ("unused", "global", "goal")
    ])
    def first_check(question, evidence, **kwargs):
        return EvidenceAssessment(
            False, "原因缺失", source="model", missing_requirements=["原因"],
            evaluated_chunk_ids=[item.id for item in evidence], evaluated_chunk_count=len(evidence),
            supporting_chunk_indices=[2, 3],
            goal_coverage=[{"goal_id": "income", "status": "conflict" if conflict else "supported",
                            "supporting_chunk_ids": ["goal"]},
                           {"goal_id": "reason", "status": "missing",
                            "supporting_chunk_ids": ["global"]}],
        )
    monkeypatch.setattr(verification, "assess_evidence_sufficiency", first_check)
    previous = verification.verify_evidence(query, old)
    current = old.copy()
    current.merge(collection_from_records([{"chunk_id": "new", "content_with_weight": "新增原因"}]))
    return query, current, previous


def test_incremental_union_reindexes_and_binds_complete_scope(monkeypatch):
    query, current, previous = incremental_fixture(monkeypatch)
    calls = []
    def check(question, evidence, **kwargs):
        calls.append([item.id for item in evidence])
        return EvidenceAssessment(True, "完整", source="model", supporting_chunk_indices=[1, 3],
                                  evaluated_chunk_ids=calls[-1], evaluated_chunk_count=len(evidence))
    monkeypatch.setattr(verification, "assess_evidence_sufficiency", check)
    report = verification.verify_evidence(query, current, previous=previous)
    assert calls == [["global", "goal", "new"]]
    context = build_answer_context(query, current, report)
    assert [block.evidence_id for block in context.blocks] == ["global", "new"]
    assert verification.verify_evidence(query, current, previous=report) is report
    assert len(calls) == 1
    assert previous.evaluated_chunk_ids == ["unused", "global", "goal"]


@pytest.mark.parametrize("failure", ["insufficient", "input_limit", "service_unavailable"])
def test_incremental_restores_material_only_for_evidence_shortfall(monkeypatch, failure):
    query, current, previous = incremental_fixture(monkeypatch)
    calls = []
    def check(question, evidence, **kwargs):
        calls.append([item.id for item in evidence])
        if len(calls) == 2:
            return assessment(evidence)
        return EvidenceAssessment(False, "未通过", source="model" if failure == "insufficient" else "fallback",
                                  verification_status="completed" if failure == "insufficient" else "unavailable",
                                  diagnostics={"failure_kind": failure})
    monkeypatch.setattr(verification, "assess_evidence_sufficiency", check)
    report = verification.verify_evidence(query, current, previous=previous)
    assert calls[0] == ["global", "goal", "new"]
    restored = failure != "service_unavailable"
    assert len(calls) == (2 if restored else 1)
    assert report.diagnostics["incremental_restored_full_input"] is restored
    if restored:
        assert calls[1] == [item.id for item in current]


@pytest.mark.parametrize("change", ["content", "removed", "query", "policy", "conflict", "tampered"])
def test_incremental_invalid_old_scope_uses_all_candidates(monkeypatch, change):
    query, current, previous = incremental_fixture(monkeypatch, conflict=change == "conflict")
    if change == "content":
        current = collection_from_records([
            {"chunk_id": item.id, "content_with_weight": "修改" if item.id == "global" else item.text}
            for item in current
        ])
    elif change == "removed":
        current = current.select(["unused", "goal", "new"])
    elif change == "query":
        query = make_query("另一个问题")
    elif change == "policy":
        monkeypatch.setenv("RAG_EVIDENCE_SUFFICIENCY_CONTEXT_TOKENS", "12345")
    elif change == "tampered":
        previous.goal_coverage.clear()
    calls = []
    def check(question, evidence, **kwargs):
        calls.append([item.id for item in evidence])
        return assessment(evidence)
    monkeypatch.setattr(verification, "assess_evidence_sufficiency", check)
    verification.verify_evidence(query, current, previous=previous)
    assert calls == [[item.id for item in current]]
