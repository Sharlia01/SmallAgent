"""充分性标签独立于系统决定，并在逐题、归因和汇总中保持一致。"""
from copy import deepcopy

import pytest

from evaluation.end_to_end import aggregate_e2e_results, evaluate_refusal, evaluate_sufficiency
from evaluation.retrieval import evaluate_retrieval_case, aggregate_retrieval_stages

pytestmark = pytest.mark.unit


def table_sample(*, requirement=False):
    sample = {"id": "table", "answerable": True, "relevant_evidence": [
        {"document_name": "报告.pdf", "evidence_type": "table", "row_cells": ["收入", "100"], "text": "收入100"}
    ]}
    if requirement:
        sample["evidence_requirements"] = [{"id": "income", "alternatives": [[
            {"type": "table_row", "document_name": "报告.pdf", "row_cells": ["收入", "100"]}
        ]]}]
    return sample


def table_chunk(identifier, binding):
    return {"chunk_id": identifier, "document_name": "报告.pdf", "content_with_weight": "收入100",
            "evidence_type_kwd": "table", "table_cells_slice_obj": {"rows": [{"cells": ["收入", "100"]}]},
            **({"table_binding_kwd": binding} if binding is not None else {})}


def score(sample, chunks, *, actual=False):
    return evaluate_retrieval_case(sample, {
        "evaluation_scope": "agent_evidence_sets", "chunks": chunks if actual else [],
        "chunks_before_sufficiency": chunks, "retrieval_calls": [],
        "evidence_sufficiency": {"source": "rule", "sufficient": actual},
    }, latency_ms=1, top_k=5, match_threshold=1)


def e2e(case):
    return {**case, "latency_ms": {"total": 1}, "answer_metrics": {}, "citation_metrics": {},
            "response_mode": "refuse", "refusal_metrics": evaluate_refusal(case, "refuse"),
            "sufficiency_metrics": evaluate_sufficiency(case, case)}


@pytest.mark.parametrize("requirement", [False, True])
@pytest.mark.parametrize("binding,expected", [("bound", True), ("unbound", False), (None, None), ("legacy", None)])
def test_table_annotation_form_does_not_change_reference(requirement, binding, expected):
    sample = table_sample(requirement=requirement)
    case = score(sample, [table_chunk("table", binding)])
    # 都是检索命中；结构可靠性是另一项判断。
    assert case["verification_evidence_metrics"]["recall"] == 1
    assert case["evidence_sufficiency_reference"]["expected_sufficient"] is expected
    sufficiency = evaluate_sufficiency(sample, case)
    assert sufficiency["expected_sufficient"] is expected
    assert case["gate_metrics"]["pre_gate_evidence_sufficient"] is expected
    summary = aggregate_e2e_results([e2e(case)])
    assert summary["gate_metrics"]["false_rejection_with_sufficient_evidence_count"] == int(expected is True)
    assert summary["sufficiency_metrics"]["false_rejection_count"] == int(expected is True)
    assert summary["sufficiency_metrics"]["decision_count"] == int(expected is not None)
    assert summary["sufficiency_metrics"]["expected_unknown_count"] == int(expected is None)
    assert summary["gate_metrics"]["evidence_attribution_unavailable_answerable_decision_count"] == int(expected is None)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("requirement", [False, True])
def test_later_usable_chunk_is_not_hidden_by_first_unbound_match(reverse, requirement):
    chunks = [table_chunk("bad", "unbound"), table_chunk("good", "bound")]
    if reverse:
        chunks.reverse()
    case = score(table_sample(requirement=requirement), chunks)
    assert case["evidence_sufficiency_reference"]["expected_sufficient"] is True
    match = case["evidence_matches_before_sufficiency"][0]
    if requirement:
        match = match["alternative_matches"][0]["atom_matches"][0]
    assert len(match["candidate_matches"]) == 2
    assert next(c for c in match["candidate_matches"] if c["matched_chunk_id"] == "good")["usable"] is True


@pytest.mark.parametrize("reverse", [False, True])
def test_complete_text_alternative_can_replace_unusable_table(reverse):
    sample = table_sample(requirement=True)
    alternatives = sample["evidence_requirements"][0]["alternatives"]
    alternatives.append([{"type": "text", "document_name": "报告.pdf", "text": "正文明确收入一百"}])
    if reverse:
        alternatives.reverse()
    chunks = [table_chunk("bad", "unbound"),
              {"chunk_id": "text", "document_name": "报告.pdf", "content_with_weight": "正文明确收入一百"}]
    case = score(sample, chunks)
    assert case["evidence_sufficiency_reference"]["expected_sufficient"] is True
    match = case["evidence_matches_before_sufficiency"][0]
    assert {a["usable"] for a in match["alternative_matches"]} == {True, False}


def test_atoms_from_different_incomplete_alternatives_cannot_be_combined():
    atom = lambda text: {"type": "text", "document_name": "报告.pdf", "text": text}
    sample = {"id": "partial", "answerable": True, "evidence_requirements": [
        {"id": "complete_pair", "alternatives": [[atom("收入一百"), atom("成本五十")],
                                                   [atom("营收两百"), atom("支出八十")]]}
    ]}
    chunks = [{"chunk_id": str(i), "document_name": "报告.pdf", "content_with_weight": text}
              for i, text in enumerate(["收入一百", "支出八十"])]
    case = score(sample, chunks)
    assert case["evidence_sufficiency_reference"]["expected_sufficient"] is False
    assert all(not a["usable"] for a in case["evidence_matches_before_sufficiency"][0]["alternative_matches"])


@pytest.mark.parametrize("requirement", [False, True])
@pytest.mark.parametrize("status,expected", [("extracted", True), ("unextracted", False), (None, None)])
def test_figure_uses_same_reference_for_both_metric_groups(requirement, status, expected):
    figure = {"document_name": "报告.pdf", "evidence_type": "figure", "locator": "图1 发电量", "text": "发电量下降"}
    sample = {"id": "figure", "answerable": True, "relevant_evidence": [figure]}
    if requirement:
        sample["evidence_requirements"] = [{"id": "figure", "alternatives": [[{**figure, "type": "figure"}]]}]
    chunk = {"chunk_id": "figure", "document_name": "报告.pdf", "content_with_weight": "图1 发电量",
             **({"visual_status_kwd": status} if status else {})}
    case = score(sample, [chunk])
    assert case["verification_evidence_metrics"]["recall"] == 1
    assert case["gate_metrics"]["pre_gate_evidence_sufficient"] is expected
    assert evaluate_sufficiency(sample, case)["expected_sufficient"] is expected


def test_unknown_alternative_does_not_become_false_and_known_failure_dominates_and():
    sample = table_sample(requirement=True)
    sample["evidence_requirements"][0]["alternatives"].append([
        {"type": "text", "document_name": "报告.pdf", "text": "未找到的正文"}
    ])
    case = score(sample, [table_chunk("unknown", None)])
    assert case["evidence_sufficiency_reference"]["expected_sufficient"] is None
    sample["evidence_requirements"].append({"id": "missing", "alternatives": [[
        {"type": "text", "document_name": "报告.pdf", "text": "必须另有培训条款"}
    ]]})
    assert score(sample, [table_chunk("unknown", None)])["evidence_sufficiency_reference"]["expected_sufficient"] is False


@pytest.mark.parametrize("expected", [True, False])
def test_explicit_annotation_is_shared_and_never_overwritten_by_recall(expected):
    sample = table_sample()
    sample["expected_evidence_sufficient"] = expected
    case = score(sample, [table_chunk("unbound", "unbound")])
    assert case["evidence_sufficiency_reference"]["source"] == "annotation"
    # 人为修改 recall，确认汇总不重新推断已有标签。
    case["verification_evidence_metrics"]["recall"] = float(not expected)
    assert evaluate_sufficiency({}, case)["expected_sufficient"] is expected
    gate = aggregate_retrieval_stages([case], top_k=5)["gate_metrics"]
    assert gate["false_rejection_with_sufficient_evidence_count"] == int(expected)
    assert gate["correct_rejection_after_insufficient_evidence_count"] == int(not expected)


def test_reference_is_independent_of_system_verdict():
    sample = table_sample()
    chunks = [table_chunk("table", "bound")]
    accepted = score(sample, chunks, actual=True)
    rejected = score(sample, chunks, actual=False)
    assert accepted["evidence_sufficiency_reference"] == rejected["evidence_sufficiency_reference"]


def test_missing_snapshot_and_legacy_records_stay_unknown_even_with_perfect_recall():
    case = score(table_sample(), [table_chunk("table", "bound")])
    legacy = deepcopy(case)
    legacy.pop("evidence_sufficiency_reference")
    for match in legacy["evidence_matches_before_sufficiency"]:
        match.pop("usable")
        match.pop("candidate_matches")
    assert evaluate_sufficiency(table_sample(), legacy)["expected_sufficient"] is None
    gate = aggregate_retrieval_stages([legacy], top_k=5)["gate_metrics"]
    assert gate["false_rejection_with_sufficient_evidence_count"] == 0
    assert gate["evidence_attribution_unavailable_answerable_decision_count"] == 1
    unknown = score(table_sample(), None)
    assert unknown["evidence_sufficiency_reference"]["status"] == "unknown"
    assert score(table_sample(), [])["evidence_sufficiency_reference"]["status"] == "insufficient"
