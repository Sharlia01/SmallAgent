"""独立问答与端到端评测共用 Agent；外部报告不能替代服务端核验。"""
import pytest
from agent import loop
from evaluation.agent import execute_agent_turn
from tests.agent_helpers import ModelClient, initial, records, coverage_verifier, task
from tests.unit.test_agent_orchestrator import AnswerClient
from service.core.retrieval.result import adapt_retrieval_records
from service.core.query.contract import RetrievalTask

pytestmark = pytest.mark.unit


def test_evaluation_preserves_contract_and_returns_full_evidence(monkeypatch):
    monkeypatch.setattr(loop, "verify_evidence", coverage_verifier())
    planner, answer = ModelClient([initial()]), AnswerClient()
    calls = []
    def retrieve(index, query, **options):
        calls.append(query)
        return {"chunks": records(), "evidence_sufficiency": {"sufficient": False}}
    result = execute_agent_turn(index_names="test", question="员工培训要求", planner_client=planner,
                                answer_client=answer, retrieval_function=retrieve)
    assert result.response_mode == "answer"
    assert result.raw_retrieval["query_contract"]["user_question"] == "员工培训要求"
    assert result.raw_retrieval["query_understanding"]["goals"][0]["id"] == "g1"
    assert len(result.documents) == 1 and len(calls) == 1
    assert result.raw_retrieval["evidence_sufficiency"]["sufficient"]


def test_external_positive_verdict_cannot_force_answer(monkeypatch):
    monkeypatch.setattr(loop, "verify_evidence", coverage_verifier(2))
    answer = AnswerClient()
    result = execute_agent_turn(index_names="test", question="员工培训要求",
        planner_client=ModelClient([initial(), {"tasks": []}]), answer_client=answer,
        retrieval_function=lambda *args, **kwargs: {"chunks": records(), "evidence_sufficiency": {"sufficient": True}})
    assert result.response_mode == "refuse" and not answer.calls
    assert len(result.raw_retrieval["chunks_before_sufficiency"]) == 1


@pytest.mark.parametrize("typed", [False, True])
@pytest.mark.parametrize("required_count", [2, 3])
def test_evaluation_snapshot_keeps_all_evidence_on_answer_and_refusal(monkeypatch, typed, required_count):
    checked = []
    monkeypatch.setattr(loop, "verify_evidence", coverage_verifier(required_count, checked))

    def retrieve(index, query, **options):
        # 外部快照与主要结果不必互相包含；两边的资料都要进入核验和评测。
        if typed:
            return adapt_retrieval_records(records("c1"), {"candidate_chunks": records("c2")},
                                           task=RetrievalTask(query, ("g1",)))
        return {"chunks": records("c1"), "chunks_before_sufficiency": records("c2")}

    result = execute_agent_turn(index_names="test", question="员工培训要求",
        planner_client=ModelClient([initial(), {"tasks": []}]), answer_client=AnswerClient(),
        retrieval_function=retrieve)
    snapshot = result.raw_retrieval["chunks_before_sufficiency"]
    assert [item.id for item in checked[0][1]] == ["c2", "c1"]
    assert [chunk["chunk_id"] for chunk in snapshot] == ["c2", "c1"]
    assert result.response_mode == ("answer" if required_count == 2 else "refuse")
    if required_count == 3:
        assert result.raw_retrieval["chunks"] == [] and result.documents == []


def test_evaluation_snapshot_merges_replanned_results_in_verification_order(monkeypatch):
    checked = []
    monkeypatch.setattr(loop, "verify_evidence", coverage_verifier(2, checked))
    responses = iter([
        {"chunks": records("c1")},
        {"chunks": records("c1") + records("c2")},
    ])
    result = execute_agent_turn(index_names="test", question="员工培训要求",
        planner_client=ModelClient([initial(), {"tasks": [task(id="s2", query="员工培训要求 完整制度")]}]),
        answer_client=AnswerClient(), retrieval_function=lambda *args, **kwargs: next(responses))
    assert result.response_mode == "answer"
    assert [len(evidence) for _, evidence in checked] == [1, 2]
    assert [chunk["chunk_id"] for chunk in result.raw_retrieval["chunks_before_sufficiency"]] == ["c1", "c2"]


@pytest.mark.parametrize("refuse", [False, True])
def test_repaired_sixth_evidence_is_scored_and_only_selected_context_is_delivered(monkeypatch, refuse):
    from evaluation.retrieval import evaluate_retrieval_case, aggregate_agent_evidence, aggregate_retrieval_stages
    from evaluation.end_to_end import evaluate_sufficiency
    from service.core.evidence.assessment import finalize_assessment
    from domain.assessment import EvidenceAssessment

    def verify(context, evidence, **kwargs):
        sufficient = len(evidence) == 6 and not refuse
        return finalize_assessment(context, evidence, EvidenceAssessment(
            sufficient, "只采用最后一次找到的培训条款", source="rule",
            evaluated_chunk_ids=[item.id for item in evidence],
            supporting_chunk_indices=[6] if sufficient else [],
        ))

    monkeypatch.setattr(loop, "verify_evidence", verify)
    noise = [records(f"noise-{i}", text=f"无关信息{i}")[0] for i in range(5)]
    gold = records("gold")
    responses = iter([{"chunks": noise}, {"chunks": gold}])
    answer = AnswerClient()
    result = execute_agent_turn(
        index_names="test", question="员工培训要求", retrieval_options={"page_size": 5},
        planner_client=ModelClient([initial(), {"tasks": [task(id="s2", query="培训制度全文")]}, {"tasks": []}]),
        answer_client=answer, retrieval_function=lambda *args, **kwargs: next(responses),
    )
    assert result.raw_retrieval["evaluation_scope"] == "agent_evidence_sets"
    assert len(result.raw_retrieval["chunks_before_sufficiency"]) == 6
    assert [c["query"] for c in result.raw_retrieval["retrieval_calls"]] == ["员工培训要求", "培训制度全文"]
    assert [c["chunk_id"] for c in result.raw_retrieval["chunks"]] == ([] if refuse else ["gold"])
    if not refuse:
        assert "每年培训一次" in answer.calls[0]["messages"][-1]["content"]
        assert "无关信息" not in answer.calls[0]["messages"][-1]["content"]
    sample = {"id": "sixth-evidence", "answerable": True,
              "relevant_evidence": [{"document_name": "制度", "text": "员工培训要求：每年培训一次"}]}
    case = evaluate_retrieval_case(sample, result.raw_retrieval, latency_ms=1, top_k=5, match_threshold=1)
    assert case["retrieval_metrics"] is None and case["post_gate_metrics"] is None
    assert case["verification_evidence_metrics"]["recall"] == 1
    assert case["answer_evidence_metrics"]["recall"] == (0 if refuse else 1)
    decision = evaluate_sufficiency(sample, case)
    assert decision["expected_sufficient"] is True
    assert decision["unsafe_accept"] is False
    assert decision["false_rejection"] is refuse
    summary = aggregate_agent_evidence([case])
    assert summary["verification_evidence_metrics"]["recall"] == 1
    calls = summary["retrieval_call_metrics"]["by_top_k"]["5"]
    assert calls["query_count"] == 2 and calls["hit_at_5"] == 0.5
    gate = aggregate_retrieval_stages([case], top_k=5)["gate_metrics"]
    assert gate["evidence_sufficient_answerable_decision_count"] == 1
    assert gate["false_rejection_with_sufficient_evidence_count"] == int(refuse)
