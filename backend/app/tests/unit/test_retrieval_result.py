"""类型化检索结果在输出边界保留评测信息。"""

from tests.evidence_helpers import bind_assessment

import json

import numpy as np
import pytest

from domain.assessment import EvidenceAssessment
from service.core.evidence_adapter import collection_from_records
from service.core.query_contract import QueryGoal
from service.core.query_contract import QueryRequest
from service.core import retrieval
from service.core.retrieval_result import retrieval_record, retrieval_result
from tests.evidence_helpers import make_query


pytestmark = pytest.mark.unit


@pytest.mark.parametrize("sufficient", [True, False])
def test_output_preserves_candidates_query_and_ranking_when_gate_changes(sufficient):
    query = make_query("差旅如何审批？", (QueryGoal("approval", "差旅审批要求", source_spans=("审批",)),))
    candidates = collection_from_records([
        {"chunk_id": "approval", "docnm_kwd": "制度.pdf", "content_with_weight": "差旅申请需要审批。",
         "rerank_score": np.float32(0.8), "retrieval_ranks": {"keyword_original": 2}},
        {"chunk_id": "other", "content_with_weight": "其他规定。"},
    ])
    evidence = candidates.select(["approval"]) if sufficient else collection_from_records([])
    assessment = EvidenceAssessment(sufficient, "逐目标核验", [] if sufficient else ["审批要求"],
                                     [1], "rule", 2, ["approval", "other"])
    diagnostics = {"query_rewrite": {"original": query.question, "changed": False},
                   "retrieval_fusion": {"method": "weighted_rrf"},
                   "total_before_evidence_sufficiency": 37,
                   "doc_aggs": [{"doc_name": "制度.pdf", "count": 37}]}
    retrieved = retrieval_result(query, evidence, candidates, bind_assessment(query, candidates or evidence, assessment), diagnostics, total_candidates=37)
    exported = json.loads(json.dumps(retrieval_record(retrieved), ensure_ascii=False))

    assert retrieved.total_candidates == 37
    assert exported["total"] == (37 if sufficient else 0)
    assert exported["total_before_evidence_sufficiency"] == 37
    assert [item["chunk_id"] for item in exported["chunks_before_sufficiency"]] == ["approval", "other"]
    assert exported["chunks_before_sufficiency"][0]["rerank_score"] == pytest.approx(0.8)
    assert exported["chunks_before_sufficiency"][0]["retrieval_ranks"] == {"keyword_original": 2}
    assert exported["query_contract"] == query.request.to_dict()
    assert exported["query_intent"]["goals"][0]["id"] == "approval"
    assert exported["query_rewrite"] == diagnostics["query_rewrite"]
    assert exported["retrieval_fusion"] == diagnostics["retrieval_fusion"]
    assert bool(exported["doc_aggs"]) is sufficient
    assert exported["evidence_sufficiency"]["sufficient"] is sufficient
    assert "query_contract" not in retrieved.diagnostics


def test_explicit_request_is_preserved_by_the_unified_retrieval_entry(monkeypatch):
    request = QueryRequest.from_user("差旅如何审批？", [{"role": "user", "content": "请查看差旅制度"}])
    observed = []

    def execute(index, question, *, context, **options):
        observed.append(context.request)
        return retrieval_result(context, collection_from_records([]), collection_from_records([]), None, {})

    monkeypatch.setattr(retrieval, "_retrieve_evidence_result", execute)
    result = retrieval.retrieve_evidence("test-index", request.user_question, request=request)
    assert observed == [request]
    assert result.request is request
    assert retrieval_record(result)["query_contract"] == request.to_dict()
