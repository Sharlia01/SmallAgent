"""Merged evidence must be ranked for the original question, with hop coverage."""

from copy import deepcopy
import pytest

from service.core import retrieval
from service.core.query_intent import QuerySubqueryPlan
from service.core.sequential_retrieval import BridgeExtraction


pytestmark = pytest.mark.unit
QUESTION = "这份研报对甲公司是看多还是看空，给了什么评级？"


def make_chunk(identifier, content, *, score, fusion):
    return {
        "chunk_id": identifier,
        "doc_id": "report",
        "docnm_kwd": "甲公司.pdf",
        "content_with_weight": content,
        "rerank_score": score,
        "similarity": score,
        "rrf_score": 0.01,
        "final_ranking": {"final_fusion_score": fusion},
    }


@pytest.fixture
def hops():
    rating = make_chunk("rating", "维持甲公司“优于大市”评级。", score=0.85, fusion=0.08)
    return [
        (
            QuerySubqueryPlan(id="rating", query=QUESTION, output_slot="rating"),
            QUESTION,
            {"total": 20, "chunks": [
                make_chunk("list", "相关研究报告列表：甲公司历年研究报告。", score=0.59, fusion=0.081),
                rating,
                make_chunk("duplicate", "投资建议：甲公司维持优于大市评级。", score=0.84, fusion=0.077),
            ]},
        ),
        (
            QuerySubqueryPlan(id="definition", query="优于大市的评级定义", depends_on=["rating"]),
            "优于大市的评级定义",
            {"total": 10, "chunks": [
                make_chunk("definition", "优于大市：股价表现优于市场代表性指数10%以上。", score=0.999, fusion=0.091),
                # The same chunk has a different score for the second question.
                {**rating, "rerank_score": 0.3},
            ]},
        ),
    ]


def merge(hops, *, page_size=3, page=1, question=QUESTION):
    return retrieval._merge_sequential_hops(
        original_question=question,
        hop_results=hops,
        bridge=BridgeExtraction(
            slot="rating", value="优于大市", source_chunk_index=2,
            source_chunk_id="rating", document_id="report", source="rule",
        ),
        options=retrieval.Dealer.RetrievalOptions(page=page, page_size=page_size),
    )


def stub_scores(monkeypatch, hops, scores):
    by_content = {
        chunk["content_with_weight"]: scores[chunk["chunk_id"]]
        for _, _, result in hops for chunk in result["chunks"]
    }
    calls = []

    def rerank(question, documents):
        calls.append((question, documents))
        return [by_content[text] for text in documents], None

    monkeypatch.setattr(retrieval, "rerank_similarity", rerank)
    return calls


def test_original_question_rerank_removes_pinned_report_list(monkeypatch, hops):
    original_hops = deepcopy(hops)
    calls = stub_scores(monkeypatch, hops, {
        "list": 0.1, "rating": 0.95, "duplicate": 0.9, "definition": 0.7,
    })
    result = merge(hops)

    assert [c["chunk_id"] for c in result["chunks"]] == ["rating", "duplicate", "definition"]
    assert len(calls) == 1
    assert calls[0][0] == QUESTION
    assert len(calls[0][1]) == 4  # One request, deduplicated across both hops.
    rating = result["chunks"][0]
    assert rating["rerank_score"] == rating["similarity"] == 0.95
    assert [s["rerank_score"] for s in rating["sequential_subqueries"]] == [0.85, 0.3]
    assert rating["final_ranking"]["semantic_reranker_rank"] == 1
    assert result["retrieval_fusion"]["coverage_first"] is False
    assert set(result["retrieval_fusion"]["final_rerank"]["coverage_chunk_ids"]) == {"rating", "definition"}
    assert hops == original_hops


def test_low_scoring_dependency_is_retained_but_not_pinned_first(monkeypatch, hops):
    stub_scores(monkeypatch, hops, {
        "list": 0.3, "rating": 0.95, "duplicate": 0.9, "definition": 0.01,
    })
    result = merge(hops, page_size=2)
    assert [c["chunk_id"] for c in result["chunks"]] == ["rating", "definition"]
    assert result["chunks"][1]["rerank_score"] == 0.01


def test_second_hop_can_lead_when_more_relevant_to_original_question(monkeypatch, hops):
    stub_scores(monkeypatch, hops, {
        "list": 0.1, "rating": 0.7, "duplicate": 0.6, "definition": 0.95,
    })
    result = merge(hops, page_size=2)
    assert [c["chunk_id"] for c in result["chunks"]] == ["definition", "rating"]


def test_shared_top_result_does_not_crowd_out_unique_second_hop_evidence(monkeypatch, hops):
    hops[1][2]["chunks"].reverse()  # Both hops can return the direct rating first.
    stub_scores(monkeypatch, hops, {
        "list": 0.1, "rating": 0.95, "duplicate": 0.9, "definition": 0.3,
    })
    result = merge(hops, page_size=2)
    assert [c["chunk_id"] for c in result["chunks"]] == ["rating", "definition"]


def test_one_slot_uses_best_original_question_result(monkeypatch, hops):
    stub_scores(monkeypatch, hops, {
        "list": 0.1, "rating": 0.7, "duplicate": 0.6, "definition": 0.95,
    })
    result = merge(hops, page_size=1)
    assert [c["chunk_id"] for c in result["chunks"]] == ["definition"]
    assert result["retrieval_fusion"]["final_rerank"]["coverage_chunk_ids"] == []


def test_reserved_dependency_is_not_repeated_on_later_pages(monkeypatch, hops):
    stub_scores(monkeypatch, hops, {
        "list": 0.3, "rating": 0.95, "duplicate": 0.9, "definition": 0.01,
    })
    first = merge(hops, page_size=2, page=1)
    second = merge(hops, page_size=2, page=2)
    assert [c["chunk_id"] for c in first["chunks"] + second["chunks"]] == [
        "rating", "definition", "duplicate", "list",
    ]
    assert merge(hops, page_size=2, page=3)["chunks"] == []


def test_failure_preserves_both_hops_with_explicit_rank_only_fallback(monkeypatch, hops):
    def fail(*_args):
        raise TimeoutError("reranker timed out")

    monkeypatch.setattr(retrieval, "rerank_similarity", fail)
    result = merge(hops, page_size=2)
    assert {c["chunk_id"] for c in result["chunks"]} == {"rating", "definition"}
    diagnostics = result["retrieval_fusion"]["final_rerank"]
    assert diagnostics["source"] == "fallback"
    assert diagnostics["method"] == "hop_rank_rrf"
    assert "TimeoutError" in diagnostics["fallback_reason"]
    for candidate in result["chunks"]:
        assert candidate["rerank_score"] is None
        assert candidate["final_ranking"]["semantic_reranker_rank"] is None
        assert candidate["final_ranking"]["fallback_rrf_score"] > 0


def test_constraints_are_checked_against_original_question(monkeypatch, hops):
    hops[1][2]["chunks"][0]["content_with_weight"] = "2025年全年实际营业收入100亿元。"
    hops[0][2]["chunks"][1]["content_with_weight"] = "2025年上半年营业收入50亿元。"
    stub_scores(monkeypatch, hops, {
        "list": 0.1, "rating": 0.99, "duplicate": 0.2, "definition": 0.8,
    })
    result = merge(hops, page_size=2, question="2025年全年实际营业收入是多少？")
    assert result["chunks"][0]["chunk_id"] == "definition"
    assert result["chunks"][1]["constraint_compatibility"]["conflict_count"] > 0


def test_empty_hops_skip_reranker(monkeypatch):
    monkeypatch.setattr(retrieval, "rerank_similarity", lambda *_: pytest.fail("unexpected API call"))
    result = merge([])
    assert result["chunks"] == []
    assert result["retrieval_fusion"]["final_rerank"]["source"] == "skipped"
