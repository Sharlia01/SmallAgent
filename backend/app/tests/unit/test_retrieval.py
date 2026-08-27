import pytest

from service.core import retrieval
from service.core.rag.nlp import search_v2
from service.core.retrieval_evaluation import (
    build_summary,
    evidence_chunk_match_score,
    evaluate_retrieval_case,
)


@pytest.mark.unit
def test_get_vector_always_targets_the_bge_vector_field(monkeypatch):
    monkeypatch.setattr(
        search_v2,
        "generate_embedding",
        lambda _text: [0.0] * 512,
    )
    dealer = object.__new__(search_v2.Dealer)

    expression = dealer.get_vector("测试问题", topk=20, similarity=0.25)

    assert expression.vector_column_name == "q_512_vec"
    assert len(expression.embedding_data) == 512
    assert expression.topn == 20
    assert expression.extra_options == {"similarity": 0.25}


@pytest.mark.unit
def test_retrieve_raw_results_reuses_production_options(monkeypatch):
    calls = []

    class FakeDealer:
        def retrieval(self, **kwargs):
            calls.append(kwargs)
            return {"total": 0, "chunks": [], "doc_aggs": []}

    monkeypatch.setattr(retrieval, "get_retrieval_dealer", lambda: FakeDealer())

    result = retrieval.retrieve_raw_results(
        "42",
        "测试问题",
        page_size=8,
        similarity_threshold=0.25,
        vector_similarity_weight=0.7,
    )

    assert result["chunks"] == []
    assert calls[0]["question"] == "测试问题"
    assert calls[0]["tenant_ids"] == "42"
    options = calls[0]["options"]
    assert options.page == 1
    assert options.page_size == 8
    assert options.similarity_threshold == 0.25
    assert options.vector_similarity_weight == 0.7


@pytest.mark.unit
def test_retrieve_content_preserves_ranking_metadata(monkeypatch):
    monkeypatch.setattr(
        retrieval,
        "retrieve_raw_results",
        lambda *_args, **_kwargs: {
            "chunks": [
                {
                    "chunk_id": "chunk-1",
                    "doc_id": "doc-1",
                    "docnm_kwd": "/uploads/国电电力.pdf",
                    "content_with_weight": "证据内容",
                    "similarity": 0.91,
                    "vector_similarity": 0.88,
                    "term_similarity": 0.44,
                    "positions": [[5, 10]],
                    "kb_id": "42",
                    "image_id": "image-1",
                }
            ]
        },
    )

    result = retrieval.retrieve_content("42", "测试问题")

    assert result == [
        {
            "id": 1,
            "rank": 1,
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "document_name": "国电电力.pdf",
            "content_with_weight": "证据内容",
            "similarity": 0.91,
            "vector_similarity": 0.88,
            "term_similarity": 0.44,
            "positions": [[5, 10]],
            "kb_id": "42",
            "image_id": "image-1",
        }
    ]


@pytest.mark.unit
def test_evidence_match_requires_same_document_and_normalizes_layout():
    evidence = {
        "document_name": "国电电力.pdf",
        "text": "营业收入 776.55 亿元，同比下降 9.52%。",
    }
    matching_chunk = {
        "docnm_kwd": "/storage/国电电力.pdf",
        "content_with_weight": "2025年上半年营业收入776.55亿元,同比下降9.52%。",
    }
    wrong_document_chunk = {
        **matching_chunk,
        "docnm_kwd": "/storage/其他公司.pdf",
    }

    assert evidence_chunk_match_score(evidence, matching_chunk) == 1.0
    assert evidence_chunk_match_score(evidence, wrong_document_chunk) == 0.0


@pytest.mark.unit
def test_multi_hop_case_computes_hit_recall_and_mrr():
    sample = {
        "id": "case-1",
        "question": "综合问题",
        "reference_answer": "标准答案",
        "answerable": True,
        "question_type": "multi_hop",
        "relevant_evidence": [
            {"document_name": "国电电力.pdf", "page": 1, "text": "第一条证据"},
            {"document_name": "国电电力.pdf", "page": 2, "text": "第二条证据"},
        ],
        "metadata": {"source_modality": "text"},
    }
    raw_result = {
        "total": 2,
        "chunks": [
            {
                "chunk_id": "second",
                "doc_id": "doc",
                "docnm_kwd": "国电电力.pdf",
                "content_with_weight": "这里包含第二条证据。",
                "similarity": 0.9,
                "vector_similarity": 0.8,
                "term_similarity": 0.5,
            },
            {
                "chunk_id": "first",
                "doc_id": "doc",
                "docnm_kwd": "国电电力.pdf",
                "content_with_weight": "这里包含第一条证据。",
                "similarity": 0.8,
                "vector_similarity": 0.7,
                "term_similarity": 0.4,
            },
        ],
    }

    result = evaluate_retrieval_case(
        sample,
        raw_result,
        latency_ms=12.5,
        top_k=5,
        match_threshold=0.8,
    )

    assert result["metrics"] == {
        "hit_at_5": True,
        "recall_at_5": 1.0,
        "mrr_at_5": 1.0,
    }
    assert result["matched_evidence_count"] == 2
    assert result["first_relevant_rank"] == 1
    assert [match["matched_rank"] for match in result["evidence_matches"]] == [2, 1]
    assert "vector" not in result["retrieval"]["chunks"][0]


@pytest.mark.unit
def test_unanswerable_case_is_excluded_from_positive_retrieval_metrics():
    sample = {
        "id": "case-2",
        "question": "文档没有答案的问题",
        "reference_answer": "文档没有说明。",
        "answerable": False,
        "question_type": "unanswerable",
        "relevant_evidence": [],
        "metadata": {},
    }
    result = evaluate_retrieval_case(
        sample,
        {"total": 0, "chunks": []},
        latency_ms=2,
        top_k=5,
        match_threshold=0.8,
    )

    assert result["metrics"] == {
        "hit_at_5": None,
        "recall_at_5": None,
        "mrr_at_5": None,
    }
    summary = build_summary([result], top_k=5)
    assert summary["overall"]["answerable_query_count"] == 0
    assert summary["overall"]["unanswerable_empty_rate"] == 1.0
