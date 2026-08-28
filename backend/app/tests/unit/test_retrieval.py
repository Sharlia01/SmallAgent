import pytest

from service.core import retrieval
from service.core.rag.nlp import search_v2
from service.core.retrieval_evaluation import (
    build_summary,
    evidence_chunk_match_score,
    evaluate_retrieval_case,
)


# 用例功能：验证检索向量表达式始终使用 BGE 的 q_512_vec 字段。
# 执行步骤：
# 1. 将 Embedding 函数替换为返回 512 维向量的伪实现。
# 2. 调用 Dealer.get_vector 生成向量检索表达式。
# 3. 验证字段名、向量维度、Top K 和相似度阈值均正确。
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


# 用例功能：验证原始检索入口会将生产检索参数完整传给 Dealer。
# 执行步骤：
# 1. 用可记录调用参数的 FakeDealer 替换真实检索器。
# 2. 使用自定义页大小、相似度阈值和向量权重执行检索。
# 3. 验证问题、租户索引和各项 RetrievalOptions 传递正确。
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


# 用例功能：验证面向聊天的检索结果会保留排名、相似度和来源元数据。
# 执行步骤：
# 1. 用包含一条完整 chunk 元数据的伪结果替换底层检索。
# 2. 调用 retrieve_content 生成聊天使用的紧凑结果。
# 3. 验证文档路径已转为文件名，且排名、分数、位置和标识字段均被保留。
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


# 用例功能：验证证据匹配会归一化文本排版，并严格限定在同一文档内。
# 执行步骤：
# 1. 构造含空格和中文标点的标准证据。
# 2. 构造同文档中内容等价、排版不同的 chunk。
# 3. 再构造内容相同但文档名不同的 chunk。
# 4. 验证同文档得分为 1，不同文档得分为 0。
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


# 用例功能：验证多跳问题能正确计算 Hit@K、Recall@K 和 MRR@K。
# 执行步骤：
# 1. 构造包含两条必需证据的可回答多跳样本。
# 2. 构造两个排名顺序与证据顺序不同的检索 chunk。
# 3. 调用检索评测函数计算 Top 5 指标和证据匹配。
# 4. 验证命中、召回率、MRR、首条相关排名及匹配排名均正确。
# 5. 验证评测结果不会写入体积较大的向量数据。
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


# 用例功能：验证不可回答问题不会被计入正向检索指标，但会计算空结果率。
# 执行步骤：
# 1. 构造一个没有相关证据的不可回答样本。
# 2. 使用空检索结果执行单样本评测。
# 3. 验证 Hit@K、Recall@K 和 MRR@K 均为 None。
# 4. 生成汇总结果，验证可回答问题数为 0，不可回答空结果率为 1。
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
