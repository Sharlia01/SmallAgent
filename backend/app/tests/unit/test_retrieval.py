from service.core.evidence.adapter import reference_records
from service.core.evidence.adapter import collection_from_records
import pytest

from domain.assessment import EvidenceAssessment
from service.core.rag.nlp import search_v2
from service.core.rag.nlp.search_v2 import reciprocal_rank_fusion
from evaluation.retrieval import (
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


@pytest.mark.unit
def test_dealer_retrieval_times_initial_retrieval_and_rerank(monkeypatch):
    dealer = object.__new__(search_v2.Dealer)
    search_result = object()
    ranked_results = object()
    clock = iter([1.0, 1.012, 2.0, 2.034])

    monkeypatch.setattr(search_v2.time, "perf_counter", lambda: next(clock))
    monkeypatch.setattr(dealer, "_build_search_request", lambda *_args: {})
    monkeypatch.setattr(dealer, "_build_index_names", lambda _ids: ["42"])
    monkeypatch.setattr(dealer, "search", lambda *_args, **_kwargs: search_result)
    monkeypatch.setattr(
        dealer,
        "_rerank_results",
        lambda *_args, **_kwargs: ranked_results,
    )
    monkeypatch.setattr(
        dealer,
        "_build_retrieval_result",
        lambda *_args, **_kwargs: {"total": 0, "chunks": []},
    )

    result = dealer.retrieval(
        "问题",
        "42",
        search_v2.Dealer.RetrievalOptions(),
    )

    assert result["stage_latency_ms"] == {
        "initial_retrieval": pytest.approx(12.0),
        "rerank": pytest.approx(34.0),
    }


# 用例功能：验证类型化检索入口会将生产检索参数完整传给 Dealer。
# 执行步骤：
# 1. 用可记录调用参数的 FakeDealer 替换真实检索器。
# 2. 使用自定义页大小、相似度阈值和向量权重执行检索。
# 3. 验证问题、租户索引和各项 RetrievalOptions 传递正确。


















# 用例功能：验证 RRF 只使用各路排名，并让多路共同命中的 chunk 优先。
@pytest.mark.unit
def test_reciprocal_rank_fusion_merges_and_deduplicates_rankings():
    result = reciprocal_rank_fusion(
        {
            "keyword_original": ["a", "b", "c"],
            "keyword_expanded": ["b", "c", "a"],
            "vector_original": ["c", "b", "d"],
        },
        {
            "keyword_original": 1.0,
            "keyword_expanded": 1.0,
            "vector_original": 1.0,
        },
        k=60,
    )

    assert result.ids == ["b", "c", "a", "d"]
    assert result.ranks["b"] == {
        "keyword_original": 2,
        "keyword_expanded": 1,
        "vector_original": 2,
    }
    assert result.scores["b"] == pytest.approx(
        1 / 62 + 1 / 61 + 1 / 62
    )


# 用例功能：验证原词、扩展词和原问题向量分别执行独立召回。
@pytest.mark.unit
def test_search_runs_three_independent_branches_and_rrf_fusion():
    query_calls = []
    vector_calls = []
    search_calls = []
    original = "国电电力未来三年能赚多少钱，估值贵不贵？"
    expanded = "国电电力 盈利预测 归母净利润 EPS PE 估值"

    responses = {
        f"text::{original}": {
            "ids": ["a", "shared"],
            "scores": [10.0, 9.0],
        },
        f"text::{expanded}": {
            "ids": ["shared", "b"],
            "scores": [11.0, 8.0],
        },
        "vector": {
            "ids": ["b", "shared", "a"],
            "scores": [0.95, 0.90, 0.85],
        },
    }

    class FakeQueryer:
        def question(self, question, min_match):
            query_calls.append((question, min_match))
            return f"text::{question}", [question]

    class FakeDataStore:
        @staticmethod
        def getTotal(response):
            return len(response["ids"])

        @staticmethod
        def getChunkIds(response):
            return response["ids"]

        @staticmethod
        def getFields(response, _source_fields):
            return {
                chunk_id: {
                    "content_ltks": f"{chunk_id} tokens",
                    "content_with_weight": f"{chunk_id} content",
                    "docnm_kwd": "doc.pdf",
                    "doc_id": "doc",
                    "kb_id": "42",
                    "_score": score,
                }
                for chunk_id, score in zip(
                    response["ids"],
                    response["scores"],
                )
            }

    dealer = object.__new__(search_v2.Dealer)
    dealer.qryr = FakeQueryer()
    dealer.dataStore = FakeDataStore()
    dealer._expand_keywords = lambda keywords: set(keywords)

    dense_expression = type(
        "DenseExpression",
        (),
        {"embedding_data": [0.1, 0.2]},
    )()

    def fake_get_vector(question, topk, similarity):
        vector_calls.append((question, topk, similarity))
        return dense_expression

    def fake_search(
        _context,
        _idx_names,
        _kb_ids,
        _highlight_fields,
        expressions,
        rank_feature,
    ):
        assert len(expressions) == 1
        expression = expressions[0]
        branch = expression if isinstance(expression, str) else "vector"
        search_calls.append((branch, rank_feature))
        return responses[branch]

    dealer.get_vector = fake_get_vector
    dealer._run_hybrid_search = fake_search
    context = search_v2.Dealer.SearchContext(
        filters={"available_int": 1},
        order_by=None,
        offset=0,
        limit=100,
        source_fields=["content_ltks", "content_with_weight", "kb_id"],
        topk=20,
    )

    result = dealer._search_with_question(
        {
            "keyword_question": expanded,
            "similarity": 0.25,
            "vector_similarity_weight": 0.6,
            "rrf_k": 60,
            "rerank_candidate_size": 50,
        },
        context,
        original,
        "42",
        None,
        False,
        {"pagerank_fea": 10},
    )

    assert query_calls == [(original, 0.3), (expanded, 0.3)]
    assert vector_calls == [(original, 20, 0.25)]
    assert search_calls == [
        (f"text::{original}", {"pagerank_fea": 10}),
        (f"text::{expanded}", {"pagerank_fea": 10}),
        ("vector", None),
    ]
    assert result.candidate_ids == ["shared", "b", "a"]
    assert result.diagnostics.branch_weights == pytest.approx(
        {
            "keyword_original": 0.2,
            "keyword_expanded": 0.2,
            "vector_original": 0.6,
        }
    )
    assert result.diagnostics.branch_ranks["shared"] == {
        "keyword_original": 2,
        "keyword_expanded": 1,
        "vector_original": 2,
    }


@pytest.mark.unit
def test_search_skips_duplicate_expanded_keyword_branch(monkeypatch):
    dealer = object.__new__(search_v2.Dealer)
    text_queries = []
    dealer._search_text_branch = lambda **kwargs: (
        text_queries.append(kwargs["text"])
        or search_v2.Dealer.SearchBranchResult(
            name=kwargs["name"],
            response={},
            total=0,
            ids=[],
            fields={},
            scores={},
        )
    )
    dealer._search_vector_branch = lambda **_kwargs: (
        search_v2.Dealer.SearchBranchResult(
            name="vector_original",
            response={},
            total=0,
            ids=[],
            fields={},
            scores={},
        )
    )
    dealer._merge_search_branches = lambda branches, weights, **_kwargs: (
        (branches, weights)
    )
    context = search_v2.Dealer.SearchContext(
        filters={},
        order_by=None,
        offset=0,
        limit=100,
        source_fields=[],
        topk=20,
    )

    branches, weights = dealer._search_with_question(
        {
            "keyword_question": "  同一个问题  ",
            "vector_similarity_weight": 0.6,
        },
        context,
        "同一个问题",
        "42",
        None,
        False,
        None,
    )

    assert text_queries == ["同一个问题"]
    assert [branch.name for branch in branches] == [
        "keyword_original",
        "vector_original",
    ]
    assert weights == pytest.approx(
        {"keyword_original": 0.4, "vector_original": 0.6}
    )


@pytest.mark.unit
def test_empty_expanded_branch_returns_its_weight_to_original_keyword():
    branches = [
        search_v2.Dealer.SearchBranchResult(
            name="keyword_original",
            response={},
            total=1,
            ids=["a"],
            fields={},
            scores={},
        ),
        search_v2.Dealer.SearchBranchResult(
            name="keyword_expanded",
            response={},
            total=0,
            ids=[],
            fields={},
            scores={},
        ),
        search_v2.Dealer.SearchBranchResult(
            name="vector_original",
            response={},
            total=1,
            ids=["b"],
            fields={},
            scores={},
        ),
    ]

    weights = search_v2.Dealer._redistribute_empty_branch_weights(
        branches,
        {
            "keyword_original": 0.2,
            "keyword_expanded": 0.2,
            "vector_original": 0.6,
        },
    )

    assert weights == pytest.approx(
        {"keyword_original": 0.4, "vector_original": 0.6}
    )


# 用例功能：验证没有硬限定时仍按原问题语义分和 RRF 排序。
@pytest.mark.unit
def test_rerank_preserves_semantic_order_without_constraints(monkeypatch):
    semantic_calls = []
    dealer = object.__new__(search_v2.Dealer)
    search_result = search_v2.Dealer.SearchResult(
        total=2,
        candidate_ids=["chunk-2", "chunk-1"],
        candidate_fields={
            "chunk-1": {"content_with_weight": "第一条原始内容"},
            "chunk-2": {"content_with_weight": "第二条原始内容"},
        },
        recall_fusion_scores={"chunk-1": 0.03, "chunk-2": 0.04},
    )
    monkeypatch.setattr(
        search_v2,
        "rerank_similarity",
        lambda query, documents: (
            semantic_calls.append((query, documents))
            or search_v2.np.array([0.8, 0.8]),
            None,
        ),
    )

    indexes, scores = dealer._rerank_results(
        search_result,
        "用户原始问题",
        search_v2.Dealer.RetrievalOptions(page_size=2),
    )

    assert semantic_calls == [
        ("用户原始问题", ["第二条原始内容", "第一条原始内容"])
    ]
    assert scores.tolist() == [0.8, 0.8]
    assert indexes == [0, 1]
    assert search_result.final_fusion_scores["chunk-2"] == pytest.approx(1 / 11)


@pytest.mark.unit
def test_table_reranker_prefers_semantic_text_over_llm_markdown(monkeypatch):
    captured = []
    dealer = object.__new__(search_v2.Dealer)
    search_result = search_v2.Dealer.SearchResult(
        total=1,
        candidate_ids=["table-row"],
        candidate_fields={
            "table-row": {
                "content_with_weight": "| 公司 | PE |\n| --- | --- |\n| 龙源电力 | 19.6 |",
                "table_semantic_text_with_weight": "公司：龙源电力；PE：19.6",
            },
        },
        recall_fusion_scores={"table-row": 0.03},
    )
    monkeypatch.setattr(
        search_v2,
        "rerank_similarity",
        lambda query, documents: (
            captured.append((query, documents)) or search_v2.np.array([0.9]),
            None,
        ),
    )

    dealer._rerank_results(
        search_result,
        "龙源电力估值",
        search_v2.Dealer.RetrievalOptions(page_size=1),
    )

    assert captured == [("龙源电力估值", ["公司：龙源电力；PE：19.6"])]


@pytest.mark.unit
def test_second_stage_fusion_uses_configured_reranker_weight(monkeypatch):
    dealer = object.__new__(search_v2.Dealer)
    search_result = search_v2.Dealer.SearchResult(
        total=2,
        candidate_ids=["retrieval-first", "semantic-first"],
        candidate_fields={
            "retrieval-first": {"content_with_weight": "候选一"},
            "semantic-first": {"content_with_weight": "候选二"},
        },
        recall_fusion_scores={
            "retrieval-first": 0.03,
            "semantic-first": 0.02,
        },
    )
    monkeypatch.setattr(
        search_v2,
        "rerank_similarity",
        lambda _query, _documents: (
            search_v2.np.array([0.7, 0.9]),
            None,
        ),
    )

    indexes, _scores = dealer._rerank_results(
        search_result,
        "普通查询",
        search_v2.Dealer.RetrievalOptions(
            page_size=2,
            final_reranker_weight=0.7,
            final_rrf_k=10,
        ),
    )

    assert indexes == [1, 0]
    assert search_result.final_fusion_scores["semantic-first"] == pytest.approx(
        0.7 / 11 + 0.3 / 12
    )

    retrieval_weighted_indexes, _scores = dealer._rerank_results(
        search_result,
        "普通查询",
        search_v2.Dealer.RetrievalOptions(
            page_size=2,
            final_reranker_weight=0.3,
            final_rrf_k=10,
        ),
    )

    assert retrieval_weighted_indexes == [0, 1]
    assert search_result.final_fusion_scores["retrieval-first"] == pytest.approx(
        0.3 / 12 + 0.7 / 11
    )


# 用例功能：验证明确的时间冲突会覆盖极小的语义分差，避免季度问题被半年片段抢占。
@pytest.mark.unit
def test_rerank_demotes_explicit_constraint_conflicts(monkeypatch):
    dealer = object.__new__(search_v2.Dealer)
    search_result = search_v2.Dealer.SearchResult(
        total=2,
        candidate_ids=["half-year", "second-quarter"],
        candidate_fields={
            "half-year": {
                "content_with_weight": (
                    "2025年上半年，公司实现营业收入776.55亿元；"
                    "归母净利润36.87亿元。"
                ),
                "docnm_kwd": "国电电力.pdf",
            },
            "second-quarter": {
                "content_with_weight": (
                    "2025年上半年经营情况。2025年第二季度，公司实现"
                    "收入378.42亿元；归母净利润18.76亿元。"
                ),
                "docnm_kwd": "国电电力.pdf",
            },
        },
        recall_fusion_scores={
            "half-year": 0.015543,
            "second-quarter": 0.015008,
        },
    )
    monkeypatch.setattr(
        search_v2,
        "rerank_similarity",
        lambda _query, _documents: (
            search_v2.np.array([0.984671, 0.982415]),
            None,
        ),
    )

    indexes, _scores = dealer._rerank_results(
        search_result,
        "国电电力2025年第二季度的收入及两项净利润指标表现如何？",
        search_v2.Dealer.RetrievalOptions(page_size=2),
    )

    assert indexes == [1, 0]
    half_year = search_result.constraint_compatibility["half-year"]
    second_quarter = search_result.constraint_compatibility[
        "second-quarter"
    ]
    assert half_year["categories"]["time"]["level"] == "conflict"
    assert (
        second_quarter["categories"]["time"]["level"]
        == "compatible"
    )


@pytest.mark.unit
def test_constraint_rerank_selects_highest_ranked_candidate(monkeypatch):
    dealer = object.__new__(search_v2.Dealer)
    search_result = search_v2.Dealer.SearchResult(
        total=2,
        candidate_ids=["compatible-low-score", "unknown-high-score"],
        candidate_fields={
            "compatible-low-score": {
                "content_with_weight": "2025年第二季度经营情况。",
            },
            "unknown-high-score": {
                "content_with_weight": "经营情况。",
            },
        },
    )
    monkeypatch.setattr(
        search_v2,
        "rerank_similarity",
        lambda _query, _documents: (
            search_v2.np.array([0.05, 0.8]),
            None,
        ),
    )

    indexes, _scores = dealer._rerank_results(
        search_result,
        "2025年第二季度经营情况如何？",
        search_v2.Dealer.RetrievalOptions(
            page_size=1,
            similarity_threshold=0.1,
        ),
    )

    assert indexes == [1]


@pytest.mark.unit
def test_retrieval_result_preserves_rrf_and_branch_diagnostics():
    dealer = object.__new__(search_v2.Dealer)
    search_result = search_v2.Dealer.SearchResult(
        total=3,
        candidate_ids=["chunk-1"],
        candidate_fields={
            "chunk-1": {
                "content_ltks": "盈利 预测",
                "content_with_weight": "盈利预测证据",
                "doc_id": "doc-1",
                "docnm_kwd": "doc.pdf",
                "kb_id": "42",
            }
        },
        recall_fusion_scores={"chunk-1": 0.031},
        diagnostics=search_v2.SearchDiagnostics(
            branch_ranks={
                "chunk-1": {
                    "keyword_original": 2,
                    "keyword_expanded": 1,
                    "vector_original": 4,
                }
            },
            branch_scores={
                "chunk-1": {
                    "keyword_original": 8.0,
                    "keyword_expanded": 12.0,
                    "vector_original": 0.88,
                }
            },
            branch_hit_counts={
                "keyword_original": 10,
                "keyword_expanded": 8,
                "vector_original": 12,
            },
            branch_weights={
                "keyword_original": 0.2,
                "keyword_expanded": 0.2,
                "vector_original": 0.6,
            },
        ),
    )

    result = dealer._build_retrieval_result(
        search_result,
        ([0], search_v2.np.array([0.93])),
        search_v2.Dealer.RetrievalOptions(page_size=1, rrf_k=60),
    )

    chunk = result["chunks"][0]
    assert chunk["similarity"] == pytest.approx(0.93)
    assert chunk["rerank_score"] == pytest.approx(0.93)
    assert chunk["rrf_score"] == pytest.approx(0.031)
    assert "rrf_contributions" not in chunk
    assert chunk["term_similarity"] == pytest.approx(12.0)
    assert chunk["vector_similarity"] == pytest.approx(0.88)
    assert "vector" not in chunk
    assert chunk["retrieval_ranks"]["keyword_expanded"] == 1
    assert chunk["constraint_compatibility"] == {}
    assert chunk["final_ranking"] == {}
    assert result["retrieval_fusion"] == {
        "method": "weighted_rrf",
        "rrf_k": 60,
        "branch_weights": search_result.diagnostics.branch_weights,
        "branch_totals": search_result.diagnostics.branch_hit_counts,
        "fused_candidate_count": 3,
        "rerank_candidate_count": 1,
        "rerank_query": "original_question",
        "final_fusion": {
            "method": "weighted_rrf",
            "rrf_k": 10,
                "weights": {
                    "semantic_reranker": 0.7,
                    "retrieval_rrf": pytest.approx(0.3),
            },
            "constraint_conflicts_first": True,
        },
    }


# 用例功能：验证面向聊天的检索结果会保留排名、相似度和来源元数据。
# 执行步骤：
# 1. 用包含一条完整 chunk 元数据的类型化结果替换底层检索。
# 2. 调用 retrieve_evidence，再从证据集合导出前端记录。
# 3. 验证文档路径已转为文件名，且排名、分数、位置和标识字段均被保留。


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
        "query_transform": {
            "original": "综合问题",
            "rewritten": "综合问题",
            "expanded": "",
            "effective_query": "综合问题",
        },
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

    assert result["post_gate_metrics"]["hit_at_5"] is True
    assert result["post_gate_metrics"]["recall_at_5"] == 1.0
    assert result["post_gate_metrics"]["precision_at_5"] == 0.4
    assert result["post_gate_metrics"]["mrr_at_5"] == 1.0
    assert result["matched_evidence_count"] == 2
    assert result["first_relevant_rank"] == 1
    assert [match["matched_rank"] for match in result["evidence_matches"]] == [2, 1]
    assert "vector" not in result["retrieval"]["chunks"][0]
    assert result["retrieval"]["query_transform"]["effective_query"] == (
        "综合问题"
    )

# 用例功能：验证同一个证据要求可以由多个检索 chunk 联合满足。
# 执行步骤：
# 1. 使用 guodian-016 的真实盈利预测和估值数据构造评测样本。
# 2. 将净利润放在排名第 2 的 chunk 中。
# 3. 将 EPS 和 PE 放在排名第 5 的 chunk 中。
# 4. 验证三个原子证据联合后，完整证据在第 5 名形成。
# 5. 验证 Hit@5、Recall@5 和 MRR@5 按完整证据排名计算。
@pytest.mark.unit
def test_multi_chunk_requirement_uses_completion_rank():
    sample = {
        "id": "guodian-016",
        "question": "研报觉得国电电力未来三年能赚多少钱，对应估值贵不贵？",
        "reference_answer": (
            "2025—2027年归母净利润分别为70.50、78.95和87.17亿元，"
            "EPS分别为0.40、0.44和0.49元，PE分别为11.4、10.2和9.2倍。"
        ),
        "answerable": True,
        "question_type": "paraphrase",
        # 暂时保留旧字段，保证评测集结构向后兼容。
        "relevant_evidence": [
            {
                "document_name": "国电电力.pdf",
                "page": 1,
                "text": (
                    "预计2025-2027年公司归母净利润分别为"
                    "70.50/78.95/87.17亿元；EPS分别为"
                    "0.40/0.44/0.49元，当前股价对应PE为"
                    "11.4/10.2/9.2x。"
                ),
            }
        ],
        # 新结构：外层 alternatives 是 OR，内层列表是 AND。
        "evidence_requirements": [
            {
                "id": "profit_and_valuation",
                "alternatives": [
                    [
                        {
                            "type": "text",
                            "document_name": "国电电力.pdf",
                            "text": (
                                "归属于母公司净利润 "
                                "5609 9831 7050 7895 8717"
                            ),
                        },
                        {
                            "type": "text",
                            "document_name": "国电电力.pdf",
                            "text": (
                                "每股收益 "
                                "0.31 0.55 0.40 0.44 0.49"
                            ),
                        },
                        {
                            "type": "text",
                            "document_name": "国电电力.pdf",
                            "text": (
                                "P/E 14.3 8.2 11.4 10.2 9.2"
                            ),
                        },
                    ]
                ],
            }
        ],
        "metadata": {"source_modality": "text"},
    }

    raw_result = {
        "total": 5,
        "chunks": [
            {
                "chunk_id": "overview",
                "docnm_kwd": "国电电力.pdf",
                "content_with_weight": "2025年上半年经营情况。",
            },
            {
                "chunk_id": "profit-table",
                "docnm_kwd": "国电电力.pdf",
                "content_with_weight": (
                    "归属于母公司净利润 "
                    "5609 9831 7050 7895 8717"
                ),
            },
            {
                "chunk_id": "disclosure",
                "docnm_kwd": "国电电力.pdf",
                "content_with_weight": "证券研究报告相关声明。",
            },
            {
                "chunk_id": "rating",
                "docnm_kwd": "国电电力.pdf",
                "content_with_weight": "投资评级：优于大市。",
            },
            {
                "chunk_id": "valuation-table",
                "docnm_kwd": "国电电力.pdf",
                "content_with_weight": (
                    "每股收益 0.31 0.55 0.40 0.44 0.49；"
                    "P/E 14.3 8.2 11.4 10.2 9.2"
                ),
            },
        ],
    }

    result = evaluate_retrieval_case(
        sample,
        raw_result,
        latency_ms=10,
        top_k=5,
        match_threshold=0.8,
        metric_ks=(3, 5),
    )

    assert result["post_gate_metrics"]["hit_at_5"] is True
    assert result["post_gate_metrics"]["recall_at_5"] == 1.0
    assert result["post_gate_metrics"]["mrr_at_5"] == pytest.approx(0.2)
    assert result["post_gate_metrics"]["all_requirements_hit_at_3"] is False
    assert result["post_gate_metrics"]["all_requirements_hit_at_5"] is True
    assert result["first_relevant_rank"] == 5

# 用例功能：验证评测器能够按表格标题和完整数据行匹配 HTML 表格证据。
# 执行步骤：
# 1. 使用国电电力利润预测表中的真实财务费用数据构造标准证据。
# 2. 构造包含表头、目标行和干扰行的 HTML 表格 chunk。
# 3. 要求评测器匹配“财务预测与估值”表中的完整财务费用行。
# 4. 验证该表格证据在排名第 1 的 chunk 中命中。
@pytest.mark.unit
def test_table_row_requirement_matches_html_table():
    sample = {
        "id": "guodian-022",
        "question": "利润预测表中，国电电力2026E财务费用是多少？",
        "reference_answer": "2026E财务费用为10049百万元。",
        "answerable": True,
        "question_type": "fact",
        "relevant_evidence": [],
        "evidence_requirements": [
            {
                "id": "financial_expense_row",
                "alternatives": [
                    [
                        {
                            "type": "table_row",
                            "document_name": "国电电力.pdf",
                            "caption": "财务预测与估值",
                            "row_cells": [
                                "财务费用",
                                "6711",
                                "6551",
                                "8389",
                                "10049",
                                "10216",
                            ],
                        }
                    ]
                ],
            }
        ],
        "metadata": {"source_modality": "table"},
    }

    raw_result = {
        "total": 1,
        "chunks": [
            {
                "chunk_id": "profit-table",
                "docnm_kwd": "国电电力.pdf",
                "content_with_weight": """
                    <table>
                      <caption>财务预测与估值</caption>
                      <tr>
                        <th>利润表（百万元）</th>
                        <th>2023</th>
                        <th>2024</th>
                        <th>2025E</th>
                        <th>2026E</th>
                        <th>2027E</th>
                      </tr>
                      <tr>
                        <td>研发费用</td>
                        <td>741</td>
                        <td>555</td>
                        <td>548</td>
                        <td>568</td>
                        <td>570</td>
                      </tr>
                      <tr>
                        <td>财务费用</td>
                        <td>6711</td>
                        <td>6551</td>
                        <td>8389</td>
                        <td>10049</td>
                        <td>10216</td>
                      </tr>
                    </table>
                """,
            }
        ],
    }

    result = evaluate_retrieval_case(
        sample,
        raw_result,
        latency_ms=10,
        top_k=5,
        match_threshold=0.8,
    )

    assert result["post_gate_metrics"]["hit_at_5"] is True
    assert result["post_gate_metrics"]["recall_at_5"] == 1.0
    assert result["post_gate_metrics"]["mrr_at_5"] == 1.0
    assert result["first_relevant_rank"] == 1

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

    assert result["post_gate_metrics"]["hit_at_5"] is None
    assert result["post_gate_metrics"]["recall_at_5"] is None
    assert result["post_gate_metrics"]["precision_at_5"] is None
    assert result["post_gate_metrics"]["mrr_at_5"] is None
    summary = build_summary([result], top_k=5)
    assert summary["overall"]["answerable_query_count"] == 0
    post_gate = summary["overall"]["post_gate_metrics"]
    assert post_gate["unanswerable_empty_rate"] == 1.0
    assert post_gate["answerable_empty_rate"] is None
    assert summary["overall"]["gate_metrics"][
        "unanswerable_rejection_rate"
    ] is None
    assert summary["overall"]["gate_metrics"]["fallback_rate"] is None


@pytest.mark.unit
def test_relevant_table_evidence_uses_table_parser():
    sample = {
        "id": "guodian-022",
        "question": "利润预测表中，国电电力2026E财务费用是多少？",
        "reference_answer": "2026E财务费用为10049百万元。",
        "answerable": True,
        "relevant_evidence": [
            {
                "document_name": "国电电力.pdf",
                "page": 5,
                "evidence_type": "table",
                "locator": (
                    "财务预测与估值｜利润表（百万元）"
                    "｜财务费用行｜2026E列"
                ),
                "text": "财务费用 6711 6551 8389 10049 10216",
            }
        ],
    }

    raw_result = {
        "total": 1,
        "chunks": [
            {
                "chunk_id": "profit-table",
                "docnm_kwd": "国电电力.pdf",
                "content_with_weight": """
                    <table>
                      <tr>
                        <th>项目</th>
                        <th>2023</th>
                        <th>2024</th>
                        <th>2025E</th>
                        <th>2026E</th>
                        <th>2027E</th>
                      </tr>
                      <tr>
                        <td>财务费用</td>
                        <td>6711</td>
                        <td>6551</td>
                        <td>8389</td>
                        <td>10049</td>
                        <td>10216</td>
                      </tr>
                    </table>
                """,
            }
        ],
    }

    result = evaluate_retrieval_case(
        sample,
        raw_result,
        latency_ms=10,
        top_k=5,
        match_threshold=0.8,
    )

    assert result["post_gate_metrics"]["hit_at_5"] is True
    assert result["post_gate_metrics"]["recall_at_5"] == 1.0
    assert result["post_gate_metrics"]["mrr_at_5"] == 1.0
    assert result["first_relevant_rank"] == 1


@pytest.mark.unit
@pytest.mark.parametrize(
    ("sample_id", "question", "locator", "evidence_text", "figure_content"),
    [
        (
            "guodian-027",
            "根据图3，在2020至2024年中，国电电力哪一年的归母净利润为负？",
            "图3：国电电力归母净利润及增速（单位：亿元）",
            "图3中，2021年的归母净利润柱位于零轴下方。",
            "图3：国电电力归母净利润及增速（单位：亿元）\n"
            "归母净利润\n同比增速\n2020 2021 2022 2023 2024",
        ),
        (
            "guodian-028",
            "根据图2，2025年的单季营业收入柱展示了哪两个季度？",
            "图2：国电电力单季营业收入（单位：亿元）",
            "图2中，2025年仅在Q1和Q2显示了营业收入柱。",
            "图2：国电电力单季营业收入（单位：亿元）\n"
            "2023 2024 2025\nQ1 Q2 Q3 Q4",
        ),
    ],
)
def test_relevant_figure_evidence_uses_locator(
    sample_id,
    question,
    locator,
    evidence_text,
    figure_content,
):
    sample = {
        "id": sample_id,
        "question": question,
        "reference_answer": "标准答案",
        "answerable": True,
        "relevant_evidence": [
            {
                "document_name": "国电电力.pdf",
                "page": 2,
                "evidence_type": "figure",
                "locator": locator,
                "text": evidence_text,
            }
        ],
    }
    raw_result = {
        "total": 1,
        "chunks": [
            {
                "chunk_id": f"{sample_id}-figure",
                "docnm_kwd": "国电电力.pdf",
                "content_with_weight": figure_content,
            }
        ],
    }

    result = evaluate_retrieval_case(
        sample,
        raw_result,
        latency_ms=10,
        top_k=5,
        match_threshold=0.8,
    )

    assert result["post_gate_metrics"]["hit_at_5"] is True
    assert result["post_gate_metrics"]["recall_at_5"] == 1.0
    assert result["post_gate_metrics"]["mrr_at_5"] == 1.0
    assert result["first_relevant_rank"] == 1


@pytest.mark.unit
def test_relevant_figure_evidence_rejects_a_different_figure():
    sample = {
        "id": "figure-mismatch",
        "question": "图3展示了什么？",
        "reference_answer": "标准答案",
        "answerable": True,
        "relevant_evidence": [
            {
                "document_name": "国电电力.pdf",
                "page": 2,
                "evidence_type": "figure",
                "locator": "图3：国电电力归母净利润及增速（单位：亿元）",
                "text": "图3中的视觉证据。",
            }
        ],
    }
    raw_result = {
        "total": 1,
        "chunks": [
            {
                "chunk_id": "wrong-figure",
                "docnm_kwd": "国电电力.pdf",
                "content_with_weight": (
                    "图2：国电电力单季营业收入（单位：亿元）\n"
                    "2023 2024 2025\nQ1 Q2 Q3 Q4"
                ),
            }
        ],
    }

    result = evaluate_retrieval_case(
        sample,
        raw_result,
        latency_ms=10,
        top_k=5,
        match_threshold=0.8,
    )

    assert result["post_gate_metrics"]["hit_at_5"] is False
    assert result["post_gate_metrics"]["recall_at_5"] == 0.0
    assert result["post_gate_metrics"]["mrr_at_5"] == 0.0
    assert result["first_relevant_rank"] is None


@pytest.mark.unit
def test_relevant_table_evidence_falls_back_to_flattened_text():
    sample = {
        "id": "guodian-021",
        "question": "财务预测表中，国电电力2026E固定资产是多少？",
        "reference_answer": "2026E固定资产为467602百万元。",
        "answerable": True,
        "relevant_evidence": [
            {
                "document_name": "国电电力.pdf",
                "page": 5,
                "evidence_type": "table",
                "locator": (
                    "财务预测与估值｜资产负债表（百万元）"
                    "｜固定资产行｜2026E列"
                ),
                "text": "固定资产 360117 383684 427026 467602 453840",
            }
        ],
    }
    raw_result = {
        "total": 1,
        "chunks": [
            {
                "chunk_id": "balance-sheet-continuation",
                "docnm_kwd": "国电电力.pdf",
                "content_with_weight": (
                    "74958 固定资产360117 383684 427026 467602 453840 "
                    "10122 7034 6682 6330 5979 无形资产及其他"
                ),
            }
        ],
    }

    result = evaluate_retrieval_case(
        sample,
        raw_result,
        latency_ms=10,
        top_k=5,
        match_threshold=0.8,
    )

    assert result["post_gate_metrics"]["hit_at_5"] is True
    assert result["post_gate_metrics"]["recall_at_5"] == 1.0
    assert result["post_gate_metrics"]["mrr_at_5"] == 1.0
    assert result["first_relevant_rank"] == 1


@pytest.mark.unit
@pytest.mark.parametrize("recall_scores", [{"z": 0.03, "a": 0.03}, {}])
def test_final_fusion_preserves_retrieval_order_for_ties_and_plain_search(monkeypatch, recall_scores):
    result = search_v2.Dealer.SearchResult(
        total=2, candidate_ids=["z", "a"],
        candidate_fields={"z": {"content_with_weight": "正文甲"},
                          "a": {"content_with_weight": "正文乙"}},
        recall_fusion_scores=recall_scores,
    )
    monkeypatch.setattr(search_v2, "rerank_similarity", lambda *_: ([0.8, 0.8], None))
    indexes, _ = object.__new__(search_v2.Dealer)._rerank_results(
        result, "查询", search_v2.Dealer.RetrievalOptions(page_size=2, final_reranker_weight=0),
    )
    assert indexes == [0, 1]
    assert result.candidate_ids == ["z", "a"]
    assert result.final_fusion_scores == pytest.approx({"z": 1 / 11, "a": 1 / 12})
