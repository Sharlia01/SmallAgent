#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
import logging
from collections import defaultdict
#数据类，类似于其他语言中的结构体
from dataclasses import dataclass, field as dataclass_field

from service.core.rag.settings import TAG_FLD, PAGERANK_FLD
from service.core.rag.utils import rmSpace
from service.core.rag.nlp import rag_tokenizer, query
import numpy as np
from service.core.constraint_compatibility import (
    evaluate_constraint_compatibility,
)
from service.core.rag.utils.doc_store_conn import (
    DocStoreConnection,
    MatchDenseExpr,
    OrderByExpr,
)
from service.core.rag.nlp.model import (
    EMBEDDING_VECTOR_FIELD,
    generate_embedding,
    rerank_similarity,
)

def index_name(uid): return f"{uid}"


@dataclass
class RrfFusionResult:
    ids: list[str]
    scores: dict[str, float]
    ranks: dict[str, dict[str, int]]
    contributions: dict[str, dict[str, float]]


def reciprocal_rank_fusion(
    rankings: dict[str, list[str]],
    weights: dict[str, float],
    *,
    k: int = 60,
) -> RrfFusionResult:
    """Fuse independent rankings without comparing their raw score scales."""
    # 将多路独立检索的排名结果合并成一个统一的排序，而且只用名次，不比较各自的原始分数
    if k <= 0:
        raise ValueError("RRF k must be greater than zero")

    scores = defaultdict(float)
    ranks = defaultdict(dict)
    contributions = defaultdict(dict)

    for source, chunk_ids in rankings.items():
        weight = float(weights.get(source, 1.0))
        if weight <= 0:
            continue

        seen_ids = set()
        for rank, chunk_id in enumerate(chunk_ids, start=1):
            if not chunk_id or chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)
            # contribution表示每个ID从各路拿到的分数贡献
            contribution = weight / (k + rank)
            scores[chunk_id] += contribution
            ranks[chunk_id][source] = rank
            contributions[chunk_id][source] = contribution

    # 按分数降序、名次升序、ID升序排序，确保结果稳定
    ordered_ids = sorted(
        scores,
        key=lambda chunk_id: (
            -scores[chunk_id],
            min(ranks[chunk_id].values()),
            chunk_id,
        ),
    )

    #将defaultdict转换为普通dict，以便序列化和返回
    return RrfFusionResult(
        ids=ordered_ids,
        scores=dict(scores),
        ranks={chunk_id: dict(value) for chunk_id, value in ranks.items()},
        contributions={
            chunk_id: dict(value)
            for chunk_id, value in contributions.items()
        },
    )


class Dealer:
    def __init__(self, dataStore: DocStoreConnection):
        self.qryr = query.FulltextQueryer()
        self.dataStore = dataStore

    @dataclass
    class SearchResult:
        total: int
        ids: list[str]
        query_vector: list[float] | None = None
        field: dict | None = None
        highlight: dict | None = None
        aggregation: list | dict | None = None
        keywords: list[str] | None = None
        group_docs: list[list] | None = None
        rrf_scores: dict[str, float] = dataclass_field(default_factory=dict)
        retrieval_ranks: dict[str, dict[str, int]] = dataclass_field(
            default_factory=dict
        )
        retrieval_scores: dict[str, dict[str, float]] = dataclass_field(
            default_factory=dict
        )
        rrf_contributions: dict[str, dict[str, float]] = dataclass_field(
            default_factory=dict
        )
        branch_totals: dict[str, int] = dataclass_field(default_factory=dict)
        branch_weights: dict[str, float] = dataclass_field(default_factory=dict)
        fused_candidate_count: int = 0
        constraint_compatibility: dict[str, dict] = dataclass_field(
            default_factory=dict
        )
        final_ranking: dict[str, dict] = dataclass_field(default_factory=dict)

    @dataclass
    class SearchContext:
        """一次搜索过程中反复使用的公共数据。"""

        filters: dict
        order_by: OrderByExpr
        offset: int
        limit: int
        source_fields: list[str]
        topk: int

    @dataclass
    class SearchExecutionResult:
        """执行完 Elasticsearch 搜索后得到的中间结果。"""

        response: object
        total: int
        query_vector: list[float] = dataclass_field(default_factory=list)
        keywords: set[str] = dataclass_field(default_factory=set)

    @dataclass
    class SearchBranchResult:
        name: str
        response: object
        total: int
        ids: list[str]
        fields: dict[str, dict]
        scores: dict[str, float]
        keywords: set[str] = dataclass_field(default_factory=set)
        highlight: dict[str, str] = dataclass_field(default_factory=dict)

    @dataclass
    class RetrievalOptions:
        page: int = 1  # 搜索结果页码，默认获取搜索结果的第一页
        page_size: int = 5  # 表示每页最多5个chunk, 综合起来就是返回最相关的5个chunk
        similarity_threshold: float = 0.1  # 最低相似度要求
        # 原问题向量分支在加权 RRF 中的权重。
        vector_similarity_weight: float = 0.3
        top: int = 1024  # 初步向量检索的最大候选数
        candidate_size: int = 100  # 每一路独立召回的候选数量
        rerank_candidate_size: int = 20  # RRF 后进入语义重排的数量
        rrf_k: int = 60  # RRF 排名平滑常数
        final_reranker_weight: float = 0.7  # 二阶段融合中的重排模型权重
        final_rrf_k: int = 10  # 二阶段 RRF 平滑常数
        highlight: bool = False  # 是否返回高亮内容
        doc_ids: list[str] | None = None #用于限定只搜索哪些文档
        aggs: bool = True  # 是否返回文档聚合统计
        #用于对chunk的tag_fea和pagerank_fea进行加分，默认给pagerank_fea加10分
        #但是这个没被保存到chunk中，暂时没生效
        rank_feature: dict = dataclass_field(
            default_factory=lambda: {PAGERANK_FLD: 10}
        )
        kb_ids: list[str] | None = None

        def __post_init__(self):
            positive_fields = {
                "page": self.page,
                "page_size": self.page_size,
                "top": self.top,
                "candidate_size": self.candidate_size,
                "rerank_candidate_size": self.rerank_candidate_size,
                "rrf_k": self.rrf_k,
                "final_rrf_k": self.final_rrf_k,
            }
            for name, value in positive_fields.items():
                if value <= 0:
                    raise ValueError(f"{name} must be greater than zero")
            if not 0 <= self.similarity_threshold <= 1:
                raise ValueError(
                    "similarity_threshold must be between zero and one"
                )
            if not 0 <= self.vector_similarity_weight <= 1:
                raise ValueError(
                    "vector_similarity_weight must be between zero and one"
                )
            if not 0 <= self.final_reranker_weight <= 1:
                raise ValueError(
                    "final_reranker_weight must be between zero and one"
                )

    def get_vector(self, txt, topk=10, similarity=0.1):
        qv = generate_embedding(txt)
        shape = np.array(qv).shape
        if len(shape) > 1:
            raise Exception(
                f"Dealer.get_vector returned array's shape {shape} doesn't match expectation(exact one dimension).")
        embedding_data = [float(v) for v in qv]
        return MatchDenseExpr(
            EMBEDDING_VECTOR_FIELD,
            embedding_data,
            "float",
            "cosine",
            topk,
            {"similarity": similarity},
        )

    def get_filters(self, req):
        condition = dict()
        for key, field in {"kb_ids": "kb_id", "doc_ids": "doc_id"}.items():
            #如果请求中包含kb_ids或doc_ids，并且不为None，则将其添加到过滤条件中
            if key in req and req[key] is not None:
                condition[field] = req[key]

        for key in ["knowledge_graph_kwd", "available_int", "entity_kwd", "from_entity_kwd", "to_entity_kwd", "removed_kwd"]:
            #如果请求中包含这些字段，并且不为None，则将其添加到过滤条件中
            if key in req and req[key] is not None:
                condition[key] = req[key]
        return condition

    def _build_search_context(self, req):
        """准备过滤条件、分页范围和需要从 ES 读取的字段。"""
        page = int(req.get("page", 1)) - 1
        topk = int(req.get("topk", 1024))
        page_size = int(req.get("size", topk))
        source_fields = list(req.get("fields", [
            "docnm_kwd", "content_ltks", "kb_id", "img_id", "title_tks",
            "important_kwd", "position_int", "doc_id", "page_num_int",
            "top_int", "create_timestamp_flt", "knowledge_graph_kwd",
            "question_kwd", "question_tks", "available_int",
            "content_with_weight", PAGERANK_FLD, TAG_FLD,
        ]))
        return self.SearchContext(
            filters=self.get_filters(req),
            order_by=OrderByExpr(),
            offset=page * page_size,
            limit=page_size,
            source_fields=source_fields,
            topk=topk,
        )

    def _search_without_question(self, req, context, idx_names, kb_ids):
        """没有用户问题时，只按过滤和排序条件列出 chunk。"""
        if req.get("sort"):
            context.order_by.asc("page_num_int")
            context.order_by.asc("top_int")
            context.order_by.desc("create_timestamp_flt")

        response = self.dataStore.search(
            context.source_fields, [], context.filters, [], context.order_by,
            context.offset, context.limit, idx_names, kb_ids,
        )
        return self.SearchExecutionResult(
            response=response,
            total=self.dataStore.getTotal(response),
        )

    @staticmethod
    def _expand_keywords(keywords):
        """加入细粒度分词，并使用集合自动去重。"""
        expanded_keywords = set()
        for keyword in keywords:
            expanded_keywords.add(keyword)
            for token in rag_tokenizer.fine_grained_tokenize(keyword).split():
                if len(token) >= 2:
                    expanded_keywords.add(token)
        return expanded_keywords

    def _run_hybrid_search(
        self, context, idx_names, kb_ids, highlight_fields,
        match_expressions, rank_feature,
    ):
        """Execute one independent retrieval branch in Elasticsearch."""
        return self.dataStore.search(
            context.source_fields,
            highlight_fields,
            context.filters,
            match_expressions,
            context.order_by,
            context.offset,
            context.limit,
            idx_names,
            kb_ids,
            #rank_feature主要给重要的chunk额外加分
            rank_feature=rank_feature,
        )

    @staticmethod
    def _query_identity(question):
        return " ".join(str(question or "").split()).casefold()

    @staticmethod
    def _rrf_branch_weights(req, has_expanded_query):
        vector_weight = float(req.get("vector_similarity_weight", 0.6))
        vector_weight = min(1.0, max(0.0, vector_weight))
        keyword_weight = 1.0 - vector_weight

        if has_expanded_query:
            weights = {
                "keyword_original": keyword_weight / 2.0,
                "keyword_expanded": keyword_weight / 2.0,
                "vector_original": vector_weight,
            }
        else:
            weights = {
                "keyword_original": keyword_weight,
                "vector_original": vector_weight,
            }
        return {
            name: weight for name, weight in weights.items() if weight > 0
        }

    @staticmethod
    def _redistribute_empty_branch_weights(branches, requested_weights):
        """Keep modality weights stable when one retrieval branch is empty."""
        active_names = {branch.name for branch in branches if branch.ids}
        active_keywords = sorted(
            name
            for name in active_names
            if name.startswith("keyword_")
        )
        active_vectors = sorted(
            name
            for name in active_names
            if name.startswith("vector_")
        )
        if not active_names:
            return requested_weights

        keyword_budget = sum(
            weight
            for name, weight in requested_weights.items()
            if name.startswith("keyword_")
        )
        vector_budget = sum(
            weight
            for name, weight in requested_weights.items()
            if name.startswith("vector_")
        )

        if not active_keywords:
            vector_budget += keyword_budget
            keyword_budget = 0.0
        if not active_vectors:
            keyword_budget += vector_budget
            vector_budget = 0.0

        weights = {}
        if active_keywords:
            weight = keyword_budget / len(active_keywords)
            weights.update({name: weight for name in active_keywords})
        if active_vectors:
            weight = vector_budget / len(active_vectors)
            weights.update({name: weight for name in active_vectors})
        return weights

    def _format_branch_response(
        self,
        *,
        name,
        response,
        source_fields,
        keywords=None,
        highlight=False,
    ):
        ids = self.dataStore.getChunkIds(response)
        fields = self.dataStore.getFields(
            response,
            list(dict.fromkeys([*source_fields, "_score"])),
        )
        scores = {}
        for chunk_id in ids:
            source = fields.get(chunk_id, {})
            try:
                scores[chunk_id] = float(source.pop("_score", 0.0))
            except (TypeError, ValueError):
                scores[chunk_id] = 0.0

        expanded_keywords = self._expand_keywords(keywords or [])
        highlights = {}
        if highlight:
            highlights = self.dataStore.getHighlight(
                response,
                list(expanded_keywords),
                "content_with_weight",
            )
        return self.SearchBranchResult(
            name=name,
            response=response,
            total=self.dataStore.getTotal(response),
            ids=ids,
            fields=fields,
            scores=scores,
            keywords=expanded_keywords,
            highlight=highlights,
        )

    def _search_text_branch(
        self,
        *,
        name,
        text,
        context,
        idx_names,
        kb_ids,
        highlight,
        rank_feature,
    ):
        highlight_fields = (
            ["content_ltks", "title_tks"] if highlight else []
        )
        match_text, keywords = self.qryr.question(text, min_match=0.3)
        response = self._run_hybrid_search(
            context,
            idx_names,
            kb_ids,
            highlight_fields,
            [match_text],
            rank_feature,
        )

        # Relax only this lexical branch. Metadata and document filters remain
        # unchanged so fallback cannot escape the caller's retrieval scope.
        if self.dataStore.getTotal(response) == 0:
            match_text, keywords = self.qryr.question(text, min_match=0.1)
            response = self._run_hybrid_search(
                context,
                idx_names,
                kb_ids,
                highlight_fields,
                [match_text],
                rank_feature,
            )

        return self._format_branch_response(
            name=name,
            response=response,
            source_fields=context.source_fields,
            keywords=keywords,
            highlight=highlight,
        )

    def _search_vector_branch(
        self,
        *,
        question,
        context,
        idx_names,
        kb_ids,
        similarity,
    ):
        match_dense = self.get_vector(
            question,
            context.topk,
            similarity,
        )
        response = self._run_hybrid_search(
            context,
            idx_names,
            kb_ids,
            [],
            [match_dense],
            None,
        )
        branch = self._format_branch_response(
            name="vector_original",
            response=response,
            source_fields=[*context.source_fields, EMBEDDING_VECTOR_FIELD],
        )
        return branch, list(match_dense.embedding_data)

    def _merge_search_branches(
        self,
        branches,
        branch_weights,
        *,
        query_vector,
        rrf_k,
        rerank_candidate_size,
    ):
        rankings = {branch.name: branch.ids for branch in branches}
        fusion = reciprocal_rank_fusion(
            rankings,
            branch_weights,
            k=rrf_k,
        )
        fused_candidate_count = len(fusion.ids)
        selected_ids = fusion.ids[:rerank_candidate_size]
        # 将列表转换成集合
        selected_id_set = set(selected_ids)

        fields = {}
        highlights = {}
        keywords = set()
        retrieval_scores = defaultdict(dict)

        for branch in branches:
            keywords.update(branch.keywords)
            # branch.fields表示文档的原始字段
            for chunk_id, source in branch.fields.items():
                if chunk_id not in selected_id_set:
                    continue
                target = fields.setdefault(chunk_id, {})
                for field_name, value in source.items():
                    if field_name not in target or target[field_name] is None:
                        target[field_name] = value
            # branch.scores表示文档的原始分数
            for chunk_id, score in branch.scores.items():
                if chunk_id in selected_id_set:
                    retrieval_scores[chunk_id][branch.name] = score
            for chunk_id, value in branch.highlight.items():
                if chunk_id in selected_id_set and chunk_id not in highlights:
                    highlights[chunk_id] = value

        selected_ids = [
            chunk_id for chunk_id in selected_ids if chunk_id in fields
        ]
        return self.SearchResult(
            total=fused_candidate_count,
            ids=selected_ids,
            query_vector=query_vector,
            field=fields,
            highlight=highlights,
            aggregation=[],
            keywords=sorted(keywords),
            rrf_scores={
                chunk_id: fusion.scores[chunk_id]
                for chunk_id in selected_ids
            },
            retrieval_ranks={
                chunk_id: fusion.ranks[chunk_id]
                for chunk_id in selected_ids
            },
            retrieval_scores={
                chunk_id: dict(retrieval_scores.get(chunk_id, {}))
                for chunk_id in selected_ids
            },
            rrf_contributions={
                chunk_id: fusion.contributions[chunk_id]
                for chunk_id in selected_ids
            },
            branch_totals={branch.name: branch.total for branch in branches},
            branch_weights=dict(branch_weights),
            fused_candidate_count=fused_candidate_count,
        )

    def _search_with_question(
        self, req, context, question, idx_names, kb_ids,
        highlight, rank_feature,
    ):
        """Run independent lexical/vector retrieval and fuse with RRF."""
        # Candidate retrieval always starts at rank one in every branch. Final
        # pagination is applied after semantic reranking.
        context.offset = 0
        expanded_question = req.get("keyword_question") or question
        has_expanded_query = (
            self._query_identity(expanded_question)
            != self._query_identity(question)
        )
        # 这里存储着每个分支的权重，向量、扩展查询、原始查询
        branch_weights = self._rrf_branch_weights(req, has_expanded_query)
        branches = []

        if "keyword_original" in branch_weights:
            branches.append(
                self._search_text_branch(
                    name="keyword_original",
                    text=question,
                    context=context,
                    idx_names=idx_names,
                    kb_ids=kb_ids,
                    highlight=highlight,
                    rank_feature=rank_feature,
                )
            )

        if "keyword_expanded" in branch_weights:
            branches.append(
                self._search_text_branch(
                    name="keyword_expanded",
                    text=expanded_question,
                    context=context,
                    idx_names=idx_names,
                    kb_ids=kb_ids,
                    highlight=highlight,
                    rank_feature=rank_feature,
                )
            )

        query_vector = []
        if "vector_original" in branch_weights:
            vector_branch, query_vector = self._search_vector_branch(
                question=question,
                context=context,
                idx_names=idx_names,
                kb_ids=kb_ids,
                similarity=req.get("similarity", 0.1),
            )
            branches.append(vector_branch)

        #当某一路检索“没有召回结果”时，将其权重重新分配给其他检索分支，以保持总权重不变。
        branch_weights = self._redistribute_empty_branch_weights(
            branches,
            branch_weights,
        )

        return self._merge_search_branches(
            branches,
            branch_weights,
            query_vector=query_vector,
            rrf_k=int(req.get("rrf_k", 60)),
            rerank_candidate_size=int(
                req.get("rerank_candidate_size", 20)
            ),
        )

    def _format_search_result(self, execution, source_fields):
        """把 Elasticsearch 原始响应包装成统一的 SearchResult。"""
        keywords = list(execution.keywords)
        response = execution.response
        return self.SearchResult(
            total=execution.total,
            ids=self.dataStore.getChunkIds(response),
            query_vector=execution.query_vector,
            aggregation=self.dataStore.getAggregation(response, "docnm_kwd"),
            highlight=self.dataStore.getHighlight(
                response, keywords, "content_with_weight",
            ),
            #field是一个字典，键是chunk_id，值是该chunk的所有字段及其值
            field=self.dataStore.getFields(response, source_fields),
            keywords=keywords,
        )

    def search(self, req, idx_names: str | list[str],
               kb_ids: list[str], highlight=False,
               rank_feature: dict | None = None):
        """从 Elasticsearch 初步召回候选 chunk，供 retrieval() 重排。"""
        context = self._build_search_context(req)
        question = req.get("question", "")

        if question:
            search_result = self._search_with_question(
                req, context, question, idx_names, kb_ids,
                highlight, rank_feature,
            )
        else:
            execution = self._search_without_question(
                req, context, idx_names, kb_ids,
            )
            search_result = self._format_search_result(
                execution,
                context.source_fields,
            )

        logging.debug(f"Dealer.search TOTAL: {search_result.total}")
        return search_result

    @staticmethod
    def trans2floats(txt):
        return [float(t) for t in txt.split("\t")]

    def _rank_feature_scores(self, query_rfea, search_res):
        ## For rank feature(tag_fea) scores.
        rank_fea = []
        pageranks = []
        for chunk_id in search_res.ids:
            pageranks.append(search_res.field[chunk_id].get(PAGERANK_FLD, 0))
        pageranks = np.array(pageranks, dtype=float)

        if not query_rfea:
            return np.array([0 for _ in range(len(search_res.ids))]) + pageranks

        q_denor = np.sqrt(np.sum([s*s for t,s in query_rfea.items() if t != PAGERANK_FLD]))
        for i in search_res.ids:
            nor, denor = 0, 0
            for t, sc in eval(search_res.field[i].get(TAG_FLD, "{}")).items():
                if t in query_rfea:
                    nor += query_rfea[t] * sc
                denor += sc * sc
            if denor == 0:
                rank_fea.append(0)
            else:
                rank_fea.append(nor/np.sqrt(denor)/q_denor)
        return np.array(rank_fea)*10. + pageranks

    def rerank_by_model(self, search_result, query):
        """Rerank RRF candidates using only the user's original question."""
        documents = []
        for chunk_id in search_result.ids:
            source = search_result.field[chunk_id]
            content = source.get("content_with_weight")
            if not content:
                content = rmSpace(source.get("content_ltks", ""))
            documents.append(str(content))

        semantic_scores, _ = rerank_similarity(query, documents)
        return np.asarray(semantic_scores, dtype=float)

    def hybrid_similarity(self, ans_embd, ins_embd, ans, inst):
        return self.qryr.hybrid_similarity(ans_embd,
                                           ins_embd,
                                           rag_tokenizer.tokenize(ans).split(),
                                           rag_tokenizer.tokenize(inst).split())

    def _build_search_request(
        self,
        question,
        options: RetrievalOptions,
        keyword_question: str | None = None,
    ):
        """将检索配置转换为底层搜索请求。"""
        request = {
            "question": question,
            "keyword_question": keyword_question or question,
            "kb_ids": options.kb_ids,
            "doc_ids": options.doc_ids,
            # 每一路先独立召回候选，再由 RRF 截断后进入语义重排。
            "size": min(
                options.top,
                max(
                    1,
                    options.candidate_size,
                    options.page * options.page_size,
                ),
            ),
            "vector": True,
            "topk": options.top,
            "similarity": options.similarity_threshold,
            "vector_similarity_weight": options.vector_similarity_weight,
            "rrf_k": max(1, options.rrf_k),
            "rerank_candidate_size": min(
                options.top,
                max(
                    options.rerank_candidate_size,
                    options.page * options.page_size,
                    1,
                ),
            ),
            #作用是标记chunk是否可用，入库时可以设置某个chunk为不可用来屏蔽它，避免被检索到
            "available_int": 1,
        }

        return request

    @staticmethod
    def _build_index_names(tenant_ids: str | list[str]):
        """将用户 ID 统一转换成 Elasticsearch 索引名列表。"""
        if isinstance(tenant_ids, str):
            tenant_ids = tenant_ids.split(",")
        return [index_name(tenant_id) for tenant_id in tenant_ids]

    def _rerank_results(
        self,
        search_result: SearchResult,
        question: str,
        options: RetrievalOptions,
    ):
        """Fuse semantic and retrieval ranks, then demote hard conflicts."""
        if not search_result.ids:
            return [], np.array([], dtype=float)

        if question:
            semantic_scores = self.rerank_by_model(search_result, question)
        else:
            semantic_scores = np.ones(len(search_result.ids), dtype=float)

        compatibility_by_id = {}
        for chunk_id in search_result.ids:
            source = search_result.field[chunk_id]
            content = source.get("content_with_weight")
            if not content:
                content = rmSpace(source.get("content_ltks", ""))
            # 计算约束兼容性
            decision = evaluate_constraint_compatibility(
                question,
                str(content),
                document_name=str(source.get("docnm_kwd", "")),
            )
            compatibility_by_id[chunk_id] = decision.to_dict()
        search_result.constraint_compatibility = compatibility_by_id

        semantic_order = sorted(
            range(len(search_result.ids)),
            key=lambda index: (
                -float(semantic_scores[index]),
                -float(
                    search_result.rrf_scores.get(
                        search_result.ids[index],
                        0.0,
                    )
                ),
                search_result.ids[index],
            ),
        )
        semantic_ids = [search_result.ids[index] for index in semantic_order]
        semantic_rank_by_id = {
            chunk_id: rank
            for rank, chunk_id in enumerate(semantic_ids, start=1)
        }
        retrieval_rrf_ids = sorted(
            search_result.ids,
            key=lambda chunk_id: (
                -float(search_result.rrf_scores.get(chunk_id, 0.0)),
                chunk_id,
            ),
        )
        retrieval_rrf_rank_by_id = {
            chunk_id: rank
            for rank, chunk_id in enumerate(retrieval_rrf_ids, start=1)
        }
        reranker_weight = float(options.final_reranker_weight)
        final_weights = {
            "semantic_reranker": reranker_weight,
            "retrieval_rrf": 1.0 - reranker_weight,
        }
        final_fusion = reciprocal_rank_fusion(
            {
                "semantic_reranker": semantic_ids,
                "retrieval_rrf": retrieval_rrf_ids,
            },
            final_weights,
            k=options.final_rrf_k,
        )
        final_ranking = {}
        for chunk_id in search_result.ids:
            final_ranking[chunk_id] = {
                "semantic_reranker_rank": semantic_rank_by_id[chunk_id],
                "retrieval_rrf_rank": retrieval_rrf_rank_by_id[chunk_id],
                "final_fusion_score": final_fusion.scores[chunk_id],
                "final_fusion_contributions": (
                    final_fusion.contributions.get(chunk_id, {})
                ),
            }
        search_result.final_ranking = final_ranking

        # 明确冲突优先降级；其余候选由重排名次与第一阶段 RRF 名次共同决定。
        ordered_indexes = sorted(
            range(len(search_result.ids)),
            key=lambda index: (
                int(
                    compatibility_by_id[
                        search_result.ids[index]
                    ]["conflict_count"]
                ),
                -float(
                    final_ranking[
                        search_result.ids[index]
                    ]["final_fusion_score"]
                ),
                -float(semantic_scores[index]),
                -float(
                    search_result.rrf_scores.get(
                        search_result.ids[index],
                        0.0,
                    )
                ),
                search_result.ids[index],
            ),
        )
        ordered_indexes = [
            index
            for index in ordered_indexes
            if semantic_scores[index] >= options.similarity_threshold
        ]
        start = (options.page - 1) * options.page_size
        end = options.page * options.page_size
        return ordered_indexes[start:end], semantic_scores

    @staticmethod
    def _build_chunk(
        chunk_id,
        source,
        rerank_score,
        rrf_score,
        retrieval_ranks,
        retrieval_scores,
        rrf_contributions,
        constraint_compatibility,
        final_ranking,
        vector_column,
        zero_vector,
    ):
        """把一个底层搜索结果转换成对外返回的文本片段。"""
        keyword_scores = [
            score
            for branch, score in retrieval_scores.items()
            if branch.startswith("keyword_")
        ]
        return {
            "chunk_id": chunk_id,
            "content_ltks": source["content_ltks"],
            "content_with_weight": source["content_with_weight"],
            "doc_id": source.get("doc_id", ""),
            "docnm_kwd": source.get("docnm_kwd", ""),
            "kb_id": source["kb_id"],
            "important_kwd": source.get("important_kwd", []),
            "image_id": source.get("img_id", ""),
            "similarity": rerank_score,
            "rerank_score": rerank_score,
            "rrf_score": rrf_score,
            "vector_similarity": retrieval_scores.get(
                "vector_original",
                0.0,
            ),
            "term_similarity": max(keyword_scores, default=0.0),
            "retrieval_ranks": retrieval_ranks,
            "retrieval_scores": retrieval_scores,
            "rrf_contributions": rrf_contributions,
            "constraint_compatibility": constraint_compatibility,
            "final_ranking": final_ranking,
            "vector": source.get(vector_column, zero_vector),
            "positions": source.get("position_int", []),
        }

    @staticmethod
    def _aggregate_documents(chunks):
        """统计每份文档在最终结果中命中的片段数量。"""
        aggregations = {}
        for chunk in chunks:
            document_name = chunk["docnm_kwd"]
            if document_name not in aggregations:
                aggregations[document_name] = {
                    "doc_id": chunk["doc_id"],
                    "count": 0,
                }
            aggregations[document_name]["count"] += 1

        sorted_aggregations = sorted(
            aggregations.items(),
            key=lambda item: item[1]["count"],
            reverse=True,
        )
        return [
            {
                "doc_name": document_name,
                "doc_id": aggregation["doc_id"],
                "count": aggregation["count"],
            }
            for document_name, aggregation in sorted_aggregations
        ]

    def _build_retrieval_result(
        self,
        search_result: SearchResult,
        ranked_results,
        options: RetrievalOptions,
    ):
        """过滤并组装最终的检索结果。"""
        indexes, rerank_scores = ranked_results
        query_vector = search_result.query_vector or []
        zero_vector = [0.0] * len(query_vector)
        chunks = []

        for result_index in indexes:
            # 兼容性排序后语义分不再严格单调，低分项不能阻断后续结果。
            if rerank_scores[result_index] < options.similarity_threshold:
                continue
            if len(chunks) >= options.page_size:
                break

            chunk_id = search_result.ids[result_index]
            source = search_result.field[chunk_id]
            #组装chunk字典
            chunk = self._build_chunk(
                chunk_id,
                source,
                float(rerank_scores[result_index]),
                float(search_result.rrf_scores.get(chunk_id, 0.0)),
                dict(search_result.retrieval_ranks.get(chunk_id, {})),
                dict(search_result.retrieval_scores.get(chunk_id, {})),
                dict(search_result.rrf_contributions.get(chunk_id, {})),
                dict(
                    search_result.constraint_compatibility.get(
                        chunk_id,
                        {},
                    )
                ),
                dict(search_result.final_ranking.get(chunk_id, {})),
                EMBEDDING_VECTOR_FIELD,
                zero_vector,
            )

            # 开了高亮且有高亮片段时，把高亮文本挂到chunk上
            if options.highlight and search_result.highlight:
                #rmSpace函数用于去掉高亮内容中的多余空格
                chunk["highlight"] = rmSpace(
                    search_result.highlight.get(
                        chunk_id,
                        chunk["content_with_weight"],
                    )
                )
            chunks.append(chunk)

        document_aggregations = (
            self._aggregate_documents(chunks) if options.aggs else []
        )

        return {
            "total": search_result.total,
            "chunks": chunks,
            "doc_aggs": document_aggregations,
            "retrieval_fusion": {
                "method": "weighted_rrf",
                "rrf_k": options.rrf_k,
                "branch_weights": search_result.branch_weights,
                "branch_totals": search_result.branch_totals,
                "fused_candidate_count": (
                    search_result.fused_candidate_count
                ),
                "rerank_candidate_count": len(search_result.ids),
                "rerank_query": "original_question",
                "final_fusion": {
                    "method": "weighted_rrf",
                    "rrf_k": options.final_rrf_k,
                    "weights": {
                        "semantic_reranker": (
                            options.final_reranker_weight
                        ),
                        "retrieval_rrf": (
                            1.0 - options.final_reranker_weight
                        ),
                    },
                    "constraint_conflicts_first": True,
                },
            },
        }

    def retrieval(
        self,
        question: str,
        tenant_ids: str | list[str],
        options: RetrievalOptions,
        keyword_question: str | None = None,
    ):

        #根据用户问题和检索配置，生成一份搜索请求参数
        request = self._build_search_request(
            question,
            options,
            keyword_question,
        )

        search_result = self.search(
            request,
            self._build_index_names(tenant_ids),
            options.kb_ids,
            options.highlight,
            rank_feature=options.rank_feature, #关键字参数，python可以只传一部分参数
        )


        ranked_results = self._rerank_results(
            search_result,
            question,
            options,
        )

        return self._build_retrieval_result(
            search_result,
            ranked_results,
            options,
        )

    def sql_retrieval(self, sql, fetch_size=128, format="json"):
        tbl = self.dataStore.sql(sql, fetch_size, format)
        return tbl

    def chunk_list(self, doc_id: str, tenant_id: str,
                   kb_ids: list[str], max_count=1024,
                   offset=0,
                   fields=["docnm_kwd", "content_with_weight", "img_id"]):
        condition = {"doc_id": doc_id}
        res = []
        bs = 128
        for p in range(offset, max_count, bs):
            es_res = self.dataStore.search(fields, [], condition, [], OrderByExpr(), p, bs, index_name(tenant_id),
                                           kb_ids)
            dict_chunks = self.dataStore.getFields(es_res, fields)
            for id, doc in dict_chunks.items():
                doc["id"] = id
            if dict_chunks:
                res.extend(dict_chunks.values())
            if len(dict_chunks.values()) < bs:
                break
        return res

    def all_tags(self, tenant_id: str, kb_ids: list[str], S=1000):
        res = self.dataStore.search([], [], {}, [], OrderByExpr(), 0, 0, index_name(tenant_id), kb_ids, ["tag_kwd"])
        return self.dataStore.getAggregation(res, "tag_kwd")

    def all_tags_in_portion(self, tenant_id: str, kb_ids: list[str], S=1000):
        res = self.dataStore.search([], [], {}, [], OrderByExpr(), 0, 0, index_name(tenant_id), kb_ids, ["tag_kwd"])
        res = self.dataStore.getAggregation(res, "tag_kwd")
        total = np.sum([c for _, c in res])
        return {t: (c + 1) / (total + S) for t, c in res}

    def tag_content(self, tenant_id: str, kb_ids: list[str], doc, all_tags, topn_tags=3, keywords_topn=30, S=1000):
        idx_nm = index_name(tenant_id)
        match_txt = self.qryr.paragraph(doc["title_tks"] + " " + doc["content_ltks"], doc.get("important_kwd", []), keywords_topn)
        res = self.dataStore.search([], [], {}, [match_txt], OrderByExpr(), 0, 0, idx_nm, kb_ids, ["tag_kwd"])
        aggs = self.dataStore.getAggregation(res, "tag_kwd")
        if not aggs:
            return False
        cnt = np.sum([c for _, c in aggs])
        tag_fea = sorted([(a, round(0.1*(c + 1) / (cnt + S) / max(1e-6, all_tags.get(a, 0.0001)))) for a, c in aggs],
                         key=lambda x: x[1] * -1)[:topn_tags]
        doc[TAG_FLD] = {a: c for a, c in tag_fea if c > 0}
        return True

    def tag_query(self, question: str, tenant_ids: str | list[str], kb_ids: list[str], all_tags, topn_tags=3, S=1000):
        if isinstance(tenant_ids, str):
            idx_nms = index_name(tenant_ids)
        else:
            idx_nms = [index_name(tid) for tid in tenant_ids]
        match_txt, _ = self.qryr.question(question, min_match=0.0)
        res = self.dataStore.search([], [], {}, [match_txt], OrderByExpr(), 0, 0, idx_nms, kb_ids, ["tag_kwd"])
        aggs = self.dataStore.getAggregation(res, "tag_kwd")
        if not aggs:
            return {}
        cnt = np.sum([c for _, c in aggs])
        tag_fea = sorted([(a, round(0.1*(c + 1) / (cnt + S) / max(1e-6, all_tags.get(a, 0.0001)))) for a, c in aggs],
                         key=lambda x: x[1] * -1)[:topn_tags]
        return {a: max(1, c) for a, c in tag_fea}
