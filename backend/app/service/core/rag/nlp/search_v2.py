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
import re
#数据类，类似于其他语言中的结构体
from dataclasses import dataclass, field

from service.core.rag.settings import TAG_FLD, PAGERANK_FLD
from service.core.rag.utils import rmSpace
from service.core.rag.nlp import rag_tokenizer, query
import numpy as np
from service.core.rag.utils.doc_store_conn import DocStoreConnection, MatchDenseExpr, FusionExpr, OrderByExpr
from service.core.rag.nlp.model import generate_embedding, rerank_similarity

def index_name(uid): return f"{uid}"


class Dealer:
    RERANK_PAGE_LIMIT = 3

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
        query_vector: list[float] = field(default_factory=list)
        keywords: set[str] = field(default_factory=set)

    @dataclass
    class RetrievalOptions:
        page: int = 1  # 搜索结果页码，默认获取搜索结果的第一页
        page_size: int = 5  # 表示每页最多5个chunk, 综合起来就是返回最相关的5个chunk
        similarity_threshold: float = 0.1  # 最低相似度要求
        vector_similarity_weight: float = 0.3  # 语义向量相似度权重
        top: int = 1024  # 初步向量检索的最大候选数
        highlight: bool = False  # 是否返回高亮内容
        doc_ids: list[str] | None = None #用于限定只搜索哪些文档
        aggs: bool = True  # 是否返回文档聚合统计
        #用于对chunk的tag_fea和pagerank_fea进行加分，默认给pagerank_fea加10分
        #但是这个没被保存到chunk中，暂时没生效
        rank_feature: dict = field(
            default_factory=lambda: {PAGERANK_FLD: 10}
        )
        kb_ids: list[str] | None = None
        embd_mdl: object | None = None

    def get_vector(self, txt, emb_mdl, topk=10, similarity=0.1):
        qv = generate_embedding(txt)
        shape = np.array(qv).shape
        if len(shape) > 1:
            raise Exception(
                f"Dealer.get_vector returned array's shape {shape} doesn't match expectation(exact one dimension).")
        embedding_data = [float(v) for v in qv]
        vector_column_name = f"q_{len(embedding_data)}_vec"
        return MatchDenseExpr(vector_column_name, embedding_data, 'float', 'cosine', topk, {"similarity": similarity})

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
        """把关键词和向量表达式交给 Elasticsearch 执行。"""
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

    def _search_with_question(
        self, req, context, question, idx_names, kb_ids,
        emb_mdl, highlight, rank_feature,
    ):
        """
        根据用户问题执行一次“关键词 + 向量”的混合搜索。

        可以先把这个函数理解成四步：
        1. 把问题转换成关键词查询；
        2. 把问题转换成向量查询；
        3. 把两种查询交给 Elasticsearch 一起搜索；
        4. 如果没有结果，就放宽条件再搜索一次。
        """

        highlight_fields = ["content_ltks", "title_tks"] if highlight else []

        # match_text：关键词搜索参数，包括搜哪些字段、什么文字等
        match_text, keywords = self.qryr.question(question, min_match=0.3)

        # match_dense表示向量搜索参数，里面有问题对应的向量和向量比较方法
        match_dense = self.get_vector(
            question, emb_mdl, context.topk, req.get("similarity", 0.1),
        )

        # 问题对应的向量，例如 [0.12, -0.08, ...]。
        query_vector = match_dense.embedding_data

        # 不同向量模型产生的向量维度可能不同，例如 768 维或 1024 维。
        # 文档向量保存在类似 q_1024_vec 的字段中，所以这里要把对应字段加入返回列表。
        context.source_fields.append(f"q_{len(query_vector)}_vec")

        # 混合搜索参数，表示把关键词和向量的得分按 5%/95% 的比例加权。
        fusion = FusionExpr(
            "weighted_sum", context.topk, {"weights": "0.05, 0.95"},
        )

        # 第一次执行混合搜索：
        response = self._run_hybrid_search(
            context, idx_names, kb_ids, highlight_fields,
            [match_text, match_dense, fusion], rank_feature,
        )

        # 从 Elasticsearch 原始响应中读取命中的 chunk 总数。
        total = self.dataStore.getTotal(response)

        # 如果第一次一个 chunk 都没有搜到，就放宽搜索条件再试一次。
        if total == 0:
            # 把关键词匹配要求从 30% 降到 10%，让关键词搜索更宽松。
            # 下划线 _ 表示这里不需要接收第二个返回值（keywords）。
            match_text, _ = self.qryr.question(question, min_match=0.1)

            # 移除“只搜索指定文档”的限制。
            context.filters.pop("doc_id", None)

            # 调整向量检索的最低相似度要求，然后重新搜索。
            match_dense.extra_options["similarity"] = 0.17
            response = self._run_hybrid_search(
                context, idx_names, kb_ids, highlight_fields,
                [match_text, match_dense, fusion], rank_feature,
            )

            # 重试以后，需要重新统计命中的 chunk 数量。
            total = self.dataStore.getTotal(response)

        # 返回一个中间结果对象，后面 _format_search_result() 会继续把它整理成 SearchResult。
        return self.SearchExecutionResult(
            response=response,
            total=total,
            query_vector=query_vector,
            # 进一步细分关键词并去重，供后面的高亮处理使用。
            keywords=self._expand_keywords(keywords),
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
               kb_ids: list[str], emb_mdl=None, highlight=False,
               rank_feature: dict | None = None):
        """从 Elasticsearch 初步召回候选 chunk，供 retrieval() 重排。"""
        context = self._build_search_context(req)
        question = req.get("question", "")

        if question:
            execution = self._search_with_question(
                req, context, question, idx_names, kb_ids,
                emb_mdl, highlight, rank_feature,
            )
        else:
            execution = self._search_without_question(
                req, context, idx_names, kb_ids,
            )

        logging.debug(f"Dealer.search TOTAL: {execution.total}")
        return self._format_search_result(execution, context.source_fields)

    @staticmethod
    def trans2floats(txt):
        return [float(t) for t in txt.split("\t")]

    def insert_citations(self, answer, chunks, chunk_v,
                         embd_mdl, tkweight=0.1, vtweight=0.9):
        assert len(chunks) == len(chunk_v)
        if not chunks:
            return answer, set([])
        pieces = re.split(r"(```)", answer)
        if len(pieces) >= 3:
            i = 0
            pieces_ = []
            while i < len(pieces):
                if pieces[i] == "```":
                    st = i
                    i += 1
                    while i < len(pieces) and pieces[i] != "```":
                        i += 1
                    if i < len(pieces):
                        i += 1
                    pieces_.append("".join(pieces[st: i]) + "\n")
                else:
                    pieces_.extend(
                        re.split(
                            r"([^\|][；。？!！\n]|[a-z][.?;!][ \n])",
                            pieces[i]))
                    i += 1
            pieces = pieces_
        else:
            pieces = re.split(r"([^\|][；。？!！\n]|[a-z][.?;!][ \n])", answer)
        for i in range(1, len(pieces)):
            if re.match(r"([^\|][；。？!！\n]|[a-z][.?;!][ \n])", pieces[i]):
                pieces[i - 1] += pieces[i][0]
                pieces[i] = pieces[i][1:]
        idx = []
        pieces_ = []
        for i, t in enumerate(pieces):
            if len(t) < 5:
                continue
            idx.append(i)
            pieces_.append(t)
        logging.debug("{} => {}".format(answer, pieces_))
        if not pieces_:
            return answer, set([])

        ans_v, _ = embd_mdl.encode(pieces_)
        for i in range(len(chunk_v)):
            if len(ans_v[0]) != len(chunk_v[i]):
                chunk_v[i] = [0.0]*len(ans_v[0])
                logging.warning("The dimension of query and chunk do not match: {} vs. {}".format(len(ans_v[0]), len(chunk_v[i])))

        assert len(ans_v[0]) == len(chunk_v[0]), "The dimension of query and chunk do not match: {} vs. {}".format(
            len(ans_v[0]), len(chunk_v[0]))

        chunks_tks = [rag_tokenizer.tokenize(self.qryr.rmWWW(ck)).split()
                      for ck in chunks]
        cites = {}
        thr = 0.63
        while thr > 0.3 and len(cites.keys()) == 0 and pieces_ and chunks_tks:
            for i, a in enumerate(pieces_):
                sim, tksim, vtsim = self.qryr.hybrid_similarity(ans_v[i],
                                                                chunk_v,
                                                                rag_tokenizer.tokenize(
                                                                    self.qryr.rmWWW(pieces_[i])).split(),
                                                                chunks_tks,
                                                                tkweight, vtweight)
                mx = np.max(sim) * 0.99
                logging.debug("{} SIM: {}".format(pieces_[i], mx))
                if mx < thr:
                    continue
                cites[idx[i]] = list(
                    set([str(ii) for ii in range(len(chunk_v)) if sim[ii] > mx]))[:4]
            thr *= 0.8

        res = ""
        seted = set([])
        for i, p in enumerate(pieces):
            res += p
            if i not in idx:
                continue
            if i not in cites:
                continue
            for c in cites[i]:
                assert int(c) < len(chunk_v)
            for c in cites[i]:
                if c in seted:
                    continue
                res += f" ##{c}$$"
                seted.add(c)

        return res, seted

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

    def rerank_by_model(self, sres, query, tkweight=0.3,
                        vtweight=0.7, cfield="content_ltks",
                        rank_feature: dict | None = None):

        #提取问题关键词
        _, keywords = self.qryr.question(query)

        for i in sres.ids:
            if isinstance(sres.field[i].get("important_kwd", []), str):
                sres.field[i]["important_kwd"] = [sres.field[i]["important_kwd"]]

        ins_tw = []
        #整理每个候选chunk的词
        for i in sres.ids:
            content_ltks = sres.field[i][cfield].split()
            title_tks = [t for t in sres.field[i].get("title_tks", "").split() if t]
            important_kwd = sres.field[i].get("important_kwd", [])
            tks = content_ltks + title_tks + important_kwd
            ins_tw.append(tks)

        #关键词覆盖分数
        tksim = self.qryr.token_similarity(keywords, ins_tw)

        #重排模型分数
        vtsim, _ = rerank_similarity(query, [rmSpace(" ".join(tks)) for tks in ins_tw])
        ## For rank feature(tag_fea) scores. 额外重要性分数
        rank_fea = self._rank_feature_scores(rank_feature, sres)

        return tkweight * (np.array(tksim)+rank_fea) + vtweight * vtsim, tksim, vtsim

    def hybrid_similarity(self, ans_embd, ins_embd, ans, inst):
        return self.qryr.hybrid_similarity(ans_embd,
                                           ins_embd,
                                           rag_tokenizer.tokenize(ans).split(),
                                           rag_tokenizer.tokenize(inst).split())

    def _build_search_request(self, question, options: RetrievalOptions):
        """将检索配置转换为底层搜索请求。"""
        request = {
            "question": question,
            "kb_ids": options.kb_ids,
            "doc_ids": options.doc_ids,
            #size目的是先从Elasticsearch中获取足够多的候选片段，再进行重新排序
            "size": max(options.page_size * self.RERANK_PAGE_LIMIT, 128),
            "vector": True,
            "topk": options.top,
            "similarity": options.similarity_threshold,
            #作用是标记chunk是否可用，入库时可以设置某个chunk为不可用来屏蔽它，避免被检索到
            "available_int": 1,
        }

        #代码只对前3页执行重排，如果查看第4页及以后的结果，不重排，直接返回5条结果
        if options.page > self.RERANK_PAGE_LIMIT:
            request["page"] = options.page
            request["size"] = options.page_size
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
        """对初步搜索结果重新计算相似度并排序。"""

        #若页码超过第3页或者没有搜索结果，则不进行重排，直接返回默认分数和索引
        if options.page > self.RERANK_PAGE_LIMIT or search_result.total == 0:
            default_scores = [1] * len(search_result.ids)
            indexes = list(range(len(search_result.ids)))
            return indexes, default_scores, default_scores, default_scores

        #每个chunk的综合分数、关键词分数、重排分数
        final_scores, term_scores, vector_scores = self.rerank_by_model(
            search_result,
            question,
            1 - options.vector_similarity_weight,
            options.vector_similarity_weight,
            rank_feature=options.rank_feature,
        )

        #表示取排序后的前5条
        start = (options.page - 1) * options.page_size
        end = options.page * options.page_size
        indexes = np.argsort(final_scores * -1)[start:end]
        return indexes, final_scores, term_scores, vector_scores

    @staticmethod
    def _build_chunk(
        chunk_id,
        source,
        similarity,
        term_similarity,
        vector_similarity,
        vector_column,
        zero_vector,
    ):
        """把一个底层搜索结果转换成对外返回的文本片段。"""
        return {
            "chunk_id": chunk_id,
            "content_ltks": source["content_ltks"],
            "content_with_weight": source["content_with_weight"],
            "doc_id": source.get("doc_id", ""),
            "docnm_kwd": source.get("docnm_kwd", ""),
            "kb_id": source["kb_id"],
            "important_kwd": source.get("important_kwd", []),
            "image_id": source.get("img_id", ""),
            "similarity": similarity,
            "vector_similarity": vector_similarity,
            "term_similarity": term_similarity,
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
        indexes, final_scores, term_scores, vector_scores = ranked_results
        query_vector = search_result.query_vector or []
        vector_column = f"q_{len(query_vector)}_vec"
        zero_vector = [0.0] * len(query_vector)
        chunks = []

        for result_index in indexes:
            if final_scores[result_index] < options.similarity_threshold:
                break
            if len(chunks) >= options.page_size:
                break

            chunk_id = search_result.ids[result_index]
            source = search_result.field[chunk_id]
            chunk = self._build_chunk(
                chunk_id,
                source,
                final_scores[result_index],
                term_scores[result_index],
                vector_scores[result_index],
                vector_column,
                zero_vector,
            )
            if options.highlight and search_result.highlight:
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
        }

    def retrieval(self, question: str, tenant_ids: str | list[str], options: RetrievalOptions):

        #根据用户问题和检索配置，生成一份搜索请求参数
        request = self._build_search_request(question, options)

        search_result = self.search(
            request,
            self._build_index_names(tenant_ids),
            options.kb_ids,
            options.embd_mdl,
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
