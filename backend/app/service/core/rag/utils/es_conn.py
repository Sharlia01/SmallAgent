#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
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
import time
import os
import json

import copy
from elasticsearch import BadRequestError, Elasticsearch
from elasticsearch.helpers import scan
from elasticsearch_dsl import UpdateByQuery, Q, Search, Index
from service.core.rag.utils import singleton
from service.core.api.utils.file_utils import get_project_base_directory
from service.core.rag.utils.doc_store_conn import MatchExpr, OrderByExpr, MatchTextExpr, MatchDenseExpr, FusionExpr
from service.core.rag.nlp import is_english
from dotenv import load_dotenv

load_dotenv()

ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
ATTEMPT_TIME = 2
PAGERANK_FLD = "pagerank_fea"
TAG_FLD = "tag_feas"

logger = logging.getLogger('ragflow.es_conn')


@singleton
class ESConnection:
    def __init__(self):
        self.info = {}
        es_password = os.getenv("ELASTIC_PASSWORD")
        if not es_password:
            raise RuntimeError("ELASTIC_PASSWORD must be configured")

        logger.info(f"Connecting to Elasticsearch at {ES_HOST}")
        self.es = Elasticsearch(
            [ES_HOST],  # Elasticsearch URL
            basic_auth=(os.getenv("ELASTIC_USERNAME", "elastic"), es_password),
            verify_certs=False,  # 禁用 SSL 证书验证
            timeout=600
        )
        logger.info("Elasticsearch connection established")

        fp_mapping = os.path.join(get_project_base_directory(), "conf", "mapping.json")
        self.mapping = json.load(open(fp_mapping, "r"))


    """
    Helper functions for search result
    """

    def getTotal(self, res):
        if isinstance(res["hits"]["total"], type({})):
            return res["hits"]["total"]["value"]
        return res["hits"]["total"]

    def getChunkIds(self, res):
        return [d["_id"] for d in res["hits"]["hits"]]
    

    def getHighlight(self, res, keywords: list[str], fieldnm: str):
        ans = {}
        for d in res["hits"]["hits"]:
            hlts = d.get("highlight")
            if not hlts:
                continue
            txt = "...".join([a for a in list(hlts.items())[0][1]])
            if not is_english(txt.split()):
                ans[d["_id"]] = txt
                continue

            txt = d["_source"][fieldnm]
            txt = re.sub(r"[\r\n]", " ", txt, flags=re.IGNORECASE | re.MULTILINE)
            txts = []
            for t in re.split(r"[.?!;\n]", txt):
                for w in keywords:
                    t = re.sub(r"(^|[ .?/'\"\(\)!,:;-])(%s)([ .?/'\"\(\)!,:;-])" % re.escape(w), r"\1<em>\2</em>\3", t,
                               flags=re.IGNORECASE | re.MULTILINE)
                if not re.search(r"<em>[^<>]+</em>", t, flags=re.IGNORECASE | re.MULTILINE):
                    continue
                txts.append(t)
            ans[d["_id"]] = "...".join(txts) if txts else "...".join([a for a in list(hlts.items())[0][1]])

        return ans
    

    def getAggregation(self, res, fieldnm: str):
        agg_field = "aggs_" + fieldnm
        if "aggregations" not in res or agg_field not in res["aggregations"]:
            return list()
        bkts = res["aggregations"][agg_field]["buckets"]
        return [(b["key"], b["doc_count"]) for b in bkts]

    def getFields(self, res, fields: list[str]) -> dict[str, dict]:
        res_fields = {}
        if not fields:
            return {}
        for d in self.__getSource(res):
            m = {n: d.get(n) for n in fields if d.get(n) is not None}
            for n, v in m.items():
                if isinstance(v, list):
                    m[n] = v
                    continue
                if not isinstance(v, str):
                    m[n] = str(m[n])
                # if n.find("tks") > 0:
                #     m[n] = rmSpace(m[n])

            if m:
                res_fields[d["id"]] = m
        return res_fields


    def __getSource(self, res):
        rr = []
        for d in res["hits"]["hits"]:
            d["_source"]["id"] = d["_id"]
            d["_source"]["_score"] = d["_score"]
            rr.append(d["_source"])
        return rr

    """
    Database operations
    """
    def _ensure_index(self, index_name: str) -> None:
        """Create a missing index with the configured dynamic templates."""
        if self.es.indices.exists(index=index_name):
            return

        try:
            self.es.indices.create(
                index=index_name,
                settings=self.mapping.get("settings", {}),
                mappings=self.mapping.get("mappings", {}),
            )
        except BadRequestError as error:
            # Another upload may create the same user index after exists().
            if "resource_already_exists_exception" not in str(error):
                raise

    def insert(self, documents: list[dict], indexName: str, knowledgebaseId: str = None) -> list[str]:
        # Refers to https://www.elastic.co/guide/en/elasticsearch/reference/current/docs-bulk.html
        operations = []
        for d in documents:
            assert "_id" not in d
            assert "id" in d
            d_copy = copy.deepcopy(d)
            meta_id = d_copy.pop("id", "")
            operations.append(
                {"index": {"_index": indexName, "_id": meta_id}})
            operations.append(d_copy)

        res = []
        for _ in range(ATTEMPT_TIME):
            try:
                res = []
                self._ensure_index(indexName)
                r = self.es.bulk(index=(indexName), operations=operations,
                                 refresh=False, timeout="60s")
                if re.search(r"False", str(r["errors"]), re.IGNORECASE):
                    return res

                for item in r["items"]:
                    for action in ["create", "delete", "index", "update"]:
                        if action in item and "error" in item[action]:
                            res.append(str(item[action]["_id"]) + ":" + str(item[action]["error"]))
                return res
            except Exception as e:
                res.append(str(e))
                logger.warning("ESConnection.insert got exception: " + str(e))
                if re.search(r"(Timeout|time out)", str(e), re.IGNORECASE):
                    time.sleep(3)
                    continue
                return res
        return res

    def replace_document(self, documents: list[dict], index_name: str, doc_id: str) -> dict:
        """Write new chunks first, then delete only the superseded snapshot IDs.

        A failed write leaves old chunks intact. This is not an atomic ES swap;
        readers can briefly see both versions, and a retry reconciles partial writes.
        """
        if not documents or any(d.get("doc_id") != doc_id or d.get("kb_id") != index_name for d in documents):
            raise ValueError("Replacement requires nonempty chunks from one document and index")
        self._ensure_index(index_name)
        self.es.indices.refresh(index=index_name)
        previous_ids = {hit["_id"] for hit in scan(
            self.es, index=index_name,
            query={"query": {"term": {"doc_id": doc_id}}, "_source": False},
        )}
        for offset in range(0, len(documents), 100):
            errors = self.insert(documents[offset:offset + 100], index_name)
            if errors:
                raise RuntimeError(f"ES replacement failed; old chunks retained: {errors}")
        self.es.indices.refresh(index=index_name)
        stale_ids = sorted(previous_ids - {d["id"] for d in documents})
        for offset in range(0, len(stale_ids), 100):
            response = self.es.bulk(operations=[
                {"delete": {"_index": index_name, "_id": identifier}}
                for identifier in stale_ids[offset:offset + 100]
            ], refresh=True)
            if response.get("errors"):
                raise RuntimeError("New chunks saved but stale chunk cleanup failed; retry reindex")
        return {"indexed": len(documents), "removed": len(stale_ids)}
    

    @staticmethod
    def _normalize_index_names(index_names):
        """把逗号分隔的索引名统一转换成列表。"""
        if isinstance(index_names, str):
            index_names = index_names.split(",")
        assert isinstance(index_names, list) and len(index_names) > 0
        return index_names

    @staticmethod
    def _build_filter_query(condition, knowledgebase_ids):
        """把普通字典过滤条件转换成 Elasticsearch bool filter。"""
        assert "_id" not in condition

        bool_query = Q("bool", must=[])
        # 保留原有行为：把知识库范围加入调用方传入的 condition。
        condition["kb_id"] = knowledgebase_ids

        for field, value in condition.items():
            if field == "available_int":
                if value == 0:
                    bool_query.filter.append(
                        Q("range", available_int={"lt": 1})
                    )
                else:
                    bool_query.filter.append(
                        Q(
                            "bool",
                            must_not=Q("range", available_int={"lt": 1}),
                        )
                    )
                continue

            # None、空字符串和空列表不生成过滤条件。
            if not value:
                continue
            if isinstance(value, list):
                bool_query.filter.append(Q("terms", **{field: value}))
            elif isinstance(value, (str, int)):
                bool_query.filter.append(Q("term", **{field: value}))
            else:
                raise Exception(
                    f"Condition `{field}={value}` value type is "
                    f"{type(value)}, expected to be int, str or list."
                )
        return bool_query

    @staticmethod
    def _get_vector_similarity_weight(match_expressions):
        """从 FusionExpr 中读取向量得分所占的权重。"""
        vector_weight = 0.5
        for expression in match_expressions:
            if (
                isinstance(expression, FusionExpr)
                and expression.method == "weighted_sum"
                and "weights" in expression.fusion_params
            ):
                # 当前混合检索固定接收：文本、向量、融合规则三个表达式。
                assert (
                    len(match_expressions) == 3
                    and isinstance(match_expressions[0], MatchTextExpr)
                    and isinstance(match_expressions[1], MatchDenseExpr)
                    and isinstance(match_expressions[2], FusionExpr)
                )
                vector_weight = float(
                    expression.fusion_params["weights"].split(",")[1]
                )
        return vector_weight

    @staticmethod
    def _normalize_minimum_should_match(minimum_should_match):
        """把 0.3 这样的比例转换成 Elasticsearch 使用的 '30%'。"""
        if isinstance(minimum_should_match, float):
            return f"{int(minimum_should_match * 100)}%"
        return minimum_should_match

    def _apply_match_expressions(
        self, search_query, bool_query, match_expressions
    ):
        """把文本和向量搜索参数翻译成 Elasticsearch DSL。"""
        vector_weight = self._get_vector_similarity_weight(match_expressions)

        for expression in match_expressions:
            if isinstance(expression, MatchTextExpr):
                minimum_should_match = self._normalize_minimum_should_match(
                    expression.extra_options.get("minimum_should_match", 0.0)
                )
                bool_query.must.append(
                    Q(
                        "query_string",
                        fields=expression.fields,
                        type="best_fields",
                        query=expression.matching_text,
                        minimum_should_match=minimum_should_match,
                        boost=1,
                    )
                )
                # ES 的 bool 查询负责关键词得分，所以使用剩余权重。
                bool_query.boost = 1.0 - vector_weight

            elif isinstance(expression, MatchDenseExpr):
                similarity = expression.extra_options.get("similarity", 0.0)
                search_query = search_query.knn(
                    expression.vector_column_name,
                    expression.topn,
                    expression.topn * 2,
                    query_vector=list(expression.embedding_data),
                    filter=bool_query.to_dict(),
                    similarity=similarity,
                )

        return search_query

    @staticmethod
    def _apply_rank_features(bool_query, rank_feature):
        """添加 PageRank 或标签等额外加分条件。"""
        if not rank_feature:
            return

        for field, score in rank_feature.items():
            if field != PAGERANK_FLD:
                field = f"{TAG_FLD}.{field}"
            bool_query.should.append(
                Q("rank_feature", field=field, linear={}, boost=score)
            )

    @staticmethod
    def _apply_sorting(search_query, order_by):
        """把 OrderByExpr 转换成 Elasticsearch 排序参数。"""
        if not order_by:
            return search_query

        orders = []
        for field, order in order_by.fields:
            direction = "asc" if order == 0 else "desc"
            if field in ["page_num_int", "top_int"]:
                order_info = {
                    "order": direction,
                    "unmapped_type": "float",
                    "mode": "avg",
                    "numeric_type": "double",
                }
            elif field.endswith("_int") or field.endswith("_flt"):
                order_info = {
                    "order": direction,
                    "unmapped_type": "float",
                }
            else:
                order_info = {
                    "order": direction,
                    "unmapped_type": "text",
                }
            orders.append({field: order_info})
        return search_query.sort(*orders)

    @staticmethod
    def _apply_highlights_and_aggregations(
        search_query, highlight_fields, aggregation_fields
    ):
        """添加搜索结果高亮和分组统计。"""
        for field in highlight_fields:
            search_query = search_query.highlight(field)

        for field in aggregation_fields:
            search_query.aggs.bucket(
                f"aggs_{field}", "terms", field=field, size=1000000
            )
        return search_query

    def _build_search_body(
        self,
        condition,
        match_expressions,
        order_by,
        offset,
        limit,
        knowledgebase_ids,
        highlight_fields,
        aggregation_fields,
        rank_feature,
    ):
        """把所有搜索参数组装成最终发送给 Elasticsearch 的 JSON。"""
        bool_query = self._build_filter_query(
            condition, knowledgebase_ids
        )
        search_query = self._apply_match_expressions(
            Search(), bool_query, match_expressions
        )

        # 保留原顺序：KNN 的 filter 构造完之后，再添加额外加分条件。
        self._apply_rank_features(bool_query, rank_feature)
        dense_only = bool(match_expressions) and all(
            isinstance(expression, MatchDenseExpr)
            for expression in match_expressions
        )
        # A KNN-only branch already carries metadata constraints in knn.filter.
        # Adding the same bool query at the top level would turn it into a
        # disjunctive hybrid request and pollute the independent vector rank.
        if not dense_only:
            search_query = search_query.query(bool_query)
        search_query = self._apply_highlights_and_aggregations(
            search_query, highlight_fields, aggregation_fields
        )
        search_query = self._apply_sorting(search_query, order_by)

        if limit > 0:
            search_query = search_query[offset:offset + limit]
        return search_query.to_dict()

    def _execute_search(self, index_names, query_body):
        """真正向 Elasticsearch 发送请求；超时时按原规则重试。"""
        logger.debug(
            f"ESConnection.search {index_names} query: "
            + json.dumps(query_body)
        )

        for _ in range(ATTEMPT_TIME):
            try:
                response = self.es.search(
                    index=index_names,
                    body=query_body,
                    timeout="600s",
                    track_total_hits=True,
                    _source=True,
                )
                if str(response.get("timed_out", "")).lower() == "true":
                    raise Exception("Es Timeout.")
                logger.debug(
                    f"ESConnection.search {index_names} res: "
                    + str(response)
                )
                return response
            except Exception as error:
                logger.exception(
                    f"ESConnection.search {index_names} query: "
                    + str(query_body)
                )
                if str(error).find("Timeout") > 0:
                    continue
                raise error

        logger.error("ESConnection.search timeout for 3 times!")
        raise Exception("ESConnection.search timeout.")

    def search(
        self,
        selectFields: list[str],
        highlightFields: list[str],
        condition: dict,
        matchExprs: list[MatchExpr],
        orderBy: OrderByExpr,
        offset: int,
        limit: int,
        indexNames: str | list[str],
        knowledgebaseIds: list[str],
        aggFields: list[str] = [],
        rank_feature: dict | None = None,
    ):
        """组装并执行 Elasticsearch 搜索。"""
        # 原实现没有用 selectFields 限制 _source，这里暂时保持该行为。
        _ = selectFields
        index_names = self._normalize_index_names(indexNames)
        query_body = self._build_search_body(
            condition=condition,
            match_expressions=matchExprs,
            order_by=orderBy,
            offset=offset,
            limit=limit,
            knowledgebase_ids=knowledgebaseIds,
            highlight_fields=highlightFields,
            aggregation_fields=aggFields,
            rank_feature=rank_feature,
        )

        return self._execute_search(index_names, query_body)

    def delete(self, condition: dict, indexName: str, knowledgebaseId: str) -> int:
        """
        删除符合条件的文档
        
        Args:
            condition: 删除条件
            indexName: 索引名称
            knowledgebaseId: 知识库ID
            
        Returns:
            删除的文档数量
        """
        try:
            # 构建删除查询
            query = {
                "query": {
                    "bool": {
                        "must": []
                    }
                }
            }
            
            # 添加知识库ID条件
            if knowledgebaseId:
                query["query"]["bool"]["must"].append({"term": {"kb_id": knowledgebaseId}})
            
            # 添加其他条件
            for field, value in condition.items():
                if isinstance(value, list):
                    query["query"]["bool"]["must"].append({"terms": {field: value}})
                elif isinstance(value, str) and value.startswith("*") and value.endswith("*"):
                    # 通配符查询（两端都有*）
                    query["query"]["bool"]["must"].append({"wildcard": {field: value}})
                elif isinstance(value, str) and (value.startswith("*") or value.endswith("*")):
                    # 通配符查询（一端有*）
                    query["query"]["bool"]["must"].append({"wildcard": {field: value}})
                else:
                    # 精确匹配
                    query["query"]["bool"]["must"].append({"term": {field: value}})
            
            # 打印调试信息
            print(f"ES 删除查询: {json.dumps(query, ensure_ascii=False, indent=2)}")
            print(f"索引名: {indexName}")
            
            # 执行删除
            response = self.es.delete_by_query(
                index=indexName,
                body=query,
                refresh=True
            )
            
            print(f"ES 删除响应: {response}")
            
            return response["deleted"]
            
        except Exception as e:
            logger.error(f"Failed to delete documents: {str(e)}")
            print(f"ES 删除失败: {str(e)}")
            return 0
