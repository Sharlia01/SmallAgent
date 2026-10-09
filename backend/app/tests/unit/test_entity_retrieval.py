"""所有独立检索分支都必须携带相同的公司范围。"""

from copy import deepcopy

import pytest

from service.core import retrieval, entity_retrieval
from service.core.entity_attribution import EntityRegistry
from service.core.entity_retrieval import resolve_company_constraint, COMPANY_FILTER_KEY
from service.core.query_expansion import QueryExpansionResult
from service.core.rag.nlp.search_v2 import Dealer
from service.core.rag.utils.doc_store_conn import MatchTextExpr, MatchDenseExpr, OrderByExpr
from service.core.rag.utils.es_conn import ESConnection


pytestmark = pytest.mark.unit


@pytest.fixture
def registry(monkeypatch):
    registry = EntityRegistry({"version": "test", "entities": [
        {"entity_id": "a", "canonical_name": "甲公司", "kind": "company",
         "aliases": ["Alpha", "同名"], "stock_codes": ["123456.SH"]},
        {"entity_id": "b", "canonical_name": "乙公司", "kind": "company",
         "aliases": ["Beta", "同名"], "stock_codes": []},
        {"entity_id": "sub", "canonical_name": "甲公司科技有限公司", "kind": "company",
         "aliases": [], "stock_codes": []},
        {"entity_id": "broker", "canonical_name": "甲证券", "kind": "publisher",
         "aliases": [], "stock_codes": []},
    ]})
    monkeypatch.setattr(entity_retrieval, "get_entity_registry", lambda: registry)
    return registry


@pytest.mark.parametrize("question,ids,status", [
    ("Alpha 2023年营收", ("a",), "resolved"),
    ("123456.SH的营收", ("a",), "resolved"),
    ("123456万元的营收", (), "no_registered_company"),
    ("比较甲公司和乙公司的营收", ("a", "b"), "resolved"),
    ("甲证券对乙公司的预测", ("b",), "resolved"),
    ("同名的营收", (), "ambiguous_company"),
    ("甲公司和同名的营收", (), "ambiguous_company"),
    ("请查询甲公司科技有限公司的营收", ("sub",), "resolved"),
    ("甲公司新能源有限公司的营收", (), "unregistered_company"),
    ("甲公司上海分公司的营收", (), "unregistered_company"),
    ("比较甲公司与未登记能源有限公司", (), "unregistered_company"),
    ("营业收入是什么", (), "no_registered_company"),
])
def test_query_scope_uses_reviewed_identity_and_keeps_uncertainty(registry, question, ids, status):
    scope = resolve_company_constraint(question)
    assert scope.entity_ids == ids
    assert scope.status == status
    assert scope.enabled == bool(ids)
    assert scope.registry_version == registry.version


def test_rewritten_company_cannot_change_original_scope(registry):
    dealer = object.__new__(Dealer)
    request = dealer._build_search_request(
        "甲公司营收", Dealer.RetrievalOptions(doc_ids=["doc-a"], kb_ids=["kb-a"]),
        keyword_question="乙公司 营业收入",
    )
    assert dealer.get_filters(request) == {
        "kb_id": ["kb-a"], "doc_id": ["doc-a"], "available_int": 1,
        COMPANY_FILTER_KEY: ["a"],
    }
    assert dealer.get_filters({"question": "乙公司营收"})[COMPANY_FILTER_KEY] == ["b"]
    assert COMPANY_FILTER_KEY not in dealer.get_filters({"question": "同名营收"})


def test_every_branch_and_keyword_retry_keep_identical_company_scope(registry):
    calls = []

    class Store:
        def search(self, fields, highlights, condition, expressions, order, offset, limit,
                   indexes, kb_ids, **kwargs):
            calls.append((deepcopy(condition), expressions[0]))
            return {"hits": {"total": {"value": 0}, "hits": []}}

        getTotal = staticmethod(lambda response: response["hits"]["total"]["value"])
        getChunkIds = staticmethod(lambda response: [])
        getFields = staticmethod(lambda response, fields: {})

    class Queryer:
        def question(self, text, min_match):
            return MatchTextExpr(["content_ltks"], text, 10,
                                 {"minimum_should_match": min_match}), []

    dealer = object.__new__(Dealer)
    dealer.dataStore = Store()
    dealer.qryr = Queryer()
    dealer._expand_keywords = lambda keywords: set(keywords)
    dealer.get_vector = lambda *args, **kwargs: MatchDenseExpr(
        "q_512_vec", [0.1, 0.2], "float", "cosine", 10, {"similarity": 0.1},
    )
    request = dealer._build_search_request("甲公司2023营收", Dealer.RetrievalOptions(
        kb_ids=["kb-a"], doc_ids=["doc-a"],
    ), "乙公司 营业收入")
    result = dealer.search(request, ["42"], ["kb-a"])

    # 原词、原词放宽、扩展词、扩展词放宽、向量；全部无结果也不取消公司限制。
    assert len(calls) == 5
    assert all(condition == {
        "kb_id": ["kb-a"], "doc_id": ["doc-a"], "available_int": 1,
        COMPANY_FILTER_KEY: ["a"],
    } for condition, _ in calls)
    assert [expression.extra_options["minimum_should_match"] for _, expression in calls[:4]] == [
        0.3, 0.1, 0.3, 0.1,
    ]
    assert isinstance(calls[-1][1], MatchDenseExpr)
    assert result.ids == []


def test_keyword_and_knn_use_same_prefilter_with_existing_scopes(registry, monkeypatch):
    monkeypatch.setenv("ELASTIC_PASSWORD", "test-only-password")
    connection = ESConnection()
    condition = {COMPANY_FILTER_KEY: ["a"], "doc_id": ["doc-a"], "kb_id": ["kb-a"], "available_int": 1}
    saved_condition = deepcopy(condition)
    expressions = [
        MatchTextExpr(["content_ltks"], "营收", 10, {"minimum_should_match": 0.3}),
        MatchDenseExpr("q_512_vec", [0.1, 0.2], "float", "cosine", 10, {"similarity": 0.1}),
    ]
    bodies = [connection._build_search_body(
        condition=condition, match_expressions=[expression], order_by=OrderByExpr(),
        offset=0, limit=10, knowledgebase_ids=None, highlight_fields=[], aggregation_fields=[],
        rank_feature=None,
    ) for expression in expressions]
    assert condition == saved_condition
    lexical_filter = bodies[0]["query"]["bool"]["filter"]
    assert lexical_filter == bodies[1]["knn"]["filter"]["bool"]["filter"]
    assert "query" not in bodies[1]  # 向量候选在搜索前筛选，不在搜完后删除。
    assert {"terms": {"doc_id": ["doc-a"]}} in lexical_filter
    assert {"terms": {"kb_id": ["kb-a"]}} in lexical_filter
    company_filter = lexical_filter[0]["bool"]
    assert company_filter["minimum_should_match"] == 1
    assert company_filter["should"][0] == {"terms": {"subject_entity_ids": ["a"]}}
    assert company_filter["should"][1]["bool"]["must_not"] == [{"exists": {"field": "subject_entity_ids"}}]


def test_parallel_scopes_are_separate_and_anonymous_repair_keeps_parent(registry, monkeypatch):
    calls = []

    class StoreDealer:
        def retrieval(self, **kwargs):
            calls.append(kwargs)
            return {"chunks": []}

    # 即使扩展时加进乙公司，原本甲公司的子查询仍用甲公司的限制。
    monkeypatch.setattr(retrieval, "expand_query", lambda *args, **kwargs:
                        QueryExpansionResult.noop("乙公司营业收入", reason="test"))
    options = Dealer.RetrievalOptions()
    dealer = retrieval._BudgetedDealer(StoreDealer(), transform_nodes=True,
                                      company_constraint=resolve_company_constraint("比较甲公司和乙公司"))
    for question in ["甲公司的营收", "乙公司的营收", "2023年净利润"]:
        dealer.retrieval(question=question, tenant_ids="42", options=options)
    assert [call["options"].company_constraint.entity_ids for call in calls] == [
        ("a",), ("b",), ("a", "b"),
    ]
    assert calls[-1]["options"].company_constraint.source == "inherited_company_set"
    assert options.company_constraint is None  # 不修改并行分支共用的配置。
    dealer.company_constraint = resolve_company_constraint("甲公司的营收")
    dealer.retrieval(question="2023年营业收入", tenant_ids="42", options=options)
    assert calls[-1]["options"].company_constraint.entity_ids == ("a",)
    dealer.retrieval(question="未登记能源有限公司营收", tenant_ids="42", options=options)
    assert not calls[-1]["options"].company_constraint.enabled


def test_company_diagnostics_survive_chat_result_format(registry):
    diagnostic = resolve_company_constraint("甲公司2023营收").to_dict()
    _, diagnostics = retrieval.format_retrieval_results({"company_constraint": diagnostic, "chunks": []})
    assert diagnostics["company_constraint"] == diagnostic
    assert diagnostic["untagged_policy"] == "exclude_when_enabled"
    assert diagnostic["candidate_ownership"] == "not_verified"
