"""实体标注必须保留归属不确定性和各来源角色。"""

from tests.evidence_helpers import make_evidence_record, make_tool_result

import json
import pytest

from service.core.entities.attribution import (
    EntityRegistry, annotate_chunk, annotate_document, get_entity_registry,
)
from service.core.ingestion import pipeline as file_parse
from service.core.persistence import evidence_store
from service.core.retrieval.pipeline import format_retrieved_chunk
from service.core.evidence.sufficiency import _prepare_evidence
from service.core.rag.nlp.search_v2 import Dealer
from service.core.rag.utils.es_conn import ESConnection
from service.core.evidence.adapter import collection_from_records, evidence_from_record, reference_records
from agent.loop import collect_tool_evidence
from service.core.evidence.adapter import reference_records
from agent.schemas import ToolResult


pytestmark = pytest.mark.unit


@pytest.fixture
def registry():
    return EntityRegistry({"version": "test", "entities": [
        {"entity_id": "tencent", "canonical_name": "腾讯控股", "kind": "company",
         "aliases": ["腾讯", "Tencent", "Tencent Holdings Limited"], "stock_codes": ["0700.HK"]},
        {"entity_id": "alibaba", "canonical_name": "阿里巴巴", "kind": "company",
         "aliases": ["Alibaba"], "stock_codes": ["9988.HK"]},
        {"entity_id": "tencent-sub", "canonical_name": "腾讯科技有限公司", "kind": "company",
         "aliases": [], "stock_codes": []},
        {"entity_id": "broker", "canonical_name": "甲证券", "kind": "publisher",
         "aliases": [], "stock_codes": []},
    ]})


def _row(headers, cells, row_index=2, binding="bound"):
    return {"content_with_weight": "资料", "content_ltks": "资料", "content_sm_ltks": "资料",
            "title_tks": "report", "evidence_type_kwd": "table", "table_binding_kwd": binding,
            "table_cells_slice_obj": {"binding_status": binding, "rows": [
                {"row_index": row_index, "headers": headers, "cells": cells},
            ]}}


def test_reviewed_aliases_codes_and_original_quotes(registry):
    text = "Ｔｅｎｃｅｎｔ Holdings Limited、腾 讯和0700.HK属于同一查询对象。"
    mentions = registry.mentions(text)
    assert registry.resolved_ids(mentions) == ["tencent"]
    assert all(text[mention["start"]:mention["end"]] == mention["quote"] for mention in mentions)
    assert registry.mentions("营业收入为603193百万元，年份2023。") == []
    assert registry.mentions("xTencentish") == []
    assert registry.resolved_ids(registry.mentions("The Tencent revenue increased.")) == ["tencent"]


def test_subsidiary_is_separate_and_unregistered_longer_name_not_merged(registry):
    assert registry.resolved_ids(registry.mentions("腾讯科技有限公司")) == ["tencent-sub"]
    assert registry.mentions("腾讯金融科技有限公司") == []
    assert annotate_document([], "腾讯金融科技有限公司.pdf", registry)["document_subject_entity_ids"] == []
    assert annotate_document([], "腾讯音乐.pdf", registry)["document_subject_entity_ids"] == []
    assert annotate_document([], "腾讯广东分公司.pdf", registry)["document_subject_entity_ids"] == []
    mixed = annotate_document([], "腾讯音乐与阿里巴巴.pdf", registry)
    assert mixed["document_subject_status_kwd"] == "ambiguous"
    assert mixed["document_subject_entity_ids"] == []


def test_source_publisher_topic_and_mentioned_company_are_separate(registry):
    item = {"content_with_weight": "发布方：甲证券\n阿里巴巴营业收入为80亿元。"}
    document = annotate_document([item], "腾讯2023年报告.pdf", registry)
    annotation = annotate_chunk(item, document, registry)
    assert annotation["document_subject_entity_ids"] == ["tencent"]
    assert annotation["publisher_entity_ids"] == ["broker"]
    assert annotation["mentioned_entity_ids"] == ["alibaba", "broker"]
    assert annotation["candidate_subject_entity_ids"] == ["alibaba"]
    assert annotation["subject_entity_ids"] == []
    assert annotation["entity_attribution_status_kwd"] == "mentions_only"


def test_company_can_publish_its_own_report(registry):
    document = annotate_document([{"content_with_weight": "发布方：腾讯\n年度报告"}], "腾讯年报.pdf", registry)
    assert document["publisher_entity_ids"] == ["tencent"]


def test_research_credit_is_candidate_and_plain_mention_is_not_publisher(registry):
    credit = annotate_document([{"content_with_weight": "资料来源：甲证券研究所预测"}], "腾讯.pdf", registry)
    assert credit["publisher_entity_ids"] == ["broker"]
    assert credit["publisher_status_kwd"] == "candidate"
    plain = annotate_document([{"content_with_weight": "甲证券的收入同比增加"}], "腾讯.pdf", registry)
    assert plain["publisher_entity_ids"] == []


def test_anonymous_document_needs_explicit_topic_and_never_body_frequency(registry):
    repeated = {"content_with_weight": "腾讯收入增长。" * 10}
    assert annotate_document([repeated], "report.pdf", registry)["document_subject_entity_ids"] == []
    explicit = annotate_document([{"content_with_weight": "研究对象：Tencent"}], "report.pdf", registry)
    assert explicit["document_subject_entity_ids"] == ["tencent"]
    unknown = annotate_document([{"content_with_weight": "研究对象：新公司"}], "report.pdf", registry)
    assert unknown["document_subject_entity_ids"] == []
    assert unknown["unresolved_entity_names_kwd"] == ["新公司"]


def test_peer_table_does_not_relabel_document(registry):
    row = _row(["公司", "收入"], ["阿里巴巴", "80"])
    row["table_semantic_text_with_weight"] = "公司名称：阿里巴巴\n收入：80"
    document = annotate_document([row], "腾讯.pdf", registry)
    annotation = annotate_chunk(row, document, registry)
    assert document["document_subject_entity_ids"] == ["tencent"]
    assert annotation["subject_entity_ids"] == ["alibaba"]
    assert annotation["entity_attribution_status_kwd"] == "explicit_table"
    assert annotation["entity_attribution_obj"]["table_bindings"] == [{
        "source": "table_company_column", "row_index": 2, "entity_id": "alibaba",
        "company_column_indices": [0], "company_quotes": ["阿里巴巴"], "value_column_indices": [1],
    }]


def test_comparison_headers_keep_value_columns(registry):
    row = _row(["指标", "腾讯 / 2023", "阿里巴巴 / 2023"], ["营业收入", "120", "80"])
    annotation = annotate_chunk(row, annotate_document([row], "比较.pdf", registry), registry)
    assert annotation["subject_entity_ids"] == ["alibaba", "tencent"]
    bindings = annotation["entity_attribution_obj"]["table_bindings"]
    assert [(value["entity_id"], value["value_column_indices"]) for value in bindings] == [
        ("tencent", [1]), ("alibaba", [2]),
    ]


@pytest.mark.parametrize("row", [
    _row(["公司", "收入"], ["未登记公司", "80"]),
    _row(["公司", "收入"], ["", "80"]),
    _row(["公司", "阿里巴巴收入"], ["腾讯", "80"]),
    _row(["公司", "收入"], ["腾讯", "80"], binding="unbound"),
])
def test_unresolved_conflicting_and_unbound_rows_do_not_inherit_filename(registry, row):
    annotation = annotate_chunk(row, annotate_document([row], "腾讯.pdf", registry), registry)
    assert annotation["subject_entity_ids"] == []
    assert annotation["candidate_subject_entity_ids"] == []
    assert annotation["entity_attribution_obj"]["table_bindings"] == []


def test_metric_table_context_is_candidate_and_not_verified(registry):
    row = _row(["指标", "2023"], ["收入", "120"])
    annotation = annotate_chunk(row, annotate_document([row], "腾讯.pdf", registry), registry)
    assert annotation["subject_entity_ids"] == []
    assert annotation["candidate_subject_entity_ids"] == ["tencent"]
    assert annotation["entity_attribution_status_kwd"] == "document_context"
    assert annotation["entity_attribution_obj"]["fact_verification_status"] == "not_verified"


def test_ambiguous_alias_is_not_silently_chosen():
    registry = EntityRegistry({"version": "1", "entities": [
        {"entity_id": "a", "canonical_name": "甲公司", "kind": "company", "aliases": ["华科"]},
        {"entity_id": "b", "canonical_name": "乙公司", "kind": "company", "aliases": ["华科"]},
    ]})
    document = annotate_document([], "华科.pdf", registry)
    assert document["document_subject_status_kwd"] == "ambiguous"
    assert document["document_subject_entity_ids"] == []
    assert document["entity_attribution_obj"]["document_subject_evidence"][0]["entity_ids"] == ["a", "b"]


def test_new_annotations_survive_ingestion_parent_storage_and_retrieval(monkeypatch, tmp_path, registry):
    row = _row(["公司", "收入"], ["阿里巴巴", "80"])
    row["table_id_kwd"] = "a" * 64
    row["_table_object"] = {"table_id": "a" * 64, "cells_json": {
        "binding_status": "bound", "data_rows": row["table_cells_slice_obj"]["rows"],
    }, "provenance": {}}
    original = json.dumps(row, ensure_ascii=False, sort_keys=True)
    monkeypatch.setattr(file_parse, "get_entity_registry", lambda: registry)
    monkeypatch.setattr(file_parse, "batch_generate_embeddings", lambda texts: [[0.1] * 512 for _ in texts])
    monkeypatch.setattr(evidence_store, "get_project_base_directory", lambda: str(tmp_path))
    indexed = file_parse.process_items([row], "腾讯.pdf", "tenant-1")[0]
    assert indexed["subject_entity_ids"] == ["alibaba"]
    assert indexed["document_subject_entity_ids"] == ["tencent"]
    assert json.dumps(row, ensure_ascii=False, sort_keys=True) == original
    parent = evidence_store.load_table_object(indexed["table_object_ref_kwd"])
    assert parent["entity_attribution"]["rows"][0]["subject_entity_ids"] == ["alibaba"]
    assert parent["entity_attribution"]["document"]["document_subject_entity_ids"] == ["tencent"]
    # Elasticsearch 必须保留字典结构，否则来源证明会退化为 Python 字符串。
    monkeypatch.setenv("ELASTIC_PASSWORD", "test-only-password")
    fields = ESConnection().getFields({"hits": {"hits": [{"_id": indexed["id"], "_score": 1.0, "_source": indexed}]}}, list(indexed))
    assert isinstance(fields[indexed["id"]]["entity_attribution_obj"], dict)
    source = fields[indexed["id"]]
    chunk = Dealer._build_chunk(indexed["id"], source, 1, 1, {}, {}, {}, {}, {}, "q_512_vec", [])
    formatted = format_retrieved_chunk(chunk, 1)
    tool_source = evidence_from_record(formatted)
    references = reference_records(collect_tool_evidence([make_tool_result(
        tool_name="search_knowledge_base", query="腾讯收入", content="检索结果", sources=[formatted],
    )]))
    assert references[0]["subject_entity_ids"] == ["alibaba"]
    assert references[0]["entity_attribution_obj"] == indexed["entity_attribution_obj"]


def test_default_registry_and_empty_legacy_documents():
    registry = get_entity_registry()
    assert registry.exact("603193.SH") == ["company:cn:603193"]
    document = annotate_document([], "未知资料.pdf", registry)
    assert document["document_subject_status_kwd"] == "unknown"
    annotation = annotate_chunk({"content_with_weight": "收入为100。"}, document, registry)
    assert annotation["entity_attribution_status_kwd"] == "unknown"
    assert annotation["subject_entity_ids"] == []
    assert "entity_attribution_obj" not in format_retrieved_chunk({"content_with_weight": "旧资料"}, 1)


def test_peer_company_short_name_and_code_override_report_subject():
    registry = get_entity_registry()
    row = _row(["代码", "公司简称", "PE 25E"], ["001289.SZ", "龙源电力", "19.6"])
    document = annotate_document([row], "国电电力.pdf", registry)
    annotation = annotate_chunk(row, document, registry)
    assert annotation["document_subject_entity_ids"] == ["company:cn:600795"]
    assert annotation["subject_entity_ids"] == ["company:cn:001289"]
    binding = annotation["entity_attribution_obj"]["table_bindings"][0]
    assert binding["company_column_indices"] == [1]
    assert binding["stock_code_column_indices"] == [0]
    assert binding["value_column_indices"] == [2]
    assert binding["company_quotes"] == ["龙源电力"]


def test_conflicting_stock_code_cannot_confirm_row_owner(registry):
    row = _row(["公司简称", "股票代码", "收入"], ["腾讯", "9988.HK", "80"])
    annotation = annotate_chunk(row, annotate_document([row], "腾讯.pdf", registry), registry)
    assert annotation["entity_attribution_status_kwd"] == "ambiguous"
    assert annotation["subject_entity_ids"] == []
    assert annotation["candidate_subject_entity_ids"] == []


def test_model_evidence_keeps_compact_labels_without_duplicating_audit(registry):
    row = _row(["公司", "收入"], ["腾讯", "120"])
    annotation = annotate_chunk(row, annotate_document([row], "腾讯.pdf", registry), registry)
    chunk = {**row, **annotation, "chunk_id": "row"}
    evidence = _prepare_evidence(collection_from_records([chunk]))[0].model_record(1)
    assert evidence["subject_entity_ids"] == ["tencent"]
    assert evidence["entity_attribution_status_kwd"] == "explicit_table"
    assert "entity_attribution_obj" not in evidence
    assert chunk["entity_attribution_obj"]["table_bindings"]
