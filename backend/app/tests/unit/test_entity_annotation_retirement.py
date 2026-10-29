"""公司归属交给统一核验，旧标签不能干扰召回或污染原始证据。"""

from copy import deepcopy

import numpy as np
import pytest

from service.core.evidence.adapter import evidence_from_record, source_record
from service.core.evidence.sufficiency import VerificationItem
from service.core.ingestion import pipeline as ingestion
from service.core.persistence import evidence_store
from service.core.rag.nlp.search_v2 import Dealer
from service.core.retrieval.entities import CompanyHints
from service.core.retrieval.table_context import expand_table_chunk_context


pytestmark = pytest.mark.unit


def _legacy_row():
    return {
        "row_index": 3,
        "context": "可比公司",
        "preamble": "单位：亿元",
        "headers": ["公司", "证券代码", "2025年收入"],
        "cells": ["华能国际", "600011", "100"],
        "entity_attribution": {"subject_entity_ids": ["wrong-company"]},
    }


def test_ingestion_preserves_company_columns_without_saving_ownership(monkeypatch, tmp_path):
    monkeypatch.setattr(ingestion, "batch_generate_embeddings", lambda texts: [[0.1] * 512 for _ in texts])
    monkeypatch.setattr(evidence_store, "get_project_base_directory", lambda: str(tmp_path))
    row = _legacy_row()
    table_id = "a" * 64
    item = {
        "content_with_weight": "华能国际 | 600011 | 100",
        "content_ltks": "华能国际 600011 100",
        "content_sm_ltks": "华能国际 600011 100",
        "title_tks": "国电电力研报",
        "evidence_type_kwd": "table",
        "table_id_kwd": table_id,
        "table_binding_kwd": "bound",
        "table_cells_slice_obj": {"rows": [row]},
        "subject_entity_ids": ["wrong-company"],
        "_table_object": {
            "table_id": table_id,
            "cells_json": {"binding_status": "bound", "data_rows": [row]},
            "entity_attribution": {"rows": [{"subject_entity_ids": ["wrong-company"]}]},
        },
    }
    original = deepcopy(item)

    chunk = ingestion.process_items([item], "国电电力研报.pdf", "kb-1")[0]
    parent = evidence_store.load_table_object(chunk["table_object_ref_kwd"])

    expected_row = {key: value for key, value in row.items() if key != "entity_attribution"}
    assert chunk["content_with_weight"] == item["content_with_weight"]
    assert chunk["docnm_kwd"] == "国电电力研报.pdf"
    assert chunk["table_cells_slice_obj"]["rows"] == [expected_row]
    assert parent["cells_json"]["data_rows"] == [expected_row]
    assert "subject_entity_ids" not in chunk
    assert "entity_attribution" not in parent
    assert item == original


def test_legacy_parent_and_chunk_reach_verifier_with_raw_structure_only(monkeypatch):
    monkeypatch.setenv("RAG_TABLE_CONTEXT_EXPANSION_ENABLED", "true")
    row = _legacy_row()
    original = deepcopy(row)
    chunk = {
        "chunk_id": "legacy-table-row",
        "doc_id": "d-1",
        "docnm_kwd": "国电电力研报.pdf",
        "content_with_weight": "旧片段",
        "evidence_type_kwd": "table",
        "table_binding_kwd": "bound",
        "table_id_kwd": "table-1",
        "table_row_start_int": 3,
        "subject_entity_ids": ["wrong-company"],
        "table_cells_slice_obj": {"rows": [row]},
    }
    parent = {
        "table_id": "table-1",
        "cells_json": {"binding_status": "bound", "data_rows": [row]},
        "entity_attribution": {"rows": [{"subject_entity_ids": ["wrong-company"]}]},
    }

    expanded = expand_table_chunk_context(chunk, "华能国际的收入是多少", table_object=parent)
    evidence = evidence_from_record(expanded)
    model_input = VerificationItem(evidence, evidence.text).model_record(1)
    output = source_record(evidence)

    assert "华能国际" in model_input["content"]
    assert "600011" in model_input["content"]
    for record in (model_input, output):
        exported = record["table_context_obj"]["rows"][0]
        assert exported["row_index"] == 3
        assert exported["headers"] == row["headers"]
        assert exported["cells"] == row["cells"]
        assert exported["column_indices"] == [0, 1, 2]
        assert exported["source_row"]["cells"] == row["cells"]
        assert "entity_attribution" not in str(record)
        assert "subject_entity_ids" not in str(record)
    assert "entity_attribution" not in expanded["table_context_obj"]["rows"][0]["source_row"]
    assert row == original


def test_peer_mentions_and_missing_or_conflicting_labels_do_not_drop_candidates():
    texts = {
        "peer-mention": "本期收入100亿元，与华能国际相比增速更快。",
        "untagged": "本期营业收入100亿元。",
        "wrong-tag": "公司营业收入100亿元。",
    }
    fields = {
        key: {"content_ltks": text, "content_with_weight": text,
              "docnm_kwd": "国电电力2025年报告.pdf", "doc_id": "d-1", "kb_id": "kb-1"}
        for key, text in texts.items()
    }
    fields["peer-mention"]["candidate_subject_entity_ids"] = ["华能国际"]
    fields["wrong-tag"]["subject_entity_ids"] = ["华能国际"]
    result = Dealer.SearchResult(total=3, candidate_ids=list(texts), candidate_fields=fields)
    options = Dealer.RetrievalOptions(company_hints=CompanyHints(
        entity_names=("国电电力",), source="model_understanding",
    ))
    dealer = object.__new__(Dealer)
    dealer.rerank_by_model = lambda search_result, question: np.array([0.9, 0.8, 0.7])

    ranked = dealer._rerank_results(result, "国电电力2025年收入是多少", options)
    retrieved = dealer._build_retrieval_result(result, ranked, options)

    assert {chunk["chunk_id"] for chunk in retrieved["chunks"]} == set(texts)
    assert {chunk["content_with_weight"] for chunk in retrieved["chunks"]} == set(texts.values())
    assert all("subject_entity_ids" not in str(chunk) for chunk in retrieved["chunks"])
