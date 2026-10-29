"""索引、来源解析器和前端 JSON 与领域证据之间的唯一适配入口。"""

from collections.abc import Iterable, Mapping
from dataclasses import fields
from typing import Any

from domain.evidence import (
    Evidence, EvidenceCollection, EvidenceSource, EvidenceLocation, EvidenceStructure,
    RetrievalTrace, fingerprint, plain,
    EvidenceTable, EvidenceFigure,
)
from domain.assessment import EvidenceAssessment
from service.core.evidence.metadata import evidence_metadata, primary_table_row, without_entity_annotations
from service.core.evidence.rows import table_row_from_record

RANKING_FIELDS = (
    "similarity", "rerank_score", "rrf_score", "vector_similarity", "term_similarity",
    "retrieval_ranks", "retrieval_scores", "rrf_contributions", "constraint_compatibility",
    "final_ranking", "sequential_subqueries", "parallel_subqueries",
)


def _table_from_record(metadata: dict[str, Any]) -> EvidenceTable:
    if not metadata:
        return EvidenceTable()
    context = metadata.get("table_context_obj") or {}
    context = context if isinstance(context, Mapping) else {}
    sliced = metadata.get("table_cells_slice_obj") or {}
    sliced = sliced if isinstance(sliced, Mapping) else {}
    rows = context.get("rows") or sliced.get("rows")
    if not rows:
        rows = [primary_table_row(metadata)]
    rows = rows if isinstance(rows, (list, tuple)) else ()
    return EvidenceTable(
        str(metadata.get("table_binding_kwd") or "unbound"),
        str(context.get("binding_source") or metadata.get("table_binding_source_kwd") or "deterministic"),
        not context or bool(context.get("reliable")), str(context.get("caption") or sliced.get("caption") or ""),
        tuple(table_row_from_record(row) for row in rows if isinstance(row, Mapping)), metadata,
    )


def evidence_from_record(record: dict[str, Any], *, source_kind="knowledge_base") -> Evidence:
    """索引记录只在边界解析一次；已有筛选正文和状态不能冒充来源数据。"""
    kind = record.get("source_type") or source_kind
    text = record.get("content_with_weight", record.get("content", ""))
    if not isinstance(text, str):
        raise ValueError("证据正文必须为字符串")
    document_id = str(record.get("document_id") or record.get("doc_id") or "")
    name = str(record.get("document_name") or record.get("docnm_kwd") or "未知文档").replace("\\", "/").split("/")[-1]
    source = EvidenceSource(kind, document_id, name, record.get("url"),
                            str(record.get("kb_id") or ""), str(record.get("image_id") or ""),
                            tuple(record.get("web_sources") or ()))
    location = EvidenceLocation(
        positions=tuple(record.get("positions") or ()), page_number=record.get("page_num_int"),
        table_id=record.get("table_id_kwd"), row=record.get("table_row_int"),
        row_start=record.get("table_row_start_int"), row_end=record.get("table_row_end_int"),
    )
    attributes = evidence_metadata(record)
    table = {key: value for key, value in attributes.items() if key.startswith("table_")}
    figure = {key: value for key, value in attributes.items() if key.startswith("visual_") or key == "image_ref_kwd"}
    coordinates = {key: attributes[key] for key in ("page_num_int", "position_int", "top_int") if key in attributes}
    structure = EvidenceStructure(
        kind=str(attributes.get("evidence_type_kwd") or "text"),
        table=_table_from_record(table),
        figure=EvidenceFigure(str(figure.get("visual_status_kwd") or "unextracted"), figure),
        locator=str(attributes.get("locator_kwd") or ""),
        parser_version=str(attributes.get("parser_version_kwd") or ""),
        coordinates=coordinates,
    )
    candidate = record.get("table_candidate_obj") or {}
    identifier = str(candidate.get("candidate_id") or record.get("chunk_id") or record.get("source_id") or "")
    if not identifier:
        identifier = f"{kind}-{fingerprint((source, location, text))}"
    return Evidence(identifier, text, source, location, structure)


def collection_from_records(records: Iterable[dict[str, Any]], *, query="", source_kind="knowledge_base") -> EvidenceCollection:
    result = EvidenceCollection()
    for record in records:
        result.add(evidence_from_record(record, source_kind=source_kind),
                   RetrievalTrace(query, {key: record[key] for key in RANKING_FIELDS if key in record}))
    return result


def structure_record(structure: EvidenceStructure) -> dict[str, Any]:
    """在模型及 API 边界导出结构；纯领域对象也能提供完整表格信息。"""
    result = {
        **plain(structure.table.metadata), **plain(structure.figure.metadata),
        **plain(structure.coordinates),
    }
    if structure.kind != "text":
        result["evidence_type_kwd"] = structure.kind
    if structure.kind == "table":
        table = structure.table
        result["table_binding_kwd"] = table.binding
        if not table.metadata:
            result["table_context_obj"] = {
                "binding_source": table.binding_source, "reliable": table.reliable, "caption": table.caption,
                "rows": [{"headers": list(row.headers), "cells": list(row.cells), "row_index": row.index,
                          "column_indices": list(row.columns), "context": row.context, "preamble": row.preamble,
                          "source_row": {"headers": list(row.original_headers), "cells": list(row.original_cells)}}
                         for row in table.rows],
            }
    if structure.kind == "figure":
        result["visual_status_kwd"] = structure.figure.status
    if structure.locator:
        result["locator_kwd"] = structure.locator
    if structure.parser_version:
        result["parser_version_kwd"] = structure.parser_version
    # 直接构造的领域对象也可能携带旧表格元数据，导出边界统一清理。
    return without_entity_annotations(result)


def source_record(evidence: Evidence) -> dict[str, Any]:
    """在 JSON 输出边界导出原文及出处，不携带本次查询判定。"""
    result = {
        **structure_record(evidence.structure),
        "chunk_id": evidence.id, "document_id": evidence.source.document_id,
        "document_name": evidence.source.name, "doc_id": evidence.source.document_id,
        "docnm_kwd": evidence.source.name, "content_with_weight": evidence.text,
        "positions": plain(evidence.location.positions), "source_type": evidence.source.kind,
        "kb_id": evidence.source.knowledge_base_id, "image_id": evidence.source.image_id,
    }
    if evidence.source.url is not None:
        result["url"] = evidence.source.url
    if evidence.source.links:
        result["web_sources"] = plain(evidence.source.links)
    for name, value in (
        ("page_num_int", evidence.location.page_number),
        ("table_id_kwd", evidence.location.table_id),
        ("table_row_int", evidence.location.row),
        ("table_row_start_int", evidence.location.row_start),
        ("table_row_end_int", evidence.location.row_end),
    ):
        if value is not None:
            result[name] = plain(value)
    return result


def source_records(evidence: EvidenceCollection) -> list[dict[str, Any]]:
    return [source_record(item) for item in evidence]


def reference_record(evidence: Evidence, number: int, traces=()) -> dict[str, Any]:
    """输出前端协议，领域对象不使用前端或 Elasticsearch 专属字段名。"""
    raw = source_record(evidence)
    raw.pop("doc_id")
    raw.pop("docnm_kwd")
    result = {
        "id": number, "rank": number, **raw,
        "similarity": None, "rerank_score": None, "rrf_score": None,
        "vector_similarity": None, "term_similarity": None,
        "retrieval_ranks": {}, "retrieval_scores": {}, "rrf_contributions": {},
        "kb_id": evidence.source.knowledge_base_id, "image_id": evidence.source.image_id,
    }
    if traces:
        result.update(plain(traces[0].scores))
    for name in ("similarity", "rerank_score", "rrf_score", "vector_similarity", "term_similarity"):
        try:
            result[name] = float(result[name]) if result[name] is not None else None
        except (TypeError, ValueError):
            result[name] = None
    if evidence.source.kind == "web":
        result.update(url=evidence.source.url, web_sources=plain(evidence.source.links))
    return result


def reference_records(evidence: EvidenceCollection) -> list[dict[str, Any]]:
    return [reference_record(item, index, evidence.traces(item.id)) for index, item in enumerate(evidence, 1)]


def assessment_from_record(record: dict[str, Any]) -> EvidenceAssessment:
    """读取历史报告内容，不恢复可复用的核验范围；当前来源须重新核验。"""
    diagnostics = dict(record.get("diagnostics") or {})
    diagnostics.update({key: record[key] for key in ("outcome", "failure_kind", "failure", "attempts") if key in record})
    names = {item.name for item in fields(EvidenceAssessment) if item.init} - {
        "diagnostics", "query_version", "evidence_version", "policy_version", "require_semantic",
    }
    data = {key: value for key, value in record.items() if key in names}
    data.setdefault("sufficient", False)
    data.setdefault("reason", "证据不足")
    # 历史字典不能自报查询版本和核验模式。
    return EvidenceAssessment(**data, diagnostics=diagnostics)
