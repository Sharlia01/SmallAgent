"""索引、来源解析器和前端 JSON 与领域证据之间的唯一适配入口。"""

from collections.abc import Iterable, Mapping
from dataclasses import fields, replace
from typing import Any

from domain.evidence import (
    Evidence, EvidenceCollection, EvidenceSource, EvidenceLocation, EvidenceStructure,
    RetrievalTrace, fingerprint, plain,
    EvidenceTableRow, EvidenceTable, EvidenceFigure, EvidenceAttribution,
)
from domain.assessment import EvidenceAssessment
from domain.facts import FinancialFactAssessment, FactRequirement, FinancialFact
from service.core.evidence.metadata import evidence_metadata
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
        rows = [{"headers": metadata.get("table_headers_kwd"), "cells": metadata.get("table_cells_kwd"),
                 "row_index": metadata.get("table_row_int", metadata.get("table_row_start_int"))}]
    rows = rows if isinstance(rows, (list, tuple)) else ()
    return EvidenceTable(
        str(metadata.get("table_binding_kwd") or "unbound"),
        str(context.get("binding_source") or metadata.get("table_binding_source_kwd") or "deterministic"),
        not context or bool(context.get("reliable")), str(context.get("caption") or ""),
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
    attributes.pop("financial_facts", None)
    attributes.pop("answer_context", None)
    table = {key: value for key, value in attributes.items() if key.startswith("table_")}
    figure = {key: value for key, value in attributes.items() if key.startswith("visual_") or key == "image_ref_kwd"}
    coordinates = {key: attributes[key] for key in ("page_num_int", "position_int", "top_int") if key in attributes}
    other = set(table) | set(figure) | set(coordinates) | {"evidence_type_kwd", "locator_kwd", "parser_version_kwd"}
    attribution = {key: value for key, value in attributes.items() if key not in other}
    structure = EvidenceStructure(str(attributes.get("evidence_type_kwd") or "text"),
                                  _table_from_record(table),
                                  EvidenceFigure(str(figure.get("visual_status_kwd") or "unextracted"), figure),
                                  EvidenceAttribution(tuple(attribution.get("document_subject_entity_ids") or ()),
                                      str(attribution.get("document_subject_status_kwd") or "unknown"),
                                      tuple((attribution.get("entity_attribution_obj") or {}).get("document_subject_evidence") or ()), attribution),
                                  str(attributes.get("locator_kwd") or ""), str(attributes.get("parser_version_kwd") or ""), coordinates)
    identifier = str(record.get("chunk_id") or record.get("source_id") or "")
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
        **plain(structure.table.metadata), **plain(structure.figure.metadata), **plain(structure.attribution.metadata),
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
    attribution = structure.attribution
    if attribution.subject_ids:
        result["document_subject_entity_ids"] = list(attribution.subject_ids)
        result["document_subject_status_kwd"] = attribution.subject_status
        if not attribution.metadata and attribution.subject_proof:
            result["entity_attribution_obj"] = {"document_subject_evidence": plain(attribution.subject_proof)}
    if structure.locator:
        result["locator_kwd"] = structure.locator
    if structure.parser_version:
        result["parser_version_kwd"] = structure.parser_version
    return result


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


def _legacy_fact_measure(item: dict[str, Any]) -> dict[str, Any]:
    """兼容历史报告把增长率拼在指标名后的表示；不恢复历史核验权限。"""
    from service.core.finance.metrics import parse_metric_mentions
    data = dict(item)
    if "measure_kind" not in data and "/" in data.get("metric", ""):
        mentions = parse_metric_mentions(data["metric"].replace("/", ""))
        if len(mentions) == 1:
            mention = mentions[0]
            data.update(metric=mention.metric, measure_kind=mention.measure_kind,
                        comparison_basis=mention.comparison_basis)
    return data


def _facts_from_record(record: dict[str, Any]) -> FinancialFactAssessment:
    """恢复服务端报告中的财务事实；来源片段的状态字段不进入此入口。"""
    names = {item.name for item in fields(FinancialFactAssessment)}
    data = {key: value for key, value in record.items() if key in names}
    data["requirements"] = [FactRequirement(**_legacy_fact_measure(item)) for item in data.get("requirements", [])]
    data["facts"] = [FinancialFact(**_legacy_fact_measure(item)) for item in data.get("facts", [])]
    return FinancialFactAssessment(**data)


def assessment_from_record(record: dict[str, Any]) -> EvidenceAssessment:
    """读取历史报告内容，不恢复可复用的核验范围；当前来源须重新核验。"""
    facts = _facts_from_record(record.get("financial_fact_bundle") or {})
    if not facts.derived_evidence and record.get("derived_evidence"):
        facts = replace(facts, derived_evidence=record["derived_evidence"])
    diagnostics = dict(record.get("diagnostics") or {})
    diagnostics.update({key: record[key] for key in ("failure_kind", "failure", "attempts") if key in record})
    names = {item.name for item in fields(EvidenceAssessment) if item.init} - {
        "facts", "diagnostics", "query_version", "evidence_version", "policy_version", "require_semantic",
    }
    data = {key: value for key, value in record.items() if key in names}
    data.setdefault("sufficient", False)
    data.setdefault("reason", "证据不足")
    # 历史字典不能自报查询版本和核验模式。
    return EvidenceAssessment(**data, facts=facts, diagnostics=diagnostics)
