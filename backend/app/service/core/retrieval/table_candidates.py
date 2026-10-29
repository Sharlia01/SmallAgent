"""按表格身份截取排名前缀，在最终重排后构建表格候选。"""

from dataclasses import dataclass
import hashlib
import json
from typing import Any

from domain.table_retrieval import IndexedTableRow, TableCandidate
from service.core.evidence.table_retrieval_adapter import candidate_record
from service.core.evidence.metadata import primary_table_row
from service.core.rag.utils import num_tokens_from_string, truncate


MAX_CANDIDATE_TOKENS = 1024


@dataclass
class GroupedCandidates:
    ids: list[str]  #表格组的代表行原始chunk ID
    source_fields: dict[str, dict[str, Any]]
    representative_by_id: dict[str, str]
    tables: dict[str, TableCandidate]

    @property
    def fields(self):
        """旧调用方的输出视图，生产排序直接使用来源快照和 tables。"""
        return {key: candidate_record(self.tables[key]) if key in self.tables else source
                for key, source in self.source_fields.items()}


def _table_key(source: dict[str, Any]) -> tuple[str, ...] | None:
    # 表号之外还限制知识库、文档及父表版本，禁止跨来源拼表。
    if source.get("table_binding_kwd") != "bound" or not source.get("table_id_kwd"):
        return None
    if not source.get("table_object_ref_kwd"):
        return None
    return tuple(str(source.get(key) or "") for key in (
        "kb_id", "doc_id", "table_id_kwd", "table_object_ref_kwd",
    ))


def _rows(chunk_id: str, source: dict[str, Any]) -> list[dict[str, Any]]:
    existing = source.get("table_candidate_obj")
    if isinstance(existing, dict):
        return existing.get("rows", [])
    primary = primary_table_row(source)
    headers, cells = primary.get("headers"), primary.get("cells")
    try:
        row_index = int(primary.get("row_index"))
    except (ValueError, TypeError):
        return []
    if not isinstance(headers, list) or not headers or not isinstance(cells, list) or len(headers) != len(cells):
        return []
    sliced = source.get("table_cells_slice_obj")
    sliced = sliced if isinstance(sliced, dict) else {}
    return [{
        "chunk_id": str(source.get("chunk_id") or chunk_id), "row_index": row_index,
        "headers": headers, "cells": cells,
        "context": str(primary.get("context") or sliced.get("context") or ""),
        "preamble": str(primary.get("preamble") or ""),
        "semantic_text": str(source.get("table_semantic_text_with_weight") or source.get("content_with_weight") or ""),
    }]


def candidate_identity(chunk_id: str, source: dict[str, Any]) -> tuple:
    table_key = _table_key(source)
    return ("table", table_key) if table_key and _rows(chunk_id, source) else ("chunk", chunk_id)


def select_candidate_ids(
    ids: list[str], fields: dict[str, dict[str, Any]], *,
    limit: int, offset: int = 0,
) -> list[str]:
    """按首次出现的候选身份计数，满额立即停止；同表行不重复占名额。

    翻页跳过此前已出现的身份，避免同表剩余行在后页重新成为候选。
    """
    if limit <= 0:
        return []
    identities, selected, seen_ids = {}, [], set()
    for chunk_id in ids:
        if chunk_id not in fields or chunk_id in seen_ids:
            continue
        seen_ids.add(chunk_id)
        key = candidate_identity(chunk_id, fields[chunk_id])
        identities.setdefault(key, len(identities))
        if identities[key] >= offset:
            selected.append(chunk_id)
        if len(identities) >= offset + limit:
            break
    return selected


def group_table_candidates(
    ids: list[str], fields: dict[str, dict[str, Any]], *, limit: int | None = None,
    rerank_scores: dict[str, float] | None = None,
) -> GroupedCandidates:
    """合并最终入选前缀中的同表行；不调用模型，也不再次删减命中行。"""
    if limit is not None:
        ids = select_candidate_ids(ids, fields, limit=limit)
    groups, representatives = {}, {}
    for chunk_id in ids:
        if chunk_id not in fields:
            continue
        source = fields[chunk_id]
        key = candidate_identity(chunk_id, source)
        groups.setdefault(key, []).append((chunk_id, source, _rows(chunk_id, source)))
        representatives[chunk_id] = groups[key][0][0]

    # 保存同表全部已评分行号，扩展时不将截取之外的行重新当作邻居补回。
    scored_indices = {}
    for chunk_id, score in sorted((rerank_scores or {}).items(), key=lambda item: -item[1]):
        if chunk_id in fields:
            key = candidate_identity(chunk_id, fields[chunk_id])
            indices = scored_indices.setdefault(key, [])
            for row in _rows(chunk_id, fields[chunk_id]):
                if row['row_index'] not in indices:
                    indices.append(row['row_index'])

    result_ids, result_fields, tables = [], {}, {}
    for key, members in groups.items():
        anchor, source, _ = members[0]
        result_ids.append(anchor)
        result_fields[anchor] = source
        if key[0] != "table":
            continue
        selected, member_ids, seen_rows = [], [], set()
        for chunk_id, member, rows in members:
            prior = member.get("table_candidate_obj") or {}
            member_ids.extend(prior.get("member_chunk_ids") or [str(member.get("chunk_id") or chunk_id)])
            for row in rows:
                if row["row_index"] in seen_rows:
                    continue
                seen_rows.add(row["row_index"])
                selected.append({**row, **({"rerank_score": rerank_scores[chunk_id]}
                                          if rerank_scores is not None and chunk_id in rerank_scores else {})})
        sliced = source.get("table_cells_slice_obj")
        caption = str(sliced.get("caption") or "") if isinstance(sliced, dict) else ""
        semantic_text = "\n".join(row["semantic_text"] for row in selected)
        selected_ids = {row["chunk_id"] for row in selected}
        candidate_id = "table-group-" + hashlib.sha256(json.dumps(
            [key, sorted(selected_ids)], ensure_ascii=False,
        ).encode()).hexdigest()
        tables[anchor] = TableCandidate(
            candidate_id=candidate_id,
            source_rows=tuple(IndexedTableRow(cid, member) for cid, member, _ in members),
            matched_rows=tuple(selected), member_chunk_ids=tuple(dict.fromkeys(member_ids)),
            omitted_chunk_ids=tuple(cid for cid in dict.fromkeys(member_ids) if cid not in selected_ids),
            scored_row_indices=tuple(scored_indices.get(key, [])),
            ranking_source="model" if rerank_scores is not None else "skipped",
            ranking_text=truncate(semantic_text, MAX_CANDIDATE_TOKENS),
            ranking_text_truncated=num_tokens_from_string(semantic_text) > MAX_CANDIDATE_TOKENS,
            caption=caption,
        )
    return GroupedCandidates(result_ids, result_fields, representatives, tables)
