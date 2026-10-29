"""从解析、检索到引用全程携带的来源字段。"""

from collections.abc import Mapping
import json


# 仅用于读取旧索引/父表时丢弃历史推断，不再生成或解释这些归属字段。
_RETIRED_ENTITY_FIELDS = frozenset({
    "entity_annotation_version_kwd", "entity_registry_version_kwd",
    "document_subject_entity_ids", "document_subject_entity_names_kwd",
    "document_subject_status_kwd", "publisher_entity_ids", "publisher_entity_names_kwd",
    "publisher_status_kwd", "mentioned_entity_ids", "mentioned_entity_names_kwd",
    "subject_entity_ids", "candidate_subject_entity_ids", "unresolved_entity_names_kwd",
    "entity_attribution_status_kwd", "entity_attribution_obj", "entity_attribution",
})


def without_entity_annotations(value):
    """保留原始表格数据，去掉嵌在旧表格行或元数据中的实体归属结论。"""
    if isinstance(value, Mapping):
        return {key: without_entity_annotations(item) for key, item in value.items()
                if key not in _RETIRED_ENTITY_FIELDS}
    if isinstance(value, (list, tuple)):
        return [without_entity_annotations(item) for item in value]
    return value


EVIDENCE_FIELDS = (
    "page_num_int",
    "position_int",
    "top_int",
    "evidence_type_kwd",
    "locator_kwd",
    "table_binding_kwd",
    "table_binding_source_kwd",
    "table_model_kwd",
    "table_row_int",
    "table_id_kwd",
    "table_object_ref_kwd",
    "table_chunk_kind_kwd",
    "table_row_start_int",
    "table_row_end_int",
    "table_semantic_text_with_weight",
    "table_markdown_excerpt_with_weight",
    "table_cells_slice_obj",
    "table_schema_version_kwd",
    "table_context_obj",
    "table_candidate_obj",
    "visual_status_kwd",
    "visual_model_kwd",
    "image_ref_kwd",
    "parser_version_kwd",
)


def evidence_metadata(source):
    # 新旧索引共用白名单；旧标签即使还在 ES 中，也不会进入 Evidence 或模型请求。
    return without_entity_annotations({key: source[key] for key in EVIDENCE_FIELDS if key in source})


def primary_table_row(source):
    """只读取结构化主行；缺少结构时返回空值，不猜测旧字段。"""
    sliced = source.get("table_cells_slice_obj")
    rows = sliced.get("rows") if isinstance(sliced, Mapping) else None
    if isinstance(rows, (list, tuple)) and rows:
        primary_index = source.get("table_row_int", source.get("table_row_start_int"))
        if len(rows) == 1 and isinstance(rows[0], Mapping):
            return rows[0]
        for row in rows:
            if isinstance(row, Mapping) and str(row.get("row_index")) == str(primary_index):
                return row
        return {}
    return {}


def index_evidence_metadata(source):
    """入库元数据只保留结构化单元格及必要的内容视图。"""
    metadata = evidence_metadata(source)
    if metadata.get("evidence_type_kwd") != "table":
        return metadata
    # 通用正文保留展示内容，扩展前摘录在检索合组时按需生成。
    if source.get("content_with_weight") == metadata.get("table_markdown_excerpt_with_weight"):
        metadata.pop("table_markdown_excerpt_with_weight", None)
    return metadata


def ranking_text(source):
    """重排时优先使用紧凑的表格语义文本，旧数据使用兼容回退。"""
    return (
        source.get("table_semantic_text_with_weight")
        or source.get("content_with_weight")
        or source.get("content_ltks")
        or ""
    )


def rerank_text(source, *, content=None):
    """仅为模型重排补充原始出处，不推断正文中数值的所属主体。"""
    text = str(ranking_text(source) if content is None else content)
    context = {}
    name = source.get("document_name") or source.get("docnm_kwd")
    if isinstance(name, str) and name.strip():
        context["文档名"] = name
    table = source.get("table_cells_slice_obj")
    caption = table.get("caption") if isinstance(table, Mapping) else None
    if isinstance(caption, str) and caption.strip():
        context["表题"] = caption
    if not context:
        return text
    # JSON 转义来源字段中的换行，避免把文档名混入正文。
    return ("来源上下文（不代表正文中每个数值的主体）："
            + json.dumps(context, ensure_ascii=False) + "\n正文：\n" + text)
