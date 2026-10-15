"""从解析、检索到引用全程携带的来源字段。"""

from service.core.entities.attribution import ENTITY_ATTRIBUTION_FIELDS

EVIDENCE_FIELDS = (
    "page_num_int",
    "position_int",
    "top_int",
    "evidence_type_kwd",
    "locator_kwd",
    "table_binding_kwd",
    "table_binding_source_kwd",
    "table_model_kwd",
    "table_headers_kwd",
    "table_cells_kwd",
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
    "visual_status_kwd",
    "visual_model_kwd",
    "image_ref_kwd",
    "parser_version_kwd",
    "financial_facts",
    "answer_context",
    *ENTITY_ATTRIBUTION_FIELDS,
)

def evidence_metadata(source):
    return {key: source[key] for key in EVIDENCE_FIELDS if key in source}


def ranking_text(source):
    """重排时优先使用紧凑的表格语义文本，旧数据使用兼容回退。"""
    return (
        source.get("table_semantic_text_with_weight")
        or source.get("content_with_weight")
        or source.get("content_ltks")
        or ""
    )
