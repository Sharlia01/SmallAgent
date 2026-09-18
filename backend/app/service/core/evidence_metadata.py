"""Provenance fields carried from parsing through retrieval and citations."""

EVIDENCE_FIELDS = (
    "page_num_int", "position_int", "top_int", "evidence_type_kwd",
    "locator_kwd", "table_binding_kwd", "table_row_int", "visual_status_kwd",
    "visual_model_kwd", "image_ref_kwd", "parser_version_kwd",
)


def evidence_metadata(source):
    return {key: source[key] for key in EVIDENCE_FIELDS if key in source}
