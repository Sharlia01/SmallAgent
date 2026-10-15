"""评测侧的独立充分性标签：只读 gold 匹配及来源状态，不读系统放行决定。"""

from typing import Any, Iterable


def all_supported(values: Iterable[bool | None]) -> bool | None:
    values = list(values)
    if any(value is False for value in values):
        return False
    return True if values and all(value is True for value in values) else None


def any_supported(values: Iterable[bool | None]) -> bool | None:
    values = list(values)
    if any(value is True for value in values):
        return True
    return False if values and all(value is False for value in values) else None


def candidate_support(evidence_type: str, chunk: dict[str, Any]) -> dict[str, Any]:
    """文本命中之外，检查被命中的来源是否具有可用的模态结构。"""
    modalities = {evidence_type, str(chunk.get("evidence_type_kwd") or "").casefold()}
    checks = []
    if "figure" in modalities:
        status = chunk.get("visual_status_kwd")
        usable = (True if status == "extracted" else
                  False if status in {"unextracted", "disabled", "unavailable", "failed", "empty"} else None)
        checks.append((usable, "visual_facts_extracted" if usable is True else
                       "visual_facts_unavailable" if usable is False else "visual_status_unknown"))
    if modalities.intersection({"table", "table_row"}):
        status = chunk.get("table_binding_kwd")
        usable = True if status == "bound" else False if status == "unbound" else None
        checks.append((usable, "table_bound" if usable is True else
                       "table_unbound" if usable is False else "table_binding_unknown"))
    if not checks:
        checks.append((True, "text_match"))
    return {"usable": all_supported(value for value, _ in checks),
            "usability_reasons": [reason for _, reason in checks]}


def sufficiency_reference(value: bool | None, *, source: str, reason: str) -> dict[str, Any]:
    return {"version": "1.0", "expected_sufficient": value,
            "status": "sufficient" if value is True else "insufficient" if value is False else "unknown",
            "source": source, "reason": reason}


def assess_expected_sufficiency(
    sample: dict[str, Any], matches: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """每个要求均须成立；要求内任一完整可用方案成立即可。"""
    explicit = sample.get("expected_evidence_sufficient")
    if isinstance(explicit, bool):
        return sufficiency_reference(explicit, source="annotation", reason="explicit_expected_evidence_sufficient")
    if matches is None:
        return sufficiency_reference(None, source="unavailable", reason="verification_snapshot_missing")
    if sample.get("answerable") is False:
        return sufficiency_reference(False, source="answerability", reason="unanswerable_sample")
    if sample.get("answerable") is not True or not matches:
        return sufficiency_reference(None, source="unavailable", reason="gold_annotations_missing")
    # 旧记录没有完整候选及可用性判断，不能从 recall 或最早一次匹配推测充分性。
    value = all_supported(match.get("usable") for match in matches)
    return sufficiency_reference(value, source="gold_evidence",
                                 reason="all_requirements_usable" if value is True else
                                 "required_evidence_missing_or_unusable" if value is False else
                                 "evidence_usability_unknown")


def stored_sufficiency_reference(case: dict[str, Any]) -> dict[str, Any]:
    """汇总复用已保存的判断，包括显式 unknown；不从召回率重新推断。"""
    reference = case.get("evidence_sufficiency_reference")
    if isinstance(reference, dict) and "expected_sufficient" in reference:
        return reference
    return sufficiency_reference(None, source="unavailable", reason="stored_reference_missing")
