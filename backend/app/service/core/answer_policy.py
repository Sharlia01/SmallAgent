"""Deterministic answer/refusal policy shared by production and evaluation."""

from __future__ import annotations

from typing import Any, Iterable


DEFAULT_REFUSAL_MESSAGE = "当前资料不足以可靠回答这个问题。"
MAX_REFUSAL_REQUIREMENTS = 5


def normalize_missing_requirements(values: Any) -> list[str]:
    """Return bounded, unique requirement text safe for a user-facing refusal."""
    if not isinstance(values, list):
        return []

    normalized: list[str] = []
    for value in values:
        if not isinstance(value, str):
            continue
        text = " ".join(value.split()).strip()
        if text and text not in normalized:
            normalized.append(text[:160])
        if len(normalized) >= MAX_REFUSAL_REQUIREMENTS:
            break
    return normalized


def render_refusal(missing_requirements: Any = None) -> str:
    """Render a stable refusal instead of asking a model to improvise one."""
    requirements = normalize_missing_requirements(missing_requirements)
    if not requirements:
        return DEFAULT_REFUSAL_MESSAGE
    return (
        f"{DEFAULT_REFUSAL_MESSAGE}"
        f"当前缺少：{'；'.join(requirements)}。"
    )


def refusal_from_tool_results(
    tool_results: Iterable[Any],
) -> tuple[str, dict[str, Any]] | None:
    """Return a refusal only when successful KB searches produced no sources.

    A web result, session document, or a later successful KB retry must be able
    to answer the request. Operational tool failures are deliberately excluded:
    they should follow the existing error/fallback path instead of being
    misreported as a trustworthy "not in the documents" decision.
    """
    results = list(tool_results)
    if any(getattr(result, "sources", None) for result in results):
        return None

    empty_kb_results = [
        result
        for result in results
        if getattr(result, "tool_name", None) == "search_knowledge_base"
        and getattr(result, "success", False)
        and not getattr(result, "sources", None)
    ]
    if not empty_kb_results:
        return None

    latest = empty_kb_results[-1]
    metadata = getattr(latest, "metadata", None) or {}
    decision = metadata.get("evidence_sufficiency") or {}
    missing = decision.get("missing_requirements") or []
    return render_refusal(missing), dict(decision)
