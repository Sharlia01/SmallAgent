"""Check that goal constraints do not add unsupported concrete details."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from service.core.query_semantics import reporting_periods
from service.core.query_validation import QueryValidationError, validate_relation_bindings


@dataclass(frozen=True)
class GroundedConstraint:
    value: str
    normalized: str
    kind: str
    source_spans: tuple[dict, ...]
    normalization: str = "identity"

    def to_dict(self) -> dict:
        return asdict(self)


def normalize_constraint(value: str, question: str) -> tuple[str, str]:
    if "同期" in value:
        explicit = re.sub(r"同期|[（()）]", "", value).strip()
        if re.search(r"(?:19|20)\d{2}", explicit):
            return explicit, "explicit_period_replaces_reference"
        periods = reporting_periods(question)
        if len(periods) == 1:
            return value.replace("同期", next(iter(periods))), "unique_period_reference"
    if value in {"一兆瓦时", "一千瓦时"} and re.search(r"多少钱|电价|单价|每.*价格", question):
        return "每" + value[1:], "price_unit_basis"
    return value, "identity"


def ground_constraint(value: str, question: str) -> GroundedConstraint:
    normalized, rule = normalize_constraint(value, question)
    start = question.find(value)
    if start >= 0:
        return GroundedConstraint(value, normalized, "literal",
                                  ({"quote": value, "start": start, "end": start + len(value)},), rule)

    # Colloquial wording may be paraphrased. Only newly introduced concrete
    # names, dates, numbers, and scopes are grounds for rejecting a constraint.
    from service.core.query_expansion import validate_query_preservation
    added = validate_query_preservation(question, normalized).added_risky_terms
    if added:
        raise QueryValidationError("goal_source_not_grounded", "constraint adds details absent from the question",
                                   actual=normalized, offending=added, source=question)
    validate_relation_bindings(question, [normalized])
    return GroundedConstraint(value, normalized, "paraphrase", (), rule)
