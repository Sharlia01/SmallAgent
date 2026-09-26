"""Source-scoped validation shared by analysis, planning and execution.

Source spans prove provenance; they are not mandatory literal query strings.
These guards handle explicit bindings only. They never infer an answer to an
unresolved relation from the order of entities in a user's question.
"""

from __future__ import annotations

import re
from typing import Iterable

from service.core.query_semantics import (
    QUERY_CONCEPTS, canonical_text, constraint_text, domain_policy, metric_terms, reporting_periods,
)


class QueryValidationError(ValueError):
    def __init__(self, rule: str, detail: str, *, field: str | None = None,
                 node_id: str | None = None, goal_id: str | None = None,
                 expected=None, actual=None, offending=None, source=None):
        self.rule = rule
        self.field = field
        self.node_id = node_id
        self.goal_id = goal_id
        self.context = {key: value for key, value in {
            "expected": expected, "actual": actual, "offending": offending, "source": source,
        }.items() if value is not None}
        super().__init__(f"{rule}: {detail}")


META_ANCHORS = {"分别", "截至"}
DEPENDENT_REFERENCE = re.compile(
    r"该类电源|该电源板块|上述(?:确定的)?电源板块|上述(?:确定的)?电源类型|"
    r"第一步确定的电源类型|该评级|这个评级|该政策|这项政策|该对象|前一步结果"
)
UNKNOWN_LOCATOR = re.compile(r"哪(?:一)?(?:张|幅|个)?(?:图|表)|哪(?:张|幅|个)图表")
NON_TOPICS = {"同比", "环比", "预测", "实际", "截至", "分别"}


def required_constraint_terms(values: Iterable[str], *, dependent: bool = False) -> list[str]:
    from service.core.query_expansion import extract_query_anchors
    terms = []
    for value in values:
        if dependent:
            value = DEPENDENT_REFERENCE.sub("", value)
        anchors = [a for a in extract_query_anchors(value) if a not in META_ANCHORS]
        concepts = [concept for concept, aliases in QUERY_CONCEPTS.items()
                    if any(alias in value for alias in aliases)]
        terms.extend(anchors)
        terms.extend(scope for scope in domain_policy().get("business_scopes", []) if scope in value)
        terms.extend(term for term in ("实际", "累计", "新增", "单季度", "全年", "扣非",
                                       "只有", "最高", "最低", "最多", "最少") if term in value)
        if re.search(r"(?:19|20)\d{2}E(?![A-Za-z0-9])", value, re.I):
            terms.append("预测")
        terms.extend(concepts)
        # Unrecognized names/labels remain hard constraints. Do not silently
        # erase a company merely because the domain recognizer lacks its name.
        if value and not anchors and not concepts and not re.search(
            r"谁|哪|什么|多少|如何|是否|是不是|能否|怎么|贵不贵|掉了|上涨|下降", value,
        ):
            terms.append(value)
    return list(dict.fromkeys(terms))


def missing_constraints(terms: Iterable[str], query: str) -> list[str]:
    normalized = constraint_text(query)
    return [term for term in terms if constraint_text(term) not in normalized]


def source_clauses(question: str) -> list[str]:
    # A conjunction introducing its own dated subject starts a local scope.
    # Do not split “收入和利润” or a candidate set such as “图2和图3”.
    clauses = []
    for sentence in re.split(r"[，,；;。！？?：:]", question):
        start = 0
        for conjunction in re.finditer(r"以及|同时|和|与", sentence):
            before, after = sentence[start:conjunction.start()], sentence[conjunction.end():]
            dated_subject = re.match(r"[\u4e00-\u9fffA-Za-z·]{2,20}(?:19|20)\d{2}", after)
            dated_metric = re.match(r"(?:19|20)\d{2}", after) and _topics(before)
            if dated_subject or dated_metric:
                clauses.append(before.strip())
                start = conjunction.end()
        clauses.append(sentence[start:].strip())
    return [part for part in clauses if part]


def shared_scope_terms(question: str) -> list[str]:
    from service.core.query_expansion import (
        _extract_organizations, YEAR_OR_DATE_PATTERN, RELATIVE_TIME_PATTERN,
        NEGATION_PATTERN,
    )
    entities = list(_extract_organizations(question))
    shared = entities if len(entities) == 1 else []
    dates = list(dict.fromkeys(m.group().strip() for m in YEAR_OR_DATE_PATTERN.finditer(question)))
    if len(dates) == 1:
        shared.extend(dates)
    # A relative period embedded in one of several dates is local, not global.
    relatives = list(dict.fromkeys(m.group() for m in RELATIVE_TIME_PATTERN.finditer(question)))
    if len(dates) <= 1 and len(relatives) == 1:
        shared.extend(relatives)
    # Only leading exclusions are unambiguously global. Other exclusions are
    # checked against their source clause and the goal that owns it.
    leading = source_clauses(question)[0] if question else ""
    if re.match(r"^(?:请)?(?:不含|不包括|不考虑)", leading):
        shared.extend(m.group() for m in NEGATION_PATTERN.finditer(leading))
        shared.extend(required_constraint_terms([leading]))
    return list(dict.fromkeys(shared))


def _topics(text: str) -> set[str]:
    return set(metric_terms(text)) - NON_TOPICS


def _years(text: str) -> set[str]:
    return set(re.findall(r"(?:19|20)\d{2}", text))


def _subject_years(text: str) -> list[tuple[str, str]]:
    from service.core.query_expansion import _extract_organizations, ENTITY_LEADING_CONTEXT_PATTERN
    pairs = []
    for clause in source_clauses(text):
        # Table labels and narrative verbs are not entity identifiers.
        subjects = _extract_organizations(clause)
        if not subjects:
            # A source-local leading name also covers companies without a
            # registered organization suffix. Exclude document/field labels.
            prefix = ENTITY_LEADING_CONTEXT_PATTERN.sub("", clause)
            match = re.match(r"([\u4e00-\u9fff·]{2,12}?)(?:的|在)?\s*((?:19|20)\d{2})", prefix)
            if match and not _topics(match[1]) and not re.search(r"表|图|财务|预测|报告|截至|季度|半年|年度|同比|环比", match[1]):
                subjects.add(match[1])
        if len(subjects) == 1:
            pairs.extend((next(iter(subjects)), year) for year in _years(clause))
    return pairs


def validate_relation_bindings(question: str, candidates: Iterable[str]) -> None:
    from service.core.query_expansion import LOCATOR_PATTERN, YEAR_OR_DATE_PATTERN, _extract_organizations
    candidates = list(candidates)
    locators = {canonical_text(m.group()) for m in LOCATOR_PATTERN.finditer(question)}
    unknown_locator = UNKNOWN_LOCATOR.search(question) and len(locators) > 1
    subjects: dict[str, set[str]] = {}
    for subject, year in _subject_years(question):
        subjects.setdefault(subject, set()).add(year)
    if len(subjects) == 1:
        # A shared subject may be omitted in the second dated clause. The
        # metric-local check below still prevents exchanging the two dates.
        next(iter(subjects.values())).update(_years(question))
    clauses = source_clauses(question)
    for candidate in candidates:
        # Include single-sentence plans containing several bindings: preserving
        # all locator tokens elsewhere must not disguise a fabricated pairing.
        for clause in source_clauses(candidate):
            selected = {canonical_text(m.group()) for m in LOCATOR_PATTERN.finditer(clause)}
            if unknown_locator and selected and selected < locators and _topics(clause):
                raise QueryValidationError("unresolved_relation_bound", "a locator-to-fact relation must remain a question")
        for subject, year in _subject_years(candidate):
            if subject in subjects and year not in subjects[subject]:
                raise QueryValidationError("subject_time_rebound", "goal-local subject was assigned another subject's year")
        candidate_topics = _topics(candidate)
        for topic in candidate_topics:
            sources = [clause for clause in clauses if topic in _topics(clause)]
            if len(sources) != 1:
                continue  # repeated metrics need explicit goal source spans
            source = sources[0]
            # A goal can own several clauses; only compare its clause for this
            # topic, never unrelated dates or entities from another goal.
            parts = [c for c in source_clauses(candidate) if topic in _topics(c)]
            for part in parts:
                if _years(source) and _years(part) and not _years(part) <= _years(source):
                    raise QueryValidationError("metric_time_rebound", f"time scope changed for {topic}")
                source_periods, candidate_periods = reporting_periods(source), reporting_periods(part)
                if source_periods and candidate_periods and not candidate_periods <= source_periods:
                    raise QueryValidationError("metric_period_rebound", f"reporting period changed for {topic}",
                                               expected=sorted(source_periods), actual=sorted(candidate_periods), source=source)
                source_entities = _extract_organizations(source)
                candidate_entities = _extract_organizations(part)
                if source_entities and candidate_entities and not candidate_entities <= source_entities:
                    raise QueryValidationError("metric_entity_rebound", f"entity scope changed for {topic}")
                explicit_locators = {canonical_text(m.group()) for m in LOCATOR_PATTERN.finditer(source)}
                chosen = {canonical_text(m.group()) for m in LOCATOR_PATTERN.finditer(part)}
                if not unknown_locator and explicit_locators and chosen and not chosen <= explicit_locators:
                    raise QueryValidationError("metric_locator_rebound", f"locator scope changed for {topic}")
                scopes = [scope for scope in domain_policy().get("business_scopes", []) if scope in source]
                if len(scopes) == 1 and scopes[0] not in part:
                    raise QueryValidationError("metric_scope_dropped", f"business scope dropped for {topic}")


def validate_goals(question: str, goals) -> None:
    from service.core.query_expansion import extract_query_anchors, validate_query_preservation
    if not goals:
        return
    for index, goal in enumerate(goals):
        try:
            validate_relation_bindings(question, [goal.text])
        except QueryValidationError as error:
            error.field = f"goals[{index}].text"
            error.goal_id = goal.id
            raise
    combined = " ".join(goal.text + " " + " ".join(goal.constraints) for goal in goals)
    missing = missing_constraints(
        [a for a in extract_query_anchors(question) if a not in META_ANCHORS], combined,
    )
    if missing:
        raise QueryValidationError("analysis_missing_requirements", "analysis dropped original requirements: " + ", ".join(missing), field="goals")
    for index, goal in enumerate(goals):
        added = validate_query_preservation(question, goal.text).added_risky_terms
        if added:
            raise QueryValidationError("analysis_invented_details", "analysis invented exact goal details", field=f"goals[{index}].text", goal_id=goal.id,
                                       offending=added, actual=goal.text, source=question)
        if goal.source_spans:
            try:
                validate_relation_bindings("；".join(goal.source_spans), [goal.text])
            except QueryValidationError as error:
                error.field = f"goals[{index}].text"
                error.goal_id = goal.id
                raise
