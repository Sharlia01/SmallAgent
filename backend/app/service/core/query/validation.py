"""分析、规划和执行共用的来源范围校验。

原文范围用于证明来源，不要求检索词逐字照抄。
这些规则只校验明确绑定，不能根据用户提及实体的顺序
推断尚未确认的关系答案。
"""

from __future__ import annotations

import re
from typing import Iterable

from service.core.query.semantics import (
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
    from service.core.query.preservation import extract_query_anchors
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
        # 未识别的名称和标签仍属于硬约束，
        # 不能因为领域识别器不认识某公司就静默删除它。
        if value and not anchors and not concepts and not re.search(
            r"谁|哪|什么|多少|如何|是否|是不是|能否|怎么|贵不贵|掉了|上涨|下降", value,
        ):
            terms.append(value)
    return list(dict.fromkeys(terms))


def missing_constraints(terms: Iterable[str], query: str) -> list[str]:
    normalized = constraint_text(query)
    return [term for term in terms if constraint_text(term) not in normalized]


def source_clauses(question: str) -> list[str]:
    # 连接词引入自己的带日期主体时，开始新的局部范围。
    # “收入和利润”或“图2和图3”等候选集合不在此拆分。
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
    from service.core.query.preservation import (
        _extract_organizations, YEAR_OR_DATE_PATTERN, RELATIVE_TIME_PATTERN,
        NEGATION_PATTERN,
    )
    entities = list(_extract_organizations(question))
    shared = entities if len(entities) == 1 else []
    dates = list(dict.fromkeys(m.group().strip() for m in YEAR_OR_DATE_PATTERN.finditer(question)))
    if len(dates) == 1:
        shared.extend(dates)
    # 多个日期中的相对期间属于局部分句，不能扩展为全局约束。
    relatives = list(dict.fromkeys(m.group() for m in RELATIVE_TIME_PATTERN.finditer(question)))
    if len(dates) <= 1 and len(relatives) == 1:
        shared.extend(relatives)
    # 仅问题开头的排除条件能明确视为全局约束；
    # 其他排除条件按原始分句及其所属目标校验。
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
    from service.core.query.preservation import _extract_organizations, ENTITY_LEADING_CONTEXT_PATTERN
    pairs = []
    for clause in source_clauses(text):
        # 表格标签和叙述动词不是实体标识。
        subjects = _extract_organizations(clause)
        if not subjects:
            # 来源分句开头的名称也可用于识别没有常见组织后缀的公司，
            # 但应排除文档标签和字段标签。
            prefix = ENTITY_LEADING_CONTEXT_PATTERN.sub("", clause)
            match = re.match(r"([\u4e00-\u9fff·]{2,12}?)(?:的|在)?\s*((?:19|20)\d{2})", prefix)
            if match and not _topics(match[1]) and not re.search(r"表|图|财务|预测|报告|截至|季度|半年|年度|同比|环比", match[1]):
                subjects.add(match[1])
        if len(subjects) == 1:
            pairs.extend((next(iter(subjects)), year) for year in _years(clause))
    return pairs


def validate_relation_bindings(question: str, candidates: Iterable[str]) -> None:
    from service.core.query.preservation import LOCATOR_PATTERN, YEAR_OR_DATE_PATTERN, _extract_organizations
    candidates = list(candidates)
    locators = {canonical_text(m.group()) for m in LOCATOR_PATTERN.finditer(question)}
    unknown_locator = UNKNOWN_LOCATOR.search(question) and len(locators) > 1
    subjects: dict[str, set[str]] = {}
    for subject, year in _subject_years(question):
        subjects.setdefault(subject, set()).add(year)
    if len(subjects) == 1:
        # 第二个带日期分句可省略共享主体；
        # 下方按指标进行的局部检查仍会防止两个日期被交换。
        next(iter(subjects.values())).update(_years(question))
    clauses = source_clauses(question)
    for candidate in candidates:
        # 包含多个绑定的单句计划也需要校验；
        # 在其他位置保留全部定位词，不能掩盖虚构的对应关系。
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
                continue  # 重复指标必须有明确的目标原文范围。
            source = sources[0]
            # 一个目标可能覆盖多个分句；只比较当前主题所属的分句，
            # 不能拿其他目标的无关日期或实体参与匹配。
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
    from service.core.query.preservation import extract_query_anchors, validate_query_preservation
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
