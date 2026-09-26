"""Layers 2 and 3: route an analyzed query, then build its retrieval plan."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field, replace
from string import Formatter
from typing import Any, Literal

from service.core.query_expansion import (
    validate_query_preservation, extract_query_anchors,
)
from service.core.query_contract import QueryGoal, MAX_GOALS
from service.core.query_runtime import (
    validated_completion, observe_contract, observation_context,
    record_observation, failure_details,
)
from service.core.query_semantics import canonical_text
from service.core.query_validation import (
    QueryValidationError, missing_constraints, required_constraint_terms,
    shared_scope_terms, validate_goals, validate_relation_bindings,
)
from service.core.query_intent import (
    RATING_SEQUENTIAL_PATTERN,
    QueryAnalysis,
    analyze_query_intent,
    get_query_intent_client,
    query_intent_model,
    query_intent_timeout_seconds,
)


logger = logging.getLogger(__name__)

MAX_SUBQUERY_CHARS = 300
MAX_SUBQUERIES = MAX_GOALS
PLAN_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,31}$")

RetrievalMode = Literal["single", "parallel", "sequential", "graph", "clarify"]
RouteSource = Literal["rule", "analysis"]
PlanSource = Literal["none", "rule", "model", "fallback"]


PARALLEL_PLAN_SYSTEM_PROMPT = """
你是知识库 RAG 的并行检索计划器。路由器已经确定该问题必须使用 parallel；你只负责生成可执行子查询，不得重新判断检索模式，不要回答问题。

要求：
1. 生成2至8个可独立执行的子查询，每个查询对应一个不同检索目标。
2. 每个子查询必须保留与自身目标相关的公司、时间、指标、表号、图号、否定词、单位和限定条件。
3. 不得新增原问题中没有的年份、日期、数字、实体、图表号或事实。
4. 不得把纯计算、求和、差值、比例或大小比较生成为检索查询。
5. 不得生成近义重复查询。
6. “未来三年、近两年、今年、明年”等相对时间必须原样保留，不得猜测具体年份。

只返回 JSON：
{
  "subqueries":[
    {
      "id":"first_fact",
      "query":"第一个事实的完整检索问题",
      "query_template":"",
      "depends_on":[],
      "output_slot":null,
      "required_slots":[],
      "inherit_document_scope":false
    },
    {
      "id":"second_fact",
      "query":"第二个事实的完整检索问题",
      "query_template":"",
      "depends_on":[],
      "output_slot":null,
      "required_slots":[],
      "inherit_document_scope":false
    }
  ]
}
""".strip()

GRAPH_PLAN_SYSTEM_PROMPT = """
生成有界的知识库检索依赖图，只检索证据，不回答问题，不把计算本身作为检索节点。
最多8个节点。独立节点使用query；依赖节点使用query_template引用上游output_slot，
depends_on列出生产所需槽位的节点，required_slots列出用到的槽位。禁止循环和猜测桥接值。
每个节点的goal_ids必须对应输入的语义目标；覆盖全部目标且不生成近义重复节点。
保留各目标的constraints。已知具体值直接使用，不要额外检索获取已知值。
用户要求同一报告时，依赖节点inherit_document_scope=true；其他情况也不得扩大明确范围。
只返回JSON：{"subqueries":[{"id":"lookup","query":"完整查询", "query_template":"",
"depends_on":[],"output_slot":"entity","required_slots":[],"inherit_document_scope":false,"goal_ids":["goal1"]}]}
""".strip()


SEQUENTIAL_PLAN_SYSTEM_PROMPT = """
你是知识库 RAG 的递进检索计划器。路由器已经确定该问题必须使用 sequential；你只负责生成严格的两跳计划，不得重新判断检索模式，不要回答问题。

要求：
1. 第一步使用 query 直接检索，并通过 output_slot 声明需要从证据中提取的桥接值。
2. 第二步不得提前猜测桥接值；query 必须为空，通过 query_template 中的 {slot} 引用第一步结果。
3. 第二步 depends_on 必须只包含第一步 id，required_slots 必须只包含第一步 output_slot。
4. 用户明确说“这份研报、该报告、同一文档”时，第二步 inherit_document_scope 为 true。
5. 不得新增原问题中没有的年份、日期、数字、实体、图表号或事实。

只返回 JSON：
{
  "subqueries":[
    {
      "id":"first_lookup",
      "query":"第一跳检索问题",
      "query_template":"",
      "depends_on":[],
      "output_slot":"bridge_value",
      "required_slots":[],
      "inherit_document_scope":false
    },
    {
      "id":"second_lookup",
      "query":"",
      "query_template":"{bridge_value}对应的第二跳检索问题",
      "depends_on":["first_lookup"],
      "output_slot":null,
      "required_slots":["bridge_value"],
      "inherit_document_scope":false
    }
  ]
}
""".strip()


@dataclass(frozen=True)
class QuerySubqueryPlan:
    """One executable step in a bounded retrieval plan."""

    id: str
    query: str = ""
    query_template: str = ""
    depends_on: list[str] = field(default_factory=list)
    output_slot: str | None = None
    required_slots: list[str] = field(default_factory=list)
    inherit_document_scope: bool = False
    goal_ids: list[str] = field(default_factory=list)
    required_constraints: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RetrievalRoute:
    """Layer-2 decision derived only from query-analysis signals."""

    mode: RetrievalMode
    reason: str
    source: RouteSource

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalPlan:
    """Layer-3 executable plan; single retrieval has no subqueries."""

    mode: RetrievalMode
    subqueries: list[QuerySubqueryPlan] = field(default_factory=list)
    source: PlanSource = "none"
    fallback_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreparedRetrievalRequest:
    """The three pre-execution layers retained for diagnostics."""

    analysis: QueryAnalysis
    route: RetrievalRoute
    plan: RetrievalPlan


def select_retrieval_route(analysis: QueryAnalysis) -> RetrievalRoute:
    """Choose a retrieval method without creating any subqueries."""

    if analysis.intent in {"ambiguous_query", "invalid_query"}:
        return RetrievalRoute(mode="clarify", reason=analysis.clarification or "请补充明确的查询对象和条件。", source="analysis")
    if any(len(goal.candidate_locators) > 1 for goal in analysis.goals):
        return RetrievalRoute(mode="parallel", reason="逐个覆盖未知关系的候选图表，禁止预先绑定事实", source="rule")
    if analysis.has_dependency and analysis.goals and (
        len(analysis.goals) > 2 or sum(not goal.depends_on for goal in analysis.goals) > 1
    ):
        return RetrievalRoute(mode="graph", reason="存在混合或多层检索依赖", source="analysis")
    if analysis.has_dependency:
        return RetrievalRoute(
            mode="sequential",
            reason="后续检索必须使用前一步取得的桥接值",
            source="analysis",
        )

    if (
        analysis.target_count >= 2
        and analysis.targets_independent
        and not analysis.single_retrieval_likely_covers_all
    ):
        return RetrievalRoute(
            mode="parallel",
            reason=(
                "存在多个独立检索目标，"
                "单次结果集不太可能完整覆盖"
            ),
            source="analysis",
        )

    return RetrievalRoute(
        mode="single",
        reason=(
            "一次检索很可能覆盖全部目标，"
            "或不存在可独立拆分的目标"
        ),
        source="rule",
    )


def _normalize_plan_text(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise QueryValidationError("invalid_plan_text", f"query plan {field_name} must be a string", field=field_name)
    normalized = " ".join(value.split()).strip()
    if len(normalized) > MAX_SUBQUERY_CHARS:
        raise QueryValidationError("plan_text_too_long", f"query plan {field_name} is too long", field=field_name)
    return normalized


def _parse_identifier_list(value: Any, *, field_name: str) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and PLAN_ID_PATTERN.fullmatch(item)
        for item in value
    ):
        raise QueryValidationError("invalid_identifier_list", f"query plan {field_name} is invalid", field=field_name)
    return list(dict.fromkeys(value))


def _parse_subquery(payload: Any) -> QuerySubqueryPlan:
    if not isinstance(payload, dict):
        raise QueryValidationError("invalid_node_schema", "query plan step must be an object")
    subquery_id = payload.get("id")
    if not isinstance(subquery_id, str) or not PLAN_ID_PATTERN.fullmatch(
        subquery_id
    ):
        raise QueryValidationError("invalid_node_id", "query plan step id is invalid", field="id")

    output_slot = payload.get("output_slot")
    if output_slot is not None and (
        not isinstance(output_slot, str)
        or not PLAN_ID_PATTERN.fullmatch(output_slot)
    ):
        raise QueryValidationError("invalid_output_slot", "query plan output_slot is invalid", field="output_slot")
    inherit_document_scope = payload.get("inherit_document_scope", False)
    if not isinstance(inherit_document_scope, bool):
        raise QueryValidationError("invalid_boolean", "inherit_document_scope must be a boolean", field="inherit_document_scope")

    return QuerySubqueryPlan(
        id=subquery_id,
        query=_normalize_plan_text(payload.get("query", ""), field_name="query"),
        query_template=_normalize_plan_text(
            payload.get("query_template", ""),
            field_name="query_template",
        ),
        depends_on=_parse_identifier_list(
            payload.get("depends_on", []),
            field_name="depends_on",
        ),
        output_slot=output_slot,
        required_slots=_parse_identifier_list(
            payload.get("required_slots", []),
            field_name="required_slots",
        ),
        inherit_document_scope=inherit_document_scope,
        goal_ids=_parse_identifier_list(payload.get("goal_ids", []), field_name="goal_ids"),
    )


def _parse_plan_steps(content: str) -> list[QuerySubqueryPlan]:
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise QueryValidationError("invalid_plan_schema", "query plan response must be a JSON object", field="response")
    raw_subqueries = payload.get("subqueries")
    if not isinstance(raw_subqueries, list):
        raise QueryValidationError("invalid_subquery_list", "query plan subqueries must be a list", field="subqueries")
    if len(raw_subqueries) > MAX_SUBQUERIES:
        raise QueryValidationError("too_many_subqueries", "query plan contains too many subqueries", field="subqueries")
    steps = []
    for index, item in enumerate(raw_subqueries):
        try:
            steps.append(_parse_subquery(item))
        except QueryValidationError as error:
            error.field = f"subqueries[{index}]" + (f".{error.field}" if error.field else "")
            if isinstance(item, dict) and isinstance(item.get("id"), str):
                error.node_id = item["id"][:32]
            raise
    return steps


def _validate_sequential_plan(subqueries: list[QuerySubqueryPlan]) -> None:
    if len(subqueries) != 2:
        raise QueryValidationError("invalid_sequential_size", "sequential retrieval must contain exactly two steps", field="subqueries")
    first, second = subqueries
    if first.id == second.id:
        raise QueryValidationError("duplicate_sequential_node", "sequential retrieval step ids must be unique", field="subqueries[].id")
    if not first.query or first.query_template or first.depends_on:
        raise QueryValidationError("invalid_first_step", "first sequential step is invalid", field="subqueries[0]")
    if not first.output_slot:
        raise QueryValidationError("missing_bridge_slot", "first sequential step must declare output_slot", field="subqueries[0].output_slot")
    if first.required_slots:
        raise QueryValidationError("unexpected_required_slots", "first sequential step cannot require bridge slots", field="subqueries[0].required_slots")
    if second.query or not second.query_template:
        raise QueryValidationError("missing_query_template", "second sequential step must use query_template", field="subqueries[1].query_template")
    if second.output_slot is not None:
        raise QueryValidationError("unexpected_output_slot", "second sequential step cannot declare output_slot", field="subqueries[1].output_slot")
    if second.depends_on != [first.id]:
        raise QueryValidationError("invalid_sequential_dependency", "second sequential step has an invalid dependency", field="subqueries[1].depends_on")
    if second.required_slots != [first.output_slot]:
        raise QueryValidationError("sequential_slot_mismatch", "second sequential step must consume the first output slot", field="subqueries[1].required_slots")
    if "{" + first.output_slot + "}" not in second.query_template:
        raise QueryValidationError("template_slot_missing", "second query template does not reference the output slot", field="subqueries[1].query_template")


def _validate_parallel_plan(subqueries: list[QuerySubqueryPlan]) -> None:
    if not 2 <= len(subqueries) <= MAX_SUBQUERIES:
        raise QueryValidationError("invalid_parallel_size", "parallel retrieval must contain two to eight steps", field="subqueries")
    if len({step.id for step in subqueries}) != len(subqueries):
        raise QueryValidationError("duplicate_parallel_node", "parallel retrieval step ids must be unique", field="subqueries[].id")
    for step in subqueries:
        if not step.query or step.query_template:
            raise QueryValidationError("invalid_parallel_query", "parallel retrieval steps must use direct queries", field="subqueries[].query")
        if step.depends_on or step.output_slot or step.required_slots:
            raise QueryValidationError("dependent_parallel_node", "parallel retrieval steps must be independent", field="subqueries[].depends_on")
        if step.inherit_document_scope:
            raise QueryValidationError("invalid_parallel_scope", "parallel retrieval cannot inherit a prior scope", field="subqueries[].inherit_document_scope")


def _validate_parallel_subquery_drift(
    original_question: str,
    subqueries: list[QuerySubqueryPlan],
    *, goals: tuple[QueryGoal, ...] = (),
) -> None:
    drifted_steps = []
    for step in subqueries:
        validation = validate_query_preservation(original_question, step.query)
        if validation.added_risky_terms:
            drifted_steps.append(
                {
                    "subquery_id": step.id,
                    "added_risky_terms": validation.added_risky_terms,
                }
            )
    if drifted_steps:
        raise QueryValidationError("parallel_invented_constraints",
                                   f"parallel subqueries invented exact constraints: {drifted_steps}",
                                   field="subqueries[].query", node_id=drifted_steps[0]["subquery_id"])

    # Every applicable shared scope must survive in every independent query.
    # Goal-bearing plans use locally compiled constraints below. Uniqueness of
    # a date in a whole request does not establish that it applies to all goals.
    shared = [] if goals else shared_scope_terms(original_question)
    for step in subqueries:
        if missing_constraints(shared, step.query):
            raise QueryValidationError("shared_constraint_dropped", "parallel subquery dropped a shared constraint", field="subqueries[].query")
    combined = " ".join(step.query for step in subqueries)
    missing = [anchor for anchor in extract_query_anchors(original_question)
               if anchor not in {"分别", "截至"} and canonical_text(anchor) not in canonical_text(combined)]
    if missing:
        raise QueryValidationError("parallel_requirements_dropped", f"parallel plan dropped requirements: {missing}", field="subqueries[].query")
    _validate_step_bindings(original_question, subqueries)


def _validate_step_bindings(question: str, steps: list[QuerySubqueryPlan]) -> None:
    for index, step in enumerate(steps):
        try:
            validate_relation_bindings(question, [step.query or step.query_template])
        except QueryValidationError as error:
            error.field = f"subqueries[{index}]." + ("query_template" if step.query_template else "query")
            error.node_id = step.id
            raise


def validate_dependency_graph(subqueries: list[QuerySubqueryPlan]) -> list[list[QuerySubqueryPlan]]:
    """Validate all templates and dependencies before any retrieval is run."""
    if not 1 <= len(subqueries) <= MAX_SUBQUERIES:
        raise QueryValidationError("invalid_graph_size", "invalid graph size", field="subqueries")
    by_id = {step.id: step for step in subqueries}
    if len(by_id) != len(subqueries):
        raise QueryValidationError("duplicate_graph_node", "duplicate graph node", field="subqueries[].id")
    producers = {}
    identities = set()
    for step in subqueries:
        if step.output_slot:
            if step.output_slot in producers:
                raise QueryValidationError("duplicate_output_slot", "duplicate output slot", field="subqueries[].output_slot", node_id=step.id)
            producers[step.output_slot] = step.id
        if bool(step.query) == bool(step.query_template):
            raise QueryValidationError("query_template_exclusivity", "node needs exactly one query or template", field="subqueries[].query_template", node_id=step.id)
        identity = canonical_text(step.query or step.query_template)
        if identity in identities:
            raise QueryValidationError("duplicate_query", "duplicate semantic query", field="subqueries[].query", node_id=step.id)
        identities.add(identity)
        if any(dep not in by_id or dep == step.id for dep in step.depends_on):
            raise QueryValidationError("unknown_or_self_dependency", "unknown or self dependency", field="subqueries[].depends_on", node_id=step.id)
        if step.inherit_document_scope and not step.depends_on:
            raise QueryValidationError("scope_without_dependency", "scope inheritance requires a dependency", field="subqueries[].inherit_document_scope", node_id=step.id)
    for step in subqueries:
        fields = []
        try:
            template_fields = list(Formatter().parse(step.query_template))
        except ValueError as error:
            raise QueryValidationError("invalid_query_template", str(error),
                                       field="subqueries[].query_template", node_id=step.id) from error
        for _, name, spec, conversion in template_fields:
            if name is None:
                continue
            if spec or conversion or not PLAN_ID_PATTERN.fullmatch(name):
                raise QueryValidationError("invalid_template_field", "invalid template field", field="subqueries[].query_template", node_id=step.id)
            fields.append(name)
        if set(fields) != set(step.required_slots):
            raise QueryValidationError("template_slot_mismatch", "template fields do not match required slots", field="subqueries[].required_slots", node_id=step.id)
        if step.query_template and not fields:
            raise QueryValidationError("template_without_slots", "dependent template has no slots", field="subqueries[].query_template", node_id=step.id)
        if any(producers.get(slot) not in step.depends_on for slot in fields):
            raise QueryValidationError("slot_producer_missing", "slot producer is not a dependency", field="subqueries[].depends_on", node_id=step.id)
        if step.depends_on and set(step.depends_on) != {producers.get(slot) for slot in fields}:
            raise QueryValidationError("unused_dependency", "dependency does not supply a consumed slot", field="subqueries[].depends_on", node_id=step.id)
    completed, waves = set(), []
    while len(completed) < len(subqueries):
        ready = [step for step in subqueries if step.id not in completed and set(step.depends_on) <= completed]
        if not ready:
            raise QueryValidationError("cyclic_dependency_graph", "cyclic dependency graph", field="subqueries[].depends_on")
        waves.append(ready)
        completed.update(step.id for step in ready)
    return waves


def _validate_goal_coverage(question: str, steps: list[QuerySubqueryPlan], goals: tuple[QueryGoal, ...]) -> list[QuerySubqueryPlan]:
    validate_goals(question, goals)
    _validate_step_bindings(question, steps)
    by_id = {goal.id: goal for goal in goals}
    if not goals:
        # Compatibility with saved v2 plans: at least guard invented details and
        # loss across the plan, even when stable goal IDs are unavailable.
        combined = " ".join(step.query or step.query_template for step in steps)
        for step in steps:
            text = re.sub(r"\{[a-z][a-z0-9_]*\}", "", step.query or step.query_template)
            if validate_query_preservation(question, text).added_risky_terms:
                raise QueryValidationError("plan_invented_constraints", "plan invented exact constraints", field="subqueries[].query")
        if any(canonical_text(a) not in canonical_text(combined) for a in extract_query_anchors(question)
               if a not in {"分别", "截至"}):
            raise QueryValidationError("plan_anchor_coverage", "plan does not cover original anchors", field="subqueries")
        return steps
    if {g for step in steps for g in step.goal_ids} != set(by_id):
        raise QueryValidationError("goal_coverage_mismatch", "plan must cover every declared goal and no unknown goals", field="subqueries[].goal_ids")
    owners = {goal_id: {step.id for step in steps if goal_id in step.goal_ids} for goal_id in by_id}
    validated = []
    for step in steps:
        if not step.goal_ids:
            raise QueryValidationError("node_without_goal", "every node must own a goal", field="subqueries[].goal_ids", node_id=step.id)
        constraints = required_constraint_terms(
            [c for goal_id in step.goal_ids for c in by_id[goal_id].constraints],
            dependent=bool(step.depends_on),
        )
        text = step.query or step.query_template
        missing = missing_constraints(constraints, text)
        if (missing and step.output_slot and step.query and re.search(r"排名|排序", text)
                and all(re.search(r"(?:最高|最低|最多|最少)$", term) for term in missing)):
            # Retrieving the ranked candidate set is valid, but the bridge
            # still needs an explicit selection operator. Compile it from the
            # accepted goal instead of guessing a winning category.
            text = " ".join([text, *missing])
            step = replace(step, query=_normalize_plan_text(text, field_name="query"))
            validate_relation_bindings(question, [text])
            record_observation("query_planning_normalization", "normalization", "accepted", source="rule",
                               node_id=step.id, end_reason="compiled_selection_constraints", constraints=missing)
            missing = missing_constraints(constraints, text)
        if missing:
            raise QueryValidationError("goal_local_constraint_dropped", "node dropped goal-local constraints: " + ", ".join(missing),
                                       field="subqueries[].query_template" if step.query_template else "subqueries[].query", node_id=step.id,
                                       expected=constraints, actual=text, offending=missing)
        for goal_id in step.goal_ids:
            validation = validate_query_preservation(by_id[goal_id].text, text)
            missing_source_anchors = [anchor for anchor in validation.missing_anchors
                                      if anchor in question and anchor not in {"分别", "截至"}]
            if validation.semantic_conflicts or missing_source_anchors:
                raise QueryValidationError("goal_semantics_changed", "node changed goal semantics: " +
                                           ", ".join([*validation.semantic_conflicts, *missing_source_anchors]),
                                           field="subqueries[].query", node_id=step.id, goal_id=goal_id)
        literal = re.sub(r"\{[a-z][a-z0-9_]*\}", "", text)
        if validate_query_preservation(question, literal).added_risky_terms:
            raise QueryValidationError("node_invented_constraints", "node invented exact constraints", field="subqueries[].query", node_id=step.id)
        for goal_id in step.goal_ids:
            for dependency in by_id[goal_id].depends_on:
                if not owners[dependency].intersection(step.depends_on):
                    raise QueryValidationError("goal_dependency_dropped", "plan lost a goal dependency", field="subqueries[].depends_on", node_id=step.id, goal_id=goal_id)
        validated.append(replace(step, required_constraints=constraints))
    return validated


def _validate_steps(steps: list[QuerySubqueryPlan], question: str, route: RetrievalRoute, goals) -> list[QuerySubqueryPlan]:
    if any(goal.candidate_locators for goal in goals):
        expected = _candidate_read_steps(question, goals)
        if route.mode != "parallel" or steps != expected:
            raise QueryValidationError("invalid_candidate_reads", "candidate reads must match the server-compiled object scopes", field="subqueries")
        validate_goals(question, goals)
        validate_dependency_graph(steps)
        return steps
    if route.mode == "parallel":
        _validate_parallel_plan(steps)
        _validate_parallel_subquery_drift(question, steps, goals=goals)
    elif route.mode == "sequential":
        _validate_sequential_plan(steps)
    validate_dependency_graph(steps)
    return _validate_goal_coverage(question, steps, goals)


def _validated_steps(content: str, question: str, route: RetrievalRoute, goals) -> list[QuerySubqueryPlan]:
    return _validate_steps(_parse_plan_steps(content), question, route, goals)


def _candidate_read_steps(question: str, goals: tuple[QueryGoal, ...]) -> list[QuerySubqueryPlan]:
    shared = shared_scope_terms(question)
    return [QuerySubqueryPlan(id=f"candidate_{index + 1}",
                             query=" ".join([locator, "完整内容 文字及视觉事实", *shared]),
                             goal_ids=[goal.id], required_constraints=[locator, *shared])
            for index, (goal, locator) in enumerate(
                (goal, locator) for goal in goals for locator in goal.candidate_locators)]


def _rating_sequential_plan(question: str) -> RetrievalPlan:
    return RetrievalPlan(
        mode="sequential",
        source="rule",
        subqueries=[
            QuerySubqueryPlan(
                id="rating_lookup",
                query=question,
                output_slot="rating",
            ),
            QuerySubqueryPlan(
                id="rating_definition",
                query_template=(
                    "“{rating}”的评级定义、评级标准、股价表现与"
                    "市场代表性指数是什么？"
                ),
                depends_on=["rating_lookup"],
                required_slots=["rating"],
                inherit_document_scope=(
                    "这份研报" in question
                    or "该研报" in question
                    or "这份报告" in question
                    or "该报告" in question
                ),
            ),
        ],
    )


def build_retrieval_plan(
    question: str,
    route: RetrievalRoute,
    *,
    client: Any | None = None,
    goals: tuple[QueryGoal, ...] = (),
) -> RetrievalPlan:
    """Build a plan only after the router has fixed the retrieval mode."""

    if route.mode in {"single", "clarify"}:
        return RetrievalPlan(mode=route.mode)
    if any(goal.candidate_locators for goal in goals):
        steps = _validate_steps(_candidate_read_steps(question, goals), question, route, goals)
        record_observation("query_planning", "decision", "accepted", source="rule",
                           end_reason="compiled_candidate_reads", node_count=len(steps))
        return RetrievalPlan(mode="parallel", subqueries=steps, source="rule")
    if route.mode == "parallel" and goals:
        # Independent validated goals already describe executable searches.
        # Compile first; ask a second model only when compilation is rejected.
        steps = [QuerySubqueryPlan(id=goal.id, query=" ".join(dict.fromkeys([
            goal.text, *missing_constraints(required_constraint_terms(goal.constraints), goal.text),
        ])), goal_ids=[goal.id]) for goal in goals]
        try:
            with observation_context(stage_source="goal_compiler", source="rule"):
                steps = observe_contract(
                    "query_planning", json.dumps({"subqueries": [asdict(step) for step in steps]}),
                    parse=_parse_plan_steps, validate=lambda value: _validate_steps(value, question, route, goals),
                )
            record_observation("query_planning", "decision", "accepted", source="rule",
                               end_reason="compiled_validated_goals", node_count=len(steps))
            return RetrievalPlan(mode="parallel", subqueries=steps, source="rule")
        except ValueError as error:
            record_observation("query_planning_compile", "decision", "rejected", source="rule",
                               end_reason="needs_model_plan", **failure_details(error))
    if (
        route.mode == "sequential"
        and RATING_SEQUENTIAL_PATTERN.search(question)
        and not goals
    ):
        plan = _rating_sequential_plan(question)
        validate_dependency_graph(plan.subqueries)
        return plan

    try:
        resolved_client = client or get_query_intent_client()
        system_prompt = (
            PARALLEL_PLAN_SYSTEM_PROMPT
            if route.mode == "parallel"
            else (GRAPH_PLAN_SYSTEM_PROMPT if route.mode == "graph" else SEQUENTIAL_PLAN_SYSTEM_PROMPT)
        )
        if goals:
            system_prompt += (
                '\n每个节点必须包含"goal_ids":["输入目标的id"]，完整覆盖所有目标。'
                '保留各目标适用的主体、时间、数值、单位和业务口径；constraints为来源证据，'
                '允许口语改为专业术语。依赖节点用槽位替换“该类电源”等指代。'
                '不要把某个目标的局部时间扩散到所有节点，也不要提前解答未知对应关系。'
            )
        subqueries = validated_completion(resolved_client, "query_planning",
            parse=_parse_plan_steps,
            validate=lambda steps: _validate_steps(steps, question, route, goals),
            model=query_intent_model(),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps({"question": question, "goals": [g.to_dict() for g in goals]}, ensure_ascii=False) if goals else question},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=2200,
            stream=False,
            timeout=query_intent_timeout_seconds(),
            extra_body={"enable_thinking": False},
        )
        return RetrievalPlan(
            mode=route.mode,
            subqueries=subqueries,
            source="model",
        )
    except Exception as error:
        logger.warning(
            "RAG %s plan generation failed: %s: %s",
            route.mode,
            type(error).__name__,
            str(error)[:500],
        )
        # Independent compilation has already been tried before the model.
        # Repeating the same rejected deterministic plan cannot repair it.
        record_observation("query_planning_fallback", "fallback", "applied", source="fallback",
                           end_reason="single_query_fallback", requested_mode=route.mode,
                           selected_mode="single", **failure_details(error))
        return RetrievalPlan(
            mode="single",
            source="fallback",
            fallback_reason=(
                f"{route.mode} 查询计划生成或保真校验失败"
                f"（{type(error).__name__}: {str(error)[:500]}）"
            ),
        )


def prepare_retrieval_request(
    question: str,
    *,
    client: Any | None = None,
) -> PreparedRetrievalRequest:
    """Coordinate analysis, routing, and planning before execution."""

    analysis = analyze_query_intent(question, client=client)
    record_observation("query_analysis_result", "decision", "selected", source=analysis.source,
                       end_reason=analysis.source, intent=analysis.intent, goal_count=len(analysis.goals))
    route = select_retrieval_route(analysis)
    plan = build_retrieval_plan(question, route, client=client, goals=analysis.goals)
    record_observation("query_plan_result", "decision", "selected", source=plan.source,
                       end_reason="fallback_selected" if plan.fallback_reason else "plan_selected",
                       requested_mode=route.mode, selected_mode=plan.mode, node_count=len(plan.subqueries),
                       fallback_reason=plan.fallback_reason)
    return PreparedRetrievalRequest(
        analysis=analysis,
        route=route,
        plan=plan,
    )
