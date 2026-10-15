"""Agent 的执行计划和确定性校验；本模块不调用模型或检索服务。"""

from dataclasses import asdict, dataclass
from string import Formatter
from typing import Any
import re

from service.core.query.contract import QueryUnderstanding
from service.core.query.context import QueryContext
from service.core.query.preservation import validate_query_preservation, extract_query_anchors
from service.core.query.semantics import canonical_text
from service.core.query.temporal import normalize_relative_years
from service.core.query.validation import (
    QueryValidationError, missing_constraints, required_constraint_terms,
    validate_relation_bindings, shared_scope_terms,
)

MAX_SUBQUERIES = 8
MAX_SUBQUERY_CHARS = 2000
PLAN_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
SAME_DOCUMENT = re.compile(r"同一(?:份)?(?:报告|研报|文档)")

@dataclass(frozen=True)
class AgentTask:
    """尚未执行的任务。模板只能引用声明的前置任务，不能提前猜测结果。"""

    id: str  #当前任务的唯一编号，补查时必须使用新编号
    tool: str = "search_knowledge_base"  #要调用的工具名称，必须是已注册工具
    query: str = ""   #可以直接执行的查询文本
    query_template: str = ""  #等待前置结果填入的查询模板 如"{company}最新公告" 与 query 二选一，不能同时填写。
    depends_on: tuple[str, ...] = ()  #当前任务依赖的前置任务编号，要求前置任务提供所需变量，不只是控制执行先后。
    output_slot: str | None = None  #从本任务证据中提取并输出的变量名称，实际值要从检索证据中提取，可以被后续任务的 query_template 引用。
    required_slots: tuple[str, ...] = ()  #当前模板需要的变量名称
    inherit_document_scope: bool = False #是否限定在前置证据所在的同一文档内检索
    goal_ids: tuple[str, ...] = ()  #当前任务服务的用户目标编号
    top_k: int = 5                  #希望工具返回的结果数量
    binding_key: str | None = None  # 补查输出绑定到固定需求的引用名，不改写原始需求。

    def to_dict(self):
        return asdict(self)

@dataclass(frozen=True)
class AgentPlan:
    tasks: tuple[AgentTask, ...] = ()
    clarification: str | None = None

    def to_dict(self):
        return {"tasks": [task.to_dict() for task in self.tasks], "clarification": self.clarification}

@dataclass(frozen=True)
class InitialDecision:
    understanding: QueryUnderstanding
    plan: AgentPlan



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

def validate_dependency_graph(subqueries: list[AgentTask]) -> list[list[AgentTask]]:
    """开始检索前校验全部模板和依赖关系。"""
    if not 1 <= len(subqueries) <= 80:
        raise QueryValidationError("invalid_graph_size", "invalid graph size", field="subqueries")
    by_id = {step.id: step for step in subqueries}
    if len(by_id) != len(subqueries):
        raise QueryValidationError("duplicate_graph_node", "duplicate graph node", field="subqueries[].id")
    producers = {}
    #检查编号重复、依赖不存在、循环依赖、模板变量与前置输出不匹配等问题；不合法就抛异常。
    for step in subqueries:
        if step.output_slot:
            if step.output_slot in producers:
                raise QueryValidationError("duplicate_output_slot", "duplicate output slot", field="subqueries[].output_slot", node_id=step.id)
            producers[step.output_slot] = step.id
        if bool(step.query) == bool(step.query_template):
            raise QueryValidationError("query_template_exclusivity", "node needs exactly one query or template", field="subqueries[].query_template", node_id=step.id)
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
    # 生成执行批次：把依赖关系转换为执行器可以逐批处理的结构。
    completed, waves = set(), []
    while len(completed) < len(subqueries):
        ready = [step for step in subqueries if step.id not in completed and set(step.depends_on) <= completed]
        if not ready:
            raise QueryValidationError("cyclic_dependency_graph", "cyclic dependency graph", field="subqueries[].depends_on")
        waves.append(ready)
        completed.update(step.id for step in ready)
    return waves


def decode_plan(payload: Any) -> AgentPlan:
    if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), list):
        raise QueryValidationError("invalid_plan", "计划必须包含 tasks 列表")
    if set(payload) - {"tasks", "clarification"}:
        raise QueryValidationError("unexpected_plan_field", "计划不能修改用户需求或注入运行上下文")
    if len(payload["tasks"]) > MAX_SUBQUERIES:
        raise QueryValidationError("too_many_tasks", "单次计划最多8个任务")
    clarification = payload.get("clarification")
    if clarification is not None and (not isinstance(clarification, str) or not clarification.strip() or len(clarification) > 200):
        raise QueryValidationError("invalid_clarification", "澄清问题必须是简短文本")
    tasks = []
    for raw in payload["tasks"]:
        if not isinstance(raw, dict):
            raise QueryValidationError("invalid_task", "任务必须是对象")
        allowed = {"id", "tool", "query", "query_template", "depends_on", "output_slot", "binding_key",
                   "required_slots", "inherit_document_scope", "goal_ids", "top_k"}
        if set(raw) - allowed:
            raise QueryValidationError("unexpected_task_field", "任务含有未声明字段")
        identifier, tool = raw.get("id"), raw.get("tool")
        if not isinstance(identifier, str) or not PLAN_ID_PATTERN.fullmatch(identifier) or not isinstance(tool, str):
            raise QueryValidationError("invalid_task_identity", "任务编号或工具名称无效")
        output = raw.get("output_slot")
        if output is not None and (not isinstance(output, str) or not PLAN_ID_PATTERN.fullmatch(output)):
            raise QueryValidationError("invalid_output_slot", "输出变量名称无效")
        binding_key = raw.get("binding_key")
        if binding_key is not None and (not isinstance(binding_key, str) or not PLAN_ID_PATTERN.fullmatch(binding_key) or not output):
            raise QueryValidationError("invalid_binding_key", "固定需求绑定必须具有合法引用名和输出变量")
        scope, top_k = raw.get("inherit_document_scope", False), raw.get("top_k", 5)
        if type(scope) is not bool or type(top_k) is not int or not 1 <= top_k <= 20:
            raise QueryValidationError("invalid_options", "文档范围或结果数量无效")
        tasks.append(AgentTask(
            id=identifier, tool=tool,
            query=_normalize_plan_text(raw.get("query", ""), field_name="query"),
            query_template=_normalize_plan_text(raw.get("query_template", ""), field_name="query_template"),
            depends_on=tuple(_parse_identifier_list(raw.get("depends_on", []), field_name="depends_on")),
            output_slot=output,
            binding_key=binding_key,
            required_slots=tuple(_parse_identifier_list(raw.get("required_slots", []), field_name="required_slots")),
            inherit_document_scope=scope,
            goal_ids=tuple(_parse_identifier_list(raw.get("goal_ids", []), field_name="goal_ids")),
            top_k=top_k,
        ))
    if clarification and tasks:
        raise QueryValidationError("clarification_with_tasks", "需要澄清时不得同时提交执行任务")
    return AgentPlan(tuple(tasks), clarification)


def validate_plan(plan: AgentPlan, context: QueryContext, tools: set[str], *,
                  previous: tuple[AgentTask, ...] = (), initial: bool = False) -> AgentPlan:
    """需求保持不变；重规划可以补充任务，但不能覆盖旧任务或引用未知目标。"""
    if plan.clarification:
        return plan
    financial = context.understanding.financial
    if financial:
        producers = {key for task in (*previous, *plan.tasks) for key in (task.output_slot, task.binding_key) if key}
        references = {ref for item in financial.facts for ref in (item.entity_ref, item.period_ref) if ref}
        if references - producers - {"document_subject"}:
            raise QueryValidationError("financial_binding_without_producer", "财务需求引用的变量必须由任务提供")
    references = {ref for item in financial.facts for ref in (item.entity_ref, item.period_ref) if ref} if financial else set()
    for task in plan.tasks:
        if task.output_slot == "document_subject" or task.binding_key == "document_subject":
            raise QueryValidationError("reserved_document_binding", "文档主体不能由普通任务变量覆盖")
        if task.binding_key and task.binding_key not in references:
            raise QueryValidationError("unknown_financial_binding", "绑定目标必须属于固定财务需求")
    if not plan.tasks:
        if initial and context.understanding.requires_evidence:
            # 会话文档也可能足够；是否充分仍由执行循环核验。
            return plan
        return plan
    previous_ids = {task.id for task in previous}
    if previous_ids.intersection(task.id for task in plan.tasks):
        raise QueryValidationError("task_id_reused", "补查必须使用新任务编号")
    combined = [*previous, *plan.tasks]
    tasks_by_id = {task.id: task for task in combined}
    # 总任务数由请求预算控制；单轮模型输出仍受8个任务的限制。
    validate_dependency_graph(combined)
    goals = {goal.id: goal for goal in context.goals}
    def absolute(text):
        return normalize_relative_years(text, context.request.reference_date)
    owners = {goal_id: {task.id for task in combined if goal_id in task.goal_ids} for goal_id in goals}
    identities = set()
    for task in plan.tasks:
        identity = (task.tool, canonical_text(task.query or task.query_template), task.depends_on, task.top_k)
        if identity in identities:
            raise QueryValidationError("duplicate_task", "同轮的相同搜索应合并目标后只执行一次")
        identities.add(identity)
        if task.tool not in tools:
            raise QueryValidationError("unknown_tool", "计划使用了未注册工具")
        if not task.goal_ids or not set(task.goal_ids) <= goals.keys():
            raise QueryValidationError("unknown_goal_ids", "任务必须引用已确认目标")
        if task.inherit_document_scope and task.tool != "search_knowledge_base":
            raise QueryValidationError("unsupported_document_scope", "网页工具不能继承内部文档范围")
        literal = absolute(re.sub(r"\{[a-z][a-z0-9_]*\}", "", task.query or task.query_template))
        reference = absolute("；".join(goals[g].text + " " + " ".join(goals[g].constraints) for g in task.goal_ids))
        same_document = SAME_DOCUMENT.findall(reference)
        if task.depends_on and same_document and not task.inherit_document_scope:
            raise QueryValidationError("document_scope_not_bound", "同一文档要求必须继承证据中的文档范围")
        scope_text = literal
        if task.inherit_document_scope:
            if any(tasks_by_id[parent].tool != "search_knowledge_base" for parent in task.depends_on):
                raise QueryValidationError("invalid_document_source", "内部文档范围只能继承知识库证据")
            # 文档范围由服务端过滤器落实，不必强迫检索词重复“同一报告”。
            scope_text += " " + " ".join(same_document)
        # 拆分年份或指标允许分多次检索；每次仍须保留共享主体、来源和排除条件。
        if missing_constraints(shared_scope_terms(reference), literal):
            raise QueryValidationError("task_scope_dropped", "子查询遗漏共享范围约束")
        source_constraints = [absolute(constraint) for g in task.goal_ids for constraint in goals[g].constraints
                              if re.search(r"仅|只查|不含|不包括|不考虑|同一|这份|该报告", constraint)]
        if missing_constraints(required_constraint_terms(source_constraints, dependent=bool(task.depends_on)), scope_text):
            raise QueryValidationError("task_scope_dropped", "子查询遗漏来源或排除要求")
        if task.tool == "search_web" and any(re.search(r"仅|只", constraint) and
                                               re.search(r"内部|知识库|文档|报告|研报", constraint)
                                               for constraint in source_constraints):
            raise QueryValidationError("source_scope_changed", "限定内部来源的任务不能改用网页搜索")
        preservation = validate_query_preservation(reference, literal)
        if preservation.added_risky_terms or preservation.semantic_conflicts:
            raise QueryValidationError("task_query_drift", "任务新增了无依据条件或改变了指标含义")
        validate_relation_bindings(absolute(context.request.resolved_question), [literal])
        for goal_id in task.goal_ids:
            for dependency in goals[goal_id].depends_on:
                if dependency not in task.goal_ids and not owners[dependency].intersection(task.depends_on):
                    raise QueryValidationError("goal_dependency_dropped", "任务遗漏了目标依赖")
    if initial:
        if {g for task in combined for g in task.goal_ids} != set(goals):
            raise QueryValidationError("goal_coverage_mismatch", "初始计划必须覆盖全部目标")
    # 一个比较目标可以分两次搜索；按目标合并检查，避免要求每个子查询重复全部年份。
    for goal_id in {g for task in plan.tasks for g in task.goal_ids}:
        goal = goals[goal_id]
        texts = absolute("；".join(task.query or task.query_template for task in combined if goal_id in task.goal_ids))
        if any(task.inherit_document_scope for task in combined if goal_id in task.goal_ids):
            texts += " " + " ".join(SAME_DOCUMENT.findall(goal.text + " " + " ".join(goal.constraints)))
        terms = required_constraint_terms([absolute(value) for value in goal.constraints], dependent=bool(goal.depends_on))
        terms += [a for a in extract_query_anchors(absolute(goal.text)) if a not in {"分别", "截至"}]
        if missing_constraints(terms, texts):
            raise QueryValidationError("goal_constraints_dropped", "目标查询集合遗漏约束")
    return plan
