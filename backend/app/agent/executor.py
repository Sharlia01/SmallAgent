"""执行已校验的 Agent 任务；依赖结果由证据绑定，工具不再自行规划。"""

from dataclasses import dataclass, replace
from typing import Any

from agent.plans import AgentTask, AgentPlan, validate_dependency_graph
from agent.schemas import ToolResult
from service.core.evidence.adapter import reference_records
from service.core.query.context import QueryContext
from service.core.query.contract import RetrievalTask
from service.core.query.temporal import normalize_relative_years
from service.core.evidence.bindings import BridgeExtraction, extract_bridge_value, resolve_query_template
from service.core.runtime.execution import run_independent
from service.core.runtime.context import QueryBudgetExceeded, current_runtime
from agent.tools.restore_table_rows import RestoreTableRowsTool, recovery_hints


@dataclass
class TaskExecution:
    task: AgentTask
    status: str
    result: ToolResult | None = None #某一个任务的工具结果，和任务绑定
    binding: BridgeExtraction | None = None  # 给下一个依赖此任务的任务，下一个任务的变量和从证据中提取的实际值
    error: str | None = None
    cache_hit: bool = False

    def observation(self, *, include_content: bool = True) -> dict[str, Any]:
        # 重规划只需状态、桥接来源和有限片段；完整证据仍保存在运行状态中供核验。
        result = self.result
        snippets = []
        if result is not None and include_content:
            snippets = [{"id": item.id, "source": item.source.name, "text": item.text[:800]}
                        for item in list(result.evidence)[:3]]
        return {
            "task_id": self.task.id, "status": self.status, "error": self.error or (result.error if result else None),
            "cache_hit": self.cache_hit,
            "retrieval_status": ("succeeded" if result.success else "failed") if result else "not_executed",
            "binding_status": ("resolved" if self.binding else "unresolved") if self.task.output_slot else "not_requested",
            "binding": self.binding.to_dict() if self.binding else None,
            "query": result.query if result else None,
            "result_count": len(result.evidence) if result else 0,
            "snippets": snippets,
            "recoverable_rows": recovery_hints(result),
        }


class ToolExecutor:
    def __init__(self, tools, *, bridge_extractor=None):
        self.tools = {tool.name: tool for tool in tools}
        if len(self.tools) != len(tools):
            raise ValueError("工具名称不能重复")
        if "search_knowledge_base" in self.tools:
            self.tools.setdefault(RestoreTableRowsTool.name, RestoreTableRowsTool())
        self._bridge_extractor = bridge_extractor or extract_bridge_value

    def definitions(self):
        return [{"type": "function", "function": {
            "name": tool.name, "description": tool.description, "parameters": tool.parameters,
        }} for tool in self.tools.values()]

    def is_repeated_direct_plan(self, plan: AgentPlan, context: QueryContext, cache: dict) -> bool:
        """仅拦截整份均为成功缓存命中的直接搜索；依赖、绑定和恢复任务交给执行器。"""
        if not plan.tasks:
            return False
        for task in plan.tasks:
            if (task.tool not in {"search_knowledge_base", "search_web"}
                    or not task.query or task.query_template or task.depends_on
                    or task.output_slot or task.required_slots or task.binding_key or task.inherit_document_scope):
                return False
            if any(goal.subject_slot for goal in context.goals if goal.id in task.goal_ids):
                return False
            names = context.company_names(task.goal_ids, subject_refs=task.subject_refs)
            scope = None
            if names is not None:
                from service.core.retrieval.entities import company_hints_from_names
                scope = company_hints_from_names(names)
            query = normalize_relative_years(task.query, context.request.reference_date)
            result = cache.get((task.tool, query, (), task.top_k, scope))
            if result is None or not result.success:
                return False
        return True

    def execute_task(self, task: AgentTask, context: QueryContext,
                     completed: dict[str, TaskExecution], cache: dict) -> TaskExecution:
        # 获取前置任务的变量的绑定结果
        parents = [completed[parent] for parent in task.depends_on]
        if any(parent.binding is None for parent in parents):
            return TaskExecution(task, "blocked", error="前置任务尚未取得有来源的变量值")
        slots = {parent.binding.slot: parent.binding.value for parent in parents}
        # 将查询模板中的变量替换成前置任务的结果
        query = resolve_query_template(task.query_template, slots) if task.query_template else task.query
        # 依据请求的参考日期处理“去年”等相对年份
        if task.tool != RestoreTableRowsTool.name:
            query = normalize_relative_years(query, context.request.reference_date)

        # 如果任务要求在前置证据所在的同一份文档中继续查找，就检查
        document_ids = ()
        if task.inherit_document_scope:
            scopes = {parent.binding.document_id for parent in parents}
            if len(scopes) != 1 or not all(scopes):
                return TaskExecution(task, "blocked", error="前置证据的文档范围缺失或冲突")
            # 内部文档标识不能从网页来源继承。
            if any(parent.task.tool != "search_knowledge_base" for parent in parents):
                return TaskExecution(task, "blocked", error="内部文档范围只能来自知识库任务")
            document_ids = tuple(sorted(scopes))
        names = context.company_names(task.goal_ids, subject_refs=task.subject_refs)
        scope = None
        if names is not None:
            from service.core.retrieval.entities import company_hints_from_names
            subject_slots = {g.subject_slot for g in context.goals if g.id in task.goal_ids
                             and g.subject_slot and g.subject_slot != (task.binding_key or task.output_slot)}
            subject_values = {parent.task.binding_key or parent.binding.slot: parent.binding.value
                              for parent in parents}
            # 主体必须来自当前任务实际消费的前置证据，不能猜测或借用其他分支。
            if subject_slots - subject_values.keys():
                return TaskExecution(task, "blocked", error="主体变量尚无经过来源校验的证据绑定")
            names = tuple(dict.fromkeys((*names, *(subject_values[slot] for slot in sorted(subject_slots)))))
            scope = company_hints_from_names(names)
        request = RetrievalTask(query, task.goal_ids, document_ids, scope)
        identity = (task.tool, query, document_ids, task.top_k, scope)
        if identity in cache:
            result = replace(cache[identity], task=request)
            status = "cached"
        else:
            # 没有缓存时，执行工具并记录结果
            tool = self.tools[task.tool]
            try:
                if task.tool != "search_knowledge_base" and current_runtime() is not None:
                    current_runtime().reserve("retrieval")
                kwargs = {"top_k": task.top_k} if "top_k" in tool.parameters.get("properties", {}) else {}
                if isinstance(tool, RestoreTableRowsTool):
                    result = tool.run_from_history(context, task=request, completed=completed)
                elif callable(getattr(tool, "run_for_task", None)):
                    result = tool.run_for_task(context, task=request, **kwargs)
                elif document_ids:
                    raise ValueError("工具不支持服务端文档范围")
                else:
                    result = tool.run(query=query, **kwargs)
                result.task = request
            except QueryBudgetExceeded:
                raise
            except Exception as error:
                result = ToolResult.failure(tool_name=task.tool, query=query,
                                            error=str(error)[:300], diagnostics={"error_type": type(error).__name__})
                result.task = request
            status = "succeeded" if result.success and result.evidence else "empty" if result.success else "failed"
            if result.success:
                # 空结果同样缓存；同词重复搜索不会带来新证据，须由 Planner 改变策略。
                cache[identity] = result
        execution = TaskExecution(task, status, result, cache_hit=status == "cached")
        # 从证据中提取输出变量，给依赖这个任务的下面的任务使用
        if task.output_slot and result.success:
            if result.evidence:
                try:
                    entity_options = {"entity_names": names} if self._bridge_extractor is extract_bridge_value else {}
                    execution.binding = self._bridge_extractor(
                        slot=task.output_slot, original_question=context.question,
                        first_query=query, chunks=reference_records(result.evidence), **entity_options,
                    )
                except QueryBudgetExceeded as error:
                    # 检索已完成，桥接预算耗尽不能抹去刚取得的证据。
                    execution.status = "budget_exhausted"
                    execution.error = str(error)[:300]
                    return execution
                except Exception as error:
                    execution.error = type(error).__name__
            if execution.binding is None:
                execution.status = "unresolved_binding"
        return execution

    def execute_plan(self, plan: AgentPlan, context: QueryContext,
                     completed: dict[str, TaskExecution], cache: dict) -> list[TaskExecution]:
        """按依赖分批执行；失败分支不覆盖已经成功的兄弟分支。"""
        all_tasks = [state.task for state in completed.values()] + list(plan.tasks)
        waves = validate_dependency_graph(all_tasks) if all_tasks else []
        results = []
        for wave in waves:
            # pending表示本轮尚未完成的任务
            pending = [task for task in wave if task.id not in completed]
            if not pending:
                continue
            # 同波任务不共享可变缓存写操作；完成后按计划顺序合并。
            snapshots = {task.id: dict(cache) for task in pending}
            outputs = run_independent({task.id: lambda task=task: self.execute_task(
                task, context, completed, snapshots[task.id]) for task in pending})
            for task in pending:
                value = outputs[task.id]
                if isinstance(value, Exception):
                    value = TaskExecution(task, "budget_exhausted" if isinstance(value, QueryBudgetExceeded) else "failed",
                                          error=str(value)[:300])
                cache.update(snapshots[task.id])
                completed[task.id] = value
                results.append(value)
        return results
