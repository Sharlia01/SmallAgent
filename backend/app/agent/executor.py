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


@dataclass
class TaskExecution:
    task: AgentTask
    status: str
    result: ToolResult | None = None #某一个任务的工具结果，和任务绑定
    binding: BridgeExtraction | None = None
    error: str | None = None

    def observation(self, *, include_content: bool = True) -> dict[str, Any]:
        # 重规划只需状态、桥接来源和有限片段；完整证据仍保存在运行状态中供核验。
        result = self.result
        snippets = []
        if result is not None and include_content:
            evidence = result.candidates.copy()
            evidence.merge(result.evidence)
            snippets = [{"id": item.id, "source": item.source.name, "text": item.text[:800]}
                        for item in list(evidence)[:3]]
        return {
            "task_id": self.task.id, "status": self.status, "error": self.error or (result.error if result else None),
            "binding": self.binding.to_dict() if self.binding else None,
            "query": result.query if result else None,
            "result_count": len(result.evidence) if result else 0,
            "snippets": snippets,
        }


class ToolExecutor:
    def __init__(self, tools, *, bridge_extractor=None):
        self.tools = {tool.name: tool for tool in tools}
        if len(self.tools) != len(tools):
            raise ValueError("工具名称不能重复")
        self._bridge_extractor = bridge_extractor or extract_bridge_value

    def definitions(self):
        return [{"type": "function", "function": {
            "name": tool.name, "description": tool.description, "parameters": tool.parameters,
        }} for tool in self.tools.values()]

    def execute_task(self, task: AgentTask, context: QueryContext,
                     completed: dict[str, TaskExecution], cache: dict) -> TaskExecution:
        parents = [completed[parent] for parent in task.depends_on]
        if any(parent.binding is None for parent in parents):
            return TaskExecution(task, "blocked", error="前置任务尚未取得有来源的变量值")
        slots = {parent.binding.slot: parent.binding.value for parent in parents}
        query = resolve_query_template(task.query_template, slots) if task.query_template else task.query
        # 检索使用与财务要求相同的参考年份，不能让“去年”在两层含义不同。
        query = normalize_relative_years(query, context.request.reference_date)
        document_ids = ()
        if task.inherit_document_scope:
            scopes = {parent.binding.document_id for parent in parents}
            if len(scopes) != 1 or not all(scopes):
                return TaskExecution(task, "blocked", error="前置证据的文档范围缺失或冲突")
            # 内部文档标识不能从网页来源继承。
            if any(parent.task.tool != "search_knowledge_base" for parent in parents):
                return TaskExecution(task, "blocked", error="内部文档范围只能来自知识库任务")
            document_ids = tuple(sorted(scopes))
        request = RetrievalTask(query, task.goal_ids, document_ids)
        identity = (task.tool, query, document_ids, task.top_k)
        if identity in cache:
            result = replace(cache[identity], task=request)
            status = "cached"
        else:
            tool = self.tools[task.tool]
            try:
                if task.tool != "search_knowledge_base" and current_runtime() is not None:
                    current_runtime().reserve("retrieval")
                kwargs = {"top_k": task.top_k} if "top_k" in tool.parameters.get("properties", {}) else {}
                if callable(getattr(tool, "run_for_task", None)):
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
            status = "succeeded" if result.success and (result.evidence or result.candidates) else "empty" if result.success else "failed"
            if result.success:
                # 空结果同样缓存；同词重复搜索不会带来新证据，须由 Planner 改变策略。
                cache[identity] = result
        execution = TaskExecution(task, status, result)
        if task.output_slot and result.success:
            evidence = result.candidates.copy()
            evidence.merge(result.evidence)
            if evidence:
                try:
                    execution.binding = self._bridge_extractor(
                        slot=task.output_slot, original_question=context.question,
                        first_query=query, chunks=reference_records(evidence),
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
