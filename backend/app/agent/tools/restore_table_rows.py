"""恢复本轮已授权检索中因上下文容量裁剪的命中行，不访问外部存储。"""

from agent.schemas import ToolResult
from domain.evidence import EvidenceCollection, plain
from service.core.evidence.adapter import structure_record
from service.core.rag.utils import num_tokens_from_string
import json

MAX_RECOVERY_TOKENS = 4096


class RestoreTableRowsTool:
    name = "restore_table_rows"
    description = "恢复本轮搜索中因上下文容量裁剪的表格命中行；query 必须是 observations 中的 recovery_ref。不是重新搜索，线索不是可回答证据。每次恢复一行及原始上下文。"
    parameters = {"type": "object", "properties": {
        "query": {"type": "string", "description": "逐字复制本轮 observation 的 recovery_ref"}},
        "required": ["query"], "additionalProperties": False}

    def run(self, *, query):
        return ToolResult.failure(tool_name=self.name, query=query, error="恢复必须绑定当前运行的已完成检索")

    def run_from_history(self, context, *, task, completed):
        matches = [item for state in completed.values()
                   if state.result and state.result.success and state.task.tool == "search_knowledge_base"
                   for item in state.result.recoverable_evidence if item.id == task.query]
        if not matches or len({item.version for item in matches}) != 1:
            return ToolResult.failure(tool_name=self.name, query=task.query, error="恢复编号不属于本轮已完成检索，或来源版本冲突")
        item = matches[0]
        table = item.structure.table
        if (item.source.kind != "knowledge_base" or table.binding != "bound" or not table.reliable
                or len(table.rows) != 1 or not table.rows[0].grounded):
            return ToolResult.failure(tool_name=self.name, query=task.query, error="恢复行缺少可靠的原始表格结构")
        payload = json.dumps({"source": plain(item.source), "content": item.text,
                              "structure": structure_record(item.structure)}, ensure_ascii=False)
        if num_tokens_from_string(payload) > MAX_RECOVERY_TOKENS:
            return ToolResult.failure(tool_name=self.name, query=task.query,
                                      error="单行及其上下文超过恢复预算，未截断或猜测单元格")
        return ToolResult(self.name, task.query, item.text, evidence=EvidenceCollection([item]), task=task,
                          diagnostics={"recovery_ref": item.id, "recovery_reason": "context_capacity"})


def recovery_hints(result):
    """仅提供来源和原始非数字单元格作为定位线索，不将遗漏值交给规划器。"""
    hints = []
    if result:
        for item in result.recoverable_evidence[:40]:
            if not item.structure.table.rows:
                continue
            row = item.structure.table.rows[0]
            labels = [cell for cell in row.original_cells if any(char.isalpha() for char in cell)]
            hints.append({"recovery_ref": item.id, "source": item.source.name,
                          "table_id": item.location.table_id, "row_index": row.index,
                          "row_labels": [label[:120] for label in labels[:6]], "reason": "context_capacity",
                          "status": "not_evaluated"})
    return hints
