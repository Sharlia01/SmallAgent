"""查询期表格对象：来源快照、命中候选和展示上下文互不改写。"""
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from domain.evidence import freeze


@dataclass(frozen=True)
class IndexedTableRow:
    chunk_id: str
    source: Mapping[str, Any]

    def __post_init__(self):
        object.__setattr__(self, 'source', freeze(self.source))


@dataclass(frozen=True)
class TableCandidate:
    candidate_id: str  # 表格组候选编号，由表格来源和选中行的 chunk ID 生成，用于对外标识该候选
    source_rows: tuple[IndexedTableRow, ...]  # 本组最终入选记录的不可变来源快照，首条为最终排名最高的代表行
    matched_rows: tuple[Mapping[str, Any], ...]  # 筛选后保留的命中行，含行号、表头、单元格、语义文本等
    member_chunk_ids: tuple[str, ...]  # 本组最终入选的原始 chunk ID，按最终排名顺序去重
    omitted_chunk_ids: tuple[str, ...]  # 本组成员中因重复行等未进入 matched_rows 的 chunk ID
    scored_row_indices: tuple[int, ...]  # 已评分的行的组内顺序，保存的是表格行号，不是chunk ID
    ranking_source: str  # 命中行评分来源：model 为统一重排模型评分，skipped 为未提供评分
    ranking_text: str  # 最终命中行的语义文本拼接后按 token 预算截断，供下游使用；不再执行整组重排
    ranking_text_truncated: bool  # 拼接后的语义文本是否因超出 token 预算而被截断，不表示是否遗漏了成员行
    caption: str  # 从代表行的结构化表格数据中读取的表题，缺失时为空字符串

    def __post_init__(self):
        object.__setattr__(self, 'matched_rows', tuple(freeze(row) for row in self.matched_rows))

    @property
    def source(self) -> Mapping[str, Any]:
        """排序代表行仅提供来源身份，不能冒充整个候选的内容。"""
        return self.source_rows[0].source

    @property
    def matched_row_indices(self) -> tuple[int, ...]:
        return tuple(row['row_index'] for row in self.matched_rows)


@dataclass(frozen=True)
class TableContext:
    candidate: TableCandidate | IndexedTableRow
    text: str
    rows: tuple[Mapping[str, Any], ...]
    diagnostics: Mapping[str, Any]
    recovery_rows: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self):
        object.__setattr__(self, 'rows', tuple(freeze(row) for row in self.rows))
        object.__setattr__(self, 'diagnostics', freeze(self.diagnostics))
        object.__setattr__(self, 'recovery_rows', tuple(freeze(row) for row in self.recovery_rows))

    @property
    def matched_rows(self):
        indices = set(self.diagnostics.get("matched_row_indices", ()))
        return tuple(row for row in self.rows if row['row_index'] in indices)

    @property
    def auxiliary_rows(self):
        indices = set(self.diagnostics.get("matched_row_indices", ()))
        return tuple(row for row in self.rows if row['row_index'] not in indices)
