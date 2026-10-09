"""原始证据、证据集合及独立的回答上下文。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, fields
from collections.abc import Iterable, Iterator, Mapping
from types import MappingProxyType
from typing import Any, Literal

SourceKind = Literal["knowledge_base", "web", "session_document"]


def freeze(value: Any) -> Any:
    """复制并冻结嵌套来源数据，避免调用方修改已经核验的快照。"""
    if isinstance(value, Mapping):
        return MappingProxyType({key: freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze(item) for item in value)
    return value


def plain(value: Any) -> Any:
    """仅在适配和序列化边界恢复 JSON 数据。"""
    if isinstance(value, Mapping):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return {item.name: plain(getattr(value, item.name)) for item in fields(value)}
    return value


def fingerprint(value: Any) -> str:
    encoded = json.dumps(plain(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class EvidenceSource:
    kind: SourceKind
    document_id: str
    name: str
    url: str | None = None
    knowledge_base_id: str = ""
    image_id: str = ""
    # 搜索摘要的网页链接是出处线索，不能被当作已经读取的网页正文。
    links: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self):
        if self.kind not in {"knowledge_base", "web", "session_document"}:
            raise ValueError("不支持的证据来源类型")
        object.__setattr__(self, "links", freeze(self.links))


@dataclass(frozen=True, slots=True)
class EvidenceLocation:
    positions: tuple[tuple[int | float, ...], ...] = ()
    # 索引中的页码可能是单值，也可能是跨页片段的页码列表。
    page_number: int | tuple[int, ...] | None = None
    table_id: str | None = None
    row: int | None = None
    row_start: int | None = None
    row_end: int | None = None

    def __post_init__(self):
        for item in fields(self):
            object.__setattr__(self, item.name, freeze(getattr(self, item.name)))


@dataclass(frozen=True, slots=True)
class EvidenceTableRow:
    """表格行及其在原表中的映射，校验器不读取索引字段。"""

    headers: tuple[str, ...] = ()
    cells: tuple[str, ...] = ()
    index: int | None = None
    columns: tuple[int, ...] = ()
    original_headers: tuple[str, ...] = ()
    original_cells: tuple[str, ...] = ()
    context: str = ""
    preamble: str = ""

    def __post_init__(self):
        for name in ("headers", "cells", "columns", "original_headers", "original_cells"):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    @property
    def valid(self) -> bool:
        return bool(self.cells) and len(self.headers) == len(self.cells) == len(self.columns)

    @property
    def grounded(self) -> bool:
        return (self.valid and len(set(self.columns)) == len(self.columns)
                and all(type(column) is int and 0 <= column < len(self.original_cells)
                        and column < len(self.original_headers)
                        and self.cells[index] == self.original_cells[column]
                        and self.headers[index] == self.original_headers[column]
                        for index, column in enumerate(self.columns)))

    def original(self) -> EvidenceTableRow:
        return EvidenceTableRow(self.original_headers, self.original_cells, self.index,
                                tuple(range(len(self.original_cells))),
                                self.original_headers, self.original_cells, self.context, self.preamble)


@dataclass(frozen=True, slots=True)
class EvidenceTable:
    binding: str = "unbound"
    binding_source: str = "deterministic"
    reliable: bool = True
    caption: str = ""
    rows: tuple[EvidenceTableRow, ...] = ()
    # 解析器附加数据仅供边界输出；业务读取上面的规范字段。
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "rows", tuple(self.rows))
        object.__setattr__(self, "metadata", freeze(self.metadata))


@dataclass(frozen=True, slots=True)
class EvidenceFigure:
    status: str = "unextracted"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "metadata", freeze(self.metadata))


@dataclass(frozen=True, slots=True)
class EvidenceAttribution:
    subject_ids: tuple[str, ...] = ()
    subject_status: str = "unknown"
    subject_proof: tuple[Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "subject_ids", tuple(self.subject_ids))
        object.__setattr__(self, "subject_proof", freeze(self.subject_proof))
        object.__setattr__(self, "metadata", freeze(self.metadata))


@dataclass(frozen=True, slots=True)
class EvidenceStructure:
    """解析器提供的可核查结构；只存来源数据，不存本次查询的结论。

    来源适配器将不同解析器版本收口为表格、图表和归属类型。
    核验器直接读取规范字段，解析器附加数据仅在输出边界使用。
    """

    kind: str = "text"
    table: EvidenceTable = field(default_factory=EvidenceTable)
    figure: EvidenceFigure = field(default_factory=EvidenceFigure)
    attribution: EvidenceAttribution = field(default_factory=EvidenceAttribution)
    locator: str = ""
    parser_version: str = ""
    coordinates: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "coordinates", freeze(self.coordinates))


@dataclass(frozen=True, slots=True)
class RetrievalTrace:
    """某次检索命中记录，排名和分数不改变原始证据的身份。"""

    query: str = ""
    scores: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "scores", freeze(self.scores))


@dataclass(frozen=True, slots=True)
class Evidence:
    id: str
    text: str
    source: EvidenceSource
    location: EvidenceLocation = field(default_factory=EvidenceLocation)
    structure: EvidenceStructure = field(default_factory=EvidenceStructure)

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("证据必须有非空的稳定标识")
        if not isinstance(self.text, str):
            raise ValueError("证据原文必须是字符串")

    @property
    def version(self) -> str:
        return fingerprint(self)


@dataclass(frozen=True, slots=True)
class EvidenceConflict:
    evidence_id: str
    retained_version: str
    incoming_version: str


class EvidenceCollection:
    """按稳定标识合并证据，集中维护顺序、检索轨迹和身份冲突。"""

    def __init__(self, evidence: Iterable[Evidence] = ()):
        self._items: dict[str, Evidence] = {}
        self._traces: dict[str, list[RetrievalTrace]] = {}
        self._conflicts: list[EvidenceConflict] = []
        for item in evidence:
            self.add(item)

    def __iter__(self) -> Iterator[Evidence]:
        return iter(self._items.values())

    def __len__(self) -> int:
        return len(self._items)

    def get(self, evidence_id: str) -> Evidence:
        return self._items[evidence_id]

    def traces(self, evidence_id: str) -> tuple[RetrievalTrace, ...]:
        return tuple(self._traces.get(evidence_id, ()))

    @property
    def conflicts(self) -> tuple[EvidenceConflict, ...]:
        return tuple(self._conflicts)

    @property
    def version(self) -> str:
        # 顺序也影响引用，改变顺序必须让旧判定失效。
        return fingerprint(([item.version for item in self], self.conflicts))

    def add(self, item: Evidence, trace: RetrievalTrace | None = None) -> None:
        previous = self._items.get(item.id)
        if previous is not None and previous.version != item.version:
            conflict = EvidenceConflict(item.id, previous.version, item.version)
            if conflict not in self._conflicts:
                self._conflicts.append(conflict)
        else:
            self._items.setdefault(item.id, item)
        if trace is not None and trace not in self._traces.setdefault(item.id, []):
            self._traces[item.id].append(trace)

    def merge(self, other: EvidenceCollection) -> None:
        for item in other:
            self.add(item)
            for trace in other.traces(item.id):
                self.add(item, trace)
        for conflict in other.conflicts:
            if conflict not in self._conflicts:
                self._conflicts.append(conflict)

    def copy(self) -> EvidenceCollection:
        result = EvidenceCollection()
        result.merge(self)
        return result

    def select(self, identifiers: Iterable[str]) -> EvidenceCollection:
        """按指定顺序选取原对象，保留检索轨迹和选中来源的身份冲突。"""
        result = EvidenceCollection()
        for identifier in identifiers:
            item = self.get(identifier)
            result.add(item)
            for trace in self.traces(identifier):
                result.add(item, trace)
        result._conflicts = [item for item in self.conflicts if item.evidence_id in result._items]
        return result


@dataclass(frozen=True, slots=True)
class AnswerBlock:
    evidence_id: str
    text: str
    fact_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Citation:
    number: int
    evidence: Evidence


@dataclass(frozen=True, slots=True)
class AnswerContext:
    """筛选完成的模型输入；原文只通过引用映射提供给展示层。"""

    question: str
    blocks: tuple[AnswerBlock, ...]
    citations: tuple[Citation, ...]
    financial: bool = False

    def __post_init__(self):
        ids = {citation.evidence.id for citation in self.citations}
        if len(ids) != len(self.citations) or any(block.evidence_id not in ids for block in self.blocks):
            raise ValueError("回答上下文的引用映射不完整或存在重复来源")
        if tuple(item.number for item in self.citations) != tuple(range(1, len(self.citations) + 1)):
            raise ValueError("引用编号必须连续且从1开始")
