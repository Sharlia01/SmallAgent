"""入库阶段使用保守规则进行实体标注，不调用模型。

报告主体、发布方、正文提及和表格明确归属是不同角色。
名称提及或继承的文档上下文不能直接升级为数值归属已核验。
未登记或存在歧义的名称保留为未消解状态。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any
import unicodedata

from domain.evidence import EvidenceTableRow
from service.core.resources.paths import config_file


ANNOTATION_VERSION = "entity-attribution-v2"
ENTITY_ATTRIBUTION_FIELDS = (
    "entity_annotation_version_kwd", "entity_registry_version_kwd",
    "document_subject_entity_ids", "document_subject_entity_names_kwd",
    "document_subject_status_kwd", "publisher_entity_ids",
    "publisher_entity_names_kwd", "publisher_status_kwd",
    "mentioned_entity_ids", "mentioned_entity_names_kwd",
    "subject_entity_ids", "candidate_subject_entity_ids",
    "unresolved_entity_names_kwd", "entity_attribution_status_kwd",
    "entity_attribution_obj",
)
_SUBJECT_LABEL = re.compile(
    r"(?m)^\s*(?:公司名称|证券简称|股票简称|研究对象|研究标的|报告主体)\s*[:：]\s*([^\n；;|]{1,80})"
)
_PUBLISHER_LABEL = re.compile(
    r"(?m)^\s*(?:资料发布方|发布方|发布机构|编制单位|出具机构|作者机构)\s*[:：]\s*([^\n；;|]{1,80})"
)
OPENING_COMPANY_CODE = re.compile(
    r"(?m)^\s*(?:证券研究报告[\s|｜:：]*)?([^\n（）()|｜]{1,40}?)\s*[（(]\s*(\d{6}\.(?:SH|SZ))\s*[）)]", re.I,
)
_COMPANY_HEADERS = {
    "公司", "公司名称", "公司简称", "公司全称", "企业", "企业名称", "企业简称",
    "证券简称", "股票简称", "标的公司", "company", "companyname",
}
_CODE_HEADERS = {"代码", "股票代码", "证券代码", "stockcode", "ticker"}
_TOPIC_TITLE_SUFFIX = re.compile(
    r"^(?:年度|半年度|年报|中报|季报|研报|报告|研究|点评|业绩|财务|盈利|估值|公告|"
    r"收入|营业收入|营收|净利润|利润|现金|分红|公司点评|与|和|对比|比较)"
)


def _ascii_word(character: str) -> bool:
    return character.isascii() and character.isalnum()


def _normalized_with_offsets(text: str) -> tuple[str, list[int]]:
    characters, offsets = [], []
    for index, character in enumerate(text):
        for normalized in unicodedata.normalize("NFKC", character).casefold():
            if not normalized.isspace():
                characters.append(normalized)
                offsets.append(index)
    return "".join(characters), offsets


def _normalized(text: str) -> str:
    return _normalized_with_offsets(text)[0]


@dataclass(frozen=True)
class Entity:
    entity_id: str
    canonical_name: str
    kind: str

# 将实体表整理成容易查找的形式
class EntityRegistry:
    """仅使用已审核别名；名称碰撞时保留歧义。"""

    def __init__(self, payload: dict[str, Any]):
        self.entities: dict[str, Entity] = {}
        self.aliases: dict[str, set[str]] = {}
        records = payload.get("entities")
        if not isinstance(records, list) or not str(payload.get("version") or "").strip():
            raise ValueError("Entity registry requires version and entities")
        for record in records:
            entity_id = record["entity_id"]
            name = record["canonical_name"]
            kind = record["kind"]
            if not isinstance(entity_id, str) or not entity_id.strip() or entity_id in self.entities:
                raise ValueError("Entity registry requires unique, nonempty IDs")
            if not isinstance(name, str) or not name.strip() or kind not in {"company", "publisher"}:
                raise ValueError("Invalid entity name or kind")
            self.entities[entity_id] = Entity(entity_id, name, kind)
            aliases = record.get("aliases", [])
            codes = record.get("stock_codes", [])
            if not isinstance(aliases, list) or not isinstance(codes, list):
                raise ValueError("Entity aliases and stock_codes must be lists")
            for alias in [name, *aliases, *codes]:
                if not isinstance(alias, str) or not _normalized(alias):
                    raise ValueError("Entity aliases must be nonempty strings")
                self.aliases.setdefault(_normalized(alias), set()).add(entity_id)
        # 优先匹配较长名称，避免母公司的短别名覆盖已登记的子公司或全称。
        # 普通数字不作为证券代码别名。
        for alias in self.aliases:
            if alias.isdigit():
                raise ValueError("Stock codes require an exchange suffix; bare numbers are ambiguous")
        self.pattern = re.compile("|".join(
            re.escape(alias) for alias in sorted(self.aliases, key=lambda value: (-len(value), value))
        )) if self.aliases else None
        digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:12]
        self.version = f"{payload['version']}:{digest}"

    def exact(self, text: str, kind: str | None = "company") -> list[str]:
        return sorted(entity_id for entity_id in self.aliases.get(_normalized(text), ())
                      if kind is None or self.entities[entity_id].kind == kind)

    def mentions(self, text: str) -> list[dict[str, Any]]:
        if not self.pattern:
            return []
        normalized, offsets = _normalized_with_offsets(text)
        result = []
        for match in self.pattern.finditer(normalized):
            alias = match.group()
            start, end = offsets[match.start()], offsets[match.end() - 1] + 1
            # 根据原文检查边界：规范化会去除 OCR 多余空格，
            # 但普通英文单词之间的空格仍应视为名称边界。
            if _ascii_word(alias[0]) and start and _ascii_word(text[start - 1]):
                continue
            if _ascii_word(alias[-1]) and end < len(text) and _ascii_word(text[end]):
                continue
            # 未登记的较长法人名称不能继承较短母公司名称的编号，
            # 例如“腾讯金融科技有限公司”不能自动绑定为“腾讯”。
            if re.match(r"[\u4e00-\u9fff]{0,20}?(?:股份有限公司|有限责任公司|有限公司)", normalized[match.end():]):
                continue
            candidates = sorted(self.aliases[alias])
            result.append({
                "quote": text[start:end], "start": start, "end": end,
                "entity_ids": candidates,
                "status": "resolved" if len(candidates) == 1 else "ambiguous",
            })
        return result

    def resolved_ids(self, mentions: list[dict[str, Any]], kind: str | None = None) -> list[str]:
        return sorted({mention["entity_ids"][0] for mention in mentions
                       if mention["status"] == "resolved"
                       and (kind is None or self.entities[mention["entity_ids"][0]].kind == kind)})

    def names(self, entity_ids: list[str]) -> list[str]:
        return [self.entities[entity_id].canonical_name for entity_id in entity_ids]


@lru_cache(maxsize=8)
def _load_registry(path: str, modified_ns: int, size: int) -> EntityRegistry:
    return EntityRegistry(json.loads(Path(path).read_text(encoding="utf-8")))


def get_entity_registry() -> EntityRegistry:
    path = Path(os.getenv("RAG_ENTITY_REGISTRY_PATH") or config_file("entity_registry.json")).resolve()
    stat = path.stat()
    return _load_registry(str(path), stat.st_mtime_ns, stat.st_size)


def _item_text(item: dict[str, Any]) -> str:
    return str(item.get("table_semantic_text_with_weight") or item.get("content_with_weight") or "")

# 先标注整份资料在研究哪个实体
def annotate_document(items: list[dict[str, Any]], file_name: str, registry: EntityRegistry) -> dict[str, Any]:
    """根据文件名或明确标签标注报告主体，不按正文提及频率推断。"""
    filename = file_name.replace("\\", "/").rsplit("/", 1)[-1]

    # stem是去掉扩展名后的文件名
    stem = Path(filename).stem
    title_mentions = registry.mentions(stem)
    for mention in title_mentions:
        suffix = _normalized(stem[mention["end"]:])
        is_company = any(registry.entities[value].kind == "company" for value in mention["entity_ids"])
        # 未登记的较长公司或分支机构名称中包含短名，不能证明报告主体已确定；
        # 常见报告标题后缀仍可正常匹配。
        if is_company and suffix and "\u4e00" <= suffix[0] <= "\u9fff" and not _TOPIC_TITLE_SUFFIX.match(suffix):
            mention["status"] = "unknown"
    subjects = registry.resolved_ids(title_mentions, "company")
    subject_quotes = [{"source": "filename", **mention} for mention in title_mentions
                      if any(registry.entities[value].kind == "company" for value in mention["entity_ids"])]
    ambiguous_subject = any(quote["status"] in {"ambiguous", "unknown"} for quote in subject_quotes)
    publishers = registry.resolved_ids(title_mentions, "publisher")
    publisher_quotes = [{"source": "filename", **mention} for mention in title_mentions
                        if any(registry.entities[value].kind == "publisher" for value in mention["entity_ids"])]
    unresolved = []
    text_item_count = 0
    for item_index, item in enumerate(items):
        text = _item_text(item)
        is_table = item.get("evidence_type_kwd") == "table" or "table_cells_slice_obj" in item
        if not is_table:
            text_item_count += 1
            if text_item_count <= 3:
                for match in OPENING_COMPANY_CODE.finditer(text):
                    ids = registry.exact(match[1].strip())
                    code_ids = registry.exact(match[2])
                    if len(ids) == 1 and ids == code_ids:
                        subjects.extend(ids)
                        subject_quotes.append({"source": "opening_company_and_code", "item_index": item_index,
                                               "quote": match.group(), "entity_ids": ids, "status": "resolved",
                                               "position_int": item.get("position_int") or []})
        for pattern, kind, target, quotes in (
            (_SUBJECT_LABEL, "company", subjects, subject_quotes),
            (_PUBLISHER_LABEL, "publisher", publishers, publisher_quotes),
        ):
            # 可比公司表中的公司列描述各行主体，不能作为整份文档的主体。
            # 报告主体标签只从文档开头提取。
            if is_table or (kind == "company" and text_item_count > 3):
                continue
            for match in pattern.finditer(text):
                label = match.group(1).strip()
                ids = registry.exact(label, None if kind == "publisher" else kind)
                quotes.append({"source": "explicit_label", "item_index": item_index,
                               "quote": match.group(), "entity_ids": ids,
                               "status": "resolved" if len(ids) == 1 else "ambiguous" if ids else "unknown"})
                if len(ids) == 1:
                    target.extend(ids)
                elif not ids:
                    unresolved.append(label)
                if kind == "company" and len(ids) != 1:
                    ambiguous_subject = True
    subjects = sorted(set(subjects))
    publishers = sorted(set(publishers))
    # 研究署名只能识别可能的发布方，不能据此确认作者机构。
    credits = []
    if not publishers:
        for item_index, item in enumerate(items):
            text = _item_text(item)
            for mention in registry.mentions(text):
                if mention["status"] != "resolved":
                    continue
                entity_id = mention["entity_ids"][0]
                suffix = _normalized(text[mention["end"]:mention["end"] + 12])
                if registry.entities[entity_id].kind == "publisher" and suffix.startswith(("研究所", "经济研究所", "研究院")):
                    credits.append({"source": "research_credit", "item_index": item_index, **mention})
        publishers = registry.resolved_ids(credits, "publisher")
    subject_status = "ambiguous" if ambiguous_subject else "multiple" if len(subjects) > 1 else "resolved" if subjects else "unknown"
    if ambiguous_subject:
        subjects = []
    publisher_status = "candidate" if credits and publishers else "resolved" if publishers else "unknown"
    if len(publishers) > 1 or any(quote["status"] != "resolved" for quote in publisher_quotes):
        publisher_status, publishers = "ambiguous", []
    return {
        "entity_annotation_version_kwd": ANNOTATION_VERSION,
        "entity_registry_version_kwd": registry.version,
        "document_subject_entity_ids": subjects,
        "document_subject_entity_names_kwd": registry.names(subjects),
        "document_subject_status_kwd": subject_status,
        "publisher_entity_ids": publishers,
        "publisher_entity_names_kwd": registry.names(publishers),
        "publisher_status_kwd": publisher_status,
        "unresolved_entity_names_kwd": sorted(set(unresolved)),
        "entity_attribution_obj": {
            "document_name": filename,
            "document_subject_evidence": subject_quotes,
            "publisher_evidence": publisher_quotes or credits,
        },
    }


def _table_bindings(item: dict[str, Any], registry: EntityRegistry) -> tuple[list[str], str, list[dict[str, Any]], list[str]]:
    """入库记录在此适配，在线核验直接使用规范表格行。"""
    from service.core.evidence.rows import table_row_from_record

    cells_slice = item.get("table_cells_slice_obj") or {}
    binding = "unbound" if item.get("table_binding_kwd") == "unbound" else cells_slice.get("binding_status", "bound")
    rows = tuple(table_row_from_record(row) for row in cells_slice.get("rows") or ())
    return resolve_table_bindings(rows, registry, binding=binding)


def resolve_table_bindings(rows: Sequence[EvidenceTableRow], registry: EntityRegistry, *,
                           binding="bound") -> tuple[list[str], str, list[dict[str, Any]], list[str]]:
    """入库和在线核验共用的原表行归属规则。"""
    if binding == "unbound":
        return [], "unbound_table", [], []
    bindings, unresolved, subjects = [], [], set()
    has_company_column = False
    uncertain = False
    for row in rows:
        headers = [str(value) for value in row.headers]
        cells = [str(value) for value in row.cells]
        if len(headers) != len(cells):
            uncertain = True
            continue
        # 先找出哪些列是公司名列，然后读取改列的单元格，查公司编号
        company_columns = [index for index, header in enumerate(headers)
                           if _normalized(header.split("/")[-1]) in _COMPANY_HEADERS]
        if company_columns:
            has_company_column = True
            row_entities = set()
            for index in company_columns:
                ids = registry.exact(cells[index])
                if len(ids) != 1:
                    uncertain = True
                    unresolved.append(cells[index])
                else:
                    row_entities.update(ids)
            if len(row_entities) != 1:
                uncertain = True
                continue
            entity_id = next(iter(row_entities))
            code_columns = [index for index, header in enumerate(headers)
                            if _normalized(header.split("/")[-1]) in _CODE_HEADERS]
            code_entities = {value for index in code_columns for value in registry.exact(cells[index])}
            # 如果股票代码查出来属于另一家公司，这一行就不能确认归属
            if code_entities - {entity_id}:
                uncertain = True
                continue
            # 携带实体信息的表头存在冲突时，不能静默覆盖。
            header_entities = {value for header in headers for value in registry.resolved_ids(registry.mentions(header), "company")}
            if header_entities - {entity_id}:
                uncertain = True
                continue
            subjects.add(entity_id)
            binding = {"source": "table_company_column", "row_index": row.index,
                       "entity_id": entity_id, "company_column_indices": company_columns,
                       "company_quotes": [cells[index] for index in company_columns],
                       "value_column_indices": [index for index in range(len(headers))
                                                if index not in company_columns + code_columns]}
            if code_columns:
                binding.update({"stock_code_column_indices": code_columns,
                                "stock_code_quotes": [cells[index] for index in code_columns]})
            bindings.append(binding)
        else:
            for index, header in enumerate(headers):
                mentions = registry.mentions(header)
                ids = registry.resolved_ids(mentions, "company")
                if any(mention["status"] == "ambiguous" for mention in mentions):
                    uncertain = True
                if len(ids) == 1:
                    subjects.update(ids)
                    bindings.append({"source": "table_company_header", "row_index": row.index,
                                     "entity_id": ids[0], "value_column_indices": [index], "header_quote": header})
                elif len(ids) > 1:
                    uncertain = True
    if uncertain:
        return [], "ambiguous", [], unresolved
    if bindings:
        return sorted(subjects), "explicit_table", bindings, unresolved
    return [], "unknown" if has_company_column else "document_context", [], unresolved


def annotate_chunk(item: dict[str, Any], document: dict[str, Any], registry: EntityRegistry) -> dict[str, Any]:
    text = _item_text(item)
    mentions = registry.mentions(text)
    mentioned_ids = registry.resolved_ids(mentions)
    company_mentions = registry.resolved_ids(mentions, "company")
    subjects, bindings, unresolved = [], [], []
    is_table = item.get("evidence_type_kwd") == "table" or "table_cells_slice_obj" in item
    #如果是表格chunk，则需要明确表格的哪一行属于哪一个实体
    if is_table:
        subjects, status, bindings, unresolved = _table_bindings(item, registry)
    else:
        # 普通文字即使出现公司名，也只是记为mentions，不能直接认定为研究对象
        status = "mentions_only" if company_mentions else "document_context"
    if any(mention["status"] == "ambiguous" for mention in mentions) and not subjects:
        status = "ambiguous"
    # 上下文可供后续核验使用，但本身不能证明数值归属。
    candidates = sorted(set(subjects or company_mentions or document["document_subject_entity_ids"]))
    if status in {"unknown", "ambiguous", "unbound_table"}:
        candidates = []
    if status == "document_context" and not candidates:
        status = "unknown"
    return {
        **document,
        "mentioned_entity_ids": mentioned_ids,  # 提到的所有实体ID
        "mentioned_entity_names_kwd": registry.names(mentioned_ids), #上面这些实体ID对应的名称列表
        "subject_entity_ids": subjects,  #表示数据归属哪个实体
        "candidate_subject_entity_ids": candidates,  # 表示数据可能归属的实体
        # 被识别成实体但在实体库中没匹配上的实体名字
        "unresolved_entity_names_kwd": sorted(set(document["unresolved_entity_names_kwd"] + unresolved)), 
        "entity_attribution_status_kwd": status, #归属状态
        "entity_attribution_obj": {
            **document["entity_attribution_obj"], "mentions": mentions,
            "mention_text_field": "table_semantic_text_with_weight" if item.get("table_semantic_text_with_weight") else "content_with_weight",
            "table_bindings": bindings,
            "fact_verification_status": "not_verified",
        },
    }

# 明确表格的哪一行属于哪一个实体
def annotate_table_object(table: dict[str, Any], document: dict[str, Any], registry: EntityRegistry) -> dict[str, Any]:
    """在完整、可审计的父表中保存归属关系及原始坐标。"""
    cells_json = table.get("cells_json") or {}
    rows = []

    # 从表格中拿到所有数据行
    for row in cells_json.get("data_rows") or []:
        # 判断表格这一行的数据归属于哪个主体实体
        annotation = annotate_chunk({
            "evidence_type_kwd": "table",
            "table_cells_slice_obj": {"binding_status": cells_json.get("binding_status"), "rows": [row]},
        }, document, registry)
        rows.append({"row_index": row.get("row_index"), "subject_entity_ids": annotation["subject_entity_ids"],
                     "status": annotation["entity_attribution_status_kwd"],
                     "bindings": annotation["entity_attribution_obj"]["table_bindings"]})

    return {"version": ANNOTATION_VERSION, "document": document, "rows": rows}
