"""用户问题侧的实体名称注册表；不推断文档或正文的数据归属。"""

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any
import unicodedata

from service.core.resources.paths import config_file


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

