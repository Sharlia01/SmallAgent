"""查询侧公司线索：用于相关性排序和诊断，不限制哪些文档片段可以召回。"""

from __future__ import annotations

from dataclasses import dataclass, replace
import re

from service.core.entities.registry import EntityRegistry, get_entity_registry


_LEGAL_NAME = re.compile(
    r"[\u4e00-\u9fffA-Za-z·]{2,40}?(?:股份有限公司|有限责任公司|有限公司|分公司)"
)


@dataclass(frozen=True)
class CompanyHints:
    # 名称用于排序；ID 和状态仅记录名称规范化结果，不用于过滤来源。
    entity_ids: tuple[str, ...] = ()
    entity_names: tuple[str, ...] = ()
    status: str = "no_registered_company"
    source: str = "original_question"
    registry_version: str = ""
    unresolved_quotes: tuple[str, ...] = ()

    @property
    def resolved(self) -> bool:
        return bool(self.entity_ids) and self.status == "resolved"

    def to_dict(self) -> dict:
        return {
            "usage": "ranking_hint",
            "entity_ids": list(self.entity_ids),
            "entity_names": list(self.entity_names),
            "status": self.status,
            "source": self.source,
            "registry_version": self.registry_version,
            "unresolved_quotes": list(self.unresolved_quotes),
        }


def resolve_company_hints(
    question: str,
    registry: EntityRegistry | None = None,
    *,
    source: str = "original_question",
) -> CompanyHints:
    """只从原问题/当前子问题取公司，不从改写或扩展词中取。"""
    registry = registry or get_entity_registry()
    mentions = [mention for mention in registry.mentions(question)
                if any(registry.entities[value].kind == "company"
                       for value in mention["entity_ids"])]
    ambiguous = [mention["quote"] for mention in mentions
                 if mention["status"] != "resolved"]
    # 未登记的完整公司/分公司名称，不能套用其中较短的母公司简称。
    # 已登记全名之前的“请查询”等文字不影响匹配。
    unknown = [match.group() for match in _LEGAL_NAME.finditer(question)
               if not any(mention["status"] == "resolved"
                          and mention["start"] >= match.start()
                          and mention["end"] == match.end()
                          for mention in mentions)]
    if ambiguous or unknown:
        return CompanyHints(
            status="ambiguous_company" if ambiguous else "unregistered_company",
            source=source,
            registry_version=registry.version,
            unresolved_quotes=tuple(dict.fromkeys([*ambiguous, *unknown]))[:20],
        )
    entity_ids = registry.resolved_ids(mentions, "company")
    return CompanyHints(
        entity_ids=tuple(entity_ids),
        entity_names=tuple(registry.names(entity_ids)),
        status="resolved" if entity_ids else "no_registered_company",
        source=source,
        registry_version=registry.version,
    )


def subquery_company_hints(
    question: str, parent: CompanyHints,
) -> CompanyHints:
    """子查询有自己的公司就用自己的；省略公司时继承原问题的公司集合。"""
    local = resolve_company_hints(question, source="subquery")
    if local.status == "no_registered_company" and parent.resolved:
        source = "inherited_single_company" if len(parent.entity_ids) == 1 else "inherited_company_set"
        return replace(parent, source=source)
    return local


def company_hints_from_names(names: tuple[str, ...], *, registry=None) -> CompanyHints:
    """仅规范化已由模型或证据确认的完整公司名，不从搜索语句识别公司。"""
    registry = registry or get_entity_registry()
    names = tuple(dict.fromkeys(names))
    ids, unknown, ambiguous = [], [], []
    for name in names:
        matches = registry.exact(name, kind="company")
        if len(matches) == 1:
            ids.extend(matches)
        elif matches:
            ambiguous.append(name)
        else:
            unknown.append(name)
    # 混合未知主体时保留全部名称；部分身份未匹配不影响这些名称参与排序。
    if ambiguous:
        status = "ambiguous_company"
    elif unknown:
        status = "unregistered_company"
    elif ids:
        status = "resolved"
    else:
        status = "no_registered_company"
    return CompanyHints(
        entity_ids=tuple(dict.fromkeys(ids)),
        entity_names=names,
        status=status,
        source="model_understanding",
        registry_version=registry.version,
        unresolved_quotes=tuple(ambiguous + unknown),
    )
