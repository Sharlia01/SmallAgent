"""关键词检索和向量检索共用的公司范围规则，不调用模型。"""

from __future__ import annotations

from dataclasses import dataclass, replace
import re

from service.core.entity_attribution import EntityRegistry, get_entity_registry


COMPANY_FILTER_KEY = "company_entity_ids"
CANDIDATE_STATUSES = ("document_context", "mentions_only")
_LEGAL_NAME = re.compile(
    r"[\u4e00-\u9fffA-Za-z·]{2,40}?(?:股份有限公司|有限责任公司|有限公司|分公司)"
)


@dataclass(frozen=True)
class CompanyConstraint:
    # 用固定编号筛选；名称只用于给人查看。多个编号表示“其中任意一家”。
    entity_ids: tuple[str, ...] = ()
    entity_names: tuple[str, ...] = ()
    status: str = "no_registered_company"
    source: str = "original_question"
    registry_version: str = ""
    unresolved_quotes: tuple[str, ...] = ()

    @property
    def enabled(self) -> bool:
        return bool(self.entity_ids) and self.status == "resolved"

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "entity_ids": list(self.entity_ids),
            "entity_names": list(self.entity_names),
            "status": self.status,
            "source": self.source,
            "registry_version": self.registry_version,
            "unresolved_quotes": list(self.unresolved_quotes),
            # 缺标签的资料不能当成“属于任意公司”，需先重新入库。
            "untagged_policy": "exclude_when_enabled",
            "candidate_ownership": "not_verified",
        }


def resolve_company_constraint(
    question: str,
    registry: EntityRegistry | None = None,
    *,
    source: str = "original_question",
) -> CompanyConstraint:
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
        return CompanyConstraint(
            status="ambiguous_company" if ambiguous else "unregistered_company",
            source=source,
            registry_version=registry.version,
            unresolved_quotes=tuple(dict.fromkeys([*ambiguous, *unknown]))[:20],
        )
    entity_ids = registry.resolved_ids(mentions, "company")
    return CompanyConstraint(
        entity_ids=tuple(entity_ids),
        entity_names=tuple(registry.names(entity_ids)),
        status="resolved" if entity_ids else "no_registered_company",
        source=source,
        registry_version=registry.version,
    )


def subquery_company_constraint(
    question: str, parent: CompanyConstraint,
) -> CompanyConstraint:
    """子查询有自己的公司就用自己的；省略公司时继承原问题的公司集合。"""
    local = resolve_company_constraint(question, source="subquery")
    if local.status == "no_registered_company" and parent.enabled:
        source = "inherited_single_company" if len(parent.entity_ids) == 1 else "inherited_company_set"
        return replace(parent, source=source)
    return local
