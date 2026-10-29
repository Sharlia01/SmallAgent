"""模型识别的实体及其来源。这里只验证引用、定位原文并映射已登记身份。"""
from dataclasses import dataclass, replace
import logging
import re

from service.core.entities.registry import get_entity_registry
from service.core.query.validation import QueryValidationError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QueryEntity:
    ref: str  #给queryunderstanding中的subject_refs用的
    mention: str  #实体名字
    source_quote: str #在用户查询中的原片段
    occurrence: int = 0
    start: int = -1  # 无法唯一定位时 start/end 均为 -1；status 仅表示身份映射状态。
    end: int = -1
    entity_id: str | None = None  #用于关联系统登记的实体身份
    status: str = "unregistered"


def decode_entities(raw) -> tuple[QueryEntity, ...]:
    if not isinstance(raw, list) or len(raw) > 16:
        raise QueryValidationError("invalid_entities", "必须显式提供 entities 列表，无已知公司时使用 []")
    result = []
    for row in raw:
        if not isinstance(row, dict) or set(row) - {"ref", "mention", "source_quote", "occurrence"}:
            raise QueryValidationError("invalid_entity_schema", "实体只能提供名称、原文和引用，不得提供数据库ID")
        ref, mention, quote = row.get("ref"), row.get("mention"), row.get("source_quote")
        occurrence = row.get("occurrence", 0)
        if (not isinstance(ref, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,31}", ref)
                or not isinstance(mention, str) or not mention.strip() or len(mention) > 120
                or not isinstance(quote, str) or not quote.strip() or len(quote) > 2000
                or type(occurrence) is not int or occurrence < 0):
            raise QueryValidationError("invalid_entity_schema", "实体名称、原文或出现序号无效")
        result.append(QueryEntity(ref, mention, quote, occurrence))
    if len({item.ref for item in result}) != len(result):
        raise QueryValidationError("duplicate_entity_ref", "实体引用编号必须唯一")
    return tuple(result)


def ground_entities(entities, question: str, *, registry=None):
    '''公司名称必须逐字来自输入问题'''
    registry = registry or get_entity_registry()
    result = []
    for item in entities:
        # occurrence 是 source_quote 在原问题中的第几次出现；偏移由服务端计算。
        starts = [m.start() for m in re.finditer(re.escape(item.source_quote), question)]
        if item.mention not in item.source_quote or item.occurrence >= len(starts):
            raise QueryValidationError("entity_not_grounded", "公司名称和来源片段必须逐字来自输入问题", offending=item.mention)
        start = end = -1
        if item.source_quote.count(item.mention) == 1:
            start = starts[item.occurrence] + item.source_quote.index(item.mention)
            end = start + len(item.mention)
        else:
            logger.info("ambiguous_entity_span: ref=%s; entity offsets left unresolved", item.ref)
        ids = registry.exact(item.mention, kind=None)
        result.append(replace(item, start=start, end=end,
                              entity_id=ids[0] if len(ids) == 1 else None,
                              status="resolved" if len(ids) == 1 else "ambiguous" if ids else "unregistered"))
    return tuple(result)


def validate_entity_goals(entities, goals, *, clarification=False):
    """只拒绝不存在的实体引用；文本提及但未引用的实体保留诊断。"""
    from service.core.query.preservation import _extract_organizations
    by_ref = {item.ref: item for item in entities}
    names = tuple(e.mention for e in entities)
    # 目标中的实体引用必须全部存在于queryunderstanding的entities中
    for goal in goals:
        refs = (*goal.subject_refs, *goal.publisher_refs)
        if not set(refs) <= by_ref.keys():
            raise QueryValidationError("invalid_goal_entity_binding", "目标的主体和发布方必须引用已声明实体", goal_id=goal.id)
        text = goal.text + "；" + "；".join(goal.constraints)
        present = _extract_organizations(text, names)
        for entity in entities:
            # 如果目标文本中有实体但是subject_ref中没有这个实体，则记录一条告警日志
            if entity.mention in present and entity.ref not in refs and not any(
                by_ref[ref].mention == entity.mention for ref in refs
            ):
                logger.info("undeclared_goal_entity: goal_id=%s ref=%s", goal.id, entity.ref)


def validate_known_entity_additions(text, entities):
    """登记名称仅用于检测新增已知身份，不用来重新决定自然语言的实体边界。"""
    registry = get_entity_registry()
    names = [item.mention for item in entities]
    # 先遮蔽完整模型名称，避免未登记子公司中的母公司别名被当作另一个主体。
    remaining = text
    for name in sorted(set(names), key=len, reverse=True):
        remaining = remaining.replace(name, " ")
    if registry.mentions(remaining):
        raise QueryValidationError("undeclared_company", "文本引入了未声明的已登记公司名称", actual=text)


def entity_identity(name: str) -> str:
    """比较已识别名称的身份；未登记名称保持独立，绝不做包含/模糊匹配。"""
    ids = get_entity_registry().exact(name, kind=None)
    return ids[0] if len(ids) == 1 else "name:" + name
