"""需求输出的解码、来源校验与规范化；模型调用统一由 AgentPlanner 管理。"""
import json
from dataclasses import replace
from service.core.query.contract import QueryUnderstanding, decode_goals, validate_goal_sources
from service.core.query.validation import QueryValidationError
from service.core.query.entities import decode_entities, ground_entities, validate_entity_goals

def decode_understanding(content: str) -> QueryUnderstanding:
    payload = json.loads(content)
    if not isinstance(payload, dict) or type(payload.get("requires_evidence")) is not bool:
        raise QueryValidationError("invalid_understanding", "requires_evidence must be boolean")

    # 如果模型返回需要用户澄清
    clarification = payload.get("clarification")
    if clarification is not None and (
        not isinstance(clarification, str) or not clarification.strip() or len(clarification) > 200
    ):
        raise QueryValidationError("invalid_clarification", "clarification must be a short question")
    raw_goals = payload.get("goals")
    goals = () if clarification and raw_goals == [] else decode_goals(raw_goals)
    return QueryUnderstanding(goals, payload["requires_evidence"], clarification,
                              entities=decode_entities(payload.get("entities")), entity_names=())

def validate_understanding(value: QueryUnderstanding, question: str, *, reference_date: str = "") -> QueryUnderstanding:
    # question不是原始用户问题而是经过了指代消解后的用户问题
    entities = ground_entities(value.entities, question)
    names = tuple(dict.fromkeys(item.mention for item in entities)) if value.entity_names is not None else None
    # 校验目标没有形成循环依赖以及目标id唯一
    goals = validate_goal_sources(value.goals, question)
    if names is not None:
        validate_entity_goals(entities, goals, clarification=bool(value.clarification))
    return replace(value, goals=goals, entities=entities, entity_names=names)
