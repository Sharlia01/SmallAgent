"""需求输出的解码与保真校验；模型调用统一由 AgentPlanner 管理。"""
import json
from dataclasses import replace
from service.core.query.contract import QueryUnderstanding, decode_goals, validate_goal_sources
from service.core.query.validation import QueryValidationError, validate_goals
from agent.financial_requirements import decode_financial_requirements, validate_financial_requirements

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
    if "financial" not in payload:
        raise QueryValidationError("missing_financial_contract", "必须显式声明 financial；无财务需求时为 null")
    financial = decode_financial_requirements(payload["financial"])
    return QueryUnderstanding(goals, payload["requires_evidence"], clarification, financial=financial)

def validate_understanding(value: QueryUnderstanding, question: str, *, reference_date: str = "") -> QueryUnderstanding:
    goals = validate_goal_sources(value.goals, question)
    validate_goals(question, goals)
    financial = value.financial
    if not value.clarification:
        financial = validate_financial_requirements(financial, goals, question, reference_date=reference_date)
    return replace(value, goals=goals, financial=financial)
