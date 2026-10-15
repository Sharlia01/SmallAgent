"""旧证据回归使用固定需求夹具；理解生成与校验由独立测试覆盖。

夹具是迁移前冻结并核对的需求输入，不在测试运行时重新解析用户问题，
也不从被测证据反推要求，避免把期望与被测实现绑在一起。
"""

import json
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

from domain.facts import FactRequirement, FinancialRequirements, CalculationRequirement, UnresolvedFinancialRequirement


@lru_cache(maxsize=1)
def _fixtures():
    rows = [json.loads(line) for line in (Path(__file__).parent / 'fixtures/financial_requirements.jsonl').read_text().splitlines()]
    return {(row['question'], tuple(row['goal_ids'])): row for row in rows}


def financial_spec(question, goals=()):
    row = _fixtures().get((question, tuple(goal.id for goal in goals))) or _fixtures().get((question, ()))
    if row is None:
        return None
    facts = tuple(FactRequirement(**{**item, 'source_spans': tuple(item.get('source_spans', ()))}) for item in row['facts'])
    calculations = tuple(CalculationRequirement(**{**item, 'operand_ids': tuple(item['operand_ids']), 'source_spans': tuple(item['source_spans'])}) for item in row['calculations'])
    unresolved = tuple(UnresolvedFinancialRequirement(**{**item, 'source_spans': tuple(item['source_spans'])}) for item in row['unresolved'])
    if len(goals) == 1 and not row['goal_ids']:
        facts = tuple(replace(item, goal_id=goals[0].id) for item in facts)
        calculations = tuple(replace(item, goal_id=goals[0].id) for item in calculations)
        unresolved = tuple(replace(item, goal_id=goals[0].id) for item in unresolved)
    return FinancialRequirements(facts, calculations, unresolved) if facts or calculations or unresolved else None


def assess_for_question(question, evidence, goals=(), **kwargs):
    from service.core.finance.facts import assess_financial_facts
    return assess_financial_facts(financial_spec(question, goals), evidence, **kwargs)
