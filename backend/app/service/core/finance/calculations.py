"""执行已声明的财务计算；只使用逐要求核验通过的事实，不重新理解问题。"""

from dataclasses import asdict
from decimal import Decimal, InvalidOperation, localcontext

from domain.facts import CalculationInput, FinancialRequirements
from service.core.finance.rules import period_order_key, period_value


def calculation_status(record, operand_statuses):
    """保留输入问题的性质，避免口径歧义和除零都被解释成需要更多搜索。"""
    if record["status"] == "supported":
        return "supported"
    for status in ("conflict", "ambiguous", "unsupported"):
        if status in operand_statuses:
            return status
    reason = record.get("reason")
    if reason in {"zero_denominator", "nonpositive_change_base", "unsupported_operation", "invalid_operands"}:
        return "unsupported"
    if reason in {"incompatible_or_unknown_fact_scopes", "incompatible_scope_or_units",
                  "incompatible_calculation_periods", "approximate_inputs"}:
        return "ambiguous"
    return "missing"


def _calculation_input(fact, chunk_indices):
    source = fact.source
    return CalculationInput(
        fact.metric, (fact.period or {}).get("label", "未知期间"), fact.value, fact.unit, fact.entity_id,
        source["chunk_id"], chunk_indices[source["chunk_id"]],
        table_id=source.get("table_id"), row=source.get("row"), column=source.get("column"),
        source_label=source.get("row_label", fact.metric), source_header=source.get("header", ""),
        source_value=source.get("cell", fact.raw_value), document_id=source["document_id"],
        fact_id=fact.fact_id, value_kind=fact.value_kind, statement_scope=fact.statement_scope,
        business_scope=fact.business_scope, currency=fact.currency, value_qualifier=fact.value_qualifier,
        measure_kind=fact.measure_kind, comparison_basis=fact.comparison_basis,
    )


def _select_operand(requirement_id, supporting_facts, chunk_indices):
    candidates = [fact for fact in supporting_facts.get(requirement_id, ())
                  if fact.attribution_status == "verified" and fact.verification_status == "verified"
                  and fact.source["chunk_id"] in chunk_indices]
    identities = {(fact.entity_id, str(fact.period), fact.metric, fact.value, fact.unit,
                   fact.value_kind, fact.statement_scope, fact.business_scope, fact.currency,
                   fact.measure_kind, fact.comparison_basis, fact.value_qualifier) for fact in candidates}
    if not candidates:
        return None, "missing_input"
    if len(identities) != 1:
        return None, "conflicting_inputs"
    return _calculation_input(candidates[0], chunk_indices), None


def _scope_error(left, right, operation):
    # 期间可能在执行依赖任务后才确定，计算前仍需核验已绑定的期间关系。
    if operation == "divide_percent" and left.period != right.period:
        return "incompatible_calculation_periods"
    if operation in {"change_percent", "decline_percent"}:
        current, base = (left, right) if operation == "change_percent" else (right, left)
        current_period, base_period = period_value(current.period), period_value(base.period)
        if not current_period or not base_period or current_period["granularity"] != base_period["granularity"]:
            return "incompatible_calculation_periods"
        if period_order_key(current_period) <= period_order_key(base_period):
            return "calculation_direction_reversed"
    shared_table = bool(left.table_id) and left.table_id == right.table_id and left.document_id == right.document_id
    incompatible = any(getattr(left, key) != getattr(right, key) for key in
                       ("value_kind", "statement_scope", "business_scope", "currency", "measure_kind", "comparison_basis"))
    unknown = any(getattr(item, key) == "unknown" for item in (left, right)
                  for key in ("value_kind", "statement_scope", "currency"))
    if incompatible or (unknown and not shared_table):
        return "incompatible_or_unknown_fact_scopes"
    if left.value_qualifier == "approximate" or right.value_qualifier == "approximate":
        return "approximate_inputs"
    # 跨公司只允许明确声明的差值；比例与跨期变化必须属于同一主体。
    if not left.scope or not right.scope or (left.scope != right.scope and operation not in {"subtract", "absolute_difference"}):
        return "incompatible_scope_or_units"
    same_unit = left.unit is not None and left.unit == right.unit
    per_share = shared_table and operation == "divide_percent" and left.unit is None and right.unit is None
    if not same_unit and not per_share:
        return "incompatible_scope_or_units"
    return None


def _compute(operation, left, right):
    if operation not in {"subtract", "absolute_difference", "change_percent", "decline_percent", "divide_percent"}:
        return None, "unsupported_operation"
    try:
        a, b = Decimal(left.value), Decimal(right.value)
    except (InvalidOperation, TypeError):
        return None, "invalid_numeric_input"
    if not a.is_finite() or not b.is_finite():
        return None, "invalid_numeric_input"
    with localcontext() as context:
        context.prec = 28
        if operation == "divide_percent":
            return (None, "zero_denominator") if b == 0 else (a / b * 100, None)
        if operation in {"change_percent", "decline_percent"}:
            base = b if operation == "change_percent" else a
            return (None, "nonpositive_change_base") if base <= 0 else ((a - b) / base * 100, None)
        return (abs(a - b) if operation == "absolute_difference" else a - b), None


def build_derived_evidence(requirements: FinancialRequirements, evidence, *, supporting_facts: dict) -> list[dict]:
    """操作数由 requirement_id 关联，不能从其他已核验事实中重新挑选。"""
    chunk_indices = {item.id: index for index, item in enumerate(evidence, 1)}
    output = []
    for calculation in requirements.calculations:
        record = {"id": calculation.id, "goal_id": calculation.goal_id,
                  "requirement": "；".join(calculation.source_spans), "operation": calculation.operation,
                  "status": "unresolved", "inputs": []}
        output.append(record)
        if len(calculation.operand_ids) != 2 or len(set(calculation.operand_ids)) != 2:
            record["reason"] = "invalid_operands"
            continue
        inputs = []
        for identifier in calculation.operand_ids:
            operand, error = _select_operand(identifier, supporting_facts, chunk_indices)
            if error:
                record.setdefault("missing_inputs", []).append({"requirement_id": identifier, "reason": error})
            else:
                inputs.append(operand)
        record["inputs"] = [asdict(item) for item in inputs]
        if len(inputs) != 2:
            record["reason"] = "inputs_not_grounded"
            continue
        left, right = inputs
        error = _scope_error(left, right, calculation.operation)
        if error:
            record["reason"] = error
            continue
        result, error = _compute(calculation.operation, left, right)
        if error:
            record["reason"] = error
            continue
        percent = calculation.operation.endswith("percent")
        unit = "%" if percent else "个百分点" if left.unit == "%" else left.unit
        record.update(status="supported", result=format(result, "f"), unit=unit,
                      unit_basis="explicit" if left.unit is not None else "same_table_per_share",
                      supporting_chunk_indices=list(dict.fromkeys(item.chunk_index for item in inputs)),
                      supporting_chunk_ids=list(dict.fromkeys(item.chunk_id for item in inputs)),
                      input_fact_ids=[item.fact_id for item in inputs])
    return output
