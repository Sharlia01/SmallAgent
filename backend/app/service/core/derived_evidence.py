"""从已绑定的表格单元格进行确定性计算，并保留输入出处。

不执行模型生成的表达式。未支持的计算仍由核验器判断，
不因计算类型未登记就直接认定整个问题不可回答。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict
from collections.abc import Sequence
from domain.evidence import EvidenceCollection
from domain.facts import FinancialFact, CalculationInput
from decimal import Decimal, InvalidOperation, localcontext

from service.core.query_semantics import constraint_text, metric_terms, reporting_periods


def _decimal(value: str) -> Decimal | None:
    value = unicodedata.normalize("NFKC", str(value)).strip().replace(",", "").replace("−", "-")
    if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?%?", value):
        return None
    try:
        number = Decimal(value.rstrip("%"))
        return number if number.is_finite() else None
    except InvalidOperation:
        return None


def _facts(evidence: EvidenceCollection) -> list[CalculationInput]:
    facts = []
    for index, item in enumerate(evidence, 1):
        table = item.structure.table
        if table.binding != "bound" or not table.reliable:
            continue
        for row in table.rows:
            if not row.grounded:
                continue
            label = row.cells[0]
            metrics = metric_terms(label)
            if len(metrics) != 1:
                continue
            unit_context = " ".join((label, row.context, row.preamble, table.caption))
            unit_match = re.search(r"百万元|亿元|万元|美元|人民币元|元|万千瓦|亿千瓦时|千瓦时|兆瓦时|倍", unit_context)
            for column, (header, cell) in enumerate(zip(row.headers[1:], row.cells[1:]), 1):
                periods = reporting_periods(header)
                value = _decimal(cell)
                if len(periods) != 1 or value is None:
                    continue
                unit = "%" if cell.strip().endswith("%") else unit_match.group() if unit_match else None
                facts.append(CalculationInput(
                    metrics[0], next(iter(periods)), str(value), unit,
                    item.source.document_id or item.source.name, item.id, index,
                    table_id=item.location.table_id, row=row.index, column=row.columns[column],
                    source_label=label, source_header=header, source_value=cell,
                    document_name=item.source.name, document_id=item.source.document_id,
                ))
    return facts


def _unique_fact(facts, metric, period, scope=None):
    values = [fact for fact in facts if fact.metric == metric and fact.period == period
              and (scope is None or fact.scope == scope)]
    identities = {(fact.value, fact.unit, fact.scope) for fact in values}
    if not values:
        return None, "missing_input"
    if len(identities) != 1:
        return None, "conflicting_inputs"
    return values[0], None


def build_derived_evidence(question: str, evidence: EvidenceCollection, *, verified_facts: Sequence[FinancialFact] | None = None) -> list[dict]:
    """返回已核验的派生事实；计算成功不代表整个问题已得到回答。"""
    periods = reporting_periods(question)
    requirements = []
    if re.search(r"派息率|股利支付率|分红支付率", question) and len(periods) == 1:
        period = next(iter(periods))
        requirements.append(("payout_ratio", "每股派息率", "divide_percent",
                             [("每股红利", period), ("每股收益", period)]))
    # 仅对明确列出的两个年份、同一个指标执行计算，
    # 不在此消解相对年份，也不猜测开放式计算的操作数。
    from service.core.financial_facts import financial_metrics
    metrics = financial_metrics(question) or [m for m in metric_terms(question) if m not in {"预测", "实际", "同比", "环比", "分别", "截至"}]
    ordered = sorted(periods, key=lambda value: question.find(value[:4]))
    if len(periods) == 2 and len(set(metrics)) == 1 and re.search(r"相差|差额|增加多少|减少多少|下降多少|增长率|降幅|增幅", question):
        operation = "change_percent" if re.search(r"增长率|降幅|增幅", question) else "subtract"
        operands = [(metrics[0], ordered[1]), (metrics[0], ordered[0])]
        if re.search(r"减少|下降|降幅", question):
            operation = "decline_percent" if operation == "change_percent" else "subtract"
            operands.reverse()
        if "相差" in question:
            operation = "absolute_difference"
        requirements.append(("period_change", "跨期变化", operation, operands))
    if verified_facts is not None and len(periods) == 1 and len(metrics) == 1 and re.search(r"相差|差额|高多少|低多少", question):
        from service.core.entity_attribution import get_entity_registry
        registry = get_entity_registry()
        entity_order = list(dict.fromkeys(mention["entity_ids"][0] for mention in registry.mentions(question)
                                        if mention["status"] == "resolved" and registry.entities[mention["entity_ids"][0]].kind == "company"))
        if len(entity_order) == 2:
            period = next(iter(periods))
            operation = "absolute_difference" if "相差" in question or "差额" in question else "subtract"
            if "低多少" in question:
                entity_order.reverse()
            requirements.append(("entity_difference", "跨公司差值", operation,
                                 [(metrics[0], period, entity) for entity in entity_order]))
    if not requirements:
        return []
    if verified_facts is None:
        facts = _facts(evidence)
    else:
        chunk_indices = {item.id: index for index, item in enumerate(evidence, 1)}
        facts = []
        for fact in verified_facts:
            if (fact.attribution_status != "verified" or fact.verification_status != "verified"
                    or fact.source["chunk_id"] not in chunk_indices):
                continue
            source = fact.source
            facts.append(CalculationInput(
                fact.metric, fact.period["label"], fact.value, fact.unit, fact.entity_id,
                source["chunk_id"], chunk_indices[source["chunk_id"]],
                table_id=source.get("table_id"), row=source.get("row"), column=source.get("column"),
                source_label=source.get("row_label", fact.metric), source_header=source.get("header", fact.period["label"]),
                source_value=source.get("cell", fact.raw_value), document_id=source["document_id"],
                fact_id=fact.fact_id, value_kind=fact.value_kind, statement_scope=fact.statement_scope,
                business_scope=fact.business_scope, currency=fact.currency, value_qualifier=fact.value_qualifier,
            ))
    # 旧证据以文档为范围边界；明确的多公司请求需要逐个绑定操作数，
    # 不能仅按文件归属隐式合并不同报告中的数值。
    from service.core.query_expansion import _extract_organizations
    subjects = _extract_organizations(question)
    if len(subjects) == 1 and verified_facts is None:
        subject = constraint_text(next(iter(subjects)))
        facts = [fact for fact in facts if subject in constraint_text(fact.document_name)]
    output = []
    for identifier, label, operation, operands in requirements:
        record = {"id": identifier, "requirement": label, "operation": operation, "status": "unresolved", "inputs": []}
        output.append(record)
        if len(subjects) > 1 and identifier != "entity_difference":
            record["reason"] = "multiple_entity_scopes"
            continue
        scoped_facts = facts
        if operation == "divide_percent":
            tables = [{fact.table_id for fact in facts if fact.table_id and
                       fact.metric == operand[0] and fact.period == operand[1]} for operand in operands]
            common_tables = set.intersection(*tables)
            if common_tables:
                scoped_facts = [fact for fact in facts if fact.table_id in common_tables]
        errors = []
        for operand in operands:
            metric, period = operand[:2]
            fact, error = _unique_fact(scoped_facts, metric, period, operand[2] if len(operand) == 3 else None)
            if error:
                errors.append({"metric": metric, "period": period, "reason": error})
            else:
                record["inputs"].append(fact)
        if errors:
            record.update(reason="inputs_not_grounded", missing_inputs=errors)
            continue
        left, right = record["inputs"]
        if verified_facts is not None:
            shared_table = bool(left.table_id) and left.table_id == right.table_id
            incompatible = any(getattr(left, key) != getattr(right, key) for key in
                               ("value_kind", "statement_scope", "business_scope", "currency"))
            unknown_scope = any(getattr(left, key) == "unknown" or getattr(right, key) == "unknown"
                                for key in ("value_kind", "statement_scope", "currency"))
            if incompatible or (unknown_scope and not shared_table):
                record["reason"] = "incompatible_or_unknown_fact_scopes"
                continue
            if left.value_qualifier == "approximate" or right.value_qualifier == "approximate":
                record["reason"] = "approximate_inputs"
                continue
        same_scope = bool(left.scope) and (left.scope == right.scope or identifier == "entity_difference")
        same_unit = left.unit is not None and left.unit == right.unit
        # 同一行跨期间的数值共享稳定的表格单位，即使单位未在单元格中重复标明。
        same_row = (bool(left.table_id) and left.table_id == right.table_id and left.row == right.row
                    and left.unit is None and right.unit is None)
        per_share_basis = (operation == "divide_percent" and bool(left.table_id)
                           and left.table_id == right.table_id
                           and left.unit is None and right.unit is None)
        if not same_scope or not (same_unit or same_row or per_share_basis):
            record["reason"] = "incompatible_scope_or_units"
            continue
        a, b = Decimal(left.value), Decimal(right.value)
        with localcontext() as context:
            context.prec = 28
            if operation == "divide_percent":
                if b == 0:
                    record["reason"] = "zero_denominator"
                    continue
                result = a / b * 100
            elif operation in {"change_percent", "decline_percent"}:
                denominator = b if operation == "change_percent" else a
                if denominator <= 0:
                    record["reason"] = "nonpositive_change_base"
                    continue
                result = (a - b) / denominator * 100
            else:
                result = abs(a - b) if operation == "absolute_difference" else a - b
        unit = "%" if operation.endswith("percent") else "个百分点" if left.unit == "%" else left.unit or "原表单位"
        record.update(status="supported", result=format(result, "f"), unit=unit,
                      unit_basis="explicit" if same_unit else "same_row" if same_row else "same_table_per_share",
                      supporting_chunk_indices=list(dict.fromkeys(fact.chunk_index for fact in record["inputs"])),
                      supporting_chunk_ids=list(dict.fromkeys(fact.chunk_id for fact in record["inputs"])))
        if verified_facts is not None:
            record["input_fact_ids"] = [fact.fact_id for fact in record["inputs"]]
    for record in output:
        record["inputs"] = [asdict(item) for item in record["inputs"]]
    return output
