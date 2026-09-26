"""Deterministic arithmetic over bound table cells, with input provenance.

No generated expressions are evaluated. Unsupported calculations remain the
verifier's responsibility; absence from this registry is not a refusal reason.
"""

from __future__ import annotations

import re
import unicodedata
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


def _facts(evidence: list[dict]) -> list[dict]:
    facts = []
    for item in evidence:
        if item.get("table_binding_kwd") != "bound":
            continue
        context = item.get("table_context_obj") or {}
        rows = context.get("rows", []) if isinstance(context, dict) and context.get("reliable") else []
        rows = [*rows, {"headers": item.get("table_headers_kwd"), "cells": item.get("table_cells_kwd"),
                       "row_index": item.get("table_row_int")}]
        for row in rows:
            if not isinstance(row, dict):
                continue
            headers, cells = row.get("headers"), row.get("cells")
            if not isinstance(headers, list) or not isinstance(cells, list) or len(headers) != len(cells) or not cells:
                continue
            label = str(cells[0])
            metrics = metric_terms(label)
            if len(metrics) != 1:
                continue
            unit_context = label + " " + str(row.get("context", "")) + " " + str(row.get("preamble", ""))
            unit_match = re.search(r"百万元|亿元|万元|美元|人民币元|元|万千瓦|亿千瓦时|千瓦时|兆瓦时|倍", unit_context)
            for column, (header, cell) in enumerate(zip(headers[1:], cells[1:]), 1):
                periods = reporting_periods(str(header))
                value = _decimal(str(cell))
                if len(periods) != 1 or value is None:
                    continue
                unit = "%" if str(cell).strip().endswith("%") else unit_match.group() if unit_match else None
                facts.append({"metric": metrics[0], "period": next(iter(periods)), "value": str(value),
                              "unit": unit, "scope": item.get("document_id") or item.get("document_name"),
                              "document_name": item.get("document_name", ""), "table_id": item.get("table_id_kwd"),
                              "chunk_index": item["index"], "chunk_id": item["chunk_id"],
                              "row": row.get("row_index"), "column": column,
                              "source_label": label, "source_header": str(header), "source_value": str(cell)})
    return facts


def _unique_fact(facts, metric, period):
    values = [fact for fact in facts if fact["metric"] == metric and fact["period"] == period]
    identities = {(fact["value"], fact["unit"], fact["scope"]) for fact in values}
    if not values:
        return None, "missing_input"
    if len(identities) != 1:
        return None, "conflicting_inputs"
    return values[0], None


def build_derived_evidence(question: str, evidence: list[dict]) -> list[dict]:
    """Return checked derived facts; never declare the whole question answered."""
    periods = reporting_periods(question)
    requirements = []
    if re.search(r"派息率|股利支付率|分红支付率", question) and len(periods) == 1:
        period = next(iter(periods))
        requirements.append(("payout_ratio", "每股派息率", "divide_percent",
                             [("每股红利", period), ("每股收益", period)]))
    # Explicit arithmetic over the same metric in two named years. Do not
    # resolve relative years or infer operands for open-ended computations.
    metrics = [m for m in metric_terms(question) if m not in {"预测", "实际", "同比", "环比", "分别", "截至"}]
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
    if not requirements:
        return []
    facts = _facts(evidence)
    # A document is the scope boundary. Explicit multi-entity requests need a
    # richer binding contract and must not be silently computed across reports.
    from service.core.query_expansion import _extract_organizations
    subjects = _extract_organizations(question)
    if len(subjects) == 1:
        subject = constraint_text(next(iter(subjects)))
        facts = [fact for fact in facts if subject in constraint_text(fact["document_name"])]
    output = []
    for identifier, label, operation, operands in requirements:
        record = {"id": identifier, "requirement": label, "operation": operation, "status": "unresolved", "inputs": []}
        output.append(record)
        if len(subjects) > 1:
            record["reason"] = "multiple_entity_scopes"
            continue
        scoped_facts = facts
        if operation == "divide_percent":
            tables = [{fact["table_id"] for fact in facts if fact["table_id"] and
                       fact["metric"] == metric and fact["period"] == period} for metric, period in operands]
            common_tables = set.intersection(*tables)
            if common_tables:
                scoped_facts = [fact for fact in facts if fact["table_id"] in common_tables]
        errors = []
        for metric, period in operands:
            fact, error = _unique_fact(scoped_facts, metric, period)
            if error:
                errors.append({"metric": metric, "period": period, "reason": error})
            else:
                record["inputs"].append(fact)
        if errors:
            record.update(reason="inputs_not_grounded", missing_inputs=errors)
            continue
        left, right = record["inputs"]
        same_scope = bool(left["scope"]) and left["scope"] == right["scope"]
        same_unit = left["unit"] is not None and left["unit"] == right["unit"]
        # Same row across periods has a stable (possibly implicit) table unit.
        same_row = (bool(left["table_id"]) and left["table_id"] == right["table_id"] and left["row"] == right["row"]
                    and left["unit"] is None and right["unit"] is None)
        per_share_basis = (operation == "divide_percent" and bool(left["table_id"])
                           and left["table_id"] == right["table_id"]
                           and left["unit"] is None and right["unit"] is None)
        if not same_scope or not (same_unit or same_row or per_share_basis):
            record["reason"] = "incompatible_scope_or_units"
            continue
        a, b = Decimal(left["value"]), Decimal(right["value"])
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
        unit = "%" if operation.endswith("percent") else "个百分点" if left["unit"] == "%" else left["unit"] or "原表单位"
        record.update(status="supported", result=format(result, "f"), unit=unit,
                      unit_basis="explicit" if same_unit else "same_row" if same_row else "same_table_per_share",
                      supporting_chunk_indices=list(dict.fromkeys(fact["chunk_index"] for fact in record["inputs"])))
    return output
