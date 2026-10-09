"""财务事实级核验与确定性的回答上下文构建。

入库标注和模型 JSON 都只是候选，不代表核验结果。
仅放行由规则支持的原文局部关系；复杂正文保持未知，
不能仅因公司名称和数字共现就建立归属。
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from collections.abc import Sequence
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any

from domain.evidence import Evidence, EvidenceCollection, AnswerBlock, fingerprint, plain
from domain.assessment import EvidenceAssessment
from domain.facts import VERSION, FactRequirement, FinancialFact, FactEvidenceBundle

from service.core.entity_attribution import EntityRegistry, OPENING_COMPANY_CODE, resolve_table_bindings, get_entity_registry
from service.core.query_semantics import domain_policy, metric_terms, reporting_periods

if TYPE_CHECKING:
    from service.core.query_contract import QueryGoal


FINANCIAL_METRICS = {
    "营业收入", "营业成本", "净利润", "归母净利润", "扣非归母净利润",
    "基本每股收益", "稀释每股收益", "每股收益", "每股红利", "市盈率", "市净率",
    "经营活动现金流量净额", "企业自由现金流", "股权自由现金流",
    "毛利率", "净利率", "资产负债率", "财务费用率", "财务费用金额",
    "财务费用", "销售费用率", "研发费用率", "管理费用率",
    "销售费用", "研发费用", "管理费用", "股息率",
}
BALANCE_METRICS = (
    "资产总额", "总资产", "负债总额", "总负债", "所有者权益", "股东权益",
    "固定资产", "货币资金", "净资产", "营业利润", "利润总额",
)
NUMBER = r"[+\-−]?\d+(?:[,，]\d{3})*(?:\.\d+)?"
UNIT = r"人民币亿元|人民币万元|人民币元|亿元|百万元|万元|元(?:/股)?|美元|港元|%|％|倍"
VALUE = re.compile(rf"(?P<value>{NUMBER})\s*(?P<unit>{UNIT})?")
VALUE_CELL = re.compile(rf"\s*(?P<value>{NUMBER}|[（(]{NUMBER}[）)])\s*(?P<unit>{UNIT})?\s*")
PERIOD = re.compile(r"(?:19|20)\d{2}(?:[年/-]\d{1,2}(?:[月/-]\d{1,2}日?)?|年)?(?:[AEae]|上半年|下半年|全年|年度|第?[一二三四1234]季度|[Qq][1-4])?")
CLAUSE = re.compile(r"[^。；;\n]+")
COMPLEX = re.compile(r"分别|前者|后者|其|该公司|本公司|公司为|子公司|母公司|旗下|其中")
NUMERIC_REQUEST = re.compile(rf"多少|数值|数据|比较|对比|相差|增幅|降幅|{NUMBER}\s*(?:{UNIT})")


def financial_metrics(text: str) -> list[str]:
    metrics = [metric for metric in metric_terms(text) if metric in FINANCIAL_METRICS]
    metrics.extend(re.findall("|".join(BALANCE_METRICS), text))
    return list(dict.fromkeys(metrics))


def decimal_value(raw: str) -> str | None:
    normalized = unicodedata.normalize("NFKC", raw).strip().replace(",", "").replace("−", "-")
    if normalized.startswith("(") and normalized.endswith(")"):
        normalized = "-" + normalized[1:-1]
    try:
        value = Decimal(normalized)
        return format(value, "f") if value.is_finite() else None
    except InvalidOperation:
        return None


def _original_value_text(text: str, start: int, end: int) -> str:
    """将规范化数值的字符范围映射回原始单元格文本。"""
    offset = 0
    first = None
    for index, character in enumerate(text):
        following = offset + len(unicodedata.normalize("NFKC", character))
        if first is None and offset <= start < following:
            first = index
        if first is not None and offset < end <= following:
            return text[first:index + 1]
        offset = following
    raise ValueError("numeric span is outside the original cell")


def period_value(text: str) -> dict[str, Any] | None:
    dates = re.findall(r"((?:19|20)\d{2})[年/-](\d{1,2})[月/-](\d{1,2})日?(?!\d)", text)
    if dates:
        try:
            values = {date(int(year), int(month), int(day)) for year, month, day in dates}
        except ValueError:
            return None
        if len(values) != 1:
            return None
        value = next(iter(values))
        return {"year": value.year, "granularity": "date", "label": value.isoformat()}
    periods = reporting_periods(text)
    # A 是明确的实际值标记，与没有标记的年份区分处理。
    periods = {re.sub(r"(?<=\d)a$", "", period) for period in periods}
    if len(periods) != 1:
        return None
    label = next(iter(periods)).replace("预测", "")
    year = int(label[:4])
    suffix = label[4:]
    suffix = "" if suffix in {"全年", "年度"} else suffix
    granularity = "half_year" if "半年" in suffix else "quarter" if "季度" in suffix else "date" if suffix else "year"
    return {"year": year, "granularity": granularity, "label": str(year) + suffix}


def _scopes(text: str) -> tuple[str, str, str]:
    forecast = bool(re.search(r"预测|预计|预期|(?:19|20)\d{2}E(?![A-Za-z0-9])", text, re.I))
    actual = bool(re.search(r"实际|(?:19|20)\d{2}A(?![A-Za-z0-9])", text, re.I))
    kind = "forecast" if forecast and not actual else "actual" if actual and not forecast else "unknown"
    consolidated, parent = "合并" in text, bool(re.search(r"母公司(?:口径|报表)|母公司财务|母公司营业", text))
    scope = "consolidated" if consolidated and not parent else "parent" if parent and not consolidated else "unknown"
    businesses = [name for name in domain_policy()["business_scopes"] if name in text]
    business = businesses[0] if len(businesses) == 1 else "unknown"
    return kind, scope, business


def _currency(text: str) -> str:
    currencies = [code for code, pattern in (("CNY", "人民币"), ("USD", "美元"), ("HKD", "港元|港币"))
                  if re.search(pattern, text)]
    return currencies[0] if len(currencies) == 1 else "unknown"


def _company_mentions(text: str, registry: EntityRegistry) -> list[dict[str, Any]]:
    return [mention for mention in registry.mentions(text)
            if any(registry.entities[identifier].kind == "company" for identifier in mention["entity_ids"])]


def _scope_text(text: str, registry: EntityRegistry) -> str:
    # 公司名称中可能含有“风电”等业务词，但不能因此将
    # 全公司营业收入认定为该业务板块的收入。
    for mention in reversed(registry.mentions(text)):
        text = text[:mention["start"]] + " " + text[mention["end"]:]
    return text


def build_fact_requirements(question: str, goals=(), *, registry: EntityRegistry | None = None) -> list[FactRequirement]:
    """编译局部事实要求，避免对无关分句的公司、指标和期间做笛卡尔积。"""
    registry = registry or get_entity_registry()
    global_mentions = _company_mentions(question, registry)
    global_entities = registry.resolved_ids(global_mentions, "company")
    global_periods = reporting_periods(question)
    requirements = []
    parts = [(goal.id, goal.text + " " + " ".join(goal.constraints)) for goal in goals] or [
        (f"g{index}", part) for index, part in enumerate(re.split(r"[，,；;。]", question), 1) if part.strip()]
    for goal_id, text in parts:
        if re.search(r"原因|为什么|为何", text) and not NUMERIC_REQUEST.search(text):
            continue
        metrics = financial_metrics(text)
        if re.search(r"派息率|股利支付率|分红支付率", text):
            metrics = ["每股红利", "每股收益"]
        if not metrics:
            continue
        mentions = _company_mentions(text, registry)
        # 原问题只有一个明确公司时，拆分后的目标可继承该主体。
        if not mentions and len(global_entities) == 1:
            entity = registry.entities[global_entities[0]]
            mentions = [{"quote": entity.canonical_name, "entity_ids": [entity.entity_id],
                         "status": "resolved", "start": 0, "end": 0}]
        if not mentions:
            # 明确命名但未登记的公司不能静默回退到旧的片段级门控。
            from service.core.query_expansion import _extract_organizations
            unknown = _extract_organizations(text)
            mentions = [{"quote": name, "entity_ids": [], "status": "unknown",
                         "start": text.find(name), "end": text.find(name) + len(name)} for name in sorted(unknown)]
        if not mentions:
            continue  # 未明确主体的查询沿用现有证据策略。
        shared_period = period_value(text)
        if shared_period is None and not reporting_periods(text) and len(global_periods) == 1:
            shared_period = period_value(question)
        seen = set()
        company_list = len(mentions) > 1 and all(re.fullmatch(r"[、和与及\s]+", text[left["end"]:right["start"]])
                                               for left, right in zip(mentions, mentions[1:]))
        for position, mention in enumerate(mentions):
            end = mentions[position + 1]["start"] if position + 1 < len(mentions) else len(text)
            local = text[mention["end"]:end]
            local_metrics = financial_metrics(local) or metrics
            local_period = period_value(local) or shared_period
            local_periods = reporting_periods(local)
            # 明确的两期请求需要两个操作数。多公司只有在年份列表确实共享时
            # 才可继承，不能覆盖各公司局部绑定的期间。
            periods = [period_value(value) for value in sorted(local_periods)] if len(local_periods) > 1 else [local_period]
            binding = "resolved"
            if local_period is None and reporting_periods(text) and len(periods) == 1:
                binding = "unknown"
            if local_period is None and re.search(r"今年|去年|上年|上一年|本年|未来|最近|近[一二三\d]年|近年来|本季度|上季度|同期", text):
                binding = "unknown"
            # 起止年份不能证明中间各期都已覆盖；
            # 年份范围须有明确、逐期展开的目标契约才能放行。
            if re.search(r"(?:19|20)\d{2}年?\s*(?:至|到|[—–~～-])\s*(?:19|20)\d{2}", local):
                binding = "unknown"
            if len(local_metrics) > 1 and len(local_periods) > 1:
                binding = "unknown"
            entity_id = mention["entity_ids"][0] if mention["status"] == "resolved" else None
            kind, scope, business = _scopes(_scope_text(local, registry))
            currency = _currency(_scope_text(local, registry))
            if len(global_entities) == 1 or len(mentions) == 1 or company_list:
                fallback_kind, fallback_scope, fallback_business = _scopes(_scope_text(text, registry))
                kind = kind if kind != "unknown" else fallback_kind
                scope = scope if scope != "unknown" else fallback_scope
                business = business if business != "unknown" else fallback_business
                currency = currency if currency != "unknown" else _currency(_scope_text(text, registry))
            if global_mentions:
                prefix = question[:global_mentions[0]["start"]]
                prefix_kind, prefix_scope, prefix_business = _scopes(prefix)
                kind = kind if kind != "unknown" else prefix_kind
                scope = scope if scope != "unknown" else prefix_scope
                business = business if business != "unknown" else prefix_business
                currency = currency if currency != "unknown" else _currency(prefix)
            for metric in local_metrics:
                for period in periods:
                    identity = (entity_id, mention["quote"], metric, json.dumps(period, sort_keys=True), kind, scope, business, currency)
                    if identity in seen:
                        continue
                    seen.add(identity)
                    requirements.append(FactRequirement(
                        f"{goal_id}_r{len(requirements) + 1}", goal_id, entity_id, mention["quote"],
                        metric, period, kind, scope, business, binding, currency,
                    ))
    if goals:
        # 规划模型可能遗漏或过度概括数值目标；
        # 即使已有模型目标，不可变的用户问题仍作为事实要求的兜底。
        for fallback in build_fact_requirements(question, registry=registry):
            covered = any(r.entity_id == fallback.entity_id and r.metric == fallback.metric
                          and r.period == fallback.period and r.binding_status == fallback.binding_status
                          and all(getattr(fallback, key) == "unknown" or getattr(r, key) == getattr(fallback, key)
                                  for key in ("value_kind", "statement_scope", "business_scope", "currency")) for r in requirements)
            if not covered:
                requirements.append(FactRequirement(
                    f"financial_backstop_r{len(requirements) + 1}", "financial_backstop", fallback.entity_id,
                    fallback.entity_name, fallback.metric, fallback.period, fallback.value_kind,
                    fallback.statement_scope, fallback.business_scope, fallback.binding_status, fallback.currency,
                ))
    return requirements


def _fact(chunk: Evidence, registry, *, entity_id, metric, period_text, raw_value, unit, context,
          source, method, attributed=True, valid=True) -> FinancialFact:
    clean_context = _scope_text(context, registry)
    kind, scope, business = _scopes(clean_context)
    # 期间列上的值类型标记比整张表的预测题注更具体，应优先使用。
    period_kind = _scopes(period_text)[0]
    if period_kind != "unknown":
        kind = period_kind
    normalized_unit = unicodedata.normalize("NFKC", unit) if unit else None
    unit_currency = _currency(unit or "")
    currency = unit_currency if unit_currency != "unknown" else _currency(clean_context)
    source = {"chunk_id": chunk.id,
              "document_id": chunk.source.document_id,
              "positions": plain(chunk.location.positions) or plain(chunk.structure.coordinates.get("position_int") or ()), **plain(source)}
    identity = fingerprint([VERSION, source, entity_id, metric, period_text, raw_value])
    value = decimal_value(raw_value)
    period = period_value(period_text)
    return FinancialFact(
        "fact_" + identity[:24], entity_id,
        registry.entities[entity_id].canonical_name if entity_id in registry.entities else "未知主体",
        metric, period, raw_value, value, normalized_unit, currency, kind, scope, business,
        "verified" if attributed and entity_id else "unknown",
        "verified" if attributed and entity_id and valid and value is not None and period else "unknown",
        method, source,
        "approximate" if re.search(r"约|大概|大约", context) else "reported",
        len(value.partition(".")[2]) if value else 0,
    )


def _text_facts(chunk: Evidence, registry: EntityRegistry) -> list[FinancialFact]:
    text = chunk.text
    facts = []
    for clause_match in CLAUSE.finditer(text):
        clause = clause_match.group()
        mentions = _company_mentions(clause, registry)
        # 按完整的有序对应关系核验“分别”列表，不能仅检查名称和数字共现。
        if "分别" in clause and len(mentions) >= 2:
            metrics = financial_metrics(clause)
            values = list(re.finditer(rf"({NUMBER})\s*({UNIT})", clause))
            metric_pattern = _metric_pattern(metrics[0]) if len(metrics) == 1 else None
            metric_match = metric_pattern.search(clause) if metric_pattern else None
            period = period_value(clause)
            prefix = clause[:mentions[-1]["end"]]
            list_only = re.sub(r"(?:、|和|与|及|\s)+", "", prefix)
            company_list = "".join(mention["quote"] for mention in mentions)
            if (metric_match and period and list_only == company_list
                    and mentions[-1]["end"] <= metric_match.start()
                    and len(values) == len(mentions) and not COMPLEX.search(clause.replace("分别", ""))
                    and all(value.start() > clause.index("分别") for value in values)):
                for mention, value in zip(mentions, values):
                    entity = mention["entity_ids"][0] if mention["status"] == "resolved" else None
                    facts.append(_fact(chunk, registry, entity_id=entity, metric=metrics[0], period_text=clause,
                                       raw_value=value[1], unit=value[2], context=clause,
                                       source={"quote": clause, "start": clause_match.start(), "end": clause_match.end(),
                                               "entity_quote": mention["quote"], "value_start": clause_match.start() + value.start()},
                                       method="ordered_respectively"))
                continue
        # 按逗号和明确公司名称划分直接陈述；
        # 每段都必须拥有自己的主体、指标、期间和数值。
        for segment in re.finditer(r"(?:[^，,]|[,，](?=\d{3}(?:\D|$)))+", clause):
            quote = segment.group().strip()
            local_mentions = _company_mentions(quote, registry)
            metrics = financial_metrics(quote)
            if len(local_mentions) != 1 or len(metrics) != 1:
                continue
            mention = local_mentions[0]
            metric_match = _metric_pattern(metrics[0]).search(quote)
            if metric_match is None:
                continue
            tail = quote[metric_match.end():]
            # 数值前只允许关系连接词；第二个年份、其他指标、范围
            # 或夹入的公司名称都不能被误当成数值的一部分。
            value_match = re.fullmatch(rf"\s*(?:为|是|达到|达|实现|约为|约|:|：)?\s*({NUMBER})\s*({UNIT})\s*[。.]?", tail)
            if value_match is None:
                continue
            entity = mention["entity_ids"][0] if mention["status"] == "resolved" else None
            prefix = quote[:metric_match.start()]
            clean_prefix = prefix[:mention["start"]] + prefix[mention["end"]:]
            clean_prefix = PERIOD.sub("", clean_prefix)
            clean_prefix = re.sub("(?:" + "|".join(domain_policy()["business_scopes"]) + r")(?:板块|业务|分部)?", "", clean_prefix)
            clean_prefix = re.sub(r"合并报表|合并口径|母公司口径|实际|预测|预计|截至|人民币|美元|港币|港元|的|全年|年度|\s", "", clean_prefix)
            if clean_prefix or mention["end"] > metric_match.start() or COMPLEX.search(quote):
                entity = None
            start = clause_match.start() + segment.start() + len(segment.group()) - len(segment.group().lstrip())
            facts.append(_fact(chunk, registry, entity_id=entity, metric=metrics[0], period_text=prefix,
                               raw_value=value_match[1], unit=value_match[2], context=quote,
                               source={"quote": quote, "start": start, "end": start + len(quote),
                                       "entity_quote": mention["quote"]}, method="direct_statement"))
    return facts


def _metric_pattern(metric: str):
    aliases = [metric, *domain_policy()["aliases"].get(metric, [])]
    return re.compile("|".join(re.escape(alias) for alias in sorted(aliases, key=len, reverse=True)), re.I)


def _document_subject(chunk, registry) -> tuple[str | None, list[dict]]:
    attribution = chunk.structure.attribution
    proof = [item for item in attribution.subject_proof
             if item.get("source") in {"explicit_label", "opening_company_and_code"} and item.get("status") == "resolved"
             and len(item.get("entity_ids") or []) == 1]
    valid = []
    for item in proof:
        quote = str(item.get("quote") or "")
        if item["source"] == "explicit_label":
            match = re.fullmatch(r"\s*(?:公司名称|证券简称|股票简称|研究对象|研究标的|报告主体)\s*[:：]\s*(.+)", quote)
            matches = bool(match and tuple(registry.exact(match[1].strip())) == tuple(item["entity_ids"]))
        else:
            match = OPENING_COMPANY_CODE.fullmatch(quote)
            matches = bool(match and tuple(registry.exact(match[1].strip())) == tuple(registry.exact(match[2])) == tuple(item["entity_ids"]))
        if not matches:
            return None, proof
        valid.append(item)
    proof = valid
    ids = {item["entity_ids"][0] for item in proof}
    declared = set(attribution.subject_ids)
    if (len(ids) == 1 and ids == declared and attribution.subject_status == "resolved"
            and next(iter(ids)) in registry.entities):
        return next(iter(ids)), proof
    return None, proof


def _table_facts(chunk: Evidence, registry: EntityRegistry) -> list[FinancialFact]:
    table = chunk.structure.table
    if table.binding != "bound" or not table.reliable:
        return []
    facts = []
    for row in table.rows:
        if not row.grounded:
            continue
        headers, cells, columns = row.headers, row.cells, row.columns
        subject, subject_proof = _document_subject(chunk, registry)
        _, ownership_status, bindings, _ = resolve_table_bindings((row.original(),), registry)
        row_scope = " ".join((row.context, row.preamble))
        label_metrics = financial_metrics(str(cells[0]))
        section = " ".join((row.context, table.caption)).strip()
        for index, cell in enumerate(cells):
            value_match = VALUE_CELL.fullmatch(unicodedata.normalize("NFKC", str(cell)))
            if value_match is None:
                continue
            header = str(headers[index])
            header_metrics = financial_metrics(header)
            metrics = header_metrics or label_metrics
            if len(metrics) != 1:
                continue
            period_text = header if period_value(header) else row_scope + " " + table.caption
            binding_ids = {binding["entity_id"] for binding in bindings if columns[index] in binding.get("value_column_indices", [])}
            entity = next(iter(binding_ids)) if len(binding_ids) == 1 else None
            method = "table_cell_binding"
            proof = [binding for binding in bindings if columns[index] in binding.get("value_column_indices", [])]
            if not bindings and ownership_status == "document_context":
                # 没有明确公司行的财务表，需要可靠报告主体与明确章节共同支持；
                # 仅有文件名不能证明数值归属。
                local_entities = registry.resolved_ids(_company_mentions(section, registry), "company")
                statement = bool(re.search(r"(?:合并|母公司).*(?:利润表|财务报表|财务数据)|公司财务报表|财务预测|盈利预测|主要财务指标|关键财务与估值指标", section))
                statement = statement and not re.search(r"子公司|参股公司|联营|可比公司|关联公司|行业|全市场", section)
                if subject and statement and (not local_entities or local_entities == [subject]):
                    entity, proof, method = subject, subject_proof, "report_subject_and_section"
            # 单位可以来自当前值、表头、行标签、题注或章节，
            # 不能从另一条扩展行中借用。
            unit_context = " ".join((header, str(cells[0]), row_scope, table.caption))
            units = set(re.findall(UNIT, unit_context))
            unit = value_match["unit"] or (next(iter(units)) if len(units) == 1 else None)
            origin = table.binding_source
            # 模型重建的表格保留来源，但不能仅凭自身声明 bound 就认定核验通过；
            # 仍需独立的结构校验依据。
            trusted_grid = origin == "deterministic"
            qualifier = re.search(r"同比|环比|增长率|增速|增幅|降幅|占比|比重", header + " " + str(cells[0]))
            fact_metric = metrics[0] + "/" + qualifier.group() if qualifier else metrics[0]
            fact = _fact(chunk, registry, entity_id=entity, metric=fact_metric, period_text=period_text,
                               raw_value=_original_value_text(str(cell), *value_match.span("value")), unit=unit,
                               context=unit_context,
                               source={"table_id": chunk.location.table_id, "row": row.index,
                                       "column": columns[index], "header": header, "cell": str(cell),
                                       "row_label": str(cells[0]), "section": section, "caption": table.caption,
                                       "binding_evidence": proof, "binding_source": origin,
                                       "ownership_status": ownership_status},
                               method=method, attributed=trusted_grid, valid=trusted_grid)
            if ownership_status == "ambiguous":
                fact.attribution_status = "ambiguous"
            # 值类型属于该单元格自己的表头或行标签。
            # 预测表的题注也可能覆盖没有 A/E 标记的历史列，不能统一赋为预测值。
            primary_scope = _scope_text(header + " " + str(cells[0]), registry)
            primary_kind, primary_statement, primary_business = _scopes(primary_scope)
            fact.value_kind = primary_kind
            if re.search(r"(?:^|[/｜|])\s*母公司\s*(?:$|[/｜|])", primary_scope):
                primary_statement = "parent"
            if primary_statement != "unknown":
                fact.statement_scope = primary_statement
            if primary_business != "unknown":
                fact.business_scope = primary_business
            facts.append(fact)
    return facts


def extract_financial_facts(chunks: EvidenceCollection, *, registry: EntityRegistry | None = None) -> list[FinancialFact]:
    registry = registry or get_entity_registry()
    facts = []
    for chunk in chunks:
        if chunk.structure.kind == "figure" and chunk.structure.figure.status != "extracted":
            continue
        is_table = chunk.structure.kind == "table" or bool(chunk.structure.table.rows and chunk.structure.table.rows[0].cells)
        facts.extend(_table_facts(chunk, registry) if is_table else _text_facts(chunk, registry))
        if not is_table:
            from service.core.financial_fact_models import cached_facts
            facts.extend(cached_facts(chunk, registry))
    return list({fact.fact_id: fact for fact in facts}.values())


def _match(requirement: FactRequirement, fact: FinancialFact) -> str:
    if not requirement.entity_id or requirement.binding_status != "resolved":
        return "unknown"
    if fact.entity_id and fact.entity_id != requirement.entity_id:
        return "not_applicable"
    if fact.metric != requirement.metric or (requirement.period and fact.period != requirement.period):
        return "not_applicable"
    if fact.attribution_status != "verified" or fact.verification_status != "verified":
        return "unknown"
    if fact.unit is None and requirement.metric not in {"每股收益", "每股红利", "市盈率", "市净率"}:
        return "unknown"
    if fact.unit in {"%", "倍", "元/股"} and requirement.metric in {
        "营业收入", "营业成本", "净利润", "归母净利润", "扣非归母净利润",
        "经营活动现金流量净额", "企业自由现金流", "股权自由现金流",
        "财务费用金额", "财务费用", "销售费用", "研发费用", "管理费用", *BALANCE_METRICS,
    }:
        return "not_applicable"
    for key in ("value_kind", "statement_scope", "business_scope", "currency"):
        expected, actual = getattr(requirement, key), getattr(fact, key)
        if expected != "unknown" and actual == "unknown":
            return "unknown"
        if expected != "unknown" and expected != actual:
            return "not_applicable"
    if requirement.business_scope == "unknown" and fact.business_scope != "unknown":
        return "not_applicable"
    return "usable"


def normalized_value(fact: FinancialFact) -> tuple[Decimal, str, str]:
    unit = fact.unit or "unknown"
    scales = {"亿元": Decimal("100000000"), "百万元": Decimal("1000000"), "万元": Decimal("10000"), "元": Decimal(1)}
    base = unit.replace("人民币", "")
    return Decimal(fact.value) * scales.get(base, Decimal(1)), "money" if base in {*scales, "美元", "港元"} else base, fact.currency


def _infer_source_fact_requirements(
    question: str, chunks: EvidenceCollection, goals: Sequence[QueryGoal], registry: EntityRegistry,
) -> list[FactRequirement]:
    """问题未指定公司时，依据可靠的报告主体建立兜底要求。"""
    metrics = financial_metrics(question)
    if not metrics or (re.search(r"原因|为什么|为何", question) and not NUMERIC_REQUEST.search(question)):
        return []

    subjects = {_document_subject(chunk, registry)[0] for chunk in chunks} - {None}
    named_sources = any(
        _company_mentions(chunk.text, registry)
        or chunk.structure.attribution.subject_ids for chunk in chunks
    )
    if not subjects and not named_sources:
        return []

    # 来源中出现公司名称，只能说明需要核验，不能据此猜测查询主体。
    # 仅有一个可靠报告主体时才允许继承；多主体仍生成未知要求，避免绕过核验。
    entity = next(iter(subjects)) if len(subjects) == 1 else None
    from service.core.query_contract import REFERENCE
    if REFERENCE.search(question):
        entity = None  # 未消解的会话指代不能直接继承检索到的报告主体。

    kind, scope, business = _scopes(_scope_text(question, registry))
    return [
        FactRequirement(
            requirement_id=f"g1_r{index}",
            goal_id=goals[0].id if goals else "g1",
            entity_id=entity,
            entity_name=registry.entities[entity].canonical_name if entity else "待确认公司",
            metric=metric,
            period=period_value(question),
            value_kind=kind,
            statement_scope=scope,
            business_scope=business,
            binding_status="resolved" if entity else "unknown",
            currency=_currency(question),
        ) for index, metric in enumerate(metrics, 1)
    ]


def _supplement_model_facts(
    requirements: list[FactRequirement], facts: list[FinancialFact], chunks: EvidenceCollection, *,
    registry: EntityRegistry, client: Any | None,
) -> list[FinancialFact]:
    """规则事实存在缺口时，尝试补充经过独立核验的复杂正文事实。"""
    model_enabled = os.getenv("RAG_FINANCIAL_FACT_MODEL_ENABLED", "false").strip().casefold() in {"true", "1", "yes", "on"}
    if not model_enabled:
        return facts
    if all(any(_match(requirement, fact) == "usable" for fact in facts) for requirement in requirements):
        return facts

    complex_chunks = chunks.select(chunk.id for chunk in chunks if COMPLEX.search(chunk.text))
    if not complex_chunks:
        return facts

    from service.core.financial_fact_models import extract_and_verify_candidates
    from service.core.evidence_sufficiency import get_evidence_sufficiency_client
    try:
        candidates = extract_and_verify_candidates(
            complex_chunks, requirements, client or get_evidence_sufficiency_client(), registry=registry,
        )
        return list({fact.fact_id: fact for fact in [*facts, *candidates]}.values())
    except Exception:
        # 模型或配置异常时保留已有规则事实，缺失目标仍需通过后续核验。
        return facts


def _match_requirement_facts(
    requirement: FactRequirement, facts: list[FinancialFact],
) -> tuple[list[dict[str, Any]], list[FinancialFact]]:
    """记录每条事实对当前要求的匹配状态，并收集可用候选。"""
    matches = []
    candidates = []
    for fact in facts:
        status = _match(requirement, fact)
        matches.append({
            "requirement_id": requirement.requirement_id,
            "goal_id": requirement.goal_id,
            "fact_id": fact.fact_id,
            "status": status,
        })
        if status == "usable":
            candidates.append(fact)
    return matches, candidates


def _candidate_scopes_are_ambiguous(
    candidates: list[FinancialFact], values: set[tuple[Decimal, str, str]],
) -> bool:
    """判断候选是否缺少足以比较数值的共同期间和口径依据。"""
    scopes = {
        (json.dumps(fact.period, sort_keys=True), fact.value_kind, fact.statement_scope,
         fact.business_scope, fact.currency) for fact in candidates
    }
    if len(scopes) > 1:
        return True
    if len(values) <= 1:
        return False

    # 近似值与精确值不同，不能直接宣布事实冲突。
    if any(fact.value_qualifier == "approximate" for fact in candidates):
        return True

    origins = {(fact.source.get("table_id"), fact.source.get("row")) for fact in candidates}
    shared_row = len(origins) == 1 and all(fact.source.get("table_id") for fact in candidates)
    # 独立来源都写 unknown，不代表口径相同；同一原表行则保留已有结构依据。
    return not shared_row and any(
        fact.statement_scope == "unknown" or fact.value_kind == "unknown" for fact in candidates
    )


def _candidate_support_status(candidates: list[FinancialFact]) -> str:
    """将可用候选归为支持、缺失或冲突；先确认可比性，再比较数值。"""
    if not candidates:
        return "missing"
    # 使用精确十进制和统一单位比较，80 亿元与 8,000 百万元是相同数值。
    values = {normalized_value(fact) for fact in candidates}
    if _candidate_scopes_are_ambiguous(candidates, values):
        return "missing"
    if len(values) > 1:
        return "conflict"
    return "supported"


def _record_requirement_coverage(
    bundle: FactEvidenceBundle, requirement: FactRequirement,
    matches: list[dict[str, Any]], candidates: list[FinancialFact], status: str,
) -> None:
    """将单项核验结论同步写入匹配记录、目标覆盖和支持事实集合。"""
    supporting_ids = [fact.fact_id for fact in candidates] if status == "supported" else []
    if status != "supported":
        # 单条事实通过匹配后，候选集合仍可能有歧义或冲突。
        # 必须同步撤销 usable，避免回答上下文继续把这些事实当作已通过的支持。
        match_status = "conflict" if status == "conflict" else "unknown"
        for match in matches:
            if match["status"] == "usable":
                match["status"] = match_status

    bundle.matches.extend(matches)
    bundle.goal_coverage.append({
        "goal_id": requirement.goal_id,
        "requirement_id": requirement.requirement_id,
        "status": status,
        "supporting_fact_ids": supporting_ids,
    })
    bundle.selected_fact_ids.extend(supporting_ids)
    if status != "supported":
        period = (requirement.period or {}).get("label", "期间未明确")
        reason = "事实冲突" if status == "conflict" else "缺少归属和口径可确认的事实"
        bundle.missing_requirements.append(f"{requirement.entity_name} / {requirement.metric} / {period}：{reason}")


def _verify_calculation_inputs(question: str, chunks: EvidenceCollection, bundle: FactEvidenceBundle) -> None:
    """仅用已选中的事实检查计算输入，并记录计算结果或缺口。"""
    if bundle.missing_requirements or not re.search(
        r"相差|差额|高多少|低多少|增加多少|减少多少|下降多少|增长率|增幅|降幅|派息率|股利支付率|分红支付率", question,
    ):
        return

    from service.core.derived_evidence import build_derived_evidence
    # 完整片段仅作为出处；计算操作数由 selected_fact_ids 确定，不能重新取其他数值。
    selected_ids = set(bundle.selected_fact_ids)
    verified_facts = [fact for fact in bundle.facts if fact.fact_id in selected_ids]
    bundle.derived_evidence = build_derived_evidence(question, chunks, verified_facts=verified_facts)
    if not bundle.derived_evidence or any(item["status"] != "supported" for item in bundle.derived_evidence):
        bundle.missing_requirements.append("缺少实体、期间、口径和单位可比的已核验计算输入")
        bundle.goal_coverage.append({
            "goal_id": bundle.requirements[0].goal_id,
            "requirement_id": "calculation",
            "status": "missing",
            "supporting_fact_ids": [],
        })


def assess_financial_facts(
    question: str, chunks: EvidenceCollection, goals: Sequence[QueryGoal] = (), *,
    registry: EntityRegistry | None = None, allow_model: bool = False, client: Any | None = None,
) -> FactEvidenceBundle:
    """组织事实要求、候选抽取、逐目标核验和计算检查，返回统一报告。"""
    registry = registry or get_entity_registry()
    requirements = build_fact_requirements(question, goals, registry=registry)
    if not requirements:
        requirements = _infer_source_fact_requirements(question, chunks, goals, registry)
    bundle = FactEvidenceBundle(requirements=requirements)
    if not bundle.applicable:
        return bundle

    bundle.facts = extract_financial_facts(chunks, registry=registry)
    if allow_model:
        bundle.facts = _supplement_model_facts(
            bundle.requirements, bundle.facts, chunks, registry=registry, client=client,
        )

    # 按要求分别核验，同一事实对 A 目标可用，不代表它也能支持 B 目标。
    for requirement in bundle.requirements:
        matches, candidates = _match_requirement_facts(requirement, bundle.facts)
        status = _candidate_support_status(candidates)
        _record_requirement_coverage(bundle, requirement, matches, candidates, status)

    bundle.selected_fact_ids = list(dict.fromkeys(bundle.selected_fact_ids))
    # 使用已选中的事实来做计算证据的校验
    _verify_calculation_inputs(question, chunks, bundle)
    return bundle

def financial_rejection(evidence: EvidenceCollection, bundle: FactEvidenceBundle, goals=()) -> EvidenceAssessment:
    """把事实缺口转成逐目标报告，模型同意不能覆盖规则缺口。"""
    identifiers = [item.id for item in evidence]
    coverage = []
    for goal in goals:
        local = [entry for entry in bundle.goal_coverage if entry["goal_id"] == goal.id]
        status = "conflict" if any(entry["status"] == "conflict" for entry in local) else "missing" if any(
            entry["status"] == "missing" for entry in local) or not local else "supported"
        fact_ids = {identifier for entry in local for identifier in entry["supporting_fact_ids"]}
        selected = [fact for fact in bundle.facts if fact.fact_id in fact_ids]
        coverage.append({"goal_id": goal.id, "status": status,
                         "supporting_chunk_ids": list(dict.fromkeys(fact.source["chunk_id"] for fact in selected)),
                         "supporting_fact_ids": [fact.fact_id for fact in selected]})
    supported = {fact.source["chunk_id"] for fact in bundle.facts if fact.fact_id in bundle.selected_fact_ids}
    return EvidenceAssessment(
        sufficient=False, reason="财务事实未通过逐目标归属与口径核验", source="rule",
        missing_requirements=bundle.missing_requirements,
        supporting_chunk_indices=[index for index, identifier in enumerate(identifiers, 1) if identifier in supported],
        evaluated_chunk_count=len(evidence), evaluated_chunk_ids=identifiers, goal_coverage=coverage, facts=bundle,
    )

def render_fact(fact: FinancialFact, goal_ids: list[str]) -> str:
    period = (fact.period or {}).get("label", "未知")
    lines = [f"目标：{', '.join(goal_ids)}；事实ID：{fact.fact_id}", f"公司：{fact.entity_name}",
             f"指标：{fact.metric}；期间：{period}", f"数值：{'约' if fact.value_qualifier == 'approximate' else ''}{fact.value}{fact.unit or '（单位未说明）'}",
             f"实际/预测：{fact.value_kind}；报表口径：{fact.statement_scope}；币种：{fact.currency}"]
    source = fact.source
    if "table_id" in source:
        lines.append(f"原表：{source['table_id']}；原始行：{source['row']}；原始列：{source['column']}")
        lines.append(f"表头：{source['header']}；单元格原文：{source['cell']}")
    elif fact.verification_method == "direct_statement":
        lines.append("原文依据：" + source["quote"])
    # 有序列表的完整引文可能包含其他公司的数值，
    # 因此完整关系只保留在审计记录中，回答上下文仅接收当前事实。
    return "\n".join(lines)


def project_fact_context(evidence: EvidenceCollection, bundle: FactEvidenceBundle) -> tuple[AnswerBlock, ...]:
    """生成筛选正文，来源原文和查询判定分别保存在证据集合与事实报告中。"""
    if not bundle.applicable:
        return tuple(AnswerBlock(item.id, item.text) for item in evidence)
    selected = {fact.fact_id: fact for fact in bundle.facts if fact.fact_id in bundle.selected_fact_ids}
    output = []
    for item in evidence:
        local = [fact for fact in selected.values() if fact.source["chunk_id"] == item.id]
        blocks = []
        for fact in local:
            goal_ids = list(dict.fromkeys(entry["goal_id"] for entry in bundle.matches
                                         if entry["fact_id"] == fact.fact_id and entry["status"] == "usable"))
            blocks.append(render_fact(fact, goal_ids))
        local_ids = {fact.fact_id for fact in local}
        for derived in bundle.derived_evidence:
            if derived["status"] == "supported" and local_ids.intersection(derived.get("input_fact_ids") or []):
                blocks.append(f"服务端计算：{derived['requirement']}；操作：{derived['operation']}；"
                              f"结果：{derived['result']}{derived['unit']}；输入事实："
                              + ", ".join(derived["input_fact_ids"]))
        if item.structure.kind != "table":
            qualitative = [match.group().strip() for match in CLAUSE.finditer(item.text)
                           if not re.search(r"\d", match.group()) and not re.search(UNIT, match.group())
                           and not (financial_metrics(match.group()) and re.search(r"[零〇一二三四五六七八九十百千万亿两]", match.group()))]
            blocks.extend(qualitative)
        output.append(AnswerBlock(item.id, "\n\n".join(blocks), tuple(fact.fact_id for fact in local)))
    return tuple(output)
