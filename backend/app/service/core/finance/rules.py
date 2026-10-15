"""原文和表格的确定性财务事实提取；不调用模型，不判断整个问题是否充分。"""

from __future__ import annotations

import re
import unicodedata
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any

from domain.evidence import Evidence, fingerprint, plain
from domain.facts import VERSION, FinancialFact
from service.core.entities.attribution import EntityRegistry, OPENING_COMPANY_CODE, resolve_table_bindings
from service.core.query.semantics import domain_policy, reporting_periods
from service.core.finance.metrics import metric_aliases, parse_metric_mentions, table_metric_mentions


NUMBER = r"[+\-−]?\d+(?:[,，]\d{3})*(?:\.\d+)?"
UNIT = r"人民币亿元|人民币万元|人民币元|亿元|百万元|万元|元(?:/股)?|美元|港元|个百分点|%|％|倍"
VALUE_CELL = re.compile(rf"\s*(?P<value>{NUMBER}|[（(]{NUMBER}[）)])\s*(?P<unit>{UNIT})?\s*")
PERIOD = re.compile(r"(?:19|20)\d{2}(?:[年/-]\d{1,2}(?:[月/-]\d{1,2}日?)?|年)?(?:[AEae]|上半年|下半年|全年|年度|第?[一二三四1234]季度|[Qq][1-4])?")
CLAUSE = re.compile(r"[^。；;\n]+")
COMPLEX = re.compile(r"分别|前者|后者|其|该公司|本公司|公司为|子公司|母公司|旗下|其中")


def financial_metrics(text: str) -> list[str]:
    """兼容旧调用方：度量解析统一完成后，仅返回基础指标名称。"""
    return list(dict.fromkeys(mention.metric for mention in parse_metric_mentions(text)))


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


def period_order_key(period: dict[str, Any]) -> tuple[int, ...]:
    """同粒度期间的时间顺序；中文季度不能直接按字符串大小比较。"""
    label = period["label"].replace("上半年", "1").replace("下半年", "2")
    for chinese, number in zip("一二三四", "1234"):
        label = label.replace(chinese, number)
    return tuple(int(value) for value in re.findall(r"\d+", label))


def parse_scopes(text: str) -> tuple[str, str, str]:
    forecast = bool(re.search(r"预测|预计|预期|(?:19|20)\d{2}E(?![A-Za-z0-9])", text, re.I))
    actual = bool(re.search(r"实际|(?:19|20)\d{2}A(?![A-Za-z0-9])", text, re.I))
    kind = "forecast" if forecast and not actual else "actual" if actual and not forecast else "unknown"
    consolidated, parent = "合并" in text, bool(re.search(r"母公司(?:口径|报表)|母公司财务|母公司营业", text))
    scope = "consolidated" if consolidated and not parent else "parent" if parent and not consolidated else "unknown"
    businesses = [name for name in domain_policy()["business_scopes"] if name in text]
    business = businesses[0] if len(businesses) == 1 else "unknown"
    return kind, scope, business


def parse_currency(text: str) -> str:
    currencies = [code for code, pattern in (("CNY", "人民币"), ("USD", "美元"), ("HKD", "港元|港币"))
                  if re.search(pattern, text)]
    return currencies[0] if len(currencies) == 1 else "unknown"


def company_mentions(text: str, registry: EntityRegistry) -> list[dict[str, Any]]:
    return [mention for mention in registry.mentions(text)
            if any(registry.entities[identifier].kind == "company" for identifier in mention["entity_ids"])]


def scope_text(text: str, registry: EntityRegistry) -> str:
    # 公司名称中可能含有“风电”等业务词，但不能因此将
    # 全公司营业收入认定为该业务板块的收入。
    for mention in reversed(registry.mentions(text)):
        text = text[:mention["start"]] + " " + text[mention["end"]:]
    return text


def make_fact(chunk: Evidence, registry, *, entity_id, metric, period_text, raw_value, unit, context,
          source, method, attributed=True, valid=True,
          measure_kind="level", comparison_basis="not_applicable") -> FinancialFact:
    clean_context = scope_text(context, registry)
    kind, scope, business = parse_scopes(clean_context)
    # 期间列上的值类型标记比整张表的预测题注更具体，应优先使用。
    period_kind = parse_scopes(period_text)[0]
    if period_kind != "unknown":
        kind = period_kind
    normalized_unit = unicodedata.normalize("NFKC", unit) if unit else None
    unit_currency = parse_currency(unit or "")
    currency = unit_currency if unit_currency != "unknown" else parse_currency(clean_context)
    source = {"chunk_id": chunk.id,
              "document_id": chunk.source.document_id,
              "positions": plain(chunk.location.positions) or plain(chunk.structure.coordinates.get("position_int") or ()), **plain(source)}
    identity = fingerprint([VERSION, source, entity_id, metric, measure_kind, comparison_basis, period_text, raw_value])
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
        measure_kind, comparison_basis,
    )


def extract_text_facts(chunk: Evidence, registry: EntityRegistry) -> list[FinancialFact]:
    text = chunk.text
    facts = []
    for clause_match in CLAUSE.finditer(text):
        clause = clause_match.group()
        mentions = company_mentions(clause, registry)
        # 按完整的有序对应关系核验“分别”列表，不能仅检查名称和数字共现。
        if "分别" in clause and len(mentions) >= 2:
            metrics = parse_metric_mentions(clause)
            values = list(re.finditer(rf"({NUMBER})\s*({UNIT})", clause))
            pattern = metric_pattern(metrics[0].metric) if len(metrics) == 1 else None
            metric_match = pattern.search(clause) if pattern else None
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
                    facts.append(make_fact(chunk, registry, entity_id=entity, metric=metrics[0].metric, period_text=clause,
                                       raw_value=value[1], unit=value[2], context=clause,
                                       source={"quote": clause, "start": clause_match.start(), "end": clause_match.end(),
                                               "entity_quote": mention["quote"], "value_start": clause_match.start() + value.start()},
                                       method="ordered_respectively", measure_kind=metrics[0].measure_kind,
                                       comparison_basis=metrics[0].comparison_basis))
                continue
        # 按逗号和明确公司名称划分直接陈述；
        # 每段都必须拥有自己的主体、指标、期间和数值。
        for segment in re.finditer(r"(?:[^，,]|[,，](?=\d{3}(?:\D|$)))+", clause):
            quote = segment.group().strip()
            local_mentions = company_mentions(quote, registry)
            metrics = parse_metric_mentions(quote)
            if len(local_mentions) > 1 or len(metrics) != 1:
                continue
            mention = local_mentions[0] if local_mentions else None
            metric = metrics[0]
            tail = quote[metric.end:]
            # 数值前只允许关系连接词；第二个年份、其他指标、范围
            # 或夹入的公司名称都不能被误当成数值的一部分。
            value_match = re.fullmatch(rf"\s*(?:为|是|达到|达|实现|约为|约|:|：)?\s*({NUMBER})\s*({UNIT})\s*[。.]?", tail)
            if value_match is None:
                continue
            entity = mention["entity_ids"][0] if mention and mention["status"] == "resolved" else None
            prefix = quote[:metric.start]
            clean_prefix = prefix[:mention["start"]] + prefix[mention["end"]:] if mention else prefix
            clean_prefix = PERIOD.sub("", clean_prefix)
            clean_prefix = re.sub("(?:" + "|".join(domain_policy()["business_scopes"]) + r")(?:板块|业务|分部)?", "", clean_prefix)
            clean_prefix = re.sub(r"合并报表|合并口径|母公司口径|实际|预测|预计|截至|人民币|美元|港币|港元|的|全年|年度|\s", "", clean_prefix)
            if clean_prefix or (mention and mention["end"] > metric.start) or COMPLEX.search(quote):
                entity = None
            start = clause_match.start() + segment.start() + len(segment.group()) - len(segment.group().lstrip())
            facts.append(make_fact(chunk, registry, entity_id=entity, metric=metric.metric, period_text=prefix,
                               raw_value=value_match[1], unit=value_match[2], context=quote,
                               source={"quote": quote, "start": start, "end": start + len(quote),
                                       "entity_quote": mention["quote"] if mention else "",
                                       "value_start": start + metric.end + value_match.start(1)}, method="direct_statement",
                               measure_kind=metric.measure_kind, comparison_basis=metric.comparison_basis))
    return facts


def metric_pattern(metric: str):
    aliases = metric_aliases(metric)
    return re.compile("|".join(re.escape(alias) for alias in sorted(aliases, key=len, reverse=True)), re.I)


def document_subject(chunk, registry) -> tuple[str | None, list[dict]]:
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


def extract_table_facts(chunk: Evidence, registry: EntityRegistry) -> list[FinancialFact]:
    table = chunk.structure.table
    if table.binding != "bound" or not table.reliable:
        return []
    facts = []
    for row in table.rows:
        if not row.grounded:
            continue
        headers, cells, columns = row.headers, row.cells, row.columns
        subject, subject_proof = document_subject(chunk, registry)
        _, ownership_status, bindings, _ = resolve_table_bindings((row.original(),), registry)
        row_scope = " ".join((row.context, row.preamble))
        section = " ".join((row.context, table.caption)).strip()
        for index, cell in enumerate(cells):
            value_match = VALUE_CELL.fullmatch(unicodedata.normalize("NFKC", str(cell)))
            if value_match is None:
                continue
            header = str(headers[index])
            metrics = table_metric_mentions(header, str(cells[0]))
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
                local_entities = registry.resolved_ids(company_mentions(section, registry), "company")
                statement = bool(re.search(r"(?:合并|母公司).*(?:利润表|财务报表|财务数据)|公司财务报表|财务预测|盈利预测|主要财务指标|关键财务与估值指标", section))
                statement = statement and not re.search(r"子公司|参股公司|联营|可比公司|关联公司|行业|全市场", section)
                if subject and statement and (not local_entities or local_entities == [subject]):
                    entity, proof, method = subject, subject_proof, "report_subject_and_section"
            # 单位可以来自当前值、表头、行标签、题注或章节，
            # 不能从另一条扩展行中借用。
            unit_context = " ".join((header, str(cells[0]), row_scope, table.caption))
            local_units = set(re.findall(UNIT, header + " " + str(cells[0])))
            units = local_units or set(re.findall(UNIT, row_scope + " " + table.caption))
            unit = value_match["unit"] or (next(iter(units)) if len(units) == 1 else None)
            origin = table.binding_source
            # 模型重建的表格保留来源，但不能仅凭自身声明 bound 就认定核验通过；
            # 仍需独立的结构校验依据。
            trusted_grid = origin == "deterministic"
            metric = metrics[0]
            fact = make_fact(chunk, registry, entity_id=entity, metric=metric.metric, period_text=period_text,
                               raw_value=_original_value_text(str(cell), *value_match.span("value")), unit=unit,
                               context=unit_context,
                               source={"table_id": chunk.location.table_id, "row": row.index,
                                       "column": columns[index], "header": header, "cell": str(cell),
                                       "row_label": str(cells[0]), "section": section, "caption": table.caption,
                                       "binding_evidence": proof, "binding_source": origin,
                                       "ownership_status": ownership_status},
                               method=method, attributed=trusted_grid, valid=trusted_grid,
                               measure_kind=metric.measure_kind, comparison_basis=metric.comparison_basis)
            if ownership_status == "ambiguous":
                fact.attribution_status = "ambiguous"
            # 值类型属于该单元格自己的表头或行标签。
            # 预测表的题注也可能覆盖没有 A/E 标记的历史列，不能统一赋为预测值。
            primary_scope = scope_text(header + " " + str(cells[0]), registry)
            primary_kind, primary_statement, primary_business = parse_scopes(primary_scope)
            fact.value_kind = primary_kind
            if re.search(r"(?:^|[/｜|])\s*母公司\s*(?:$|[/｜|])", primary_scope):
                primary_statement = "parent"
            if primary_statement != "unknown":
                fact.statement_scope = primary_statement
            if primary_business != "unknown":
                fact.business_scope = primary_business
            facts.append(fact)
    return facts


def normalized_value(fact: FinancialFact) -> tuple[Decimal, str, str]:
    unit = fact.unit or "unknown"
    scales = {"亿元": Decimal("100000000"), "百万元": Decimal("1000000"), "万元": Decimal("10000"), "元": Decimal(1)}
    base = unit.replace("人民币", "")
    return Decimal(fact.value) * scales.get(base, Decimal(1)), "money" if base in {*scales, "美元", "港元"} else base, fact.currency
