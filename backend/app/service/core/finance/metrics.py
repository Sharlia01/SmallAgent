"""财务指标的共享解析规则；查询、正文和表格使用相同的度量语义。"""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
import unicodedata

from service.core.query.semantics import domain_policy


@dataclass(frozen=True)
class MetricMention:
    """原文中的指标提及，仅描述语义，不代表已取得数值或完成事实核验。"""

    metric: str     #标准化的基础指标，例如把“营收”统一为“营业收入”
    raw_label: str   #原文表达，便于追溯
    measure_kind: str    #原值、增长率、增量或占比
    comparison_basis: str   #同比、环比、不适用或未知
    start: int   #该表达在输入文本中的位置
    end: int

    @property
    def identity(self) -> tuple[str, str, str]:
        return self.metric, self.measure_kind, self.comparison_basis


_QUALIFIER = re.compile(
    r"(?:(?:同比|环比)(?:增长率|下降率|增长额|增加额|减少额|变动额|增速|增幅|降幅|增长|下降|增加|减少)?)"
    r"|增长率|下降率|增长额|增加额|减少额|变动额|增速|增幅|降幅|占比|比重|增长|下降|增加|减少"
)
_RATE_UNIT = re.compile(r"[+\-−]?\d+(?:\.\d+)?\s*[%％]")
_MONEY_UNIT = re.compile(r"\d+(?:\.\d+)?\s*(?:亿|万|百万)?元")
_CLAUSE_BOUNDARY = re.compile(r"[，,；;。！？?\n]")


def metric_dimension(metric: str) -> str:
    return domain_policy().get("financial_metrics", {}).get(metric, {}).get("value_dimension", "unknown")


def metric_unit_compatible(metric: str, measure_kind: str, unit: str | None) -> bool:
    """查询核验和计算共用单位约束；增长率缺少百分比单位时不猜数值尺度。"""
    if measure_kind not in {"level", "growth_rate", "absolute_change", "share"}:
        return False
    dimension = metric_dimension(metric)
    if measure_kind in {"growth_rate", "share"}:
        return unit == "%"
    if dimension == "ratio":
        return unit == ("个百分点" if measure_kind == "absolute_change" else "%")
    if dimension == "money":
        return unit in {"人民币亿元", "人民币万元", "人民币元", "亿元", "百万元", "万元", "元", "美元", "港元"}
    if dimension == "money_per_share":
        return unit in {None, "元", "元/股"}
    if dimension == "multiple":
        return unit in {None, "倍"}
    return False


def metric_aliases(metric: str) -> tuple[str, ...]:
    return (metric, *domain_policy()["aliases"].get(metric, []))


def _normalized_with_offsets(text: str) -> tuple[str, list[int]]:
    characters, offsets = [], []
    for index, character in enumerate(text):
        for normalized in unicodedata.normalize("NFKC", character).casefold():
            if not normalized.isspace():
                characters.append(normalized)
                offsets.append(index)
    return "".join(characters), offsets


def _metric_matches(text: str) -> list[tuple[str, int, int]]:
    aliases = {}
    for metric in domain_policy().get("financial_metrics", {}):
        for alias in metric_aliases(metric):
            normalized, _ = _normalized_with_offsets(alias)
            aliases[normalized] = metric
    if not aliases:
        return []
    pattern = "|".join(re.escape(alias) for alias in sorted(aliases, key=lambda value: (-len(value), value)))
    normalized, offsets = _normalized_with_offsets(text)
    matches = []
    for match in re.finditer(pattern, normalized):
        start, end = offsets[match.start()], offsets[match.end() - 1] + 1
        # 英文简称不能匹配其他单词的子串；原文坐标保留 OCR 空格。
        if match[0].isascii() and (
            (start and text[start - 1].isascii() and text[start - 1].isalnum())
            or (end < len(text) and text[end].isascii() and text[end].isalnum())
        ):
            continue
        matches.append((aliases[match[0]], start, end))
    return matches


def _measure(qualifier: str, context: str) -> tuple[str, str]:
    basis = ("year_over_year" if "同比" in qualifier else
             "period_over_period" if "环比" in qualifier else "unknown")
    if re.search(r"占比|比重", qualifier):
        return "share", "not_applicable"
    if re.search(r"下降|减少|降幅", qualifier):
        # 降幅的正数不能直接当成增长率的正数；未建模方向转换时保持未知。
        return "unknown", basis
    if "额" in qualifier or "百分点" in context:
        return "absolute_change", basis
    if not re.search(r"率|速|幅", qualifier) and _MONEY_UNIT.search(context) and not _RATE_UNIT.search(context):
        return "absolute_change", basis
    return "growth_rate", basis


def _local_qualifier(text: str, start: int, end: int, following: int) -> tuple[str, str, int]:
    suffix = text[end:following]
    boundary = _CLAUSE_BOUNDARY.search(suffix)
    suffix = suffix[:boundary.start()] if boundary else suffix
    label_suffix = suffix.lstrip(" \t的（(/／")
    match = _QUALIFIER.match(label_suffix)
    if match:
        qualifier_end = end + len(suffix) - len(label_suffix) + match.end()
        if text[qualifier_end:qualifier_end + 1] in {"）", ")"}:
            qualifier_end += 1
        return match[0], suffix, qualifier_end
    # 允许“同比营业收入”，不从另一指标或前一分句借用限定。
    prefix = text[:start]
    before = re.search(r"(同比|环比)\s*$", prefix)
    return (before[1], suffix, end) if before else ("", suffix, end)


def parse_metric_mentions(text: str) -> list[MetricMention]:
    matches = _metric_matches(text)
    output = []
    for index, (metric, start, end) in enumerate(matches):
        following = matches[index + 1][1] if index + 1 < len(matches) else len(text)
        qualifier, context, label_end = _local_qualifier(text, start, end, following)
        kind, basis = _measure(qualifier, context) if qualifier else ("level", "not_applicable")
        output.append(MetricMention(metric, text[start:label_end], kind, basis, start, label_end))
    return output


def table_metric_mentions(header: str, row_label: str) -> list[MetricMention]:
    """指标优先取数值列，其次取行名；度量限定可以来自另一侧表头。"""
    header_metrics = parse_metric_mentions(header)
    mentions = header_metrics or parse_metric_mentions(row_label)
    if len(mentions) != 1:
        return mentions
    other = row_label if header_metrics else header
    qualifier = _QUALIFIER.search(other)
    if not qualifier:
        return mentions
    kind, basis = _measure(qualifier[0], other)
    mention = mentions[0]
    if mention.measure_kind != "level" and (
        mention.measure_kind != kind
        or (mention.comparison_basis not in {"unknown", "not_applicable"} and basis != "unknown"
            and mention.comparison_basis != basis)
    ):
        return []  # 行列限定冲突，不能擅自选择其中一个。
    return [replace(mention, measure_kind=kind, comparison_basis=(
        mention.comparison_basis if basis == "unknown" and mention.measure_kind != "level" else basis))]
