"""理解输出的财务覆盖检查；只报告遗漏，不在核验阶段补造用户需求。"""

from __future__ import annotations

import re
from dataclasses import replace

from service.core.finance.metrics import MetricMention, parse_metric_mentions
from service.core.query.semantics import reporting_periods


_ANNUAL_RANGE = re.compile(
    r"(?P<start>(?:19|20)\d{2})(?P<left>[AEae]?)年?\s*(?:至|到|[—–~～-])\s*"
    r"(?P<end>(?:19|20)\d{2})(?P<right>[AEae]?)年?(?![\d月日季度])"
)
_NUMERIC_REQUEST = re.compile(r"多少|数值|数据|列出|分别|比较|对比|计算|增幅|降幅|\d+(?:\.\d+)?\s*(?:亿?元|万元|%|％)")
_FINANCIAL_NOUN = re.compile(
    r"[\u4e00-\u9fffA-Za-z]{0,24}?(?:收入|利润|费用|成本|资产|负债|账款|存货|"
    r"准备金|资本充足率|收益率|回报率|周转率|财务指标|现金流|净额|余额|金额)"
)


def is_qualitative_financial_question(text: str) -> bool:
    return bool(re.search(r"原因|为什么|为何", text) and not _NUMERIC_REQUEST.search(text))


def requirement_period_labels(text: str) -> set[str]:
    """只展开明确的连续年度，保留 A/E 标记；不猜季度、相对年份或过长范围。"""
    def expand(match: re.Match) -> str:
        start, end = int(match['start']), int(match['end'])
        left, right = match['left'].upper(), match['right'].upper()
        if not 0 <= end - start <= 20 or (left and right and left != right):
            return match[0]
        marker = left or right
        return "、".join(f"{year}{marker}年" for year in range(start, end + 1))

    return reporting_periods(_ANNUAL_RANGE.sub(expand, text))


def has_unresolved_annual_range(text: str) -> bool:
    for match in _ANNUAL_RANGE.finditer(text):
        if not 0 <= int(match['end']) - int(match['start']) <= 20:
            return True
        if match['left'] and match['right'] and match['left'].upper() != match['right'].upper():
            return True
    return False


def _unrecognized_financial_labels(text: str, registry) -> list[str]:
    # 用原文坐标遮掉已识别部分，防止一个已知指标掩盖同句中的未知目标。
    masked = list(text)
    spans = [(item.start, item.end) for item in parse_metric_mentions(text)]
    spans.extend((item['start'], item['end']) for item in registry.mentions(text))
    for start, end in spans:
        masked[start:end] = " " * (end - start)
    remaining = "".join(masked)
    remaining = re.sub(r"(?:19|20)\d{2}(?:[AEae]|年)?", " ", remaining)
    remaining = re.sub(r"(?<= )的?(?:余额|金额)(?=\s|是|为|多|[？?]|$)", " ", remaining)
    remaining = re.sub(r"请问|查询|请|列出|分别|比较|对比|公司的|公司|年度|预测|实际|合并口径|母公司口径", " ", remaining)
    return list(dict.fromkeys(match[0].strip("的与和及") for match in _FINANCIAL_NOUN.finditer(remaining)))


def unresolved_financial_targets(question: str, goals, registry) -> list[dict[str, str]]:
    parts = [(goal.id, goal.text + " " + " ".join(goal.constraints)) for goal in goals]
    # 用户原问永远保留：规划模型省略的目标也必须留下缺口。
    parts.extend(("financial_backstop", part) for part in re.split(r"[，,；;。]", question) if part.strip())
    output, seen = [], set()
    for goal_id, text in parts:
        if is_qualitative_financial_question(text) or not (_NUMERIC_REQUEST.search(text) or reporting_periods(text)):
            continue
        for label in _unrecognized_financial_labels(text, registry):
            if not label or label in seen:
                continue
            seen.add(label)
            output.append({"goal_id": goal_id, "source_text": text,
                           "metric_label": label, "reason": "metric_unresolved"})
    return output


def requires_calculation(text: str) -> bool:
    """披露增长率的取值不等于计算；跨期差值和明确计算指令另行处理。"""
    if re.search(r"派息率|股利支付率|分红支付率|相差|差额|高多少|低多少|增加多少|减少多少|下降多少", text):
        return True
    if not re.search(r"增长率|增幅|降幅", text):
        return False
    if re.search(r"计算|算出|算一下|根据.+(?:收入|利润|现金流)", text):
        return True
    return len(reporting_periods(text)) == 2 and not re.search(
        r"分别|各自|各年|各期|趋势|增长率.*(?:相差|差额)|至|到|[—–~～-]", text,
    )


def requirement_metric_mentions(text: str, *, calculation_text: str | None = None) -> list[MetricMention]:
    mentions = parse_metric_mentions(text)
    request = calculation_text or text
    if requires_calculation(request) and not re.search(r"相差|差额|高多少|低多少", request):
        # 派生增长率使用基础指标值作为操作数；普通增长率取值保持原度量。
        mentions = [replace(mention, measure_kind="level", comparison_basis="not_applicable") for mention in mentions]
    # 目标文本和 constraints 可能重复提到同一指标，不应因此判为多个指标。
    return list({mention.identity: mention for mention in mentions}.values())
