"""确定性的语义基础规则，不将不确定信息视为证明。

登记表只提供精确等价的别名，不包含仅有关联的概念。
开放式实体识别由带原文约束的查询契约处理。
"""

from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def domain_policy() -> dict:
    return json.loads((Path(__file__).parent / "conf/query_domain.json").read_text())


def canonical_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = text.replace("−", "-").replace("－", "-").replace("—", "-")
    text = re.sub(r"((?:19|20)\d{2})\s*年?\s*(?:至|到|-)\s*((?:19|20)\d{2})", r"\1-\2", text)
    # 可省略的“年”只是日期写法差异，不是新增日期；
    # 保留月、季度和预测标记，避免将不同报告期合并。
    text = re.sub(r"((?:19|20)\d{2})\s*年", r"\1", text)
    text = re.sub(r"((?:19|20)\d{2})\s*h1", r"\1上半年", text)
    text = re.sub(r"((?:19|20)\d{2})\s*h2", r"\1下半年", text)
    for number, han in enumerate("一二三四", 1):
        text = re.sub(rf"第?\s*(?:{han}|{number})\s*季度|q{number}(?!\d)", f"第{han}季度", text)
    replacements = [(alias.casefold(), canonical.casefold())
                    for canonical, aliases in domain_policy()["aliases"].items()
                    for alias in aliases]
    replacements.extend([("本年度", "今年"), ("上年度", "去年"),
                         ("这家公司", "该公司"), ("本公司", "该公司")])
    for alias, canonical in sorted(replacements, key=lambda item: -len(item[0])):
        pattern = re.escape(alias)
        if alias.isascii():
            pattern = rf"(?<![a-z]){pattern}(?![a-z])"
        text = re.sub(pattern, lambda _match: canonical, text)
    return "".join(c for c in text if c.isalnum() or c in "+-%./<>=")


# 查询语言的等价表达与指标别名单独处理；
# “赚多少钱”不能证明用户要求某一种会计口径的利润。
QUERY_CONCEPTS = {
    "盈利": ("赚多少钱", "能赚多少", "盈利"),
    "估值": ("估值贵不贵", "估值水平", "估值"),
    "上网电价": ("卖电均价", "上网电价"),
}


def constraint_text(text: str) -> str:
    text = re.sub(r"((?:19|20)\d{2})E(?![A-Za-z0-9])", r"\1预测", text, flags=re.I)
    for concept, aliases in QUERY_CONCEPTS.items():
        for alias in sorted(aliases, key=len, reverse=True):
            text = text.replace(alias, concept)
    return canonical_text(text)


def reporting_periods(text: str) -> set[str]:
    """仅在分句包含唯一明确年份时，补全其中省略的年份。

    年份列表仍按列表处理，范围不能缩减为其中一个端点。
    这是期间绑定规则，不是模糊日期相似度判断。
    """
    normalized = constraint_text(text)
    pattern = re.compile(r"((?:19|20)\d{2})(预测)?(\d{1,2}月(?:\d{1,2}日)?|上半年|下半年|第[一二三四]季度|全年|年度)?")
    matches = list(pattern.finditer(normalized))
    periods = {m.group() for m in matches}
    years = {m[1] for m in matches}
    if "预测" in normalized and "实际" not in normalized and not any(m[2] for m in matches):
        periods = {m[1] + "预测" + (m[3] or "") for m in matches}
    if len(years) == 1:
        year = next(iter(years))
        forecast = any(m[2] for m in matches) or ("预测" in normalized and "实际" not in normalized)
        prefix = year + ("预测" if forecast else "")
        partials = re.findall(r"上半年|下半年|第[一二三四]季度|全年|年度", normalized)
        periods.update(prefix + period for period in partials)
        if partials:
            periods.discard(year)
            periods.discard(year + "预测")
    return periods


def metric_terms(text: str) -> list[str]:
    canonical = canonical_text(text)
    pattern = "|".join(re.escape(item) for item in sorted(
        domain_policy()["metrics"], key=len, reverse=True,
    ))
    return list(dict.fromkeys(re.findall(pattern, canonical)))


def explicit_named_terms(text: str) -> list[str]:
    """对组织名称和引号锚点进行范围有限、精度优先的补充。"""
    terms = []
    # 保留常见人物与地点问题的两个关系参数；
    # 这里只做有限保护，不是通用中文命名实体识别。
    for match in re.finditer(
        r"(?:^|[，。；！？\s])([\u4e00-\u9fffA-Za-z·]{2,12})在"
        r"([\u4e00-\u9fffA-Za-z·]{2,12}?)(?:负责|工作|任职|担任|居住)", text,
    ):
        terms.extend(match.groups())
    terms.extend(match.group(0) for match in re.finditer(
        r"(?:项目|产品|设备|型号)\s*[A-Za-z][A-Za-z0-9_-]*", text,
    ))
    return list(dict.fromkeys(terms))


_OPERATORS = {
    "不高于": "<=", "不超过": "<=", "至多": "<=", "小于等于": "<=",
    "不低于": ">=", "不少于": ">=", "至少": ">=", "大于等于": ">=",
    "大于": ">", "高于": ">", "超过": ">",
    "小于": "<", "低于": "<", "等于": "=",
}
_OP_PATTERN = re.compile("|".join(map(re.escape, sorted(_OPERATORS, key=len, reverse=True))))


def semantic_conflicts(original: str, candidate: str) -> list[str]:
    conflicts = []
    original_ops = [_OPERATORS[m.group()] for m in _OP_PATTERN.finditer(original)]
    candidate_ops = [_OPERATORS[m.group()] for m in _OP_PATTERN.finditer(candidate)]
    original_ops += re.findall(r"[<>]=?|(?<![<>])=", original)
    candidate_ops += re.findall(r"[<>]=?|(?<![<>])=", candidate)
    if original_ops != candidate_ops:
        conflicts.append("comparison_operator_changed")
    # 反转关系仍可能保留所有词语，因此“比”关系必须保留参数顺序；
    # 按来源分句校验还会保护拆分计划中的局部绑定。
    if re.search(r"(?<![同环])比(?![较例重率])", original):
        def operands(text: str):
            return [canonical_text(part) for part in re.split(
                r"比", re.sub(r"同比|环比|比例|比较", "", text), maxsplit=1,
            )]
        years = re.findall(r"(?:19|20)\d{2}", original)
        if len(set(years)) >= 2:
            candidate_order = re.findall(r"(?:19|20)\d{2}", candidate)
            if years != candidate_order:
                conflicts.append("comparison_arguments_changed")
        left = operands(original)[0]
        # 保留比较两侧各自的明确主体。
        from service.core.query_expansion import _extract_organizations
        subjects = _extract_organizations(original)
        first = [s for s in subjects if canonical_text(s) in left]
        second = [s for s in subjects if canonical_text(s) not in left]
        if len(first) == len(second) == 1:
            normalized = canonical_text(candidate)
            if normalized.find(canonical_text(first[0])) > normalized.find(canonical_text(second[0])):
                conflicts.append("comparison_subjects_changed")
    # 正负方向和值类型都属于硬约束，正向请求也需检查。
    for term in ("不含", "不包括", "不考虑", "实际", "预测", "累计", "新增",
                 "单季度", "全年", "扣非"):
        if term in original and term not in candidate:
            if term == "预测" and re.search(r"(?:19|20)\d{2}E(?![A-Za-z0-9])", candidate, re.I):
                continue
            conflicts.append(f"constraint_dropped:{term}")
    if re.search(r"(?:19|20)\d{2}E(?![A-Za-z0-9])", original, re.I) and not (
        re.search(r"(?:19|20)\d{2}E(?![A-Za-z0-9])", candidate, re.I) or "预测" in candidate
    ):
        conflicts.append("forecast_basis_dropped")
    return conflicts


def alias_expansions(text: str, maximum: int = 8) -> list[str]:
    terms = []
    normalized = canonical_text(text)
    for canonical, aliases in domain_policy()["aliases"].items():
        if canonical_text(canonical) in normalized:
            # 相关但含义不同的指标不能作为同义词扩展。
            if canonical == "归母净利润" and "扣非归母净利润" in normalized:
                continue
            for alias in [canonical, *aliases]:
                if alias.casefold() not in text.casefold():
                    terms.append(alias)
    return list(dict.fromkeys(terms))[:maximum]
