"""对检索的硬约束执行确定性兼容检查。

语义重排负责判断主题相关性；此模块只判断候选中明确陈述的约束
是否与用户问题兼容。候选元数据缺失时保持中性，
仅将明确、可比较的不一致判为冲突。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


CompatibilityLevel = Literal["compatible", "unknown", "conflict"]

_CHINESE_DIGITS = {
    "一": "1",
    "二": "2",
    "三": "3",
    "四": "4",
}

_YEAR = r"(?:19|20)\d{2}"
_SHORT_OR_LONG_YEAR = rf"(?:{_YEAR}|\d{{2}})"

_CHINESE_QUARTER_WITH_YEAR = re.compile(
    rf"(?P<year>{_YEAR})年?(?:第)?(?P<quarter>[一二三四1-4])季度"
)
_Q_STYLE_WITH_YEAR = re.compile(
    rf"(?<!\d)(?P<year>{_SHORT_OR_LONG_YEAR})年?Q(?P<quarter>[1-4])",
    re.IGNORECASE,
)
_Q_STYLE_YEAR_LAST = re.compile(
    rf"Q(?P<quarter>[1-4])(?P<year>{_SHORT_OR_LONG_YEAR})(?!\d)",
    re.IGNORECASE,
)
_HALF_YEAR_CHINESE = re.compile(
    rf"(?P<year>{_YEAR})年?(?P<half>上半年|下半年)"
)
_HALF_YEAR_STYLE = re.compile(
    rf"(?<!\d)(?P<year>{_SHORT_OR_LONG_YEAR})年?H(?P<half>[12])",
    re.IGNORECASE,
)
_MONTH_WITH_YEAR = re.compile(
    rf"(?P<year>{_YEAR})年(?P<month>1[0-2]|0?[1-9])月"
)
_FULL_YEAR = re.compile(
    rf"(?P<year>{_YEAR})年?(?:全年|全年度|年度)"
)
_FORECAST_YEAR = re.compile(
    rf"(?<!\d)(?P<year>{_SHORT_OR_LONG_YEAR})E(?![A-Za-z])",
    re.IGNORECASE,
)
_YEAR_PATTERN = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
_STANDALONE_QUARTER = re.compile(
    r"(?:第)?(?P<quarter>[一二三四1-4])季度|Q(?P<q>[1-4])",
    re.IGNORECASE,
)
_STANDALONE_HALF = re.compile(
    r"(?P<half>上半年|下半年)|H(?P<h>[12])",
    re.IGNORECASE,
)

_LOCATOR_PATTERN = re.compile(
    r"(?:表|图|附表|附图)\s*[A-Za-z]?\s*\d+(?:\s*[-－—.]\s*\d+)*"
    r"|第\s*[一二三四五六七八九十百零〇两\d]+\s*(?:章|节|页)",
    re.IGNORECASE,
)

_ORGANIZATION_PATTERN = re.compile(
    r"[\u4e00-\u9fffA-Za-z0-9·]{2,24}"
    r"(?:股份有限公司|有限责任公司|集团有限公司|股份|集团|电力|能源|"
    r"银行|证券|科技|公司)"
)
_GENERIC_ORGANIZATIONS = {
    "上市公司",
    "该公司",
    "本公司",
    "公司",
    "电力",
    "能源",
    "银行",
    "证券",
}
_ENTITY_PREFIXES = (
    "研报觉得",
    "研报认为",
    "报告觉得",
    "报告认为",
    "帮我查询",
    "帮我查",
    "想知道",
    "请问",
    "关于",
    "根据",
    "预计",
)
_LEADING_ENTITY_TIME_PATTERN = re.compile(
    rf"^(?:{_YEAR}年?)?(?:上半年|下半年|全年|年度|"
    r"(?:第)?[一二三四1-4]季度)?"
)

_QUERY_ACTUAL_PATTERN = re.compile(
    r"实际(?:值|数据|业绩|实现)?|已实现|截至|报告期"
)
_CANDIDATE_ACTUAL_PATTERN = re.compile(
    r"实际(?:值|数据|业绩|实现)?|已实现|截至|报告期|"
    r"公司实现(?:营业收入|收入|归母净利润|扣非归母净利润|利润)"
)
_FORECAST_PATTERN = re.compile(
    rf"预测|预计|预期|盈利预测|(?<!\d){_SHORT_OR_LONG_YEAR}E(?![A-Za-z])",
    re.IGNORECASE,
)
_TARGET_PATTERN = re.compile(r"目标|规划目标|计划达到|力争")

# 这里只比较通用计量维度。业务指标识别交由语义检索处理，
# 避免枚举营收、现金流或装机容量等名称而过度适配个别语料。
_MEASUREMENT_DIMENSION_PATTERNS = {
    "rate": re.compile(r"率|比例|占比|比重|百分比|百分点|[%％]"),
    "currency_amount": re.compile(
        r"金额|人民币|百万元|千万元|亿元|万元|千元|"
        r"(?<![万亿千百])元(?!\s*[/／]\s*股)"
    ),
    "per_share": re.compile(
        r"每股|(?:元|角|分)\s*[/／]\s*股|"
        r"(?<![A-Za-z])EPS(?![A-Za-z])",
        re.IGNORECASE,
    ),
    "power": re.compile(
        r"(?:千瓦|兆瓦|吉瓦)(?!时)|"
        r"(?<![A-Za-z])(?:kW|MW|GW)(?![A-Za-z])",
        re.IGNORECASE,
    ),
    "energy": re.compile(
        r"千瓦时|兆瓦时|吉瓦时|"
        r"(?<![A-Za-z])(?:kWh|MWh|GWh)(?![A-Za-z])",
        re.IGNORECASE,
    ),
    "count": re.compile(
        r"数量|\d+(?:\.\d+)?\s*(?:台|套|座|家|户|人)(?![\u4e00-\u9fff])"
    ),
    "multiple": re.compile(r"\d+(?:\.\d+)?\s*倍"),
}

_MEASUREMENT_SUBJECT_NOISE_PATTERN = re.compile(
    rf"(?:{_YEAR})e?|\d+(?:\.\d+)?|"
    r"上半年|下半年|全年|年度|季度|月份?|年|"
    r"第一|第二|第三|第四|"
    r"是多少|多少|什么|哪个|哪一个|分别|具体|数据|指标|"
    r"公司|报告|研报|预计|预测|实际|实现|达到|约|左右|"
    r"金额|人民币|百万元|千万元|亿元|万元|千元|"
    r"百分比|百分点|比例|占比|比重|率|[%％]|"
    r"每股|元股|"
    r"千瓦时|兆瓦时|吉瓦时|kwh|mwh|gwh|"
    r"千瓦|兆瓦|吉瓦|kw|mw|gw|"
    r"更高|相差|口径|单位",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ConstraintCategoryDecision:
    """一类可比较约束的兼容性结果。"""

    level: CompatibilityLevel
    required: list[str]
    observed: list[str]
    matched: list[str]


@dataclass(frozen=True)
class ConstraintCompatibilityDecision:
    """最终检索排序使用的结构化诊断信息。"""

    categories: dict[str, ConstraintCategoryDecision]
    compatible_count: int
    conflict_count: int
    unknown_count: int

    def to_dict(self) -> dict:
        return asdict(self)


def _normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return "".join(
        character
        for character in normalized
        if character.isalnum() or "\u4e00" <= character <= "\u9fff"
    )


def _expanded_year(value: str) -> str:
    year = int(value)
    if year < 100:
        year += 2000 if year <= 69 else 1900
    return str(year)


def _quarter_number(value: str) -> str:
    return _CHINESE_DIGITS.get(value, value)


def extract_time_scopes(text: str) -> set[str]:
    """提取带明确范围的规范报告期，例如 2025-Q2。"""
    normalized = unicodedata.normalize("NFKC", str(text or ""))
    compact = re.sub(r"\s+", "", normalized)
    scopes: set[str] = set()

    for match in _CHINESE_QUARTER_WITH_YEAR.finditer(compact):
        scopes.add(
            f"{match.group('year')}-Q"
            f"{_quarter_number(match.group('quarter'))}"
        )
    for pattern in (_Q_STYLE_WITH_YEAR, _Q_STYLE_YEAR_LAST):
        for match in pattern.finditer(compact):
            scopes.add(
                f"{_expanded_year(match.group('year'))}-Q"
                f"{match.group('quarter')}"
            )
    for match in _HALF_YEAR_CHINESE.finditer(compact):
        half = "H1" if match.group("half") == "上半年" else "H2"
        scopes.add(f"{match.group('year')}-{half}")
    for match in _HALF_YEAR_STYLE.finditer(compact):
        scopes.add(
            f"{_expanded_year(match.group('year'))}-H{match.group('half')}"
        )
    for match in _MONTH_WITH_YEAR.finditer(compact):
        scopes.add(
            f"{match.group('year')}-M{int(match.group('month')):02d}"
        )
    for match in _FULL_YEAR.finditer(compact):
        scopes.add(f"{match.group('year')}-FY")
    for match in _FORECAST_YEAR.finditer(compact):
        scopes.add(f"{_expanded_year(match.group('year'))}-FY")

    # 支持年份与报告期之间夹有少量词语的表达，
    # 但仅在全文包含唯一、无歧义年份时建立绑定。
    years = set(_YEAR_PATTERN.findall(normalized))
    if len(years) == 1:
        year = next(iter(years))
        for match in _STANDALONE_QUARTER.finditer(compact):
            quarter = match.group("quarter") or match.group("q")
            scopes.add(f"{year}-Q{_quarter_number(quarter)}")
        for match in _STANDALONE_HALF.finditer(compact):
            half = match.group("half")
            half_number = (
                "1" if half == "上半年" else "2" if half else match.group("h")
            )
            scopes.add(f"{year}-H{half_number}")
        if not scopes:
            scopes.add(f"{year}-YEAR")
    elif not years and not scopes:
        for match in _STANDALONE_QUARTER.finditer(compact):
            quarter = match.group("quarter") or match.group("q")
            scopes.add(f"ANY-Q{_quarter_number(quarter)}")
        for match in _STANDALONE_HALF.finditer(compact):
            half = match.group("half")
            half_number = (
                "1" if half == "上半年" else "2" if half else match.group("h")
            )
            scopes.add(f"ANY-H{half_number}")
    return scopes


def extract_locators(text: str) -> set[str]:
    return {
        _normalize(match.group(0))
        for match in _LOCATOR_PATTERN.finditer(str(text or ""))
        if match.group(0).strip()
    }


def extract_fact_statuses(
    text: str,
    *,
    for_query: bool = False,
) -> set[str]:
    statuses = set()
    value = str(text or "")
    actual_pattern = (
        _QUERY_ACTUAL_PATTERN if for_query else _CANDIDATE_ACTUAL_PATTERN
    )
    if actual_pattern.search(value):
        statuses.add("actual")
    if _FORECAST_PATTERN.search(value):
        statuses.add("forecast")
    if _TARGET_PATTERN.search(value):
        statuses.add("target")
    return statuses


def extract_measurement_dimensions(text: str) -> set[str]:
    """提取可用于判断明确不一致的通用数值维度。"""
    value = unicodedata.normalize("NFKC", str(text or ""))
    return {
        dimension
        for dimension, pattern in _MEASUREMENT_DIMENSION_PATTERNS.items()
        if pattern.search(value)
    }


def extract_organizations(text: str) -> set[str]:
    # 规划与排序必须使用一致的实体边界；
    # “解释”“查询”等叙述前缀不属于公司名称。
    from service.core.query.preservation import _extract_organizations
    return {_normalize(value) for value in _extract_organizations(str(text or ""))}


def _measurement_subject_text(text: str, *, entity_names=None) -> str:
    """移除通用约束，保留问题中特有的主体文本。"""
    subject = _normalize(text)
    organizations = extract_organizations(text) if entity_names is None else {_normalize(name) for name in entity_names}
    for organization in organizations:
        subject = subject.replace(organization, "")
    return _MEASUREMENT_SUBJECT_NOISE_PATTERN.sub("", subject)


def _longest_common_measurement_subject(
    required: str,
    observed: str,
) -> str:
    """不依赖指标词表，保守判断主体的字面匹配。"""
    maximum_length = min(len(required), len(observed), 16)
    for size in range(maximum_length, 1, -1):
        for start in range(len(required) - size + 1):
            candidate = required[start:start + size]
            if candidate in observed:
                return candidate
    return ""


def _compare_measurement_subjects(
    question: str,
    candidate_text: str,
    *, entity_names=None,
) -> ConstraintCategoryDecision | None:
    """对包含计量维度的问题返回软性主体字面匹配结果。

    没有匹配时返回 unknown，而非 conflict，
    使别名和改写表达仍有机会进入语义重排。
    """
    if not extract_measurement_dimensions(question):
        return None
    required = _measurement_subject_text(question, entity_names=entity_names)
    if len(required) < 2:
        return None
    observed = _measurement_subject_text(candidate_text)
    matched = _longest_common_measurement_subject(required, observed)
    return ConstraintCategoryDecision(
        level="compatible" if matched else "unknown",
        required=[required],
        observed=[observed] if observed else [],
        matched=[matched] if matched else [],
    )


def _compare_sets(
    required: set[str],
    observed: set[str],
    *,
    explicit_mismatch_is_conflict: bool = True,
) -> ConstraintCategoryDecision | None:
    if not required:
        return None
    matched = required & observed
    if matched:
        level: CompatibilityLevel = "compatible"
    elif observed and explicit_mismatch_is_conflict:
        level = "conflict"
    else:
        level = "unknown"
    return ConstraintCategoryDecision(
        level=level,
        required=sorted(required),
        observed=sorted(observed),
        matched=sorted(matched),
    )


def _compare_time_scopes(
    required: set[str],
    observed: set[str],
) -> ConstraintCategoryDecision | None:
    if not required:
        return None

    matched_required = set()
    for required_scope in required:
        required_year, required_period = required_scope.split("-", 1)
        for observed_scope in observed:
            observed_year, observed_period = observed_scope.split("-", 1)
            same_year = (
                required_year == observed_year
                or required_year == "ANY"
                or observed_year == "ANY"
            )
            if not same_year:
                continue
            if (
                required_period == observed_period
                or required_period == "YEAR"
            ):
                matched_required.add(required_scope)
                break

    if matched_required:
        level: CompatibilityLevel = "compatible"
    elif not observed:
        level = "unknown"
    else:
        # 候选仅出现年份，不能证明其事实属于所要求的季度或半年，
        # 因此保持中性，不将其视为已匹配。
        required_years = {scope.split("-", 1)[0] for scope in required}
        observed_bare_years = {
            scope.split("-", 1)[0]
            for scope in observed
            if scope.endswith("-YEAR")
        }
        if required_years & observed_bare_years:
            level = "unknown"
        else:
            level = "conflict"

    return ConstraintCategoryDecision(
        level=level,
        required=sorted(required),
        observed=sorted(observed),
        matched=sorted(matched_required),
    )


def evaluate_constraint_compatibility(
    question: str,
    candidate_text: str,
    *,
    document_name: str = "",
    entity_names=None,
) -> ConstraintCompatibilityDecision:
    """比较高置信度查询约束与单条候选片段。"""
    categories: dict[str, ConstraintCategoryDecision] = {}

    comparisons = {
        "time": _compare_time_scopes(
            extract_time_scopes(question),
            extract_time_scopes(candidate_text),
        ),
        "fact_status": _compare_sets(
            extract_fact_statuses(question, for_query=True),
            extract_fact_statuses(candidate_text),
        ),
        "locator": _compare_sets(
            extract_locators(question),
            extract_locators(candidate_text),
        ),
        "measurement_dimension": _compare_sets(
            extract_measurement_dimensions(question),
            extract_measurement_dimensions(candidate_text),
        ),
        "measurement_subject": _compare_measurement_subjects(
            question,
            candidate_text,
            entity_names=entity_names,
        ),
    }
    for category, decision in comparisons.items():
        if decision is not None:
            categories[category] = decision

    # 正文片段可能从文档继承实体，缺少名称不构成明确冲突。
    # 文档元数据或正文中的实体匹配可作为诊断信息和排序同分时的依据。
    required_entities = extract_organizations(question) if entity_names is None else {_normalize(name) for name in entity_names}
    if required_entities:
        document_stem = Path(str(document_name or "")).stem
        candidate_entities = extract_organizations(candidate_text)
        candidate_haystack = _normalize(
            f"{document_stem} {candidate_text}"
        )
        matched_entities = {
            entity
            for entity in required_entities
            if entity in candidate_haystack
        }
        entity_level: CompatibilityLevel = (
            "compatible" if matched_entities else "unknown"
        )
        categories["entity"] = ConstraintCategoryDecision(
            level=entity_level,
            required=sorted(required_entities),
            observed=sorted(candidate_entities),
            matched=sorted(matched_entities),
        )

    levels = [decision.level for decision in categories.values()]
    return ConstraintCompatibilityDecision(
        categories=categories,
        compatible_count=levels.count("compatible"),
        conflict_count=levels.count("conflict"),
        unknown_count=levels.count("unknown"),
    )
