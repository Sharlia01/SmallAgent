"""查询保真与实体锚点检查。仅使用确定性规则，不生成或扩展查询。"""
import re
import unicodedata
from dataclasses import dataclass, field
from service.core.query.semantics import canonical_text, explicit_named_terms, metric_terms, semantic_conflicts, domain_policy

LOCATOR_PATTERN = re.compile(
    r"(?:表|图|附表|附图)\s*[A-Za-z]?\s*\d+(?:\s*[-－—.]\s*\d+)*"
    r"|第\s*[一二三四五六七八九十百零〇两\d]+\s*(?:章|节|页)"
)
YEAR_OR_DATE_PATTERN = re.compile(
    r"(?<!\d)(?:19|20)\d{2}\s*年\s*\d{1,2}\s*月(?:\s*\d{1,2}\s*日)?"
    r"|(?<!\d)(?:19|20)\d{2}\s*年?\s*(?:第\s*)?[一二三四1-4]\s*季度"
    r"|(?<!\d)(?:19|20)\d{2}\s*年?\s*(?:上半年|下半年|全年|年度)"
    r"|(?<!\d)(?:19|20)\d{2}\s*年?\s*[-－—至到/]\s*(?:19|20)\d{2}\s*年?"
    r"|(?<!\d)(?:\d{2}|(?:19|20)\d{2})\s*H[12]"
    r"|(?:\d{2}|(?:19|20)\d{2})?\s*Q[1-4]"
    r"|(?<!\d)(?:19|20)\d{2}\s*年?(?!\d)",
    re.IGNORECASE,
)
RELATIVE_TIME_PATTERN = re.compile(
    r"(?:未来|过去|近)\s*[一二三四五六七八九十两\d]+\s*(?:年|个月|季度)"
    r"|今年|去年|明年|本年度|上年度|本季度|上季度|上半年|下半年"
    r"|(?:第\s*)?[一二三四1-4]\s*季度"
)
NUMBER_WITH_UNIT_PATTERN = re.compile(
    r"[-+]?\d+(?:\.\d+)?\s*(?:%|％|百万元|百万美元|亿美元|美元|亿元|万元|元|倍|万吨|万千瓦|"
    r"亿千瓦时|千瓦时|兆瓦时|GW|MW|kW|x)",
    re.IGNORECASE,
)
NUMBER_PATTERN = re.compile(r"(?<![A-Za-z\d])[-+]?\d+(?:\.\d+)?")
STOCK_CODE_PATTERN = re.compile(
    r"(?<!\d)\d{6}(?:\.(?:SH|SZ|BJ))?(?!\d)",
    re.IGNORECASE,
)
QUOTED_TERM_PATTERN = re.compile(r"[“‘\"']([^”’\"']{2,40})[”’\"']")
NEGATION_PATTERN = re.compile(
    r"没有|并未|未能|未曾|未发生|未提供|未披露|未包含|不能|不含|"
    r"不包括|(?<!是)不是|不高于|不低于|不超过|不少于|不支持|不考虑|"
    r"不计算|无法|无需"
)

ORGANIZATION_PATTERN = re.compile(
    r"[\u4e00-\u9fffA-Za-z0-9·]{1,24}?"
    r"(?:股份有限公司|有限责任公司|集团有限公司|股份|集团|电力|能源|"
    r"银行|证券|科技|公司)"
)
ENTITY_PREFIXES = (
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
    "觉得",
    "认为",
)
ENTITY_LEADING_CONTEXT_PATTERN = re.compile(
    r"^(?:(?:这份|该|本)?(?:研报|研究报告|报告)(?:中|里)?"
    r"(?:对|关于|认为|觉得)?|帮我查询|帮我查|想知道|请问|关于|根据|对|"
    r"关键财务与估值指标表|查询|查找|找出|获取|比较|说明|解释|分析|介绍|列出|预计|以及|请|先|再|和|与|"
    r"(?:哪些|什么|具体)?(?:数据|信息|证据)?(?:来|用于)?支持)+"
)
GENERIC_ORGANIZATIONS = {
    "上市公司",
    "该公司",
    "本公司",
    "公司",
    "电力",
    "能源",
    "银行",
    "证券",
    "推荐公司",
    "目标公司",
}

# 这些词已是规范检索术语；用户明确使用时应原样保留，
# 不能在扩展中替换为更宽泛的概念。
CANONICAL_TERM_PATTERN = re.compile(
    r"归母净利润|扣非归母净利润|营业收入|营业成本|每股收益|市盈率|"
    r"市净率|现金流|装机容量|发电量|上网电价|利用小时|煤价|分红|"
    r"股息率|估值|毛利率|净利率|资产负债率|同比|环比|预测|实际|"
    r"截至|分别",
    re.IGNORECASE,
)

@dataclass(frozen=True)
class QueryValidationResult:
    valid: bool
    required_anchors: list[str]
    missing_anchors: list[str]
    added_risky_terms: list[str]
    semantic_conflicts: list[str] = field(default_factory=list)

def _normalize_for_match(value: str) -> str:
    return canonical_text(value)

def _unique_matches(pattern: re.Pattern, query: str) -> set[str]:
    return {
        " ".join(match.group(0).split()).strip()
        for match in pattern.finditer(query)
        if match.group(0).strip()
    }

def _extract_organizations(query: str, entity_names=None) -> set[str]:
    # Agent 传入模型确认的名称后，只定位这些完整名称；绝不按后缀重新识别。
    # None 仅兼容未运行理解模型的旧调用方，空元组表示明确没有已知实体。
    if entity_names is not None:
        if not entity_names:
            return set()
        # 完整名称优先，不能把明确声明的子公司名称再次拆成母公司。
        pattern = "|".join(re.escape(name) for name in sorted(set(entity_names), key=lambda name: (-len(name), name)))
        return {match.group() for match in re.finditer(pattern, query)}

    # “归属于母公司股东的净利润”等财务字段不是新的公司实体。
    # 使用组织后缀识别前，先遮蔽已知字段名称。
    field_names = [name for canonical, aliases in domain_policy()["aliases"].items()
                   for name in [canonical, *aliases] if not name.isascii()]
    original = query
    for name in sorted(field_names, key=len, reverse=True):
        query = query.replace(name, " ")
    # 先遮蔽时间文本再匹配实体；对已匹配名称删除时间会拼造
    # 原文中不连续的名称，例如“2025年国电电力”。
    query = YEAR_OR_DATE_PATTERN.sub(" ", query)
    query = RELATIVE_TIME_PATTERN.sub(" ", query)
    organizations = set()
    for match in ORGANIZATION_PATTERN.finditer(query):
        candidate = match.group(0).strip()
        candidate = ENTITY_LEADING_CONTEXT_PATTERN.sub("", candidate).strip()
        if (
            candidate
            and candidate in original
            and candidate not in GENERIC_ORGANIZATIONS
            # “研报推荐的公司”是待检索的描述，不能绑定成一个已知公司名。
            and not candidate.endswith("的公司")
            and len(candidate) <= 20
        ):
            organizations.add(candidate)
    return organizations

def extract_query_anchors(query: str, *, entity_names=None) -> list[str]:
    """提取查询变换必须保留的高置信度锚点词。"""
    if not isinstance(query, str) or not query.strip():
        return []

    anchors = set()
    for pattern in (
        LOCATOR_PATTERN,
        YEAR_OR_DATE_PATTERN,
        RELATIVE_TIME_PATTERN,
        NUMBER_WITH_UNIT_PATTERN,
        NUMBER_PATTERN,
        STOCK_CODE_PATTERN,
        NEGATION_PATTERN,
        CANONICAL_TERM_PATTERN,
    ):
        anchors.update(_unique_matches(pattern, query))

    anchors.update(_extract_organizations(query, entity_names))
    anchors.update(explicit_named_terms(query))
    anchors.update(metric_terms(query))
    anchors.update(
        match.group(1).strip()
        for match in QUOTED_TERM_PATTERN.finditer(query)
        if match.group(1).strip()
    )
    return sorted(anchors, key=lambda value: (_normalize_for_match(value), value))

def _extract_risky_additions(query: str, *, entity_names=None) -> set[str]:
    query = unicodedata.normalize("NFKC", query).replace("−", "-").replace("—", "-")
    risky_terms = set()
    for pattern in (
        LOCATOR_PATTERN,
        YEAR_OR_DATE_PATTERN,
        NUMBER_WITH_UNIT_PATTERN,
        NUMBER_PATTERN,
        STOCK_CODE_PATTERN,
        NEGATION_PATTERN,
    ):
        risky_terms.update(_unique_matches(pattern, query))
    risky_terms.update(_extract_organizations(query, entity_names))
    risky_terms.update(explicit_named_terms(query))
    risky_terms.update(scope for scope in domain_policy().get("business_scopes", []) if scope in query)
    return risky_terms

def _normalized_difference(left: set[str], right: set[str]) -> list[str]:
    right_normalized = {_normalize_for_match(value) for value in right}
    return sorted(
        value
        for value in left
        if _normalize_for_match(value) not in right_normalized
    )

def validate_query_preservation(
    original: str,
    candidate: str,
    *, entity_names=None,
) -> QueryValidationResult:
    """拒绝丢失锚点或无依据增加精确条件的候选查询。"""
    required_anchors = extract_query_anchors(original, entity_names=entity_names)
    normalized_candidate = _normalize_for_match(candidate)
    missing_anchors = [
        anchor
        for anchor in required_anchors
        if _normalize_for_match(anchor) not in normalized_candidate
    ]

    original_risky_terms = _extract_risky_additions(original, entity_names=entity_names)
    candidate_risky_terms = _extract_risky_additions(candidate, entity_names=entity_names)
    added_risky_terms = _normalized_difference(
        candidate_risky_terms,
        original_risky_terms,
    )
    from service.core.query.semantics import reporting_periods
    source_periods = reporting_periods(original)
    added_risky_terms = [term for term in added_risky_terms if not (
        YEAR_OR_DATE_PATTERN.fullmatch(term.strip()) and not re.search(r"月|日", term)
        and reporting_periods(term) and reporting_periods(term) <= source_periods
    )]
    conflicts = semantic_conflicts(original, candidate, entity_names=entity_names)

    return QueryValidationResult(
        valid=bool(candidate.strip()) and not missing_anchors and not added_risky_terms and not conflicts,
        required_anchors=required_anchors,
        missing_anchors=missing_anchors,
        added_risky_terms=added_risky_terms,
        semantic_conflicts=conflicts,
    )
