"""扩展检索查询，并通过确定性检查保护原始意图。"""

from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any, Literal

from dotenv import load_dotenv
from openai import OpenAI
from service.core.query_runtime import model_completion
from service.core.query_semantics import (
    alias_expansions, canonical_text, explicit_named_terms, metric_terms,
    semantic_conflicts, domain_policy,
)


load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_QUERY_EXPANSION_MODEL = "qwen3.7-flash-2026-07-15"
DEFAULT_QUERY_EXPANSION_TIMEOUT_SECONDS = 8.0
DEFAULT_QUERY_EXPANSION_MAX_TERMS = 8
MAX_QUERY_EXPANSION_INPUT_CHARS = 2000
MAX_QUERY_EXPANSION_OUTPUT_CHARS = 400

QueryExpansionSource = Literal["model", "rule", "fallback", "disabled", "skipped"]

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
    r"[\u4e00-\u9fffA-Za-z0-9·]{2,24}?"
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
    r"查询|查找|找出|获取|比较|说明|解释|分析|介绍|列出|预计|以及|请|先|再|和|与|"
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

RATING_INTERPRETATION_CUE_PATTERN = re.compile(
    r"看多|看空|含义|意思|定义|标准|代表什么|意味着什么|如何理解|到底"
)
RATING_REQUIRED_EXPANSION_TERMS = (
    "投资评级",
    "评级定义",
    "评级标准",
)
TARGET_PRICE_PATTERN = re.compile(r"目标价(?:格)?")
TARGET_PRICE_INTENT_PATTERN = re.compile(
    r"目标价(?:格)?|合理价(?:格)?|估值|值多少钱|贵不贵|便不便宜"
)

QUERY_EXPANSION_SYSTEM_PROMPT = """
你是知识库 RAG 的查询扩展器。请根据原问题和规范化问题，生成一条用于关键词召回的扩展查询；不要回答问题。

要求：
1. 保留原问题中的公司、人名、证券代码、时间、数字、单位、否定词、表号、图号、章节、页码和所有限定条件。
2. 只补充原问题中口语表达对应的常见书面语、专业字段名、缩写和同义检索词。
3. 不得猜测或新增具体年份、日期、数字、实体、结论及事实。
4. 同时覆盖原问题中的每个子意图，不得只保留其中一部分。
5. 不得补充只是同属金融领域、但用户没有询问的字段；例如只问投资评级时，不得加入目标价或盈利预测。
6. 当用户询问评级代表看多还是看空、评级的含义或标准时，应补充“投资评级、评级定义、评级标准”等检索词。
7. added_terms 只列出相对规范化问题新增的检索词，最多 {max_terms} 个。
8. 只返回 JSON 对象，格式为：
{{"expanded_query":"扩展查询","added_terms":["新增词1","新增词2"]}}

示例：
输入：{{"original_query":"国电电力未来三年的每股收益和市盈率？","normalized_query":"国电电力未来三年的每股收益和市盈率"}}
输出：{{"expanded_query":"国电电力未来三年的每股收益和市盈率 EPS PE","added_terms":["EPS","PE"]}}

输入：{{"original_query":"这份研报对国电电力到底是看多还是看空？给了什么评级？","normalized_query":"国电电力 研报 投资评级 看多 看空"}}
输出：{{"expanded_query":"国电电力 研报 投资评级 看多 看空 评级定义 评级标准","added_terms":["评级定义","评级标准"]}}
""".strip()


@dataclass(frozen=True)
class QueryValidationResult:
    valid: bool
    required_anchors: list[str]
    missing_anchors: list[str]
    added_risky_terms: list[str]
    semantic_conflicts: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class QueryExpansionResult:
    """保存完整查询变换状态，供检索诊断使用。"""

    original: str
    rewritten: str
    expanded: str
    effective_query: str
    changed: bool
    expansion_applied: bool
    added_terms: list[str]
    source: QueryExpansionSource
    fallback_reason: str | None
    rewrite_validation: QueryValidationResult
    expansion_validation: QueryValidationResult

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["validation"] = {
            "rewritten": payload.pop("rewrite_validation"),
            "expanded": payload.pop("expansion_validation"),
        }
        return payload

    @classmethod
    def noop(
        cls,
        question: str,
        *,
        reason: str = "无需查询改写和扩展",
    ) -> "QueryExpansionResult":
        validation = validate_query_preservation(question, question)
        return cls(
            original=question,
            rewritten=question,
            expanded="",
            effective_query=question,
            changed=False,
            expansion_applied=False,
            added_terms=[],
            source="skipped",
            fallback_reason=reason,
            rewrite_validation=validation,
            expansion_validation=validation,
        )


def _environment_flag(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().casefold() not in {"0", "false", "no", "off"}


def _expansion_timeout_seconds() -> float:
    raw_value = os.getenv(
        "QUERY_EXPANSION_TIMEOUT_SECONDS",
        str(DEFAULT_QUERY_EXPANSION_TIMEOUT_SECONDS),
    )
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return DEFAULT_QUERY_EXPANSION_TIMEOUT_SECONDS
    return value if value > 0 else DEFAULT_QUERY_EXPANSION_TIMEOUT_SECONDS


def _max_expansion_terms() -> int:
    raw_value = os.getenv(
        "QUERY_EXPANSION_MAX_TERMS",
        str(DEFAULT_QUERY_EXPANSION_MAX_TERMS),
    )
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        return DEFAULT_QUERY_EXPANSION_MAX_TERMS
    return value if value > 0 else DEFAULT_QUERY_EXPANSION_MAX_TERMS


@lru_cache(maxsize=1)
def get_query_expansion_client() -> OpenAI:
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    base_url = os.getenv("DASHSCOPE_BASE_URL", "").strip()
    if not api_key or not base_url:
        raise RuntimeError(
            "DASHSCOPE_API_KEY and DASHSCOPE_BASE_URL are required for query expansion"
        )
    return OpenAI(api_key=api_key, base_url=base_url)


def _normalize_for_match(value: str) -> str:
    return canonical_text(value)


def _unique_matches(pattern: re.Pattern, query: str) -> set[str]:
    return {
        " ".join(match.group(0).split()).strip()
        for match in pattern.finditer(query)
        if match.group(0).strip()
    }


def _extract_organizations(query: str) -> set[str]:
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
            and len(candidate) <= 20
        ):
            organizations.add(candidate)
    return organizations


def extract_query_anchors(query: str) -> list[str]:
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

    anchors.update(_extract_organizations(query))
    anchors.update(explicit_named_terms(query))
    anchors.update(metric_terms(query))
    anchors.update(
        match.group(1).strip()
        for match in QUOTED_TERM_PATTERN.finditer(query)
        if match.group(1).strip()
    )
    return sorted(anchors, key=lambda value: (_normalize_for_match(value), value))


def _extract_risky_additions(query: str) -> set[str]:
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
    risky_terms.update(_extract_organizations(query))
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
) -> QueryValidationResult:
    """拒绝丢失锚点或无依据增加精确条件的候选查询。"""
    required_anchors = extract_query_anchors(original)
    normalized_candidate = _normalize_for_match(candidate)
    missing_anchors = [
        anchor
        for anchor in required_anchors
        if _normalize_for_match(anchor) not in normalized_candidate
    ]

    original_risky_terms = _extract_risky_additions(original)
    candidate_risky_terms = _extract_risky_additions(candidate)
    added_risky_terms = _normalized_difference(
        candidate_risky_terms,
        original_risky_terms,
    )
    from service.core.query_semantics import reporting_periods
    source_periods = reporting_periods(original)
    added_risky_terms = [term for term in added_risky_terms if not (
        YEAR_OR_DATE_PATTERN.fullmatch(term.strip()) and not re.search(r"月|日", term)
        and reporting_periods(term) and reporting_periods(term) <= source_periods
    )]
    conflicts = semantic_conflicts(original, candidate)

    return QueryValidationResult(
        valid=bool(candidate.strip()) and not missing_anchors and not added_risky_terms and not conflicts,
        required_anchors=required_anchors,
        missing_anchors=missing_anchors,
        added_risky_terms=added_risky_terms,
        semantic_conflicts=conflicts,
    )


def _response_content(completion: Any) -> str:
    choices = getattr(completion, "choices", None) or []
    if not choices:
        raise ValueError("query expansion model returned no choices")
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise ValueError("query expansion model returned empty content")
    return content.strip()


def _parse_expansion(content: str) -> tuple[str, list[str]]:
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise ValueError("query expansion response must be a JSON object")

    expanded_query = payload.get("expanded_query")
    added_terms = payload.get("added_terms")
    if not isinstance(expanded_query, str):
        raise ValueError("query expansion response must contain expanded_query")
    if not isinstance(added_terms, list) or not all(
        isinstance(term, str) for term in added_terms
    ):
        raise ValueError("query expansion added_terms must be a string list")

    expanded_query = " ".join(expanded_query.split()).strip()
    added_terms = list(
        dict.fromkeys(" ".join(term.split()).strip() for term in added_terms)
    )
    if not expanded_query:
        raise ValueError("expanded query is empty")
    if len(expanded_query) > MAX_QUERY_EXPANSION_OUTPUT_CHARS:
        raise ValueError("expanded query is too long")
    if any(not term or len(term) > 40 for term in added_terms):
        raise ValueError("query expansion contains an invalid added term")
    if len(added_terms) > _max_expansion_terms():
        raise ValueError("query expansion contains too many added terms")

    normalized_expanded_query = _normalize_for_match(expanded_query)
    if any(
        _normalize_for_match(term) not in normalized_expanded_query
        for term in added_terms
    ):
        raise ValueError("query expansion added term is absent from expanded_query")
    return expanded_query, added_terms


def _apply_domain_expansion_policy(
    original: str,
    rewritten: str,
    expanded: str,
    added_terms: list[str],
) -> tuple[str, list[str]]:
    """为重要检索意图应用范围有限的确定性保护规则。"""
    filtered_terms = list(added_terms)

    # 目标价不是“投资评级”的同义词。用户没有询问价格或估值时，移除模型
    # 顺手补入的目标价，避免关键词分支被带到另一个检索意图。
    if not TARGET_PRICE_INTENT_PATTERN.search(original):
        expanded = TARGET_PRICE_PATTERN.sub(" ", expanded)
        filtered_terms = [
            term
            for term in filtered_terms
            if not TARGET_PRICE_PATTERN.search(term)
        ]

    required_terms = []
    if (
        "评级" in original
        and RATING_INTERPRETATION_CUE_PATTERN.search(original)
    ):
        required_terms.extend(RATING_REQUIRED_EXPANSION_TERMS)

    normalized_expanded = _normalize_for_match(expanded)
    for term in required_terms:
        if _normalize_for_match(term) not in normalized_expanded:
            expanded = f"{expanded} {term}"
            normalized_expanded = _normalize_for_match(expanded)

    normalized_rewritten = _normalize_for_match(rewritten)
    required_added_terms = [
        term
        for term in required_terms
        if _normalize_for_match(term) not in normalized_rewritten
    ]
    combined_terms = list(
        dict.fromkeys([*required_added_terms, *filtered_terms])
    )

    maximum_terms = _max_expansion_terms()
    retained_terms = combined_terms[:maximum_terms]
    dropped_terms = combined_terms[maximum_terms:]
    for term in dropped_terms:
        if term not in required_terms:
            expanded = expanded.replace(term, " ")

    return " ".join(expanded.split()).strip(), retained_terms


def _fallback_result(
    original: str,
    rewritten: str,
    *,
    expanded: str = "",
    added_terms: list[str] | None = None,
    expansion_validation: QueryValidationResult | None = None,
    source: QueryExpansionSource = "fallback",
    reason: str,
) -> QueryExpansionResult:
    rewrite_validation = validate_query_preservation(original, rewritten)
    effective_query = rewritten if rewrite_validation.valid else original
    if expansion_validation is None:
        expansion_validation = validate_query_preservation(original, "")

    return QueryExpansionResult(
        original=original,
        rewritten=rewritten,
        expanded=expanded,
        effective_query=effective_query,
        changed=effective_query != original,
        expansion_applied=False,
        added_terms=added_terms or [],
        source=source,
        fallback_reason=reason,
        rewrite_validation=rewrite_validation,
        expansion_validation=expansion_validation,
    )


def expand_query(
    original: str,
    rewritten: str,
    *,
    client: Any | None = None,
    allow_model: bool = True,
) -> QueryExpansionResult:
    """生成一条扩展关键词查询，并检查原始意图是否保留。"""
    original_query = original.strip() if isinstance(original, str) else ""
    rewritten_query = rewritten.strip() if isinstance(rewritten, str) else ""

    if not original_query:
        return QueryExpansionResult.noop(original_query, reason="原查询为空")
    if not rewritten_query:
        rewritten_query = original_query
    if not _environment_flag("QUERY_EXPANSION_ENABLED", True):
        return _fallback_result(
            original_query,
            rewritten_query,
            source="disabled",
            reason="查询扩展已关闭，使用规范化查询",
        )
    if (
        len(original_query) > MAX_QUERY_EXPANSION_INPUT_CHARS
        or len(rewritten_query) > MAX_QUERY_EXPANSION_INPUT_CHARS
    ):
        return _fallback_result(
            original_query,
            rewritten_query,
            reason="查询过长，跳过扩展",
        )

    rewrite_validation = validate_query_preservation(original_query, rewritten_query)
    if not rewrite_validation.valid:
        return _fallback_result(original_query, rewritten_query,
                                reason="规范化查询未通过保真校验，保留原问题")
    safe_query = rewritten_query if rewrite_validation.valid else original_query
    terms = alias_expansions(safe_query, _max_expansion_terms())
    if terms and (safe_query == original_query or not allow_model):
        expanded = f"{safe_query} {' '.join(terms)}"
        validation = validate_query_preservation(original_query, expanded)
        if validation.valid and len(expanded) <= MAX_QUERY_EXPANSION_OUTPUT_CHARS:
            return QueryExpansionResult(
                original=original_query, rewritten=rewritten_query, expanded=expanded,
                effective_query=expanded, changed=expanded != original_query,
                expansion_applied=True, added_terms=terms, source="rule",
                fallback_reason=None, rewrite_validation=rewrite_validation,
                expansion_validation=validation,
            )
    if not allow_model:
        return _fallback_result(original_query, rewritten_query, source="skipped",
                                reason="没有可用的词表别名，跳过模型扩展")

    try:
        resolved_client = client or get_query_expansion_client()
        completion = model_completion(resolved_client, "query_expansion",
            model=os.getenv(
                "QUERY_EXPANSION_MODEL",
                os.getenv("QUERY_REWRITE_MODEL", DEFAULT_QUERY_EXPANSION_MODEL),
            ),
            messages=[
                {
                    "role": "system",
                    "content": QUERY_EXPANSION_SYSTEM_PROMPT.format(
                        max_terms=_max_expansion_terms()
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "original_query": original_query,
                            "normalized_query": safe_query,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=192,
            stream=False,
            timeout=_expansion_timeout_seconds(),
            extra_body={"enable_thinking": False},
        )

        #对模型生成的扩展查询做“意图保留校验”，保证扩展后的查询不会把用户本来的意思改丢/改偏
        expanded_query, added_terms = _parse_expansion(
            _response_content(completion)
        )
        expanded_query, added_terms = _apply_domain_expansion_policy(
            original_query,
            rewritten_query,
            expanded_query,
            added_terms,
        )
        rewrite_validation = validate_query_preservation(
            original_query,
            rewritten_query,
        )
        expansion_validation = validate_query_preservation(
            original_query,
            expanded_query,
        )
        if not expansion_validation.valid:
            fallback_target = (
                "规范化查询" if rewrite_validation.valid else "原查询"
            )
            return _fallback_result(
                original_query,
                rewritten_query,
                expanded=expanded_query,
                added_terms=added_terms,
                expansion_validation=expansion_validation,
                reason=f"扩展查询保留校验失败，使用{fallback_target}",
            )

        logger.info(
            "Query expansion completed: changed=%s added_terms=%s",
            expanded_query != original_query,
            len(added_terms),
        )
        return QueryExpansionResult(
            original=original_query,
            rewritten=rewritten_query,
            expanded=expanded_query,
            effective_query=expanded_query,
            changed=expanded_query != original_query,
            expansion_applied=True,
            added_terms=added_terms,
            source="model",
            fallback_reason=None,
            rewrite_validation=rewrite_validation,
            expansion_validation=expansion_validation,
        )
    except Exception as error:
        logger.warning(
            "Query expansion failed; using validated fallback: %s",
            type(error).__name__,
        )
        return _fallback_result(
            original_query,
            rewritten_query,
            reason=f"查询扩展失败（{type(error).__name__}），使用校验后的回退查询",
        )
