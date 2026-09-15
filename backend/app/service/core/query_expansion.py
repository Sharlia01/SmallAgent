"""Retrieval query expansion with deterministic intent-preservation checks."""

from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Literal

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_QUERY_EXPANSION_MODEL = "qwen3.7-flash-2026-07-15"
DEFAULT_QUERY_EXPANSION_TIMEOUT_SECONDS = 8.0
DEFAULT_QUERY_EXPANSION_MAX_TERMS = 8
MAX_QUERY_EXPANSION_INPUT_CHARS = 2000
MAX_QUERY_EXPANSION_OUTPUT_CHARS = 400

QueryExpansionSource = Literal["model", "fallback", "disabled", "skipped"]

LOCATOR_PATTERN = re.compile(
    r"(?:表|图|附表|附图)\s*[A-Za-z]?\s*\d+(?:\s*[-－—.]\s*\d+)*"
    r"|第\s*[一二三四五六七八九十百零〇两\d]+\s*(?:章|节|页)"
)
YEAR_OR_DATE_PATTERN = re.compile(
    r"(?<!\d)(?:19|20)\d{2}\s*年\s*\d{1,2}\s*月(?:\s*\d{1,2}\s*日)?"
    r"|(?<!\d)(?:19|20)\d{2}\s*年?\s*(?:第\s*)?[一二三四1-4]\s*季度"
    r"|(?<!\d)(?:19|20)\d{2}\s*年?\s*(?:上半年|下半年|全年|年度)"
    r"|(?<!\d)(?:19|20)\d{2}(?:\s*[-－—至到/]\s*(?:(?:19|20)?\d{2}))?\s*年?(?!\d)"
    r"|(?<!\d)(?:\d{2}|(?:19|20)\d{2})\s*H[12]"
    r"|(?:\d{2}|(?:19|20)\d{2})?\s*Q[1-4]",
    re.IGNORECASE,
)
RELATIVE_TIME_PATTERN = re.compile(
    r"(?:未来|过去|近)\s*[一二三四五六七八九十两\d]+\s*(?:年|个月|季度)"
    r"|今年|去年|明年|本年度|上年度|本季度|上季度|上半年|下半年"
    r"|(?:第\s*)?[一二三四1-4]\s*季度"
)
NUMBER_WITH_UNIT_PATTERN = re.compile(
    r"[-+]?\d+(?:\.\d+)?\s*(?:%|％|亿元|万元|元|倍|万吨|万千瓦|"
    r"亿千瓦时|千瓦时|兆瓦时|GW|MW|kW|x)",
    re.IGNORECASE,
)
NUMBER_PATTERN = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?")
STOCK_CODE_PATTERN = re.compile(
    r"(?<!\d)\d{6}(?:\.(?:SH|SZ|BJ))?(?!\d)",
    re.IGNORECASE,
)
QUOTED_TERM_PATTERN = re.compile(r"[“‘\"']([^”’\"']{2,40})[”’\"']")
NEGATION_PATTERN = re.compile(
    r"没有|并未|未能|未曾|未发生|未提供|未披露|未包含|不能|不含|"
    r"不包括|不是|不高于|不低于|不超过|不少于|不支持|不考虑|"
    r"不计算|无法|无需"
)

ORGANIZATION_PATTERN = re.compile(
    r"[\u4e00-\u9fffA-Za-z0-9·]{2,24}"
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

# These are already canonical retrieval terms. If the user supplied one, an
# expansion should retain it verbatim rather than replace it with a broader
# concept.
CANONICAL_TERM_PATTERN = re.compile(
    r"归母净利润|扣非归母净利润|营业收入|营业成本|每股收益|市盈率|"
    r"市净率|现金流|装机容量|发电量|上网电价|利用小时|煤价|分红|"
    r"股息率|估值|毛利率|净利率|资产负债率|同比|环比|预测|实际|"
    r"截至|分别",
    re.IGNORECASE,
)

QUERY_EXPANSION_SYSTEM_PROMPT = """
你是知识库 RAG 的查询扩展器。请根据原问题和规范化问题，生成一条用于关键词召回的扩展查询；不要回答问题。

要求：
1. 保留原问题中的公司、人名、证券代码、时间、数字、单位、否定词、表号、图号、章节、页码和所有限定条件。
2. 只补充原问题中口语表达对应的常见书面语、专业字段名、缩写和同义检索词。
3. 不得猜测或新增具体年份、日期、数字、实体、结论及事实。
4. 同时覆盖原问题中的每个子意图，不得只保留其中一部分。
5. added_terms 只列出相对规范化问题新增的检索词，最多 {max_terms} 个。
6. 只返回 JSON 对象，格式为：
{{"expanded_query":"扩展查询","added_terms":["新增词1","新增词2"]}}

示例：
输入：{{"original_query":"研报觉得国电电力未来三年能赚多少钱，对应估值贵不贵？","normalized_query":"国电电力未来三年盈利预测与估值"}}
输出：{{"expanded_query":"国电电力 未来三年 盈利预测 归母净利润 每股收益 EPS 市盈率 PE 估值","added_terms":["归母净利润","每股收益","EPS","市盈率","PE"]}}
""".strip()


@dataclass(frozen=True)
class QueryValidationResult:
    valid: bool
    required_anchors: list[str]
    missing_anchors: list[str]
    added_risky_terms: list[str]


@dataclass(frozen=True)
class QueryExpansionResult:
    """Full query transformation state retained for retrieval diagnostics."""

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
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(
        character
        for character in normalized
        if character.isalnum() or "\u4e00" <= character <= "\u9fff"
    )


def _unique_matches(pattern: re.Pattern, query: str) -> set[str]:
    return {
        " ".join(match.group(0).split()).strip()
        for match in pattern.finditer(query)
        if match.group(0).strip()
    }


def _extract_organizations(query: str) -> set[str]:
    organizations = set()
    for match in ORGANIZATION_PATTERN.finditer(query):
        candidate = match.group(0).strip()
        for prefix in ENTITY_PREFIXES:
            prefix_position = candidate.rfind(prefix)
            if prefix_position >= 0:
                candidate = candidate[prefix_position + len(prefix):]
        candidate = RELATIVE_TIME_PATTERN.sub("", candidate).strip()
        if (
            candidate
            and candidate not in GENERIC_ORGANIZATIONS
            and len(candidate) <= 20
        ):
            organizations.add(candidate)
    return organizations


def extract_query_anchors(query: str) -> list[str]:
    """Extract high-confidence terms that transformations must preserve."""
    if not isinstance(query, str) or not query.strip():
        return []

    anchors = set()
    for pattern in (
        LOCATOR_PATTERN,
        YEAR_OR_DATE_PATTERN,
        RELATIVE_TIME_PATTERN,
        NUMBER_WITH_UNIT_PATTERN,
        STOCK_CODE_PATTERN,
        NEGATION_PATTERN,
        CANONICAL_TERM_PATTERN,
    ):
        anchors.update(_unique_matches(pattern, query))

    anchors.update(_extract_organizations(query))
    anchors.update(
        match.group(1).strip()
        for match in QUOTED_TERM_PATTERN.finditer(query)
        if match.group(1).strip()
    )
    return sorted(anchors, key=lambda value: (_normalize_for_match(value), value))


def _extract_risky_additions(query: str) -> set[str]:
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
    """Reject candidates that drop anchors or invent risky exact details."""
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

    return QueryValidationResult(
        valid=bool(candidate.strip()) and not missing_anchors and not added_risky_terms,
        required_anchors=required_anchors,
        missing_anchors=missing_anchors,
        added_risky_terms=added_risky_terms,
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
) -> QueryExpansionResult:
    """Generate one expanded keyword query and validate intent preservation."""
    original_query = original.strip() if isinstance(original, str) else ""
    rewritten_query = rewritten.strip() if isinstance(rewritten, str) else ""

    if not original_query:
        return QueryExpansionResult.noop(original_query, reason="原查询为空")
    if not rewritten_query or rewritten_query == original_query:
        return QueryExpansionResult.noop(
            original_query,
            reason="规范化查询未发生变化，跳过扩展",
        )
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

    try:
        resolved_client = client or get_query_expansion_client()
        completion = resolved_client.chat.completions.create(
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
                            "normalized_query": rewritten_query,
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
