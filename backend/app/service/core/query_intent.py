"""RAG query intent analysis used to select query preprocessing safely."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Literal

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_QUERY_INTENT_MODEL = "qwen3.7-flash-2026-07-15"
DEFAULT_QUERY_INTENT_TIMEOUT_SECONDS = 6.0
MAX_QUERY_INTENT_INPUT_CHARS = 2000
MAX_QUERY_INTENT_REASON_CHARS = 200

QueryIntentName = Literal[
    "precise_lookup",
    "colloquial_lookup",
    "complex_lookup",
    "ambiguous_query",
    "contextual_query",
    "general_lookup",
    "invalid_query",
]
QueryIntentSource = Literal["rule", "model", "fallback", "disabled"]

VALID_QUERY_INTENTS = {
    "precise_lookup",
    "colloquial_lookup",
    "complex_lookup",
    "ambiguous_query",
    "contextual_query",
    "general_lookup",
    "invalid_query",
}

# Exact document locators carry unusually high retrieval value. Rewriting them
# is more likely to cause query drift than to improve recall.
PRECISE_LOCATOR_PATTERN = re.compile(
    r"(?:表|图|附表|附图)\s*[A-Za-z]?\s*\d+(?:\s*[-－—.]\s*\d+)*"
    r"|第\s*[一二三四五六七八九十百零〇两\d]+\s*(?:章|节|页)"
)

COLLOQUIAL_PATTERNS = (
    re.compile(r"赚(?:了|到)?多少钱|能赚多少|赚不赚钱"),
    re.compile(r"贵不贵|便不便宜|值不值得"),
    re.compile(r"咋样|怎么样|怎么回事|为啥|啥时候|啥意思"),
    re.compile(r"涨了多少|跌了多少|掉了多少|卖了多少"),
    re.compile(r"靠什么(?:赚钱|增长)?|有啥|多少个钱"),
)

CONTEXTUAL_PATTERN = re.compile(
    r"^(?:那|那么|然后|还有|它|他|她|这个|那个|上述|前者|后者)"
)

QUERY_INTENT_SYSTEM_PROMPT = """
你是知识库 RAG 的查询分析器。你的任务是判断用户问题的检索意图，以及是否需要在检索前改写；不要回答问题。

intent 只能取以下值之一：
- precise_lookup：已有明确实体、时间、指标、表号、图号或限定条件，可直接检索。
- colloquial_lookup：包含口语、俗称或与文档术语不一致的表达。
- complex_lookup：包含多个需要分别检索再合并的问题。
- ambiguous_query：核心实体或条件缺失，无法安全确定检索目标。
- contextual_query：依赖上一轮对话中的实体或条件。
- general_lookup：普通知识库查询，不属于以上类型。

判断规则：
1. 只有口语规范化、简称展开或补全检索术语能明显提升召回时，need_rewrite 才为 true。
2. 已经正式、精确、可直接检索的问题，need_rewrite 必须为 false。
3. 表号、图号、章节、页码、公司、人名、时间、数字、单位、否定词和限定条件都是不可丢失的精确信息。
4. 只有问题包含两个及以上需要独立检索的子问题时，need_decompose 才为 true。
5. ambiguous_query 不得通过猜测补全，因此 need_rewrite 为 false。
6. 只返回 JSON 对象，不输出其他文字。

返回格式：
{"intent":"precise_lookup","need_rewrite":false,"need_decompose":false,"reason":"简短原因"}
""".strip()


@dataclass(frozen=True)
class QueryIntentDecision:
    """Structured decision made before RAG query rewriting."""

    intent: QueryIntentName
    need_rewrite: bool
    need_decompose: bool
    reason: str
    source: QueryIntentSource

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _environment_flag(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().casefold() not in {"0", "false", "no", "off"}


def _intent_timeout_seconds() -> float:
    raw_value = os.getenv(
        "RAG_QUERY_INTENT_TIMEOUT_SECONDS",
        str(DEFAULT_QUERY_INTENT_TIMEOUT_SECONDS),
    )
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return DEFAULT_QUERY_INTENT_TIMEOUT_SECONDS
    return value if value > 0 else DEFAULT_QUERY_INTENT_TIMEOUT_SECONDS


@lru_cache(maxsize=1)
def get_query_intent_client() -> OpenAI:
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    base_url = os.getenv("DASHSCOPE_BASE_URL", "").strip()
    if not api_key or not base_url:
        raise RuntimeError(
            "DASHSCOPE_API_KEY and DASHSCOPE_BASE_URL are required for query intent analysis"
        )
    return OpenAI(api_key=api_key, base_url=base_url)


def _response_content(completion: Any) -> str:
    choices = getattr(completion, "choices", None) or []
    if not choices:
        raise ValueError("query intent model returned no choices")
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise ValueError("query intent model returned empty content")
    return content.strip()


def _parse_intent_decision(content: str) -> QueryIntentDecision:
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise ValueError("query intent response must be a JSON object")

    intent = payload.get("intent")
    need_rewrite = payload.get("need_rewrite")
    need_decompose = payload.get("need_decompose")
    reason = payload.get("reason")

    if intent not in VALID_QUERY_INTENTS:
        raise ValueError("query intent response contains an invalid intent")
    if not isinstance(need_rewrite, bool):
        raise ValueError("query intent need_rewrite must be a boolean")
    if not isinstance(need_decompose, bool):
        raise ValueError("query intent need_decompose must be a boolean")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("query intent reason must be a non-empty string")

    normalized_reason = " ".join(reason.split()).strip()
    if len(normalized_reason) > MAX_QUERY_INTENT_REASON_CHARS:
        raise ValueError("query intent reason is too long")

    # The model may identify ambiguity correctly but still request a rewrite.
    # Without missing context, such a rewrite would have to invent facts.
    if intent == "ambiguous_query":
        need_rewrite = False

    return QueryIntentDecision(
        intent=intent,
        need_rewrite=need_rewrite,
        need_decompose=need_decompose,
        reason=normalized_reason,
        source="model",
    )


def _rule_based_decision(question: str) -> QueryIntentDecision | None:
    if PRECISE_LOCATOR_PATTERN.search(question):
        return QueryIntentDecision(
            intent="precise_lookup",
            need_rewrite=False,
            need_decompose=False,
            reason="包含明确的表、图、章节或页码定位信息",
            source="rule",
        )

    if any(pattern.search(question) for pattern in COLLOQUIAL_PATTERNS):
        return QueryIntentDecision(
            intent="colloquial_lookup",
            need_rewrite=True,
            need_decompose=False,
            reason="包含需要转换为文档术语的口语表达",
            source="rule",
        )

    if CONTEXTUAL_PATTERN.search(question):
        return QueryIntentDecision(
            intent="contextual_query",
            need_rewrite=True,
            need_decompose=False,
            reason="查询包含依赖上下文的指代或承接表达",
            source="rule",
        )

    return None


def analyze_query_intent(
    question: str,
    *,
    client: Any | None = None,
) -> QueryIntentDecision:
    """Classify a RAG query and conservatively decide whether to rewrite it.

    Exact locators and common conversational forms use deterministic rules.
    Other queries are classified by the configured model. Any provider or
    parsing failure falls back to no rewrite so intent analysis cannot degrade
    retrieval availability or silently introduce query drift.
    """
    normalized_question = question.strip() if isinstance(question, str) else ""
    if not normalized_question:
        return QueryIntentDecision(
            intent="invalid_query",
            need_rewrite=False,
            need_decompose=False,
            reason="查询为空",
            source="rule",
        )

    if not _environment_flag("RAG_QUERY_INTENT_ENABLED", True):
        return QueryIntentDecision(
            intent="general_lookup",
            need_rewrite=True,
            need_decompose=False,
            reason="意图识别已关闭，沿用原有的全量改写策略",
            source="disabled",
        )

    if len(normalized_question) > MAX_QUERY_INTENT_INPUT_CHARS:
        return QueryIntentDecision(
            intent="general_lookup",
            need_rewrite=False,
            need_decompose=False,
            reason="查询过长，跳过意图识别和改写",
            source="fallback",
        )

    rule_decision = _rule_based_decision(normalized_question)
    if rule_decision is not None:
        return rule_decision

    try:
        resolved_client = client or get_query_intent_client()
        completion = resolved_client.chat.completions.create(
            model=os.getenv(
                "RAG_QUERY_INTENT_MODEL",
                os.getenv("QUERY_REWRITE_MODEL", DEFAULT_QUERY_INTENT_MODEL),
            ),
            messages=[
                {"role": "system", "content": QUERY_INTENT_SYSTEM_PROMPT},
                {"role": "user", "content": normalized_question},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=160,
            stream=False,
            timeout=_intent_timeout_seconds(),
            extra_body={"enable_thinking": False},
        )
        decision = _parse_intent_decision(_response_content(completion))
        logger.info(
            "RAG query intent analyzed: intent=%s need_rewrite=%s need_decompose=%s source=%s",
            decision.intent,
            decision.need_rewrite,
            decision.need_decompose,
            decision.source,
        )
        return decision
    except Exception as error:
        logger.warning(
            "RAG query intent analysis failed; skipping rewrite: %s",
            type(error).__name__,
        )
        return QueryIntentDecision(
            intent="general_lookup",
            need_rewrite=False,
            need_decompose=False,
            reason="意图识别失败，保守使用原查询",
            source="fallback",
        )
