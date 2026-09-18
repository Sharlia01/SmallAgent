"""RAG query intent analysis used to select query preprocessing safely."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field
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
MAX_SUBQUERY_CHARS = 300
MAX_SUBQUERIES = 2

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
RetrievalMode = Literal["single", "parallel", "sequential"]

VALID_QUERY_INTENTS = {
    "precise_lookup",
    "colloquial_lookup",
    "complex_lookup",
    "ambiguous_query",
    "contextual_query",
    "general_lookup",
    "invalid_query",
}
VALID_RETRIEVAL_MODES = {"single", "parallel", "sequential"}
PLAN_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,31}$")

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
RATING_SEQUENTIAL_PATTERN = re.compile(
    r"(?=.*评级)(?=.*(?:看多|看空|含义|意思|定义|标准|代表|意味))"
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
4. retrieval_mode 只能是 single、parallel、sequential：
   - single：一次检索可以完整回答。
   - parallel：多个子问题互不依赖，可以分别检索。
   - sequential：后一个子问题必须使用前一步从证据中取得的术语、实体或结论。
5. 只有 retrieval_mode 不是 single 时，need_decompose 才为 true。
6. sequential 必须正好给出两个 subqueries。第一步通过 output_slot 声明要从证据提取的槽位；第二步通过 query_template 和 required_slots 使用该槽位。不得提前猜测槽位值。
7. 当用户询问“研报给了什么评级，以及该评级代表看多还是看空”时，应判为 sequential：先查询评级，再查询该评级的定义。
8. 用户提到“这份研报、该报告、同一文档”时，第二步 inherit_document_scope 应为 true。
9. ambiguous_query 不得通过猜测补全，因此 need_rewrite 为 false。
10. single 的 subqueries 必须为空。只返回 JSON 对象，不输出其他文字。

返回格式：
{
  "intent":"complex_lookup",
  "need_rewrite":true,
  "need_decompose":true,
  "retrieval_mode":"sequential",
  "reason":"后一步依赖前一步得到的评级名称",
  "subqueries":[
    {
      "id":"rating_lookup",
      "query":"国电电力研报的投资评级是什么？",
      "query_template":"",
      "depends_on":[],
      "output_slot":"rating",
      "required_slots":[],
      "inherit_document_scope":false
    },
    {
      "id":"rating_definition",
      "query":"",
      "query_template":"“{rating}”的评级定义和评级标准是什么？",
      "depends_on":["rating_lookup"],
      "output_slot":null,
      "required_slots":["rating"],
      "inherit_document_scope":true
    }
  ]
}
""".strip()


@dataclass(frozen=True)
class QuerySubqueryPlan:
    """One executable step in a bounded retrieval plan."""

    id: str
    query: str = ""
    query_template: str = ""
    depends_on: list[str] = field(default_factory=list)
    output_slot: str | None = None
    required_slots: list[str] = field(default_factory=list)
    inherit_document_scope: bool = False


@dataclass(frozen=True)
class QueryIntentDecision:
    """Structured decision made before RAG query rewriting."""

    intent: QueryIntentName
    need_rewrite: bool
    need_decompose: bool
    reason: str
    source: QueryIntentSource
    retrieval_mode: RetrievalMode = "single"
    subqueries: list[QuerySubqueryPlan] = field(default_factory=list)

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
    retrieval_mode = payload.get("retrieval_mode")

    if intent not in VALID_QUERY_INTENTS:
        raise ValueError("query intent response contains an invalid intent")
    if not isinstance(need_rewrite, bool):
        raise ValueError("query intent need_rewrite must be a boolean")
    if not isinstance(need_decompose, bool):
        raise ValueError("query intent need_decompose must be a boolean")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("query intent reason must be a non-empty string")

    # Older providers may still return the original boolean-only contract.
    # Preserve that response while requiring a full plan before execution.
    if retrieval_mode is None:
        retrieval_mode = "parallel" if need_decompose else "single"
    if retrieval_mode not in VALID_RETRIEVAL_MODES:
        raise ValueError("query intent response contains an invalid retrieval mode")

    raw_subqueries = payload.get("subqueries", [])
    if not isinstance(raw_subqueries, list):
        raise ValueError("query intent subqueries must be a list")
    if len(raw_subqueries) > MAX_SUBQUERIES:
        raise ValueError("query intent contains too many subqueries")
    subqueries = [_parse_subquery(item) for item in raw_subqueries]

    if retrieval_mode == "single" and subqueries:
        raise ValueError("single retrieval cannot contain subqueries")
    if retrieval_mode == "sequential":
        _validate_sequential_plan(subqueries)
    need_decompose = retrieval_mode != "single"

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
        retrieval_mode=retrieval_mode,
        subqueries=subqueries,
    )


def _normalize_plan_text(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"query intent {field_name} must be a string")
    normalized = " ".join(value.split()).strip()
    if len(normalized) > MAX_SUBQUERY_CHARS:
        raise ValueError(f"query intent {field_name} is too long")
    return normalized


def _parse_identifier_list(value: Any, *, field_name: str) -> list[str]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and PLAN_ID_PATTERN.fullmatch(item)
        for item in value
    ):
        raise ValueError(f"query intent {field_name} is invalid")
    return list(dict.fromkeys(value))


def _parse_subquery(payload: Any) -> QuerySubqueryPlan:
    if not isinstance(payload, dict):
        raise ValueError("query intent subquery must be an object")
    subquery_id = payload.get("id")
    if not isinstance(subquery_id, str) or not PLAN_ID_PATTERN.fullmatch(
        subquery_id
    ):
        raise ValueError("query intent subquery id is invalid")

    output_slot = payload.get("output_slot")
    if output_slot is not None and (
        not isinstance(output_slot, str)
        or not PLAN_ID_PATTERN.fullmatch(output_slot)
    ):
        raise ValueError("query intent output_slot is invalid")
    inherit_document_scope = payload.get("inherit_document_scope", False)
    if not isinstance(inherit_document_scope, bool):
        raise ValueError("inherit_document_scope must be a boolean")

    return QuerySubqueryPlan(
        id=subquery_id,
        query=_normalize_plan_text(payload.get("query", ""), field_name="query"),
        query_template=_normalize_plan_text(
            payload.get("query_template", ""),
            field_name="query_template",
        ),
        depends_on=_parse_identifier_list(
            payload.get("depends_on", []),
            field_name="depends_on",
        ),
        output_slot=output_slot,
        required_slots=_parse_identifier_list(
            payload.get("required_slots", []),
            field_name="required_slots",
        ),
        inherit_document_scope=inherit_document_scope,
    )


def _validate_sequential_plan(subqueries: list[QuerySubqueryPlan]) -> None:
    if len(subqueries) != 2:
        raise ValueError("sequential retrieval must contain exactly two steps")
    first, second = subqueries
    if first.id == second.id:
        raise ValueError("sequential retrieval step ids must be unique")
    if not first.query or first.query_template or first.depends_on:
        raise ValueError("first sequential step is invalid")
    if not first.output_slot:
        raise ValueError("first sequential step must declare output_slot")
    if first.required_slots:
        raise ValueError("first sequential step cannot require bridge slots")
    if second.query or not second.query_template:
        raise ValueError("second sequential step must use query_template")
    if second.output_slot is not None:
        raise ValueError("second sequential step cannot declare output_slot")
    if second.depends_on != [first.id]:
        raise ValueError("second sequential step has an invalid dependency")
    if second.required_slots != [first.output_slot]:
        raise ValueError("second sequential step must consume the first output slot")
    if "{" + first.output_slot + "}" not in second.query_template:
        raise ValueError("second query template does not reference the output slot")


def _rule_based_decision(question: str) -> QueryIntentDecision | None:
    if PRECISE_LOCATOR_PATTERN.search(question):
        return QueryIntentDecision(
            intent="precise_lookup",
            need_rewrite=False,
            need_decompose=False,
            reason="包含明确的表、图、章节或页码定位信息",
            source="rule",
        )

    if RATING_SEQUENTIAL_PATTERN.search(question):
        return QueryIntentDecision(
            intent="complex_lookup",
            need_rewrite=True,
            need_decompose=True,
            reason="评级含义依赖先取得研报给出的具体评级名称",
            source="rule",
            retrieval_mode="sequential",
            subqueries=[
                QuerySubqueryPlan(
                    id="rating_lookup",
                    query=question,
                    output_slot="rating",
                ),
                QuerySubqueryPlan(
                    id="rating_definition",
                    query_template=(
                        "“{rating}”的评级定义、评级标准、股价表现与"
                        "市场代表性指数是什么？"
                    ),
                    depends_on=["rating_lookup"],
                    required_slots=["rating"],
                    inherit_document_scope=(
                        "这份研报" in question
                        or "该研报" in question
                        or "这份报告" in question
                        or "该报告" in question
                    ),
                ),
            ],
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
            max_tokens=520,
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
