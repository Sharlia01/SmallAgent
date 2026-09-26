"""Layer 1: analyze a RAG question without building a retrieval plan."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from typing import Any, Literal

from dotenv import load_dotenv
from openai import OpenAI
from service.core.query_contract import QueryGoal, decode_goals, validate_goal_sources, MAX_GOALS
from service.core.query_runtime import validated_completion, record_observation, failure_details
from service.core.query_validation import QueryValidationError, validate_goals, UNKNOWN_LOCATOR
from service.core.query_semantics import alias_expansions, metric_terms


load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_QUERY_INTENT_MODEL = "qwen3.7-flash-2026-07-15"
DEFAULT_QUERY_INTENT_TIMEOUT_SECONDS = 6.0
MAX_QUERY_INTENT_INPUT_CHARS = 2000
MAX_QUERY_INTENT_REASON_CHARS = 200
MAX_RETRIEVAL_TARGETS = MAX_GOALS

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
COLLOQUIAL_MULTI_FACET_PATTERN = re.compile(
    r"(?:，|,|；|;|？|\?).*(?:对应|另外|同时|以及|多少|是什么|如何|查询)"
    r"|另一份|另一个|并查询|并查|以及|同时|分别"
)
EXPLICIT_RATING_PATTERN = re.compile(r"[“‘\"'][^”’\"']{2,24}[”’\"']\s*(?:投资)?评级")


QUERY_ANALYSIS_SYSTEM_PROMPT = """
你是知识库 RAG 的查询分析器。只分析问题的检索特征，不生成子查询，不回答问题，也不选择或描述执行步骤。

intent 只能取以下值之一：
- precise_lookup：已有明确实体、时间、指标、表号、图号或限定条件，可直接检索。
- colloquial_lookup：包含口语、俗称或与文档术语不一致的表达。
- complex_lookup：包含多个事实目标，或者存在前后依赖的多跳目标。complex 不代表一定要拆分。
- ambiguous_query：核心实体或条件缺失，无法安全确定检索目标。
- contextual_query：依赖上一轮对话中的实体或条件。
- general_lookup：普通知识库查询，不属于以上类型。

分析以下检索信号：
1. target_count：不同检索目标的数量，1至8。多个问句、多个年份、计算步骤不自动构成多个检索目标。
2. targets_independent：这些目标能否直接根据原问题各自检索。后一个目标必须使用前一个答案时为 false。
3. has_dependency：后一个目标是否必须取得前一个目标的答案、术语或实体后才能构造查询。
4. single_retrieval_likely_covers_all：一次检索的同一结果行、同一段落或同一局部证据是否很可能覆盖全部事实目标。

判断原则：
- 默认把问题视为一个检索目标，除非存在明确不同的实体、业务指标、主题、图表定位或证据范围。
- 同一指标的多个年份、同一表格行中的多个值、同一段落中的“数值和原因”，以及对已检索数值做求和、差值、比例或大小比较，都视为一次检索可能覆盖。
- 两个业务指标即使属于同一张表，只要通常位于不同行、单个 Top-K 容易只召回其中一个，就视为一次检索不一定覆盖。
- 近义改写不算不同目标。
- “先查评级，再解释该评级含义”存在依赖；“先查两个数，再计算差值”不存在检索依赖，计算不是检索目标。
- 只有口语规范化、简称展开或补全检索术语能明显提升召回时，need_rewrite 才为 true。
- 已经正式、精确、可直接检索的问题，need_rewrite 必须为 false。
- ambiguous_query 不得通过猜测补全，因此 need_rewrite 必须为 false。
- 不得丢失或猜测表号、图号、章节、页码、公司、人名、时间、数字、单位、否定词和限定条件。
- 语言表达、上下文依赖和任务复杂度可以同时存在，不能因一个表号或口语词就忽略其他目标。
- goals 为语义目标清单，不是执行子查询。每个目标有唯一 id、完整目标描述 text、适用 constraints 和依赖目标 id 列表 depends_on。id 推荐小写字母、数字、下划线，不超过32字符。
- source_spans 必须逐字引用原文；constraints 可保留原词或做确定的术语、日期规范化，并继承同一句中明确共享的修饰语，不得增加新事实。口语措辞不是必须逐字保留的查询模板。
- 用户问“哪张图/哪个对象对应某事实”时，该对应关系是待求答案。必须保留候选范围和未知关系，禁止按提及顺序把图号或对象与事实配对。
- 同一个修饰语限定多个指标时必须继承，例如“光伏发电量和上网电量”的两个目标都属于光伏。
- 不同主体的年份必须绑定到各自目标；全局的“不含预测值”等条件必须分配到每个适用目标。constraints 包含指标或目标主题原文。
- need_expansion 独立于 need_rewrite，正式字段也可以需要同义词召回。language_features 可取 colloquial、contextual、abbreviation；task_types 可取 lookup、definition、comparison、calculation、explanation、summary。
- 歧义无法安全消解时 intent=ambiguous_query，clarification 填需要用户补充的具体问题；不要猜主体。

只返回以下 JSON，不得包含 retrieval_mode、need_decompose、subqueries 或其他字段：
{
  "intent":"complex_lookup",
  "need_rewrite":false,
  "target_count":2,
  "targets_independent":true,
  "has_dependency":false,
  "single_retrieval_likely_covers_all":false,
  "reason":"两个指标通常位于不同表格行，单次结果集不一定同时覆盖",
  "need_expansion":false,
  "language_features":[],
  "task_types":["lookup"],
  "clarification":null,
  "goals":[
    {"id":"first_metric","text":"第一个指标的语义需求","constraints":["原问题中的指标原文"],"depends_on":[],"source_spans":["该目标的完整原文分句"]},
    {"id":"second_metric","text":"第二个指标的语义需求","constraints":["原问题中的指标原文"],"depends_on":[],"source_spans":["该目标的完整原文分句"]}
  ]
}
""".strip()


@dataclass(frozen=True)
class QueryAnalysis:
    """Semantic query features consumed by the deterministic router."""

    intent: QueryIntentName
    need_rewrite: bool
    target_count: int
    targets_independent: bool
    has_dependency: bool
    single_retrieval_likely_covers_all: bool
    reason: str
    source: QueryIntentSource
    goals: tuple[QueryGoal, ...] = ()
    need_expansion: bool = False
    language_features: tuple[str, ...] = ()
    task_types: tuple[str, ...] = ()
    clarification: str | None = None
    validation_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.validation_error is None:
            payload.pop("validation_error")
        return payload


def _environment_flag(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().casefold() not in {"0", "false", "no", "off"}


def query_intent_timeout_seconds() -> float:
    raw_value = os.getenv(
        "RAG_QUERY_INTENT_TIMEOUT_SECONDS",
        str(DEFAULT_QUERY_INTENT_TIMEOUT_SECONDS),
    )
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return DEFAULT_QUERY_INTENT_TIMEOUT_SECONDS
    return value if value > 0 else DEFAULT_QUERY_INTENT_TIMEOUT_SECONDS


def query_intent_model() -> str:
    return os.getenv(
        "RAG_QUERY_INTENT_MODEL",
        os.getenv("QUERY_REWRITE_MODEL", DEFAULT_QUERY_INTENT_MODEL),
    )


@lru_cache(maxsize=1)
def get_query_intent_client() -> OpenAI:
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    base_url = os.getenv("DASHSCOPE_BASE_URL", "").strip()
    if not api_key or not base_url:
        raise RuntimeError(
            "DASHSCOPE_API_KEY and DASHSCOPE_BASE_URL are required for query analysis"
        )
    return OpenAI(api_key=api_key, base_url=base_url)


def response_content(completion: Any) -> str:
    choices = getattr(completion, "choices", None) or []
    if not choices:
        raise ValueError("query analysis model returned no choices")
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise ValueError("query analysis model returned empty content")
    return content.strip()


def _normalized_reason(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise QueryValidationError("invalid_reason", "query analysis reason must be a non-empty string", field="reason")
    normalized = " ".join(value.split()).strip()
    if len(normalized) > MAX_QUERY_INTENT_REASON_CHARS:
        raise QueryValidationError("invalid_reason", "query analysis reason is too long", field="reason")
    return normalized


def _decode_query_analysis(content: str, question: str = "") -> QueryAnalysis:
    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise QueryValidationError("invalid_analysis_schema", "query analysis response must be a JSON object", field="response")

    forbidden_fields = {"need_decompose", "retrieval_mode", "subqueries"}
    if forbidden_fields.intersection(payload):
        raise QueryValidationError("unexpected_planning_fields", "query analysis response contains planning fields",
                                   field=sorted(forbidden_fields.intersection(payload))[0])

    intent = payload.get("intent")
    need_rewrite = payload.get("need_rewrite")
    target_count = payload.get("target_count")
    targets_independent = payload.get("targets_independent")
    has_dependency = payload.get("has_dependency")
    likely_covers = payload.get("single_retrieval_likely_covers_all")

    if intent not in VALID_QUERY_INTENTS:
        raise QueryValidationError("invalid_intent", "query analysis response contains an invalid intent", field="intent")
    if not isinstance(need_rewrite, bool):
        raise QueryValidationError("invalid_boolean", "query analysis need_rewrite must be a boolean", field="need_rewrite")
    if not isinstance(target_count, int) or isinstance(target_count, bool) or not (
        1 <= target_count <= MAX_RETRIEVAL_TARGETS
    ):
        raise QueryValidationError("invalid_target_count", "query analysis target_count must be between 1 and 8", field="target_count")
    if not isinstance(targets_independent, bool):
        raise QueryValidationError("invalid_boolean", "query analysis targets_independent must be a boolean", field="targets_independent")
    if not isinstance(has_dependency, bool):
        raise QueryValidationError("invalid_boolean", "query analysis has_dependency must be a boolean", field="has_dependency")
    if not isinstance(likely_covers, bool):
        raise QueryValidationError("invalid_boolean", "query analysis single_retrieval_likely_covers_all must be a boolean",
                                   field="single_retrieval_likely_covers_all")

    if intent == "ambiguous_query":
        need_rewrite = False

    goals = decode_goals(payload["goals"]) if "goals" in payload else ()
    need_expansion = payload.get("need_expansion", bool(alias_expansions(question)))
    if not isinstance(need_expansion, bool):
        raise QueryValidationError("invalid_boolean", "need_expansion must be a boolean", field="need_expansion")
    dimensions = {}
    for name, allowed in {
        "language_features": {"colloquial", "contextual", "abbreviation"},
        "task_types": {"lookup", "definition", "comparison", "calculation", "explanation", "summary"},
    }.items():
        values = payload.get(name, [])
        if not isinstance(values, list) or not all(isinstance(v, str) and v in allowed for v in values):
            raise QueryValidationError("invalid_analysis_dimension", f"invalid {name}", field=name)
        dimensions[name] = tuple(dict.fromkeys(values))
    clarification = payload.get("clarification")
    if clarification is not None and (not isinstance(clarification, str) or not clarification.strip() or len(clarification) > 200):
        raise QueryValidationError("invalid_clarification", "invalid clarification", field="clarification")

    return QueryAnalysis(
        intent=intent,
        need_rewrite=need_rewrite,
        target_count=target_count,
        targets_independent=targets_independent,
        has_dependency=has_dependency,
        single_retrieval_likely_covers_all=likely_covers,
        reason=_normalized_reason(payload.get("reason")),
        source="model",
        goals=goals,
        need_expansion=need_expansion and intent != "ambiguous_query",
        clarification=clarification if intent == "ambiguous_query" else None,
        **dimensions,
    )


def _validate_query_analysis(analysis: QueryAnalysis, question: str) -> QueryAnalysis:
    goals = validate_goal_sources(analysis.goals, question)
    validate_goals(question, goals)
    if goals:
        count = len(goals)
        dependency = any(goal.depends_on for goal in goals)
        signals = {"target_count": count, "has_dependency": dependency,
                   "targets_independent": count > 1 and not dependency}
        changes = {key: {"before": getattr(analysis, key), "after": value}
                   for key, value in signals.items() if getattr(analysis, key) != value}
        if changes:
            record_observation("query_analysis_normalization", "normalization", "accepted",
                               end_reason="derived_signals_from_validated_goals", changes=changes)
        return replace(analysis, goals=goals, **signals)
    # Legacy responses without goals cannot use a graph as the source of truth.
    # A single independent lookup is a valid lookup, not a reason to retry.
    if analysis.target_count == 1 and analysis.targets_independent:
        analysis = replace(analysis, targets_independent=False)
    if analysis.has_dependency and analysis.target_count < 2:
        raise QueryValidationError("dependency_target_count_conflict", "dependent retrieval requires at least two targets", field="target_count")
    if analysis.has_dependency and analysis.targets_independent:
        raise QueryValidationError("dependency_independence_conflict", "dependent retrieval targets cannot be independent", field="targets_independent")
    return analysis


def _parse_query_analysis(content: str, question: str = "") -> QueryAnalysis:
    return _validate_query_analysis(_decode_query_analysis(content, question), question)


def _rule_based_analysis(question: str) -> QueryAnalysis | None:
    # Fast paths are deliberately abstaining: a local cue is not proof that
    # the entire request has one goal.
    multi_facet = bool(COLLOQUIAL_MULTI_FACET_PATTERN.search(question))
    precise_locators = {
        "".join(match.group(0).split()).casefold()
        for match in PRECISE_LOCATOR_PATTERN.finditer(question)
    }
    if UNKNOWN_LOCATOR.search(question) and 2 <= len(precise_locators) <= MAX_GOALS:
        # The answer is the mapping, so retrieve the candidate objects without
        # attaching a requested fact to any one of them before evidence exists.
        goal = QueryGoal(id="locator_relation", text=question, source_spans=(question,),
                         candidate_locators=tuple(sorted(precise_locators)))
        return QueryAnalysis(intent="complex_lookup", need_rewrite=False, target_count=1,
                             targets_independent=False, has_dependency=False,
                             single_retrieval_likely_covers_all=False, source="rule",
                             reason="未知图表与事实的对应关系，先完整读取各候选对象再联合判定",
                             goals=(goal,), task_types=("lookup", "comparison"))
    if len(precise_locators) == 1 and not multi_facet:
        return QueryAnalysis(
            intent="precise_lookup",
            need_rewrite=False,
            target_count=1,
            targets_independent=False,
            has_dependency=False,
            single_retrieval_likely_covers_all=True,
            reason="包含单个明确的表、图、章节或页码定位信息",
            source="rule",
        )

    if (RATING_SEQUENTIAL_PATTERN.search(question) and not EXPLICIT_RATING_PATTERN.search(question)
            and not multi_facet and not metric_terms(question)):
        return QueryAnalysis(
            intent="complex_lookup",
            need_rewrite=False,
            target_count=2,
            targets_independent=False,
            has_dependency=True,
            single_retrieval_likely_covers_all=False,
            reason="评级含义查询依赖先取得研报给出的具体评级名称",
            source="rule",
        )

    has_colloquial_language = any(
        pattern.search(question) for pattern in COLLOQUIAL_PATTERNS
    )
    if multi_facet or RATING_SEQUENTIAL_PATTERN.search(question):
        return None

    if has_colloquial_language:
        return QueryAnalysis(
            intent="colloquial_lookup",
            need_rewrite=True,
            target_count=1,
            targets_independent=False,
            has_dependency=False,
            single_retrieval_likely_covers_all=True,
            reason="包含需要转换为文档术语的口语表达",
            source="rule",
        )

    if CONTEXTUAL_PATTERN.search(question):
        return QueryAnalysis(
            intent="contextual_query",
            need_rewrite=True,
            target_count=1,
            targets_independent=False,
            has_dependency=False,
            single_retrieval_likely_covers_all=True,
            reason="查询包含依赖上下文的指代或承接表达",
            source="rule",
        )

    return None


def analyze_query_intent(
    question: str,
    *,
    client: Any | None = None,
) -> QueryAnalysis:
    """Analyze query semantics without selecting a route or making a plan."""

    normalized_question = question.strip() if isinstance(question, str) else ""
    if not normalized_question:
        return QueryAnalysis(
            intent="invalid_query",
            need_rewrite=False,
            target_count=1,
            targets_independent=False,
            has_dependency=False,
            single_retrieval_likely_covers_all=True,
            reason="查询为空",
            source="rule",
        )

    if not _environment_flag("RAG_QUERY_INTENT_ENABLED", True):
        return QueryAnalysis(
            intent="general_lookup",
            need_rewrite=True,
            target_count=1,
            targets_independent=False,
            has_dependency=False,
            single_retrieval_likely_covers_all=True,
            reason="查询分析已关闭，沿用原有的全量改写策略",
            source="disabled",
        )

    if len(normalized_question) > MAX_QUERY_INTENT_INPUT_CHARS:
        return QueryAnalysis(
            intent="general_lookup",
            need_rewrite=False,
            target_count=1,
            targets_independent=False,
            has_dependency=False,
            single_retrieval_likely_covers_all=True,
            reason="查询过长，跳过查询分析和改写",
            source="fallback",
        )

    rule_analysis = _rule_based_analysis(normalized_question)
    if rule_analysis is not None:
        return replace(
            rule_analysis,
            need_expansion=rule_analysis.need_rewrite or (
                not precise_locator(normalized_question) and bool(alias_expansions(normalized_question))
            ),
            language_features=tuple(name for name, present in (
                ("colloquial", any(p.search(normalized_question) for p in COLLOQUIAL_PATTERNS)),
                ("contextual", bool(CONTEXTUAL_PATTERN.search(normalized_question))),
            ) if present),
        )

    try:
        resolved_client = client or get_query_intent_client()
        analysis = validated_completion(resolved_client, "query_analysis",
            parse=lambda content: _decode_query_analysis(content, normalized_question),
            validate=lambda value: _validate_query_analysis(value, normalized_question),
            model=query_intent_model(),
            messages=[
                {"role": "system", "content": QUERY_ANALYSIS_SYSTEM_PROMPT},
                {"role": "user", "content": normalized_question},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=1800,
            stream=False,
            timeout=query_intent_timeout_seconds(),
            extra_body={"enable_thinking": False},
        )
        logger.info(
            "RAG query analyzed: intent=%s targets=%s dependency=%s source=%s",
            analysis.intent,
            analysis.target_count,
            analysis.has_dependency,
            analysis.source,
        )
        return analysis
    except Exception as error:
        logger.warning("RAG query analysis failed: %s: %s", type(error).__name__, str(error)[:500])
        record_observation("query_analysis_fallback", "fallback", "applied",
                           source="rule", end_reason="conservative_single_query", **failure_details(error))

        has_colloquial_language = any(
            pattern.search(normalized_question)
            for pattern in COLLOQUIAL_PATTERNS
        )
        return QueryAnalysis(
            intent=(
                "colloquial_lookup"
                if has_colloquial_language
                else "general_lookup"
            ),
            need_rewrite=has_colloquial_language,
            target_count=1,
            targets_independent=False,
            has_dependency=False,
            single_retrieval_likely_covers_all=True,
            reason=(
                "查询分析失败；"
                "口语问题回退为受保真校验的单查询改写"
                if has_colloquial_language
                else "查询分析失败，保守使用原查询"
            ),
            source="fallback",
            need_expansion=has_colloquial_language or bool(alias_expansions(normalized_question)),
            validation_error=f"{type(error).__name__}: {str(error)[:500]}",
        )


def precise_locator(question: str) -> bool:
    return bool(PRECISE_LOCATOR_PATTERN.search(question))
