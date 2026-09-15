"""LLM-backed query normalization shared by every retrieval caller."""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_QUERY_REWRITE_MODEL = "qwen3.7-flash-2026-07-15"
DEFAULT_QUERY_REWRITE_TIMEOUT_SECONDS = 10.0
MAX_QUERY_REWRITE_INPUT_CHARS = 2000
MAX_QUERY_REWRITE_OUTPUT_CHARS = 300

QUERY_REWRITE_SYSTEM_PROMPT = """
你是知识库检索查询规范化器。请把用户问题改写成适合文档检索的一条正式查询语句。

要求：
1. 将口语表达替换为常见书面语、专业术语，以及更可能出现在原文中的字段名称。
2. 完整保留公司、人名、时间、数字、单位、否定词和限定条件，不改变用户意图。
3. 可以补充口语表达对应的通用术语，但不得回答问题、猜测事实或生成具体数值。
4. 已经适合检索的问题保持原意，不做无关扩展。
5. 只返回 JSON 对象，格式为 {"query": "改写后的查询"}。

示例：
用户问题：这家公司今年卖货赚了多少，和去年比怎么样？
输出：{"query":"该公司本年度营业收入、归母净利润及同比变化"}

用户问题：这家公司接下来几年能赚多少，股票贵不贵？
输出：{"query":"该公司未来几年盈利预测、归母净利润、每股收益 EPS 和市盈率 PE 估值"}
""".strip()


def _environment_flag(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().casefold() not in {"0", "false", "no", "off"}


def _rewrite_timeout_seconds() -> float:
    raw_value = os.getenv(
        "QUERY_REWRITE_TIMEOUT_SECONDS",
        str(DEFAULT_QUERY_REWRITE_TIMEOUT_SECONDS),
    )
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return DEFAULT_QUERY_REWRITE_TIMEOUT_SECONDS
    return value if value > 0 else DEFAULT_QUERY_REWRITE_TIMEOUT_SECONDS


@lru_cache(maxsize=1)
def get_query_rewrite_client() -> OpenAI:
    api_key = os.getenv("DASHSCOPE_API_KEY", "").strip()
    base_url = os.getenv("DASHSCOPE_BASE_URL", "").strip()
    if not api_key or not base_url:
        raise RuntimeError(
            "DASHSCOPE_API_KEY and DASHSCOPE_BASE_URL are required for query rewrite"
        )
    return OpenAI(api_key=api_key, base_url=base_url)


def _response_content(completion: Any) -> str:
    choices = getattr(completion, "choices", None) or []
    if not choices:
        raise ValueError("query rewrite model returned no choices")
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise ValueError("query rewrite model returned empty content")
    return content.strip()


def _parse_rewritten_query(content: str) -> str:
    payload = json.loads(content)
    rewritten_query = payload.get("query") if isinstance(payload, dict) else None
    if not isinstance(rewritten_query, str):
        raise ValueError("query rewrite response must contain a string query")

    rewritten_query = " ".join(rewritten_query.split()).strip()
    if not rewritten_query:
        raise ValueError("rewritten query is empty")
    if len(rewritten_query) > MAX_QUERY_REWRITE_OUTPUT_CHARS:
        raise ValueError("rewritten query is too long")
    return rewritten_query


def rewrite_query(question: str, *, client: Any | None = None) -> str:
    """Return one retrieval-oriented rewrite, falling back to the input.

    Query rewriting is an optional quality enhancement: missing credentials,
    provider failures, malformed output, and oversized inputs must never make
    knowledge-base retrieval unavailable.
    """
    original_question = question.strip() if isinstance(question, str) else ""
    if not original_question:
        return original_question
    if not _environment_flag("QUERY_REWRITE_ENABLED", True):
        return original_question

    # Skip query rewriting for inputs that are too long
    if len(original_question) > MAX_QUERY_REWRITE_INPUT_CHARS:
        logger.info(
            "Skipping query rewrite because input is too long: chars=%s",
            len(original_question),
        )
        return original_question

    try:
        resolved_client = client or get_query_rewrite_client()
        completion = resolved_client.chat.completions.create(
            model=os.getenv(
                "QUERY_REWRITE_MODEL",
                DEFAULT_QUERY_REWRITE_MODEL,
            ),
            messages=[
                {"role": "system", "content": QUERY_REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": original_question},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=128,
            stream=False,
            timeout=_rewrite_timeout_seconds(),
            extra_body={"enable_thinking": False},
        )
        rewritten_query = _parse_rewritten_query(
            _response_content(completion)
        )
        logger.info(
            "Query rewrite completed: changed=%s original_chars=%s rewritten_chars=%s",
            rewritten_query != original_question,
            len(original_question),
            len(rewritten_query),
        )
        return rewritten_query
    except Exception as error:
        logger.warning(
            "Query rewrite failed; using original question: %s",
            type(error).__name__,
        )
        return original_question
