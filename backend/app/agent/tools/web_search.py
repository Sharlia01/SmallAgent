"""Real-time web search exposed through the standard Agent tool protocol."""

import hashlib
import os
import re
from typing import Any, Callable
from urllib.parse import urlparse

import dashscope

from agent.schemas import ToolResult, ToolSource
from utils import logger


DEFAULT_WEB_SEARCH_MODEL = "qwen-plus"
MAX_WEB_SOURCES = 10
EMPTY_WEB_RESULT_MESSAGE = "实时网络搜索没有返回可验证的来源。"
FAILED_WEB_RESULT_MESSAGE = "实时网络搜索暂时不可用。"
PROVIDER_CITATION_PATTERN = re.compile(r"\s*\[ref_\d+\]", re.IGNORECASE)


def _value(container: Any, key: str, default=None):
    """Read one field from DashScope dict-like or attribute-style objects."""
    if container is None:
        return default
    if isinstance(container, dict):
        return container.get(key, default)
    get_value = getattr(container, "get", None)
    if callable(get_value):
        return get_value(key, default)
    return getattr(container, key, default)


def _stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def _safe_web_url(value: Any) -> str:
    url = str(value or "").strip()
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return url


def _escape_markdown_label(value: str) -> str:
    escaped = value.replace("\\", "\\\\")
    return escaped.replace("[", "\\[").replace("]", "\\]")


def _call_dashscope_web_search(
    *,
    api_key: str,
    model: str,
    query: str,
    native_base_url: str | None,
):
    """Call DashScope natively because the compatible API omits sources."""
    if native_base_url:
        dashscope.base_http_api_url = native_base_url

    return dashscope.Generation.call(
        api_key=api_key,
        model=model,
        messages=[{"role": "user", "content": query}],
        enable_search=True,
        search_options={
            "forced_search": True,
            "search_strategy": "turbo",
            "enable_source": True,
            "enable_citation": True,
            "citation_format": "[ref_<number>]",
        },
        result_format="message",
    )


def _extract_answer(output: Any) -> str:
    choices = _value(output, "choices", []) or []
    if not choices:
        return ""
    message = _value(choices[0], "message")
    answer = str(_value(message, "content", "") or "").strip()
    # DashScope's ref_N values identify entries inside one provider response.
    # They are not the document numbers used by our final-answer citation UI.
    return PROVIDER_CITATION_PATTERN.sub("", answer)


def _extract_sources(output: Any) -> list[ToolSource]:
    search_info = _value(output, "search_info", {}) or {}
    search_results = _value(search_info, "search_results", []) or []
    sources = []

    for position, result in enumerate(search_results[:MAX_WEB_SOURCES], start=1):
        url = _safe_web_url(_value(result, "url", ""))
        title = str(_value(result, "title", "") or "").strip()
        site_name = str(
            _value(result, "site_name", _value(result, "siteName", "")) or ""
        ).strip()
        source_index = _value(result, "index", position)

        if not url or not title:
            continue

        sources.append(
            ToolSource(
                source_type="web",
                source_id=_stable_id("web", url),
                title=title,
                content=(
                    f"来源网站：{site_name}\n搜索结果标题：{title}"
                    if site_name
                    else f"搜索结果标题：{title}"
                ),
                url=url,
                metadata={
                    "site_name": site_name,
                    "icon": _value(result, "icon", "") or "",
                    "index": source_index,
                },
            )
        )

    return sources


def _format_web_content(answer: str, sources: list[ToolSource]) -> str:
    sections = []
    if answer:
        sections.append(f"**实时网络搜索摘要：**\n{answer}")

    source_lines = []
    for position, source in enumerate(sources, start=1):
        site_name = source.metadata.get("site_name")
        site_suffix = f"（{site_name}）" if site_name else ""
        source_lines.append(
            f"{position}. "
            f"[{_escape_markdown_label(source.title)}]({source.url}){site_suffix}"
        )
    if source_lines:
        sections.append("**网络来源：**\n" + "\n".join(source_lines))

    return "\n\n".join(sections)


class WebSearchTool:
    """Search the public web and return a cited, provider-neutral result."""

    name = "search_web"
    description = (
        "搜索实时互联网信息，适合最新新闻、政策变化、天气、价格、市场数据"
        "以及明确要求联网查询的问题。不要用于仅依赖内部文档的问题。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "包含时间、地点和主题等必要限定条件的网络搜索词。",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        native_base_url: str | None = None,
        search_callable: Callable[..., Any] | None = None,
    ):
        self._api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "")
        self._model = model or os.getenv(
            "WEB_SEARCH_MODEL",
            DEFAULT_WEB_SEARCH_MODEL,
        )
        self._native_base_url = native_base_url or os.getenv(
            "DASHSCOPE_NATIVE_BASE_URL"
        )
        self._search_callable = search_callable or _call_dashscope_web_search

    def run(self, *, query: str) -> ToolResult:
        """Force a real web search and preserve its source links."""
        normalized_query = query.strip() if isinstance(query, str) else ""
        if not normalized_query:
            return ToolResult.failure(
                tool_name=self.name,
                query=normalized_query,
                error="query 不能为空。",
                metadata={"error_type": "validation_error"},
            )

        if not self._api_key:
            return ToolResult.failure(
                tool_name=self.name,
                query=normalized_query,
                error="实时网络搜索未配置 API Key。",
                metadata={"error_type": "configuration_error"},
            )

        try:
            response = self._search_callable(
                api_key=self._api_key,
                model=self._model,
                query=normalized_query,
                native_base_url=self._native_base_url,
            )
            status_code = _value(response, "status_code", 200)
            if status_code != 200:
                raise RuntimeError(
                    str(_value(response, "message", "DashScope search failed"))
                )

            output = _value(response, "output", {}) or {}
            answer = _extract_answer(output)
            sources = _extract_sources(output)
            request_id = str(_value(response, "request_id", "") or "")
        except Exception as error:
            logger.exception("Agent Web 搜索失败: query=%s", normalized_query)
            return ToolResult.failure(
                tool_name=self.name,
                query=normalized_query,
                error=FAILED_WEB_RESULT_MESSAGE,
                metadata={"error_type": type(error).__name__},
            )

        if not sources:
            return ToolResult(
                tool_name=self.name,
                query=normalized_query,
                content=answer,
                success=False,
                error=EMPTY_WEB_RESULT_MESSAGE,
                metadata={
                    "error_type": "empty_search_results",
                    "model": self._model,
                    "request_id": request_id,
                },
            )

        document_key = request_id or normalized_query
        return ToolResult(
            tool_name=self.name,
            query=normalized_query,
            content=_format_web_content(answer, sources),
            sources=sources,
            metadata={
                "result_count": len(sources),
                "model": self._model,
                "request_id": request_id,
                "document_id": _stable_id("web-search", document_key),
                "source_type": "web",
            },
        )


def search_web(query: str) -> ToolResult:
    """Functional entry point for callers that do not manage tool instances."""
    return WebSearchTool().run(query=query)
