from types import SimpleNamespace

import pytest

from agent.tools import web_search


def successful_response():
    return SimpleNamespace(
        status_code=200,
        request_id="request-1",
        output=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="杭州明天有雨[ref_1]。",
                    )
                )
            ],
            search_info={
                "search_results": [
                    {
                        "index": 1,
                        "title": "杭州天气预报",
                        "url": "https://example.com/weather",
                        "site_name": "示例天气网",
                        "icon": "https://example.com/favicon.ico",
                    }
                ]
            },
        ),
    )


@pytest.mark.unit
def test_web_search_returns_cited_standard_result():
    calls = []

    def fake_search(**kwargs):
        calls.append(kwargs)
        return successful_response()

    tool = web_search.WebSearchTool(
        api_key="test-key",
        model="qwen-plus",
        native_base_url="https://example.com/api/v1",
        search_callable=fake_search,
    )

    result = tool.run(query="  杭州明天天气  ")

    assert result.success is True
    assert result.query == "杭州明天天气"
    assert calls == [
        {
            "api_key": "test-key",
            "model": "qwen-plus",
            "query": "杭州明天天气",
            "native_base_url": "https://example.com/api/v1",
        }
    ]
    assert "杭州明天有雨。" in result.content
    assert "ref_1" not in result.content
    assert "1. [杭州天气预报]" in result.content
    assert "[杭州天气预报](https://example.com/weather)" in result.content
    assert result.sources[0].source_type == "web"
    assert result.sources[0].url == "https://example.com/weather"
    assert result.sources[0].metadata["site_name"] == "示例天气网"
    assert result.metadata["result_count"] == 1
    assert result.metadata["document_id"].startswith("web-search-")


@pytest.mark.unit
def test_web_search_requires_api_key(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    result = web_search.WebSearchTool(api_key="").run(query="最新消息")

    assert result.success is False
    assert result.metadata["error_type"] == "configuration_error"


@pytest.mark.unit
def test_web_search_rejects_empty_query():
    result = web_search.WebSearchTool(api_key="test-key").run(query="   ")

    assert result.success is False
    assert result.metadata["error_type"] == "validation_error"


@pytest.mark.unit
def test_web_search_requires_verifiable_sources():
    response = successful_response()
    response.output.search_info = {"search_results": []}
    tool = web_search.WebSearchTool(
        api_key="test-key",
        search_callable=lambda **_kwargs: response,
    )

    result = tool.run(query="最新消息")

    assert result.success is False
    assert result.content == "杭州明天有雨。"
    assert result.metadata["error_type"] == "empty_search_results"


@pytest.mark.unit
def test_web_search_ignores_unsafe_source_urls():
    response = successful_response()
    response.output.search_info["search_results"][0]["url"] = (
        "javascript:alert(1)"
    )
    tool = web_search.WebSearchTool(
        api_key="test-key",
        search_callable=lambda **_kwargs: response,
    )

    result = tool.run(query="最新消息")

    assert result.success is False
    assert result.sources == []
    assert result.metadata["error_type"] == "empty_search_results"


@pytest.mark.unit
def test_web_search_converts_provider_error_into_failure():
    def fail_search(**_kwargs):
        raise TimeoutError("provider timeout")

    tool = web_search.WebSearchTool(
        api_key="test-key",
        search_callable=fail_search,
    )

    result = tool.run(query="最新消息")

    assert result.success is False
    assert result.error == web_search.FAILED_WEB_RESULT_MESSAGE
    assert result.metadata["error_type"] == "TimeoutError"
