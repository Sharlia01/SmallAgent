from io import StringIO
import json

import httpx
import pytest

from rag_cli import (
    CliError,
    InteractiveCli,
    RagApiClient,
    format_source,
    iter_sse_events,
    normalize_terminal_text,
)


@pytest.mark.unit
def test_iter_sse_events_supports_comments_and_multiline_data():
    events = list(
        iter_sse_events(
            [
                ": keepalive",
                "event: message",
                'data: {"first":',
                'data: "value"}',
                "",
                "event: end",
                "data: [DONE]",
            ]
        )
    )

    assert events[0].event == "message"
    assert events[0].data == '{"first":\n"value"}'
    assert events[1].event == "end"
    assert events[1].data == "[DONE]"


@pytest.mark.unit
def test_api_client_login_and_create_session_send_bearer_token():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/login":
            assert json.loads(request.content) == {
                "username": "alice",
                "password": "secret",
            }
            return httpx.Response(200, json={"access_token": "jwt-token"})
        assert request.url.path == "/create_session"
        assert request.headers["Authorization"] == "Bearer jwt-token"
        return httpx.Response(200, json={"session_id": "session-1"})

    http_client = httpx.Client(
        base_url="http://testserver",
        transport=httpx.MockTransport(handler),
    )
    api = RagApiClient("http://testserver", client=http_client)

    api.login("alice", "secret")
    session_id = api.create_session()

    assert session_id == "session-1"
    assert len(requests) == 2
    http_client.close()


@pytest.mark.unit
def test_stream_chat_decodes_documents_answer_and_refusal_mode():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/chat_on_docs"
        assert request.url.params["session_id"] == "session-1"
        assert request.url.params["include_recommended_questions"] == "false"
        assert request.headers["Authorization"] == "Bearer jwt-token"
        assert json.loads(request.content) == {"message": "未知问题"}
        body = (
            "event: message\n"
            'data: {"documents": []}\n\n'
            "event: message\n"
            'data: {"role":"assistant","content":"证据不足。",'
            '"thinking":false,"response_mode":"refuse"}\n\n'
            "event: end\n"
            "data: [DONE]\n\n"
        )
        return httpx.Response(
            200,
            text=body,
            headers={"content-type": "text/event-stream"},
        )

    http_client = httpx.Client(
        base_url="http://testserver",
        transport=httpx.MockTransport(handler),
    )
    api = RagApiClient("http://testserver", client=http_client)
    api.token = "jwt-token"

    payloads = list(api.stream_chat("session-1", "未知问题"))

    assert payloads == [
        {"documents": []},
        {
            "role": "assistant",
            "content": "证据不足。",
            "thinking": False,
            "response_mode": "refuse",
        },
    ]
    http_client.close()


@pytest.mark.unit
def test_stream_chat_turns_sse_error_into_cli_error():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=(
                "event: error\n"
                'data: {"role":"error","content":"模型不可用"}\n\n'
            ),
        )

    http_client = httpx.Client(
        base_url="http://testserver",
        transport=httpx.MockTransport(handler),
    )
    api = RagApiClient("http://testserver", client=http_client)
    api.token = "jwt-token"

    with pytest.raises(CliError, match="模型不可用"):
        list(api.stream_chat("session-1", "问题"))
    http_client.close()


@pytest.mark.unit
def test_stream_chat_replaces_invalid_terminal_surrogate_before_json_encoding():
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content) == {"message": "杭州�天气"}
        return httpx.Response(
            200,
            text="event: end\ndata: [DONE]\n\n",
        )

    http_client = httpx.Client(
        base_url="http://testserver",
        transport=httpx.MockTransport(handler),
    )
    api = RagApiClient("http://testserver", client=http_client)
    api.token = "jwt-token"

    assert list(api.stream_chat("session-1", "杭州\udcff天气")) == []
    assert normalize_terminal_text("正常中文") == "正常中文"
    surrogate_escaped_chinese = "今天".encode("utf-8").decode(
        "ascii",
        errors="surrogateescape",
    )
    assert normalize_terminal_text(surrogate_escaped_chinese) == "今天"
    http_client.close()


@pytest.mark.unit
def test_interactive_cli_renders_stream_and_keeps_source_order():
    class FakeApi:
        def stream_chat(self, session_id, question):
            assert session_id == "session-1"
            assert question == "年收入是多少？"
            yield {
                "documents": [
                    {
                        "document_name": "报告.pdf",
                        "source_type": "knowledge_base",
                        "page_num_int": [3],
                        "chunk_id": "chunk-3",
                    }
                ]
            }
            yield {
                "role": "assistant",
                "content": "收入为 10 亿元。##1$$",
                "thinking": False,
            }
            yield {"recommended_questions": ["利润是多少？"]}

    output = StringIO()
    cli = InteractiveCli(FakeApi(), session_id="session-1", output=output)

    cli.ask("年收入是多少？")

    rendered = output.getvalue()
    assert "收入为 10 亿元。##1$$" in rendered
    assert "[1] 报告.pdf（knowledge_base，页码 3，chunk chunk-3）" in rendered
    assert "1. 利润是多少？" in rendered


@pytest.mark.unit
def test_format_source_includes_web_url():
    rendered = format_source(
        {
            "document_name": "实时搜索",
            "source_type": "web",
            "url": "https://example.com/source",
        },
        2,
    )

    assert rendered.startswith("  [2] 实时搜索（web）")
    assert "https://example.com/source" in rendered
