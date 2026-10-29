"""回答流必须在下一次评分请求前归还连接，不能依赖垃圾回收。"""

import gc
import json
from types import SimpleNamespace

import httpx
from openai import OpenAI
import pytest

from agent.loop import AgentRun
from agent.orchestrator import AgentExecution
from service.core.answering import chat
from service.core.answering.streaming import collect_answer_stream


pytestmark = pytest.mark.unit


def chunk(text="", *, finish=None, thinking=""):
    return SimpleNamespace(choices=[SimpleNamespace(
        delta=SimpleNamespace(content=text, reasoning_content=thinking), finish_reason=finish,
    )])


class TrackedStream:
    def __init__(self, chunks, *, error=None, close_error=None):
        self.chunks = chunks
        self.error = error
        self.close_error = close_error
        self.close_count = 0

    def __iter__(self):
        yield from self.chunks
        if self.error:
            raise self.error

    def close(self):
        self.close_count += 1
        if self.close_error:
            raise self.close_error


@pytest.mark.parametrize("finish", [None, "stop"])
def test_normal_or_early_finish_closes_stream_and_keeps_final_text(finish):
    stream = TrackedStream([chunk("结尾正文", finish=finish, thinking="推理内容")],
                           error=AssertionError("结束事件后不能继续读取") if finish else None)
    result = collect_answer_stream(stream)
    assert (result.answer, result.thinking, result.chunk_count) == ("结尾正文", "推理内容", 1)
    assert stream.close_count == 1


def test_read_error_closes_stream_without_masking_original_exception(caplog):
    original = TimeoutError("响应读取超时")
    stream = TrackedStream([chunk("部分内容")], error=original, close_error=OSError("连接关闭失败"))
    with pytest.raises(TimeoutError) as caught:
        collect_answer_stream(stream)
    assert caught.value is original
    assert stream.close_count == 1
    assert "关闭回答模型响应流失败" in caplog.text


def test_empty_stream_is_closed_and_plain_iterables_remain_supported():
    stream = TrackedStream([])
    assert collect_answer_stream(stream).answer == ""
    assert stream.close_count == 1
    assert collect_answer_stream([chunk("列表响应")]).answer == "列表响应"


def test_real_sdk_stream_is_closed_before_reusing_client_for_judges():
    bodies = []
    judged = []

    class AnswerBody(httpx.SyncByteStream):
        closed = False

        def __iter__(self):
            for text, finish in [("回答", None), ("结束", "stop")]:
                value = {"id": "answer", "object": "chat.completion.chunk", "created": 0,
                         "model": "answer", "choices": [{"index": 0, "delta": {"content": text},
                                                           "finish_reason": finish}]}
                yield ("data: " + json.dumps(value, ensure_ascii=False) + "\n\n").encode()
            # collect_answer_stream 在 finish_reason 退出时，底层还没读到这个结束事件。
            yield b"data: [DONE]\n\n"

        def close(self):
            self.closed = True

    def handler(request):
        payload = json.loads(request.content)
        if payload.get("stream"):
            body = AnswerBody()
            bodies.append(body)
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=body)
        assert bodies and all(body.closed for body in bodies), "评分前必须显式关闭回答响应"
        judged.append(payload["model"])
        return httpx.Response(200, json={
            "id": "judge", "object": "chat.completion", "created": 0, "model": "judge",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "{}"},
                         "finish_reason": "stop"}],
        })

    with OpenAI(api_key="test", base_url="https://model.example/v1", max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(handler))) as client:
        for _ in range(3):
            stream = client.chat.completions.create(model="answer", messages=[], stream=True)
            assert collect_answer_stream(stream).answer == "回答结束"
            assert stream.response.is_closed
            # 连续样本共用客户端，随后再发忠实度、引用评分请求。
            gc.collect()
            for model in ("faithfulness", "citation"):
                client.chat.completions.create(model=model, messages=[], stream=False)
    assert len(bodies) == 3 and judged == ["faithfulness", "citation"] * 3


@pytest.mark.parametrize("exit_at", ["documents", "partial_answer", "finished", "read_error"])
def test_online_chat_releases_stream_on_finish_disconnect_or_error(monkeypatch, exit_at):
    stream = TrackedStream(
        [chunk("第一段"), chunk("结尾", finish="stop")],
        error=TimeoutError("读取失败") if exit_at == "read_error" else None,
    )
    if exit_at == "read_error":
        stream.chunks = [chunk("第一段")]
    execution = AgentExecution(agent_run=AgentRun(), answer_stream=stream)
    monkeypatch.setattr(chat, "load_conversation_history", lambda *_args: [])
    monkeypatch.setattr(chat, "get_quick_parse_content", lambda *_args: None)
    monkeypatch.setattr(chat, "_create_agent_orchestrator", lambda _user: SimpleNamespace(run=lambda **_kwargs: execution))
    monkeypatch.setattr(chat, "update_session_name", lambda *_args: None)
    saved = []
    monkeypatch.setattr(chat, "write_chat_to_db", lambda *args: saved.append(args))
    response = chat.get_chat_completion("test", "用户问题", user_id=1, include_recommended_questions=False)
    assert "documents" in next(response)

    if exit_at in {"documents", "partial_answer"}:
        if exit_at == "partial_answer":
            assert "第一段" in next(response)
        response.close()
        assert not saved
    else:
        events = list(response)
        if exit_at == "finished":
            assert saved[0][2] == "第一段结尾"
        else:
            assert any("event: error" in event for event in events)
            assert not saved
    assert stream.close_count == 1
