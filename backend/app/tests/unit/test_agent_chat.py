from types import SimpleNamespace

import pytest

from agent.orchestrator import (
    AgentExecution,
    AgentRun,
    tool_results_to_retrieved_content,
)
from agent.schemas import ToolResult, ToolSource
from service.core import chat


@pytest.mark.unit
def test_generate_session_name_uses_first_question_without_model_call():
    assert (
        chat.generate_session_name("  今天杭州天气怎么样？\n请简短回答  ")
        == "今天杭州天气怎么样？ 请简短回答"
    )
    assert chat.generate_session_name("问题" * 20) == ("问题" * 15)
    assert chat.generate_session_name("   ") == "新对话"


@pytest.mark.unit
def test_chat_builds_two_model_orchestrator_with_rag_and_web_tools(monkeypatch):
    captured = {}

    class FakeOrchestrator:
        def __init__(self, *, tools, **kwargs):
            captured["tool_names"] = [tool.name for tool in tools]
            captured.update(kwargs)

    monkeypatch.setattr(chat, "OpenAI", lambda **_kwargs: object())
    monkeypatch.setattr(chat, "AgentOrchestrator", FakeOrchestrator)

    orchestrator = chat._create_agent_orchestrator(42)

    assert isinstance(orchestrator, FakeOrchestrator)
    assert captured["tool_names"] == ["search_knowledge_base", "search_web"]
    assert captured["planner_model"] == chat.AGENT_MODEL
    assert captured["answer_model"] == chat.CHAT_MODEL
    assert captured["planner_client"] is captured["answer_client"]
    assert captured["fallback_tool"].name == "search_knowledge_base"


@pytest.mark.unit
def test_load_conversation_history_is_user_scoped_and_chronological(monkeypatch):
    class FakeResult:
        def fetchall(self):
            # Database query returns newest turns first.
            return [
                SimpleNamespace(user_question="后一个问题", model_answer="后一个回答"),
                SimpleNamespace(user_question="前一个问题", model_answer="前一个回答"),
            ]

    class FakeDatabase:
        def __init__(self):
            self.statement = None
            self.params = None
            self.closed = False

        def execute(self, statement, params):
            self.statement = str(statement)
            self.params = params
            return FakeResult()

        def close(self):
            self.closed = True

    database = FakeDatabase()
    monkeypatch.setattr(chat, "get_db", lambda: iter([database]))

    history = chat.load_conversation_history("session-1", 42)

    assert history == [
        {"role": "user", "content": "前一个问题"},
        {"role": "assistant", "content": "前一个回答"},
        {"role": "user", "content": "后一个问题"},
        {"role": "assistant", "content": "后一个回答"},
    ]
    assert "s.user_id = :user_id" in database.statement
    assert "ORDER BY m.created_at DESC" in database.statement
    assert database.params == {
        "session_id": "session-1",
        "user_id": 42,
        "max_turns": chat.CHAT_HISTORY_MAX_TURNS,
    }
    assert database.closed is True


@pytest.mark.unit
def test_load_conversation_history_truncates_oversized_latest_turn(monkeypatch):
    class FakeResult:
        def fetchall(self):
            return [
                SimpleNamespace(user_question="问" * 30, model_answer="答" * 50),
                SimpleNamespace(user_question="旧问题", model_answer="旧回答"),
            ]

    class FakeDatabase:
        def execute(self, _statement, _params):
            return FakeResult()

        def close(self):
            pass

    monkeypatch.setattr(chat, "get_db", lambda: iter([FakeDatabase()]))
    monkeypatch.setattr(chat, "CHAT_HISTORY_MAX_CHARS", 20)

    history = chat.load_conversation_history("session-1", 42)

    assert [message["role"] for message in history] == ["user", "assistant"]
    assert sum(len(message["content"]) for message in history) == 20
    assert all("旧" not in message["content"] for message in history)


@pytest.mark.unit
def test_tool_results_are_deduplicated_and_adapted_for_existing_chat():
    source = ToolSource(
        source_type="knowledge_base",
        source_id="chunk-1",
        title="制度.pdf",
        content="差旅需要审批。",
        score=0.9,
        metadata={
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "positions": [[1, 2]],
            "kb_id": "42",
        },
    )
    results = [
        ToolResult(
            tool_name="search_knowledge_base",
            query="差旅",
            content="证据",
            sources=[source],
        ),
        ToolResult(
            tool_name="search_knowledge_base",
            query="差旅审批",
            content="重复证据",
            sources=[source],
        ),
    ]

    references = tool_results_to_retrieved_content(results)

    assert references == [
        {
            "id": 1,
            "rank": 1,
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "document_name": "制度.pdf",
            "content_with_weight": "差旅需要审批。",
            "similarity": 0.9,
            "rerank_score": None,
            "rrf_score": None,
            "vector_similarity": None,
            "term_similarity": None,
            "retrieval_ranks": {},
            "retrieval_scores": {},
            "rrf_contributions": {},
            "positions": [[1, 2]],
            "kb_id": "42",
            "image_id": "",
            "source_type": "knowledge_base",
        }
    ]


@pytest.mark.unit
def test_web_tool_result_becomes_one_cited_web_document():
    web_sources = [
        ToolSource(
            source_type="web",
            source_id="web-1",
            title="最新政策",
            content="搜索结果标题：最新政策",
            url="https://example.com/policy",
            metadata={"site_name": "政府网站", "index": 1},
        ),
        ToolSource(
            source_type="web",
            source_id="web-2",
            title="政策解读",
            content="搜索结果标题：政策解读",
            url="https://example.com/explanation",
            metadata={"site_name": "权威媒体", "index": 2},
        ),
    ]
    result = ToolResult(
        tool_name="search_web",
        query="最新政策",
        content="实时摘要与来源链接",
        sources=web_sources,
        metadata={"document_id": "web-search-1"},
    )

    references = tool_results_to_retrieved_content([result, result])

    assert len(references) == 1
    assert references[0]["document_id"] == "web-search-1"
    assert references[0]["document_name"] == "实时网络搜索：最新政策"
    assert references[0]["source_type"] == "web"
    assert references[0]["content_with_weight"] == "实时摘要与来源链接"
    assert len(references[0]["web_sources"]) == 2
    assert references[0]["web_sources"][0]["url"] == (
        "https://example.com/policy"
    )


@pytest.mark.unit
def test_chat_uses_agent_evidence_in_stream_and_persistence(monkeypatch):
    source = ToolSource(
        source_type="knowledge_base",
        source_id="chunk-1",
        title="制度.pdf",
        content="差旅需要审批。",
        metadata={"document_id": "doc-1", "chunk_id": "chunk-1"},
    )
    agent_run = AgentRun(
        tool_results=[
            ToolResult(
                tool_name="search_knowledge_base",
                query="差旅",
                content="证据",
                sources=[source],
            )
        ],
        steps=2,
    )
    saved = {}
    history = [
        {"role": "user", "content": "差旅制度适用于谁？"},
        {"role": "assistant", "content": "适用于全体员工。"},
    ]

    monkeypatch.setattr(
        chat,
        "load_conversation_history",
        lambda _session_id, _user_id: history,
    )
    monkeypatch.setattr(chat, "get_quick_parse_content", lambda _session_id: None)

    answer_stream = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason=None,
                    delta=SimpleNamespace(
                        content="需要审批。",
                        reasoning_content=None,
                    ),
                )
            ]
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    delta=SimpleNamespace(
                        content=None,
                        reasoning_content=None,
                    ),
                )
            ]
        ),
    ]
    references = tool_results_to_retrieved_content(agent_run.tool_results)
    execution = AgentExecution(
        planning=agent_run,
        retrieved_content=references,
        response_documents=references,
        answer_messages=[{"role": "user", "content": "answer prompt"}],
        answer_stream=answer_stream,
    )

    class FakeOrchestrator:
        def run(self, **kwargs):
            saved["orchestrator_args"] = kwargs
            return execution

    monkeypatch.setattr(
        chat,
        "_create_agent_orchestrator",
        lambda user_id: saved.update({"orchestrator_user_id": user_id})
        or FakeOrchestrator(),
    )
    monkeypatch.setattr(
        chat,
        "_generate_recommended_questions_safely",
        lambda *_args: [],
    )
    monkeypatch.setattr(
        chat,
        "write_chat_to_db",
        lambda *args: saved.update({"write_args": args}),
    )
    monkeypatch.setattr(
        chat,
        "update_session_name",
        lambda *args: saved.update({"name_args": args}),
    )

    events = list(
        chat.get_chat_completion(
            "session-1",
            "差旅如何审批？",
            user_id=42,
        )
    )

    assert any('"documents"' in event and "制度.pdf" in event for event in events)
    assert any("需要审批。" in event for event in events)
    assert events[-1] == "event: end\ndata: [DONE]\n\n"
    assert saved["write_args"][2] == "需要审批。"
    assert saved["write_args"][3][0]["document_id"] == "doc-1"
    assert saved["name_args"] == ("session-1", "差旅如何审批？", 42)
    assert saved["orchestrator_user_id"] == 42
    assert saved["orchestrator_args"]["conversation_history"] == history
    assert saved["orchestrator_args"]["question"] == "差旅如何审批？"


@pytest.mark.unit
def test_chat_streams_and_persists_direct_refusal(monkeypatch):
    saved = {}
    execution = AgentExecution(
        planning=AgentRun(steps=1, stop_reason="completed"),
        retrieved_content=[],
        response_documents=[],
        answer_messages=[],
        answer_stream=None,
        direct_answer="当前资料不足以可靠回答这个问题。",
        response_mode="refuse",
        evidence_sufficiency={"sufficient": False, "source": "model"},
    )

    class FakeOrchestrator:
        def run(self, **_kwargs):
            return execution

    monkeypatch.setattr(chat, "load_conversation_history", lambda *_args: [])
    monkeypatch.setattr(chat, "get_quick_parse_content", lambda *_args: None)
    monkeypatch.setattr(
        chat,
        "_create_agent_orchestrator",
        lambda *_args: FakeOrchestrator(),
    )
    monkeypatch.setattr(
        chat,
        "_generate_recommended_questions_safely",
        lambda *_args: pytest.fail("CLI 模式不应生成推荐问题"),
    )
    monkeypatch.setattr(
        chat,
        "write_chat_to_db",
        lambda *args: saved.update({"write_args": args}),
    )
    monkeypatch.setattr(chat, "update_session_name", lambda *_args: None)

    events = list(
        chat.get_chat_completion(
            "session-1",
            "未知资料是什么？",
            user_id=42,
            include_recommended_questions=False,
        )
    )

    assert any('"response_mode": "refuse"' in event for event in events)
    assert any("当前资料不足" in event for event in events)
    assert not any("recommended_questions" in event for event in events)
    assert saved["write_args"][2] == "当前资料不足以可靠回答这个问题。"
    assert saved["write_args"][4] == []
