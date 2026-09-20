import logging
from types import SimpleNamespace

import pytest

from agent.orchestrator import AgentOrchestrator, MinimalAgent
from agent.schemas import ToolResult, ToolSource


def assistant_message(*, content="", tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls or [])


def tool_call(call_id, name, arguments):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def completion(message):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
    )


class FakeCompletions:
    def __init__(self, responses):
        self._responses = iter(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return next(self._responses)


class FakeTool:
    name = "search_knowledge_base"
    description = "搜索内部知识库"
    parameters = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }

    def __init__(self):
        self.calls = []

    def run(self, **kwargs):
        self.calls.append(kwargs)
        return ToolResult(
            tool_name=self.name,
            query=kwargs["query"],
            content="[1] 制度.pdf\n证据",
            sources=[
                ToolSource(
                    source_type="knowledge_base",
                    source_id="chunk-1",
                    title="制度.pdf",
                    content="证据",
                )
            ],
        )


class EmptyTool(FakeTool):
    def run(self, **kwargs):
        self.calls.append(kwargs)
        return ToolResult(
            tool_name=self.name,
            query=kwargs["query"],
            content="",
            sources=[],
            metadata={
                "evidence_sufficiency": {
                    "sufficient": False,
                    "source": "model",
                    "missing_requirements": ["审批时限"],
                }
            },
        )


class FakeWebTool(FakeTool):
    name = "search_web"
    description = "搜索互联网"

    def run(self, **kwargs):
        self.calls.append(kwargs)
        return ToolResult(
            tool_name=self.name,
            query=kwargs["query"],
            content="外部证据",
            sources=[
                ToolSource(
                    source_type="web",
                    source_id="web-1",
                    title="外部资料",
                    content="外部证据",
                    url="https://example.com/source",
                )
            ],
        )


def make_agent(responses, tool, max_steps=3):
    completions = FakeCompletions(responses)
    client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
    )
    agent = MinimalAgent(
        client=client,
        model="test-model",
        tools=[tool],
        max_steps=max_steps,
    )
    return agent, completions


@pytest.mark.unit
def test_agent_skips_tool_for_direct_question():
    tool = FakeTool()
    agent, completions = make_agent(
        [completion(assistant_message(content="READY"))],
        tool,
    )

    result = agent.run(question="你好")

    assert result.steps == 1
    assert result.stop_reason == "completed"
    assert result.tool_results == []
    assert tool.calls == []
    assert completions.calls[0]["tool_choice"] == "auto"
    assert completions.calls[0]["parallel_tool_calls"] is True
    assert completions.calls[0]["extra_body"] == {"enable_thinking": False}


@pytest.mark.unit
def test_agent_places_conversation_history_before_current_question():
    tool = FakeTool()
    agent, completions = make_agent(
        [completion(assistant_message(content="READY"))],
        tool,
    )
    history = [
        {"role": "user", "content": "介绍一下项目 A"},
        {"role": "assistant", "content": "项目 A 是一个检索系统。"},
        {"role": "system", "content": "这条非法历史消息应被忽略"},
        {"role": "assistant", "content": "   "},
    ]

    agent.run(
        question="它支持实时搜索吗？",
        conversation_history=history,
    )

    messages = completions.calls[0]["messages"]
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert messages[1]["content"] == "介绍一下项目 A"
    assert messages[2]["content"] == "项目 A 是一个检索系统。"
    assert "它支持实时搜索吗？" in messages[3]["content"]


@pytest.mark.unit
def test_agent_stops_after_successful_tool_returns_evidence():
    tool = FakeTool()
    agent, completions = make_agent(
        [
            completion(
                assistant_message(
                    tool_calls=[
                        tool_call(
                            "call-1",
                            "search_knowledge_base",
                            '{"query":"差旅制度","top_k":3}',
                        )
                    ]
                )
            ),
        ],
        tool,
    )

    result = agent.run(question="公司的差旅制度是什么？")

    assert result.steps == 1
    assert result.stop_reason == "evidence_ready"
    assert tool.calls == [{"query": "差旅制度", "top_k": 3}]
    assert result.tool_results[0].sources[0].source_id == "chunk-1"
    assert len(completions.calls) == 1


@pytest.mark.unit
def test_agent_executes_all_parallel_tools_before_stopping():
    rag_tool = FakeTool()
    web_tool = FakeWebTool()
    completions = FakeCompletions(
        [
            completion(
                assistant_message(
                    tool_calls=[
                        tool_call("call-rag", rag_tool.name, '{"query":"内部制度"}'),
                        tool_call("call-web", web_tool.name, '{"query":"最新政策"}'),
                    ]
                )
            )
        ]
    )
    agent = MinimalAgent(
        client=SimpleNamespace(
            chat=SimpleNamespace(completions=completions),
        ),
        model="test-model",
        tools=[rag_tool, web_tool],
    )

    result = agent.run(question="比较内部制度和最新政策")

    assert result.stop_reason == "evidence_ready"
    assert rag_tool.calls == [{"query": "内部制度"}]
    assert web_tool.calls == [{"query": "最新政策"}]
    assert [item.tool_name for item in result.tool_results] == [
        "search_knowledge_base",
        "search_web",
    ]
    assert len(completions.calls) == 1


@pytest.mark.unit
def test_agent_returns_invalid_json_as_tool_observation():
    tool = FakeTool()
    agent, _completions = make_agent(
        [
            completion(
                assistant_message(
                    tool_calls=[
                        tool_call(
                            "call-bad",
                            "search_knowledge_base",
                            "not-json",
                        )
                    ]
                )
            ),
            completion(assistant_message(content="READY")),
        ],
        tool,
    )

    result = agent.run(question="查询资料")

    assert tool.calls == []
    assert result.tool_results[0].success is False
    assert result.tool_results[0].metadata["error_type"] == "invalid_json"


@pytest.mark.unit
def test_agent_retries_empty_results_until_configured_step_limit():
    tool = EmptyTool()

    #创建agent, 并传入两个模拟的模型响应
    agent, completions = make_agent(
        [
            completion(
                assistant_message(
                    tool_calls=[
                        tool_call("call-1", tool.name, '{"query":"第一次"}')
                    ]
                )
            ),
            completion(
                assistant_message(
                    tool_calls=[
                        tool_call("call-2", tool.name, '{"query":"第二次"}')
                    ]
                )
            ),
        ],
        tool,
        max_steps=2,
    )

    result = agent.run(question="复杂问题")

    assert result.steps == 2
    assert result.stop_reason == "max_steps"
    assert len(result.tool_results) == 2
    assert len(completions.calls) == 2
    assert all(not result.sources for result in result.tool_results)


@pytest.mark.unit
def test_orchestrator_uses_small_planner_and_large_answer_model(caplog):
    tool = FakeTool()
    planner_completions = FakeCompletions(
        [
            completion(
                assistant_message(
                    tool_calls=[
                        tool_call(
                            "call-1",
                            tool.name,
                            '{"query":"差旅制度"}',
                        )
                    ]
                )
            ),
        ]
    )
    answer_stream = ["answer-chunk"]
    answer_completions = FakeCompletions([answer_stream])
    planner_client = SimpleNamespace(
        chat=SimpleNamespace(completions=planner_completions)
    )
    answer_client = SimpleNamespace(
        chat=SimpleNamespace(completions=answer_completions)
    )
    orchestrator = AgentOrchestrator(
        planner_client=planner_client,
        planner_model="small-planner",
        answer_client=answer_client,
        answer_model="large-answer",
        tools=[tool],
        fallback_tool=tool,
    )

    caplog.set_level(logging.INFO, logger="agent.orchestrator")
    execution = orchestrator.run(
        session_id="session-1",
        question="差旅制度是什么？",
        conversation_history=[{"role": "user", "content": "继续介绍"}],
    )

    assert execution.planning.steps == 1
    assert execution.planning.stop_reason == "evidence_ready"
    assert execution.answer_stream is answer_stream
    assert execution.response_documents[0]["document_id"] == "chunk-1"
    assert planner_completions.calls[0]["model"] == "small-planner"
    assert planner_completions.calls[0]["parallel_tool_calls"] is True
    assert len(planner_completions.calls) == 1
    assert answer_completions.calls[0]["model"] == "large-answer"
    assert answer_completions.calls[0]["stream"] is True
    assert "[1] 证据" in execution.answer_messages[-1]["content"]
    assert "session_id=session-1 stage=planner_model" in caplog.text
    assert "session_id=session-1 stage=tool" in caplog.text
    assert "session_id=session-1 stage=planning_total" in caplog.text
    assert "session_id=session-1 stage=answer_request" in caplog.text


@pytest.mark.unit
def test_orchestrator_falls_back_without_repeating_answer_generation():
    class FailingCompletions:
        def create(self, **_kwargs):
            raise RuntimeError("planner unavailable")

    fallback_tool = FakeTool()
    planner_client = SimpleNamespace(
        chat=SimpleNamespace(completions=FailingCompletions())
    )
    answer_stream = ["fallback-answer"]
    answer_completions = FakeCompletions([answer_stream])
    answer_client = SimpleNamespace(
        chat=SimpleNamespace(completions=answer_completions)
    )
    orchestrator = AgentOrchestrator(
        planner_client=planner_client,
        planner_model="small-planner",
        answer_client=answer_client,
        answer_model="large-answer",
        tools=[fallback_tool],
        fallback_tool=fallback_tool,
    )

    execution = orchestrator.run(
        session_id="session-1",
        question="内部制度是什么？",
    )

    assert execution.planning.stop_reason == "planner_fallback"
    assert fallback_tool.calls == [{"query": "内部制度是什么？"}]
    assert execution.answer_stream is answer_stream
    assert len(answer_completions.calls) == 1


@pytest.mark.unit
def test_orchestrator_returns_deterministic_refusal_for_insufficient_kb():
    tool = EmptyTool()
    planner_completions = FakeCompletions(
        [
            completion(
                assistant_message(
                    tool_calls=[
                        tool_call("call-1", tool.name, '{"query":"审批时限"}')
                    ]
                )
            ),
            completion(assistant_message(content="READY")),
        ]
    )
    answer_completions = FakeCompletions([])
    orchestrator = AgentOrchestrator(
        planner_client=SimpleNamespace(
            chat=SimpleNamespace(completions=planner_completions)
        ),
        planner_model="small-planner",
        answer_client=SimpleNamespace(
            chat=SimpleNamespace(completions=answer_completions)
        ),
        answer_model="large-answer",
        tools=[tool],
    )

    execution = orchestrator.run(
        session_id="session-1",
        question="审批时限是多久？",
    )

    assert execution.response_mode == "refuse"
    assert execution.answer_stream is None
    assert "审批时限" in execution.direct_answer
    assert answer_completions.calls == []


@pytest.mark.unit
def test_orchestrator_keeps_prompt_and_frontend_citation_order_identical():
    planner_client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions([]))
    )
    answer_completions = FakeCompletions([["answer"]])
    answer_client = SimpleNamespace(
        chat=SimpleNamespace(completions=answer_completions)
    )
    orchestrator = AgentOrchestrator(
        planner_client=planner_client,
        planner_model="small-planner",
        answer_client=answer_client,
        answer_model="large-answer",
        tools=[],
    )
    provided_evidence = [
        {
            "id": 1,
            "document_id": "doc-1",
            "document_name": "制度.pdf",
            "content_with_weight": "知识库证据",
            "source_type": "knowledge_base",
        }
    ]

    execution = orchestrator.run(
        session_id="session-1",
        question="请综合说明",
        session_context="会话文档证据",
        retrieved_content=provided_evidence,
    )

    assert [
        document["document_id"] for document in execution.response_documents
    ] == ["doc-1", "quick_parse_session-1_0"]
    answer_prompt = execution.answer_messages[-1]["content"]
    assert "[1] 知识库证据" in answer_prompt
    assert "[2] 会话文档证据" in answer_prompt
    assert execution.planning.stop_reason == "provided_evidence"
