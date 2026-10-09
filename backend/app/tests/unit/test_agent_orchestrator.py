from tests.evidence_helpers import make_evidence_record, make_tool_result, make_query
from service.core.evidence_adapter import collection_from_records
from tests.evidence_helpers import bind_assessment
from service.core.evidence_adapter import assessment_from_record
from service.core.query_contract import QueryGoal
import logging
from dataclasses import replace
from types import SimpleNamespace

import pytest

from agent.orchestrator import AgentOrchestrator, AgentRun, MinimalAgent
from agent.schemas import ToolResult
from domain.assessment import EvidenceAssessment


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
        return make_tool_result(
            tool_name=self.name,
            query=kwargs["query"],
            content="[1] 制度.pdf\n证据",
            sources=[
                make_evidence_record(
                    source_type="knowledge_base",
                    source_id="chunk-1",
                    title="制度.pdf",
                    content="证据",
                )
            ],
        )

    def run_for_context(self, context, **kwargs):
        result = self.run(**kwargs)
        if result.assessment:
            # 模拟工具在当前请求下生成新报告；生产代码禁止给旧报告重新贴范围。
            assessment = replace(result.assessment, query_version="", evidence_version="", policy_version="")
            result.assessment = bind_assessment(context, result.candidates or result.evidence, assessment)
        return result


class EmptyTool(FakeTool):
    def run(self, **kwargs):
        self.calls.append(kwargs)
        return make_tool_result(
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
        return make_tool_result(
            tool_name=self.name,
            query=kwargs["query"],
            content="外部证据",
            sources=[
                make_evidence_record(
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
    assert result.tool_results[0].evidence.get("chunk-1").id == "chunk-1"
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
    assert result.tool_results[0].diagnostics["error_type"] == "invalid_json"


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
    assert all(not result.evidence for result in result.tool_results)


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

    assert execution.agent_run.steps == 1
    assert execution.agent_run.stop_reason == "evidence_ready"
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
    assert "session_id=session-1 stage=agent_run_total" in caplog.text
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

    assert execution.agent_run.stop_reason == "planner_fallback"
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
def test_orchestrator_keeps_prompt_and_frontend_citation_order_identical(monkeypatch):
    result = make_tool_result(
        tool_name="search_knowledge_base", query="制度", content="知识库证据",
        sources=[make_evidence_record(
            source_type="knowledge_base", source_id="chunk-1", title="制度.pdf",
            content="知识库证据", metadata={"document_id": "doc-1"},
        )],
    )
    orchestrator, answer = orchestrator_with_results(monkeypatch, [result], answer_responses=[["answer"]])

    execution = orchestrator.run(
        session_id="session-1",
        question="请综合说明",
        session_context="会话文档证据",
    )

    assert [
        document["document_id"] for document in execution.response_documents
    ] == ["doc-1", "quick_parse_session-1_0"]
    answer_prompt = execution.answer_messages[-1]["content"]
    assert "[1] 知识库证据" in answer_prompt
    assert "[2] 会话文档证据" in answer_prompt
    assert execution.agent_run.tool_results == [result]
    assert len(answer.calls) == 1


def orchestrator_with_results(monkeypatch, results, *, answer_responses):
    """固定规划结果，让测试集中检查证据到最终回答的处理行为。"""
    planner_completions = FakeCompletions([])
    answer_completions = FakeCompletions(answer_responses)
    orchestrator = AgentOrchestrator(
        planner_client=SimpleNamespace(chat=SimpleNamespace(completions=planner_completions)),
        planner_model="small-planner",
        answer_client=SimpleNamespace(chat=SimpleNamespace(completions=answer_completions)),
        answer_model="large-answer",
        tools=[],
    )
    monkeypatch.setattr(orchestrator, "_run_agent", lambda **kwargs: AgentRun(tool_results=results))
    return orchestrator, answer_completions


@pytest.mark.unit
def test_single_verified_result_keeps_resolved_question_without_duplicate_verification(monkeypatch):
    result = FakeTool().run(query="审批时限")
    context = make_query("差旅审批时限是多久？", (QueryGoal("approval", "差旅审批时限", ("差旅",)),))
    result.assessment = bind_assessment(context, result.evidence, assessment_from_record({"sufficient": True, "source": "model"}))
    monkeypatch.setattr("agent.orchestrator.query_context_for", lambda request: context)
    orchestrator, answer = orchestrator_with_results(monkeypatch, [result], answer_responses=[["answer"]])
    monkeypatch.setattr(
        "service.core.evidence_verification.assess_evidence_sufficiency",
        lambda *args, **kwargs: pytest.fail("单次已通过的检索无需重复进行集合核验"),
    )

    execution = orchestrator.run(session_id="session-1", question="它的审批时限是多久？")

    assert "差旅审批时限是多久？" in execution.answer_messages[-1]["content"]
    assert execution.assessment is result.assessment
    assert len(answer.calls) == 1
    assert execution.query_runtime["events"][-1]["phase"] == "pipeline"


@pytest.mark.unit
@pytest.mark.parametrize("sufficient", [True, False])
def test_merged_evidence_keeps_latest_goals_candidate_order_and_session_document(monkeypatch, sufficient):
    first = FakeTool().run(query="差旅制度")
    candidate = {
        "chunk_id": "candidate-3", "document_id": "doc-3",
        "document_name": "审批制度.pdf", "content_with_weight": "候选证据",
    }
    context = make_query("差旅审批时限是多久？", (QueryGoal("approval", "差旅审批时限", ("差旅",)),))
    monkeypatch.setattr("agent.orchestrator.query_context_for", lambda request: context)
    first.candidates = collection_from_records([candidate])
    first.assessment = bind_assessment(context, first.evidence, assessment_from_record({"sufficient": True}))
    second = make_tool_result(
        tool_name="search_knowledge_base", query="审批时限", content="第二条证据",
        sources=[make_evidence_record(
            source_type="knowledge_base", source_id="chunk-2",
            title="审批制度.pdf", content="第二条证据",
        )],
        metadata={
            "query_contract": {"resolved_question": "差旅审批时限是多久？"},
            "query_intent": {"goals": [{
                "id": "approval", "text": "差旅审批时限", "constraints": ["差旅"],
            }]},
            "candidate_chunks": [dict(candidate)],
            "evidence_sufficiency": {"sufficient": True},
        },
    )
    checked = []

    def verify(question, documents, *, goals, context=None, **_verification_options):
        checked.append((question, documents, goals))
        return EvidenceAssessment(
            sufficient=sufficient, reason="集合核验",
            missing_requirements=[] if sufficient else ["审批时限"],
            supporting_chunk_indices=[1, 2, 3, 4] if sufficient else [],
            source="model", evaluated_chunk_count=len(documents),
            evaluated_chunk_ids=[document.id for document in documents],
        )

    monkeypatch.setattr("service.core.evidence_verification.assess_evidence_sufficiency", verify)
    orchestrator, answer = orchestrator_with_results(
        monkeypatch, [first, second], answer_responses=[["answer"]] if sufficient else [],
    )

    execution = orchestrator.run(
        session_id="session-1", question="它的审批时限是多久？", session_context="会话文档证据",
    )

    assert len(checked) == 1
    question, documents, goals = checked[0]
    assert question == "差旅审批时限是多久？"
    assert [(goal.id, goal.constraints) for goal in goals] == [("approval", ("差旅",))]
    assert [document.id for document in documents] == [
        "candidate-3", "chunk-1", "chunk-2", "quick_parse_session-1_0",
    ]
    assert list(documents)[1].text == "证据"
    assert list(documents)[0].text == "候选证据"
    assert execution.evidence_sufficiency["sufficient"] is sufficient
    if sufficient:
        assert len(answer.calls) == 1
        assert len(execution.retrieved_content) == 3
        assert len(execution.response_documents) == 4
        assert "[4] 会话文档证据" in execution.answer_messages[-1]["content"]
    else:
        assert execution.response_mode == "refuse"
        assert "审批时限" in execution.direct_answer
        assert execution.response_documents == execution.answer_messages == []
        assert len(execution.evidence) == 4
        assert answer.calls == []


@pytest.mark.unit
def test_clarification_returns_before_decoding_goals_or_verifying_evidence(monkeypatch):
    result = make_tool_result(
        tool_name="search_knowledge_base", query="审批时限", content="请明确制度名称。",
        metadata={
            "response_mode": "clarify", "clarification": "请明确制度名称。",
            "query_contract": {"resolved_question": "它的审批时限？"},
            "query_intent": {"goals": [{"id": "incomplete"}]},
        },
    )
    orchestrator, answer = orchestrator_with_results(monkeypatch, [result], answer_responses=[])
    monkeypatch.setattr(
        "service.core.evidence_verification.assess_evidence_sufficiency",
        lambda *args, **kwargs: pytest.fail("需要澄清时不应继续核验"),
    )

    execution = orchestrator.run(session_id="session-1", question="审批时限？")

    assert execution.response_mode == "clarify"
    assert execution.direct_answer == "请明确制度名称。"
    assert execution.retrieved_content == execution.response_documents == execution.answer_messages == []
    assert execution.answer_stream is None
    assert answer.calls == []


@pytest.mark.unit
def test_direct_question_runs_planner_without_tool_evidence():
    planner = FakeCompletions([completion(assistant_message(content="READY"))])
    answer = FakeCompletions([["answer"]])
    orchestrator = AgentOrchestrator(
        planner_client=SimpleNamespace(chat=SimpleNamespace(completions=planner)),
        planner_model="small-planner",
        answer_client=SimpleNamespace(chat=SimpleNamespace(completions=answer)),
        answer_model="large-answer",
        tools=[],
    )

    execution = orchestrator.run(session_id="session-1", question="你好")

    assert len(planner.calls) == 1
    assert execution.agent_run.steps == 1
    assert execution.agent_run.stop_reason == "completed"
    assert execution.retrieved_content == []
    assert execution.response_mode == "answer"
    assert execution.response_documents == []
    assert len(answer.calls) == 1


@pytest.mark.unit
def test_empty_tool_result_does_not_refuse_when_session_document_is_available(monkeypatch):
    result = EmptyTool().run(query="审批时限")
    orchestrator, answer = orchestrator_with_results(monkeypatch, [result], answer_responses=[["answer"]])

    monkeypatch.setattr("service.core.evidence_verification.assess_evidence_sufficiency",
        lambda question, documents, **kwargs: EvidenceAssessment(
            True, "会话文档明确给出时限", [], [1], "model", 1, [next(iter(documents)).id]))
    execution = orchestrator.run(
        session_id="session-1", question="审批时限是多久？", session_context="审批时限为三个工作日。",
    )

    assert execution.response_mode == "answer"
    assert execution.direct_answer is None
    assert "[1] 审批时限为三个工作日。" in execution.answer_messages[-1]["content"]
    assert len(answer.calls) == 1


@pytest.mark.unit
def test_answer_request_failure_preserves_exception_and_performance_log(monkeypatch, caplog):
    error = RuntimeError("answer unavailable")

    def fail_answer(**kwargs):
        raise error

    orchestrator, answer = orchestrator_with_results(monkeypatch, [], answer_responses=[])
    monkeypatch.setattr(answer, "create", fail_answer)
    caplog.set_level(logging.INFO, logger="agent.orchestrator")

    with pytest.raises(RuntimeError) as raised:
        orchestrator.run(session_id="session-1", question="你好")

    assert raised.value is error
    assert "stage=answer_request model=large-answer status=error" in caplog.text
    assert error.query_runtime["events"][-1]["end_reason"] == "unhandled_error"
