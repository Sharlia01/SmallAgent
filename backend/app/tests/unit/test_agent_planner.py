"""需求与首轮计划一次生成，重规划只改变执行策略。"""
import json
import pytest
from agent.planner import AgentPlanner
from agent.plans import decode_plan, validate_plan, AgentTask
from service.core.query.context import prepare_query_context
from service.core.query.contract import QueryGoal, QueryUnderstanding
from service.core.query.validation import QueryValidationError
from tests.agent_helpers import ModelClient, SearchTool, initial, task, goal

pytestmark = pytest.mark.unit


def planner(client):
    return AgentPlanner(client=client, model="test", tool_definitions=[{
        "function": {"name": "search_knowledge_base", "description": "知识库"}}])


def test_initial_understanding_and_plan_use_one_model_call():
    client = ModelClient([initial()])
    context = prepare_query_context("员工培训要求")
    decision = planner(client).analyze_and_plan(context)
    assert len(client.calls) == 1
    assert decision.understanding is context.understanding
    assert decision.plan.tasks[0].goal_ids == ("g1",)
    assert context.goals[0].source_spans == ("员工培训要求",)


@pytest.mark.parametrize(("configured", "remaining", "expected"), [
    (None, 120, 45), ("60", 120, 60), ("60", 7, 7),
])
def test_planning_and_replanning_timeout_respects_remaining_budget(monkeypatch, configured, remaining, expected):
    from service.core.runtime.context import QueryRuntime, query_runtime
    monkeypatch.delenv("AGENT_PLANNER_TIMEOUT_SECONDS", raising=False)
    if configured is not None:
        monkeypatch.setenv("AGENT_PLANNER_TIMEOUT_SECONDS", configured)
    runtime = QueryRuntime()
    monkeypatch.setattr(runtime, "remaining", lambda: remaining)
    client = ModelClient([initial(), {"tasks": []}])
    instance = planner(client)
    context = prepare_query_context("员工培训要求")
    with query_runtime(runtime):
        decision = instance.analyze_and_plan(context)
        instance.replan(context, previous=decision.plan.tasks, observations=[], assessment={})
    assert [call["timeout"] for call in client.calls] == [expected, expected]


def test_invalid_plan_repair_does_not_commit_partial_understanding():
    bad = initial(tasks=[task(tool="unknown")])
    client = ModelClient([bad, bad])
    context = prepare_query_context("员工培训要求")
    with pytest.raises(QueryValidationError):
        planner(client).analyze_and_plan(context)
    assert context.understanding is None
    assert len(client.calls) == 2


def test_clarification_does_not_need_search_plan():
    client = ModelClient([initial(goals=[], tasks=[], clarification="需要查询哪份制度？")])
    context = prepare_query_context("查询制度")
    result = planner(client).analyze_and_plan(context)
    assert result.understanding.clarification and not result.plan.tasks


def test_replan_preserves_fixed_understanding_and_uses_feedback():
    client = ModelClient([initial(), {"tasks": [task(id="s2", query="员工培训要求 完整内容")]}])
    instance = planner(client)
    context = prepare_query_context("员工培训要求")
    decision = instance.analyze_and_plan(context)
    previous = context.understanding
    plan = instance.replan(context, previous=decision.plan.tasks, observations=[], assessment={"missing_requirements": ["培训频率"]})
    assert context.understanding is previous
    assert plan.tasks[0].id == "s2"
    assert "培训频率" in client.calls[-1]["messages"][-1]["content"]


@pytest.mark.parametrize("slot_fields", [{}, {"output_slot": None}])
def test_direct_replan_does_not_require_output_slot(slot_fields):
    client = ModelClient([initial(), {"tasks": [task(id="s2", **slot_fields)]}])
    instance = planner(client)
    context = prepare_query_context("员工培训要求")
    decision = instance.analyze_and_plan(context)
    result = instance.replan(context, previous=decision.plan.tasks, observations=[], assessment={})
    assert result.tasks[0].output_slot is None
    assert len(client.calls) == 2


@pytest.mark.parametrize("invalid_slot", ["", "公司", "Company", "company-name", "a" * 33, 123])
def test_replan_output_slot_failure_reaches_log_and_repair_request(invalid_slot, caplog):
    client = ModelClient([
        initial(),
        {"tasks": [task(id="s2", output_slot=invalid_slot)]},
        {"tasks": [task(id="s2", output_slot=None)]},
    ])
    instance = planner(client)
    context = prepare_query_context("员工培训要求")
    decision = instance.analyze_and_plan(context)
    fixed_understanding = context.understanding
    result = instance.replan(context, previous=decision.plan.tasks, observations=[], assessment={})

    # 校验失败只触发现有的一次纠错；日志和纠错请求都应能定位具体任务及错误值。
    assert len(client.calls) == 3
    assert result.tasks[0].output_slot is None
    assert context.understanding is fixed_understanding
    record = next(record for record in caplog.records
                  if "agent_replan validation failed" in record.getMessage())
    details = json.loads(record.getMessage().split(": ", 1)[1])
    assert details["failure_fields"] == ["tasks[].output_slot"]
    assert details["node_id"] == "s2"
    assert details["validation_context"]["actual"] == invalid_slot
    assert "^[a-z][a-z0-9_]{0,31}$" in details["validation_context"]["expected"]
    repair_message = client.calls[-1]["messages"][-1]["content"]
    assert json.dumps(details, ensure_ascii=False) in repair_message


def test_json_provider_accepts_initial_repair_and_replan_requests():
    """模拟线上 JSON 模式的准入检查，覆盖实际补充规划请求构造。"""
    class JsonProvider(ModelClient):
        def create(self, **kwargs):
            if kwargs.get("response_format") == {"type": "json_object"}:
                if not any("json" in str(message.get("content", "")).lower()
                           for message in kwargs["messages"]):
                    raise ValueError("messages must contain the word json")
            return super().create(**kwargs)

    client = JsonProvider([
        initial(tasks=[task(tool="unknown")]),
        initial(),
        {"tasks": [task(id="s2", query="员工培训要求 完整内容")]},
    ])
    context = prepare_query_context("员工培训要求")
    instance = planner(client)
    decision = instance.analyze_and_plan(context)
    fixed = context.understanding
    result = instance.replan(context, previous=decision.plan.tasks, observations=[], assessment={})
    assert result.tasks[0].id == "s2"
    assert context.understanding is fixed
    assert len(client.calls) == 3
    for call in client.calls:
        assert any(message["role"] == "system" and "json" in message["content"].lower()
                   for message in call["messages"])


def test_one_goal_can_use_separate_year_queries():
    question = "A公司2023年和2022年净利润比较"
    context = prepare_query_context(question)
    context.understanding = QueryUnderstanding((QueryGoal("g1", question, constraints=("A公司", "2023年", "2022年")),))
    plan = decode_plan({"tasks": [task(query="A公司2023年净利润"), task(id="s2", query="A公司2022年净利润")]})
    validate_plan(plan, context, {"search_knowledge_base"}, initial=True)


@pytest.mark.parametrize("change", [
    {"goal_ids": ["unknown"]}, {"tool": "unknown"},
    {"query": "员工培训要求", "depends_on": ["missing"]},
])
def test_plan_rejects_unknown_goal_tool_or_dependency(change):
    context = prepare_query_context("A公司2023年员工培训要求")
    context.understanding = QueryUnderstanding((QueryGoal("g1", context.question, constraints=("A公司", "2023年")),))
    data = task(query=context.question)
    data.update(change)
    with pytest.raises(QueryValidationError):
        validate_plan(decode_plan({"tasks": [data]}), context, {"search_knowledge_base"}, initial=True)


def test_unknown_dependency_cannot_be_replaced_with_guessed_entity():
    question = "研报推荐的公司；该公司的公告"
    context = prepare_query_context(question)
    context.understanding = QueryUnderstanding((QueryGoal("g1", "研报推荐的公司"),
        QueryGoal("g2", "该公司的公告", depends_on=("g1",), subject_slot="company")))
    with pytest.raises(QueryValidationError, match="subject_slot_not_bound"):
        validate_plan(decode_plan({"tasks": [task(query="研报推荐的公司", output_slot="company"),
            task(id="s2", query="A公司公告", goal_ids=("g2",))]}), context, {"search_knowledge_base"}, initial=True)


def test_replan_query_does_not_have_to_repeat_goal_constraints():
    context = prepare_query_context("A公司2023年员工培训要求")
    context.understanding = QueryUnderstanding((QueryGoal("g1", context.question, constraints=("A公司", "2023年")),))
    previous = (AgentTask("s1", query=context.question, goal_ids=("g1",)),)
    plan = decode_plan({"tasks": [task(id="s2", query="2023年员工培训要求")]})
    assert validate_plan(plan, context, {"search_knowledge_base"}, previous=previous) is plan


def test_model_cannot_supply_tenant_or_server_context():
    for extra in ({"tenant_id": "other"}, {"_context": {}}, {"document_ids": ["foreign"]}):
        with pytest.raises(QueryValidationError, match="unexpected_task_field"):
            decode_plan({"tasks": [{**task(), **extra}]})


def test_colloquial_goal_keeps_original_provenance():
    question = "国电电力未来三年能赚多少？"
    payload = initial(goals=[{"id": "g1", "text": "国电电力未来三年盈利预测", "constraints": ["国电电力", "未来三年"],
                              "source_spans": [question], "subject_refs": ["e1"]}], tasks=[task(query="国电电力未来三年盈利预测")],
                      entities=[dict(ref="e1", mention="国电电力", source_quote=question)])
    context = prepare_query_context(question)
    decision = planner(ModelClient([payload])).analyze_and_plan(context)
    assert decision.understanding.goals[0].text == "国电电力未来三年盈利预测"
    assert decision.understanding.goals[0].source_spans == (question,)
    assert context.request.user_question == question


def test_within_question_dependency_can_resolve_pronoun_without_guessing_company():
    question = "先查研报推荐的公司，再查它的公告"
    context = prepare_query_context(question)
    assert context.request.clarification  # 历史消解没有具体公司，这是待确认的提示。
    payload = initial(goals=[
        goal("研报推荐的公司", "g1"), goal("它的公告", "g2", depends_on=("g1",)),
    ], tasks=[task(query="研报推荐的公司", output_slot="company"),
              task(id="s2", query="", query_template="{company}的公告", goal_ids=("g2",),
                   depends_on=["s1"], required_slots=["company"])])
    decision = planner(ModelClient([payload])).analyze_and_plan(context)
    assert context.request.clarification is None
    assert context.request.user_question == question
    assert decision.plan.tasks[1].query_template == "{company}的公告"


def test_unresolved_pronoun_cannot_be_silently_searched_as_a_concrete_query():
    context = prepare_query_context("它的收入是多少？")
    payload = initial(goals=[goal("它的收入是多少？")], tasks=[task(query="它的收入是多少？")])
    with pytest.raises(ValueError):
        planner(ModelClient([payload, payload])).analyze_and_plan(context)
    assert context.request.clarification


@pytest.mark.parametrize("inherit", [True, False])
@pytest.mark.parametrize("constraint", ["同一报告", "不要求来自同一份报告"])
def test_document_scope_is_not_inferred_from_goal_keywords(inherit, constraint):
    context = prepare_query_context("研报推荐的公司及同一报告中的评级")
    context.understanding = QueryUnderstanding((
        QueryGoal("g1", "研报推荐的公司"),
        QueryGoal("g2", "评级", constraints=(constraint,), depends_on=("g1",)),
    ))
    plan = decode_plan({"tasks": [
        task(query="研报推荐的公司", output_slot="company"),
        task(id="s2", query="", query_template="{company}评级", goal_ids=("g2",),
             depends_on=["s1"], required_slots=["company"], inherit_document_scope=inherit),
    ]})
    assert validate_plan(plan, context, {"search_knowledge_base"}, initial=True) is plan
    assert plan.tasks[1].inherit_document_scope is inherit


def test_logical_goal_dependency_allows_parallel_retrieval():
    question = "查询营业收入及同比变化"
    payload = initial(goals=[goal("营业收入", "g1"), goal("同比变化", "g2", depends_on=("g1",))],
                      tasks=[task(query="营业收入"), task(id="s2", query="同比变化", goal_ids=("g2",))])
    client = ModelClient([payload])
    decision = planner(client).analyze_and_plan(prepare_query_context(question))
    assert len(client.calls) == 1
    assert decision.understanding.goals[1].depends_on == ("g1",)
    assert all(not item.depends_on for item in decision.plan.tasks)


@pytest.mark.parametrize("constraint", ["只查公开研报", "仅使用内部材料"])
def test_source_choice_is_not_rejected_by_constraint_keywords(constraint):
    context = prepare_query_context("研报观点")
    context.understanding = QueryUnderstanding((QueryGoal("g1", "研报观点", constraints=(constraint,)),))
    plan = decode_plan({"tasks": [task(tool="search_web", query="公开研报观点")]})
    assert validate_plan(plan, context, {"search_web"}, initial=True) is plan


@pytest.mark.parametrize("separate_subjects", [False, True])
def test_same_query_tasks_are_diagnostic_only(caplog, separate_subjects):
    context = prepare_query_context("比较两家公司营业收入")
    context.understanding = QueryUnderstanding((QueryGoal("g1", context.question, subject_refs=("e1", "e2")),))
    plan = decode_plan({"tasks": [
        task(query="营业收入", subject_refs=["e1"]),
        task(id="s2", query="营业收入", subject_refs=["e2"] if separate_subjects else ["e1"]),
    ]})
    with caplog.at_level("INFO", logger="agent.plans"):
        assert validate_plan(plan, context, {"search_knowledge_base"}, initial=True) is plan
    assert len(plan.tasks) == 2
    assert "duplicate_task: task_id=s2 similar_task_id=s1" in caplog.text


@pytest.mark.parametrize("empty", [False, True])
def test_goal_without_search_task_is_diagnostic_only(caplog, empty):
    context = prepare_query_context("查询收入并计算差额")
    context.understanding = QueryUnderstanding((QueryGoal("g1", "查询收入"),
                                               QueryGoal("g2", "计算差额", depends_on=("g1",))))
    plan = decode_plan({"tasks": [] if empty else [task(query="收入")]})
    with caplog.at_level("INFO", logger="agent.plans"):
        assert validate_plan(plan, context, {"search_knowledge_base"}, initial=True) is plan
    assert "goal_coverage_mismatch" in caplog.text and "g2" in caplog.text
    assert len(context.goals) == 2


@pytest.mark.parametrize(("producer_tool", "consumer_tool", "rule"), [
    ("search_web", "search_knowledge_base", "invalid_document_source"),
    ("search_knowledge_base", "search_web", "unsupported_document_scope"),
])
def test_explicit_document_scope_still_requires_internal_sources(producer_tool, consumer_tool, rule):
    context = prepare_query_context("查询公司公告")
    context.understanding = QueryUnderstanding((QueryGoal("g1", context.question),))
    plan = decode_plan({"tasks": [
        task(tool=producer_tool, query="公司", output_slot="company"),
        task(id="s2", tool=consumer_tool, query="", query_template="{company}公告",
             depends_on=["s1"], required_slots=["company"], inherit_document_scope=True),
    ]})
    with pytest.raises(QueryValidationError, match=rule):
        validate_plan(plan, context, {"search_web", "search_knowledge_base"})


def test_unused_answer_slots_are_removed_without_model_retry():
    client = ModelClient([
        initial(tasks=[task(output_slot="growth_rate")]),
        {"tasks": [task(id="s2", query="员工培训要求 完整内容", output_slot="growth_rate_v2")]},
    ])
    instance = planner(client)
    context = prepare_query_context("员工培训要求")
    decision = instance.analyze_and_plan(context)
    assert decision.plan.tasks[0].output_slot is None
    result = instance.replan(context, previous=decision.plan.tasks, observations=[], assessment={})
    assert result.tasks[0].output_slot is None
    assert result.tasks[0].query == "员工培训要求 完整内容"
    assert len(client.calls) == 2
