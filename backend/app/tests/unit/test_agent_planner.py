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


def test_one_goal_can_use_separate_year_queries():
    question = "A公司2023年和2022年净利润比较"
    context = prepare_query_context(question)
    context.understanding = QueryUnderstanding((QueryGoal("g1", question, constraints=("A公司", "2023年", "2022年")),))
    plan = decode_plan({"tasks": [task(query="A公司2023年净利润"), task(id="s2", query="A公司2022年净利润")]})
    validate_plan(plan, context, {"search_knowledge_base"}, initial=True)


@pytest.mark.parametrize("change", [
    {"goal_ids": ["unknown"]}, {"tool": "unknown"}, {"query": "B公司2024年员工培训要求"},
    {"query": "员工培训要求", "depends_on": ["missing"]},
])
def test_plan_rejects_invented_goal_tool_constraint_or_dependency(change):
    context = prepare_query_context("A公司2023年员工培训要求")
    context.understanding = QueryUnderstanding((QueryGoal("g1", context.question, constraints=("A公司", "2023年")),))
    data = task(query=context.question)
    data.update(change)
    with pytest.raises(QueryValidationError):
        validate_plan(decode_plan({"tasks": [data]}), context, {"search_knowledge_base"}, initial=True)


def test_unknown_dependency_cannot_be_replaced_with_guessed_entity():
    question = "研报推荐的公司；该公司的公告"
    context = prepare_query_context(question)
    context.understanding = QueryUnderstanding((QueryGoal("g1", "研报推荐的公司"), QueryGoal("g2", "该公司的公告", depends_on=("g1",))))
    with pytest.raises(QueryValidationError):
        validate_plan(decode_plan({"tasks": [task(query="研报推荐的公司"),
            task(id="s2", query="A公司公告", goal_ids=("g2",))]}), context, {"search_knowledge_base"}, initial=True)


def test_replan_cannot_drop_company_even_if_previous_search_had_it():
    context = prepare_query_context("A公司2023年员工培训要求")
    context.understanding = QueryUnderstanding((QueryGoal("g1", context.question, constraints=("A公司", "2023年")),))
    previous = (AgentTask("s1", query=context.question, goal_ids=("g1",)),)
    with pytest.raises(QueryValidationError, match="scope_dropped"):
        validate_plan(decode_plan({"tasks": [task(id="s2", query="2023年员工培训要求")]}), context,
                      {"search_knowledge_base"}, previous=previous)


def test_model_cannot_supply_tenant_or_server_context():
    for extra in ({"tenant_id": "other"}, {"_context": {}}, {"document_ids": ["foreign"]}):
        with pytest.raises(QueryValidationError, match="unexpected_task_field"):
            decode_plan({"tasks": [{**task(), **extra}]})


def test_colloquial_goal_keeps_original_provenance():
    question = "国电电力未来三年能赚多少？"
    payload = initial(goals=[{"id": "g1", "text": "国电电力未来三年盈利预测", "constraints": ["国电电力", "未来三年"],
                              "source_spans": [question]}], tasks=[task(query="国电电力未来三年盈利预测")])
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
def test_same_report_requirement_uses_document_filter(inherit):
    context = prepare_query_context("研报推荐的公司及同一报告中的评级")
    context.understanding = QueryUnderstanding((
        QueryGoal("g1", "研报推荐的公司"),
        QueryGoal("g2", "评级", constraints=("同一报告",), depends_on=("g1",)),
    ))
    plan = decode_plan({"tasks": [
        task(query="研报推荐的公司", output_slot="company"),
        task(id="s2", query="", query_template="{company}评级", goal_ids=("g2",),
             depends_on=["s1"], required_slots=["company"], inherit_document_scope=inherit),
    ]})
    if inherit:
        validate_plan(plan, context, {"search_knowledge_base"}, initial=True)
    else:
        with pytest.raises(QueryValidationError, match="document_scope_not_bound"):
            validate_plan(plan, context, {"search_knowledge_base"}, initial=True)
