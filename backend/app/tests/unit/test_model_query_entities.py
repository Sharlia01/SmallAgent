"""目标直接保存自然语言主体；验证理解、任务选择、检索和核验传递。"""
import json
import pytest

from agent.executor import ToolExecutor
from agent.planner import AgentPlanner
from agent.plans import decode_plan, validate_plan
from agent.understanding import decode_understanding, validate_understanding
from service.core.query.context import prepare_query_context
from service.core.query.contract import RetrievalTask, decode_goals
from service.core.query.validation import QueryValidationError
from tests.agent_helpers import ModelClient, SearchTool, initial, task, records

pytestmark = pytest.mark.unit


def goal(text, subjects=(), id="g1", source=None, constraints=(), **kwargs):
    return dict(id=id, text=text, source_spans=[source or text], constraints=list(constraints),
                subjects=list(subjects), **kwargs)


def run_plan(question, payload):
    client = ModelClient([payload, payload])
    context = prepare_query_context(question)
    planner = AgentPlanner(client=client, model="test", tool_definitions=[{"function": {"name": "search_knowledge_base"}}])
    decision = planner.analyze_and_plan(context)
    return context, decision, client


@pytest.mark.parametrize(("question", "query"), [
    ("润本股份25H1新增合作的渠道是什么？", "润本股份2025年上半年新增合作的渠道"),
    ("润本股份2025E至2027E的营业收入是多少？", "润本股份2025E、2026E、2027E的营业收入"),
])
def test_equivalent_time_expressions_reach_retrieval(question, query):
    context, decision, client = run_plan(question, initial(
        goals=[goal(query, subjects=("润本股份",), source=question)], tasks=[task(query=query)]))
    assert len(client.calls) == 1
    tool = SearchTool([records()])
    result = ToolExecutor([tool]).execute_task(decision.plan.tasks[0], context, {}, {})
    assert result.status == "succeeded"
    assert tool.calls[0].query == query


@pytest.mark.parametrize("subject", ["国电电力", "国电电力新星科技有限公司", "甲公司的煤电板块", "星河项目"])
def test_goal_subjects_need_no_registry_or_shared_entity_list(subject, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("must not infer subject identity or name boundaries")
    monkeypatch.setattr("service.core.entities.registry.get_entity_registry", forbidden)
    question = f"为什么{subject}的收入下降？"
    context, decision, client = run_plan(question, initial(
        goals=[goal(question, subjects=(subject,))], tasks=[task(query=question)]))
    assert len(client.calls) == 1
    assert context.goals[0].subjects == (subject,)
    assert set(context.understanding.to_dict()) == {
        "goals", "requires_evidence", "clarification", "source", "failure_reason"}
    tool = SearchTool([records()])
    ToolExecutor([tool]).execute_task(decision.plan.tasks[0], context, {}, {})
    assert tool.calls[0].subject_names == (subject,)


def test_multiple_goals_keep_their_own_subjects():
    question = "比较甲公司的煤电板块和水电板块收入"
    subjects = ("甲公司的煤电板块", "甲公司的水电板块")
    context, decision, _ = run_plan(question, initial(goals=[
        goal("煤电板块收入", subjects=(subjects[0],), source=question),
        goal("水电板块收入", subjects=(subjects[1],), id="g2", source=question),
    ], tasks=[task(query="煤电收入"), task(id="s2", query="水电收入", goal_ids=("g2",))]))
    tool = SearchTool([records(), records("c2")])
    executor = ToolExecutor([tool])
    for node in decision.plan.tasks:
        executor.execute_task(node, context, {}, {})
    assert [call.subject_names for call in tool.calls] == [(subjects[0],), (subjects[1],)]
    assert context.for_goals(("g2",)).subject_names() == (subjects[1],)


def test_comparison_tasks_select_goal_subjects_and_reject_outside_names():
    question = "比较甲公司的煤电板块和水电板块收入"
    subjects = ("甲公司的煤电板块", "甲公司的水电板块")
    context, decision, _ = run_plan(question, initial(
        goals=[goal(question, subjects=subjects)], tasks=[
            task(query="营业收入", subjects=[subjects[0]]),
            task(id="s2", query="营业收入", subjects=[subjects[1]])]))
    tool = SearchTool([records(), records("c2")])
    executor, cache = ToolExecutor([tool]), {}
    for node in decision.plan.tasks:
        executor.execute_task(node, context, {}, cache)
    assert [call.subject_names for call in tool.calls] == [(subjects[0],), (subjects[1],)]
    for selected in (["乙公司的煤电板块"], []):
        bad = decode_plan({"tasks": [task(query="收入", subjects=selected)]})
        with pytest.raises(QueryValidationError, match="invalid_task_subjects"):
            validate_plan(bad, context, {"search_knowledge_base"})


@pytest.mark.parametrize("invalid", [None, "甲公司", [""], ["  "], [42], [{}], ["a" * 401], ["a"] * 17])
def test_subject_schema_rejects_malformed_values(invalid):
    row = {**goal("收入"), "subjects": invalid}
    with pytest.raises(QueryValidationError, match="invalid_subjects"):
        decode_goals([row])
    if invalid is not None:
        with pytest.raises(QueryValidationError, match="invalid_subjects"):
            decode_plan({"tasks": [task(subjects=invalid)]})
    else:
        assert decode_plan({"tasks": [task(subjects=None)]}).tasks[0].subjects is None


def test_normalized_subjects_use_goal_source_spans_without_exact_name_matching():
    question = "甲公司煤电和水电板块的收入"
    payload = initial(goals=[goal(question, subjects=("甲公司的煤电板块", "甲公司的水电板块"))])["understanding"]
    value = validate_understanding(decode_understanding(json.dumps(payload)), question)
    assert value.goals[0].subjects == ("甲公司的煤电板块", "甲公司的水电板块")
    assert value.goals[0].source_spans == (question,)
    payload["goals"][0]["source_spans"] = ["不存在的引文"]
    with pytest.raises(QueryValidationError, match="goal_source_not_grounded"):
        validate_understanding(decode_understanding(json.dumps(payload)), question)


def test_publisher_requirement_reaches_verifier_as_constraint():
    from service.core.evidence.verification import verify_evidence
    from service.core.evidence.adapter import collection_from_records
    question = "国信证券关于国电电力的研报观点"
    context, decision, _ = run_plan(question, initial(goals=[goal(
        question, subjects=("国电电力",), constraints=["来源：国信证券发布的研报"])], tasks=[task(query=question)]))
    tool = SearchTool([records()])
    ToolExecutor([tool]).execute_task(decision.plan.tasks[0], context, {}, {})
    assert tool.calls[0].subject_names == ("国电电力",)
    checker = ModelClient([{"sufficient": True, "reason": "指定来源提供观点", "missing_requirements": [],
                            "goal_coverage": [{"goal_id": "g1", "status": "supported", "supporting_chunk_indices": [1]}]}])
    verify_evidence(context, collection_from_records(records(text=question)), client=checker)
    payload = json.loads(checker.calls[0]["messages"][1]["content"])
    assert payload["goals"][0]["constraints"] == ["来源：国信证券发布的研报"]
    assert payload["goals"][0]["subjects"] == ["国电电力"]
    assert set(payload["goals"][0]) == {
        "id", "text", "depends_on", "source_spans", "constraints", "subjects", "subject_slot"}


def test_local_pronoun_is_resolved_by_planner_not_regex():
    question = "介绍国电电力，并说明它的项目进展"
    context, _, client = run_plan(question, initial(
        goals=[goal("介绍国电电力并说明国电电力的项目进展", subjects=("国电电力",), source=question)],
        tasks=[task(query="国电电力介绍和项目进展")]))
    assert len(client.calls) == 1
    assert context.request.user_question == context.request.resolved_question == question
    assert context.request.clarification is None and not context.request.bindings


def test_unknown_subject_is_bound_only_after_evidence():
    from service.core.evidence.bindings import BridgeExtraction
    question = "先查研报推荐的公司，再查它的公告"
    context, decision, _ = run_plan(question, initial(goals=[
        goal("研报推荐的公司", subjects=(), source="研报推荐的公司"),
        goal("它的公告", subjects=(), id="g2", depends_on=["g1"], subject_slot="company"),
    ], tasks=[task(query="研报推荐的公司", output_slot="company"),
        task(id="s2", query="", query_template="{company}公告", depends_on=["s1"],
             required_slots=["company"], goal_ids=("g2",))]))
    tool = SearchTool([records(text="推荐国电电力"), records("c2")])
    def bridge(**kwargs):
        return BridgeExtraction(slot="company", value="国电电力",
                                source_chunk_id="c1", document_id="doc1",
                                source_quote="推荐国电电力")
    executor = ToolExecutor([tool], bridge_extractor=bridge)
    first = executor.execute_task(decision.plan.tasks[0], context, {}, {})
    assert first.binding
    executor.execute_task(decision.plan.tasks[1], context, {"s1": first}, {})
    assert tool.calls[0].subject_names == ()
    assert tool.calls[1].subject_names == ("国电电力",)


@pytest.mark.parametrize("names", [(), ("国电电力新星科技有限公司",), ("国电电力",)])
def test_pipeline_uses_supplied_scope_even_when_empty_or_unregistered(names, monkeypatch):
    from service.core.retrieval import pipeline
    seen = []
    class Dealer:
        def retrieval(self, **kwargs):
            seen.append(kwargs["options"].subject_names)
            return {"chunks": [], "total": 0}
    monkeypatch.setattr(pipeline, "get_retrieval_dealer", lambda: Dealer())
    def forbidden(*args, **kwargs):
        raise AssertionError("must not re-extract company from search text")
    monkeypatch.setattr("service.core.entities.registry.get_entity_registry", forbidden)
    scope = names
    pipeline.retrieve_evidence("1", "华能国际公告", task=RetrievalTask("华能国际公告", ("g1",), subject_names=scope))
    assert seen == [scope]




def test_subject_discovery_task_does_not_require_its_own_output():
    from agent.plans import AgentTask
    from service.core.query.contract import QueryUnderstanding, QueryGoal, QueryRequest
    from service.core.query.context import QueryContext
    from service.core.evidence.bindings import BridgeExtraction
    context = QueryContext(QueryRequest.from_user("先查推荐公司，再查营业收入"), QueryUnderstanding(
        (QueryGoal("g1", "推荐公司及其营业收入", subject_slot="company"),)), True)
    tool = SearchTool([records(text="推荐国电电力"), records("c2")])
    executor = ToolExecutor([tool], bridge_extractor=lambda **kw: BridgeExtraction(
        "company", "国电电力", "c1", "doc1", "推荐国电电力"))
    first = executor.execute_task(AgentTask("s1", query="推荐公司", goal_ids=("g1",), output_slot="company"), context, {}, {})
    assert first.status == "succeeded" and first.binding
    second = executor.execute_task(AgentTask("s2", query_template="{company}营业收入", goal_ids=("g1",),
                                             depends_on=("s1",), required_slots=("company",)), context, {"s1": first}, {})
    assert second.status == "succeeded"
    assert tool.calls[-1].subject_names == ("国电电力",)


def test_context_model_can_prefer_current_question_over_old_company():
    from service.core.query.contract import QueryRequest, resolve_request
    request = QueryRequest.from_user("介绍国电电力，再介绍它的项目", [{"role": "user", "content": "华能国际"}])
    client = ModelClient([{"bindings": [dict(mention="它", value="国电电力", source_turn=-1, source_quote="国电电力")]}])
    resolved = resolve_request(request, client=client)
    assert resolved.resolved_question == "介绍国电电力，再介绍国电电力的项目"
    assert resolved.bindings[0].source_turn == -1


def test_replan_can_mention_another_company_without_changing_entity_filter():
    question = "国电电力项目进展"
    context, decision, _ = run_plan(question, initial(goals=[goal(question, subjects=("国电电力",))],
                                                     tasks=[task(query=question)]))
    good = decode_plan({"tasks": [task(id="s2", query="国电电力项目进展完整报告")]})
    validate_plan(good, context, {"search_knowledge_base"}, previous=decision.plan.tasks)
    expanded = decode_plan({"tasks": [task(id="s2", query="国电电力及华能国际项目进展")]})
    expanded = validate_plan(expanded, context, {"search_knowledge_base"}, previous=decision.plan.tasks)
    tool = SearchTool([records()])
    result = ToolExecutor([tool]).execute_task(expanded.tasks[0], context, {}, {})
    assert result.status == "succeeded"
    assert tool.calls[0].subject_names == ("国电电力",)


def test_business_segment_subject_reaches_verifier_without_identity_mapping():
    from service.core.evidence.verification import verify_evidence
    from service.core.evidence.adapter import collection_from_records
    subject = "国电电力的煤电板块"
    question = f"{subject}2025年营业收入是多少"
    context, decision, _ = run_plan(question, initial(
        goals=[goal(question, subjects=(subject,))], tasks=[task(query=question)]))
    assert context.goals[0].subjects == (subject,)
    tool = SearchTool([records(text=f"{subject}2025年营业收入为100亿元。")])
    ToolExecutor([tool]).execute_task(decision.plan.tasks[0], context, {}, {})
    assert tool.calls[0].subject_names == (subject,)
    checker = ModelClient([{"sufficient": True, "reason": "主体及收入有证据", "missing_requirements": [],
                            "goal_coverage": [{"goal_id": "g1", "status": "supported", "supporting_chunk_indices": [1]}]}])
    verify_evidence(context, collection_from_records(records(text=f"{subject}收入100亿元")), client=checker)
    payload = json.loads(checker.calls[0]["messages"][1]["content"])
    assert payload["goals"][0]["subjects"] == [subject]
    assert "entity_id" not in json.dumps(payload["goals"])


def test_discovered_segment_binding_accepts_model_subject_wording():
    from agent.plans import AgentTask
    from service.core.query.contract import QueryUnderstanding, QueryGoal, QueryRequest
    from service.core.query.context import QueryContext
    from service.core.evidence.bindings import extract_bridge_value
    quote = "甲公司利润最高的是煤电板块。"
    client = ModelClient([dict(found=True, value="煤电板块", subject="甲公司的煤电业务",
                              source_quote=quote, source_chunk_index=1)])
    context = QueryContext(QueryRequest.from_user("先查甲公司利润最高的板块，再查其发电量"), QueryUnderstanding(
        (QueryGoal("g1", "甲公司利润最高板块的发电量", subject_slot="segment"),)), True)
    tool = SearchTool([records(text=quote), records("c2")])
    def bridge(**kwargs):
        return extract_bridge_value(**kwargs, entity_names=("甲公司",), client=client)
    executor = ToolExecutor([tool], bridge_extractor=bridge)
    first = executor.execute_task(AgentTask("s1", query="甲公司利润最高的板块", goal_ids=("g1",),
                                            output_slot="segment"), context, {}, {})
    assert first.binding.value == "煤电板块"
    second = executor.execute_task(AgentTask("s2", query_template="甲公司{segment}发电量", goal_ids=("g1",),
                                             depends_on=("s1",), required_slots=("segment",)), context, {"s1": first}, {})
    assert second.status == "succeeded"
    assert tool.calls[-1].query == "甲公司煤电板块发电量"
    assert tool.calls[-1].subject_names == ("煤电板块",)


@pytest.mark.parametrize("initial_plan", [True, False])
def test_plan_fills_subjects_once_and_executor_uses_snapshot(initial_plan, monkeypatch):
    from dataclasses import replace
    question = "比较甲公司的煤电板块和水电板块收入"
    coal, water = "甲公司的煤电板块", "甲公司的水电板块"
    context, _, _ = run_plan(question, initial(goals=[
        goal("煤电收入", subjects=(coal,), source=question),
        goal("水电收入", subjects=(water, coal), id="g2", source=question),
    ], tasks=[task(query=question, goal_ids=("g1", "g2"))]))
    raw = decode_plan({"tasks": [task(query="收入", goal_ids=("g1", "g2"))]})
    plan = validate_plan(raw, context, {"search_knowledge_base"}, initial=initial_plan)
    assert raw.tasks[0].subjects is None
    assert plan.tasks[0].subjects == (coal, water)
    assert validate_plan(plan, context, {"search_knowledge_base"}) is plan
    # 模拟后续上下文变化：执行和缓存判断使用已校验任务快照。
    context.understanding = replace(context.understanding, goals=(replace(context.goals[0], subjects=("其他主体",)),))
    monkeypatch.setattr(type(context), "subject_names", lambda *args, **kwargs: pytest.fail("执行时不得再次继承主体"))
    tool, cache = SearchTool([records()]), {}
    executor = ToolExecutor([tool])
    executor.execute_task(plan.tasks[0], context, {}, cache)
    assert tool.calls[0].subject_names == (coal, water)
    assert executor.is_repeated_direct_plan(plan, context, cache)
