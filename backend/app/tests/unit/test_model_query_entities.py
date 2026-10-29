"""模型实体结果贯穿理解、校验、执行和检索；不访问模型或搜索服务。"""
from dataclasses import replace
import json

import pytest

from agent.executor import ToolExecutor
from agent.planner import AgentPlanner
from agent.plans import decode_plan, validate_plan
from agent.understanding import decode_understanding, validate_understanding
from service.core.query.context import prepare_query_context
from service.core.query.contract import RetrievalTask
from service.core.query.entities import decode_entities, ground_entities
from service.core.query.validation import QueryValidationError
from service.core.retrieval.entities import company_hints_from_names
from tests.agent_helpers import ModelClient, SearchTool, initial, task, records

pytestmark = pytest.mark.unit


def entity(name, ref="e1", quote=None, **kwargs):
    return dict(ref=ref, mention=name, source_quote=quote or name, **kwargs)


def goal(text, refs=("e1",), id="g1", source=None, constraints=(), **kwargs):
    return dict(id=id, text=text, source_spans=[source or text], constraints=list(constraints), subject_refs=list(refs), **kwargs)


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
        goals=[goal(query, source=question)], tasks=[task(query=query)],
        entities=[entity("润本股份")]))
    assert len(client.calls) == 1
    tool = SearchTool([records()])
    result = ToolExecutor([tool]).execute_task(decision.plan.tasks[0], context, {}, {})
    assert result.status == "succeeded"
    assert tool.calls[0].query == query


@pytest.mark.parametrize("prefix", ["为什么", "请帮我分析一下", "我想了解一下"])
def test_natural_language_prefix_does_not_become_company(prefix, monkeypatch):
    from service.core.query import preservation
    class NoRegex:
        def finditer(self, text):
            raise AssertionError("Agent must not infer company boundaries with suffix regex")
    monkeypatch.setattr(preservation, "ORGANIZATION_PATTERN", NoRegex())
    question = prefix + "国电电力2025年上半年营业收入下降" + ("的原因" if prefix != "为什么" else "")
    text = "分析国电电力2025年上半年营业收入下降的原因"
    context, decision, client = run_plan(question, initial(
        goals=[goal(text, source=question)], tasks=[task(query=text)],
        entities=[entity("国电电力", quote=question)]))
    assert len(client.calls) == 1
    item = context.understanding.entities[0]
    assert question[item.start:item.end] == "国电电力"
    assert item.entity_id == "company:cn:600795"
    tool = SearchTool([records()])
    result = ToolExecutor([tool]).execute_task(decision.plan.tasks[0], context, {}, {})
    assert result.status == "succeeded"
    assert tool.calls[0].company_hints.entity_ids == ("company:cn:600795",)


def test_unregistered_subsidiary_does_not_inherit_parent_identity():
    name = "国电电力新星科技有限公司"
    question = f"为什么{name}的项目延期"
    context, decision, _ = run_plan(question, initial(
        goals=[goal(f"分析{name}的项目延期原因", source=question)],
        tasks=[task(query=f"{name}的项目延期原因")], entities=[entity(name)]))
    assert context.understanding.entities[0].entity_id is None
    tool = SearchTool([records()])
    ToolExecutor([tool]).execute_task(decision.plan.tasks[0], context, {}, {})
    scope = tool.calls[0].company_hints
    assert scope.status == "unregistered_company"
    assert scope.entity_names == (name,)
    assert not scope.entity_ids


def test_publisher_is_not_used_as_research_subject_filter():
    question = "国信证券关于国电电力的研报观点"
    context, decision, _ = run_plan(question, initial(
        goals=[goal(question, publisher_refs=["p1"])], tasks=[task(query=question)],
        entities=[entity("国电电力"), entity("国信证券", "p1")]))
    tool = SearchTool([records()])
    ToolExecutor([tool]).execute_task(decision.plan.tasks[0], context, {}, {})
    assert tool.calls[0].company_hints.entity_names == ("国电电力",)


@pytest.mark.parametrize("change", [
    {"mention": "华能国际"}, {"source_quote": "不存在的原文"},
    {"entity_id": "company:cn:600795"}, {"occurrence": True},
])
def test_invalid_model_entities_are_rejected(change):
    row = {**entity("国电电力"), **change}
    with pytest.raises(QueryValidationError):
        ground_entities(decode_entities([row]), "国电电力项目进展")


def test_repeated_mentions_have_server_computed_positions():
    question = "国电电力项目；国电电力公告"
    result = ground_entities(decode_entities([entity("国电电力", occurrence=1)]), question)
    assert result[0].start == question.rindex("国电电力")
    assert result[0].end == result[0].start + 4


def test_ambiguous_mention_offsets_do_not_block_retrieval():
    question = "国电电力的项目和国电电力的公告"
    context, decision, client = run_plan(question, initial(
        goals=[goal("项目和公告", source=question)],
        entities=[entity("国电电力", quote=question)], tasks=[task(query="项目和公告")]))
    item = context.understanding.entities[0]
    assert (item.start, item.end) == (-1, -1)
    assert item.status == "resolved"
    assert item.entity_id == "company:cn:600795"
    assert item.source_quote == question
    assert len(client.calls) == 1
    tool = SearchTool([records()])
    result = ToolExecutor([tool]).execute_task(decision.plan.tasks[0], context, {}, {})
    assert result.status == "succeeded"
    assert tool.calls[0].company_hints.entity_names == ("国电电力",)


def test_subject_can_publish_its_own_report_with_unused_background_entity():
    question = "背景是华能国际的项目，查询国电电力发布的项目公告"
    context, decision, client = run_plan(question, initial(
        goals=[goal("项目公告", source="国电电力发布的项目公告", publisher_refs=["e1"])],
        entities=[entity("国电电力"), entity("华能国际", "e2")],
        tasks=[task(query="项目公告")]))
    assert len(client.calls) == 1
    assert context.goals[0].subject_refs == context.goals[0].publisher_refs == ("e1",)
    tool = SearchTool([records()])
    ToolExecutor([tool]).execute_task(decision.plan.tasks[0], context, {}, {})
    assert tool.calls[0].company_hints.entity_names == ("国电电力",)


def test_entity_mentioned_without_role_is_diagnostic_only(caplog):
    question = "国电电力项目进展"
    with caplog.at_level("INFO", logger="service.core.query.entities"):
        context, _, client = run_plan(question, initial(
            goals=[goal(question, refs=())], entities=[entity("国电电力")], tasks=[task(query=question)]))
    assert len(client.calls) == 1
    assert "undeclared_goal_entity" in caplog.text
    assert context.goals[0].subject_refs == ()


@pytest.mark.parametrize("query", ["项目建设状态", "华能国际2028年项目进展"])
def test_goal_and_search_wording_do_not_trigger_semantic_repair(query):
    question = "国电电力2024年项目进展"
    context, decision, client = run_plan(question, initial(
        goals=[goal("项目建设状态", source=question, constraints=["2028年项目进展"])],
        entities=[entity("国电电力")], tasks=[task(query=query)]))
    assert len(client.calls) == 1
    assert context.goals[0].constraints == ("2028年项目进展",)
    tool = SearchTool([records()])
    result = ToolExecutor([tool]).execute_task(decision.plan.tasks[0], context, {}, {})
    assert result.status == "succeeded"
    assert tool.calls[0].query == query
    assert tool.calls[0].company_hints.entity_names == ("国电电力",)


@pytest.mark.parametrize("role", ["subject_refs", "publisher_refs"])
def test_unknown_entity_references_still_fail(role):
    question = "国电电力项目进展"
    row = goal(question)
    row[role] = ["missing"]
    payload = initial(goals=[row], entities=[entity("国电电力")])["understanding"]
    with pytest.raises(QueryValidationError, match="invalid_goal_entity_binding"):
        validate_understanding(decode_understanding(json.dumps(payload)), question)


def test_ambiguous_mention_still_requires_a_real_quote_occurrence():
    question = "国电电力项目和国电电力公告"
    with pytest.raises(QueryValidationError, match="entity_not_grounded"):
        ground_entities(decode_entities([entity("国电电力", quote=question, occurrence=1)]), question)


def test_missing_entity_contract_cannot_fall_back_to_regex():
    payload = initial()["understanding"]
    del payload["entities"]
    with pytest.raises(QueryValidationError, match="invalid_entities"):
        decode_understanding(json.dumps(payload))


def test_omitted_registered_company_does_not_trigger_rule_based_rejection():
    payload = initial(goals=[goal("项目进展", refs=())], entities=[])["understanding"]
    value = validate_understanding(decode_understanding(json.dumps(payload)), "国电电力项目进展")
    assert value.entities == ()
    assert value.goals[0].text == "项目进展"


def test_model_goal_relations_are_preserved_without_rule_based_rejection():
    question = "国电电力2024年营业收入下降原因；华能国际2023年净利润下降原因"
    payload = initial(goals=[
        goal("华能国际2024年营业收入下降原因", refs=("e2",), source=question.split("；")[0]),
        goal("国电电力2023年净利润下降原因", refs=("e1",), id="g2", source=question.split("；")[1]),
    ], entities=[entity("国电电力"), entity("华能国际", "e2")])["understanding"]
    value = validate_understanding(decode_understanding(json.dumps(payload)), question)
    assert [g.subject_refs for g in value.goals] == [("e2",), ("e1",)]
    assert [g.text for g in value.goals] == [g["text"] for g in payload["goals"]]


def test_single_comparison_goal_can_filter_each_company_separately():
    question = "比较国电电力与华能国际的项目进展"
    context, decision, _ = run_plan(question, initial(
        goals=[goal(question, refs=("e1", "e2"))], entities=[entity("国电电力"), entity("华能国际", "e2")],
        tasks=[task(query="国电电力项目进展", subject_refs=["e1"]),
               task(id="s2", query="华能国际项目进展", subject_refs=["e2"])]))
    tool = SearchTool([records(), records("c2")])
    executor = ToolExecutor([tool])
    for node in decision.plan.tasks:
        executor.execute_task(node, context, {}, {})
    assert [call.company_hints.entity_names for call in tool.calls] == [("国电电力",), ("华能国际",)]
    bad = decode_plan({"tasks": [task(query="华能国际项目进展", subject_refs=["missing"])]})
    with pytest.raises(QueryValidationError, match="invalid_task_subjects"):
        validate_plan(bad, context, {"search_knowledge_base"})


def test_local_pronoun_is_resolved_by_planner_not_regex():
    question = "介绍国电电力，并说明它的项目进展"
    context, _, client = run_plan(question, initial(
        goals=[goal("介绍国电电力并说明国电电力的项目进展", source=question)],
        tasks=[task(query="国电电力介绍和项目进展")], entities=[entity("国电电力")]))
    assert len(client.calls) == 1
    assert context.request.user_question == question
    assert context.request.resolved_question == question
    assert context.request.clarification is None
    assert not context.request.bindings  # 规划前没有用后缀规则替换原文


def test_unknown_subject_is_bound_only_after_evidence():
    from service.core.evidence.bindings import BridgeExtraction
    question = "先查研报推荐的公司，再查它的公告"
    context, decision, _ = run_plan(question, initial(goals=[
        goal("研报推荐的公司", refs=(), source="研报推荐的公司"),
        goal("它的公告", refs=(), id="g2", depends_on=["g1"], subject_slot="company"),
    ], entities=[], tasks=[task(query="研报推荐的公司", output_slot="company"),
        task(id="s2", query="", query_template="{company}公告", depends_on=["s1"],
             required_slots=["company"], goal_ids=("g2",))]))
    tool = SearchTool([records(text="推荐国电电力"), records("c2")])
    def bridge(**kwargs):
        return BridgeExtraction(slot="company", value="国电电力", source_chunk_index=1, source="model",
                                source_chunk_id="c1", document_id="doc1",
                                source_quote="推荐国电电力")
    executor = ToolExecutor([tool], bridge_extractor=bridge)
    first = executor.execute_task(decision.plan.tasks[0], context, {}, {})
    assert first.binding
    executor.execute_task(decision.plan.tasks[1], context, {"s1": first}, {})
    assert tool.calls[0].company_hints.entity_names == ()
    assert tool.calls[1].company_hints.entity_names == ("国电电力",)
    assert tool.calls[1].company_hints.resolved


@pytest.mark.parametrize("names", [(), ("国电电力新星科技有限公司",), ("国电电力",)])
def test_pipeline_uses_supplied_scope_even_when_empty_or_unregistered(names, monkeypatch):
    from service.core.retrieval import pipeline
    seen = []
    class Dealer:
        def retrieval(self, **kwargs):
            seen.append(kwargs["options"].company_hints)
            return {"chunks": [], "total": 0}
    monkeypatch.setattr(pipeline, "get_retrieval_dealer", lambda: Dealer())
    def forbidden(*args, **kwargs):
        raise AssertionError("must not re-extract company from search text")
    monkeypatch.setattr(pipeline, "resolve_company_hints", forbidden)
    scope = company_hints_from_names(names)
    pipeline.retrieve_evidence("1", "华能国际公告", task=RetrievalTask("华能国际公告", ("g1",), company_hints=scope))
    assert seen == [scope]




def test_ambiguous_registered_alias_is_not_assigned_an_arbitrary_id():
    from service.core.entities.registry import EntityRegistry
    registry = EntityRegistry({"version": "test", "entities": [
        {"entity_id": f"company:{key}", "canonical_name": f"星河{key}公司", "kind": "company", "aliases": ["星河"]}
        for key in ("甲", "乙")
    ]})
    value = ground_entities(decode_entities([entity("星河")]), "星河的项目进展", registry=registry)[0]
    assert value.status == "ambiguous" and value.entity_id is None
    scope = company_hints_from_names(("星河",), registry=registry)
    assert scope.status == "ambiguous_company" and not scope.resolved


def test_subject_discovery_task_does_not_require_its_own_output():
    from agent.plans import AgentTask
    from service.core.query.contract import QueryUnderstanding, QueryGoal, QueryRequest
    from service.core.query.context import QueryContext
    from service.core.evidence.bindings import BridgeExtraction
    context = QueryContext(QueryRequest.from_user("先查推荐公司，再查营业收入"), QueryUnderstanding(
        (QueryGoal("g1", "推荐公司及其营业收入", subject_slot="company"),), entity_names=()), True)
    tool = SearchTool([records(text="推荐国电电力"), records("c2")])
    executor = ToolExecutor([tool], bridge_extractor=lambda **kw: BridgeExtraction(
        "company", "国电电力", 1, "c1", "doc1", "model", "推荐国电电力"))
    first = executor.execute_task(AgentTask("s1", query="推荐公司", goal_ids=("g1",), output_slot="company"), context, {}, {})
    assert first.status == "succeeded" and first.binding
    second = executor.execute_task(AgentTask("s2", query_template="{company}营业收入", goal_ids=("g1",),
                                             depends_on=("s1",), required_slots=("company",)), context, {"s1": first}, {})
    assert second.status == "succeeded"
    assert tool.calls[-1].company_hints.entity_names == ("国电电力",)


def test_context_model_can_prefer_current_question_over_old_company():
    from service.core.query.contract import QueryRequest, resolve_request
    request = QueryRequest.from_user("介绍国电电力，再介绍它的项目", [{"role": "user", "content": "华能国际"}])
    client = ModelClient([{"bindings": [dict(mention="它", value="国电电力", source_turn=-1, source_quote="国电电力")]}])
    resolved = resolve_request(request, client=client)
    assert resolved.resolved_question == "介绍国电电力，再介绍国电电力的项目"
    assert resolved.bindings[0].source_turn == -1


def test_replan_can_mention_another_company_without_changing_entity_filter():
    question = "国电电力项目进展"
    context, decision, _ = run_plan(question, initial(goals=[goal(question)], entities=[entity("国电电力")],
                                                     tasks=[task(query=question)]))
    good = decode_plan({"tasks": [task(id="s2", query="国电电力项目进展完整报告")]})
    validate_plan(good, context, {"search_knowledge_base"}, previous=decision.plan.tasks)
    expanded = decode_plan({"tasks": [task(id="s2", query="国电电力及华能国际项目进展")]})
    validate_plan(expanded, context, {"search_knowledge_base"}, previous=decision.plan.tasks)
    tool = SearchTool([records()])
    result = ToolExecutor([tool]).execute_task(expanded.tasks[0], context, {}, {})
    assert result.status == "succeeded"
    assert tool.calls[0].company_hints.entity_names == ("国电电力",)
