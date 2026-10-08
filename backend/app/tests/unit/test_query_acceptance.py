"""Acceptance regressions from baselineV2_v4; model outputs here are fixtures.

The saved run did not retain rejected model responses. These tests exercise
valid equivalents and invalid counterexamples through parsing AND execution,
not claim a replay of the missing provider outputs.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from service.core.query_contract import QueryGoal, parse_goals
from service.core.query_execution import execute_graph
from service.core.query_expansion import _extract_organizations, validate_query_preservation
from service.core.query_intent import _parse_query_analysis, analyze_query_intent
from service.core.query_planning import (
    QuerySubqueryPlan, RetrievalRoute, _validate_goal_coverage,
    build_retrieval_plan, prepare_retrieval_request,
)
from service.core.query_runtime import QueryRuntime, query_runtime
from service.core.query_validation import QueryValidationError, validate_goals
from service.core.sequential_retrieval import BridgeExtraction


pytestmark = pytest.mark.unit


class ScriptedClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls.append(kwargs)
        value = next(self.responses)
        if isinstance(value, Exception):
            raise value
        content = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=content), finish_reason="stop",
        )], usage=SimpleNamespace(total_tokens=10))


def analysis(goals, *, dependent=False):
    return dict(intent="complex_lookup", need_rewrite=False, target_count=len(goals),
                targets_independent=not dependent and len(goals) > 1,
                has_dependency=dependent, single_retrieval_likely_covers_all=False,
                reason="独立事实或依赖事实需要分别检索", goals=goals)


DIVIDENDS = "国电电力2025—2027年的现金分红底线是什么，2025年上半年拟如何分红？"
DIVIDEND_GOALS = [
    dict(id="floor", text="国电电力2025—2027年的现金分红底线", constraints=["国电电力", "2025—2027年", "现金分红底线"]),
    dict(id="interim", text="国电电力2025年上半年的分红安排", constraints=["国电电力", "2025年上半年", "分红"]),
]
VALUATION = "研报觉得国电电力未来三年能赚多少钱，对应估值贵不贵？"
VALUATION_GOALS = [
    dict(id="profit", text="国电电力未来三年的盈利预测", constraints=["国电电力", "未来三年", "赚多少钱"]),
    dict(id="value", text="国电电力未来三年的估值水平", constraints=["国电电力", "未来三年", "估值贵不贵"]),
]
FIGURES = "不要混淆图2和图3：哪张图显示2021年归母净利润为负，哪张图显示2025年只有第一、第二季度营业收入柱？"
FIGURE_GOALS = [
    dict(id="profit", text="在图2和图3中，哪张图显示2021年归母净利润为负？", constraints=["图2", "图3", "2021年", "归母净利润为负"]),
    dict(id="revenue", text="在图2和图3中，哪张图显示2025年只有第一、第二季度营业收入柱？", constraints=["图2", "图3", "2025年", "只有第一、第二季度营业收入柱"]),
]


@pytest.mark.parametrize("filename", [
    "guodian_power_eval_v2.jsonl",
    "power_reports_eval_v2_dev.jsonl",
    "power_reports_eval_v2_test.jsonl",
])
def test_every_benchmark_question_preserves_itself(filename):
    dataset = Path(__file__).resolve().parents[3] / "evals" / "data" / filename
    rows = [json.loads(line) for line in dataset.read_text().splitlines()]
    for row in rows:
        result = validate_query_preservation(row["question"], row["question"])
        assert result.valid, (row["id"], result)


@pytest.mark.parametrize("text", [
    "2025年上半年国电电力各电源板块归母净利润", "查询国电电力2025年的收入",
    "查找国电电力的财务费用", "国电电力和华能能源的营业收入",
])
def test_entity_extraction_only_returns_contiguous_names(text):
    entities = _extract_organizations(text)
    assert "国电电力" in entities
    assert all(entity in text and not entity.startswith(("查询", "查找", "2025")) for entity in entities)


@pytest.mark.parametrize(("original", "candidate"), [
    ("2025年上半年国电电力归母净利润", "查询国电电力2025年上半年归母净利润"),
    ("国电电力2026E固定资产是多少", "国电电力2026年固定资产预测值"),
    ("国电电力2025年每股收益", "国电电力2025 EPS"),
    ("2025年上半年营业收入", "2025H1营收"),
])
def test_equivalent_queries_are_accepted(original, candidate):
    assert validate_query_preservation(original, candidate).valid


@pytest.mark.parametrize(("question", "goals", "queries"), [
    (DIVIDENDS, DIVIDEND_GOALS, ["国电电力2025-2027年现金分红底线", "国电电力2025年上半年分红安排"]),
    (VALUATION, VALUATION_GOALS, ["国电电力未来三年盈利预测", "国电电力未来三年估值水平"]),
    (FIGURES, FIGURE_GOALS, [g["text"] for g in FIGURE_GOALS]),
])
def test_accepted_analysis_and_parallel_plan_reach_each_retrieval_node(question, goals, queries):
    client = ScriptedClient([analysis(goals), {"subqueries": [
        dict(id=g["id"], query=q, goal_ids=[g["id"]]) for g, q in zip(goals, queries)
    ]}])
    calls = []
    with query_runtime() as runtime:
        prepared = prepare_retrieval_request(question, client=client)
        assert prepared.analysis.source == "model"
        assert prepared.plan.mode == "parallel", prepared.plan
        assert prepared.plan.source == "model"
        results = execute_graph(prepared.plan.subqueries, original_question=question,
                                retrieve=lambda q, scope: calls.append(q) or {"chunks": [{"chunk_id": q}]})
    assert set(calls) == set(queries)
    assert all(node.status == "succeeded" for node in results)
    assert runtime.counts["model"] == 2


@pytest.mark.parametrize("constraint", ["贵不贵", "贵"])
def test_valuation_constraint_from_original_question_keeps_both_goals(constraint):
    goals = [VALUATION_GOALS[0], dict(VALUATION_GOALS[1],
                                    constraints=["国电电力", "未来三年", constraint])]
    client = ScriptedClient([analysis(goals)])
    prepared = prepare_retrieval_request(VALUATION, client=client)
    assert prepared.analysis.source == "model"
    assert len(prepared.analysis.goals) == 2
    assert prepared.plan.mode == "parallel"
    assert len(client.calls) == 1
    provenance = prepared.analysis.goals[1].constraint_provenance[-1]
    assert provenance["source_spans"][0]["quote"] == constraint


def test_goal_constraint_cannot_add_a_year_absent_from_the_question():
    with pytest.raises(QueryValidationError, match="goal_source_not_grounded"):
        parse_goals([dict(VALUATION_GOALS[1], constraints=["国电电力", "2028年"])], VALUATION)


def test_paraphrased_constraint_does_not_claim_to_quote_the_question():
    goal = parse_goals([dict(VALUATION_GOALS[1], constraints=["是否昂贵"])], VALUATION)[0]
    assert goal.constraint_provenance[0]["kind"] == "paraphrase"
    assert goal.constraint_provenance[0]["source_spans"] == ()


def test_constraint_cannot_move_a_year_to_a_different_metric():
    question = "2025年营业收入和2026年净利润"
    with pytest.raises(QueryValidationError, match="metric_time_rebound"):
        parse_goals([dict(id="revenue", text="2025年营业收入", constraints=["2026年营业收入"])], question)


def test_price_unit_basis_remains_compatible_with_guodian_015():
    question = "上半年卖电均价掉了多少，现在一兆瓦时大概多少钱？"
    goal = parse_goals([dict(id="price", text="现在每兆瓦时电价是多少",
                             constraints=["现在", "一兆瓦时", "多少钱"])], question)[0]
    assert goal.constraints == ("现在", "每兆瓦时", "多少钱")


def test_bridge_reference_becomes_slot_and_executes_with_grounded_value():
    question = "2025年上半年新增装机最多的是哪类电源？该类电源的累计装机和同期归母净利润如何？"
    goals = [
        dict(id="identify", text="2025年上半年新增装机最多的电源类型", constraints=["2025年上半年", "新增装机最多"]),
        dict(id="detail", text="该类电源的累计装机和同期归母净利润", constraints=["该类电源", "累计装机", "同期归母净利润"], depends_on=["identify"]),
    ]
    first = dict(id="first", query="2025年上半年新增装机最多的电源类型", output_slot="power_type", goal_ids=["identify"])
    second = dict(id="second", query_template="{power_type}同期累计装机容量和归母净利润",
                  depends_on=["first"], required_slots=["power_type"], goal_ids=["detail"], inherit_document_scope=True)
    client = ScriptedClient([analysis(goals, dependent=True), {"subqueries": [first, second]}])
    calls = []
    with query_runtime():
        prepared = prepare_retrieval_request(question, client=client)
        assert prepared.plan.mode == "sequential", prepared.plan
        results = execute_graph(prepared.plan.subqueries, original_question=question,
            retrieve=lambda q, scope: calls.append((q, scope)) or {"chunks": [{"chunk_id": "evidence", "doc_id": "report"}]},
            bridge_extractor=lambda **kw: BridgeExtraction(kw["slot"], "光伏", 1, "evidence", "report", "rule"))
    assert [r.status for r in results] == ["succeeded", "succeeded"]
    assert calls[-1] == ("光伏同期累计装机容量和归母净利润", ["report"])


@pytest.mark.parametrize("wrong", [
    "查找图2中显示2021年归母净利润为负的信息",
    "查找图3中显示2021年归母净利润为负的信息",  # even a lucky answer cannot be assumed
])
def test_unknown_figure_binding_is_rejected_before_retrieval(wrong):
    goals = [dict(FIGURE_GOALS[0], text=wrong), FIGURE_GOALS[1]]
    with pytest.raises(QueryValidationError, match="unresolved_relation_bound"):
        _parse_query_analysis(json.dumps(analysis(goals), ensure_ascii=False), FIGURES)


def test_bad_relationship_is_corrected_once_without_freezing_wrong_goals():
    wrong = analysis([dict(FIGURE_GOALS[0], text="图2的2021年归母净利润为负"), FIGURE_GOALS[1]])
    client = ScriptedClient([wrong, analysis(FIGURE_GOALS)])
    with query_runtime() as runtime:
        result = analyze_query_intent(FIGURES, client=client)
    assert result.source == "model"
    assert "哪张图" in result.goals[0].text
    events = [e for e in runtime.events if e["stage"] == "query_analysis_validation"]
    assert [e["status"] for e in events] == ["rejected", "accepted"]
    assert events[0]["rule"] == "unresolved_relation_bound"
    assert "unresolved_relation_bound" in client.calls[1]["messages"][-1]["content"]


def test_plan_correction_preserves_mode_and_executes_valid_nodes():
    goals = parse_goals(DIVIDEND_GOALS, DIVIDENDS)
    good = {"subqueries": [dict(id=g.id, query=g.text, goal_ids=[g.id]) for g in goals]}
    bad = {"subqueries": [dict(id=g.id, query=g.text) for g in goals]}
    client = ScriptedClient([bad, good])
    with query_runtime() as runtime:
        plan = build_retrieval_plan(DIVIDENDS, RetrievalRoute("parallel", "multi", "analysis"), goals=goals, client=client)
    assert plan.mode == "parallel" and plan.source == "model"
    assert len(client.calls) == 2
    assert [e["status"] for e in runtime.events if e["stage"] == "query_planning_validation"] == ["rejected", "accepted"]


def test_real_retrieval_entry_runs_parallel_plan_without_mocking_preparation(monkeypatch):
    from service.core import query_intent, query_planning, retrieval
    from service.core.evidence_sufficiency import EvidenceSufficiencyDecision

    client = ScriptedClient([analysis(DIVIDEND_GOALS), {"subqueries": [
        dict(id=g["id"], query=g["text"], goal_ids=[g["id"]]) for g in DIVIDEND_GOALS
    ]}])
    monkeypatch.setattr(query_intent, "get_query_intent_client", lambda: client)
    monkeypatch.setattr(query_planning, "get_query_intent_client", lambda: client)
    monkeypatch.setenv("RAG_QUERY_INTENT_ENABLED", "true")
    monkeypatch.setenv("RAG_PARALLEL_RETRIEVAL_ENABLED", "true")
    calls = []
    def retrieve(**kwargs):
        query = kwargs["question"]
        calls.append(query)
        return {"total": 1, "chunks": [{"chunk_id": query, "doc_id": "report",
                 "content_with_weight": query, "rerank_score": 0.9}], "doc_aggs": []}
    monkeypatch.setattr(retrieval, "get_retrieval_dealer", lambda: SimpleNamespace(retrieval=retrieve))
    monkeypatch.setattr(retrieval, "rerank_similarity", lambda q, docs: ([0.9] * len(docs), None))
    monkeypatch.setattr(retrieval, "check_evidence_sufficiency", lambda q, chunks, **kw:
        EvidenceSufficiencyDecision(True, "目标证据齐全", [], [1, 2], "model", len(chunks),
                                     [chunk["chunk_id"] for chunk in chunks]))
    result = retrieval.retrieve_raw_results("test-index", DIVIDENDS)
    assert result["retrieval_plan"]["source"] == "model"
    assert result["parallel_retrieval"]["applied"] is True
    assert set(calls) == {g["text"] for g in DIVIDEND_GOALS}
    assert {c["chunk_id"] for c in result["chunks"]} == set(calls)
    assert result["query_runtime"]["counts"] == {"model": 2, "retrieval": 2}


def test_valid_independent_goals_survive_repeated_plan_format_errors():
    client = ScriptedClient(["bad json", "bad json"])
    plan = build_retrieval_plan(DIVIDENDS, RetrievalRoute("parallel", "multi", "analysis"),
                                goals=parse_goals(DIVIDEND_GOALS, DIVIDENDS), client=client)
    assert plan.mode == "parallel" and plan.source == "rule"
    assert len(plan.subqueries) == 2
    assert "compiled validated goals" in plan.fallback_reason


def test_repeated_bad_analysis_discards_invalid_goals_and_records_reason():
    bad = analysis([dict(FIGURE_GOALS[0], text="图2的2021年归母净利润为负"), FIGURE_GOALS[1]])
    client = ScriptedClient([bad, bad])
    result = analyze_query_intent(FIGURES, client=client)
    assert result.source == "fallback" and not result.goals
    assert "unresolved_relation_bound" in result.validation_error
    assert len(client.calls) == 2


def test_correction_does_not_overrun_model_budget():
    client = ScriptedClient(["bad json"])
    with query_runtime(QueryRuntime(max_model_calls=1)):
        result = analyze_query_intent("国电电力2025年营业收入是多少？", client=client)
    assert result.source == "fallback" and len(client.calls) == 1
    assert "QueryBudgetExceeded" in result.validation_error


def test_truncated_analysis_gets_one_larger_output_budget():
    client = ScriptedClient([analysis(DIVIDEND_GOALS), analysis(DIVIDEND_GOALS)])
    create = client.chat.completions.create
    def truncated_first(**kwargs):
        response = create(**kwargs)
        if len(client.calls) == 1:
            response.choices[0].finish_reason = "length"
        return response
    client.chat.completions.create = truncated_first
    with query_runtime() as runtime:
        result = analyze_query_intent(DIVIDENDS, client=client)
    assert result.source == "model"
    assert client.calls[1]["max_tokens"] > client.calls[0]["max_tokens"]
    assert any(e.get("rule") == "response_truncated" for e in runtime.events)


def test_transport_failure_is_not_retried_as_semantic_correction():
    client = ScriptedClient([ConnectionError("offline")])
    result = analyze_query_intent(DIVIDENDS, client=client)
    assert result.source == "fallback" and len(client.calls) == 1


@pytest.mark.parametrize(("question", "texts", "rule"), [
    ("国电电力2024年利润和华能国际2025年利润", ["国电电力2025年利润", "华能国际2024年利润"], "subject_time_rebound"),
    ("国电电力2025年营业收入和2026年净利润", ["国电电力2026年营业收入", "国电电力2025年净利润"], "metric_time_rebound"),
    ("图2营业收入，图3归母净利润", ["图3营业收入", "图2归母净利润"], "metric_locator_rebound"),
    ("国电电力2025年上半年光伏发电量和上网电量同比增幅", ["国电电力2025年上半年光伏发电量同比增幅", "国电电力2025年上半年上网电量同比增幅"], "metric_scope_dropped"),
])
def test_wrong_local_bindings_are_rejected(question, texts, rule):
    with pytest.raises(QueryValidationError, match=rule):
        validate_goals(question, tuple(QueryGoal(f"g{i}", text) for i, text in enumerate(texts)))


@pytest.mark.parametrize(("original", "candidate"), [
    ("2026E营业收入", "2026年实际营业收入"),
    ("2025年上半年营业收入", "2025年全年营业收入"),
    ("国电电力2025年营收", "华能能源2025年营收"),
    ("先查归母净利润最高的电源类型", "查询煤电归母净利润"),
])
def test_acceptance_fixes_do_not_allow_invented_values_or_scopes(original, candidate):
    assert not validate_query_preservation(original, candidate).valid


def test_goal_local_company_without_known_suffix_remains_required():
    question = "华能国际2025年营业收入和2026年净利润是多少？"
    goals = parse_goals([
        dict(id="revenue", text="华能国际2025年营业收入", constraints=["华能国际", "2025年", "营业收入"]),
        dict(id="profit", text="华能国际2026年净利润", constraints=["华能国际", "2026年", "净利润"]),
    ], question)
    with pytest.raises(QueryValidationError, match="goal-local"):
        _validate_goal_coverage(question, [
            QuerySubqueryPlan("a", query="2025年营业收入", goal_ids=["revenue"]),
            QuerySubqueryPlan("b", query="华能国际2026年净利润", goal_ids=["profit"]),
        ], goals)


def test_reporting_period_cannot_move_to_another_metric_in_same_year():
    question = "2025年上半年营业收入和2025年下半年净利润"
    with pytest.raises(QueryValidationError, match="metric_period_rebound"):
        validate_goals(question, (
            QueryGoal("a", "2025年下半年营业收入"), QueryGoal("b", "2025年上半年净利润"),
        ))
