"""理解输出、事实需求、执行绑定和核验版本的边界回归。"""

from dataclasses import replace
from types import SimpleNamespace
import json

import pytest

from agent.financial_requirements import decode_financial_requirements, validate_financial_requirements
from agent.financial_bindings import bind_financial_requirements
from agent.planner import AgentPlanner
from agent.plans import AgentTask, AgentPlan, validate_plan
from agent.executor import TaskExecution
from domain.facts import FinancialRequirements, FactRequirement, CalculationRequirement
from service.core.entities.attribution import EntityRegistry, annotate_document
from service.core.query.contract import QueryGoal, QueryUnderstanding, QueryRequest
from service.core.query.context import QueryContext
from service.core.query.validation import QueryValidationError
from service.core.evidence.adapter import collection_from_records, assessment_from_record
from service.core.evidence.bindings import BridgeExtraction
from service.core.evidence.verification import verify_evidence
from service.core.finance.facts import assess_financial_facts
from tests.agent_helpers import ModelClient, initial, task

pytestmark = pytest.mark.unit


@pytest.fixture
def registry(monkeypatch):
    value = EntityRegistry({'version': 'financial-contract', 'entities': [
        {'entity_id': 'A', 'canonical_name': 'A公司', 'kind': 'company'},
        {'entity_id': 'B', 'canonical_name': 'B公司', 'kind': 'company'},
    ]})
    for module in ('agent.financial_requirements', 'agent.financial_bindings',
                   'service.core.finance.facts', 'service.core.evidence.assessment'):
        monkeypatch.setattr(module + '.get_entity_registry', lambda: value)
    return value


def fact(source, *, id='r1', entity='A公司', period='2023年', metric='营业收入', **fields):
    return dict(id=id, goal_id='g1', entity_name=entity, metric=metric, period=period,
                source_spans=[source], **fields)


def contract(*facts, calculations=(), unresolved=()):
    return dict(facts=list(facts), calculations=list(calculations), unresolved=list(unresolved))


def validate(question, payload, goals=None):
    goals = goals or (QueryGoal('g1', question, source_spans=(question,)),)
    return validate_financial_requirements(decode_financial_requirements(payload), goals, question,
                                          reference_date='2026-10-16')


def test_initial_call_generates_and_normalizes_fact_requirements(registry):
    question = 'A公司2023年合并口径营业收入是多少？'
    payload = initial(goals=[dict(id='g1', text=question, constraints=[], source_spans=[question])],
                      tasks=[task(query=question)], financial=contract(fact(question, statement_scope='consolidated')))
    client = ModelClient([payload])
    planner = AgentPlanner(client=client, model='test', tool_definitions=[{'function': {'name': 'search_knowledge_base'}}])
    context = QueryContext(QueryRequest.from_user(question), _resolved=True)
    result = planner.analyze_and_plan(context)
    assert len(client.calls) == 1
    requirement = result.understanding.financial.facts[0]
    assert requirement.entity_id == 'A' and requirement.period['year'] == 2023
    assert requirement.statement_scope == 'consolidated'
    assert result.understanding is context.understanding


@pytest.mark.parametrize('extra', [{'entity_id':'B'}, {'verification_status':'verified'}, {'period': {'year':2023}}])
def test_model_cannot_inject_server_fields_or_period_objects(registry, extra):
    question = 'A公司2023年营业收入是多少？'
    row = fact(question)
    row.update(extra)
    with pytest.raises(QueryValidationError):
        validate(question, contract(row))


def test_missing_financial_contract_triggers_initial_response_repair(registry):
    question = 'A公司2023年营业收入是多少？'
    goal = dict(id='g1', text=question, constraints=[], source_spans=[question])
    invalid = initial(goals=[goal], tasks=[task(query=question)])
    valid = initial(goals=[goal], tasks=[task(query=question)], financial=contract(fact(question)))
    client = ModelClient([invalid, valid])
    planner = AgentPlanner(client=client, model='test', tool_definitions=[{'function': {'name': 'search_knowledge_base'}}])
    context = QueryContext(QueryRequest.from_user(question), _resolved=True)
    result = planner.analyze_and_plan(context)
    assert len(client.calls) == 2 and result.understanding.financial.facts


@pytest.mark.parametrize('query', ['A公司2023年营业收入和净利润分别是多少？', 'A公司2022年和2023年营业收入分别是多少？', 'A公司和B公司2023年营业收入是多少？'])
def test_omitted_metric_period_or_company_is_rejected(registry, query):
    with pytest.raises(QueryValidationError):
        validate(query, contract(fact(query)))


def test_scope_is_local_and_not_borrowed_from_other_company(registry):
    first, second = 'A公司2023年合并口径营业收入是多少', 'B公司2023年营业收入是多少'
    question = first + '，' + second
    parsed = validate(question, contract(fact(first, statement_scope='consolidated'), fact(second, id='r2', entity='B公司')))
    assert [item.statement_scope for item in parsed.facts] == ['consolidated', 'unknown']
    with pytest.raises(QueryValidationError, match='scope_not_grounded'):
        validate(question, contract(fact(first, statement_scope='consolidated'),
                                    fact(second, id='r2', entity='B公司', statement_scope='consolidated')))


def test_company_list_explicitly_shares_scope(registry):
    query = 'A公司和B公司2023年合并口径营业收入'
    value = validate(query, contract(fact(query, statement_scope='consolidated'),
                                    fact(query, id='r2', entity='B公司', statement_scope='consolidated')))
    assert all(item.statement_scope == 'consolidated' for item in value.facts)


def test_relative_period_uses_request_date_and_invented_year_is_rejected(registry):
    query = 'A公司去年营业收入是多少？'
    value = validate(query, contract(fact(query, period='去年')))
    assert value.facts[0].period['year'] == 2025
    with pytest.raises(QueryValidationError, match='period_not_grounded'):
        validate(query, contract(fact(query, period='2024年')))


@pytest.mark.parametrize('search_query', ['A公司去年营业收入', 'A公司2025年营业收入'])
def test_relative_year_is_consistent_across_planning_execution_and_verification(registry, search_query):
    from agent.executor import ToolExecutor
    from tests.agent_helpers import SearchTool, records
    question = 'A公司去年营业收入是多少？'
    payload = initial(goals=[dict(id='g1', text=question, constraints=[], source_spans=[question])],
                      tasks=[task(query=search_query)], financial=contract(fact(question, period='去年')))
    context = QueryContext(replace(QueryRequest.from_user(question), reference_date='2026-10-16'), _resolved=True)
    tool = SearchTool([records(text='A公司2025年营业收入为80亿元。')])
    executor = ToolExecutor([tool])
    planner = AgentPlanner(client=ModelClient([payload]), model='test', tool_definitions=executor.definitions())
    planned = planner.analyze_and_plan(context)
    executions = executor.execute_plan(planned.plan, context, {}, {})
    assert tool.calls[0].query == 'A公司2025年营业收入'
    assert context.financial_requirements.facts[0].period['year'] == 2025
    assert verify_evidence(context, executions[0].result.evidence, require_semantic=False).sufficient


def test_unsupported_metric_is_explicit_and_cannot_bypass_verification(registry):
    query = 'A公司2023年资本充足率是多少？'
    with pytest.raises(QueryValidationError, match='unsupported_requirement_missing'):
        validate(query, None)
    value = validate(query, contract(unresolved=[dict(id='u1', goal_id='g1', metric_label='资本充足率',
                                                     reason='unsupported_metric', source_spans=[query])]))
    result = assess_financial_facts(value, collection_from_records([]))
    assert result.applicable and not result.sufficient


@pytest.mark.parametrize('kind', ['none', 'fact', 'unresolved'])
def test_financial_applicability_depends_on_requirements_without_evidence(registry, kind):
    query = 'A公司2023年营业收入是多少？'
    if kind == 'none':
        requirements = None
    elif kind == 'fact':
        requirements = validate(query, contract(fact(query)))
    else:
        query = 'A公司2023年资本充足率是多少？'
        requirements = validate(query, contract(unresolved=[dict(
            id='u1', goal_id='g1', metric_label='资本充足率',
            reason='unsupported_metric', source_spans=[query],
        )]))
    bundle = assess_financial_facts(requirements, collection_from_records([]))
    assert bundle.applicable is (kind != 'none')
    assert not bundle.sufficient
    assert not bundle.facts and not bundle.derived_evidence
    assert bool(bundle.missing_requirements) is (kind != 'none')
    assert 'verification_required' not in bundle.to_dict()


@pytest.mark.parametrize('legacy_flag', [None, False, True])
@pytest.mark.parametrize('has_unresolved', [False, True])
def test_legacy_financial_report_derives_applicability_from_actual_requirements(legacy_flag, has_unresolved):
    # 旧标记缺失或与实际需求矛盾时，也必须保留未解决需求，不能据此放行回答。
    record = {'unresolved_targets': [dict(id='u1', goal_id='g1', metric_label='资本充足率',
                                         reason='unsupported_metric')] if has_unresolved else []}
    if legacy_flag is not None:
        record['verification_required'] = legacy_flag
    bundle = assessment_from_record({'financial_fact_bundle': record}).facts
    assert bundle.applicable is has_unresolved
    assert not bundle.sufficient
    assert 'verification_required' not in bundle.to_dict()


def test_calculation_without_declared_fact_operands_is_rejected(registry):
    query = '计算A公司2022年和2023年营业收入增长率'
    calculation = dict(id='growth', goal_id='g1', operation='change_percent',
                       operand_ids=['current', 'base'], source_spans=[query])
    with pytest.raises(QueryValidationError, match='invalid_calculation_operands'):
        validate(query, contract(calculations=[calculation]))


def test_calculation_operands_keep_explicit_base_direction(registry):
    query = '计算A公司2022年和2023年营业收入增长率'
    rows = [fact(query, id='base', period='2022年'), fact(query, id='current')]
    calculation = dict(id='growth', goal_id='g1', operation='change_percent',
                       operand_ids=['current', 'base'], source_spans=[query])
    value = validate(query, contract(*rows, calculations=[calculation]))
    assert value.calculations[0].operand_ids == ('current', 'base')
    with pytest.raises(QueryValidationError, match='direction_reversed'):
        validate(query, contract(*rows, calculations=[{**calculation, 'operand_ids':['base', 'current']}]))


def test_quarter_comparison_uses_chronological_order(registry):
    query = '计算A公司2023年第二季度和2023年第三季度营业收入增长率'
    rows = [fact(query, id='base', period='2023年第二季度'),
            fact(query, id='current', period='2023年第三季度')]
    calculation = dict(id='growth', goal_id='g1', operation='change_percent',
                       operand_ids=['current', 'base'], source_spans=[query])
    assert validate(query, contract(*rows, calculations=[calculation])).calculations
    with pytest.raises(QueryValidationError, match='direction_reversed'):
        validate(query, contract(*rows, calculations=[{**calculation, 'operand_ids':['base', 'current']}]))


def test_calculator_rechecks_periods_after_runtime_binding(registry):
    facts = (FactRequirement('r1', 'g1', 'A', 'A公司', '营业收入',
                             {'year':2022, 'granularity':'year', 'label':'2022'}),
             FactRequirement('r2', 'g1', 'A', 'A公司', '营业收入',
                             {'year':2023, 'granularity':'year', 'label':'2023'}))
    requirements = FinancialRequirements(facts, (CalculationRequirement('c1', 'g1', 'change_percent', ('r1','r2')),))
    evidence = collection_from_records([dict(chunk_id='c1', doc_id='d1', content_with_weight=
        'A公司2022年实际合并口径营业收入为60人民币亿元。A公司2023年实际合并口径营业收入为80人民币亿元。')])
    result = assess_financial_facts(requirements, evidence)
    assert not result.sufficient
    assert result.derived_evidence[0]['reason'] == 'calculation_direction_reversed'


def test_disclosed_growth_rate_is_not_reinterpreted_as_calculation(registry):
    query = 'A公司2023年营业收入同比增长率是多少？'
    value = validate(query, contract(fact(query, measure_kind='growth_rate', comparison_basis='year_over_year')))
    assert not value.calculations
    with pytest.raises(QueryValidationError, match='requirement_missing'):
        validate(query, contract(fact(query)))


def test_qualitative_question_needs_no_numeric_contract(registry):
    assert validate('A公司净利润下降原因是什么？', None) is None


def binding_context():
    question = '先查研报推荐的公司，再查其2023年营业收入'
    requirement = FactRequirement('r1', 'g1', None, '', '营业收入',
                                  {'year':2023, 'granularity':'year', 'label':'2023'},
                                  entity_ref='company', binding_status='deferred', source_spans=(question,))
    return QueryContext(QueryRequest.from_user(question), QueryUnderstanding(
        (QueryGoal('g1', question),), financial=FinancialRequirements((requirement,))), True)


def test_dependency_binding_preserves_understanding_and_invalidates_report(registry):
    context = binding_context()
    evidence = collection_from_records([dict(chunk_id='c1', doc_id='d1', content_with_weight='推荐A公司。A公司2023年营业收入为80亿元。')])
    original = context.understanding
    previous = verify_evidence(context, evidence, require_semantic=False)
    task = AgentTask('s1', query='研报推荐公司', goal_ids=('g1',), output_slot='company')
    state = TaskExecution(task, 'succeeded', binding=BridgeExtraction('company','A公司',1,'c1','d1','rule',source_quote='推荐A公司'))
    bind_financial_requirements(context, evidence, {'s1':state})
    result = verify_evidence(context, evidence, previous=previous, require_semantic=False)
    assert result is not previous and result.sufficient
    assert context.understanding is original and original.financial.facts[0].entity_id is None
    assert context.financial_requirements.facts[0].entity_id == 'A'
    assert context.version != previous.query_version


def test_unproven_binding_cannot_fill_a_requirement(registry):
    context = binding_context()
    evidence = collection_from_records([dict(chunk_id='c1', doc_id='d1', content_with_weight='推荐A公司')])
    task = AgentTask('s1', query='公司', output_slot='company')
    state = TaskExecution(task, 'succeeded', binding=BridgeExtraction('company','B公司',1,'c1','d1','rule',source_quote='推荐B公司'))
    bind_financial_requirements(context, evidence, {'s1':state})
    assert context.financial_requirements.facts[0].binding_status == 'deferred'
    assert not context.financial_bindings


def test_plan_must_supply_declared_financial_slot(registry):
    context = binding_context()
    with pytest.raises(QueryValidationError, match='without_producer'):
        validate_plan(AgentPlan(), context, {'search_knowledge_base'}, initial=True)


def test_document_subject_requires_explicit_request_and_reliable_attribution(registry):
    query = '这份报告中2023年营业收入是多少？'
    value = validate(query, contract(fact(query, entity='', entity_ref='document_subject')))
    context = QueryContext(QueryRequest.from_user(query), QueryUnderstanding((QueryGoal('g1', query),), financial=value), True)
    raw = dict(chunk_id='c1', doc_id='d1', document_name='A公司.pdf', content_with_weight='A公司2023年营业收入为80亿元。')
    bind_financial_requirements(context, collection_from_records([raw]), {})
    assert context.financial_requirements.facts[0].binding_status == 'deferred'
    document = annotate_document([{'content_with_weight':'报告主体：A公司'}], 'A公司.pdf', registry)
    evidence = collection_from_records([{**raw, **document}])
    bind_financial_requirements(context, evidence, {})
    assert context.financial_requirements.facts[0].entity_id == 'A'
    assert verify_evidence(context, evidence, require_semantic=False).sufficient


def test_financial_verifier_does_not_parse_user_requirements(registry, monkeypatch):
    query = 'A公司2023年营业收入是多少？'
    value = validate(query, contract(fact(query)))
    monkeypatch.setattr('agent.financial_coverage.unresolved_financial_targets', lambda *args: pytest.fail('核验不能重做理解'))
    monkeypatch.setattr('agent.financial_coverage.requirement_metric_mentions', lambda *args, **kwargs: pytest.fail('核验不能重建要求'))
    evidence = collection_from_records([dict(chunk_id='c1', doc_id='d1', content_with_weight='A公司2023年营业收入为80亿元。')])
    assert assess_financial_facts(value, evidence).sufficient
    with pytest.raises(TypeError):
        assess_financial_facts(query, evidence)


def test_goal_scope_keeps_calculation_inputs_from_other_goals(registry):
    value = FinancialRequirements((FactRequirement('r1','g1','A','A公司','营业收入',None),
                                   FactRequirement('r2','g2','A','A公司','营业收入',None)),
                                  (CalculationRequirement('c1','g3','subtract',('r1','r2')),))
    context = QueryContext(QueryRequest.from_user('比较'), QueryUnderstanding(tuple(QueryGoal(f'g{i}','比较') for i in (1,2,3)), financial=value))
    scoped = context.for_goals(('g3',))
    assert len(scoped.financial_requirements.facts) == 2
    assert scoped.financial_requirements.calculations == value.calculations
    assert scoped.version != context.version


def test_replanned_output_can_fill_original_financial_reference(registry):
    context = binding_context()
    evidence = collection_from_records([dict(chunk_id='c1', doc_id='d1', content_with_weight='推荐A公司')])
    previous = AgentTask('s1', query='研报推荐的公司', goal_ids=('g1',), output_slot='company')
    retry = AgentTask('s2', query='研报推荐的公司2023年营业收入', goal_ids=('g1',),
                      output_slot='company_retry', binding_key='company')
    validate_plan(AgentPlan((retry,)), context, {'search_knowledge_base'}, previous=(previous,))
    state = TaskExecution(retry, 'succeeded', binding=BridgeExtraction('company_retry','A公司',1,'c1','d1','rule',source_quote='推荐A公司'))
    bind_financial_requirements(context, evidence, {'s1':TaskExecution(previous,'unresolved_binding'), 's2':state})
    assert context.financial_requirements.facts[0].entity_id == 'A'
    assert context.financial_bindings[0].slot == 'company'
    assert context.understanding.financial.facts[0].entity_ref == 'company'


def test_conflicting_bindings_keep_original_reference_unresolved(registry):
    context = binding_context()
    evidence = collection_from_records([dict(chunk_id='c1', doc_id='d1', content_with_weight='推荐A公司'),
                                        dict(chunk_id='c2', doc_id='d2', content_with_weight='推荐B公司')])
    executions = {}
    for index, company in enumerate(('A公司', 'B公司'), 1):
        task = AgentTask(f's{index}', query='公司', output_slot=f'company{index}', binding_key='company')
        executions[task.id] = TaskExecution(task,'succeeded',binding=BridgeExtraction(
            task.output_slot, company, 1, f'c{index}', f'd{index}', 'rule', source_quote='推荐'+company))
    bind_financial_requirements(context, evidence, executions)
    assert context.financial_requirements.facts[0].binding_status == 'deferred'
    assert not context.financial_bindings


def test_forecast_actual_are_bound_to_each_period(registry):
    query = 'A公司2023A和2024E营业收入分别是多少？'
    value = validate(query, contract(fact(query, period='2023A', value_kind='actual'),
                                    fact(query, id='r2', period='2024E', value_kind='forecast')))
    assert [item.value_kind for item in value.facts] == ['actual','forecast']
    with pytest.raises(QueryValidationError, match='scope'):
        validate(query, contract(fact(query, period='2023A', value_kind='forecast'),
                                 fact(query, id='r2', period='2024E', value_kind='forecast')))


def test_metric_company_year_relationship_cannot_be_swapped(registry):
    query = 'A公司2022年营业收入，B公司2023年营业收入'
    with pytest.raises(QueryValidationError, match='relation_changed'):
        validate(query, contract(fact(query), fact(query, id='r2', entity='B公司', period='2022年')))


def test_complete_agent_run_uses_initial_contract_for_financial_verification(registry, monkeypatch):
    from agent import loop
    from tests.agent_helpers import SearchTool, records
    question = 'A公司2023年营业收入是多少？'
    payload = initial(goals=[dict(id='g1', text=question, constraints=[], source_spans=[question])],
                      tasks=[task(query=question)], financial=contract(fact(question)))
    client = ModelClient([payload])
    tool = SearchTool([records(text='A公司2023年营业收入为80亿元。')])
    monkeypatch.setattr(loop, 'verify_evidence', lambda context, evidence, previous=None, **kwargs:
                        verify_evidence(context, evidence, previous=previous, require_semantic=False))
    context = QueryContext(QueryRequest.from_user(question), _resolved=True)
    run = loop.AgentLoop(client=client, model='test', tools=[tool]).run(context=context)
    assert run.stop_reason == 'evidence_ready' and run.assessment.sufficient
    assert len(client.calls) == 1 and len(tool.calls) == 1
    assert run.assessment.facts.requirements[0].requirement_id == 'r1'
    assert run.assessment.query_version == context.version


def test_calculation_goal_can_reference_facts_owned_by_other_goals(registry):
    from service.core.evidence.sufficiency import _bind_fact_support
    from domain.assessment import EvidenceAssessment
    facts = (FactRequirement('r1','first','A','A公司','营业收入',{'year':2022,'granularity':'year','label':'2022'}),
             FactRequirement('r2','second','A','A公司','营业收入',{'year':2023,'granularity':'year','label':'2023'}))
    requirements = FinancialRequirements(facts, (CalculationRequirement('c1','compare','subtract',('r2','r1')),))
    pending = assess_financial_facts(requirements, collection_from_records([]))
    assert pending.applicable and not pending.sufficient
    assert pending.derived_evidence[0]['reason'] == 'inputs_not_grounded'
    evidence = collection_from_records([dict(chunk_id='c1',doc_id='d1', content_with_weight='A公司2022年实际合并口径营业收入为60人民币亿元。A公司2023年实际合并口径营业收入为80人民币亿元。')])
    bundle = assess_financial_facts(requirements, evidence)
    assert bundle.applicable and bundle.sufficient and bundle.derived_evidence[0]['result'] == '20'
    assessment = EvidenceAssessment(True, '计算有依据', evaluated_chunk_ids=['c1'], supporting_chunk_indices=[1],
                                    goal_coverage=[dict(goal_id='compare', status='supported', supporting_chunk_ids=['c1'])])
    result = _bind_fact_support(assessment, bundle)
    assert len(result.goal_coverage[0]['supporting_fact_ids']) == 2
