"""混合核验的行为回归：规则漏识别可修复，来源伪造和服务故障不能被放行。"""

from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from domain.facts import FactRequirement, FinancialRequirements, UnresolvedFinancialRequirement
from service.core.evidence.adapter import collection_from_records
from service.core.entities.attribution import EntityRegistry
from service.core.finance.facts import assess_financial_facts, project_fact_context
from service.core.finance.model_extraction import review_candidates
from service.core.runtime.context import query_runtime, QueryBudgetExceeded

pytestmark = pytest.mark.unit


@pytest.fixture
def registry(monkeypatch):
    registry = EntityRegistry({'version': 'hybrid', 'entities': [
        {'entity_id': 'A', 'canonical_name': 'A公司', 'kind': 'company'},
        {'entity_id': 'B', 'canonical_name': 'B公司', 'kind': 'company'},
    ]})
    monkeypatch.setenv('RAG_FINANCIAL_VERIFICATION_MODE', 'hybrid')
    for module in ('service.core.finance.facts', 'service.core.entities.attribution', 'service.core.evidence.assessment'):
        monkeypatch.setattr(module + '.get_entity_registry', lambda: registry)
    return registry


def requirements(metric='营业收入', **changes):
    item = FactRequirement('r1', 'g1', 'B', 'B公司', metric,
                           {'year':2023, 'granularity':'year', 'label':'2023'})
    return FinancialRequirements((replace(item, **changes),))


def evidence(text):
    return collection_from_records([dict(chunk_id='c1', doc_id='d1', content_with_weight=text)])


def reference(sources, quote, *, kind='text', row=None, column=None):
    source = next(item for item in sources if item['kind'] == kind and quote in item['text']
                  and (row is None or item.get('row') == row) and (column is None or item.get('column') == column))
    start = source['text'].index(quote)
    return dict(source_id=source['source_id'], start=start, end=start + len(quote), quote=quote)


def candidate(sources, *, metric='营业收入', value='80', unit='亿元', **changes):
    row = dict(entity_id='B', metric=metric, period_text='2023年', raw_value=value, unit=unit,
               measure_kind='level', comparison_basis='not_applicable', field_evidence={
        key: [reference(sources, quote)] for key, quote in
        (('entity','B公司'), ('metric',metric), ('period','2023年'), ('value',value), ('unit',unit))})
    value_ref = row['field_evidence']['value'][0]
    source = next(item for item in sources if item['source_id'] == value_ref['source_id'])
    start = source['text'].rindex(value)
    value_ref.update(start=start, end=start + len(value))
    row.update(changes)
    return row


class Client:
    def __init__(self, extract=None, *, verdict='supported', failure=None, mutate_verdict=None):
        self.extract = extract or (lambda sources: [candidate(sources)])
        self.verdict, self.failure, self.mutate_verdict = verdict, failure, mutate_verdict
        self.calls = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        payload = json.loads(kwargs['messages'][1]['content'])
        self.calls.append(payload)
        if 'entities' in payload:
            response = {'candidates': self.extract(payload['sources'])}
        else:
            if self.failure:
                raise self.failure
            response = {'verdicts': [dict(fact_id=item['fact_id'], status=self.verdict,
                reason_code='source_relationship', references=[ref for refs in item['source']['field_evidence'].values() for ref in refs])
                for item in payload['candidates']]}
            if self.mutate_verdict:
                self.mutate_verdict(response)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(response)), finish_reason='stop')])


def assess(text, client, registry, **kwargs):
    return assess_financial_facts(requirements(**kwargs), evidence(text), registry=registry, allow_model=True, client=client)


def test_cross_sentence_relation_without_trigger_word_can_be_supported(registry):
    text = 'B公司发布年度报告。2023年营业收入录得80亿元。'
    client = Client()
    result = assess(text, client, registry)
    assert result.sufficient and len(client.calls) == 2
    assert result.requirement_results[0].status == 'supported'
    assert result.relationship_results[0].references
    assert '80亿元' in project_fact_context(evidence(text), result)[0].text


def test_unknown_relation_is_ambiguous_and_excluded_from_answer(registry):
    text = 'B公司发布年度报告。2023年营业收入录得80亿元。'
    result = assess(text, Client(verdict='unknown'), registry)
    assert not result.sufficient and result.verification_status == 'completed'
    assert result.requirement_results[0].status == 'ambiguous'
    assert not result.selected_fact_ids
    assert '80' not in project_fact_context(evidence(text), result)[0].text


@pytest.mark.parametrize('failure', [TimeoutError('timeout'), ValueError('invalid JSON'), QueryBudgetExceeded('budget')])
def test_model_failure_is_not_missing_material(registry, failure):
    result = assess('B公司发布年度报告。2023年营业收入录得80亿元。', Client(failure=failure), registry)
    assert not result.sufficient and result.verification_status == 'unavailable'
    assert result.diagnostics['failure_kind'] == type(failure).__name__
    assert not result.selected_fact_ids


@pytest.mark.parametrize('change', ['quote', 'value', 'entity', 'scope'])
def test_forged_candidate_is_rejected_before_relationship_call(registry, change):
    def extract(sources):
        item = candidate(sources)
        if change == 'quote':
            item['field_evidence']['metric'][0]['quote'] = '假的引文'
        elif change == 'value':
            item['raw_value'] = '180'
        elif change == 'entity':
            item['entity_id'] = 'A'
        else:
            item['currency'] = 'USD'
        return [item]
    client = Client(extract)
    result = assess('B公司发布年度报告。2023年营业收入录得80亿元。', client, registry)
    assert not result.sufficient and len(client.calls) == 1
    assert result.diagnostics['rejected_candidates']


def test_partial_number_is_rejected(registry):
    client = Client(lambda sources: [candidate(sources, value='80')])
    result = assess('B公司发布年度报告。2023年营业收入录得180亿元。', client, registry)
    assert not result.sufficient
    assert result.diagnostics['rejected_candidates'][0]['reason_code'] == 'partial_numeric_value'


def test_supported_verdict_must_cover_all_candidate_evidence(registry):
    client = Client(mutate_verdict=lambda result: result['verdicts'][0].update(references=[]))
    result = assess('B公司发布年度报告。2023年营业收入录得80亿元。', client, registry)
    assert result.verification_status == 'unavailable' and not result.sufficient


def test_empty_extraction_is_completed_missing_not_unavailable(registry):
    result = assess('B公司发布年度报告。', Client(lambda sources: []), registry)
    assert result.verification_status == 'completed'
    assert result.requirement_results[0].status == 'missing'


def test_clear_rule_fact_does_not_require_model(registry):
    client = Client()
    result = assess('B公司2023年营业收入为80亿元。', client, registry)
    assert result.sufficient and not client.calls


def test_scope_completion_replaces_same_value_rule_candidate(registry):
    def extract(sources):
        item = candidate(sources, statement_scope='consolidated')
        item['field_evidence']['statement_scope'] = [reference(sources, '合并口径')]
        return [item]
    result = assess('B公司2023年营业收入为80亿元。以上采用合并口径。', Client(extract), registry, statement_scope='consolidated')
    assert result.sufficient
    assert any(fact.verification_status == 'superseded' for fact in result.facts)
    assert len(result.selected_fact_ids) == 1


def test_requirement_and_source_changes_do_not_reuse_incomplete_extraction(registry):
    text = 'B公司披露业绩：2023年营业收入录得80亿元；净利润录得20亿元。'
    client = Client()
    with query_runtime():
        first = assess(text, client, registry)
        second = assess(text, client, registry)
        assert first.sufficient and second.sufficient and len(client.calls) == 2
        client.extract = lambda sources: [candidate(sources, metric='净利润', value='20')]
        third = assess(text, client, registry, metric='净利润')
        assert third.sufficient and len(client.calls) == 4
        assess(text.replace('录得80', '录得81'), client, registry, metric='净利润')
        assert len(client.calls) == 6


def test_relationship_failure_is_not_cached_as_empty_result(registry):
    text = 'B公司发布年度报告。2023年营业收入录得80亿元。'
    client = Client(failure=TimeoutError())
    with query_runtime():
        assert not assess(text, client, registry).sufficient
        client.failure = None
        assert assess(text, client, registry).sufficient
        assert len(client.calls) == 3  # 候选可复用，失败的关系结论不可复用。


def test_unsupported_metric_stays_explicit(registry):
    contract = FinancialRequirements(unresolved=(UnresolvedFinancialRequirement('u1','g1','资本充足率','unsupported_metric'),))
    result = assess_financial_facts(contract, evidence('B公司资本充足率为10%'), allow_model=True)
    assert result.requirement_results[0].status == 'unsupported' and not result.sufficient


def table_evidence(*, origin='deterministic'):
    from domain.evidence import EvidenceCollection, EvidenceTable, EvidenceTableRow, EvidenceStructure
    chunk = next(iter(evidence('原始财务表')))
    headers = ('指标', '本报告年度')
    cells = ('销售收入', '80')  # 有原始单元格，但常规行列规则无法确认指标与期间。
    row = EvidenceTableRow(headers, cells, 0, (0,1), headers, cells,
                           context='报告主体：B公司；本报告年度为2023年；销售收入对应营业收入。', preamble='单位：亿元')
    table = EvidenceTable(binding='bound', binding_source=origin, rows=(row,), caption='财务数据')
    return EvidenceCollection([replace(chunk, location=replace(chunk.location, table_id='t1'),
                                       structure=EvidenceStructure(kind='table', table=table))])


def table_candidate(sources):
    fields = {key: [reference(sources, quote, kind=kind)] for key, quote, kind in (
        ('entity','B公司','table_context'), ('metric','销售收入对应营业收入','table_context'),
        ('period','2023年','table_context'), ('value','80','table_cell'), ('unit','亿元','table_preamble'))}
    fields['period'].append(reference(sources, '本报告年度', kind='table_header', column=1))
    return dict(entity_id='B', metric='营业收入', period_text='2023年', raw_value='80', unit='亿元',
                measure_kind='level', comparison_basis='not_applicable', field_evidence=fields)


def test_table_relationship_can_use_original_cell_header_and_context(registry):
    chunks = table_evidence()
    client = Client(lambda sources: [table_candidate(sources)])
    result = assess_financial_facts(requirements(), chunks, registry=registry, allow_model=True, client=client)
    assert result.sufficient and len(client.calls) == 2
    selected = next(fact for fact in result.facts if fact.fact_id in result.selected_fact_ids)
    assert (selected.source['row'], selected.source['column'], selected.source['cell']) == (0,1,'80')
    assert '80亿元' in project_fact_context(chunks, result)[0].text


def test_model_reconstructed_grid_cannot_verify_itself(registry):
    client = Client(lambda sources: [table_candidate(sources)])
    result = assess_financial_facts(requirements(), table_evidence(origin='model'), registry=registry, allow_model=True, client=client)
    assert not result.sufficient and not client.calls
    assert result.requirement_results[0].reason_code == 'original_table_required'


def test_table_coordinate_forgery_is_rejected(registry):
    def extract(sources):
        item = table_candidate(sources)
        item['field_evidence']['value'][0]['source_id'] = 'invented_cell'
        return [item]
    client = Client(extract)
    result = assess_financial_facts(requirements(), table_evidence(), registry=registry, allow_model=True, client=client)
    assert not result.sufficient and len(client.calls) == 1


def test_ambiguous_raw_fact_is_kept_for_audit(registry):
    result = assess_financial_facts(requirements(), evidence('2023年营业收入为80亿元。'), registry=registry)
    assert result.facts and result.facts[0].entity_id is None
    assert result.requirement_results[0].status == 'ambiguous'


def test_context_limit_is_not_reported_as_absence(registry, monkeypatch):
    monkeypatch.setenv('RAG_FINANCIAL_FACT_MAX_CHUNKS', '1')
    chunks = evidence('B公司年度报告')
    chunks.merge(collection_from_records([dict(chunk_id='c2', doc_id='d1', content_with_weight='补充片段')]))
    result = assess_financial_facts(requirements(), chunks, registry=registry, allow_model=True, client=Client(lambda sources: []))
    assert result.requirement_results[0].status == 'ambiguous'
    assert result.requirement_results[0].reason_code == 'source_context_limit'


@pytest.mark.parametrize('failure, expected', [(TimeoutError(), 'verification_failed'),
                                              (QueryBudgetExceeded('budget'), 'budget_exhausted'),
                                              (None, 'clarification_required')])
def test_agent_distinguishes_ambiguity_from_verification_failure(registry, monkeypatch, failure, expected):
    from agent.loop import AgentLoop
    from service.core.query.context import QueryContext
    from service.core.query.contract import QueryRequest
    from service.core.evidence.verification import verify_evidence
    from tests.agent_helpers import ModelClient, SearchTool, initial, task
    monkeypatch.setattr('agent.financial_requirements.get_entity_registry', lambda: registry)
    question = 'B公司2023年营业收入是多少？'
    payload = initial(goals=[dict(id='g1', text=question, source_spans=[question])], tasks=[task(query=question)],
                      financial=dict(facts=[dict(id='r1',goal_id='g1',entity_name='B公司',metric='营业收入',period='2023年',source_spans=[question])], calculations=[],unresolved=[]))
    client = Client(failure=failure, verdict='unknown' if failure is None else 'supported')
    monkeypatch.setattr('agent.loop.verify_evidence', lambda context, chunks, **kwargs: verify_evidence(context, chunks, client=client, **kwargs))
    tool = SearchTool([[dict(chunk_id='c1',doc_id='d1',content_with_weight='B公司发布年度报告。2023年营业收入录得80亿元。')]])
    model = ModelClient([payload] + ([{'tasks': [], 'clarification': '请明确收入口径'}] if failure is None else []))
    run = AgentLoop(client=model, model='test',tools=[tool]).run(context=QueryContext(QueryRequest.from_user(question),_resolved=True))
    assert run.stop_reason == expected and len(tool.calls) == 1
    if failure is None:
        assert len(model.calls) == 2 and run.assessment.verification_status == 'completed'
        feedback = json.loads(model.calls[-1]['messages'][-1]['content'])['assessment']['financial_fact_bundle']
        assert feedback['requirement_results'][0]['status'] == 'ambiguous'
    else:
        assert len(model.calls) == 1 and run.assessment.verification_status == 'unavailable'


def test_model_extracted_approximate_number_stays_approximate(registry):
    text = 'B公司发布年度报告。2023年营业收入录得约80亿元。'
    result = assess(text, Client(), registry)
    assert result.sufficient
    fact = next(fact for fact in result.facts if fact.fact_id in result.selected_fact_ids)
    assert fact.value_qualifier == 'approximate'
    assert '约80亿元' in project_fact_context(evidence(text), result)[0].text


def test_new_sources_are_reviewed_after_bounded_first_batch(registry, monkeypatch):
    monkeypatch.setenv('RAG_FINANCIAL_FACT_MAX_CHUNKS', '1')
    first = evidence('B公司年度报告概述')
    second = collection_from_records([dict(chunk_id='c2', doc_id='d1', content_with_weight='B公司发布年度报告。2023年营业收入录得80亿元。')])
    client = Client(lambda sources: [candidate(sources)] if any('80' in row['text'] for row in sources) else [])
    with query_runtime():
        assert not assess_financial_facts(requirements(), first, registry=registry, allow_model=True, client=client).sufficient
        first.merge(second)
        result = assess_financial_facts(requirements(), first, registry=registry, allow_model=True, client=client)
        assert result.sufficient
        assert client.calls[1]['sources'][0]['chunk_id'] == 'c2'


def test_exact_local_pairing_cannot_be_overridden_by_model(registry):
    text = 'A公司2023年营业收入为120亿元。B公司2023年营业收入为80亿元。'
    client = Client(lambda sources: [candidate(sources, value='120')])
    result = review_candidates(evidence(text), requirements().facts, client, registry=registry)
    assert result.relationships[0].status == 'rejected'
    assert result.relationships[0].reason_code == 'explicit_local_pairing_conflict'


def test_default_hybrid_mode_and_legacy_configuration(monkeypatch):
    from service.core.finance.model_policy import verification_mode
    monkeypatch.delenv('RAG_FINANCIAL_VERIFICATION_MODE', raising=False)
    monkeypatch.delenv('RAG_FINANCIAL_FACT_MODEL_ENABLED', raising=False)
    assert verification_mode() == 'hybrid'
    monkeypatch.setenv('RAG_FINANCIAL_FACT_MODEL_ENABLED', 'false')
    assert verification_mode() == 'rules'
    monkeypatch.setenv('RAG_FINANCIAL_VERIFICATION_MODE', 'hybrid')
    assert verification_mode() == 'hybrid'


def test_model_policy_change_invalidates_verified_report(registry, monkeypatch):
    from service.core.evidence.assessment import verification_policy_version
    before = verification_policy_version()
    monkeypatch.setenv('RAG_FINANCIAL_FACT_MAX_CHUNKS', '5')
    assert verification_policy_version() != before


def test_evaluation_keeps_business_states_separate_from_failures():
    from evaluation.retrieval import aggregate_query_metrics
    results = [dict(retrieval=dict(evidence_sufficiency=dict(financial_fact_bundle=dict(
        requirement_results=[dict(status='ambiguous'), dict(status='unsupported')],
        relationship_results=[dict(status='unknown')], verification_status='unavailable'))))]
    summary = aggregate_query_metrics(results)
    assert summary['financial_requirement_statuses'] == {'ambiguous': 1, 'unsupported': 1}
    assert summary['financial_relationship_statuses'] == {'unknown': 1}
    assert summary['financial_verification_unavailable_count'] == 1


def test_calculation_failure_preserves_actionable_category():
    from service.core.finance.calculations import calculation_status
    assert calculation_status({'status': 'unresolved', 'reason': 'inputs_not_grounded'}, ['ambiguous']) == 'ambiguous'
    assert calculation_status({'status': 'unresolved', 'reason': 'inputs_not_grounded'}, ['conflict']) == 'conflict'
    assert calculation_status({'status': 'unresolved', 'reason': 'zero_denominator'}, ['supported']) == 'unsupported'


def test_unit_cannot_be_sliced_out_of_larger_unit(registry):
    client = Client(lambda sources: [candidate(sources, unit='万元')])
    result = assess('B公司发布年度报告。2023年营业收入录得80百万元。', client, registry)
    assert not result.sufficient and len(client.calls) == 1
    assert result.diagnostics['rejected_candidates'][0]['reason_code'] == 'unit_not_grounded'


def test_forecast_period_marker_survives_model_normalization(registry):
    def extract(sources):
        fields = {key: [reference(sources, quote)] for key, quote in (
            ('entity','B公司'), ('metric','营业收入'), ('period','2023E'), ('value','80'), ('unit','亿元'))}
        return [dict(entity_id='B', metric='营业收入', period_text='2023E', raw_value='80', unit='亿元',
                     field_evidence=fields)]
    result = assess('B公司发布预期。2023E营业收入录得80亿元。', Client(extract), registry)
    assert result.sufficient
    assert next(fact for fact in result.facts if fact.fact_id in result.selected_fact_ids).value_kind == 'forecast'


@pytest.mark.parametrize('text, supported', [
    ('B公司发布年度报告，2023年营业收入录得，80亿元。', True),
    ('B公司发布年度报告，2023年营业收入录得80,000亿元。', False),
])
def test_numeric_boundary_distinguishes_punctuation_and_thousands_separator(registry, text, supported):
    result = assess(text, Client(), registry)
    assert result.sufficient is supported
    if not supported:
        assert result.diagnostics['rejected_candidates'][0]['reason_code'] == 'partial_numeric_value'


def test_extraction_failure_does_not_claim_no_matching_fact(registry):
    def fail(sources):
        raise TimeoutError('extraction unavailable')
    result = assess('B公司发布年度报告。2023年营业收入录得80亿元。', Client(fail), registry)
    assert result.verification_status == 'unavailable'
    assert result.requirement_results[0].status == 'ambiguous'
    assert result.requirement_results[0].reason_code == 'verification_unavailable'
    assert all('没有找到' not in message for message in result.missing_requirements)


def test_unmapped_table_rows_require_original_structure(registry):
    from domain.evidence import EvidenceCollection
    chunk = next(iter(table_evidence()))
    table = chunk.structure.table
    invalid = replace(table.rows[0], original_cells=())
    chunks = EvidenceCollection([replace(chunk, structure=replace(chunk.structure, table=replace(table, rows=(invalid,))))])
    client = Client()
    result = assess_financial_facts(requirements(), chunks, registry=registry, allow_model=True, client=client)
    assert not client.calls
    assert result.requirement_results[0].reason_code == 'original_table_required'


def test_context_limit_does_not_erase_already_confirmed_conflict(registry, monkeypatch):
    monkeypatch.setenv('RAG_FINANCIAL_FACT_MAX_CHUNKS', '1')
    chunks = collection_from_records([
        dict(chunk_id='c1',doc_id='d1',content_with_weight='B公司2023年实际合并口径营业收入为80人民币亿元。'),
        dict(chunk_id='c2',doc_id='d2',content_with_weight='B公司2023年实际合并口径营业收入为90人民币亿元。'),
    ])
    result = assess_financial_facts(requirements(), chunks, registry=registry, allow_model=True, client=Client(lambda sources: []))
    assert result.diagnostics['source_limit_reached']
    assert result.requirement_results[0].status == 'conflict'
