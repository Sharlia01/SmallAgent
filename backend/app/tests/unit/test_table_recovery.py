"""容量遗漏行从检索快照进入补查，再核验；不访问存储或模型 API。"""
import json
from dataclasses import replace

import pytest

from agent import loop
from agent.executor import ToolExecutor
from agent.loop import AgentLoop
from agent.plans import AgentTask
from agent.tools.rag_search import RagSearchTool
from service.core.evidence.table_retrieval_adapter import context_record
from service.core.evidence.adapter import structure_record
from service.core.retrieval.result import adapt_retrieval_records
from service.core.retrieval.table_context import build_table_context
from service.core.query.context import prepare_query_context
from tests.agent_helpers import ModelClient, initial, task, coverage_verifier
from tests.unit.test_table_candidates import row, candidates

pytestmark = pytest.mark.unit


def make_result(monkeypatch):
    from service.core.retrieval import table_context
    records = [row(1), row(2)]
    records[1]['table_cells_slice_obj']['rows'][0]['cells'] = ['PE', '23.4567']
    records[1]['table_semantic_text_with_weight'] = '指标：PE；2025E：23.4567'
    records[1]['content_with_weight'] = 'PE 23.4567'
    grouped = candidates(records)
    candidate = grouped.tables['t1-1']
    parent = {'table_id': 't1', 'cells_json': {'binding_status': 'bound', 'caption': '财务表（百万元）',
        'data_rows': [
            {'row_index': 1, 'headers': ['公司', '指标', '2025E'], 'cells': ['甲公司', '收入', '100'], 'preamble': '单位：百万元'},
            {'row_index': 2, 'headers': ['公司', '指标', '2025E'], 'cells': ['乙公司', 'PE', '23.4567'], 'context': '估值指标', 'preamble': '单位：倍'},
            {'row_index': 3, 'headers': ['公司', '指标', '2025E'], 'cells': ['丙公司', '未命中行', '999']},
        ]}}
    # 只让多行窗口超预算，完整单行可保留。
    monkeypatch.setenv('RAG_TABLE_CONTEXT_MAX_TOKENS', '50')
    monkeypatch.setattr(table_context, 'num_tokens_from_string', lambda text: 100 if '23.4567' in text and '100' in text else 10)
    context = build_table_context(candidate, '收入和PE', table_object=parent)
    assert context.diagnostics['omitted_matched_row_indices'] == (2,)
    raw = context_record(context)
    result = adapt_retrieval_records([raw], {})
    return result, parent


def test_omitted_values_are_private_and_recovery_preserves_full_row(monkeypatch):
    result, parent = make_result(monkeypatch)
    assert len(result.recoverable_evidence) == 1
    recovered = result.recoverable_evidence[0]
    assert recovered.structure.table.rows[0].original_cells == ('乙公司', 'PE', '23.4567')
    assert recovered.structure.table.rows[0].preamble == '单位：倍'
    assert recovered.structure.table.rows[0].context == '估值指标'
    assert '23.4567' not in json.dumps(structure_record(next(iter(result.evidence)).structure), ensure_ascii=False)
    assert '23.4567' not in next(iter(result.evidence)).text
    parent['cells_json']['data_rows'][1]['cells'][2] = '999'
    assert recovered.structure.table.rows[0].original_cells[-1] == '23.4567'


def test_agent_recovers_then_reverifies_without_another_search(monkeypatch):
    result, _ = make_result(monkeypatch)
    ref = result.recoverable_evidence[0].id
    searches = []
    def retrieve(*args, **kwargs):
        searches.append(args)
        return result
    tool = RagSearchTool(1, retriever=retrieve)
    calls = []
    monkeypatch.setattr(loop, 'verify_evidence', coverage_verifier(2, calls))
    client = ModelClient([initial(), {'tasks': [task(id='restore', tool='restore_table_rows', query=ref)]}])
    run = AgentLoop(client=client, model='test', tools=[tool]).run(context=prepare_query_context('员工培训要求'))
    assert run.stop_reason == 'evidence_ready'
    assert len(searches) == 1
    assert [len(evidence) for _, evidence in calls] == [1, 2]
    payload = json.loads(client.calls[1]['messages'][1]['content'])
    hint = payload['observations'][0]['recoverable_rows'][0]
    assert hint['recovery_ref'] == ref and hint['row_index'] == 2
    assert '23.4567' not in json.dumps(payload, ensure_ascii=False)
    assert run.evidence.get(ref).structure.table.rows[0].original_cells[-1] == '23.4567'


def test_unknown_reference_is_rejected_and_success_is_cached(monkeypatch):
    result, _ = make_result(monkeypatch)
    tool = RagSearchTool(1, retriever=lambda *a, **kw: result)
    executor = ToolExecutor([tool])
    context = prepare_query_context('员工培训要求')
    search = AgentTask('search', query='员工培训要求', goal_ids=())
    first = executor.execute_task(search, context, {}, {})
    completed = {'search': first}
    ref = result.recoverable_evidence[0].id
    recovery = AgentTask('restore', tool='restore_table_rows', query=ref)
    cache = {}
    restored = executor.execute_task(recovery, context, completed, cache)
    assert restored.status == 'succeeded'
    assert executor.execute_task(replace(recovery, id='again'), context, completed, cache).status == 'cached'
    assert executor.execute_task(recovery, context, {}, {}).status == 'failed'
    assert executor.execute_task(replace(recovery, query='/tmp/other-table.json'), context, completed, {}).status == 'failed'


def test_oversized_recovery_does_not_truncate_values(monkeypatch):
    result, _ = make_result(monkeypatch)
    from agent.tools import restore_table_rows
    monkeypatch.setattr(restore_table_rows, 'MAX_RECOVERY_TOKENS', 1)
    search_tool = RagSearchTool(1, retriever=lambda *a, **kw: result)
    executor = ToolExecutor([search_tool])
    context = prepare_query_context('员工培训要求')
    first = executor.execute_task(AgentTask('search', query='员工培训要求'), context, {}, {})
    task_value = AgentTask('restore', tool='restore_table_rows', query=result.recoverable_evidence[0].id)
    restored = executor.execute_task(task_value, context, {'search': first}, {})
    assert restored.status == 'failed' and not restored.result.evidence
    assert '预算' in restored.result.error


def test_recovery_consumes_shared_retrieval_budget(monkeypatch):
    result, _ = make_result(monkeypatch)
    from service.core.runtime.context import QueryRuntime, QueryBudgetExceeded, query_runtime
    executor = ToolExecutor([RagSearchTool(1, retriever=lambda *a, **kw: result)])
    context = prepare_query_context('员工培训要求')
    first = executor.execute_task(AgentTask('search', query='员工培训要求'), context, {}, {})
    runtime = QueryRuntime(max_retrievals=1)
    recovery = AgentTask('restore', tool='restore_table_rows', query=result.recoverable_evidence[0].id)
    with query_runtime(runtime):
        before = runtime.counts.get('retrieval', 0)
        executor.execute_task(recovery, context, {'search': first}, {})
        assert runtime.counts['retrieval'] == before + 1
        with pytest.raises(QueryBudgetExceeded):
            executor.execute_task(recovery, context, {'search': first}, {})
