"""Fixed read policy mechanics, not model quality or human reference labels."""
from copy import deepcopy
import json

import pytest

from aml_qc import llm, workflow
from aml_qc.depgraph import business_result, digest
from aml_qc.llm import FrozenModel
from test_model_safety import MODEL, ScriptedModel, case, json_message, response_message, script, semantic_response


@pytest.fixture(autouse=True)
def forbid_paid_calls(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Fixed policy mechanism tests must not read credentials or call a model service')
    monkeypatch.setattr(llm, 'settings', forbidden)
    monkeypatch.setattr(llm.httpx, 'post', forbidden)


@pytest.mark.parametrize('partial', [False, True])
def test_fixed_model_receives_actual_full_scope_rows_and_coverage(partial):
    value = case()
    if partial:
        value['coverage'][0]['status'] = 'partial'
    model = script(value)
    result = workflow.run_review(value, provider='frozen', model=model)
    data = json.loads(model.calls[2]['request']['messages'][1]['content'])
    reads = data['fixed_tool_results']
    assert [r['tool'] for r in reads] == ['check_coverage', 'query_transactions']
    assert reads[0]['result']['coverage'] == value['coverage']
    query = reads[1]
    assert query['arguments'] == {'start': value['coverage_start'], 'end': value['coverage_end']}
    assert set(query['result']['transaction_ids']) == {t['transaction_id'] for t in value['transactions']}
    assert query['result']['metrics']['count'] == len(value['transactions'])
    assert query['result']['coverage']['status'] == ('partial' if partial else 'full')
    for read in reads:
        node = result['snapshot']['nodes'][read['result_ref']]
        assert node['result'] == read['result']
        assert read['result_ref'] in result['snapshot']['nodes']['agent_stage']['dependencies']
        assert read['result_ref'] in {r['basis_ref'] for r in data['lead_basis_catalog']}
    assert all(r['request']['tools'] is None for r in model.calls)
    assert result['stats']['model_calls'] == 3 and result['stats']['adaptive_tool_calls'] == 0
    assert result['execution']['fixed_policy']['version'] == 'fixed-visible-scope-1'


def test_fixed_can_cite_the_successful_read_it_was_actually_shown():
    value = case()
    ref = 'tool:query_transactions:' + digest({'start': value['coverage_start'], 'end': value['coverage_end']})[:20]
    answer = semantic_response(value)
    answer['leads'] = [{'observation': '机制测试中的可见流水观察，业务意义未判定。',
        'question': '是否需核对额外经营关系？', 'basis_refs': [ref]}]
    model = ScriptedModel([json_message({'claims': [], 'unresolved': []}), response_message(value),
                           json_message({'gaps':answer['gaps'],'leads':answer['leads']})])
    result = workflow.run_review(value, provider='frozen', model=model)
    lead, = result['lead_candidates']
    assert lead['basis_refs'] == [ref] and lead['novelty_status'] == 'unconfirmed'
    assert any(e.get('result_ref') == ref for e in lead['evidence'])


def test_unchanged_fixed_reuses_reads_and_model_stage_without_fresh_calls():
    value = case(); recorder = script(value)
    old = workflow.run_review(value, provider='frozen', model=recorder)
    snapshot = deepcopy(old['snapshot'])
    incremental_model = FrozenModel(recorder.request_products, MODEL)
    inc = workflow.run_review(value, provider='frozen', strategy='incremental', previous=snapshot, model=incremental_model)
    full = workflow.run_review(value, provider='frozen', strategy='full', model=FrozenModel(recorder.request_products, MODEL))
    assert business_result(inc) == business_result(full)
    assert inc['snapshot']['nodes'] == full['snapshot']['nodes']
    assert old['snapshot'] == snapshot
    assert inc['stats']['model_calls'] == inc['stats']['tool_calls'] == inc['stats']['recomputed'] == 0
    assert full['stats']['reused'] == 0 and full['stats']['model_calls'] == 3
    assert [r['cache_status'] for r in inc['fixed_policy_reads']] == ['reused', 'reused']


def test_empty_query_then_first_row_cannot_reuse_the_old_semantic_request():
    new = case(); new['transactions'] = new['transactions'][:1]
    old_case = deepcopy(new); old_case['transactions'] = []
    recorder = script(old_case)
    old = workflow.run_review(old_case, provider='frozen', model=recorder)
    fresh = script(new)
    full = workflow.run_review(new, provider='frozen', model=fresh)
    assert json.loads(recorder.calls[2]['request']['messages'][1]['content'])['fixed_tool_results'][1]['result']['metrics']['count'] == 0
    assert json.loads(fresh.calls[2]['request']['messages'][1]['content'])['fixed_tool_results'][1]['result']['metrics']['count'] == 1
    assert recorder.calls[2]['request_hash'] != fresh.calls[2]['request_hash']
    products = {**recorder.request_products, **fresh.request_products}
    inc = workflow.run_review(new, provider='frozen', strategy='incremental', previous=old['snapshot'], model=FrozenModel(products, MODEL))
    assert business_result(inc) == business_result(full)
    assert inc['fixed_policy_reads'][1]['cache_status'] == 'computed'
    assert inc['stats']['model_calls'] == 1


def test_fixed_read_failure_is_recorded_and_blocks_semantic_completion(monkeypatch):
    original = workflow.execute_tool
    def fail_query(case, schema, evaluator, name, args):
        if name == 'query_transactions':
            raise ValueError('fixture: query failed')
        return original(case, schema, evaluator, name, args)
    monkeypatch.setattr(workflow, 'execute_tool', fail_query)
    value = case(); model = script(value)
    result = workflow.run_review(value, provider='frozen', model=model)
    assert result['run_status'] == 'partial' and result['semantic_results']
    assert result['fixed_policy_reads'][-1]['status'] == 'failed'
    assert result['fixed_policy_reads'][-1]['result_ref'] is None
    assert result['stats']['model_calls'] == 2 and result['stats']['adaptive_tool_calls'] == 0
    assert any(c['check_id'] == 'semantic' and c['status'] == 'failed' for c in result['required_checks'])
    assert any(i['kind'] == 'execution_failed' for i in result['open_items'])


def test_agent_still_selects_reads_itself_and_is_not_given_the_fixed_policy():
    value = case(); model = script(value, tools=[])
    result = workflow.run_review(value, provider='frozen', mode='agent', model=model)
    assert 'fixed_tool_results' not in json.loads(model.calls[2]['request']['messages'][1]['content'])
    assert result.get('fixed_policy_reads', []) == []
    assert result['stats']['tool_calls'] == 2  # features and materials; no additional fixed reads
