"""Additional §7.7 mechanism cases, not live-model or human-reference evidence.

Expected old human records are named by the changed business source, without
reading the dependency graph. P is read from the actual appended invalidation
event. Conservative extra records are reported separately from omissions.
"""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from aml_qc.depgraph import business_result, digest
from aml_qc.ingest import load_case
from aml_qc.llm import FrozenModel
from aml_qc.store import Store
from aml_qc.workflow import run_review
from test_model_safety import MODEL, script, tool_message


DATA = Path(__file__).resolve().parents[1] / 'data/synthetic'


def reviewed_state(store):
    state = store.create(load_case(DATA / 'seed-02.json'))
    cid = state['package']['case_id']
    state = store.save_run(cid, state['source_hash'], run_review(state['package']))
    for annotation in list(state['annotations']):
        if annotation['kind'] == 'scope_review' or 'confirm' not in annotation['allowed_actions']:
            continue
        state = store.review(cid, action='confirm', target_id=annotation['annotation_id'],
            reason='机制测试裁决，不是独立参考答案', actor='mechanism-fixture',
            snapshot_id=annotation['snapshot_id'], expected_event_id=None)
    return state


@pytest.mark.parametrize('mutation', ['add_claim', 'delete_claim', 'new_alert_focus'])
def test_source_mutation_business_result_and_human_impact(tmp_path, record_property, mutation):
    store = Store(tmp_path / 'mutation.sqlite3')
    prior = reviewed_state(store)
    old = deepcopy(prior['latest_run'])
    updated = deepcopy(prior['package'])
    doc = next(d for d in updated['documents'] if d['document_id'] == 'narrative')
    if mutation == 'add_claim':
        doc['text'] += '另外陈述：检查期间仅向乙公司支付二次货款。'
        expected_labels = {'count', 'alert_response'}
    elif mutation == 'delete_claim':
        doc['text'] = '收款为日用品销售收入，零售订单明细尚未提交。'
        expected_labels = {'count', 'alert_response'}
    else:
        updated['alert']['focuses'].append({'focus_id': 'new-focus', 'text': '新增关注：请解释零售订单与收款的对应。'})
        expected_labels = {'alert_response'}
    expected = {e['event_id'] for e in prior['review_events'] if e.get('label') in expected_labels}
    assert expected
    changed = store.change_source(updated['case_id'], updated, '定向变更机制验收')
    invalidation = next(e for e in reversed(changed['review_events']) if e['action'] == 'needs_review')
    predicted = {e['event_id'] for e in invalidation['review_records']}
    assert not expected - predicted, '不得漏列独立预期的旧人工记录'
    assert not changed['can_pass'] and changed['stale']
    assert all(not a['review']['valid'] for a in changed['annotations'])

    incremental = run_review(changed['package'], strategy='incremental', previous=old['snapshot'])
    full = run_review(changed['package'], strategy='full')
    assert business_result(incremental) == business_result(full)
    assert incremental['snapshot']['nodes'] == full['snapshot']['nodes']
    assert full['stats']['reused'] == 0
    assert prior['latest_run'] == old
    old_ids = {i['issue_id'] for i in old['issues'] if i['type'] == 'claim_error'}
    new_ids = {i['issue_id'] for i in full['issues'] if i['type'] == 'claim_error'}
    if mutation == 'add_claim':
        assert len(full['claims']) == 2 and len(new_ids - old_ids) == 1
        assert {c['value'] for c in full['claims']} == {1, 2}
        assert all(c['result'] == 'contradicted' and c['observed']['count'] == 3 for c in full['claim_results'])
    elif mutation == 'delete_claim':
        assert full['claims'] == [] and full['claim_results'] == []
        assert old_ids and old_ids.isdisjoint(new_ids)
        assert incremental['stats']['revoked'] >= len(old_ids)
    else:
        assert {r['focus_id'] for r in full['semantic_results']} == {'focus-1', 'new-focus'}
        assert any(i['target_id'] == 'semantic:new-focus' for i in full['open_items'])
        assert all(r['status'] == 'pending_judgement' for r in full['semantic_results'])
    current = store.save_run(updated['case_id'], changed['source_hash'], incremental)
    assert store.export(updated['case_id'])['deliverable']['annotations'] == []
    if mutation == 'delete_claim':
        absent = next(a for a in current['annotations'] if a['label'] == 'count')
        assert absent['kind'] == 'scope_review' and absent['candidate_value'] is None
    record_property('mechanism', json.dumps({'case': mutation, 'provider': 'local',
        'expected_review_records': sorted(expected), 'actual_review_records': sorted(predicted),
        'missed_records': sorted(expected-predicted), 'conservative_extra_records': sorted(predicted-expected),
        'incremental_stats': incremental['stats'], 'independent_full_stats': full['stats']}, ensure_ascii=False))


@pytest.mark.parametrize('mutation', ['read_kyc_changed', 'material_catalog_added', 'tool_return_changed'])
def test_read_context_change_reruns_whole_agent_with_exact_request_replay(record_property, mutation):
    old = load_case(DATA / 'seed-01.json')
    new = deepcopy(old)
    if mutation == 'read_kyc_changed':
        next(d for d in new['documents'] if d['document_id'] == 'kyc')['text'] += ' 合成开发变更：补充经营周期。'
    elif mutation == 'material_catalog_added':
        material = deepcopy(new['materials'][0]); material['material_id'] = 'additional-contract'
        new['materials'].append(material)
    else:
        new['transactions'] = [r for r in new['transactions'] if r['direction'] != 'out']
    new['data_version'] = '2'
    calls = tool_message('read_document', {'document_id': 'kyc'})
    calls['tool_calls'] += tool_message('query_transactions', {'direction': 'out'}, 'call-2')['tool_calls']
    products, old_products = {}, {}
    for index, value in enumerate((old, new)):
        recorder = script(value, tools=[calls])
        run_review(value, provider='frozen', mode='agent', model=recorder)
        assert not recorder.responses
        for key, response in recorder.request_products.items():
            assert key not in products or products[key] == response
            products[key] = deepcopy(response)
        if index == 0:
            old_products = deepcopy(products)
    before = run_review(old, provider='frozen', mode='agent', model=FrozenModel(products, MODEL))
    frozen = deepcopy(before['snapshot'])
    inc = run_review(new, provider='frozen', mode='agent', strategy='incremental',
                     previous=before['snapshot'], model=FrozenModel(products, MODEL))
    full = run_review(new, provider='frozen', mode='agent', model=FrozenModel(products, MODEL))
    assert business_result(inc) == business_result(full)
    # The stage stores real tool latency in its trace. §7.7 excludes this
    # volatile field, while source bindings, parameters and results must agree.
    def without_latency(value):
        if isinstance(value, dict):
            return {k: without_latency(v) for k, v in value.items() if k != 'duration_ms'}
        if isinstance(value, list):
            return [without_latency(v) for v in value]
        return value
    assert inc['snapshot']['nodes'].keys() == full['snapshot']['nodes'].keys()
    for key, node in inc['snapshot']['nodes'].items():
        other = full['snapshot']['nodes'][key]
        assert all(node[field] == other[field] for field in ['kind', 'parameters', 'dependencies', 'fingerprint'])
        assert node['hash'] == digest(node['result']) and other['hash'] == digest(other['result'])
        assert without_latency(node['result']) == without_latency(other['result'])
    assert before['snapshot'] == frozen and full['stats']['reused'] == 0
    assert next(t for t in inc['trace'] if t['tool'] == 'agent_stage')['status'] == 'computed'
    assert inc['stats']['adaptive_tool_calls'] == 2 and inc['stats']['replayed_tool_calls'] == 0
    assert any(r['request_hash'] not in old_products for r in inc['model_requests'])
    # An old answer cannot be borrowed for the changed full context.
    miss = run_review(new, provider='frozen', mode='agent', model=FrozenModel(old_products, MODEL))
    assert any(c['check_id'] == 'semantic' and c['status'] == 'failed' for c in miss['required_checks'])
    assert any('冻结请求未命中' in i['description'] for i in miss['issues'])
    if mutation == 'tool_return_changed':
        query = next(t['result'] for t in inc['trace'] if t.get('round') and t['tool'] == 'query_transactions')
        assert query['metrics']['count'] == 0 and query['transaction_ids'] == []
    record_property('mechanism', json.dumps({'case': mutation, 'provider': 'frozen_fixture_not_live_agent',
        'new_request_count': sum(r['request_hash'] not in old_products for r in inc['model_requests']),
        'incremental_stats': inc['stats'], 'independent_full_stats': full['stats']}, ensure_ascii=False))
