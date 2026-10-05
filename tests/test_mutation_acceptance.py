"""Additional §7.7 mechanism cases, not live-model or human-reference evidence.

Expected old human records are named by the changed business source, without
reading the dependency graph. P is read from the actual appended invalidation
event. Conservative extra records are reported separately from omissions.
"""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from test_store import bound_review

from aml_qc import llm
from aml_qc.depgraph import business_result, digest
from aml_qc.ingest import load_case
from aml_qc.llm import FrozenModel
from aml_qc.store import Store
from aml_qc.workflow import run_review
from test_model_safety import MODEL, ScriptedModel, json_message, script, semantic_response, tool_message


DATA = Path(__file__).resolve().parents[1] / 'data/synthetic'


@pytest.fixture(autouse=True)
def no_paid_api_or_credentials(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Mutation mechanism acceptance must never read credentials or call a live model')
    monkeypatch.setattr(llm, 'settings', forbidden)
    monkeypatch.setattr(llm.httpx, 'post', forbidden)


def confirm_produced(store, state):
    cid = state['package']['case_id']
    for annotation in list(state['annotations']):
        if annotation['kind'] == 'scope_review' or 'confirm' not in annotation['allowed_actions']:
            continue
        state = store.review(cid, action='confirm', target_id=annotation['annotation_id'],
            reason='机制测试裁决，不是独立参考答案', actor='mechanism-fixture',
            snapshot_id=annotation['snapshot_id'], expected_event_id=None)
    return state


def reviewed_state(store, package=None):
    state = store.create(package or load_case(DATA / 'seed-02.json'))
    cid = state['package']['case_id']
    state = store.save_run(cid, state['source_hash'], run_review(state['package']))
    return confirm_produced(store, state)


def impact_sets(prior, changed, labels):
    # Labels are selected by the hand-written business-change oracle below,
    # never by affected_nodes or the actual invalidation event.
    expected = {e['event_id'] for e in prior['review_events']
        if e.get('snapshot_id') == prior['latest_run']['snapshot_id']
        and (e.get('label') in labels or e.get('target_id') == 'task')}
    assert expected, 'The oracle must name existing human decisions, not an empty set'
    event = next(e for e in reversed(changed['review_events']) if e['action'] == 'needs_review')
    actual = {r['event_id'] for r in event['review_records']}
    assert expected <= actual, 'Independent expected human decisions were omitted'
    return {'expected_review_records': sorted(expected), 'actual_review_records': sorted(actual),
        'missed_records': sorted(expected - actual), 'conservative_extra_records': sorted(actual - expected)}


def independent_pair(prior_run, package, *, model_products=None, mode='fixed'):
    frozen = deepcopy(prior_run)
    options = {'provider': 'frozen', 'mode': mode} if model_products is not None else {'provider': 'local'}
    inc = run_review(deepcopy(package), strategy='incremental', previous=prior_run['snapshot'],
        **options, **({'model': FrozenModel(model_products, MODEL)} if model_products is not None else {}))
    full = run_review(deepcopy(package), strategy='full',
        **options, **({'model': FrozenModel(model_products, MODEL)} if model_products is not None else {}))
    assert business_result(inc) == business_result(full)
    assert inc['snapshot']['candidate_ids'] == full['snapshot']['candidate_ids']
    assert full['stats']['reused'] == 0 and full['stats']['recomputed'] == len(full['snapshot']['nodes'])
    assert prior_run == frozen
    for claim in full['claims']:
        source = claim['source']
        doc = next(d for d in package['documents'] if d['document_id'] == source['document_id'])
        assert source['revision'] == doc['revision']
        start, end = source['span']
        assert doc['text'][start:end] == claim['text']
    return inc, full


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
        added = next(r for r in full['semantic_results'] if r['focus_id'] == 'new-focus')
        assert any(e.get('focus_id') == 'new-focus' and e.get('revision') == changed['package']['alert']['revision']
                   for e in added['evidence'])
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
def test_read_context_change_reruns_whole_agent_with_exact_request_replay(tmp_path, record_property, mutation):
    store = Store(tmp_path / 'agent-context.sqlite3')
    initial = store.create(load_case(DATA / 'seed-01.json'))
    old = initial['package']
    calls = tool_message('read_document', {'document_id': 'kyc'})
    calls['tool_calls'] += tool_message('query_transactions', {'direction': 'out'}, 'call-2')['tool_calls']
    prior_recorder = script(old, tools=[calls])
    prior = store.save_run(old['case_id'], initial['source_hash'],
        run_review(old, provider='frozen', mode='agent', model=prior_recorder))
    assert not prior_recorder.responses
    prior = confirm_produced(store, prior)
    new = deepcopy(old)
    if mutation == 'read_kyc_changed':
        next(d for d in new['documents'] if d['document_id'] == 'kyc')['text'] += ' 合成开发变更：补充经营周期。'
    elif mutation == 'material_catalog_added':
        material = deepcopy(new['materials'][0]); material['material_id'] = 'additional-contract'
        new['materials'].append(material)
    else:
        new['transactions'] = [r for r in new['transactions'] if r['direction'] != 'out']
    new['data_version'] = '2'
    changed = store.change_source(old['case_id'], new, 'Agent已读上下文的独立机制变更')
    new = changed['package']
    labels = {'F1', 'F2', 'material_relation', 'alert_response'} if mutation == 'tool_return_changed' else {'alert_response'}
    impact = impact_sets(prior, changed, labels)
    old_products = deepcopy(prior_recorder.request_products)
    products = deepcopy(old_products)
    recorder = script(new, tools=[calls])
    run_review(new, provider='frozen', mode='agent', model=recorder)
    assert not recorder.responses
    for key, response in recorder.request_products.items():
        assert key not in products or products[key] == response
        products[key] = deepcopy(response)
    before = prior['latest_run']
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
    if mutation == 'read_kyc_changed':
        read = next(t['result'] for t in inc['trace'] if t.get('round') and t['tool'] == 'read_document')
        assert '补充经营周期' in json.dumps(read, ensure_ascii=False)
    if mutation == 'material_catalog_added':
        assert 'additional-contract' in json.dumps(inc['snapshot']['sources'], ensure_ascii=False)
    store.save_run(old['case_id'], changed['source_hash'], inc)
    assert not store.export(old['case_id'])['deliverable']['passed']
    record_property('mechanism', json.dumps({'case': mutation, 'provider': 'frozen_fixture_not_live_agent',
        **impact,
        'reviewed_snapshot_id': prior['latest_run']['snapshot_id'],
        'incremental_previous_snapshot_id': before['snapshot_id'],
        'new_request_count': sum(r['request_hash'] not in old_products for r in inc['model_requests']),
        'incremental_stats': inc['stats'], 'independent_full_stats': full['stats']}, ensure_ascii=False))


@pytest.mark.parametrize('mutation,seed,labels', [
    ('empty_query_first_transaction', 1, {'F1', 'F2', 'count', 'material_relation', 'alert_response'}),
    ('delete_unique_evidence', 1, {'F1', 'F2', 'count', 'material_relation', 'alert_response'}),
    ('prefix_moves_span', 1, {'count', 'alert_response'}),
    ('supply_missing_material', 3, {'material_relation', 'alert_response'}),
    ('only_material_link', 5, {'material_relation', 'alert_response'}),
    ('same_revision_different_entity', 1, {'count', 'material_relation', 'alert_response'}),
    ('recompute_revokes_candidate', 2, {'F1', 'F2', 'count', 'material_relation', 'alert_response'}),
])
def test_source_change_has_independent_value_evidence_and_human_impact(tmp_path, record_property, mutation, seed, labels):
    package = load_case(DATA / f'seed-{seed:02d}.json')
    if mutation == 'empty_query_first_transaction':
        package['transactions'] = [r for r in package['transactions'] if r['direction'] != 'out']
    if mutation == 'same_revision_different_entity':
        package['counterparties'].append({'counterparty_token': 'supplier-other', 'display_name_masked': '另一对象', 'type': 'business'})
    store = Store(tmp_path / 'source-acceptance.sqlite3')
    prior = reviewed_state(store, package)
    old = prior['latest_run']
    updated = deepcopy(prior['package'])
    if mutation == 'empty_query_first_transaction':
        # The first matching row has a declared identity; expected count is a
        # literal, not obtained by asking the production query implementation.
        updated['transactions'].append(next(r for r in load_case(DATA / 'seed-01.json')['transactions']
                                           if r['transaction_id'] == 'seed-01-out-00'))
    elif mutation == 'delete_unique_evidence':
        updated['transactions'] = [r for r in updated['transactions'] if r['transaction_id'] != 'seed-01-out-00']
    elif mutation == 'prefix_moves_span':
        next(d for d in updated['documents'] if d['document_id'] == 'narrative')['text'] = '新增前缀：' + next(
            d for d in updated['documents'] if d['document_id'] == 'narrative')['text']
    elif mutation == 'supply_missing_material':
        updated['materials'] = load_case(DATA / 'seed-01.json')['materials']
    elif mutation == 'only_material_link':
        updated['material_links'][0]['relation_template'] = 'contract_installments'
    elif mutation == 'same_revision_different_entity':
        updated['entity_mappings'][0]['target_token'] = 'supplier-other'
        assert updated['entity_mappings'][0]['revision'] == prior['package']['entity_mappings'][0]['revision']
    else:
        updated['transactions'] = [r for r in updated['transactions'] if r['transaction_id'] not in {'seed-02-out-01', 'seed-02-out-02'}]
    changed = store.change_source(updated['case_id'], updated, '独立预期驱动的来源变更验收')
    impact = impact_sets(prior, changed, labels)
    inc, full = independent_pair(old, changed['package'])
    assert inc['snapshot']['nodes'] == full['snapshot']['nodes']
    if mutation in {'empty_query_first_transaction', 'delete_unique_evidence', 'same_revision_different_entity', 'recompute_revokes_candidate'}:
        expected_count = 1 if mutation in {'empty_query_first_transaction', 'recompute_revokes_candidate'} else 0
        expected_ids = ['seed-02-out-00' if seed == 2 else 'seed-01-out-00'] if expected_count else []
        assert len(full['claim_results']) == 1
        result = full['claim_results'][0]
        assert result['observed']['count'] == expected_count
        assert result['result'] == ('supported' if expected_count == 1 else 'contradicted')
        query = next(e for e in result['evidence'] if e['type'] == 'query_scope')
        assert query['transaction_ids'] == expected_ids
        old_query = next(e for e in old['claim_results'][0]['evidence'] if e['type'] == 'query_scope')
        expected_old_count = 0 if mutation == 'empty_query_first_transaction' else 3 if mutation == 'recompute_revokes_candidate' else 1
        assert old['claim_results'][0]['observed']['count'] == expected_old_count
        if mutation == 'empty_query_first_transaction':
            assert old_query['transaction_ids'] == [] and old['claim_results'][0]['result'] == 'contradicted'
        if mutation == 'delete_unique_evidence':
            assert old_query['transaction_ids'] == ['seed-01-out-00']
            assert full['material_results'][0]['result'] == 'insufficient'
        if mutation != 'same_revision_different_entity':
            assert query['scope']['transaction_set_version'] != old_query['scope']['transaction_set_version']
        else:
            assert result['identity']['basis'] != old['claim_results'][0]['identity']['basis']
        if expected_count == 1:
            old_errors = {i['issue_id'] for i in old['issues'] if i['type'] == 'claim_error'}
            assert old_errors and old_errors.isdisjoint(full['snapshot']['candidate_ids'])
            assert inc['stats']['revoked'] >= len(old_errors)
        else:
            assert any(i['type'] == 'claim_error' for i in full['issues'])
    elif mutation == 'prefix_moves_span':
        assert full['claims'][0]['text'] == old['claims'][0]['text']
        assert full['claims'][0]['source']['span'] == [n + len('新增前缀：') for n in old['claims'][0]['source']['span']]
        assert full['claims'][0]['source']['revision'] != old['claims'][0]['source']['revision']
        # Logical proposition identity may remain stable when only its source
        # offsets move; the binding and dependent fingerprint must change.
        node_id = 'claim:' + old['claims'][0]['claim_id']
        assert old['claims'][0]['source'] != full['claims'][0]['source']
        assert old['snapshot']['nodes'][node_id]['fingerprint'] != full['snapshot']['nodes'][node_id]['fingerprint']
    else:
        assert full['material_results'][0]['result'] == 'corresponds'
        assert old['material_results'][0]['result'] == ('insufficient' if mutation == 'supply_missing_material' else 'pending_judgement')
        obsolete = {i['issue_id'] for i in old['issues'] if i['target_id'] == 'materials:purchase-link'}
        assert obsolete and obsolete.isdisjoint(full['snapshot']['candidate_ids'])
        assert inc['stats']['revoked'] >= len(obsolete)
    store.save_run(updated['case_id'], changed['source_hash'], inc)
    export = store.export(updated['case_id'])
    assert not export['deliverable']['passed'] and export['deliverable']['annotations'] == []
    assert prior['latest_run'] == old
    record_property('mechanism', json.dumps({'case': mutation, 'provider': 'local', **impact,
        'independent_assertions': 'literal count/IDs, source span/version or material label; current candidate removal after both runs',
        'incremental_stats': inc['stats'], 'independent_full_stats': full['stats']}, ensure_ascii=False))


def test_cross_case_migration_has_independent_human_impact_and_unselected_case_is_unchanged(tmp_path, record_property):
    store = Store(tmp_path / 'migration-acceptance.sqlite3')
    states = {}
    for number in (1, 2, 3):
        package = load_case(DATA / f'seed-{number:02d}.json')
        package.update(task_mode='annotation_only', alert=None)
        package['review_scope']['target_labels'] = ['F1', 'F2']
        states[package['case_id']] = reviewed_state(store, package)
    schema = deepcopy(states['seed-01']['package']['schema'])
    schema['schema_version'] = 'S1.1-independent-mutation'
    schema['features']['F1']['minimum_days'] = 1
    schema['features']['F2']['ratio_numerator'] = 9
    preview = store.preview_migration(base_schema_hash=digest(states['seed-01']['package']['schema']),
        target_schema=schema, selected_case_ids=['seed-01', 'seed-02'], actor='fixture-rule-reviewer', reason='明确两案范围及演示阈值')
    rows = []
    for cid in ('seed-01', 'seed-02'):
        prior = states[cid]
        changed = store.migrate_case(preview['preview_id'], cid, preview_hash=preview['preview_hash'],
            actor='fixture-operator', reason='按冻结预览逐案采用')['case']
        impact = impact_sets(prior, changed, {'F1', 'F2'})
        inc, full = independent_pair(prior['latest_run'], changed['package'])
        assert inc['snapshot']['nodes'] == full['snapshot']['nodes']
        assert {f['feature_code']: f['result'] for f in full['features']} == {'F1': 'met', 'F2': 'not_met'}
        assert all(f['schema_version'] == schema['schema_version'] for f in full['features'])
        store.save_run(cid, changed['source_hash'], inc)
        assert not store.export(cid)['deliverable']['passed']
        rows.append({'case_id': cid, **impact, 'incremental_stats': inc['stats'], 'independent_full_stats': full['stats']})
    other = store.get('seed-03')
    assert all(other[k] == states['seed-03'][k] for k in ('package', 'source_hash', 'latest_run', 'review_events'))
    assert other['package']['schema_version'] == 'S1.0'
    record_property('mechanism', json.dumps({'case': 'cross_case_schema_migration', 'provider': 'local',
        'cases': rows, 'unselected_case_unchanged': 'seed-03'}, ensure_ascii=False))


def lead_fixture(package):
    response = semantic_response(package)
    response['leads'] = [{'question': '经营资料的结算安排是否需要补充回应？',
        'observation': '经营资料存在结算安排，供本机制测试人工决定范围。', 'basis_refs': ['document:kyc']}]
    upgraded = {f['focus_id'] for f in package['review_scope'].get('upgraded_leads', [])}
    for focus in response['focuses']:
        if focus['focus_id'] in upgraded:
            focus.update(status='not_addressed', quote='', reason='机制夹具未回应人工升级范围')
    return ScriptedModel([json_message({'claims': [], 'unresolved': []}),
        json_message({'focuses': response['focuses']}),
        json_message({'gaps': response['gaps'], 'leads': response['leads']})])


@pytest.mark.parametrize('action', ['upgrade_lead', 'close_lead'])
def test_lead_scope_only_change_has_independent_human_impact_and_current_candidate_set(tmp_path, record_property, action):
    store = Store(tmp_path / 'lead-acceptance.sqlite3')
    package = load_case(DATA / 'seed-01.json')
    package['review_scope']['target_labels'] = ['F1', 'F2', 'alert_response']
    prior = store.create(package)
    def execute(state):
        return store.save_run(package['case_id'], state['source_hash'],
            run_review(state['package'], provider='frozen', model=lead_fixture(state['package'])))
    def act(state, choice):
        lead = state['lead_dispositions'][0]
        return store.review(package['case_id'], action=choice, target_id=lead['lead_id'], actor='fixture-human',
            reason='机制测试核对观察后改变必需范围', snapshot_id=state['latest_run']['snapshot_id'],
            expected_event_id=lead['expected_event_id'])
    prior = execute(prior)
    assert any(i['type'] == 'new_lead' for i in prior['latest_run']['issues'])
    if action == 'close_lead':
        prior = execute(act(prior, 'upgrade_lead'))
    prior = confirm_produced(store, prior)
    if prior['can_pass']:
        prior = bound_review(store, package['case_id'], action='confirm', target_id='task', reason='机制范围已逐标签确认')
    changed = act(prior, action)
    impact = impact_sets(prior, changed, {'alert_response'})
    for key in ('documents', 'transactions', 'materials', 'material_links', 'alert'):
        assert changed['package'][key] == prior['package'][key]
    recorder = lead_fixture(changed['package'])
    run_review(changed['package'], provider='frozen', model=recorder)
    assert not recorder.responses
    inc, full = independent_pair(prior['latest_run'], changed['package'], model_products=recorder.request_products)
    focus = changed['package']['review_scope']['lead_dispositions'][-1]['focus_id']
    targets = {r['focus_id'] for r in full['semantic_results']}
    if action == 'upgrade_lead':
        assert focus in targets and any(r['required'] and r['type'] == 'upgraded_lead' for r in full['semantic_results'])
        assert next(r for r in full['semantic_results'] if r['focus_id'] == focus)['status'] == 'not_addressed'
        assert any(i['target_id'] == 'semantic:' + focus for i in full['open_items'])
    else:
        assert focus not in targets and not changed['package']['review_scope']['upgraded_leads']
        assert all(i['target_id'] != 'semantic:' + focus for i in full['issues'] + full['open_items'])
        removed = {i['issue_id'] for i in prior['latest_run']['issues'] if i['target_id'] == 'semantic:' + focus}
        assert removed and removed.isdisjoint(full['snapshot']['candidate_ids'])
        assert inc['stats']['revoked'] >= len(removed)
        assert 'agent_stage' in full['snapshot']['nodes']
    store.save_run(package['case_id'], changed['source_hash'], inc)
    exported = store.export(package['case_id'])
    assert not exported['deliverable']['passed'] and exported['lead_dispositions']
    record_property('mechanism', json.dumps({'case': action, 'provider': 'frozen_fixture_not_live_agent', **impact,
        'incremental_stats': inc['stats'], 'independent_full_stats': full['stats']}, ensure_ascii=False))


def test_actual_failed_recheck_preserves_human_impact_and_never_exports_old_pass(tmp_path, record_property):
    store = Store(tmp_path / 'failed-acceptance.sqlite3')
    package = load_case(DATA / 'seed-01.json')
    package['review_scope']['target_labels'] = ['F1', 'F2', 'alert_response']
    initial = store.create(package)
    recorder = script(initial['package'])
    old_run = run_review(initial['package'], provider='frozen', model=recorder)
    prior = confirm_produced(store, store.save_run(package['case_id'], initial['source_hash'], old_run))
    assert prior['can_pass']
    prior = bound_review(store, package['case_id'], action='confirm', target_id='task', reason='机制测试旧快照已人工通过')
    assert store.export(package['case_id'])['deliverable']['passed']
    updated = deepcopy(prior['package'])
    next(d for d in updated['documents'] if d['document_id'] == 'kyc')['text'] += ' 新经营说明导致旧语义请求失效。'
    changed = store.change_source(package['case_id'], updated, '失败重查前真实修改资料')
    impact = impact_sets(prior, changed, {'alert_response'})
    # Deliberately omit products for the new complete semantic request. Both
    # paths must fail through the actual workflow, never reuse an old answer.
    inc, full = independent_pair(prior['latest_run'], changed['package'], model_products=recorder.request_products)
    assert full['run_status'] == 'partial'
    assert any(c['check_id'] == 'semantic' and c['status'] == 'failed' for c in full['required_checks'])
    assert any(i['type'] == 'execution_failed' and '冻结请求未命中' in i['description'] for i in full['issues'])
    current = store.save_run(package['case_id'], changed['source_hash'], inc)
    assert not current['can_pass'] and current['latest_run']['run_id'] != prior['latest_run']['run_id']
    exported = store.export(package['case_id'])
    assert not exported['deliverable']['passed'] and exported['deliverable']['annotations'] == []
    failure = next(i for i in current['latest_run']['issues'] if i['type'] == 'execution_failed')
    with pytest.raises(ValueError):
        bound_review(store, package['case_id'], action='reject', target_id=failure['issue_id'], reason='不能驳回执行失败绕过门禁')
    with pytest.raises(ValueError):
        bound_review(store, package['case_id'], action='confirm', target_id='task', reason='不能沿用旧通过')
    record_property('mechanism', json.dumps({'case': 'failed_recheck_blocks_old_export',
        'provider': 'frozen_exact_request_miss', **impact,
        'incremental_stats': inc['stats'], 'independent_full_stats': full['stats']}, ensure_ascii=False))
