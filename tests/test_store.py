"""Store invariants: immutable history, explicit decisions, current-only export."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from aml_qc.ingest import load_case
from aml_qc.store import Store, audit_integrity
from aml_qc.workflow import run_review

DATA = Path(__file__).resolve().parents[1] / 'data' / 'synthetic'


def review_binding(state, target_id):
    actions = {'confirm', 'reject', 'request_correction', 'dispute', 'reconfirm', 'close_item'}
    event = next((e for e in reversed(state['review_events'])
                  if e.get('source_hash') == state['source_hash'] and e.get('target_id') == target_id
                  and e.get('action') in actions), {})
    return {'snapshot_id': (state['latest_run'] or {}).get('snapshot_id'),
            'expected_event_id': event.get('event_id')}


def bound_review(store, case_id, **kwargs):
    for key, value in review_binding(store.get(case_id), kwargs['target_id']).items():
        kwargs.setdefault(key, value)
    return store.review(case_id, **kwargs)


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / 'audit.sqlite3')


def create(store, number=1):
    package = load_case(DATA / f'seed-{number:02d}.json')
    # These state-machine fixtures target the facts the local extractor supports.
    # Full eight-label scope and absent-candidate decisions have separate tests.
    package['review_scope']['target_labels'] = ['F1', 'F2', 'count', 'material_relation', 'alert_response']
    return store.create(package)


def run_current(store, case_id):
    state = store.get(case_id)
    result = run_review(state['package'], mode='fixed', provider='local', strategy='full')
    return store.save_run(case_id, state['source_hash'], result)


def close_manual(store, case_id):
    state = store.get(case_id)
    types = {i['issue_id']: i['type'] for i in state['latest_run']['issues']}
    for item in list(state['open_items']):
        kind = types.get(item['item_id'])
        if kind in {'manual_extraction', 'manual_focus', 'manual_material'}:
            bound_review(store, case_id, action='close_item', target_id=item['item_id'], reason='测试人工复核：已逐项检查对应范围和证据',
                         actor='synthetic-reviewer', resolution='corresponds' if kind == 'manual_material' else 'addressed')
    for item in store.get(case_id)['annotations']:
        action = 'revise' if item['kind'] == 'semantic' else 'reconfirm' if 'reconfirm' in item['allowed_actions'] else 'confirm'
        if action not in item['allowed_actions']:
            continue
        store.review(case_id, action=action, target_id=item['annotation_id'], reason='测试人工逐标签核对本次范围与证据',
                     snapshot_id=item['snapshot_id'], expected_event_id=item['review']['event_id'],
                     previous_event_id=(item.get('prior_review') or {}).get('event_id'),
                     new_value='addressed' if item['kind'] == 'semantic' else None, evidence=item['evidence'])
    return store.get(case_id)


def make_passed(store, number=1):
    state = create(store, number); cid = state['package']['case_id']
    run_current(store, cid)
    state = close_manual(store, cid)
    assert state['can_pass']
    return bound_review(store, cid, action='confirm', target_id='task', reason='复核当前快照，确认限定范围内完成')


def test_default_schema_is_pinned_and_mismatched_version_rejected(store):
    state = create(store)
    assert state['package']['schema']['schema_version'] == state['package']['schema_version'] == 'S1.0'
    assert state['package']['schema']['material_templates']['contract_installments']['amount_relation'] == 'sum_at_most'
    invalid = load_case(DATA / 'seed-02.json'); invalid['schema_version'] = 'UNSUPPORTED'
    with pytest.raises(ValueError, match='Schema'):
        store.create(invalid)


def test_non_synthetic_and_conflicting_duplicate_import_rejected(store):
    package = load_case(DATA / 'seed-01.json'); package['profile']['data_origin'] = 'live'
    with pytest.raises(ValueError, match='合成'):
        store.create(package)
    package = load_case(DATA / 'seed-01.json')
    row = deepcopy(package['transactions'][0]); row['amount'] = '9.00'; package['transactions'].append(row)
    with pytest.raises(ValueError, match='重复'):
        store.create(package)


def test_task_pass_requires_explicit_check_decisions_and_final_confirmation(store):
    state = create(store); cid = state['package']['case_id']
    run_current(store, cid)
    with pytest.raises(ValueError, match='不能通过'):
        bound_review(store, cid, action='confirm', target_id='task', reason='不能跳过未完成检查')
    state = close_manual(store, cid)
    assert state['can_pass']
    assert state['review_status'] != '本次质检范围内通过'
    assert store.export(cid)['deliverable']['passed'] is False
    state = bound_review(store, cid, action='confirm', target_id='task', reason='当前快照全部检查已人工完成')
    assert state['review_status'] == '本次质检范围内通过'
    assert store.export(cid)['deliverable']['passed'] is True


def test_confirming_problem_does_not_resolve_it_or_allow_pass(store):
    state = create(store, 2); cid = state['package']['case_id']
    state = run_current(store, cid)
    problem = next(i for i in state['latest_run']['issues'] if i['type'] == 'claim_error')
    bound_review(store, cid, action='confirm', target_id=problem['issue_id'], reason='完整流水确认有三笔，次数陈述确实错误')
    state = close_manual(store, cid)
    assert any(i['item_id'] == problem['issue_id'] for i in state['open_items'])
    assert not state['can_pass']
    with pytest.raises(ValueError, match='不能通过'):
        bound_review(store, cid, action='confirm', target_id='task', reason='仅确认问题不代表补正完成')
    exported = store.export(cid)
    assert exported['deliverable']['confirmed_issues'] == []
    assert [i['issue_id'] for i in exported['candidates_and_open_items']['current_confirmed_issues']] == [problem['issue_id']]
    assert exported['deliverable']['passed'] is False


def test_source_change_invalidates_pass_and_requires_new_snapshot_confirmation(store):
    passed = make_passed(store); cid = passed['package']['case_id']
    old_run = deepcopy(passed['latest_run']); old_events = deepcopy(passed['review_events'])
    package = deepcopy(passed['package']); package['documents'][0]['text'] = '补充：' + package['documents'][0]['text']
    stale = store.change_source(cid, package, '经办补充理由文本')
    assert stale['stale'] and stale['review_status'] == 'needs_review'
    assert stale['package']['documents'][0]['revision'] != passed['package']['documents'][0]['revision']
    assert stale['latest_run']['run_id'] == old_run['run_id']
    assert not stale['can_pass']
    with pytest.raises(ValueError, match='旧快照'):
        bound_review(store, cid, action='reconfirm', target_id='task', reason='不能直接沿用旧结论')
    exported = store.export(cid)
    assert exported['deliverable']['snapshot_id'] is None
    assert exported['deliverable']['confirmed_issues'] == []
    assert not exported['deliverable']['passed']
    new = run_current(store, cid)
    assert not new['stale'] and new['latest_run']['snapshot_id'] != old_run['snapshot_id']
    assert new['review_status'] != '本次质检范围内通过'
    closed = close_manual(store, cid)
    assert closed['can_pass'] and closed['review_status'] != '本次质检范围内通过'
    confirmed = bound_review(store, cid, action='reconfirm', target_id='task', reason='已针对新快照重新复核')
    assert confirmed['review_status'] == '本次质检范围内通过'
    assert confirmed['review_events'][:len(old_events)] == old_events
    with store.connect() as db:
        historical = json.loads(db.execute('SELECT result FROM runs WHERE run_id=?', (old_run['run_id'],)).fetchone()[0])
        assert historical['snapshot_id'] == old_run['snapshot_id']
        assert db.execute('SELECT COUNT(*) FROM sources WHERE case_id=?', (cid,)).fetchone()[0] == 2
        assert db.execute('SELECT COUNT(*) FROM runs WHERE case_id=?', (cid,)).fetchone()[0] == 2


def test_failed_rerun_cannot_export_old_pass_or_reject_failure_away(store):
    passed = make_passed(store); cid = passed['package']['case_id']
    failed = {'case_id':cid, 'run_status':'failed', 'required_checks':[{'check_id':'execution','label':'执行','status':'failed'}],
              'issues':[{'issue_id':'failure-1','type':'execution_failed','target_id':'execution','evidence':[]}],
              'open_items':[{'item_id':'failure-1','target_id':'execution','kind':'execution_failed','title':'执行失败'}]}
    state = store.save_run(cid, passed['source_hash'], failed)
    assert not state['can_pass']
    assert state['latest_run']['run_id'] != passed['latest_run']['run_id']
    assert not store.export(cid)['deliverable']['passed']
    assert store.export(cid)['deliverable']['confirmed_issues'] == []
    with pytest.raises(ValueError, match='不能通过否决'):
        bound_review(store, cid, action='reject', target_id='failure-1', reason='不能以误报处理执行失败')
    with pytest.raises(ValueError, match='不可手动关闭'):
        bound_review(store, cid, action='close_item', target_id='failure-1', reason='失败必须重新执行', resolution='not_applicable')
    with pytest.raises(ValueError, match='不能通过'):
        bound_review(store, cid, action='confirm', target_id='task', reason='不能把失败伪装为通过')


def test_partial_judgement_can_be_closed_but_negative_verdict_stays_open(store):
    state = create(store); cid = state['package']['case_id']; state = run_current(store, cid)
    focus = next(i for i in state['latest_run']['issues'] if i['type'] == 'manual_focus')
    state = bound_review(store, cid, action='close_item', target_id=focus['issue_id'], reason='全文未回应指定关注点', resolution='not_addressed')
    assert any(i['item_id'] == focus['issue_id'] for i in state['open_items'])
    assert any(c['check_id'] == 'semantic' for c in state['pending_checks'])
    assert not state['can_pass']
    state = bound_review(store, cid, action='close_item', target_id=focus['issue_id'], reason='再次裁决：相关句子已回应关注点但不证明合理性', resolution='addressed')
    assert all(i['item_id'] != focus['issue_id'] for i in state['open_items'])


@pytest.mark.parametrize('action', ['dispute', 'request_correction'])
def test_task_dispute_or_correction_survives_same_source_rerun_until_explicit_resolution(store, action):
    passed = make_passed(store); cid = passed['package']['case_id']
    state = bound_review(store, cid, action=action, target_id='task', reason='当前任务需进一步复核')
    assert not state['can_pass']
    assert any(i['item_id'] == 'task-review' for i in state['open_items'])
    assert not store.export(cid)['deliverable']['passed']
    state = run_current(store, cid)
    assert any(i['item_id'] == 'task-review' for i in state['open_items'])
    close_manual(store, cid)
    with pytest.raises(ValueError, match='不能通过'):
        bound_review(store, cid, action='confirm', target_id='task', reason='重跑本身不能解除任务争议')
    state = bound_review(store, cid, action='close_item', target_id='task', resolution='not_applicable', reason='复核裁决：撤回该项请求，原请求不适用于本次范围')
    assert state['can_pass']
    assert state['review_status'] != '本次质检范围内通过'
    state = bound_review(store, cid, action='reconfirm', target_id='task', reason='显式裁决后重新确认当前快照')
    assert state['review_status'] == '本次质检范围内通过'


def test_disputed_issue_not_exported_as_current_confirmed_annotation(store):
    state = create(store, 2); cid = state['package']['case_id']; state = run_current(store, cid)
    issue_id = next(i['issue_id'] for i in state['latest_run']['issues'] if i['type'] == 'claim_error')
    bound_review(store, cid, action='confirm', target_id=issue_id, reason='第一次复核确认问题')
    assert store.export(cid)['candidates_and_open_items']['current_confirmed_issues']
    state = bound_review(store, cid, action='dispute', target_id=issue_id, reason='对该问题标注提出争议')
    assert state['review_status'] == 'disputed'
    assert not state['can_pass']
    assert store.export(cid)['deliverable']['confirmed_issues'] == []


def test_old_run_cannot_replace_candidate_after_concurrent_source_change(store):
    state = create(store); cid = state['package']['case_id']
    result = run_review(state['package'], provider='local')
    package = deepcopy(state['package']); package['documents'][0]['text'] += ' 新补充。'
    changed = store.change_source(cid, package, '计算期间经办补充资料')
    with pytest.raises(ValueError, match='运行时来源已改变'):
        store.save_run(cid, state['source_hash'], result)
    assert store.get(cid)['source_hash'] == changed['source_hash']
    assert store.get(cid)['latest_run'] is None
    with store.connect() as db:
        assert db.execute('SELECT COUNT(*) FROM runs').fetchone()[0] == 0


def test_run_without_required_check_manifest_cannot_pass(store):
    state = create(store); cid = state['package']['case_id']
    state = store.save_run(cid, state['source_hash'], {'case_id':cid,'run_status':'completed','open_items':[],'required_checks':[]})
    assert not state['can_pass']


def test_audit_hash_chain_validates_and_detects_payload_change(store):
    state = make_passed(store); cid = state['package']['case_id']
    assert audit_integrity(state['review_events'])['valid']
    assert state['audit_integrity']['checked_events'] == len(state['review_events'])
    exported = store.export(cid)
    assert exported['audit_integrity']['valid']
    with store.connect() as db:
        row = db.execute('SELECT seq,payload FROM events WHERE case_id=? ORDER BY seq LIMIT 1', (cid,)).fetchone()
        payload = json.loads(row['payload']); payload['reason'] = '模拟未经重算链条的外部修改'
        db.execute('UPDATE events SET payload=? WHERE seq=?', (json.dumps(payload,ensure_ascii=False),row['seq']))
    state = store.get(cid)
    assert not state['audit_integrity']['valid']
    assert not state['can_pass']
    assert any(i['kind'] == 'audit_invalid' for i in state['open_items'])
    assert not store.export(cid)['deliverable']['passed']
    assert store.export(cid)['deliverable']['confirmed_issues'] == []


def test_list_coverage_and_stale_come_from_current_declared_sources(store):
    state = create(store); cid = state['package']['case_id']
    row = store.list()[0]
    assert row['coverage_summary'] == 'full' and row['stale'] is False
    run_current(store, cid)
    package = deepcopy(store.get(cid)['package']); package['coverage'][0]['status'] = 'partial'
    store.change_source(cid, package, '发现覆盖缺口')
    row = store.list()[0]
    assert row['coverage_summary'] == 'partial' and row['stale'] is True
    assert row['run_status'] == 'stale'


def test_changes_keep_frozen_schema_when_client_omits_it(store):
    state = create(store); package = deepcopy(state['package']); del package['schema']
    package['documents'][0]['text'] += ' 补充材料说明。'
    changed = store.change_source(package['case_id'], package, '正文修订')
    assert changed['package']['schema'] == state['package']['schema']
    package = deepcopy(changed['package']); del package['schema']; package['schema_version'] = 'S2.0'
    with pytest.raises(ValueError, match='Schema'):
        store.change_source(package['case_id'], package, '没有规范内容不能迁移')


def test_needs_review_event_lists_each_invalidated_human_record(store):
    prior = make_passed(store); cid = prior['package']['case_id']
    snapshot_id = prior['latest_run']['snapshot_id']
    records = [e for e in prior['review_events'] if e.get('snapshot_id') == snapshot_id]
    assert records
    package = deepcopy(prior['package']); package['documents'][0]['text'] += ' 新的说明。'
    changed = store.change_source(cid,package,'来源文本补充')
    event = next(e for e in reversed(changed['review_events']) if e['action']=='needs_review')
    assert event['previous_source_hash'] == prior['source_hash']
    assert event['source_hash'] == changed['source_hash']
    assert {r['event_id'] for r in event['review_records']} == {r['event_id'] for r in records}
    assert {(r['event_id'],r['target_id']) for r in event['review_records']} == {(r['event_id'],r['target_id']) for r in records}
    assert all(r['snapshot_id'] == snapshot_id for r in event['review_records'])


def test_lead_actions_keep_actor_and_close_even_when_origin_candidate_disappears(store):
    from aml_qc.workflow import extract_local
    from test_model_safety import ScriptedModel, json_message, semantic_response
    state = create(store); cid = state['package']['case_id']
    semantic = semantic_response(state['package'])
    semantic['leads'] = [{'question':'经营范围是否需补充说明？', 'observation':'经营资料包含经营范围，建议人工核对说明。',
                          'basis_refs':['document:kyc']}]
    model = ScriptedModel([json_message(extract_local(state['package'])),
        json_message({'focuses': semantic['focuses']}),
        json_message({'gaps': semantic['gaps'], 'leads': semantic['leads']})])
    result = run_review(state['package'],provider='frozen',model=model)
    state = store.save_run(cid,state['source_hash'],result)
    issue_id = next(i['issue_id'] for i in result['issues'] if i['type']=='new_lead')
    upgraded = store.review(cid,action='upgrade_lead',target_id=issue_id,reason='请追加核对新增线索',actor='special-reviewer',
                            snapshot_id=state['latest_run']['snapshot_id'],expected_event_id=None)
    event = next(e for e in reversed(upgraded['review_events']) if e['action']=='source_changed')
    assert event['actor']=='special-reviewer'
    assert event['context']['review_action']=='upgrade_lead'
    assert event['context']['target_id']==issue_id
    assert upgraded['package']['review_scope']['upgraded_leads'][0]['origin_issue_id']==issue_id
    refreshed = run_current(store,cid)
    assert all(i['issue_id']!=issue_id for i in refreshed['latest_run']['issues'])
    closed = store.review(cid,action='close_lead',target_id=issue_id,reason='复核后关闭现存升级线索',actor='second-reviewer',
                         snapshot_id=refreshed['latest_run']['snapshot_id'],expected_event_id=refreshed['lead_dispositions'][0]['expected_event_id'])
    assert closed['package']['review_scope']['upgraded_leads']==[]
    event = next(e for e in reversed(closed['review_events']) if e['action']=='source_changed')
    assert event['actor']=='second-reviewer' and event['context']['review_action']=='close_lead'
    assert closed['stale']


def test_task_or_non_lead_issue_cannot_be_upgraded_as_new_lead(store):
    state = create(store,2); cid = state['package']['case_id']; state = run_current(store,cid)
    with pytest.raises(ValueError,match='具体线索'):
        bound_review(store, cid,action='upgrade_lead',target_id='task',reason='不能把整个任务升级为新增线索')
    issue = next(i for i in state['latest_run']['issues'] if i['type']=='claim_error')
    with pytest.raises(ValueError,match='只有新增线索'):
        bound_review(store, cid,action='upgrade_lead',target_id=issue['issue_id'],reason='事实错误已是问题，不是新增线索')


def test_export_contains_frozen_sources_and_runs_to_resolve_old_review_spans(store):
    initial = create(store,2); cid = initial['package']['case_id']; initial = run_current(store,cid)
    issue = next(i for i in initial['latest_run']['issues'] if i['type']=='claim_error')
    reviewed = bound_review(store, cid,action='confirm',target_id=issue['issue_id'],reason='确认原版本次数矛盾')
    event_id = reviewed['review_events'][-1]['event_id']
    old_text = reviewed['package']['documents'][0]['text']
    updated = deepcopy(reviewed['package']); updated['documents'][0]['text'] = '修订后：'+old_text
    store.change_source(cid,updated,'在段首加入说明')
    run_current(store,cid)
    exported = store.export(cid)
    assert len(exported['historical_sources']) == 2
    assert len(exported['historical_runs']) == 2
    old_event = next(e for e in exported['review_events'] if e['event_id']==event_id)
    frozen_package = next(s['package'] for s in exported['historical_sources'] if s['source_hash']==old_event['source_hash'])
    old_run = next(r for r in exported['historical_runs'] if r['snapshot_id']==old_event['snapshot_id'])
    assert old_run['source_hash']==old_event['source_hash']
    span = next(e for e in old_event['evidence'] if e['type']=='document_span')
    document = next(d for d in frozen_package['documents'] if d['document_id']==span['document_id'] and d['revision']==span['revision'])
    original_claim = next(c for c in old_run['claims'] if c['source']['span']==span['span'])
    assert document['text'][span['span'][0]:span['span'][1]]==original_claim['text']
    assert document['text']==old_text
    assert exported['deliverable']['confirmed_issues']==[]
    assert exported['candidates_and_open_items']['current_confirmed_issues']==[]
    assert '不计入当前可交付集合' in exported['historical_scope']


def test_executor_change_invalidates_prior_pass_without_source_edit(store, monkeypatch):
    passed = make_passed(store)
    cid = passed['package']['case_id']
    monkeypatch.setattr('aml_qc.store.IMPLEMENTATION_HASH', 'different-reviewed-code-version')
    current = store.get(cid)
    assert current['source_hash'] == passed['source_hash']
    assert current['engine_changed'] and current['stale']
    assert not current['can_pass']
    assert not store.export(cid)['deliverable']['passed']


def test_old_page_cannot_confirm_a_new_ready_snapshot(store):
    old = make_passed(store); cid = old['package']['case_id']
    old_binding = review_binding(old, 'task')
    run_current(store, cid)
    current = close_manual(store, cid)
    assert current['can_pass'] and current['latest_run']['snapshot_id'] != old_binding['snapshot_id']
    before = deepcopy(current['review_events'])
    with pytest.raises(ValueError, match='当前快照'):
        store.review(cid, action='confirm', target_id='task', reason='旧页面不能确认未查看的新快照', **old_binding)
    assert store.get(cid)['review_events'] == before
    assert not store.export(cid)['deliverable']['passed']
    assert bound_review(store, cid, action='reconfirm', target_id='task', reason='明确核对新快照')['can_pass']


@pytest.mark.parametrize('target', ['task', 'issue'])
def test_same_snapshot_target_event_cas_rejects_stale_human_overwrite(store, target):
    state = create(store, 2); cid = state['package']['case_id']; state = run_current(store, cid)
    target_id = 'task' if target == 'task' else next(i['issue_id'] for i in state['latest_run']['issues'] if i['type'] == 'claim_error')
    binding = review_binding(state, target_id)
    first = store.review(cid, action='dispute', target_id=target_id, reason='第一位人员保留争议', **binding)
    with pytest.raises(ValueError, match='其他人员'):
        store.review(cid, action='request_correction', target_id=target_id, reason='第二位人员仍使用旧事件', **binding)
    assert store.get(cid)['review_events'] == first['review_events']
    updated = bound_review(store, cid, action='request_correction', target_id=target_id, reason='刷新后明确处理当前争议')
    assert updated['review_events'][-1]['previous_event_id'] == first['review_events'][-1]['event_id']
    assert not updated['can_pass']


@pytest.mark.parametrize('missing', ['snapshot_id', 'expected_event_id'])
def test_task_and_issue_review_require_explicit_anchors(store, missing):
    state = create(store, 2); cid = state['package']['case_id']; state = run_current(store, cid)
    for target_id in ['task', next(i['issue_id'] for i in state['latest_run']['issues'] if i['type'] == 'claim_error')]:
        binding = review_binding(state, target_id); del binding[missing]
        with pytest.raises(ValueError, match='当前快照|预期事件'):
            store.review(cid, action='dispute', target_id=target_id, reason='缺少页面所见锚点不能裁决', **binding)
    assert store.get(cid)['review_events'] == state['review_events']


def test_issue_review_rejects_prior_snapshot_but_new_source_has_own_event_chain(store):
    state = create(store, 2); cid = state['package']['case_id']; state = run_current(store, cid)
    target_id = next(i['issue_id'] for i in state['latest_run']['issues'] if i['type'] == 'claim_error')
    old = bound_review(store, cid, action='dispute', target_id=target_id, reason='旧来源判断分歧')
    binding = review_binding(old, target_id)
    package = deepcopy(old['package']); package['documents'][1]['text'] += '新增背景资料。'
    store.change_source(cid, package, '经办新增资料')
    current = run_current(store, cid)
    assert review_binding(current, target_id)['expected_event_id'] is None
    with pytest.raises(ValueError, match='当前快照'):
        store.review(cid, action='confirm', target_id=target_id, reason='旧页面不能确认新资料', **binding)
    accepted = bound_review(store, cid, action='confirm', target_id=target_id, reason='核对新来源中的当前问题')
    assert accepted['review_events'][-1]['previous_event_id'] is None
    assert accepted['review_events'][-1]['source_hash'] != old['source_hash']
