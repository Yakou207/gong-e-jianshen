"""Version-bound positive labels, human input revisions, and task/export gates."""
from copy import deepcopy
import csv
import json
from pathlib import Path

import pytest

from aml_qc import core
from aml_qc.annotations import validate_evidence
from aml_qc.exports import export_bundle
from aml_qc.ingest import load_case
from aml_qc.store import Store
from aml_qc.workflow import run_review

DATA = Path(__file__).resolve().parents[1] / 'data' / 'synthetic'
BASE_LABELS = ['F1', 'F2', 'count', 'material_relation', 'alert_response']


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / 'annotations.sqlite3')


def start(store, number=1, *, labels=None, edit=None, edit_result=None):
    package = load_case(DATA / f'seed-{number:02d}.json')
    if labels is not None:
        package['review_scope']['target_labels'] = labels
    if edit:
        edit(package)
    state = store.create(package)
    result = run_review(state['package'], provider='local')
    if edit_result:
        edit_result(result, state['package'])
    return store.save_run(package['case_id'], state['source_hash'], result)


def item(state, label, kind=None):
    return next(a for a in state['annotations'] if a['label'] == label and (kind is None or a['kind'] == kind))


def decide(store, state, annotation, action='confirm', **kwargs):
    return store.review(state['package']['case_id'], target_id=annotation['annotation_id'], action=action,
        reason=kwargs.pop('reason', '人工核对当前原文、限定范围和可解析证据'), snapshot_id=annotation['snapshot_id'],
        expected_event_id=annotation['review']['event_id'], evidence=kwargs.pop('evidence', annotation['evidence']), **kwargs)


def close_extraction(store, state):
    for issue in state['latest_run']['issues']:
        if issue['type'] == 'manual_extraction':
            state = store.review(state['package']['case_id'], target_id=issue['issue_id'], action='close_item',
                                 resolution='addressed', reason='人工完整检查理由，确认本次抽取的适用事实及限定词')
    return state


def adjudicate_produced(store, state):
    for annotation in list(state['annotations']):
        if annotation['kind'] == 'scope_review':
            continue
        action = 'revise' if annotation['kind'] == 'semantic' else 'confirm'
        kwargs = {'new_value': 'addressed'} if action == 'revise' else {}
        state = decide(store, state, annotation, action, **kwargs)
    return state


def test_positive_results_are_candidates_and_all_eight_targets_are_visible(store):
    state = start(store)
    assert {a['label'] for a in state['annotations']} == set(state['package']['schema']['labels'])
    assert item(state, 'count')['candidate_value'] == 'supported'
    assert item(state, 'material_relation')['candidate_value'] == 'corresponds'
    assert item(state, 'F1')['candidate_value'] in {'met', 'not_met'}
    assert all(a['review']['final_value'] is None for a in state['annotations'])
    assert state['annotation_pending_count'] == len(state['annotations'])
    missing = [a for a in state['annotations'] if a['kind'] == 'scope_review']
    assert {a['label'] for a in missing} == {'amount_sum', 'counterparty', 'time_range'}
    assert all(a['value'] is None and a['coverage_status'] == 'no_candidate' for a in missing)
    assert not state['can_pass']


def test_every_label_and_final_task_confirmation_needed_for_deliverable(store):
    state = start(store, labels=BASE_LABELS)
    cid = state['package']['case_id']
    state = close_extraction(store, state)
    state = adjudicate_produced(store, state)
    assert state['can_pass'] and not state['annotation_pending'] and not state['open_items']
    assert store.export(cid)['deliverable']['annotations'] == []
    state = store.review(cid, action='confirm', target_id='task', reason='逐标签与全部必需核验完成后确认任务')
    exported = store.export(cid)
    assert exported['deliverable']['passed']
    assert len(exported['deliverable']['annotations']) == len(state['annotations'])
    label = item(state, 'F1')
    state = decide(store, state, label, 'confirm')
    assert state['can_pass'] and state['review_status'] != '本次质检范围内通过'
    assert not store.export(cid)['deliverable']['annotations']


def test_abstain_and_dispute_keep_null_and_cannot_satisfy_required_label(store):
    state = start(store, labels=BASE_LABELS)
    annotation = item(state, 'F1')
    state = decide(store, state, annotation, 'abstain', evidence=[])
    reviewed = item(state, 'F1')['review']
    assert reviewed['status'] == 'abstained' and reviewed['final_value'] is None and not reviewed['valid']
    assert any(p['label'] == 'F1' for p in state['annotation_pending'])
    state = decide(store, state, item(state, 'F1'), 'dispute', evidence=[])
    assert state['review_status'] == 'disputed' and not state['can_pass']
    assert item(state, 'F1')['review']['final_value'] is None


def test_snapshot_event_cas_and_evidence_validation(store):
    state = start(store, labels=BASE_LABELS); cid = state['package']['case_id']
    annotation = item(state, 'F1')
    with pytest.raises(ValueError, match='当前快照'):
        store.review(cid, action='confirm', target_id=annotation['annotation_id'], reason='缺少快照不能确认')
    with pytest.raises(ValueError, match='证据'):
        decide(store, state, annotation, evidence=[])
    forged = {'type':'document_span','document_id':'narrative','revision':'old','span':[0,3]}
    with pytest.raises(ValueError, match='不能解析'):
        decide(store, state, annotation, evidence=[forged])
    state = decide(store, state, annotation)
    with pytest.raises(ValueError, match='其他人员裁决'):
        decide(store, state, annotation)
    assert item(state, 'F1')['review']['valid']
    with pytest.raises(ValueError, match='不允许'):
        decide(store, state, item(state, 'F1'), 'revise', new_value='not_met')


def test_old_label_decision_requires_explicit_reconfirmation_against_new_snapshot(store):
    state = start(store, labels=BASE_LABELS); cid = state['package']['case_id']
    state = decide(store, state, item(state, 'F1'))
    old_event = item(state, 'F1')['review']['event_id']
    package = deepcopy(state['package']); package['documents'][0]['text'] = '说明：' + package['documents'][0]['text']
    state = store.change_source(cid, package, '补充理由')
    assert item(state, 'F1')['allowed_actions'] == []
    state = store.save_run(cid, state['source_hash'], run_review(state['package'], provider='local'))
    annotation = item(state, 'F1')
    assert annotation['review']['status'] == 'needs_review' and annotation['prior_review']['event_id'] == old_event
    with pytest.raises(ValueError, match='旧人工事件'):
        decide(store, state, annotation, 'reconfirm')
    state = decide(store, state, annotation, 'reconfirm', previous_event_id=old_event)
    assert item(state, 'F1')['review']['valid']
    assert state['audit_integrity']['valid']


def bad_extraction(result, package):
    claim = result['claims'][0]
    claim['value'] = 99
    result['claim_results'][0] = core.verify_claim(package, claim)
    target = 'claim:' + claim['claim_id']
    result['issues'].append({'issue_id':'wrong-extraction','type':'claim_error','target_id':target,'evidence':result['claim_results'][0]['evidence']})
    result['open_items'].append({'item_id':'wrong-extraction','kind':'claim_error','target_id':target,'title':'抽取错次数'})


def test_claim_revision_recomputes_but_keeps_original_problem_pending(store):
    state = start(store, labels=BASE_LABELS, edit_result=bad_extraction)
    annotation = item(state, 'count'); original = deepcopy(annotation)
    assert annotation['candidate_value'] == 'contradicted'
    with pytest.raises(ValueError, match='claim_patch'):
        decide(store, state, annotation, 'revise', new_value='supported')
    state = decide(store, state, annotation, 'revise', claim_patch={'quote':annotation['claim']['text'], 'value':1})
    revised = item(state, 'count')
    assert revised['candidate_value'] == original['candidate_value'] == 'contradicted'
    assert revised['claim']['value'] == 99
    assert revised['review']['final_value'] is None and not revised['review']['valid']
    assert revised['review']['proposed_value'] == 'supported'
    assert revised['review']['verification']['comparison']['expected'] == 1
    assert revised['editable_claim']['value'] == 1
    assert revised['review']['origin'] == 'human_extraction_correction_pending'
    assert any(i['item_id'] == 'wrong-extraction' for i in state['open_items'])
    assert state['review_events'][-1]['candidate_value'] == 'contradicted'
    assert state['review_events'][-1]['claim_patch']['value'] == 1
    with pytest.raises(ValueError, match='唯一定位'):
        decide(store, state, revised, 'revise', claim_patch={'quote':'并不存在的陈述','value':1})


def test_partial_coverage_and_unresolved_query_cannot_be_manually_relabelled_supported(store):
    state = start(store, labels=BASE_LABELS, edit=lambda p:p['coverage'][0].update(status='partial'))
    annotation = item(state, 'count')
    state = decide(store, state, annotation, 'confirm')
    assert not item(state, 'count')['review']['valid']
    state = decide(store, state, item(state, 'count'), 'revise', claim_patch={'value':1,'operator':'exact'})
    assert item(state, 'count')['review']['final_value'] == 'insufficient_evidence'
    assert not item(state, 'count')['review']['valid'] and not state['can_pass']


def test_resolving_mistaken_extraction_identity_reexecutes_but_does_not_approve_new_object(store):
    state = start(store, 4, labels=BASE_LABELS)
    annotation = item(state, 'count')
    assert annotation['execution_status'] == 'identity_unresolved'
    assert 'confirm' not in annotation['allowed_actions']
    state = decide(store, state, annotation, 'revise', claim_patch={'counterparty_ref':'supplier-b', 'value':1})
    revised = item(state, 'count')
    assert revised['review']['verification']['execution_status'] == 'completed'
    assert revised['review']['final_value'] is None and revised['review']['proposed_value'] == 'supported'
    assert any(c['check_id'] == annotation['target_id'] for c in state['pending_checks'])


def test_pending_material_relation_allows_explicit_human_judgement_without_faking_calculation(store):
    state = start(store, labels=BASE_LABELS, edit=lambda p:p['material_links'][0].pop('relation_template'))
    annotation = item(state, 'material_relation')
    assert annotation['candidate_value'] == 'pending_judgement' and annotation['editable']
    state = decide(store, state, annotation, 'revise', new_value='corresponds')
    revised = item(state, 'material_relation')
    assert revised['candidate_value'] == 'pending_judgement'
    assert revised['review']['origin'] == 'human_judgement' and revised['review']['verification'] is None
    assert revised['review']['valid']
    assert all(i['target_id'] != revised['target_id'] for i in state['open_items'])


def test_missing_or_schema_conflicting_material_cannot_be_overridden(store):
    def edit(package):
        package['material_links'][0]['schema_version'] = 'another-version'
    state = start(store, labels=BASE_LABELS, edit=edit)
    annotation = item(state, 'material_relation')
    assert not annotation['editable']
    with pytest.raises(ValueError, match='不允许'):
        decide(store, state, annotation, 'revise', new_value='corresponds')


def test_no_candidate_is_not_automatic_pass_and_na_keeps_null_scope_decision(store):
    def edit(package):
        package.update(task_mode='annotation_only', alert=None)
        package['documents'][0]['text'] = '本次只补充商户背景，没有交易次数的陈述。'
    state = start(store, labels=['count'], edit=edit)
    annotation = item(state, 'count')
    assert annotation['kind'] == 'scope_review' and state['annotation_pending_count'] == 1
    assert not state['can_pass']
    state = decide(store, state, annotation, 'not_applicable', reason='全文只有商户背景，没有任何交易次数命题，因此次数标签不适用')
    review = item(state, 'count')['review']
    assert review['valid'] and review['adjudicated'] and review['final_value'] is None
    assert review['applicability'] == 'not_applicable'
    assert not state['can_pass'] and any(c['check_id']=='extraction' for c in state['pending_checks'])
    state = close_extraction(store, state)
    assert state['can_pass']
    assert state['latest_run']['claims'] == []


def test_failed_extraction_cannot_be_bypassed_with_no_candidate_na(store):
    def failed(result, package):
        result.update(claims=[], claim_results=[], run_status='failed', open_items=[],
                      required_checks=[{'check_id':'extraction','label':'抽取','status':'failed'}])
    state = start(store, labels=['count'], edit_result=failed)
    state = decide(store, state, item(state, 'count'), 'not_applicable')
    assert state['pending_checks'] and not state['can_pass']
    with pytest.raises(ValueError, match='不能通过'):
        store.review(state['package']['case_id'], action='confirm', target_id='task', reason='失败不能用范围裁决绕过')


def test_frozen_annotation_object_schema_and_export_survive_source_changes(store, tmp_path):
    state = start(store, labels=BASE_LABELS); cid=state['package']['case_id']
    old = deepcopy(item(state, 'F1'))
    package = deepcopy(state['package'])
    package['coverage_end']='2026-09-16T00:00:00+08:00'
    store.change_source(cid, package, '扩展检查范围')
    schema=deepcopy(package['schema']); schema['schema_version']='S1.1-test'
    from aml_qc.depgraph import digest
    plan=store.preview_migration(base_schema_hash=digest(package['schema']),target_schema=schema,
        selected_case_ids=[cid],reason='预览升级演示规范',actor='schema-reviewer')
    store.migrate_case(plan['preview_id'],cid,preview_hash=plan['preview_hash'],reason='明确迁移该案件规范',actor='schema-reviewer')
    exported = store.export(cid)
    frozen = next(a for a in exported['annotations'] if a['label']=='F1')
    assert frozen['object'] == old['object'] and frozen['schema_version']=='S1.0'
    export_bundle(exported, tmp_path/'out')
    rows=[json.loads(line) for line in (tmp_path/'out/candidates.jsonl').read_text().splitlines()]
    assert {row['schema_version'] for row in rows} == {'S1.0'}
    assert all(row['status']=='stale' and row['final_value'] is None for row in rows)
    assert (tmp_path/'out/deliverable.jsonl').read_text()==''
    current=store.get(cid)
    current=store.save_run(cid,current['source_hash'],run_review(current['package'],provider='local'))
    assert item(current,'F1')['annotation_id'] != old['annotation_id']
    assert item(current,'F1')['object']['end']==package['coverage_end']


@pytest.mark.parametrize('patch', [{'value':3}, {'end':'2026-09-02T00:00:00+08:00'}, {'operator':'at_least'}])
def test_changed_proposition_cannot_erase_original_once_vs_three_counterexample(store, patch):
    def edit(package):
        package.update(task_mode='annotation_only', alert=None)
    state = start(store, 2, labels=['count'], edit=edit)
    annotation = item(state, 'count')
    assert annotation['candidate_value'] == 'contradicted'
    state = decide(store, state, annotation, 'revise', claim_patch=patch)
    revised = item(state, 'count')
    assert revised['review']['proposed_value'] == 'supported'
    assert revised['review']['final_value'] is None and not revised['review']['valid']
    assert revised['review']['correction_status'] == 'pending_source_review'
    assert revised['claim']['value'] == 1
    assert any(i['target_id']==annotation['target_id'] for i in state['open_items'])
    state = close_extraction(store, state)
    assert not state['can_pass']
    assert not store.export(state['package']['case_id'])['deliverable']['annotations']
    if 'end' in patch:
        assert revised['object']['end'] != revised['review']['object']['end']
        assert revised['review']['object']['end'] == patch['end']


@pytest.mark.parametrize('missing', ['subject', 'counterparty', 'period', 'amount', 'currency'])
def test_untemplated_material_cannot_invent_missing_fields(store, missing):
    def edit(package):
        package['material_links'][0].pop('relation_template')
        package['materials'][0].pop(missing)
    state = start(store, labels=BASE_LABELS, edit=edit)
    annotation = item(state, 'material_relation')
    assert annotation['candidate_value']=='pending_judgement'
    assert not annotation['human_judgement_allowed'] and 'revise' not in annotation['allowed_actions']
    with pytest.raises(ValueError, match='不允许'):
        decide(store, state, annotation, 'revise', new_value='corresponds')


def test_missing_template_does_not_hide_mismatched_schema_version(store):
    def edit(package):
        package['material_links'][0].pop('relation_template')
        package['material_links'][0]['schema_version']='old-schema'
    state=start(store, labels=BASE_LABELS, edit=edit)
    assert not item(state,'material_relation')['human_judgement_allowed']


def test_only_span_repair_reverifies_unchanged_proposition(store):
    def corrupt(result, package):
        claim=result['claims'][0]
        claim['source']['span']=[0,1]
        result['claim_results'][0]=core.verify_claim(package,claim)
        target='claim:'+claim['claim_id']
        next(c for c in result['required_checks'] if c['check_id']==target)['status']='extraction_failed'
        result['open_items'].append({'item_id':'bad-span','kind':'claim_unresolved','target_id':target,'title':'陈述跨度无效'})
    state=start(store, labels=BASE_LABELS, edit_result=corrupt)
    annotation=item(state,'count')
    doc=next(d for d in state['package']['documents'] if d['document_id']=='narrative')
    proof=[{'type':'document_span','document_id':'narrative','revision':doc['revision'],'span':[0,len(doc['text'])]}]
    state=decide(store,state,annotation,'revise',claim_patch={'quote':annotation['claim']['text']},evidence=proof)
    revised=item(state,'count')
    assert revised['review']['valid'] and revised['review']['final_value']=='supported'
    assert revised['review']['correction_status'] is None
    assert not any(c['check_id']==annotation['target_id'] for c in state['pending_checks'])
    assert not any(i['item_id']=='bad-span' for i in state['open_items'])


def test_custom_reference_cannot_smuggle_old_or_fake_hash_metadata(store):
    state=start(store,labels=BASE_LABELS)
    forged={'type':'transactions','transaction_ids':[state['package']['transactions'][0]['transaction_id']],
            'transaction_set_version':'sha256:fake'}
    with pytest.raises(ValueError,match='不能解析'):
        decide(store,state,item(state,'F1'),evidence=[forged])


def test_final_export_uses_human_value_and_preserves_original_candidate(store,tmp_path):
    state=start(store,labels=BASE_LABELS)
    state=adjudicate_produced(store,close_extraction(store,state))
    cid=state['package']['case_id']
    store.review(cid,action='confirm',target_id='task',reason='逐项完成后最终确认本次质检任务')
    exported=store.export(cid)
    semantic=next(a for a in exported['deliverable']['annotations'] if a['kind']=='semantic')
    assert semantic['value']==semantic['final_value']=='addressed'
    assert semantic['machine_candidate_value']=='pending_judgement'
    assert semantic['object']==semantic['review']['object']
    export_bundle(exported,tmp_path/'passed')
    delivered=[json.loads(line) for line in (tmp_path/'passed/deliverable.jsonl').read_text().splitlines()]
    assert next(a for a in delivered if a['kind']=='semantic')['value']=='addressed'
    with (tmp_path/'passed/candidates.csv').open(encoding='utf-8-sig') as handle:
        candidate=next(row for row in csv.DictReader(handle) if row['collection']=='semantic')
    assert candidate['value']=='pending_judgement' and candidate['final_value']=='addressed'


@pytest.mark.parametrize('kind',['document_span','material','transactions'])
def test_known_reference_still_requires_valid_span_fields_or_transaction_ids(store,kind):
    state=start(store,labels=BASE_LABELS)
    package=state['package']
    doc=next(d for d in package['documents'] if d['document_id']=='narrative')
    material=package['materials'][0]
    invalid={'document_span':{'type':'document_span','document_id':'narrative','revision':doc['revision'],'span':[0,0],'text':''},
             'material':{'type':'material','material_id':material['material_id'],'revision':material['revision'],'field_paths':[]},
             'transactions':{'type':'transactions','transaction_ids':[]}}[kind]
    with pytest.raises(ValueError,match='不能解析'):
        validate_evidence(package,[invalid],[invalid])


def test_current_label_reviews_become_unusable_when_executor_changes(store,monkeypatch):
    state=start(store,labels=BASE_LABELS)
    state=adjudicate_produced(store,close_extraction(store,state))
    cid=state['package']['case_id']
    store.review(cid,action='confirm',target_id='task',reason='最终确认当前版本全部标签')
    monkeypatch.setattr('aml_qc.store.IMPLEMENTATION_HASH','different-adjudication-implementation')
    state=store.get(cid)
    assert state['engine_changed'] and state['stale']
    assert all(a['review']['status']=='needs_review' and not a['review']['valid'] and a['allowed_actions']==[] for a in state['annotations'])
    assert store.export(cid)['deliverable']['annotations']==[]


def test_material_manual_judgement_needs_material_and_selected_transaction_references(store):
    state=start(store,labels=BASE_LABELS,edit=lambda p:p['material_links'][0].pop('relation_template'))
    annotation=item(state,'material_relation')
    unrelated=item(state,'alert_response')['evidence']
    with pytest.raises(ValueError,match='对应材料字段'):
        decide(store,state,annotation,'revise',new_value='corresponds',evidence=unrelated)


def test_replacing_quote_content_without_changing_numeric_fields_still_requires_source_review(store):
    state=start(store,labels=BASE_LABELS)
    annotation=item(state,'count')
    state=decide(store,state,annotation,'revise',claim_patch={'quote':'零售订单明细尚未提交'})
    reviewed=item(state,'count')['review']
    assert reviewed['verification']['result']=='supported'
    assert reviewed['correction_status']=='pending_source_review'
    assert reviewed['final_value'] is None and not reviewed['valid']


@pytest.mark.parametrize('unreliable_field',[None,'amount','timestamp','counterparty_token'])
def test_material_judgement_allows_reliable_selected_partial_but_not_explicit_unreliable_fields(store,unreliable_field):
    def edit(package):
        package['material_links'][0].pop('relation_template')
        package['coverage'][0]['status']='partial'
        if unreliable_field:
            declaration=deepcopy(package['coverage'][0])
            declaration.update(coverage_id='unreliable-selected-field',fields=[unreliable_field],reliable=False)
            package['coverage'].append(declaration)
    state=start(store,labels=BASE_LABELS,edit=edit)
    annotation=item(state,'material_relation')
    assert annotation['human_judgement_allowed'] is (unreliable_field is None)
    if unreliable_field:
        with pytest.raises(ValueError,match='不允许'):
            decide(store,state,annotation,'revise',new_value='corresponds')
    else:
        state=decide(store,state,annotation,'revise',new_value='corresponds')
        assert item(state,'material_relation')['review']['valid']
