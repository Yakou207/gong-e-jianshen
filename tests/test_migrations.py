"""Cross-case rule migrations use frozen previews and explicit per-case commits."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from test_store import bound_review

from aml_qc.depgraph import business_result, digest
from aml_qc.ingest import load_case
from aml_qc.schema import load_schema
from aml_qc.store import Store
from aml_qc.workflow import run_review

DATA=Path(__file__).resolve().parents[1]/'data/synthetic'

@pytest.fixture
def store(tmp_path):
    return Store(tmp_path/'migration.sqlite3')

def create(store,n=1,*,run=True,edit=None):
    p=load_case(DATA/f'seed-{n:02d}.json'); p.update(task_mode='annotation_only',alert=None)
    p['review_scope']['target_labels']=['F1','F2']
    if edit: edit(p)
    s=store.create(p)
    return store.save_run(p['case_id'],s['source_hash'],run_review(s['package'],provider='local')) if run else s

def target(version='S1.1-migration-test'):
    s=load_schema(); s['schema_version']=version
    s['features']['F1']['minimum_days']=1; s['features']['F2']['ratio_numerator']=9
    return s

def preview(store,selected,schema=None,base=None):
    return store.preview_migration(base_schema_hash=base or digest(load_schema()),target_schema=schema or target(),
        selected_case_ids=selected,reason='预览演示规则变更并显式确定范围',actor='rule-reviewer')

def apply(store,p,cid):
    return store.migrate_case(p['preview_id'],cid,preview_hash=p['preview_hash'],reason='核对差异后逐案迁移',actor='operator')

def passed(store,s):
    cid=s['package']['case_id']
    for a in s['annotations']:
        s=store.review(cid,target_id=a['annotation_id'],action='confirm',reason='人工逐项核对演示特征',
            snapshot_id=a['snapshot_id'],expected_event_id=None,evidence=a['evidence'])
    for issue in s['latest_run']['issues']:
        if issue['type']=='manual_extraction':
            s=bound_review(store, cid,target_id=issue['issue_id'],action='close_item',resolution='addressed',reason='仅F1/F2范围，已核对抽取未决')
    assert s['can_pass']
    return bound_review(store, cid,target_id='task',action='confirm',reason='本次限定任务人工完成确认')


def test_two_selected_cases_migrate_individually_third_and_business_data_stay_unchanged(store):
    states={f'seed-{n:02d}':create(store,n) for n in (1,2,3)}
    states['seed-01']=passed(store,states['seed-01'])
    plan=preview(store,['seed-01','seed-02'])
    assert {r['case_id'] for r in plan['cases']}==set(states)
    assert {c['path'] for c in plan['changes']} >= {'/schema_version','/features/F1/minimum_days','/features/F2/ratio_numerator'}
    row=next(r for r in plan['cases'] if r['case_id']=='seed-01')
    ids={e['event_id'] for e in states['seed-01']['review_events']}
    assert {r['event_id'] for r in row['review_records']}==ids
    assert row['affected_rule_codes']==['F1','F2'] and 'features' in row['affected_nodes']
    assert 'source:schema' in row['changed_sources'] and not row['requires_full_recheck']
    one=apply(store,plan,'seed-01')
    assert one['case']['stale'] and not one['case']['can_pass'] and one['receipt']['run_started'] is False
    assert one['case']['latest_run']['run_id']==states['seed-01']['latest_run']['run_id']
    assert store.get('seed-02')['source_hash']==states['seed-02']['source_hash']
    assert one['preview']['preview_hash']==plan['preview_hash']
    for key in ('transactions','documents','materials','entity_mappings','coverage','review_scope','alert'):
        assert one['case']['package'][key]==states['seed-01']['package'][key]
    newlink=one['case']['package']['material_links'][0]; oldlink=states['seed-01']['package']['material_links'][0]
    assert {k:v for k,v in newlink.items() if k!='schema_version'}=={k:v for k,v in oldlink.items() if k!='schema_version'}
    assert newlink['schema_version']==target()['schema_version']
    audit=store.export('seed-01')
    assert not audit['deliverable']['passed'] and audit['deliverable']['annotations']==[]
    assert all(a['schema_version']=='S1.0' for a in audit['annotations'])
    assert audit['historical_sources'][0]['package']==states['seed-01']['package']
    assert len(audit['schema_migrations']['receipts'])==1
    invalidation=next(e for e in reversed(one['case']['review_events']) if e['action']=='needs_review')
    assert {r['event_id'] for r in invalidation['review_records']}==ids
    assert one['case']['audit_integrity']['valid']
    two=apply(store,plan,'seed-02'); assert len(two['preview']['receipts'])==2
    third=store.get('seed-03')
    for key in ('package','source_hash','latest_run','review_events'):
        assert third[key]==states['seed-03'][key]
    with pytest.raises(ValueError,match='未包含'): apply(store,plan,'seed-03')


def test_known_feature_reversals_and_independent_full_incremental_equivalence(store):
    old=create(store)
    assert {f['feature_code']:f['result'] for f in old['latest_run']['features']}=={'F1':'not_met','F2':'met'}
    new=apply(store,preview(store,['seed-01']),'seed-01')['case']
    full=run_review(new['package'],provider='local',strategy='full')
    incremental=run_review(new['package'],provider='local',strategy='incremental',previous=old['latest_run']['snapshot'])
    assert business_result(full)==business_result(incremental)
    assert {f['feature_code']:f['result'] for f in full['features']}=={'F1':'met','F2':'not_met'}
    assert incremental['stats']['recomputed']>0
    current=store.save_run('seed-01',new['source_hash'],incremental)
    assert not current['stale'] and current['annotation_pending_count']>0
    assert not store.export('seed-01')['deliverable']['passed']


def test_missing_graph_full_case_and_empty_selection_never_migrates(store):
    old=create(store,run=False); plan=preview(store,[]); row=plan['cases'][0]
    assert row['requires_full_recheck'] and row['impact_mode']=='full_case'
    assert 'features' in row['affected_nodes'] and row['review_records']==[] and not row['selected']
    with pytest.raises(ValueError,match='未包含'): apply(store,plan,'seed-01')
    assert store.get('seed-01')['source_hash']==old['source_hash']


@pytest.mark.parametrize('change',['source','run','review','implementation'])
def test_preview_rejects_source_run_review_or_implementation_drift(store,monkeypatch,change):
    old=create(store); plan=preview(store,['seed-01'])
    if change=='source':
        package=deepcopy(old['package']); package['documents'][0]['text']+=' 补充。'
        store.change_source('seed-01',package,'预览后新增资料')
    elif change=='run':
        store.save_run('seed-01',old['source_hash'],run_review(old['package'],provider='local'))
    elif change=='review':
        a=old['annotations'][0]
        store.review('seed-01',action='abstain',target_id=a['annotation_id'],reason='预览后人工弃标',
                     snapshot_id=a['snapshot_id'],expected_event_id=None,evidence=[])
    else: monkeypatch.setattr('aml_qc.store.IMPLEMENTATION_HASH','new-review-code')
    source=store.get('seed-01')['source_hash']
    with pytest.raises(ValueError,match='预览已过期'): apply(store,plan,'seed-01')
    assert store.get('seed-01')['source_hash']==source
    assert store.migration_preview(plan['preview_id'])['cases'][0]['status']=='stale'
    with store.connect() as db: assert db.execute('SELECT COUNT(*) FROM migration_receipts').fetchone()[0]==0


def test_duplicate_forged_hash_and_target_version_conflict_rejected(store):
    create(store); plan=preview(store,['seed-01'])
    with pytest.raises(ValueError,match='指纹不匹配'):
        store.migrate_case(plan['preview_id'],'seed-01',preview_hash='forged',reason='伪造预览',actor='tester')
    altered=target(); altered['features']['F1']['minimum_days']=2
    with pytest.raises(ValueError,match='绑定不同内容'): preview(store,['seed-01'],altered)
    applied=apply(store,plan,'seed-01')
    with pytest.raises(ValueError,match='已迁移'): apply(store,plan,'seed-01')
    assert store.get('seed-01')['source_hash']==applied['case']['source_hash']
    with store.connect() as db:
        assert db.execute('SELECT COUNT(*) FROM migration_receipts').fetchone()[0]==1
        assert db.execute('SELECT COUNT(*) FROM sources').fetchone()[0]==2


@pytest.mark.parametrize('mutate',[
    lambda s:s['features']['F2'].update(window_days=0),lambda s:s.pop('claims'),
    lambda s:s.update(unimplemented_rule=True),lambda s:s['claims'].update(exact_requires_full_coverage=False),
    lambda s:s['claims'].update(exact_requires_full_coverage=1),
    lambda s:s['features']['F1'].update(hidden_threshold=7),lambda s:s['labels']['F1'].pop('abstain_when'),
    lambda s:s['labels'].pop('count')])
def test_invalid_incomplete_or_unimplemented_schema_rejected(store,mutate):
    old=create(store); schema=target(); mutate(schema)
    with pytest.raises(ValueError): preview(store,['seed-01'],schema)
    assert store.get('seed-01')['source_hash']==old['source_hash']
    with store.connect() as db: assert db.execute('SELECT COUNT(*) FROM migration_previews').fetchone()[0]==0


def test_same_version_unknown_base_duplicate_and_unknown_case_rejected(store):
    create(store)
    for schema,base,selection in [(load_schema(),None,['seed-01']),(target(),'invented',['seed-01']),
                                (target(),None,['seed-01','seed-01']),(target(),None,['absent'])]:
        with pytest.raises(ValueError): preview(store,selection,schema,base)


def test_existing_and_implicit_material_version_conflicts_are_not_washed_away(store):
    def edit(p):
        conflict=deepcopy(p['material_links'][0]); conflict.update(link_id='conflict',schema_version='S0-conflict')
        implicit=deepcopy(p['material_links'][0]); implicit['link_id']='implicit'; implicit.pop('schema_version')
        p['material_links'] += [conflict,implicit]
    old=create(store,edit=edit); plan=preview(store,['seed-01']); row=plan['cases'][0]
    assert {w['code'] for w in row['material_link_warnings']}=={'existing_version_conflict','implicit_binding_pinned'}
    new=apply(store,plan,'seed-01')['case']['package']; links={r['link_id']:r for r in new['material_links']}
    assert links['conflict']['schema_version']=='S0-conflict' and links['implicit']['schema_version']=='S1.0'
    assert links[old['package']['material_links'][0]['link_id']]['schema_version']==target()['schema_version']


def test_related_other_version_and_full_per_case_differences_are_visible(store):
    create(store)
    def edit(p):
        p['schema']=load_schema(); p['schema']['schema_version']='S0-related'; p['schema_version']='S0-related'
        p['schema']['features']['F1']['minimum_days']=7
    create(store,2,edit=edit); plan=preview(store,['seed-02'])
    row=next(r for r in plan['cases'] if r['case_id']=='seed-02')
    assert next(c['before'] for c in row['changes'] if c['path']=='/features/F1/minimum_days')==7
    apply(store,plan,'seed-02')
    assert store.get('seed-01')['package']['schema_version']=='S1.0'


def test_preview_and_receipt_tampering_are_detected(store):
    create(store); plan=preview(store,['seed-01'])
    apply(store,plan,'seed-01')
    with store.connect() as db:
        row=db.execute('SELECT payload FROM migration_receipts').fetchone()
        value=json.loads(row[0]); value['reason']='外部改写'
        db.execute('UPDATE migration_receipts SET payload=?',(json.dumps(value),))
    with pytest.raises(ValueError,match='回执审计内容'): store.migration_preview(plan['preview_id'])
    with store.connect() as db:
        row=db.execute('SELECT payload FROM migration_previews').fetchone()
        value=json.loads(row[0]); value['target_schema']['features']['F1']['minimum_days']=8
        db.execute('UPDATE migration_previews SET payload=?',(json.dumps(value),))
    with pytest.raises(ValueError,match='预览审计内容'): store.migration_preview(plan['preview_id'])


def test_ordinary_source_cannot_bypass_schema_preview_and_import_versions_are_unique(store):
    old=create(store); package=deepcopy(old['package']); package['schema']=target(); package['schema_version']=target()['schema_version']
    with pytest.raises(ValueError,match='迁移预览'): store.change_source('seed-01',package,'尝试绕过预览')
    package=deepcopy(old['package']); package['schema']['features']['F1']['minimum_days']=9
    with pytest.raises(ValueError,match='迁移预览'): store.change_source('seed-01',package,'同版本偷偷改阈值')
    package['case_id']='different-id'
    with pytest.raises(ValueError,match='绑定不同内容'): store.create(package)
    plan=preview(store,['seed-01'])
    package=deepcopy(old['package']); package.update(case_id='another-id',schema=target(),schema_version=target()['schema_version'])
    package['schema']['features']['F1']['minimum_days']=2
    with pytest.raises(ValueError,match='绑定不同内容'): store.create(package)


def test_migration_source_events_and_receipt_rollback_together_on_failure(store,monkeypatch):
    old=create(store); plan=preview(store,['seed-01'])
    event=store._event
    def fail_receipt_event(db,case_id,payload):
        if payload['action']=='schema_migrated':
            raise RuntimeError('测试迁移事务后半段故障')
        return event(db,case_id,payload)
    monkeypatch.setattr(store,'_event',fail_receipt_event)
    with pytest.raises(RuntimeError,match='后半段故障'): apply(store,plan,'seed-01')
    current=store.get('seed-01')
    assert current['source_hash']==old['source_hash'] and current['review_events']==old['review_events']
    with store.connect() as db:
        assert db.execute('SELECT COUNT(*) FROM sources').fetchone()[0]==1
        assert db.execute('SELECT COUNT(*) FROM migration_receipts').fetchone()[0]==0
    assert store.migration_preview(plan['preview_id'])['cases'][0]['can_apply']
