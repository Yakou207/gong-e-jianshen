"""HTTP contract tests. Paid/model execution is always replaced locally."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from unittest.mock import patch

from fastapi.testclient import TestClient
import pytest

from aml_qc import llm
from aml_qc.ingest import load_case
from aml_qc.workflow import run_review as local_review

# api exposes a default ASGI app at import time. Keep that initialization away
# from the real workspace database and credentials, too; fixtures use tmp_path.
with TemporaryDirectory() as import_dir:
    with patch.object(llm, 'settings', return_value={'AML_QC_DB':str(Path(import_dir) / 'default.sqlite3'),
                     'DEEPSEEK_API_KEY':'','DEEPSEEK_MODEL':'mock-model'}):
        from aml_qc import api
    api.settings = llm.settings

DATA = Path(__file__).resolve().parents[1] / 'data' / 'synthetic'


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Guard even accidental test changes from invoking an external model.
    def guarded_run(package, **kwargs):
        assert kwargs.get('provider', 'local') == 'local', '测试不得调用付费模型'
        return local_review(package, **kwargs)
    monkeypatch.setattr(api, 'run_review', guarded_run)
    app = api.create_app(tmp_path / 'api-test.sqlite3', seed=False)
    with TestClient(app) as session:
        yield session


def package(number=1):
    value = load_case(DATA / f'seed-{number:02d}.json')
    value['review_scope']['target_labels'] = ['F1', 'F2', 'count', 'material_relation', 'alert_response']
    return value


def imported(client, number=1):
    response = client.post('/api/cases', json={'package':package(number),'reason':'测试导入来源说明'})
    assert response.status_code == 200, response.text
    return response.json()


def run(client, cid, **kwargs):
    response = client.post(f'/api/cases/{cid}/run', json={'mode':'fixed','provider':'local','strategy':'full',**kwargs})
    assert response.status_code == 200, response.text
    return response.json()


def review(client, cid, action, target_id, resolution=None):
    response = client.post(f'/api/cases/{cid}/reviews', json={'action':action,'target_id':target_id,
        'reason':'测试人工复核：核对当前范围和证据','actor':'test-reviewer','resolution':resolution})
    assert response.status_code == 200, response.text
    return response.json()


def close_manual(client, cid):
    state = client.get(f'/api/cases/{cid}').json()
    types = {i['issue_id']:i['type'] for i in state['latest_run']['issues']}
    for item in state['open_items']:
        if types.get(item['item_id']) in {'manual_extraction','manual_focus','manual_material'}:
            review(client,cid,'close_item',item['item_id'],'corresponds' if types[item['item_id']]=='manual_material' else 'addressed')
    for item in client.get(f'/api/cases/{cid}').json()['annotations']:
        action = 'revise' if item['kind'] == 'semantic' else 'reconfirm' if 'reconfirm' in item['allowed_actions'] else 'confirm'
        if action not in item['allowed_actions']:
            continue
        response = client.post(f'/api/cases/{cid}/reviews', json={'action':action, 'target_id':item['annotation_id'],
            'reason':'测试人工逐标签核对当前范围及证据', 'snapshot_id':item['snapshot_id'],
            'expected_event_id':item['review']['event_id'], 'previous_event_id':(item.get('prior_review') or {}).get('event_id'),
            'new_value':'addressed' if item['kind']=='semantic' else None, 'evidence':item['evidence']})
        assert response.status_code == 200, response.text
    return client.get(f'/api/cases/{cid}').json()


def passed(client):
    state = imported(client); cid = state['package']['case_id']
    run(client,cid); assert close_manual(client,cid)['can_pass']
    return review(client,cid,'confirm','task')


def test_http_import_run_review_export_and_assets(client):
    assert client.get('/api/cases').json() == {'cases':[]}
    state = imported(client); cid = state['package']['case_id']
    assert client.get(f'/api/cases/{cid}').json()['source_hash'] == state['source_hash']
    summary = client.get('/api/cases').json()['cases'][0]
    assert summary['coverage_summary'] == 'full' and summary['stale'] is False
    with client.app.state.store.connect() as db:
        assert db.execute('SELECT reason FROM sources').fetchone()[0] == '测试导入来源说明'
    state = run(client,cid)
    assert state['latest_run']['run_status'] == 'partial'
    assert state['latest_run']['agent_verified'] is False
    denied = client.post(f'/api/cases/{cid}/reviews',json={'action':'confirm','target_id':'task','reason':'尚未关闭未决项'})
    assert denied.status_code == 409
    assert close_manual(client,cid)['can_pass']
    confirmed = review(client,cid,'confirm','task')
    assert confirmed['review_status'] == '本次质检范围内通过'
    response = client.get(f'/api/cases/{cid}/export')
    assert response.status_code == 200 and response.headers['content-type'].startswith('application/json')
    assert 'attachment' in response.headers['content-disposition']
    assert response.json()['deliverable']['passed']
    assert response.json()['audit_integrity']['valid']
    html = client.get('/')
    assert html.status_code == 200 and 'text/html' in html.headers['content-type']
    assert html.headers['x-content-type-options'] == 'nosniff'
    assert "frame-ancestors 'none'" in html.headers['content-security-policy']
    assert client.get('/app.js').status_code == 200


def test_source_mutation_revokes_current_pass_and_requires_new_confirmation(client):
    state = passed(client); cid = state['package']['case_id']; old_snapshot = state['latest_run']['snapshot_id']
    updated = deepcopy(state['package']); updated['documents'][0]['text'] = '补充：' + updated['documents'][0]['text']
    response = client.post(f'/api/cases/{cid}/source',json={'package':updated,'reason':'新增经办说明'})
    assert response.status_code == 200
    assert response.json()['stale'] and not response.json()['can_pass']
    assert client.get(f'/api/cases/{cid}/export').json()['deliverable']['passed'] is False
    state = run(client,cid,strategy='incremental')
    assert not state['stale'] and state['latest_run']['snapshot_id'] != old_snapshot
    assert state['latest_run']['stats']['reused'] > 0
    assert not client.get(f'/api/cases/{cid}/export').json()['deliverable']['passed']
    assert close_manual(client,cid)['can_pass']
    assert review(client,cid,'reconfirm','task')['review_status'] == '本次质检范围内通过'


def test_failed_paid_run_replaces_old_pass_and_does_not_expose_exception_content(client, monkeypatch):
    state = passed(client); cid = state['package']['case_id']
    calls=[]
    def failed_model(package, **kwargs):
        calls.append(kwargs)
        raise RuntimeError('SENSITIVE_FAKE_PROVIDER_RESPONSE_MUST_NOT_APPEAR')
    monkeypatch.setattr(api,'run_review',failed_model)
    response = client.post(f'/api/cases/{cid}/run',json={'mode':'agent','provider':'deepseek','strategy':'incremental'})
    assert response.status_code == 200
    failed = response.json()
    assert calls[0]['provider'] == 'deepseek' and calls[0]['mode'] == 'agent'
    assert failed['latest_run']['run_status'] == 'failed'
    assert failed['latest_run']['run_id'] != state['latest_run']['run_id']
    assert not failed['can_pass']
    assert 'SENSITIVE_FAKE' not in response.text
    assert not client.get(f'/api/cases/{cid}/export').json()['deliverable']['passed']
    # The execution lock must also be released after an exception.
    monkeypatch.setattr(api,'run_review',lambda package, **kwargs: local_review(package,provider='local'))
    assert run(client,cid)['latest_run']['run_status'] == 'partial'


def test_source_changed_during_run_does_not_install_outdated_snapshot(client, monkeypatch):
    state = imported(client); cid = state['package']['case_id']
    def changed_during_run(current, **kwargs):
        result = local_review(current,provider='local')
        updated = deepcopy(current); updated['documents'][0]['text'] += ' 执行中补充。'
        client.app.state.store.change_source(cid,updated,'模拟计算期间来源变化')
        return result
    monkeypatch.setattr(api,'run_review',changed_during_run)
    response = client.post(f'/api/cases/{cid}/run',json={'provider':'local'})
    assert response.status_code == 409
    current = client.get(f'/api/cases/{cid}').json()
    assert current['source_hash'] != state['source_hash']
    assert current['latest_run'] is None
    assert not client.get(f'/api/cases/{cid}/export').json()['deliverable']['passed']


def test_concurrent_execution_rejected_without_second_model_call(client, monkeypatch):
    state = imported(client); cid = state['package']['case_id']
    started, release = Event(), Event(); calls=[]
    def blocked_run(package, **kwargs):
        calls.append(kwargs); started.set()
        assert release.wait(5), '测试释放事件未触发'
        return local_review(package,provider='local')
    monkeypatch.setattr(api,'run_review',blocked_run)
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(client.post,f'/api/cases/{cid}/run',json={'provider':'deepseek'})
        assert started.wait(5)
        try:
            second = client.post(f'/api/cases/{cid}/run',json={'provider':'deepseek'})
            assert second.status_code == 409
            assert len(calls) == 1
        finally:
            release.set()
        assert first.result(timeout=5).status_code == 200


@pytest.mark.parametrize('origin',['https://evil.example','http://localhost','https://testserver','null','http://['])
def test_cross_origin_writes_rejected(client, origin):
    response = client.post('/api/cases',json={'package':package()},headers={'Origin':origin})
    assert response.status_code == 403
    assert client.get('/api/cases').json()['cases'] == []


def test_same_origin_json_write_allowed_and_foreign_host_rejected(client):
    response = client.post('/api/cases',json={'package':package()},headers={'Origin':'http://testserver'})
    assert response.status_code == 200
    assert client.get('/api/cases',headers={'Host':'evil.example'}).status_code == 400


def test_json_and_body_size_enforced_even_without_content_length(client):
    assert client.post('/api/cases',content='{}',headers={'Content-Type':'text/plain'}).status_code == 415
    assert client.post('/api/cases',json={},headers={'Content-Length':'5000001'}).status_code == 413
    assert client.post('/api/cases',json={},headers={'Content-Length':'invalid'}).status_code == 400
    assert client.post('/api/cases',json={},headers={'Content-Length':'-1'}).status_code == 400
    response = client.post('/api/cases',content=iter([b'{' + b' ' * 5_000_001 + b'}']),headers={'Content-Type':'application/json'})
    assert response.status_code == 413
    assert client.get('/api/cases').json()['cases'] == []


def test_bad_import_missing_case_and_unsupported_modes_are_explicit(client):
    invalid = package(); invalid['transactions'][0]['direction']='sideways'
    assert client.post('/api/cases',json={'package':invalid}).status_code == 409
    assert client.get('/api/cases/missing').status_code == 404
    state = imported(client); cid = state['package']['case_id']
    assert client.post(f'/api/cases/{cid}/run',json={'mode':'agent','provider':'local'}).status_code == 409
    assert client.post(f'/api/cases/{cid}/run',json={'mode':'unknown'}).status_code == 422
    assert client.post(f'/api/cases/{cid}/source',json={'package':state['package'],'reason':''}).status_code == 409
    assert client.post('/api/cases',json={'package':package()}).status_code == 409


def test_config_exposes_only_key_presence_not_key_material(client, monkeypatch):
    monkeypatch.setattr(api,'settings',lambda:{'DEEPSEEK_API_KEY':'FAKE_SECRET_NEVER_EXPOSE','DEEPSEEK_MODEL':'mock-model'})
    response = client.get('/api/config')
    assert response.status_code == 200
    assert response.json()['deepseek_configured'] is True
    assert response.json()['model'] == 'mock-model'
    assert 'FAKE_SECRET' not in response.text


def test_annotation_http_contract_requires_snapshot_cas_and_rejects_client_old_value(client):
    state=imported(client); cid=state['package']['case_id']; state=run(client,cid)
    annotation=next(a for a in state['annotations'] if a['label']=='F1')
    payload={'action':'confirm','target_id':annotation['annotation_id'],'reason':'逐项核对F1当前计算与证据',
             'snapshot_id':annotation['snapshot_id'],'expected_event_id':None,'evidence':annotation['evidence']}
    forged=client.post(f'/api/cases/{cid}/reviews',json={**payload,'old_value':'met'})
    assert forged.status_code==422
    stale=client.post(f'/api/cases/{cid}/reviews',json={**payload,'snapshot_id':'old-snapshot'})
    assert stale.status_code==409
    success=client.post(f'/api/cases/{cid}/reviews',json=payload)
    assert success.status_code==200
    reviewed=next(a for a in success.json()['annotations'] if a['annotation_id']==annotation['annotation_id'])
    assert reviewed['review']['valid'] and reviewed['review']['final_value']==annotation['candidate_value']
    conflict=client.post(f'/api/cases/{cid}/reviews',json=payload)
    assert conflict.status_code==409
    assert not client.get(f'/api/cases/{cid}/export').json()['deliverable']['annotations']


def test_claim_http_revision_is_recorded_pending_without_replacing_original_fact(client):
    state=imported(client,2); cid=state['package']['case_id']; state=run(client,cid)
    annotation=next(a for a in state['annotations'] if a['label']=='count')
    response=client.post(f'/api/cases/{cid}/reviews',json={'action':'revise','target_id':annotation['annotation_id'],
        'reason':'测试修订命题不能自动消除原文一次和流水三次矛盾','snapshot_id':annotation['snapshot_id'],
        'expected_event_id':None,'claim_patch':{'value':3},'evidence':annotation['evidence']})
    assert response.status_code==200,response.text
    state=response.json(); reviewed=next(a for a in state['annotations'] if a['label']=='count')
    assert reviewed['candidate_value']=='contradicted' and reviewed['review']['final_value'] is None
    assert reviewed['review']['proposed_value']=='supported' and not reviewed['review']['valid']
    assert not state['can_pass'] and state['audit_integrity']['valid']


def test_migration_http_preview_commit_refuses_forged_target_and_does_not_run_model(client,monkeypatch):
    state=imported(client); cid=state['package']['case_id']; state=run(client,cid)
    catalogue=client.get('/api/schemas').json()
    base=next(s for s in catalogue['schemas'] if cid in s['case_ids'])
    target=deepcopy(base['schema']); target['schema_version']='S1.1-http-migration'
    target['features']['F1']['minimum_days']=1
    body={'base_schema_hash':base['schema_hash'],'target_schema':target,'selected_case_ids':[cid],
          'actor':'http-reviewer','reason':'测试HTTP规范迁移预览'}
    response=client.post('/api/migrations/preview',json=body)
    assert response.status_code==200,response.text
    plan=response.json(); url=f"/api/migrations/{plan['preview_id']}/cases/{cid}"
    apply={'preview_hash':plan['preview_hash'],'actor':'http-reviewer','reason':'逐案显式执行已审核预览'}
    assert client.post(url,json={**apply,'target_schema':target}).status_code==422
    assert client.post(url,json={**apply,'preview_hash':'forged'}).status_code==409
    direct=deepcopy(state['package']); direct.update(schema=target,schema_version=target['schema_version'])
    assert client.post(f'/api/cases/{cid}/source',json={'package':direct,'reason':'尝试绕过迁移'}).status_code==409
    def never_execute(*args,**kwargs):
        pytest.fail('规范迁移不得运行模型或重查')
    monkeypatch.setattr(api,'run_review',never_execute)
    response=client.post(url,json=apply)
    assert response.status_code==200,response.text
    result=response.json()
    assert result['receipt']['run_started'] is False and result['case']['stale']
    assert result['case']['latest_run']['run_id']==state['latest_run']['run_id']
    current=client.get(f"/api/migrations/{plan['preview_id']}").json()
    assert current['preview_hash']==plan['preview_hash'] and len(current['receipts'])==1
    assert current['cases'][0]['status']=='applied' and not current['cases'][0]['can_apply']
    assert client.post(url,json=apply).status_code==409
    exported=client.get(f'/api/cases/{cid}/export').json()
    assert not exported['deliverable']['passed'] and len(exported['schema_migrations']['receipts'])==1
