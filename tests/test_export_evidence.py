"""Delivery projection and source-reference checks; no human quality claims."""
from copy import deepcopy
import json

import pytest

from aml_qc import core
from aml_qc.annotations import validate_evidence
from aml_qc.exports import export_bundle
from aml_qc.store import Store
from test_annotations import BASE_LABELS, start, item, decide, close_extraction, adjudicate_produced


def test_delivered_human_label_uses_its_evidence_and_keeps_candidate_provenance(tmp_path):
    store = Store(tmp_path/'evidence.sqlite3')
    state = adjudicate_produced(store, close_extraction(store, start(store, labels=BASE_LABELS)))
    annotation = item(state, 'alert_response')
    original = deepcopy(annotation)
    doc = next(d for d in state['package']['documents'] if d['document_id'] == 'narrative')
    quote = doc['text'].split('。')[0]
    proof = [{'type':'document_span', 'document_id':'narrative', 'revision':doc['revision'],
              'span':[0,len(quote)], 'text':quote}]
    assert proof != annotation['evidence']
    state = decide(store, state, annotation, 'revise', new_value='addressed', evidence=proof,
                   reason='机制测试：人工明确选择当前原文片段')
    cid = state['package']['case_id']
    store.review(cid, action='confirm', target_id='task', reason='机制测试：最终确认限定范围')
    exported = store.export(cid)
    delivered = next(a for a in exported['deliverable']['annotations'] if a['kind'] == 'semantic')
    assert delivered['evidence'] == delivered['review']['evidence'] == proof
    assert delivered['origin'] == 'human_judgement'
    assert delivered['reason'] == delivered['review']['reason']
    assert delivered['candidate_evidence'] == original['evidence']
    assert delivered['candidate_origin'] == original['origin']
    assert next(a for a in exported['annotations'] if a['kind']=='semantic')['evidence'] == original['evidence']
    export_bundle(exported, tmp_path/'out')
    rows = [json.loads(line) for line in (tmp_path/'out/deliverable.jsonl').read_text().splitlines()]
    assert next(a for a in rows if a['kind']=='semantic')['evidence'] == proof


def test_delivered_span_repair_projects_the_reverified_claim(tmp_path):
    store = Store(tmp_path/'span.sqlite3')
    def corrupt(result, package):
        claim = result['claims'][0]; claim['source']['span'] = [0,1]
        result['claim_results'][0] = core.verify_claim(package, claim)
        target = 'claim:'+claim['claim_id']
        next(c for c in result['required_checks'] if c['check_id']==target)['status']='extraction_failed'
        result['open_items'].append({'item_id':'bad-span','kind':'claim_unresolved','target_id':target,'title':'跨度错误'})
    state = close_extraction(store, start(store, labels=BASE_LABELS, edit_result=corrupt))
    annotation = item(state, 'count'); original = deepcopy(annotation)
    doc = next(d for d in state['package']['documents'] if d['document_id']=='narrative')
    proof = [{'type':'document_span','document_id':'narrative','revision':doc['revision'],'span':[0,len(doc['text'])]}]
    state = decide(store,state,annotation,'revise',claim_patch={'quote':annotation['claim']['text']},evidence=proof)
    for a in list(state['annotations']):
        if a['label']=='count':continue
        state = decide(store,state,a,'revise',new_value='addressed') if a['kind']=='semantic' else decide(store,state,a)
    cid = state['package']['case_id']; store.review(cid,action='confirm',target_id='task',reason='机制测试：修复引用后完成')
    delivered = next(a for a in store.export(cid)['deliverable']['annotations'] if a['kind']=='claim')
    assert core.validate_span(state['package'],delivered['claim']['source'],delivered['claim']['text'])
    assert delivered['claim'] == delivered['review']['claim']
    assert delivered['candidate_claim'] == original['claim']
    assert delivered['execution_status'] == 'completed'
    assert delivered['evidence'] == delivered['review']['evidence']


@pytest.mark.parametrize('kind', ['query_scope','material_set','material_link','transaction_set','alert_focus','material','transactions'])
def test_known_reference_does_not_make_incorrect_source_metadata_valid(tmp_path,kind):
    state=start(Store(tmp_path/'refs.sqlite3'),labels=BASE_LABELS)
    refs=[r for a in state['annotations'] for r in a['evidence']]
    ref=deepcopy(next(r for r in refs if r['type']==kind))
    if kind=='query_scope':ref['scope']['transaction_set_hash']='wrong'
    elif kind=='alert_focus':ref['revision']='wrong'
    elif kind=='transactions':ref['transaction_set_version']='sha256:wrong'
    else:ref['content_hash']='wrong'
    with pytest.raises(ValueError,match='不能解析'):
        validate_evidence(state['package'],[ref],[ref])


@pytest.mark.parametrize('change',['ids','coverage','account','fields'])
def test_known_query_is_replayed_with_its_scope_and_coverage(tmp_path,change):
    state=start(Store(tmp_path/'query.sqlite3'),labels=BASE_LABELS)
    ref=deepcopy(next(r for a in state['annotations'] for r in a['evidence'] if r['type']=='query_scope' and r['transaction_ids']))
    if change=='ids':ref['transaction_ids']=[]
    elif change=='coverage':ref['coverage']['status']='unknown'
    elif change=='account':ref['scope']['account_id']='another-account'
    else:ref['scope']['fields']=[]
    with pytest.raises(ValueError,match='不能解析'):
        validate_evidence(state['package'],[ref],[ref])


def test_valid_empty_query_keeps_proof_of_its_absence_scope(tmp_path):
    state=start(Store(tmp_path/'empty.sqlite3'),labels=BASE_LABELS)
    package=state['package']; package['transactions']=[]
    q=core.query_transactions(package,{'direction':'out'})
    ref={'type':'query_scope','scope':q['scope'],'coverage':q['coverage'],'transaction_ids':q['transaction_ids']}
    assert validate_evidence(package,[ref],[ref]) == [ref]


def test_export_rechecks_final_evidence_before_delivery(tmp_path, monkeypatch):
    store = Store(tmp_path/'export-gate.sqlite3')
    state = adjudicate_produced(store, close_extraction(store, start(store, labels=BASE_LABELS)))
    cid = state['package']['case_id']
    state = store.review(cid, action='confirm', target_id='task', reason='机制测试：最终确认')
    assert state['review_status'] == '本次质检范围内通过'
    ref = next(r for a in state['annotations'] for r in a['review']['evidence'] if r['type']=='query_scope')
    ref['scope']['transaction_set_hash'] = 'wrong'
    # Simulate an inconsistent assembled export state, without altering its stored audit history.
    monkeypatch.setattr(store, 'get', lambda _: deepcopy(state))
    with pytest.raises(ValueError, match='不能解析'):
        store.export(cid)
