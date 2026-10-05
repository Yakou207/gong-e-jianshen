"""Read-only rule/feature tool mechanics; scripted models are not Agent acceptance."""
from copy import deepcopy
import json

import pytest

from aml_qc import llm, workflow
from aml_qc.depgraph import Evaluator, business_result, digest, sources_for
from aml_qc.llm import FrozenModel
from test_model_safety import MODEL, ScriptedModel, case, json_message, response_message, semantic_response, tool_message


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Tool mechanism tests must not read credentials or call live models')
    monkeypatch.setattr(llm, 'settings', forbidden)
    monkeypatch.setattr(llm.httpx, 'post', forbidden)


def evaluator(value, schema, previous=None, strategy='full'):
    return Evaluator(sources_for(value, schema, {'test': 'feature-tools'}), previous, strategy)


def test_read_schema_returns_the_bound_case_version_without_case_fact_basis():
    value=case(); schema=deepcopy(workflow.default_schema())
    schema['schema_version']='S1.1'; schema['features']['F2']['minimum_in_counterparties']=13
    engine=evaluator(value,schema)
    result=workflow.execute_tool(value,schema,engine,'read_schema',{})
    assert result['schema']==schema and result['schema_hash']==digest(schema)
    assert result['evidence_role']=='definition_only'
    result['schema']['features']['F2']['minimum_in_counterparties']=999
    assert schema['features']['F2']['minimum_in_counterparties']==13


@pytest.mark.parametrize('partial',[False,True])
def test_feature_tool_returns_actual_windows_and_coverage(partial):
    value=case();schema=workflow.default_schema()
    if partial:value['coverage'][0]['status']='partial'
    result=workflow.execute_tool(value,schema,evaluator(value,schema),'compute_features',{'feature_code':'F2'})
    feature,=result['features']
    assert feature['feature_code']=='F2'
    assert feature['result']==('undeterminable' if partial else 'met')
    assert feature['windows'][0]['metrics']['in_counterparty_count']==12
    assert feature['windows'][0]['metrics']['out_amount_cents']==105000
    assert feature['windows'][0]['coverage']['status']==('partial' if partial else 'full')
    assert feature['evidence'][0]['scope']['account_id']==value['subject_account_id']
    assert result['schema_hash']==digest(schema)


@pytest.mark.parametrize('args',[{'feature_code':'F3'},{'feature_code':None},{'account_id':'other-account'}, {'start':'2020-01-01'}])
def test_feature_tool_rejects_unknown_codes_and_scope_overrides(args):
    value=case();schema=workflow.default_schema();engine=evaluator(value,schema)
    with pytest.raises(ValueError):workflow.execute_tool(value,schema,engine,'compute_features',args)
    assert engine.nodes=={}


def test_default_feature_read_and_unchanged_read_reuse_preserve_evidence():
    value=case();schema=workflow.default_schema();old=evaluator(value,schema)
    result=workflow.execute_tool(value,schema,old,'compute_features',{})
    assert [f['feature_code'] for f in result['features']]==['F1','F2']
    inc=evaluator(value,schema,deepcopy(old.snapshot()),'incremental')
    assert workflow.execute_tool(value,schema,inc,'compute_features',{})==result
    assert inc.reused==1 and inc.recomputed==0


@pytest.mark.parametrize('mutation,expected',[('schema','not_met'),('transactions','not_met'),('coverage','undeterminable'),('period','undeterminable')])
def test_feature_read_invalidates_on_each_computation_input(mutation,expected):
    value=case();schema=workflow.default_schema();old=evaluator(value,schema)
    args={'feature_code':'F2'}
    before=workflow.execute_tool(value,schema,old,'compute_features',args)
    snapshot=deepcopy(old.snapshot());original=deepcopy(snapshot)
    new=deepcopy(value);updated=deepcopy(schema)
    if mutation=='schema':updated['features']['F2']['minimum_in_counterparties']=13
    elif mutation=='transactions':new['transactions']=[r for r in new['transactions'] if r['direction']!='out']
    elif mutation=='coverage':new['coverage'][0]['status']='partial'
    else:new['coverage_end']='2026-09-04T00:00:00+08:00'
    inc=evaluator(new,updated,snapshot,'incremental');full=evaluator(new,updated)
    actual=workflow.execute_tool(new,updated,inc,'compute_features',args)
    independent=workflow.execute_tool(new,updated,full,'compute_features',args)
    assert actual==independent and actual['features'][0]['result']==expected
    assert before['features'][0]['result']=='met'
    assert inc.recomputed==1 and inc.reused==full.reused==0
    assert snapshot==original


def model_with_reads(value, *, basis_tool='compute_features'):
    response=semantic_response(value)
    args={'feature_code':'F2'} if basis_tool=='compute_features' else {}
    response['leads']=[{'observation':'机制夹具观察，业务意义未验证。','question':'是否需要另行核查这一观察？',
                       'basis_refs':['tool:'+basis_tool+':'+digest(args)[:20]]}]
    return ScriptedModel([json_message({'claims':[],'unresolved':[]}),response_message(value),
        tool_message('read_schema',{},'schema-read'),tool_message('compute_features',{'feature_code':'F2'},'feature-read'),{'role':'assistant','content':'取证结束。'},json_message({'gaps':response['gaps'],'leads':response['leads']})])


def test_schema_read_is_not_case_evidence_but_feature_read_can_bind_a_candidate():
    value=case();model=model_with_reads(value)
    result=workflow.run_review(value,provider='frozen',mode='agent',model=model)
    lead,=result['lead_candidates']
    assert lead['basis_refs']==['tool:compute_features:'+digest({'feature_code':'F2'})[:20]]
    assert any(ref['type']=='query_scope' for ref in lead['evidence'])
    messages=model.calls[-1]['request']['messages']
    schema_return=json.loads(next(m['content'] for m in messages if m.get('tool_call_id')=='schema-read'))
    assert schema_return['lead_basis_ref'] is None
    assert schema_return['schema_hash']==digest(workflow.default_schema())
    for t in result['trace']:
        if 'round' in t:
            node=result['snapshot']['nodes'][t['result_ref']]
            assert node['result']==t['result']
            assert set(node['dependencies']) <= set(result['snapshot']['nodes']['agent_stage']['dependencies'])
    rejected=workflow.run_review(value,provider='frozen',mode='agent',model=model_with_reads(value,basis_tool='read_schema'))
    assert not rejected['lead_candidates'] and rejected['run_status']=='partial'


def test_changed_schema_reruns_agent_reads_and_matches_independent_full():
    old=case();new=deepcopy(old);new['schema']=deepcopy(workflow.default_schema())
    new['schema']['features']['F2']['minimum_in_counterparties']=13
    products={}
    for value in (old,new):
        recorder=model_with_reads(value)
        workflow.run_review(value,provider='frozen',mode='agent',model=recorder)
        products.update(recorder.request_products)
    before=workflow.run_review(old,provider='frozen',mode='agent',model=FrozenModel(products,MODEL))
    snapshot=deepcopy(before['snapshot'])
    inc=workflow.run_review(new,provider='frozen',mode='agent',strategy='incremental',previous=snapshot,model=FrozenModel(products,MODEL))
    full=workflow.run_review(new,provider='frozen',mode='agent',model=FrozenModel(products,MODEL))
    assert business_result(inc)==business_result(full) and full['stats']['reused']==0
    assert inc['stats']['model_calls']==6 and inc['stats']['replayed_tool_calls']==0
    assert before['snapshot']==snapshot
    feature_read=next(t for t in inc['trace'] if 'round' in t and t['tool']=='compute_features')
    assert feature_read['result']['features'][0]['result']=='not_met'
