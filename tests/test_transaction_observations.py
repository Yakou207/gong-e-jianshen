"""Actual-read summaries and ID-scoped references, not model quality claims."""
from copy import deepcopy
from datetime import datetime, timedelta
import json

import pytest

from aml_qc import core, workflow
from aml_qc.annotations import validate_evidence
from aml_qc.baseline import raw_case_input
from test_model_safety import ScriptedModel, case, json_message, response_message, support_response, tool_message


def run_model(value, mode, queries=()):
    messages = [json_message({'claims': [], 'unresolved': []}), response_message(value)]
    if mode == 'agent':
        messages += [tool_message('query_transactions', args, f'read-{i}') for i, args in enumerate(queries)]
        messages.append({'role': 'assistant', 'content': '取证结束。'})
    messages.append(json_message(support_response(value)))
    model = ScriptedModel(messages)
    result = workflow.run_review(value, mode=mode, provider='frozen', model=model)
    return result, model


def test_fixed_summary_uses_actual_rows_not_material_counterparty():
    value = case()
    outgoing = next(t for t in value['transactions'] if t['direction'] == 'out')
    outgoing['counterparty_token'] = 'actual-other-account'
    result, model = run_model(value, 'fixed')
    summary = json.loads(model.calls[-1]['request']['messages'][1]['content'])['transaction_observations']
    observed = next(row for row in summary[0]['rows'] if row['transaction_id'] == outgoing['transaction_id'])
    assert observed['counterparty_token'] == 'actual-other-account'
    assert observed['counterparty_token'] != value['materials'][0]['counterparty']['counterparty_token']
    assert summary[0]['result_ref'] == result['fixed_policy_reads'][1]['result_ref']
    assert summary[0]['scope'] == result['fixed_policy_reads'][1]['result']['scope']


def test_agent_summary_does_not_include_unread_other_counterparties():
    value = case()
    result, model = run_model(value, 'agent', [{'counterparty_token': 'supplier-b'}])
    message = model.calls[-1]['request']['messages'][-2]
    summary = json.loads(message['content'])['transaction_observations']
    assert len(summary) == 1
    assert summary[0]['rows']
    assert {r['counterparty_token'] for r in summary[0]['rows']} == {'supplier-b'}
    assert len(summary[0]['rows']) < len(value['transactions'])
    assert summary[0]['scope']['counterparty_token'] == 'supplier-b'
    actual = next(t for t in result['trace'] if 'round' in t)['result']
    assert summary[0]['coverage'] == actual['coverage']


def test_empty_filtered_read_then_id_lookup_retains_both_distinct_scopes():
    value = case()
    outgoing = next(t for t in value['transactions'] if t['direction'] == 'out')
    outgoing['counterparty_token'] = 'actual-other-account'
    queries = [{'direction': 'out', 'counterparty_token': 'supplier-b'},
               {'transaction_id': outgoing['transaction_id']}]
    _, model = run_model(value, 'agent', queries)
    summary = json.loads(model.calls[-1]['request']['messages'][-2]['content'])['transaction_observations']
    assert summary[0]['rows'] == []
    assert summary[0]['scope']['counterparty_token'] == 'supplier-b'
    assert summary[1]['scope']['transaction_id'] == outgoing['transaction_id']
    assert summary[1]['rows'][0]['counterparty_token'] == 'actual-other-account'
    assert summary[0]['result_ref'] != summary[1]['result_ref']


def test_failed_query_is_not_promoted_into_successful_observations():
    value = case()
    result, model = run_model(value, 'agent', [{'direction': 'invalid'}])
    summary = json.loads(model.calls[-1]['request']['messages'][-2]['content'])['transaction_observations']
    assert summary == []
    assert result['stats']['adaptive_tools_failed'] == 1
    tool = next(m for m in model.calls[-1]['request']['messages'] if m['role'] == 'tool')
    assert json.loads(tool['content'])['execution_status'] == 'failed'


def test_direct_baseline_preserves_supplied_material_transaction_identifiers():
    value = case()
    material = value['materials'][0]
    material.update(order_id='visible-order', original_payment_transaction_id='visible-in',
                    refund_transaction_id='visible-out', hidden_truth='must-not-be-projected')
    projected = raw_case_input(value)['materials'][0]
    for key in ('order_id', 'original_payment_transaction_id', 'refund_transaction_id'):
        assert projected[key] == material[key]
    assert 'hidden_truth' not in projected


@pytest.mark.parametrize('present', [True, False])
def test_id_scoped_reference_is_replayed_without_expanding_query(present):
    value = case()
    tid = value['transactions'][0]['transaction_id'] if present else 'absent-id'
    query = core.query_transactions(value, {'transaction_id': tid})
    reference = {'type': 'query_scope', **{k: query[k] for k in ('scope', 'coverage', 'transaction_ids')}}
    assert validate_evidence(value, [reference], [reference]) == [reference]
    changed = deepcopy(reference)
    changed['scope']['transaction_id'] = 'different-id' if present else value['transactions'][0]['transaction_id']
    with pytest.raises(ValueError, match='不能解析'):
        validate_evidence(value, [changed], [changed])


def test_gap_catalog_does_not_treat_material_ids_as_deterministic_links():
    value = case()
    value['material_links'] = []
    data = workflow.semantic_input(value, {'claim_results': [], 'material_results': []})
    assert value['materials']
    assert data['gap_basis_catalog'] == {'identity_unresolved': [], 'missing_linked_material': [],
        'material_mismatch': [], 'explanation_support': ['narrative']}


def test_gap_catalog_distinguishes_missing_revision_and_known_mismatch():
    value = case()
    value['material_links'] = [
        {'link_id': 'missing-version', 'material_id': value['materials'][0]['material_id'], 'material_revision': 'absent'},
        {'link_id': 'fields-conflict', 'material_id': value['materials'][0]['material_id'],
         'material_revision': value['materials'][0]['revision']}]
    checks = {'claim_results': [{'claim_id': 'not-resolved', 'execution_status': 'identity_unresolved'},
                               {'claim_id': 'resolved', 'execution_status': 'completed'}],
        'material_results': [{'link_id': 'missing-version', 'result': 'undeterminable'},
                             {'link_id': 'fields-conflict', 'result': 'mismatch'}]}
    data = workflow.semantic_input(value, checks)
    assert data['gap_basis_catalog'] == {'identity_unresolved': ['not-resolved'],
        'missing_linked_material': ['missing-version'], 'material_mismatch': ['fields-conflict'],
        'explanation_support': ['narrative']}


def actual_query_trace(value,args,ref='actual-query'):
    return {'tool':'query_transactions','status':'completed','dispatch_status':'sent','result_ref':ref,
            'result':core.query_transactions(value,args)}


@pytest.mark.parametrize('scope_kind,expected',[('id',()),('counterparty',()),('out',('out',)),
                                              ('in',('in',)),('both',('in','out')),('subscope',()),('expanded',('in','out'))])
def test_read_progress_counts_only_full_visible_unfiltered_successful_query_scopes(scope_kind,expected):
    value=case();args={}
    if scope_kind=='id':args={'transaction_id':value['transactions'][0]['transaction_id']}
    elif scope_kind=='counterparty':args={'counterparty_token':'supplier-b'}
    elif scope_kind in {'in','out'}:args={'direction':scope_kind}
    elif scope_kind=='subscope':args={'start':(datetime.fromisoformat(value['coverage_start'])+timedelta(days=1)).isoformat()}
    elif scope_kind=='expanded':
        args={'start':(datetime.fromisoformat(value['coverage_start'])-timedelta(days=1)).isoformat(),
              'end':(datetime.fromisoformat(value['coverage_end'])+timedelta(days=1)).isoformat()}
    trace=actual_query_trace(value,args)
    progress=workflow.read_scope_progress(value,[trace])
    for direction in ['in','out']:
        assert progress[direction]=={'whole_visible_interval_read':direction in expected,
                                     'successful_result_refs':['actual-query'] if direction in expected else []}
    assert '来源' in progress['meaning'] and '材料' in progress['meaning']


@pytest.mark.parametrize('failure',['trace_failed','execution_failed','rejected','other_tool'])
def test_read_progress_excludes_failed_rejected_and_non_query_results(failure):
    value=case();trace=actual_query_trace(value,{})
    if failure=='trace_failed':trace['status']='failed'
    elif failure=='execution_failed':trace['result']['execution_status']='failed'
    elif failure=='rejected':trace['dispatch_status']='not_sent'
    else:trace['tool']='compute_features'
    progress=workflow.read_scope_progress(value,[trace])
    assert all(progress[direction]=={'whole_visible_interval_read':False,'successful_result_refs':[]} for direction in ['in','out'])


def test_read_progress_deduplicates_qualifying_refs_without_copying_unread_fields_or_source_completeness():
    value=case();value['coverage'][0]['status']='partial'
    trace=actual_query_trace(value,{'direction':'out'})
    original=deepcopy(trace)
    projected_case={'coverage_start':value['coverage_start'],'coverage_end':value['coverage_end'],
                    'transactions':object(),'unread_transaction_id':'UNREAD-ID','unread_counterparty':'UNREAD-PARTY'}
    progress=workflow.read_scope_progress(projected_case,[trace,trace])
    assert progress['out']=={'whole_visible_interval_read':True,'successful_result_refs':['actual-query']}
    assert progress['in']=={'whole_visible_interval_read':False,'successful_result_refs':[]}
    assert trace==original and trace['result']['coverage']['status']=='partial'
    encoded=json.dumps(progress)
    assert not any(field in encoded for field in ['UNREAD-ID','UNREAD-PARTY','transaction_id','counterparty_token','amount','timestamp'])


def test_fixed_read_progress_is_full_both_directions_even_with_partial_source_coverage():
    value=case();value['coverage'][0]['status']='partial'
    result,model=run_model(value,'fixed')
    data=json.loads(model.calls[-1]['request']['messages'][1]['content'])
    ref=result['fixed_policy_reads'][1]['result_ref']
    assert all(data['read_scope_progress'][direction]=={'whole_visible_interval_read':True,'successful_result_refs':[ref]}
               for direction in ['in','out'])
    assert data['transaction_observations'][0]['coverage']['status']=='partial'


def test_next_agent_planner_receives_only_actual_scope_progress_and_selects_its_own_next_read():
    value=case();tid=value['transactions'][0]['transaction_id']
    class DependingModel(ScriptedModel):
        def complete(self,messages,tools=None,stage=None):
            if len(self.calls) in {3,4}:
                latest=json.loads(messages[-1]['content'])
                progress=latest['read_scope_progress']
                assert progress['in']['whole_visible_interval_read'] is False
                assert progress['out']['whole_visible_interval_read'] is (len(self.calls)==4)
                assert set(latest)=={'read_scope_progress','material_transaction_read_progress'}
                assert 'transaction_id' not in json.dumps(latest['read_scope_progress'])
                assert latest['material_transaction_read_progress']['entries']==[]
            return super().complete(messages,tools,stage)
    model=DependingModel([json_message({'claims':[],'unresolved':[]}),response_message(value),
                          tool_message('query_transactions',{'transaction_id':tid},'id-only'),
                          tool_message('query_transactions',{'direction':'out'},'whole-out'),
                          {'role':'assistant','content':'结束取证。'},json_message(support_response(value))])
    result=workflow.run_review(value,mode='agent',provider='frozen',model=model)
    final=json.loads(model.calls[-1]['request']['messages'][-2]['content'])
    assert set(final)=={'transaction_observations','read_scope_progress','material_transaction_read_progress'}
    assert len(final['transaction_observations'])==2
    assert final['read_scope_progress']['out']['successful_result_refs']==[final['transaction_observations'][1]['result_ref']]
    assert final['read_scope_progress']['in']['whole_visible_interval_read'] is False
    assert result['stats']['adaptive_tool_calls']==2 and len(model.calls)==6 and not model.responses
