"""Tool-enabled semantic entry, with final validation and finite budgets intact."""
import json

import pytest

from aml_qc import llm,workflow
from aml_qc.depgraph import digest
from test_model_safety import ScriptedModel,case,json_message,response_message,semantic_response,support_response,tool_message


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    def forbidden(*args,**kwargs):
        pytest.fail('Agent entry tests must not read credentials or call a live model')
    monkeypatch.setattr(llm,'settings',forbidden)
    monkeypatch.setattr(llm.httpx,'post',forbidden)


def test_agent_can_read_evidence_before_its_first_semantic_answer():
    value=case()
    model=ScriptedModel([json_message({'claims':[],'unresolved':[]}),response_message(value),tool_message('read_schema'),
                         {'role':'assistant','content':'取证结束。'},
                         json_message(support_response(value))])
    result=workflow.run_review(value,mode='agent',provider='frozen',model=model)
    assert model.calls[2]['request']['tools']
    context=json.loads(model.calls[2]['request']['messages'][1]['content'])
    assert 'initial_candidates' not in context
    assert context['case']['checks']['features']
    assert result['stats']['model_calls']==5
    assert result['stats']['adaptive_tools_completed']==1
    assert result['semantic_results'] and not model.responses


@pytest.mark.parametrize('draft', ['plain text', '', 'valid_json'])
def test_agent_can_finish_without_unnecessary_tools(draft):
    value=case()
    planning=json_message(semantic_response(value)) if draft=='valid_json' else {'role':'assistant','content':draft}
    model=ScriptedModel([json_message({'claims':[],'unresolved':[]}),response_message(value),planning,json_message(support_response(value))])
    result=workflow.run_review(value,mode='agent',provider='frozen',model=model)
    assert len(model.calls)==4 and model.calls[2]['request']['tools']
    assert model.calls[-1]['request']['tools'] is None and not model.responses
    assert result['stats']['adaptive_tool_calls']==0 and result['semantic_results']


def test_three_tool_rounds_have_one_bounded_final_answer():
    value=case()
    model=ScriptedModel([json_message({'claims':[],'unresolved':[]}),response_message(value),tool_message('read_schema',{},'rules'),
        tool_message('compute_features',{'feature_code':'F2'},'features'),
        tool_message('query_transactions',{'direction':'out'},'query'),json_message(support_response(value))])
    result=workflow.run_review(value,mode='agent',provider='frozen',model=model)
    assert result['adaptive_rounds']==3 and result['stats']['model_calls']==6
    assert model.calls[-1]['request']['tools'] is None
    assert result['semantic_results'] and not model.responses
    assert result['execution']['max_model_calls']==6


def test_invalid_semantic_final_is_not_repaired_or_promoted_after_reading():
    value=case();value['material_links'][0]['schema_version']='S9'
    answer=semantic_response(value)
    quote=next(d['text'] for d in value['documents'] if d['document_id']=='narrative')
    answer['gaps']=[{'basis_kind':'material_mismatch','basis_ref':'purchase-link','quote':quote,
                    'requested_material':'夹具建议','reason':'错误地把待判断当成不匹配'}]
    model=ScriptedModel([json_message({'claims':[],'unresolved':[]}),response_message(value),tool_message('read_schema'),json_message(semantic_response(value)),json_message({'gaps':answer['gaps'],'leads':answer.get('leads',[])})])
    result=workflow.run_review(value,mode='agent',provider='frozen',model=model)
    assert result['material_results'][0]['result']=='pending_judgement'
    assert result['stats']['adaptive_tools_completed']==1 and len(model.calls)==5
    assert result['run_status']=='partial' and result['semantic_results']
    assert any(c['check_id']=='semantic' and c['status']=='failed' for c in result['required_checks'])
    assert not model.responses


@pytest.mark.parametrize('draft', ['STALE-NO-TOOL-SUGGESTION', 'wrong_json'])
def test_final_review_excludes_unvalidated_no_tool_draft(draft):
    value=case()
    wrong=semantic_response(value)
    wrong['gaps']=[{'basis_kind':'explanation_support','basis_ref':'narrative',
                   'quote':wrong['focuses'][0]['quote'],'requested_material':'STALE-NO-TOOL-SUGGESTION',
                   'reason':'Unvalidated fixture suggestion'}]
    planning=json_message(wrong) if draft=='wrong_json' else {'role':'assistant','content':draft}
    model=ScriptedModel([json_message({'claims':[],'unresolved':[]}),response_message(value),planning,
                         json_message(support_response(value))])
    result=workflow.run_review(value,mode='agent',provider='frozen',model=model)
    final=model.calls[-1]['request']
    assert final['messages'][0]=={'role':'system','content':workflow.SEMANTIC_SYSTEM}
    assert final['messages'][1]==model.calls[2]['request']['messages'][1]
    assert [m['role'] for m in final['messages']]==['system','user','user','user']
    assert json.loads(final['messages'][-2]['content'])['transaction_observations']==[]
    assert 'STALE-NO-TOOL-SUGGESTION' not in json.dumps(final,ensure_ascii=False)
    assert final['tools'] is None and len(model.calls)==4 and not model.responses
    assert result['model_requests']==model.calls and result['semantic_results']


@pytest.mark.parametrize('failed_tool', [False, True])
def test_final_review_preserves_tool_requests_and_results_without_assistant_drafts(failed_tool):
    value=case()
    first=tool_message('read_document',{'document_id':'narrative'},'narrative-read')
    first['tool_calls']+=tool_message('check_coverage',{},'coverage-read')['tool_calls']
    first['content']='STALE-FIRST-SUGGESTION'
    second=tool_message('read_material',{'material_id':'purchase-contract'},'material-read')
    second['content']='STALE-SECOND-SUGGESTION'
    if failed_tool:
        second['tool_calls']+=tool_message('not_a_tool',{},'failed-read')['tool_calls']
    wrong=semantic_response(value)
    wrong['gaps']=[{'basis_kind':'explanation_support','basis_ref':'narrative',
                   'quote':wrong['focuses'][0]['quote'],'requested_material':'STALE-FINAL-SUGGESTION',
                   'reason':'Unvalidated fixture suggestion'}]
    model=ScriptedModel([json_message({'claims':[],'unresolved':[]}),response_message(value),first,second,
                         json_message(wrong),json_message(support_response(value))])
    result=workflow.run_review(value,mode='agent',provider='frozen',model=model)
    evidence_request=model.calls[-2]['request']
    final=model.calls[-1]['request']
    expected=[]
    for message in evidence_request['messages'][2:]:
        expected.append({**message,'content':None} if message['role']=='assistant' else message)
    assert final['messages'][0]=={'role':'system','content':workflow.SEMANTIC_SYSTEM}
    assert final['messages'][1]==model.calls[2]['request']['messages'][1]
    assert final['messages'][2:-2]==expected
    assert json.loads(final['messages'][-2]['content'])['transaction_observations']==[]
    assert [m.get('tool_call_id') for m in final['messages'] if m['role']=='tool']==[
        'narrative-read','coverage-read','material-read']+(['failed-read'] if failed_tool else [])
    if failed_tool:
        failed=next(m for m in final['messages'] if m.get('tool_call_id')=='failed-read')
        assert json.loads(failed['content'])['execution_status']=='failed'
    assert 'STALE-' not in json.dumps(final,ensure_ascii=False)
    assert 'STALE-FIRST-SUGGESTION' in json.dumps(evidence_request,ensure_ascii=False)
    assert 'STALE-SECOND-SUGGESTION' in json.dumps(evidence_request,ensure_ascii=False)
    assert final['tools'] is None and len(model.calls)==6 and not model.responses
    assert result['model_requests']==model.calls and result['semantic_results']
    assert result['stats']['adaptive_tools_completed']==3 and result['adaptive_rounds']==2


def combined_tools(*items):
    return {'role':'assistant','tool_calls':[
        tool_message(name,args,call_id)['tool_calls'][0] for name,args,call_id in items]}


@pytest.mark.parametrize('count',[3,4])
def test_oversized_round_executes_first_two_and_returns_every_remaining_budget_rejection(monkeypatch,count):
    value=case()
    items=[('read_document',{'document_id':'narrative'},'first'),('check_coverage',{},'second'),
           ('query_transactions',{'direction':'out'},'third'),('query_transactions',{'direction':'in'},'fourth')][:count]
    planning=combined_tools(*items)
    dispatched=[];actual=workflow.execute_tool
    def execute(*args):
        dispatched.append((args[3],args[4]))
        return actual(*args)
    monkeypatch.setattr(workflow,'execute_tool',execute)
    model=ScriptedModel([json_message({'claims':[],'unresolved':[]}),response_message(value),planning,
                         {'role':'assistant','content':'结束取证。'},json_message(support_response(value))])
    result=workflow.run_review(value,mode='agent',provider='frozen',model=model)
    assert dispatched==[(name,args) for name,args,_ in items[:2]]
    trace=[row for row in result['trace'] if 'round' in row]
    assert [row['tool_call_id'] for row in trace]==[call_id for _,_,call_id in items]
    assert [row['dispatch_status'] for row in trace]==['sent','sent']+['not_sent']*(count-2)
    assert all(row['status']=='failed' and row['result_ref'] is None and row['result']['execution_status']=='failed'
               and '预算' in row['result']['error'] for row in trace[2:])
    final=model.calls[-1]['request']['messages']
    assert next(m for m in final if m['role']=='assistant')['tool_calls']==planning['tool_calls']
    returns=[m for m in final if m['role']=='tool']
    assert [m['tool_call_id'] for m in returns]==[call_id for _,_,call_id in items]
    assert all(json.loads(m['content'])['execution_status']=='failed' and json.loads(m['content'])['lead_basis_ref'] is None
               for m in returns[2:])
    assert json.loads(final[-2]['content'])['transaction_observations']==[]
    assert result['stats']['adaptive_tool_calls']==result['stats']['adaptive_tools_completed']==2
    assert result['stats']['adaptive_tools_failed']==0 and result['stats']['adaptive_tools_rejected']==count-2
    assert len(model.calls)==5 and not model.responses and result['semantic_results']


def test_later_round_can_query_after_receiving_the_exact_budget_rejection():
    value=case();query={'transaction_id':value['transactions'][0]['transaction_id']}
    first=combined_tools(('read_schema',{},'rules'),('check_coverage',{},'coverage'),
                         ('query_transactions',query,'rejected-query'))
    class DependingModel(ScriptedModel):
        def complete(self,messages,tools=None,stage=None):
            if len(self.calls)==3:
                rejected=next(m for m in messages if m.get('tool_call_id')=='rejected-query')
                assert json.loads(rejected['content'])['execution_status']=='failed'
                assert '预算' in json.loads(rejected['content'])['error']
            return super().complete(messages,tools,stage)
    model=DependingModel([json_message({'claims':[],'unresolved':[]}),response_message(value),first,
                          tool_message('query_transactions',query,'actual-query'),
                          {'role':'assistant','content':'结束取证。'},json_message(support_response(value))])
    result=workflow.run_review(value,mode='agent',provider='frozen',model=model)
    trace=[row for row in result['trace'] if 'round' in row]
    assert trace[2]['dispatch_status']=='not_sent' and trace[2]['result_ref'] is None
    assert trace[3]['dispatch_status']=='sent' and trace[3]['status']=='completed'
    summary=json.loads(model.calls[-1]['request']['messages'][-2]['content'])['transaction_observations']
    assert len(summary)==1 and summary[0]['result_ref']==trace[3]['result_ref']
    assert [row['transaction_id'] for row in summary[0]['rows']]==[query['transaction_id']]
    assert result['stats']['adaptive_tool_calls']==3 and result['stats']['adaptive_tools_rejected']==1
    assert result['adaptive_rounds']==2 and len(model.calls)==6 and not model.responses


def test_three_oversized_rounds_never_dispatch_more_than_two_tools_per_round(monkeypatch):
    value=case();dispatched=[];actual=workflow.execute_tool
    def execute(*args):
        dispatched.append((args[3],args[4]))
        return actual(*args)
    monkeypatch.setattr(workflow,'execute_tool',execute)
    rounds=[]
    pairs=[[('read_schema',{}),('check_coverage',{})],
           [('query_transactions',{'direction':'out'}),('query_transactions',{'direction':'in'})],
           [('read_document',{'document_id':'narrative'}),('read_material',{'material_id':'purchase-contract'})]]
    for number,pair in enumerate(pairs):
        items=[(name,args,f'r{number}-sent-{index}') for index,(name,args) in enumerate(pair)]
        items += [('query_transactions',{'counterparty_token':f'unread-{number}-{index}'},f'r{number}-rejected-{index}')
                  for index in range(2)]
        rounds.append(combined_tools(*items))
    model=ScriptedModel([json_message({'claims':[],'unresolved':[]}),response_message(value),*rounds,json_message(support_response(value))])
    result=workflow.run_review(value,mode='agent',provider='frozen',model=model)
    trace=[row for row in result['trace'] if 'round' in row]
    assert dispatched==[item for pair in pairs for item in pair]
    assert all(sum(row['dispatch_status']=='sent' for row in trace if row['round']==number)==2 for number in [1,2,3])
    assert len(trace)==12 and result['stats']['adaptive_tool_calls']==6
    assert result['stats']['adaptive_tools_completed']==6 and result['stats']['adaptive_tools_failed']==0
    assert result['stats']['adaptive_tools_rejected']==6 and result['adaptive_rounds']==3
    final=model.calls[-1]['request']['messages']
    assert [m['tool_calls'] for m in final if m['role']=='assistant']==[m['tool_calls'] for m in rounds]
    assert len([m for m in final if m['role']=='tool'])==12
    assert len(json.loads(final[-2]['content'])['transaction_observations'])==2
    assert len(model.calls)==6 and not model.responses


def test_dispatched_failure_is_counted_separately_from_budget_rejections():
    value=case()
    planning=combined_tools(('not_a_tool',{},'failed'),('check_coverage',{},'completed'),
                            ('query_transactions',{'direction':'out'},'rejected'))
    model=ScriptedModel([json_message({'claims':[],'unresolved':[]}),response_message(value),planning,
                         {'role':'assistant','content':'结束取证。'},json_message(support_response(value))])
    result=workflow.run_review(value,mode='agent',provider='frozen',model=model)
    assert result['stats']['adaptive_tool_calls']==2 and result['stats']['adaptive_tools_completed']==1
    assert result['stats']['adaptive_tools_failed']==1 and result['stats']['adaptive_tools_rejected']==1


@pytest.mark.parametrize('reference_kind',['lead','gap'])
def test_rejected_tool_cannot_support_a_final_reference_or_optional_lead(reference_kind):
    value=case();args={'direction':'out'}
    ref='tool:query_transactions:'+digest(args)[:20]
    answer=semantic_response(value)
    if reference_kind=='lead':
        answer['leads']=[{'observation':'未执行的查询不能作为观察。','question':'另一个核查问题。','basis_refs':[ref]}]
    else:
        answer['gaps']=[{'basis_kind':'explanation_support','basis_ref':ref,
                        'quote':answer['focuses'][0]['quote'],'requested_material':'夹具材料',
                        'reason':'失败或拒绝查询没有提供证据。'}]
    planning=combined_tools(('read_schema',{},'rules'),('check_coverage',{},'coverage'),
                            ('query_transactions',args,'rejected'))
    model=ScriptedModel([json_message({'claims':[],'unresolved':[]}),response_message(value),planning,
                         {'role':'assistant','content':'结束取证。'},json_message({'gaps':answer['gaps'],'leads':answer.get('leads',[])})])
    result=workflow.run_review(value,mode='agent',provider='frozen',model=model)
    assert result['run_status']=='partial' and result['semantic_results'] and not result['lead_candidates']
    assert any(row['check_id']=='semantic' and row['status']=='failed' for row in result['required_checks'])
    assert result['stats']['adaptive_tool_calls']==2 and result['stats']['adaptive_tools_rejected']==1
    assert json.loads(model.calls[-1]['request']['messages'][-2]['content'])['transaction_observations']==[]
    assert len(model.calls)==5 and not model.responses
