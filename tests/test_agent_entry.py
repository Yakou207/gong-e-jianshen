"""Tool-enabled semantic entry, with final validation and finite budgets intact."""
import json

import pytest

from aml_qc import llm,workflow
from test_model_safety import ScriptedModel,case,json_message,semantic_response,tool_message


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    def forbidden(*args,**kwargs):
        pytest.fail('Agent entry tests must not read credentials or call a live model')
    monkeypatch.setattr(llm,'settings',forbidden)
    monkeypatch.setattr(llm.httpx,'post',forbidden)


def test_agent_can_read_evidence_before_its_first_semantic_answer():
    value=case()
    model=ScriptedModel([json_message({'claims':[],'unresolved':[]}),tool_message('read_schema'),
                         {'role':'assistant','content':'取证结束。'},
                         json_message(semantic_response(value))])
    result=workflow.run_review(value,mode='agent',provider='frozen',model=model)
    assert model.calls[1]['request']['tools']
    context=json.loads(model.calls[1]['request']['messages'][1]['content'])
    assert 'initial_candidates' not in context
    assert context['case']['checks']['features']
    assert result['stats']['model_calls']==4
    assert result['stats']['adaptive_tools_completed']==1
    assert result['semantic_results'] and not model.responses


@pytest.mark.parametrize('draft', ['plain text', '', 'valid_json'])
def test_agent_can_finish_without_unnecessary_tools(draft):
    value=case()
    planning=json_message(semantic_response(value)) if draft=='valid_json' else {'role':'assistant','content':draft}
    model=ScriptedModel([json_message({'claims':[],'unresolved':[]}),planning,json_message(semantic_response(value))])
    result=workflow.run_review(value,mode='agent',provider='frozen',model=model)
    assert len(model.calls)==3 and model.calls[1]['request']['tools']
    assert model.calls[-1]['request']['tools'] is None and not model.responses
    assert result['stats']['adaptive_tool_calls']==0 and result['semantic_results']


def test_three_tool_rounds_have_one_bounded_final_answer():
    value=case()
    model=ScriptedModel([json_message({'claims':[],'unresolved':[]}),tool_message('read_schema',{},'rules'),
        tool_message('compute_features',{'feature_code':'F2'},'features'),
        tool_message('query_transactions',{'direction':'out'},'query'),json_message(semantic_response(value))])
    result=workflow.run_review(value,mode='agent',provider='frozen',model=model)
    assert result['adaptive_rounds']==3 and result['stats']['model_calls']==5
    assert model.calls[-1]['request']['tools'] is None
    assert result['semantic_results'] and not model.responses
    assert result['execution']['max_model_calls']==6


def test_invalid_semantic_final_is_not_repaired_or_promoted_after_reading():
    value=case();value['material_links'][0]['schema_version']='S9'
    answer=semantic_response(value)
    quote=next(d['text'] for d in value['documents'] if d['document_id']=='narrative')
    answer['gaps']=[{'basis_kind':'material_mismatch','basis_ref':'purchase-link','quote':quote,
                    'requested_material':'夹具建议','reason':'错误地把待判断当成不匹配'}]
    model=ScriptedModel([json_message({'claims':[],'unresolved':[]}),tool_message('read_schema'),json_message(semantic_response(value)),json_message(answer)])
    result=workflow.run_review(value,mode='agent',provider='frozen',model=model)
    assert result['material_results'][0]['result']=='pending_judgement'
    assert result['stats']['adaptive_tools_completed']==1 and len(model.calls)==4
    assert result['run_status']=='partial' and not result['semantic_results']
    assert any(c['check_id']=='semantic' and c['status']=='failed' for c in result['required_checks'])
    assert not model.responses
