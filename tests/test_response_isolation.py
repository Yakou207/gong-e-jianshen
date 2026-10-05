"""Mechanism fixtures for actual request isolation, independent reuse and failure."""
from copy import deepcopy
import json

import pytest
from pydantic import ValidationError

from aml_qc import workflow
from aml_qc.contracts import ResponseOutput, SupportOutput, contract_schemas
from aml_qc.depgraph import business_result, digest
from aml_qc.llm import ModelError
from test_model_safety import case, json_message, tool_message


class IsolatedModel:
    model = 'isolated-mechanism-fixture'

    def __init__(self, *, support=None, response=None, rounds=0):
        self.calls = []
        self.support = {'gaps': [], 'leads': []} if support is None else deepcopy(support)
        self.response = deepcopy(response)
        self.rounds = rounds
        self.tool_rounds = 0

    def complete(self, messages, tools=None, stage=None):
        if len(self.calls) >= 6:
            raise ModelError('fixture call limit exceeded')
        request = deepcopy({'model': self.model, 'messages': messages, 'tools': tools})
        if stage is not None:
            request['stage'] = stage
        self.calls.append({'request_hash': digest(request), 'request': request,
                           'status': 'frozen', 'usage': {}})
        if tools:
            self.tool_rounds += 1
            if self.tool_rounds <= self.rounds:
                return tool_message('query_transactions', {'direction': 'out'}, str(self.tool_rounds))
            return {'role': 'assistant', 'content': 'No further tool request.'}
        if messages[0]['content'] == workflow.EXTRACT_SYSTEM:
            return json_message({'claims': [], 'unresolved': []})
        if messages[0]['content'] == workflow.RESPONSE_SYSTEM:
            if self.response is not None:
                return json_message(self.response)
            data = json.loads(messages[-1]['content'])
            value = data.get('input', data)
            return json_message({'focuses': [{'focus_id': f['focus_id'], 'status': 'addressed',
                'quote': value['narrative']['text'], 'reason': 'Explicit mechanism fixture'}
                for f in value['focuses']]})
        assert messages[0]['content'] == workflow.SEMANTIC_SYSTEM
        return json_message(self.support)


def response_requests(model):
    return [row['request'] for row in model.calls
            if row['request']['messages'][0]['content'] == workflow.RESPONSE_SYSTEM]


@pytest.mark.parametrize('mode', ['fixed', 'agent'])
def test_response_wire_is_identical_when_only_evidence_changes(mode):
    original = case()
    changed = deepcopy(original)
    changed['transactions'][0]['amount'] = '98765.43'
    changed['materials'][0]['amount'] = '54321.00'
    changed['materials'][0]['text'] = 'EVIDENCE-ONLY-SENTINEL'
    changed['coverage'][0]['status'] = 'partial'
    models = [IsolatedModel(), IsolatedModel()]
    for value, model in zip([original, changed], models):
        workflow.run_review(value, provider='frozen', mode=mode, model=model)
    assert len(response_requests(models[0])) == len(response_requests(models[1])) == 1
    assert response_requests(models[0]) == response_requests(models[1])
    captured = json.dumps(response_requests(models[1]), ensure_ascii=False)
    assert 'EVIDENCE-ONLY-SENTINEL' not in captured and '98765.43' not in captured
    assert all(key not in captured for key in ['transaction_observations', 'material_results', 'read_scope_progress'])


def test_evidence_change_reuses_response_but_full_run_independently_executes_it():
    original = case()
    before = workflow.run_review(original, provider='frozen', model=IsolatedModel())
    changed = deepcopy(original)
    changed['transactions'][0]['amount'] = '98765.43'
    incremental_model, full_model = IsolatedModel(), IsolatedModel()
    incremental = workflow.run_review(changed, provider='frozen', strategy='incremental',
                                      previous=before['snapshot'], model=incremental_model)
    full = workflow.run_review(changed, provider='frozen', strategy='full',
                               previous=None, model=full_model)
    assert response_requests(incremental_model) == []
    assert len(response_requests(full_model)) == 1
    assert business_result(incremental) == business_result(full)


@pytest.mark.parametrize('change', ['narrative', 'focus', 'requirements', 'scope', 'schema'])
def test_response_recomputed_when_original_task_changes(change):
    original = case()
    before_model = IsolatedModel()
    before = workflow.run_review(original, provider='frozen', model=before_model)
    value = deepcopy(original)
    if change == 'narrative':
        next(d for d in value['documents'] if d['document_id'] == 'narrative')['text'] += ' 补充原理由。'
    elif change == 'focus':
        value['alert']['focuses'][0]['text'] += ' 新需回应事项。'
    elif change == 'requirements':
        value['alert']['focuses'][0]['response_requirements'] = [
            {'kind': 'verification_result', 'text': '报告明确核对结果。'}]
    elif change == 'scope':
        value['review_scope']['target_labels'] = ['alert_response']
    else:
        value['schema'] = workflow.default_schema()
        value['schema']['labels']['alert_response']['definition'] += ' 新任务规范。'
    model = IsolatedModel()
    result = workflow.run_review(value, provider='frozen', strategy='incremental',
                                 previous=before['snapshot'], model=model)
    assert len(response_requests(model)) == 1
    assert response_requests(model)[0] != response_requests(before_model)[0]
    assert result['semantic_results']


@pytest.mark.parametrize('extra', ['focuses', 'basis_ref_id'])
def test_support_cannot_overwrite_focus_or_silently_drop_illegal_fields(extra):
    value = case()
    support = {'gaps': [], 'leads': []}
    if extra == 'focuses':
        support['focuses'] = [{'focus_id': f['focus_id'], 'status': 'not_addressed',
                              'quote': '', 'reason': 'attempted overwrite'} for f in workflow.focuses_for(value)]
    else:
        support['gaps'] = [{'basis_kind': 'explanation_support', 'basis_ref': 'narrative',
                            'basis_ref_id': 'narrative', 'quote': workflow.narrative(value)['text'],
                            'requested_material': 'fixture', 'reason': 'fixture'}]
    model = IsolatedModel(support=support)
    result = workflow.run_review(value, provider='frozen', model=model)
    assert result['run_status'] == 'partial'
    assert len(response_requests(model)) == 1
    assert len(model.calls) == 3
    assert all(f['status'] == 'addressed' for f in result['semantic_results'])
    assert any(i['type'] == 'execution_failed' for i in result['issues'])


def test_separated_stages_fit_original_six_call_limit():
    model = IsolatedModel(rounds=3)
    result = workflow.run_review(case(), provider='frozen', mode='agent', model=model)
    assert len(model.calls) == 6 and len(response_requests(model)) == 1
    assert result['adaptive_rounds'] == 3
    assert result['run_status'] == 'completed'
    assert model.calls[-1]['request']['tools'] is None


def test_two_strict_schemas_reject_cross_stage_fields():
    assert contract_schemas()['response']['additionalProperties'] is False
    assert contract_schemas()['support']['additionalProperties'] is False
    with pytest.raises(ValidationError):
        ResponseOutput.model_validate({'focuses': [], 'gaps': []})
    with pytest.raises(ValidationError):
        SupportOutput.model_validate({'focuses': [], 'gaps': []})
