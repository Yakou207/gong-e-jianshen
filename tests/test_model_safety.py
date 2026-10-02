"""Offline regressions for model boundaries and immutable replay evidence.

No test reads .env or contacts a model service. Scripted/frozen responses are
mechanism fixtures, never evidence that the live Agent passed acceptance.
"""
from copy import deepcopy
import json
from pathlib import Path

import httpx
import pytest

from aml_qc import llm
from aml_qc.depgraph import business_result, digest
from aml_qc.ingest import load_case
from aml_qc.llm import DeepSeek, FrozenModel, ModelError
from aml_qc.workflow import focuses_for, narrative, normalize_semantic, run_review


DATA = Path(__file__).resolve().parents[1] / 'data' / 'synthetic'
MODEL = 'offline-safety-fixture'


@pytest.fixture(autouse=True)
def no_credentials_or_network(monkeypatch):
    monkeypatch.setattr(llm, 'settings', lambda: {
        'DEEPSEEK_API_KEY': 'test-placeholder',
        'DEEPSEEK_BASE_URL': 'https://model.invalid',
        'DEEPSEEK_MODEL': MODEL,
    })

    def forbidden_request(*args, **kwargs):
        raise AssertionError('Safety tests must not call a live model service')

    monkeypatch.setattr(llm.httpx, 'post', forbidden_request)


def case():
    return load_case(DATA / 'seed-01.json')


def json_message(value):
    return {'role': 'assistant', 'content': json.dumps(value, ensure_ascii=False)}


def semantic_response(value):
    return {
        'focuses': [
            {'focus_id': f['focus_id'], 'status': 'addressed',
             'quote': narrative(value)['text'], 'reason': 'Offline fixture only'}
            for f in focuses_for(value)
        ],
        'gaps': [],
    }


def tool_message(name='check_coverage', arguments=None, call_id='call-1'):
    return {'role': 'assistant', 'tool_calls': [
        {'id': call_id, 'type': 'function', 'function': {
            'name': name, 'arguments': json.dumps(arguments or {})}}
    ]}


class ScriptedModel:
    """Finite fixture with exact request capture for later frozen replay."""

    model = MODEL

    def __init__(self, responses):
        self.responses = deepcopy(responses)
        self.calls = []
        self.request_products = {}

    def complete(self, messages, tools=None):
        assert self.responses, 'Unexpected extra model request'
        request = deepcopy({'model': self.model, 'messages': messages, 'tools': tools})
        key = digest(request)
        response = self.responses.pop(0)
        if key in self.request_products:
            assert self.request_products[key] == response
        self.request_products[key] = deepcopy(response)
        self.calls.append({'request_hash': key, 'request': request, 'status': 'frozen',
                           'usage': {'prompt_tokens': 0, 'completion_tokens': 0}})
        return deepcopy(response)


def script(value, extraction=None, tools=None):
    responses = [json_message(extraction if extraction is not None else
                              {'claims': [], 'unresolved': []}),
                 json_message(semantic_response(value))]
    if tools is not None:
        responses.extend(tools)
        responses.append(json_message(semantic_response(value)))
    return ScriptedModel(responses)


@pytest.mark.parametrize('extraction', [
    {'claims': ['not-an-object'], 'unresolved': []},
    {'claims': [None], 'unresolved': []},
    {'claims': None, 'unresolved': []},
    {'claims': [], 'unresolved': 'not-an-array'},
])
def test_malformed_claim_payload_keeps_required_check_unfinished(extraction):
    value = case()
    result = run_review(value, provider='frozen', model=script(value, extraction))
    extraction_check = next(c for c in result['required_checks'] if c['check_id'] == 'extraction')
    assert extraction_check['status'] != 'completed'
    assert result['run_status'] == 'partial'
    assert result['open_items']


@pytest.mark.parametrize('mutation', ['non_object', 'bad_id', 'bad_quote', 'bad_gap'])
def test_malformed_semantic_nested_payload_is_model_error(mutation):
    value = case()
    response = semantic_response(value)
    if mutation == 'non_object':
        response['focuses'] = ['not-an-object']
    elif mutation == 'bad_id':
        response['focuses'][0]['focus_id'] = []
    elif mutation == 'bad_quote':
        response['focuses'][0]['quote'] = {'not': 'text'}
    else:
        response['gaps'] = ['not-an-object']
    with pytest.raises(ModelError):
        normalize_semantic(value, response)


def test_semantic_failure_is_recorded_instead_of_reporting_complete():
    value = case()
    model = ScriptedModel([json_message({'claims': [], 'unresolved': []}),
                           json_message({'focuses': ['invalid'], 'gaps': []})])
    result = run_review(value, provider='frozen', model=model)
    assert result['run_status'] == 'partial'
    assert any(c['check_id'] == 'semantic' and c['status'] == 'failed'
               for c in result['required_checks'])
    assert any(i['type'] == 'execution_failed' for i in result['issues'])


def test_failed_tools_do_not_establish_agent_acceptance():
    value = case()
    model = script(value, tools=[tool_message('not_a_tool'),
                                 tool_message('not_a_tool', call_id='call-2')])
    # Supplied fixture prevents DeepSeek construction; the provider label alone
    # must not turn two failed requests into live Agent acceptance.
    result = run_review(value, provider='deepseek', mode='agent', model=model)
    attempts = [t for t in result['trace'] if t.get('round')]
    assert len(attempts) == 2
    assert all(t['status'] == 'failed' for t in attempts)
    assert result['agent_verified'] is False


def test_material_execution_error_cannot_be_completed_check():
    value = case()
    value['review_scope'] = {'target_labels': ['material_relation']}
    value['materials'][0]['period']['start'] = 'invalid-time'
    result = run_review(value, provider='frozen', model=script(value))
    assert result['material_results'][0]['execution_status'] == 'failed'
    assert next(c for c in result['required_checks'] if c['check_id'] == 'materials')['status'] == 'failed'
    assert result['run_status'] == 'partial'
    assert result['open_items']


def test_collection_reordering_preserves_actual_model_requests():
    original = case()
    extra_material = deepcopy(original['materials'][0])
    extra_material['material_id'] = 'zz-unlinked-material'
    original['materials'].append(extra_material)
    reordered = deepcopy(original)
    for name in ('documents', 'materials', 'transactions'):
        reordered[name].reverse()
    before = deepcopy(reordered)
    first_model, second_model = script(original), script(reordered)
    first = run_review(original, provider='frozen', model=first_model)
    second = run_review(reordered, provider='frozen', model=second_model)
    assert [r['request_hash'] for r in first_model.calls] == [r['request_hash'] for r in second_model.calls]
    assert business_result(first) == business_result(second)
    assert reordered == before, 'Canonicalization must not mutate the caller snapshot'


def test_reused_agent_stage_tool_refs_resolve_in_current_snapshot():
    value = case()
    first = run_review(value, provider='frozen', mode='agent',
                       model=script(value, tools=[tool_message()]))
    replay = FrozenModel({}, model=MODEL)
    second = run_review(value, provider='frozen', mode='agent', strategy='incremental',
                        previous=first['snapshot'], model=replay)
    assert replay.calls == [], 'Unchanged stage should reuse rather than invent a new model response'
    attempts = [t for t in second['trace'] if t.get('round') and t['status'] == 'reused']
    assert attempts
    assert second['stats']['adaptive_tool_calls'] == 0
    assert second['stats']['adaptive_tools_completed'] == 0
    assert second['stats']['replayed_tool_calls'] == len(attempts)
    assert second['adaptive_rounds'] == 0
    assert second['replayed_adaptive_rounds'] == 1
    for entry in attempts:
        node = second['snapshot']['nodes'][entry['result_ref']]
        assert node['result'] == entry['result']
        for dependency in node['dependencies']:
            assert dependency in second['snapshot']['sources'] or dependency in second['snapshot']['nodes']
    assert business_result(first) == business_result(second)


def test_final_semantic_failure_keeps_actual_tool_attempts_and_usage():
    value = case()
    model = script(value, tools=[tool_message()])
    model.responses[-1] = json_message({'focuses': ['invalid final output'], 'gaps': []})
    result = run_review(value, provider='frozen', mode='agent', model=model)
    assert result['run_status'] == 'partial'
    assert result['stats']['model_calls'] == 4
    assert result['stats']['adaptive_tool_calls'] == 1
    assert result['stats']['adaptive_tools_completed'] == 1
    assert any(t.get('round') == 1 and t['status'] == 'completed' for t in result['trace'])
    assert any(c['check_id'] == 'semantic' and c['status'] == 'failed' for c in result['required_checks'])


def test_truncation_preserves_already_reported_usage(monkeypatch):
    usage = {'prompt_tokens': 101, 'completion_tokens': 4096,
             'prompt_cache_hit_tokens': 20, 'prompt_cache_miss_tokens': 81}
    body = {'model': MODEL, 'system_fingerprint': 'offline-fingerprint', 'usage': usage,
            'choices': [{'finish_reason': 'length', 'message': {'role': 'assistant', 'content': '{'}}]}
    monkeypatch.setattr(llm.httpx, 'post', lambda *a, **k: httpx.Response(200, json=body))
    model = DeepSeek()
    with pytest.raises(ModelError, match='截断'):
        model.complete([{'role': 'user', 'content': 'Offline fixture'}])
    assert model.calls[0]['status'] == 'failed'
    assert model.calls[0]['usage'] == usage
    assert model.calls[0]['model_returned'] == MODEL


@pytest.mark.parametrize('usage', ['absent', None, {}, {'prompt_tokens': 7}])
def test_missing_usage_is_unknown_not_zero_complete(monkeypatch, usage):
    value = case()

    def fake_post(*args, **kwargs):
        system = kwargs['json']['messages'][0]['content']
        content = {'claims': [], 'unresolved': []} if '提取理由' in system else semantic_response(value)
        body = {'model': MODEL, 'choices': [{'finish_reason': 'stop', 'message': json_message(content)}]}
        if usage != 'absent':
            body['usage'] = usage
        return httpx.Response(200, json=body)

    monkeypatch.setattr(llm.httpx, 'post', fake_post)
    result = run_review(value, provider='deepseek', model=DeepSeek())
    assert result['stats']['model_calls'] == 2
    assert result['stats']['usage_complete'] is False
    assert result['stats']['input_tokens'] is None
    assert result['stats']['output_tokens'] is None


def test_request_capture_is_immutable_and_hash_matches(monkeypatch):
    body = {'usage': {'prompt_tokens': 1, 'completion_tokens': 1},
            'choices': [{'finish_reason': 'stop', 'message': json_message({})}]}
    monkeypatch.setattr(llm.httpx, 'post', lambda *a, **k: httpx.Response(200, json=body))
    messages = [{'role': 'user', 'content': 'original'}]
    model = DeepSeek()
    model.complete(messages)
    recorded = deepcopy(model.calls[0]['request'])
    messages[0]['content'] = 'changed'
    messages.append({'role': 'assistant', 'content': 'later turn'})
    assert model.calls[0]['request'] == recorded
    assert model.calls[0]['request_hash'] == digest(model.calls[0]['request'])


def test_exact_request_frozen_full_incremental_and_independent_expected_value():
    old = case()
    outgoing = deepcopy(next(t for t in old['transactions'] if t['direction'] == 'out'))
    old['transactions'] = [t for t in old['transactions'] if t['direction'] == 'in']
    old['review_scope'] = {'target_labels': ['count', 'alert_response']}
    narrative(old)['text'] = '本次仅向乙公司支付一次。'
    old['materials'], old['material_links'] = [], []
    new = deepcopy(old)
    new['transactions'].append(outgoing)
    new['data_version'] = '2'
    extraction = {'claims': [{'kind': 'count', 'operator': 'exact', 'value': 1,
                              'direction': 'out', 'counterparty_ref': '乙公司',
                              'document_id': 'narrative', 'quote': narrative(old)['text']}],
                  'unresolved': []}

    # Build fixture products from finite predetermined messages, without an old
    # graph/cache. The mechanism comparison below shares only this exact-key map.
    products = {}
    for value in (old, new):
        recorder = script(value, extraction, tools=[tool_message()])
        run_review(value, provider='frozen', mode='agent', model=recorder)
        assert not recorder.responses
        for key, response in recorder.request_products.items():
            if key in products:
                assert products[key] == response
            products[key] = deepcopy(response)

    old_run = run_review(old, provider='frozen', mode='agent', model=FrozenModel(products, MODEL))
    old_snapshot = deepcopy(old_run['snapshot'])
    incremental = run_review(new, provider='frozen', mode='agent', strategy='incremental',
                             previous=old_run['snapshot'], model=FrozenModel(products, MODEL))
    independent_full = run_review(new, provider='frozen', mode='agent', strategy='full',
                                  model=FrozenModel(products, MODEL))
    assert old_run['claim_results'][0]['result'] == 'contradicted'
    assert independent_full['claim_results'][0]['result'] == 'supported'
    assert incremental['claim_results'][0]['result'] == 'supported'
    assert business_result(incremental) == business_result(independent_full)
    assert independent_full['stats']['reused'] == 0
    assert old_run['snapshot'] == old_snapshot
    assert any(i['type'] == 'claim_error' for i in old_run['issues'])
    assert not any(i['type'] == 'claim_error' for i in independent_full['issues'])
    assert incremental['stats']['revoked'] >= 1
    assert incremental['agent_verified'] is False

    changed_request = FrozenModel(products, MODEL)
    with pytest.raises(ModelError, match='冻结请求未命中'):
        changed_request.complete([{'role': 'user', 'content': 'Different request must never reuse an answer'}])
