"""Wire generation policy and privacy mechanics, with mocked transport only."""
from contextlib import contextmanager
from copy import deepcopy
from decimal import Decimal
import json

import httpx
import pytest

from aml_qc import llm
from aml_qc.depgraph import digest
from aml_qc.evaluation_budget import OUTPUT_BOUND, BudgetError, BudgetLedger, BudgetedModel
from aml_qc.llm import DeepSeek, FrozenModel, ModelError, generation_request
from test_evaluation_budget import BUDGET, MESSAGES, PRICING, usage


TOOLS = [{'type': 'function', 'function': {'name': 'fixture', 'parameters': {'type': 'object'}}}]


@pytest.fixture(autouse=True)
def no_credentials(monkeypatch):
    @contextmanager
    def stream(method, *args, **kwargs):
        assert method == 'POST'
        response = llm.httpx.post(*args, **kwargs)
        try:
            yield response
        finally:
            response.close()
    monkeypatch.setattr(llm.httpx, 'stream', stream)
    monkeypatch.setattr(llm, 'settings', lambda: {
        'DEEPSEEK_API_KEY': 'offline-placeholder', 'DEEPSEEK_BASE_URL': 'https://model.invalid',
        'DEEPSEEK_MODEL': 'deepseek-flash'})


def transport(monkeypatch, *, finish_reason='stop', receipt=None):
    sent = []
    def post(*args, **kwargs):
        sent.append(deepcopy(kwargs['json']))
        return httpx.Response(200, json={'model': 'deepseek-flash', 'usage': usage() if receipt is None else receipt,
            'choices': [{'finish_reason': finish_reason, 'message': {
                'role': 'assistant', 'content': '{}', 'reasoning_content': 'HIDDEN-OUTPUT-REASONING'}}]})
    monkeypatch.setattr(llm.httpx, 'post', post)
    return sent


@pytest.mark.parametrize('tools,stage,thinking,cap', [
    (None, None, 'enabled', 16384), ([], None, 'enabled', 16384),
    (TOOLS, None, 'disabled', 4096), (None, 'extraction', 'disabled', 4096),
    ([], 'extraction', 'disabled', 4096), (TOOLS, 'tool_review', 'disabled', 4096),
    (None, 'final', 'enabled', 16384), ([], 'final', 'enabled', 16384),
])
def test_transport_budget_and_provider_audits_use_one_generation_policy(monkeypatch, tools, stage, thinking, cap):
    sent = transport(monkeypatch)
    events = []
    base = DeepSeek()
    model = BudgetedModel(base, BudgetLedger('30', PRICING), 'policy-fixture', BUDGET, events.append)
    messages = deepcopy(MESSAGES) + [{'role': 'assistant', 'content': 'Visible history.',
                                     'reasoning_content': 'HIDDEN-INPUT-REASONING'}]
    original = deepcopy(messages)
    result = model.complete(messages, tools=tools, stage=stage)
    payload = sent[0]
    assert payload['thinking'] == {'type': thinking} and payload['max_tokens'] == cap
    if thinking == 'enabled':
        assert payload['reasoning_effort'] == 'low' and 'temperature' not in payload
    else:
        assert payload['temperature'] == 0 and 'reasoning_effort' not in payload
    if tools:
        assert payload['tools'] == tools and payload['tool_choice'] == 'auto'
        assert 'response_format' not in payload
    else:
        assert payload['response_format'] == {'type': 'json_object'} and 'tools' not in payload
    assert BUDGET['max_output_tokens'] == llm.GENERATION['max_output_tokens'] == 16384
    assert set(payload) <= {'model', 'messages', 'max_tokens', 'temperature', 'thinking',
                            'reasoning_effort', 'tools', 'tool_choice', 'response_format', 'stream', 'stream_options'}
    record = model.calls[0]
    provider = record['provider_records'][0]
    assert record['stage'] == provider['stage'] == base.calls[0]['stage'] == events[0]['stage']
    assert record['stage'] == (stage if stage is not None else 'tool_review' if tools else 'final')
    assert events[0]['request'] == record['request'] == provider['request'] == base.calls[0]['request'] == payload
    assert events[0]['request_hash'] == record['request_hash'] == provider['request_hash'] == digest(payload)
    assert record['generation'] == provider['generation'] == base.calls[0]['generation'] == llm.GENERATION
    assert llm.GENERATION['stage_max_output_tokens'] == {'extraction': 4096, 'tool_review': 4096, 'final': 16384}
    assert llm.GENERATION['stage_thinking'] == {'extraction': 'disabled', 'tool_review': 'disabled', 'final': 'enabled'}
    assert llm.GENERATION['temperature_applies_to'] == 'non_thinking_only'
    assert llm.GENERATION['reasoning_effort'] == 'low'
    assert llm.GENERATION['reasoning_effort_applies_to'] == 'thinking_only'
    serialized = json.dumps({'events': events, 'calls': model.calls, 'provider_calls': base.calls, 'result': result})
    assert 'reasoning_content' not in serialized and 'HIDDEN-' not in serialized
    assert messages == original and len(sent) == 1


@pytest.mark.parametrize('budgeted', [False, True])
@pytest.mark.parametrize('stage,thinking', [('extraction', 'disabled'), ('final', 'enabled')])
def test_truncated_json_stays_failed_without_retry_or_reasoning_leak(monkeypatch, budgeted, stage, thinking):
    sent = transport(monkeypatch, finish_reason='length')
    base = DeepSeek()
    events = []
    model = BudgetedModel(base, BudgetLedger('30', PRICING), 'truncated-fixture', BUDGET, events.append) if budgeted else base
    with pytest.raises(ModelError, match='截断'):
        model.complete(MESSAGES, stage=stage)
    assert len(sent) == len(base.calls) == 1 and base.calls[0]['status'] == 'failed'
    assert sent[0]['thinking'] == {'type': thinking}
    assert model.calls[0]['status'] == 'failed'
    assert base.calls[0]['usage'] == model.calls[0]['usage'] == usage()
    serialized = json.dumps({'calls': model.calls, 'provider_calls': base.calls, 'events': events})
    assert 'reasoning_content' not in serialized and 'HIDDEN-' not in serialized
    if budgeted:
        assert model.ledger.held == 0 and model.ledger.spent > 0


def test_length_with_known_output_overrun_stops_and_bills_without_unresolved_label(monkeypatch):
    receipt = usage(output=OUTPUT_BOUND + 1)
    sent = transport(monkeypatch, finish_reason='length', receipt=receipt)
    base, events = DeepSeek(), []
    ledger = BudgetLedger('30', PRICING)
    model = BudgetedModel(base, ledger, 'overrun-fixture', BUDGET, events.append)
    with pytest.raises(BudgetError, match='^provider_bound_exceeded$'):
        model.complete(MESSAGES)
    assert len(sent) == len(base.calls) == 1
    assert model.calls[0]['status'] == base.calls[0]['status'] == 'failed'
    assert model.calls[0]['finish_reason'] == 'length' and model.calls[0]['usage'] == receipt
    settlement = events[-1]['settlement']
    assert settlement['status'] == 'overrun' and settlement['reason'] == 'provider_bound_exceeded'
    assert ledger.stopped and ledger.held == 0 and ledger.spent == Decimal(settlement['cost']) > 0
    assert 'budget_charge_unresolved' not in json.dumps(events)
    serialized = json.dumps({'calls': model.calls, 'provider_calls': base.calls, 'events': events})
    assert 'reasoning_content' not in serialized and 'HIDDEN-' not in serialized


@pytest.mark.parametrize('stage,output,finish_reason', [
    ('extraction', 4097, 'stop'), ('extraction', 4097, 'length'), ('final', 16384, 'length'),
])
def test_request_cap_or_length_is_business_failure_with_known_settled_cost(monkeypatch, stage, output, finish_reason):
    receipt = usage(output=output)
    sent = transport(monkeypatch, finish_reason=finish_reason, receipt=receipt)
    base, events = DeepSeek(), []
    ledger = BudgetLedger('30', PRICING)
    model = BudgetedModel(base, ledger, 'request-overrun-fixture', BUDGET, events.append)
    with pytest.raises(ModelError, match='截断|超过当次请求上限'):
        model.complete(MESSAGES, stage=stage)
    assert len(sent) == len(base.calls) == 1
    assert sent[0]['max_tokens'] == llm.GENERATION['stage_max_output_tokens'][stage]
    assert model.calls[0]['status'] == 'failed' and model.calls[0]['usage'] == receipt
    settlement = events[-1]['settlement']
    assert settlement['status'] == 'settled'
    assert not ledger.stopped and ledger.held == 0 and ledger.spent == Decimal(settlement['cost']) > 0
    assert 'reasoning_content' not in json.dumps({'calls': model.calls, 'events': events})


@pytest.mark.parametrize('stage,tools', [
    ('unknown', None), ('', None), (False, None), ([], None),
    ('extraction', TOOLS), ('final', TOOLS), ('tool_review', None), ('tool_review', []),
])
def test_invalid_or_inconsistent_stage_is_rejected_before_dispatch(monkeypatch, stage, tools):
    sent = transport(monkeypatch)
    model = DeepSeek()
    with pytest.raises(ValueError):
        generation_request(model.model, MESSAGES, tools, stage=stage)
    with pytest.raises(ValueError):
        model.complete(MESSAGES, tools=tools, stage=stage)
    with pytest.raises(ValueError):
        FrozenModel({}).complete(MESSAGES, tools=tools, stage=stage)
    assert not sent and not model.calls


def test_frozen_default_preserves_legacy_key_and_strips_hidden_reasoning():
    request = {'model': 'fixture', 'messages': deepcopy(MESSAGES), 'tools': None}
    answer = {'role': 'assistant', 'content': '{}', 'reasoning_content': 'HIDDEN-OUTPUT-REASONING'}
    model = FrozenModel({digest(request): answer}, model='fixture')
    messages = deepcopy(MESSAGES)
    messages[0]['reasoning_content'] = 'HIDDEN-INPUT-REASONING'
    before = deepcopy(messages)
    result = model.complete(messages)
    assert model.calls[0]['request_hash'] == digest(request)
    assert model.calls[0]['request'] == request
    assert messages == before and result == {'role': 'assistant', 'content': '{}'}
    assert 'reasoning_content' not in json.dumps(model.calls) and 'HIDDEN-' not in json.dumps(model.calls)
    assert answer['reasoning_content'] == 'HIDDEN-OUTPUT-REASONING'


def test_frozen_explicit_stage_is_part_of_request_identity():
    request = {'model': 'fixture', 'messages': deepcopy(MESSAGES), 'tools': None, 'stage': 'extraction'}
    model = FrozenModel({digest(request): {'role': 'assistant', 'content': '{}'}}, model='fixture')
    assert model.complete(MESSAGES, stage='extraction')['content'] == '{}'
    assert model.calls[0]['request_hash'] == digest(request)
    with pytest.raises(ModelError, match='冻结请求未命中'):
        model.complete(MESSAGES, stage='final')
    assert len(model.calls) == 1
