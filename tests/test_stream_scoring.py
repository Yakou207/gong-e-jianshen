"""Frozen SSE wire policy and saved provider hashes, using offline records only."""
from copy import deepcopy

import pytest

from aml_qc.depgraph import digest
from aml_qc.evaluation_budget import _provider_record
from aml_qc.llm import GENERATION, generation_request
from scripts.score_evaluation import verify_call_configuration


MESSAGES = [{'role': 'user', 'content': 'Offline JSON fixture.'}]
TOOLS = [{'type': 'function', 'function': {'name': 'fixture', 'parameters': {'type': 'object'}}}]


def record(stage):
    return {'stage': stage, 'status': 'completed', 'generation': deepcopy(GENERATION),
            'request': generation_request('deepseek-flash', MESSAGES,
                       TOOLS if stage == 'tool_review' else None, stage=stage)}


def method(call):
    return {'model': 'deepseek-flash', 'generation': deepcopy(call['generation'])}


@pytest.mark.parametrize('stage', ['extraction', 'tool_review', 'final'])
@pytest.mark.parametrize('inferred_stage', [False, True])
def test_saved_provider_projection_preserves_exact_stream_wire_and_hash(stage, inferred_stage):
    call = record(stage)
    call.update(request_hash=digest(call['request']), dispatch_status='sent')
    provider = _provider_record(call)
    assert provider['request'] == call['request']
    assert provider['request_hash'] == digest(provider['request'])
    if stage == 'final':
        assert provider['request']['stream'] is True
        assert provider['request']['stream_options'] == {'include_usage': True}
    else:
        assert 'stream' not in provider['request'] and 'stream_options' not in provider['request']
    call['provider_records'] = [provider]
    call['provider_request_hash'] = provider['request_hash']
    if inferred_stage:
        call.pop('stage')
    verify_call_configuration([call], method(call))


@pytest.mark.parametrize('mutation', ['missing_stream', 'disabled_stream', 'integer_stream',
    'missing_options', 'disabled_usage', 'integer_usage', 'extra_option'])
def test_declared_sse_final_rejects_wrong_wire_even_with_fresh_request_hash(mutation):
    call = record('final')
    request = call['request']
    if mutation == 'missing_stream':
        request.pop('stream')
    elif mutation == 'disabled_stream':
        request['stream'] = False
    elif mutation == 'integer_stream':
        request['stream'] = 1
    elif mutation == 'missing_options':
        request.pop('stream_options')
    elif mutation == 'extra_option':
        request['stream_options']['unfrozen_option'] = True
    else:
        request['stream_options']['include_usage'] = 1 if mutation == 'integer_usage' else False
    call['request_hash'] = digest(request)
    with pytest.raises(ValueError, match='transport_mismatch'):
        verify_call_configuration([call], method(call))


@pytest.mark.parametrize('stage', ['extraction', 'tool_review'])
@pytest.mark.parametrize('mutation', ['stream', 'options_only'])
def test_declared_sse_policy_rejects_streaming_nonfinal_stages(stage, mutation):
    call = record(stage)
    if mutation == 'stream':
        call['request']['stream'] = True
    else:
        call['request']['stream_options'] = {'include_usage': True}
    call['request_hash'] = digest(call['request'])
    with pytest.raises(ValueError, match='transport_mismatch'):
        verify_call_configuration([call], method(call))


def test_legacy_final_without_transport_declaration_keeps_original_nonstream_wire():
    call = record('final')
    call['generation'].pop('final_transport')
    call['request'].pop('stream')
    call['request'].pop('stream_options')
    call.update(request_hash=digest(call['request']), dispatch_status='sent')
    call['provider_records'] = [_provider_record(call)]
    verify_call_configuration([call], method(call))


def test_frozen_logical_lookup_compatibility_cannot_bypass_final_stream_policy():
    call = record('final')
    logical = {'model': 'deepseek-flash', 'messages': deepcopy(MESSAGES), 'tools': None, 'stage': 'final'}
    call['provider_records'] = [{'status': 'frozen', 'stage': 'final',
                                'request': logical, 'request_hash': digest(logical)}]
    verify_call_configuration([call], method(call))
    call['request']['stream'] = False
    call['request_hash'] = digest(call['request'])
    with pytest.raises(ValueError, match='transport_mismatch'):
        verify_call_configuration([call], method(call))
