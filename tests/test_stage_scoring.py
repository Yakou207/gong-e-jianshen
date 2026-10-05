"""Frozen stage-policy checks, without model calls or human quality claims."""
from copy import deepcopy

import pytest

from aml_qc.depgraph import digest
from aml_qc.llm import GENERATION, generation_request
from scripts.score_evaluation import verify_call_configuration


METHOD = {'model': 'deepseek-flash', 'generation': deepcopy(GENERATION)}
MESSAGES = [{'role': 'user', 'content': 'Offline JSON fixture.'}]
TOOLS = [{'type': 'function', 'function': {'name': 'fixture', 'parameters': {'type': 'object'}}}]


def record(stage):
    return {'stage': stage, 'request': generation_request('deepseek-flash', MESSAGES,
             TOOLS if stage == 'tool_review' else None, stage=stage),
            'generation': deepcopy(GENERATION), 'model_returned': 'deepseek-flash'}


@pytest.mark.parametrize('stage', ['extraction', 'tool_review', 'final'])
@pytest.mark.parametrize('legacy', [False, True])
def test_actual_wire_stage_cap_is_accepted_instead_of_global_ceiling(stage, legacy):
    call = record(stage)
    if legacy:
        call.pop('stage')
    verify_call_configuration([call], METHOD)


@pytest.mark.parametrize('stage', ['extraction', 'tool_review', 'final'])
@pytest.mark.parametrize('mutation', ['cap', 'thinking', 'effort', 'temperature', 'stage', 'generation'])
def test_wrong_frozen_stage_configuration_is_rejected(stage, mutation):
    call = record(stage)
    if mutation == 'cap':
        call['request']['max_tokens'] = 16384 if stage != 'final' else 4096
    elif mutation == 'thinking':
        call['request']['thinking']['type'] = 'disabled' if stage == 'final' else 'enabled'
    elif mutation == 'effort':
        call['request']['reasoning_effort'] = 'high'
    elif mutation == 'temperature':
        call['request']['temperature'] = 1
    elif mutation == 'stage':
        call['stage'] = 'final' if stage != 'final' else 'extraction'
    else:
        call['generation']['reasoning_effort'] = 'high'
    with pytest.raises(ValueError, match='generation_mismatch'):
        verify_call_configuration([call], METHOD)


def test_missing_cap_cannot_be_inferred_or_silently_skipped():
    call = record('extraction')
    call.pop('stage')
    call['request'].pop('max_tokens')
    with pytest.raises(ValueError, match='stage_ambiguous'):
        verify_call_configuration([call], METHOD)


def test_explicit_current_stage_requires_saved_wire():
    with pytest.raises(ValueError, match='request_missing'):
        verify_call_configuration([{'stage': 'final', 'status': 'completed', 'usage': {}}], METHOD)


def test_message_mutation_with_old_request_hash_is_rejected():
    call = record('final')
    call['request_hash'] = digest(call['request'])
    call['request']['messages'][0]['content'] = 'Changed source input.'
    with pytest.raises(ValueError, match='request_hash_mismatch'):
        verify_call_configuration([call], METHOD)


def test_nested_provider_wire_cannot_differ_from_reserved_intent():
    call = record('final')
    provider = record('extraction')
    provider['request_hash'] = digest(provider['request'])
    call['provider_records'] = [provider]
    with pytest.raises(ValueError, match='provider_request_mismatch'):
        verify_call_configuration([call], METHOD)


def test_frozen_exact_logical_key_is_explicit_mechanism_compatibility():
    call = record('extraction')
    logical = {'model': METHOD['model'], 'messages': deepcopy(MESSAGES), 'tools': None, 'stage': 'extraction'}
    call['provider_records'] = [{'status': 'frozen', 'request': logical, 'request_hash': digest(logical)}]
    verify_call_configuration([call], METHOD)
    call['provider_records'][0]['request']['messages'][0]['content'] = 'Wrong frozen lookup.'
    with pytest.raises(ValueError, match='provider_request_mismatch'):
        verify_call_configuration([call], METHOD)


def test_inferred_wire_stage_accepts_same_explicit_frozen_lookup_stage():
    call = record('extraction')
    call.pop('stage')
    logical = {'model': METHOD['model'], 'messages': deepcopy(MESSAGES), 'tools': None, 'stage': 'extraction'}
    call['provider_records'] = [{'status': 'frozen', 'request': logical, 'request_hash': digest(logical)}]
    verify_call_configuration([call], METHOD)


@pytest.mark.parametrize('metadata', ['stage', 'generation'])
def test_frozen_metadata_cannot_disagree_with_its_exact_lookup(metadata):
    call = record('extraction')
    logical = {'model': METHOD['model'], 'messages': deepcopy(MESSAGES), 'tools': None, 'stage': 'extraction'}
    provider = {'status': 'frozen', 'request': logical, 'request_hash': digest(logical)}
    provider[metadata] = 'final' if metadata == 'stage' else {**GENERATION, 'reasoning_effort': 'high'}
    call['provider_records'] = [provider]
    with pytest.raises(ValueError, match='generation_mismatch'):
        verify_call_configuration([call], METHOD)


def test_dispatched_request_requires_saved_hash():
    call = record('final')
    call['dispatch_status'] = 'sent'
    with pytest.raises(ValueError, match='request_hash_missing'):
        verify_call_configuration([call], METHOD)


@pytest.mark.parametrize('frozen', [False, True])
def test_provider_request_hash_points_to_saved_provider_record(frozen):
    call = record('extraction')
    actual = ({'model': METHOD['model'], 'messages': deepcopy(MESSAGES), 'tools': None, 'stage': 'extraction'}
              if frozen else deepcopy(call['request']))
    provider = {'request': actual, 'request_hash': digest(actual), 'stage': 'extraction',
                'status': 'frozen' if frozen else 'completed'}
    call['provider_records'] = [provider]
    call['provider_request_hash'] = digest(actual)
    verify_call_configuration([call], METHOD)
    call['provider_request_hash'] = '0' * 64
    with pytest.raises(ValueError, match='provider_request_hash_mismatch'):
        verify_call_configuration([call], METHOD)
