"""Resume requires the same wire and stage in intent, provider, journal and raw."""
from copy import deepcopy
import json

import pytest

from aml_qc.depgraph import digest
from aml_qc.llm import GENERATION, generation_request
from scripts.run_evaluation import validate_call_journal


def audit_fixture(tmp_path, method='Fixed'):
    request = generation_request('deepseek-flash', [{'role': 'user', 'content': 'Offline JSON fixture.'}])
    budget = {'max_calls': 6, 'max_output_tokens': 16384, 'total_token_budget': 10000000, 'currency_limit': '20'}
    provider = {'request': deepcopy(request), 'request_hash': digest(request), 'stage': 'final',
                'generation': deepcopy(GENERATION), 'status': 'completed'}
    record = {'request': request, 'request_hash': digest(request), 'stage': 'final',
              'generation': deepcopy(GENERATION), 'call_id': 'call-1', 'budget_event_id': 'call-1',
              'dispatch_status': 'sent', 'status': 'completed', 'usage': {},
              'provider_records': [provider]}
    events = [{'event': 'call_reserved', 'run_id': 'run-1', 'call_id': 'call-1', 'budget': budget,
               'stage': 'final', 'request': deepcopy(request), 'request_hash': digest(request)},
              {'event': 'call_finished', 'run_id': 'run-1', 'call_id': 'call-1', 'record': deepcopy(record)}]
    raw_record = deepcopy(record)
    if method == 'B0':
        raw_record['provider_request_hash'] = digest(request)
    raw = {'call_records' if method == 'B0' else 'model_requests': [raw_record]}
    directory = tmp_path / 'run/run-1'
    directory.mkdir(parents=True)
    path = directory / 'raw.json'
    path.write_text(json.dumps(raw))
    plans = [{'planned_run_id': 'run-1', 'method': method}]
    manifest = {'methods': {method: {'model': 'deepseek-flash', 'generation': deepcopy(GENERATION), 'budget': budget}}}
    return events, plans, manifest, path, raw


@pytest.mark.parametrize('method', ['Fixed', 'Agent', 'B0'])
def test_current_wire_is_valid_across_all_method_audits(tmp_path, method):
    events, plans, manifest, _, _ = audit_fixture(tmp_path, method)
    validate_call_journal(events, plans, manifest, tmp_path)


@pytest.mark.parametrize('change', ['raw_stage', 'journal_stage', 'raw_input', 'raw_input_and_hash', 'provider_input'])
@pytest.mark.parametrize('method', ['Fixed', 'B0'])
def test_stage_or_input_mutation_cannot_survive_resume(tmp_path, change, method):
    events, plans, manifest, path, raw = audit_fixture(tmp_path, method)
    record = next(iter(raw.values()))[0]
    if change == 'raw_stage':
        record['stage'] = 'extraction'
    elif change == 'journal_stage':
        events[0]['stage'] = events[1]['record']['stage'] = 'extraction'
    elif change in {'raw_input', 'raw_input_and_hash'}:
        record['request']['messages'][0]['content'] = 'Different input.'
        if change == 'raw_input_and_hash':
            record['request_hash'] = digest(record['request'])
    else:
        record['provider_records'][0]['request']['messages'][0]['content'] = 'Different provider input.'
        record['provider_records'][0]['request_hash'] = digest(record['provider_records'][0]['request'])
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='generation_mismatch|request_hash_mismatch|request_mismatch|differs_from_journal'):
        validate_call_journal(events, plans, manifest, tmp_path)
