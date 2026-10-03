"""Serial, single-attempt execution of a frozen B0/Fixed/Agent experiment.

Default CLI behavior is preflight only. --execute sends paid API requests. The
manifest must bind execution_directory (relative to itself), every implementation
file and prompt, a future pricing.valid_until, and zero retries. Live runs use
deepseek-flash at its official endpoint and a frozen peak/off_peak pricing period.
The operator must verify the frozen tariff against the current official price.
Until a verified holiday calendar is available, live calls are limited to
weekends and hours outside the official weekday peak intervals.

A kernel file lock serializes this LOCAL execution directory. A permanent start
marker prevents retrying a started plan, even after a process dies. Recovery may
repair an index from an immutable raw artifact, but never re-send an interrupted
attempt. This is not a cross-machine lock or an account-wide provider quota.

Budget guarantees are conditional on the frozen price and provider context/output
limits. Each call reserves the whole context bound plus maximum output, not an
estimated token count. Unknown billing keeps the reservation and stops execution.
Raw model records, journal and scoring index stay in the local execution directory.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile

from aml_qc.baseline import raw_case_input, run_baseline
from aml_qc.depgraph import digest
from aml_qc.evaluation_budget import CONTEXT_BOUND, OUTPUT_BOUND, BudgetLedger, BudgetedModel
from aml_qc.llm import DeepSeek
from aml_qc.workflow import run_review
from scripts.score_evaluation import read_object, read_ref, unique_rows
from scripts.validate_evaluation import _credential_path, _read_json, _timestamp, validate_manifest


ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATIONS = [f'aml_qc/{name}.py' for name in ('core', 'ingest', 'schema', 'contracts', 'llm', 'depgraph',
    'workflow', 'claim_edits', 'leads', 'baseline', 'scoring_inputs', 'evaluation_budget')]
IMPLEMENTATIONS += ['scripts/run_evaluation.py', 'scripts/score_evaluation.py', 'scripts/validate_evaluation.py']
PROMPTS = [f'config/prompts/v2.8/{name}.txt' for name in ('agent', 'base', 'extraction', 'semantic')]
PROMPTS += ['config/prompts/b0-v1/direct.txt']
GENERATION = {'temperature': 0, 'thinking': 'disabled', 'max_output_tokens': 4096}


def now():
    return datetime.now(timezone.utc)


def timestamp():
    return now().isoformat()


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json(path, value, *, replace=False):
    """Immutable artifacts use exclusive creation; only derived index is replaced."""
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    descriptor, name = tempfile.mkstemp(prefix='.' + path.name + '-', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, 'wb') as file:
            file.write(raw)
            file.flush()
            os.fsync(file.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    fsync_directory(path.parent)
    return hashlib.sha256(raw).hexdigest()


@contextmanager
def single_writer(directory):
    with (directory / '.runner.lock').open('a') as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('execution_directory_locked') from None
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class Journal:
    def __init__(self, path, manifest_sha):
        self.path, self.manifest_sha, self.events = path, manifest_sha, []
        if path.exists():
            raw = path.read_bytes()
            if raw and not raw.endswith(b'\n'):
                raise ValueError('journal_truncated')
            for number, line in enumerate(raw.splitlines(), 1):
                row = _read_json(line)
                if (not isinstance(row, dict) or row.get('sequence') != number
                        or row.get('manifest_sha256') != manifest_sha or not isinstance(row.get('event'), str)):
                    raise ValueError('journal_binding_or_sequence_invalid')
                self.events.append(row)

    def append(self, event):
        row = {**event, 'sequence': len(self.events) + 1, 'manifest_sha256': self.manifest_sha, 'recorded_at': timestamp()}
        raw = (json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n').encode()
        with self.path.open('ab') as file:
            file.write(raw)
            file.flush()
            os.fsync(file.fileno())
        fsync_directory(self.path.parent)
        self.events.append(row)


def verify_pricing_window(pricing, *, live):
    deadline = pricing.get('valid_until')
    if not _timestamp(deadline) or now() + timedelta(seconds=60) >= datetime.fromisoformat(deadline):
        raise ValueError('pricing_expired_or_too_near_expiry')
    if live:
        current = now().astimezone(timezone(timedelta(hours=8)))
        # The official weekday peak schedule excludes Chinese statutory
        # holidays. Without a verified calendar, that interval is unknown;
        # do not silently treat every weekday as a non-holiday.
        if current.weekday() < 5 and (9 <= current.hour < 12 or 14 <= current.hour < 18):
            raise ValueError('weekday_peak_requires_verified_holiday_calendar')
        if pricing.get('period') != 'off_peak':
            raise ValueError('frozen_pricing_period_is_not_current')
        # Do not let a request enter an interval whose tariff is unknown.
        later = current + timedelta(seconds=60)
        if later.weekday() < 5 and (9 <= later.hour < 12 or 14 <= later.hour < 18):
            raise ValueError('pricing_transition_within_request_timeout')


def preflight(manifest_path, output_dir, *, live, check_price_window=True):
    frozen = validate_manifest(manifest_path)
    if not frozen['frozen_valid']:
        raise ValueError('manifest_not_frozen_valid')
    manifest, sha = read_object(manifest_path)
    if manifest['retry_policy']['max_retries'] != 0:
        raise ValueError('execution_requires_zero_retries')
    relative = manifest.get('execution_directory')
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError('execution_directory_must_be_frozen_relative_path')
    directory = Path(output_dir).resolve()
    if directory != (Path(manifest_path).parent / relative).resolve() or _credential_path(directory):
        raise ValueError('execution_directory_binding_mismatch')
    declared = {ref['sha256'] for ref in manifest['artifacts']['tools']}
    if any(sha_file(ROOT / name) not in declared for name in IMPLEMENTATIONS):
        raise ValueError('current_execution_implementation_not_frozen')
    prompts = {ref['sha256'] for ref in manifest['artifacts']['prompts']}
    if any(sha_file(ROOT / name) not in prompts for name in PROMPTS):
        raise ValueError('current_prompt_not_frozen')
    if manifest['artifacts']['dependency_lock']['sha256'] != sha_file(ROOT / 'uv.lock'):
        raise ValueError('current_dependency_lock_not_frozen')
    frozen_schema = read_ref(Path(manifest_path).parent, manifest['artifacts']['schema'])
    for case_ref in manifest['cases']:
        case = read_ref(Path(manifest_path).parent, case_ref)
        actual_schema = case.get('schema') or read_object(ROOT / 'config/schema/S1.0.json')[0]
        if actual_schema != frozen_schema or case.get('schema_version') != frozen_schema.get('schema_version'):
            raise ValueError('case_schema_does_not_match_frozen_schema')
        if raw_case_input(case)['schema'] != frozen_schema:
            raise ValueError('b0_projected_schema_does_not_match_frozen_schema')
    for name, method in manifest['methods'].items():
        runner = 'aml_qc/baseline.py' if name == 'B0' else 'aml_qc/workflow.py'
        if method['runner']['sha256'] != sha_file(ROOT / runner) or method['generation'] != GENERATION:
            raise ValueError('unsupported_or_changed_runner_configuration')
        if method['budget']['max_calls'] != (1 if name == 'B0' else 6) or method['budget']['max_output_tokens'] != 4096:
            raise ValueError('unsupported_call_or_output_limit')
        if live and method['model'] != 'deepseek-flash':
            raise ValueError('live_model_is_not_verified_deepseek_flash')
    pricing = read_ref(Path(manifest_path).parent, manifest['pricing'])
    if check_price_window:
        verify_pricing_window(pricing, live=live)
    return manifest, sha, frozen['planned_runs'], directory, pricing


def run_binding(plan, manifest, sha):
    case = next(c for c in manifest['cases'] if c['case_id'] == plan['case_id'])
    return {'planned_run_id': plan['planned_run_id'], 'manifest_sha256': sha,
            'case_sha256': case['sha256'], 'method_config_sha256': digest(manifest['methods'][plan['method']])}


def interrupted_raw(plan, manifest, binding, events):
    """A failed execution envelope, never a fabricated successful model result."""
    calls, starts = {}, {}
    for event in events:
        if event.get('run_id') != plan['planned_run_id']:
            continue
        if event['event'] == 'call_reserved':
            starts[event['call_id']] = event
        elif event['event'] == 'call_finished':
            calls[event['call_id']] = event['record']
        elif event['event'] == 'call_blocked':
            starts[event['call_id']] = event
            calls[event['call_id']] = event['record']
    records = [calls.get(cid, {'request': start.get('request'), 'request_hash': start.get('request_hash'),
        'status': 'interrupted', 'dispatch_status': 'possibly_sent', 'usage': None, 'budget_event_id': cid}) for cid, start in starts.items()]
    # Even a crash before a durable reservation is not treated as a successful
    # run or retried. Billing is unknown unless the journal proves otherwise.
    return {'case_id': plan['case_id'], 'mode': plan['method'].lower(), 'run_status': 'failed',
        'runner_failure': 'interrupted_attempt_not_retried', 'evaluation_execution': binding,
        'execution': {'model': manifest['methods'][plan['method']]['model'], **GENERATION},
        'call_records' if plan['method'] == 'B0' else 'model_requests': records,
        'stats': {'model_calls': len(records)}}


def index_row(plan, binding, raw_path, directory, raw_sha):
    return {'planned_run_id': plan['planned_run_id'], 'case_sha256': binding['case_sha256'],
            'method_config_sha256': binding['method_config_sha256'],
            'raw_result': {'path': os.path.relpath(raw_path, directory), 'sha256': raw_sha}}


def validate_call_journal(events, plans, manifest, directory):
    """Cross-check replay inputs against the freeze and independently saved raw."""
    planned = {plan['planned_run_id']: plan for plan in plans}
    calls = {}
    for event in events:
        kind, run_id, call_id = event['event'], event.get('run_id'), event.get('call_id')
        if not kind.startswith('call_'):
            continue
        if (kind not in {'call_reserved', 'call_finished', 'call_blocked', 'call_audit_failed'}
                or run_id not in planned or not isinstance(call_id, str) or not call_id):
            raise ValueError('unplanned_or_invalid_call_event')
        call = calls.setdefault(call_id, {'run_id': run_id})
        if call['run_id'] != run_id or kind in call:
            raise ValueError('duplicate_or_reassigned_call_event')
        if kind == 'call_reserved':
            budget = manifest['methods'][planned[run_id]['method']]['budget']
            if event.get('budget') != budget:
                raise ValueError('call_event_budget_differs_from_freeze')
            if event.get('request_hash') != digest(event.get('request')):
                raise ValueError('call_event_request_hash_mismatch')
        elif kind in {'call_finished', 'call_blocked'}:
            record = event.get('record', {})
            if record.get('budget_event_id') != call_id or record.get('call_id') != call_id:
                raise ValueError('call_event_record_identity_mismatch')
            if record.get('request_hash') != digest(record.get('request')):
                raise ValueError('call_event_request_hash_mismatch')
        call[kind] = event
    for call in calls.values():
        reserved, finished, blocked = (call.get(name) for name in ('call_reserved', 'call_finished', 'call_blocked'))
        if ((finished or call.get('call_audit_failed')) and not reserved) or (finished and blocked):
            raise ValueError('inconsistent_call_event_lifecycle')
        terminal = finished or blocked
        if reserved and terminal and reserved['request_hash'] != terminal['record']['request_hash']:
            raise ValueError('call_event_request_changed')
    # Validate every saved raw before permitting ANY new dispatch, including
    # later plans whose files would otherwise only be inspected after dispatch.
    for run_id, filename in ((rid, name) for rid in planned for name in ('raw.json', 'interrupted.json')):
        plan = planned[run_id]
        path = directory / 'run' / run_id / filename
        if not path.exists():
            continue
        raw = read_object(path)[0]
        records = raw.get('call_records' if plan['method'] == 'B0' else 'model_requests', [])
        by_id = unique_rows(records, 'budget_event_id')
        expected = {cid: call for cid, call in calls.items() if call['run_id'] == run_id}
        if by_id.keys() != expected.keys():
            raise ValueError('raw_call_ids_differ_from_journal')
        for cid, record in by_id.items():
            call = expected[cid]
            terminal = call.get('call_finished') or call.get('call_blocked')
            if terminal is None:
                if (filename == 'interrupted.json' and call.get('call_reserved')
                        and record.get('dispatch_status') == 'possibly_sent' and record.get('usage') is None
                        and record.get('request_hash') == call['call_reserved']['request_hash']):
                    continue  # An interrupted request has no trustworthy usage yet.
                if call.get('call_audit_failed') and record.get('dispatch_status') == 'sent':
                    continue  # No durable finish: restore keeps this money held.
                raise ValueError('raw_call_missing_durable_outcome')
            recorded = terminal['record']
            if any(record.get(key) != recorded.get(key) for key in
                   ('dispatch_status', 'usage', 'response', 'model_returned', 'finish_reason')):
                raise ValueError('raw_call_differs_from_journal')
            request_hash = record.get('provider_request_hash') if plan['method'] == 'B0' and filename == 'raw.json' else record.get('request_hash')
            if request_hash != recorded['request_hash']:
                raise ValueError('raw_call_request_differs_from_journal')


def run_evaluation(manifest_path, output_dir, *, execute=False, model_factory=None):
    report = {'status': 'blocked', 'errors': [], 'planned_runs': [], 'simulation_only': model_factory is not None}
    try:
        live = model_factory is None
        recovering = (Path(output_dir) / 'plan.json').exists()
        manifest, sha, plans, directory, pricing = preflight(manifest_path, output_dir, live=live,
                                                            check_price_window=not recovering)
        report.update(manifest_sha256=sha, planned_runs=plans, index_path=str(directory / 'runs.json'),
                      journal_path=str(directory / 'journal.jsonl'))
        if not execute:
            report['status'] = 'planned'
            return report
        directory.mkdir(parents=True, exist_ok=True)
        with single_writer(directory):
            plan_record = {'contract_version': 'evaluation-execution-plan-1', 'manifest_sha256': sha,
                           'planned_runs': plans, 'simulation_only': not live, 'frozen_manifest': manifest}
            plan_path = directory / 'plan.json'
            if plan_path.exists():
                if read_object(plan_path)[0] != plan_record:
                    raise ValueError('existing_execution_plan_differs')
            else:
                if any(directory.iterdir()) and set(p.name for p in directory.iterdir()) != {'.runner.lock'}:
                    raise ValueError('unrecognized_execution_directory_contents')
                write_json(plan_path, plan_record)
            journal = Journal(directory / 'journal.jsonl', sha)
            validate_call_journal(journal.events, plans, manifest, directory)
            ledger = BudgetLedger(manifest['total_currency_budget'], pricing)
            ledger.restore([r for r in journal.events if r['event'].startswith('call_')])
            index_path = directory / 'runs.json'
            index = {'contract_version': 'evaluation-runs-1', 'manifest_sha256': sha, 'runs': []}
            known = {}
            if index_path.exists():
                stored = read_object(index_path)[0]
                if stored.get('manifest_sha256') != sha or stored.get('contract_version') != index['contract_version']:
                    raise ValueError('saved_index_binding_mismatch')
                known = unique_rows(stored.get('runs'), 'planned_run_id')
                if set(known) - {p['planned_run_id'] for p in plans}:
                    raise ValueError('unplanned_saved_run')
                for row in known.values():
                    read_ref(directory, row['raw_result'])
            # Rebuild only this derived index from committed artifacts. Starting
            # markers and raw files are never reset to allow another attempt.
            stopped = ledger.stopped
            for plan in plans:
                run_dir = directory / 'run' / plan['planned_run_id']
                marker, raw_path = run_dir / 'started.json', run_dir / 'raw.json'
                failure_path = run_dir / 'interrupted.json'
                binding = run_binding(plan, manifest, sha)
                if raw_path.exists():
                    if not marker.exists() or read_object(marker)[0].get('binding') != binding:
                        raise ValueError('raw_without_bound_attempt_marker')
                    raw, raw_sha = read_object(raw_path)
                    if raw.get('evaluation_execution') != binding:
                        raise ValueError('raw_execution_binding_mismatch')
                    expected = [event['raw_sha256'] for event in journal.events if event.get('run_id') == plan['planned_run_id']
                                and event['event'] in {'raw_prepared', 'raw_committed'}]
                    if not expected or any(value != raw_sha for value in expected):
                        raise ValueError('committed_raw_hash_mismatch')
                    if raw.get('runner_failure'):
                        stopped = True
                elif marker.exists():
                    if read_object(marker)[0].get('binding') != binding:
                        raise ValueError('started_attempt_binding_mismatch')
                    raw_path = failure_path
                    if failure_path.exists():
                        raw, raw_sha = read_object(failure_path)
                        if raw.get('evaluation_execution') != binding:
                            raise ValueError('interrupted_execution_binding_mismatch')
                        expected = [event['raw_sha256'] for event in journal.events if event.get('run_id') == plan['planned_run_id']
                                    and event['event'] == 'run_interrupted_recovered']
                        if not expected or any(value != raw_sha for value in expected):
                            raise ValueError('interrupted_raw_hash_mismatch')
                    else:
                        raw = interrupted_raw(plan, manifest, binding, journal.events)
                        raw_sha = write_json(raw_path, raw)
                        journal.append({'event': 'run_interrupted_recovered', 'run_id': plan['planned_run_id'], 'raw_sha256': raw_sha})
                    stopped = True
                elif stopped:
                    continue
                else:
                    # Revalidate before every new run as case/prompt files may
                    # change while a long experiment is in progress.
                    checked = preflight(manifest_path, output_dir, live=live)
                    if checked[1] != sha:
                        raise ValueError('manifest_changed_during_execution')
                    case_ref = next(c for c in manifest['cases'] if c['case_id'] == plan['case_id'])
                    case = read_ref(Path(manifest_path).parent, case_ref)
                    if case.get('claim_amendments'):
                        raise ValueError('human_claim_amendments_forbidden_in_machine_comparison')
                    run_dir.mkdir(parents=True, exist_ok=False)
                    write_json(marker, {'binding': binding, 'started_at': timestamp(), 'single_attempt': True})
                    journal.append({'event': 'run_started', 'run_id': plan['planned_run_id']})
                    placeholder = interrupted_raw(plan, manifest, binding, [])
                    placeholder['runner_failure'] = 'attempt_in_progress'
                    placeholder_path = run_dir / 'attempt.json'
                    placeholder_sha = write_json(placeholder_path, placeholder)
                    provisional = {**index, 'runs': index['runs'] + [index_row(plan, binding, placeholder_path, directory, placeholder_sha)]}
                    write_json(index_path, provisional, replace=True)
                    method = manifest['methods'][plan['method']]
                    try:
                        base = model_factory(plan, method) if model_factory is not None else DeepSeek(max_calls=method['budget']['max_calls'])
                        if getattr(base, 'model', None) != method['model']:
                            raise ValueError('configured_model_differs_from_frozen_model')
                        if live and base.config.get('DEEPSEEK_BASE_URL', '').rstrip('/') not in {
                                'https://api.deepseek.com', 'https://api.deepseek.com/v1'}:
                            raise ValueError('unverified_live_endpoint')
                        def sink(event):
                            if event['event'] == 'call_reserved':
                                checked = preflight(manifest_path, output_dir, live=live)
                                if checked[1] != sha:
                                    raise ValueError('manifest_changed_during_execution')
                            journal.append(event)
                        wrapped = BudgetedModel(base, ledger, plan['planned_run_id'], method['budget'], sink)
                        wrapped.execution_budget_spec = {'method_budget': method['budget'],
                            'pricing_hash': ledger.pricing_hash, 'context_reservation_bound': CONTEXT_BOUND,
                            'output_bound': OUTPUT_BOUND, 'initial_shared_ledger': ledger.snapshot(),
                            'implementation_sha256': sha_file(ROOT / 'aml_qc/evaluation_budget.py')}
                        provider = 'deepseek' if live else 'frozen'
                        if plan['method'] == 'B0':
                            raw = run_baseline(case, provider=provider, model=wrapped)
                        else:
                            raw = run_review(case, mode=plan['method'].lower(), provider=provider,
                                             strategy='full', previous=None, model=wrapped)
                        raw['evaluation_execution'] = binding
                        raw['evaluation_budget'] = ledger.snapshot()
                        expected_sha = hashlib.sha256((json.dumps(raw, ensure_ascii=False, sort_keys=True,
                            indent=2, allow_nan=False) + '\n').encode()).hexdigest()
                        journal.append({'event': 'raw_prepared', 'run_id': plan['planned_run_id'], 'raw_sha256': expected_sha})
                        raw_sha = write_json(raw_path, raw)
                        journal.append({'event': 'raw_committed', 'run_id': plan['planned_run_id'], 'raw_sha256': raw_sha})
                    except BaseException as error:
                        try:
                            journal.append({'event': 'runner_interrupted', 'run_id': plan['planned_run_id'], 'error_type': type(error).__name__})
                        except OSError:
                            pass
                        raise
                    stopped = ledger.stopped
                index['runs'].append(index_row(plan, binding, raw_path, directory, raw_sha))
                write_json(index_path, index, replace=True)
            if not index_path.exists():
                write_json(index_path, index)
            report.update(status='stopped' if stopped else 'completed', budget=ledger.snapshot(),
                          saved_runs=len(index['runs']), not_run=len(plans) - len(index['runs']))
    except (ValueError, TypeError, KeyError, OSError) as error:
        report['errors'].append(str(error) if isinstance(error, ValueError) else type(error).__name__)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--execute', action='store_true', help='Explicitly dispatch paid DeepSeek requests after preflight.')
    args = parser.parse_args()
    report = run_evaluation(args.manifest, args.output_dir, execute=args.execute)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return int(report['status'] == 'blocked')


if __name__ == '__main__':
    raise SystemExit(main())
