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
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import fcntl
import hashlib
from html import unescape
import json
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import urlsplit

from aml_qc.baseline import raw_case_input, run_baseline
from aml_qc.depgraph import digest
from aml_qc.evaluation_budget import CONTEXT_BOUND, OUTPUT_BOUND, BudgetLedger, BudgetedModel
from aml_qc.llm import GENERATION, REQUEST_TIMEOUT_SECONDS, DeepSeek
from aml_qc.workflow import run_review
from scripts.score_evaluation import read_object, read_ref, unique_rows, verify_call_configuration
from scripts.validate_evaluation import _credential_path, _read_json, _timestamp, validate_manifest


ROOT = Path(__file__).resolve().parents[1]
IMPLEMENTATIONS = [f'aml_qc/{name}.py' for name in ('core', 'ingest', 'schema', 'contracts', 'llm', 'depgraph',
    'workflow', 'claim_edits', 'leads', 'baseline', 'scoring_inputs', 'evaluation_budget')]
IMPLEMENTATIONS += ['scripts/run_evaluation.py', 'scripts/score_evaluation.py', 'scripts/validate_evaluation.py']
PROMPTS = [f'config/prompts/v3.1/{name}.txt' for name in ('agent', 'base', 'extraction', 'response', 'semantic')]
PROMPTS += ['config/prompts/b0-v1/direct.txt']


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


def _verified_holidays(pricing, year):
    proof = pricing.get('holiday_verification')
    if proof is None:
        return set()
    try:
        if (not isinstance(proof, dict) or proof.get('contract') != 'official-holiday-tariff-evidence-1'
                or type(proof.get('year')) is not int or proof['year'] != year
                or proof.get('model') != 'deepseek-flash' or not isinstance(proof.get('dates'), list)
                or not proof['dates'] or pricing.get('currency') != 'CNY'
                or pricing.get('unit_tokens') != 1000000 or pricing.get('period') != 'off_peak'):
            raise ValueError('invalid holiday declaration')

        def plain(value):
            value = re.sub(r'<(?:script|style)\b[^>]*>.*?</(?:script|style)>', '', value, flags=re.I | re.S)
            return re.sub(r'\s+', '', unescape(re.sub(r'<[^>]*>', '', value)))

        def capture(name):
            item = proof[name]; relative = Path(item['path']); url = urlsplit(item['url'])
            if (relative.is_absolute() or '..' in relative.parts or relative.suffix.lower() not in {'.html', '.htm'}
                    or any(part.startswith('.env') for part in relative.parts)
                    or any((ROOT / Path(*relative.parts[:n])).is_symlink() for n in range(1, len(relative.parts) + 1))
                    or url.scheme != 'https' or url.username or url.password or url.query or url.fragment
                    or not re.fullmatch(r'[a-f0-9]{64}', item['sha256'])):
                raise ValueError('unsafe holiday source capture')
            if name == 'government_capture':
                if url.hostname != 'www.gov.cn' or not url.path.startswith(('/gongbao/', '/zhengce/')):
                    raise ValueError('unverified holiday government source')
            elif item['url'] != 'https://api-docs.deepseek.com/zh-cn/quick_start/pricing/' or item['url'] != pricing.get('source_url'):
                raise ValueError('unverified holiday tariff source')
            raw = (ROOT / relative).read_bytes()
            if hashlib.sha256(raw).hexdigest() != item['sha256']:
                raise ValueError('holiday source capture changed')
            return raw.decode('utf-8')

        government = plain(capture('government_capture'))
        if f'国务院办公厅关于{year}年部分节假日安排的通知' not in government:
            raise ValueError('holiday notice year or title mismatch')
        calendar = set()
        pattern = r'(\d{1,2})月(\d{1,2})日(?:（[^）]*）)?至(?:(\d{1,2})月)?(\d{1,2})日(?:（[^）]*）)?放假'
        for month, first, last_month, last in re.findall(pattern, government):
            start = date(year, int(month), int(first)); end = date(year, int(last_month or month), int(last))
            if end < start:
                raise ValueError('unverified holiday range')
            calendar.update((start + timedelta(days=n)).isoformat() for n in range((end - start).days + 1))
        holidays = set(proof['dates'])
        if len(holidays) != len(proof['dates']) or not holidays <= calendar:
            raise ValueError('declared holiday date not in official notice')
        official = capture('pricing_capture')
        if '中国法定节假日全天均为空闲时段' not in plain(official):
            raise ValueError('holiday off-peak tariff not verified')
        rows = [[plain(cell) for cell in re.findall(r'<t[dh]\b[^>]*>(.*?)</t[dh]>', row, re.I | re.S)]
                for row in re.findall(r'<tr\b[^>]*>(.*?)</tr>', official, re.I | re.S)]
        header = next(row for row in rows if row and row[0] == '模型'
                      and any(re.fullmatch(r'deepseek-flash(?:\(\d+\))?', value) for value in row))
        column = next(i for i, value in enumerate(header)
                      if re.fullmatch(r'deepseek-flash(?:\(\d+\))?', value)) - len(header)
        rates, category = {}, None
        for row in rows:
            text = ''.join(row)
            for label, key in [('缓存命中', 'input_cache_hit'), ('缓存未命中', 'input_cache_miss'), ('百万tokens输出', 'output')]:
                if label in text:
                    category = key
            if '空闲时段' in row and category is not None:
                value = row[column]
                if not re.fullmatch(r'\d+(?:\.\d+)?元', value):
                    raise ValueError('unverified holiday price cell')
                rates[category] = Decimal(value.removesuffix('元'))
        if rates != {k: Decimal(str(v)) for k, v in pricing['rates'].items()}:
            raise ValueError('holiday tariff differs from frozen rates')
        return holidays
    except (KeyError, IndexError, TypeError, ValueError, OSError, StopIteration, ArithmeticError) as error:
        raise ValueError('holiday_evidence_invalid:' + str(error)) from error


def verify_pricing_window(pricing, *, live):
    deadline = pricing.get('valid_until')
    if not _timestamp(deadline) or now() + timedelta(seconds=REQUEST_TIMEOUT_SECONDS) >= datetime.fromisoformat(deadline):
        raise ValueError('pricing_expired_or_too_near_expiry')
    if live:
        current = now().astimezone(timezone(timedelta(hours=8)))
        holidays = _verified_holidays(pricing, current.year)
        # The official weekday peak schedule excludes Chinese statutory
        # holidays. Without a verified calendar, that interval is unknown;
        # do not silently treat every weekday as a non-holiday.
        if current.weekday() < 5 and (9 <= current.hour < 12 or 14 <= current.hour < 18) and current.date().isoformat() not in holidays:
            raise ValueError('weekday_peak_requires_verified_holiday_calendar')
        if pricing.get('period') != 'off_peak':
            raise ValueError('frozen_pricing_period_is_not_current')
        # Do not let a request enter an interval whose tariff is unknown.
        later = current + timedelta(seconds=REQUEST_TIMEOUT_SECONDS)
        if later.weekday() < 5 and (9 <= later.hour < 12 or 14 <= later.hour < 18) and later.date().isoformat() not in holidays:
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
        if method['budget']['max_calls'] != (1 if name == 'B0' else 6) or method['budget']['max_output_tokens'] != GENERATION['max_output_tokens']:
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
            verify_call_configuration([event], manifest['methods'][planned[run_id]['method']])
        elif kind in {'call_finished', 'call_blocked'}:
            record = event.get('record', {})
            if record.get('budget_event_id') != call_id or record.get('call_id') != call_id:
                raise ValueError('call_event_record_identity_mismatch')
            if record.get('request_hash') != digest(record.get('request')):
                raise ValueError('call_event_request_hash_mismatch')
            verify_call_configuration([record], manifest['methods'][planned[run_id]['method']])
        call[kind] = event
    for call in calls.values():
        reserved, finished, blocked = (call.get(name) for name in ('call_reserved', 'call_finished', 'call_blocked'))
        if ((finished or call.get('call_audit_failed')) and not reserved) or (finished and blocked):
            raise ValueError('inconsistent_call_event_lifecycle')
        terminal = finished or blocked
        if reserved and terminal and reserved['request_hash'] != terminal['record']['request_hash']:
            raise ValueError('call_event_request_changed')
        if reserved and terminal and 'stage' in reserved and reserved['stage'] != terminal['record'].get('stage'):
            raise ValueError('call_event_stage_changed')
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
            if 'stage' in record:
                verify_call_configuration([record], manifest['methods'][plan['method']])
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
            if ('stage' in record or 'stage' in recorded) and record.get('stage') != recorded.get('stage'):
                raise ValueError('raw_call_stage_differs_from_journal')
            if 'stage' in record and record.get('request') != recorded.get('request'):
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
