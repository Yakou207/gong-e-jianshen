"""Score saved machine outputs against a frozen, independently authored reference.

Run with ``python -m scripts.score_evaluation MANIFEST RUNS [--ledger LEDGER]``.
No model, business rule, reference generator or human-corrected deliverable is
executed. Output goes to stdout; redirect to a NEW file to retain earlier scores.

RUNS: {contract_version: evaluation-runs-1, manifest_sha256, runs: [
 {planned_run_id, case_sha256, method_config_sha256, raw_result: {path,sha256}}]}.
Paths are relative to RUNS. Omitted plan rows remain not_run. This first version
supports one attempt per planned run; manifests permitting retries are rejected.

LEDGER: {contract_version: evaluation-matching-1, manifest_sha256, runs: [
 {planned_run_id, raw_sha256, reference_sha256, checks: [...], issues: [...]}]}.
Each decision declares reviewer_id, signed_at, reason, method_blinded: true.
Checks use observation_id (raw JSON pointer), decision matched/duplicate/unrelated/
unresolved, reference_check_id for matched/duplicate, proposition_faithful boolean,
and evidence_support supported/unsupported/unresolved for matched checks. Issues
use prediction_id and the same decisions plus new_issue/rejected; only an explicit
rejected declaration counts a reference-external prediction as false. Matched/duplicate issues
declare reference_check_id and proposition_faithful. Values/evidence are NEVER
supplied by the ledger. The earliest mapped raw row is primary; later rows must
be duplicate. This prevents selecting the most accurate duplicate after scoring.

Reference units may add issue_expectations: [{type, truth: positive/negative/
unresolved}]. No declaration means that unit's issue domain is unavailable, not
negative. Feature labels and issue detection are separate. Pending matches grant
no TP; recall is a lower bound and pending precision bounds are reported. Actual
human identity, independence and semantic decisions are declarations, not proven.
"""
import argparse
from collections import Counter
from decimal import Decimal
import hashlib
import json
from pathlib import Path

from aml_qc.depgraph import digest, sources_for
from aml_qc.schema import LABEL_VALUES
from aml_qc.scoring_inputs import normalize_observations, normalize_issue_predictions, resolve_evidence
from scripts.validate_evaluation import _credential_path, _read_json, _timestamp, validate_manifest


UNCERTAIN = {'undeterminable', 'insufficient_evidence', 'insufficient', 'pending_judgement'}


def read_object(path, expected=None):
    path = Path(path)
    if _credential_path(path) or _credential_path(path.resolve()):
        raise ValueError('credential_file_forbidden')
    raw = path.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if expected is not None and sha != expected:
        raise ValueError('file_hash_mismatch')
    value = _read_json(raw)
    if not isinstance(value, dict):
        raise ValueError('json_object_required')
    return value, sha


def read_ref(base, ref):
    if (not isinstance(ref, dict) or not isinstance(ref.get('path'), str)
            or Path(ref['path']).is_absolute() or not isinstance(ref.get('sha256'), str)):
        raise ValueError('invalid_file_reference')
    return read_object(base / ref['path'], ref['sha256'])[0]


def unique_rows(rows, key):
    if not isinstance(rows, list):
        raise ValueError('row_list_required')
    result = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get(key), str) or not row[key] or row[key] in result:
            raise ValueError('invalid_or_duplicate_' + key)
        result[row[key]] = row
    return result


def fraction(n, d):
    return {'numerator': n, 'denominator': d, 'value': n / d if d else None}


def call_cost(calls, pricing, *, attempted, declared_calls=None, usage_complete=None):
    """Count each recorded call, including identical requests; unknown is not zero."""
    known, unknown, tokens = Decimal(0), 0, 0
    dispatched, not_sent_missing_usage = 0, False
    if not isinstance(calls, list):
        calls = []
    if attempted and not calls:
        unknown = 1
    for call in calls:
        usage = call.get('usage') if isinstance(call, dict) else None
        keys = ('prompt_cache_hit_tokens', 'prompt_cache_miss_tokens', 'completion_tokens', 'prompt_tokens')
        if isinstance(call, dict) and call.get('dispatch_status') == 'not_sent':
            zero_usage = (isinstance(usage, dict) and all(type(usage.get(k)) is int and usage[k] == 0 for k in keys)
                          and ('total_tokens' not in usage or type(usage['total_tokens']) is int and usage['total_tokens'] == 0))
            if call.get('response') is not None or (usage is not None and not zero_usage):
                unknown += 1
            elif usage is None:
                not_sent_missing_usage = True
            continue
        dispatched += 1
        if (not isinstance(usage, dict) or any(type(usage.get(k)) is not int or usage[k] < 0 for k in keys)
                or usage['prompt_cache_hit_tokens'] + usage['prompt_cache_miss_tokens'] != usage['prompt_tokens']
                or ('total_tokens' in usage and (type(usage['total_tokens']) is not int
                    or usage['total_tokens'] != usage['prompt_tokens'] + usage['completion_tokens']))):
            unknown += 1
            continue
        known += sum(Decimal(usage[k]) * Decimal(pricing['rates'][r]) for k, r in zip(keys[:3],
                     ('input_cache_hit', 'input_cache_miss', 'output'))) / Decimal(pricing['unit_tokens'])
        tokens += usage['prompt_tokens'] + usage['completion_tokens']
    if declared_calls is not None and (type(declared_calls) is not int or declared_calls < 0 or declared_calls != len(calls)):
        missing = declared_calls - len(calls) if type(declared_calls) is int else 1
        unknown += max(1, missing)
    if usage_complete is False and not unknown and not not_sent_missing_usage:
        unknown = 1
    return {'known_cost': str(known), 'total_cost': None if unknown else str(known),
            'unknown_usage_calls': unknown, 'recorded_calls': len(calls), 'dispatched_calls': dispatched,
            'known_tokens': tokens, 'cost_status': 'incomplete' if unknown else 'complete'}


def verify_call_configuration(calls, method):
    accepted = method.get('accepted_returned_models', [method['model']])
    if not isinstance(accepted, list) or not accepted or any(not isinstance(v, str) or not v for v in accepted):
        raise ValueError('invalid_returned_model_allowlist')
    for call in calls if isinstance(calls, list) else []:
        if not isinstance(call, dict):
            continue  # Invalid/missing billing records remain unknown in call_cost.
        if call.get('model_returned') is not None and call['model_returned'] not in accepted:
            raise ValueError('returned_model_not_frozen')
        request = call.get('request')
        if request is None:
            if 'stage' in call or call.get('dispatch_status') == 'sent':
                raise ValueError('recorded_request_missing')
            continue
        if not isinstance(request, dict) or request.get('model') != method['model']:
            raise ValueError('recorded_request_model_mismatch')
        if call.get('dispatch_status') == 'sent' and call.get('request_hash') is None:
            raise ValueError('recorded_request_hash_missing')
        if 'request_hash' in call and call['request_hash'] != digest(request):
            raise ValueError('recorded_request_hash_mismatch')
        generation = method['generation']
        stage = call.get('stage')
        if 'stage_max_output_tokens' in generation and stage is None:
            caps, thinking = generation['stage_max_output_tokens'], generation['stage_thinking']
            candidates = [name for name, cap in caps.items()
                          if request.get('max_tokens') == cap
                          and request.get('thinking') == {'type': thinking[name]}
                          and (name == 'tool_review') == bool(request.get('tools'))]
            if len(candidates) != 1:
                raise ValueError('recorded_request_stage_ambiguous')
            stage = candidates[0]
        current_wire = 'stage' in call or 'stage_max_output_tokens' in method['generation']
        if 'provider_request_hash' in call:
            hashes = {row.get('request_hash') for row in call.get('provider_records', []) if isinstance(row, dict)}
            if len(hashes) != 1 or call['provider_request_hash'] not in hashes or None in hashes:
                raise ValueError('recorded_provider_request_hash_mismatch')
        for provider in call.get('provider_records', []) if current_wire else []:
            if not isinstance(provider, dict) or 'provider_records' in provider:
                raise ValueError('recorded_provider_invalid')
            actual = provider.get('request')
            if provider.get('status') == 'frozen' and isinstance(actual, dict) and 'max_tokens' not in actual:
                # FrozenModel uses an exact logical key for mechanism replay.
                logical = {'model': request['model'], 'messages': request['messages'], 'tools': request.get('tools')}
                if 'stage' in actual:
                    logical['stage'] = stage
                if actual != logical or provider.get('request_hash') != digest(logical):
                    raise ValueError('recorded_provider_request_mismatch')
                if (provider.get('stage', stage) != stage
                        or provider.get('generation', generation) != generation):
                    raise ValueError('recorded_request_generation_mismatch')
                continue
            if actual is None and call.get('dispatch_status') != 'sent' and 'stage' not in provider:
                continue  # Legacy/local doubles do not claim a provider wire.
            if actual != request:
                raise ValueError('recorded_provider_request_mismatch')
            verify_call_configuration([provider], method)
        if 'stage_max_output_tokens' in generation:
            caps, thinking = generation['stage_max_output_tokens'], generation['stage_thinking']
            if (not isinstance(stage, str) or stage not in caps
                    or (stage == 'tool_review') != bool(request.get('tools'))
                    or request.get('max_tokens') != caps[stage]
                    or request.get('thinking') != {'type': thinking[stage]}):
                raise ValueError('recorded_request_generation_mismatch')
            if thinking[stage] == 'enabled':
                if request.get('reasoning_effort') != generation['reasoning_effort'] or 'temperature' in request:
                    raise ValueError('recorded_request_generation_mismatch')
            elif request.get('temperature') != generation['temperature'] or 'reasoning_effort' in request:
                raise ValueError('recorded_request_generation_mismatch')
            if call.get('generation', generation) != generation:
                raise ValueError('recorded_request_generation_mismatch')
            if generation.get('final_transport') == 'sse-with-usage-1':
                if stage == 'final':
                    options = request.get('stream_options')
                    if (request.get('stream') is not True or not isinstance(options, dict)
                            or set(options) != {'include_usage'} or options['include_usage'] is not True):
                        raise ValueError('recorded_request_transport_mismatch')
                elif ('stream_options' in request
                      or request.get('stream') is not None and request.get('stream') is not False):
                    raise ValueError('recorded_request_transport_mismatch')
            continue
        for key, expected in method['generation'].items():
            if key == 'thinking' and isinstance(expected, dict):
                expected = expected['with_tools' if request.get('tools') else 'without_tools']
            if (key == 'temperature' and method['generation'].get('temperature_applies_to') == 'non_thinking_only'
                    and request.get('thinking') == {'type': 'enabled'}):
                continue
            actual_key = 'max_tokens' if key == 'max_output_tokens' else key
            if actual_key not in request:
                continue
            actual = request[actual_key]
            if key == 'thinking' and isinstance(actual, dict):
                actual = actual.get('type')
            if actual != expected:
                raise ValueError('recorded_request_generation_mismatch')


def decisions(rows, predictions, units, *, issue=False):
    key = 'prediction_id' if issue else 'observation_id'
    indexed = unique_rows(rows, key)
    known = {r[key] for r in predictions}
    if set(indexed) - known:
        raise ValueError('unknown_prediction_in_ledger')
    seen, first = set(), {}
    for pred in predictions:
        fingerprint = digest([pred['type'], pred.get('observations'), pred.get('local_check_refs'), pred['evidence']]) if issue else digest([
            pred['label'], pred['object_scope'], pred['matching_anchor'], pred['proposition']])
        first.setdefault(fingerprint, pred[key])
        row = indexed.get(pred[key])
        if row is None:
            continue
        if (not isinstance(row.get('reviewer_id'), str) or not row['reviewer_id'].strip()
                or not _timestamp(row.get('signed_at')) or not isinstance(row.get('reason'), str)
                or not row['reason'].strip() or row.get('method_blinded') is not True):
            raise ValueError('incomplete_blinded_review_declaration')
        allowed = {'matched', 'duplicate', 'unrelated', 'unresolved'} | ({'new_issue', 'rejected'} if issue else set())
        if row.get('decision') not in allowed:
            raise ValueError('invalid_matching_decision')
        if row['decision'] in {'matched', 'duplicate'}:
            uid = row.get('reference_check_id')
            if not isinstance(uid, str) or uid not in units or type(row.get('proposition_faithful')) is not bool:
                raise ValueError('invalid_reference_match')
            if not issue and pred['label'] != units[uid]['label']:
                raise ValueError('cross_label_match')
            if not issue:
                for scope_key in ('account_id', 'direction', 'start', 'end', 'counterparty_token', 'counterparty_tokens', 'focus_id', 'material_link_id'):
                    actual, expected = pred['object_scope'].get(scope_key), units[uid]['object_scope'].get(scope_key)
                    if actual is not None and expected is not None and actual != expected:
                        raise ValueError('explicit_scope_conflict')
                for anchor_key in ('document_id', 'revision'):
                    actual = pred['matching_anchor'].get(anchor_key)
                    reference_anchor = units[uid].get('matching_anchor', {})
                    if not isinstance(reference_anchor, dict):
                        raise ValueError('invalid_reference_anchor')
                    expected = reference_anchor.get(anchor_key)
                    if actual is not None and expected is not None and actual != expected:
                        raise ValueError('explicit_anchor_conflict')
            if row['decision'] == 'matched' and first[fingerprint] != pred[key]:
                raise ValueError('cannot_skip_first_structural_duplicate')
            target = (uid, pred['type']) if issue else uid
            expected = 'duplicate' if target in seen else 'matched'
            if row['decision'] != expected:
                raise ValueError('primary_must_be_first_raw_prediction')
            seen.add(target)
            if not issue and row['decision'] == 'matched' and row.get('evidence_support') not in {
                    'supported', 'unsupported', 'unresolved'}:
                raise ValueError('evidence_support_declaration_required')
    return indexed


def score_checks(plan, units, observations, matching, case):
    matched = {r['reference_check_id']: oid for oid, r in matching.items() if r['decision'] == 'matched'}
    obs = {r['observation_id']: r for r in observations}
    scores = []
    for uid, unit in units.items():
        oid = matched.get(uid)
        prediction, decision = obs.get(oid), matching.get(oid, {})
        scorable = unit['applicability'] == 'applicable' and unit['adjudication_status'] == 'adjudicated'
        state = 'missing'
        if prediction:
            has_business_answer = prediction['execution_state'] == 'completed' or (
                prediction['execution_state'] == 'identity_unresolved' and prediction['prediction_value'] in UNCERTAIN)
            state = ('invalid_proposition' if not decision['proposition_faithful'] else
                     'completed' if has_business_answer else 'unfinished')
        value = prediction['prediction_value'] if prediction else None
        correct = scorable and state == 'completed' and value == unit['reference_value']
        evidence = prediction['evidence'] if prediction else []
        resolved = [resolve_evidence(ref, case) for ref in {digest(r): r for r in evidence}.values()]
        signatures = {digest(r) for r in evidence}
        sufficient_set = bool(evidence) and any(bool(group) and {digest(r) for r in group} <= signatures
                                                for group in unit['evidence_sets'])
        support = decision.get('evidence_support', 'unresolved')
        supported = bool(correct and sufficient_set and resolved and all(r is True for r in resolved) and support == 'supported')
        support_unknown = bool(correct and evidence and not any(r is False for r in resolved)
                               and support != 'unsupported' and not supported)
        scores.append({**plan, 'reference_check_id': uid, 'label': unit['label'], 'scorable': scorable,
            'applicability': unit['applicability'], 'adjudication_status': unit['adjudication_status'],
            'reference_value': unit['reference_value'], 'observation_id': oid, 'state': state,
            'execution_state': prediction['execution_state'] if prediction else 'not_run',
            'prediction_value': value, 'correct': bool(correct),
            'business_abstention': state == 'completed' and value in UNCERTAIN,
            'reference_requires_abstention': scorable and unit['reference_value'] in UNCERTAIN,
            'evidence_missing': not evidence, 'reference_evidence_set_satisfied': sufficient_set,
            'citation_counts': {'valid': resolved.count(True), 'invalid': resolved.count(False),
                                'unchecked': sum(r is None for r in resolved)},
            'semantic_support_declaration': support,
            'evidence_support_status': 'supported' if supported else 'unknown' if support_unknown else 'unsupported',
            'evidence_supported': supported})
    return scores


def score_issues(plan, units, predictions, matching, check_matching):
    expectations, output = {}, []
    for uid, unit in units.items():
        for expectation in unit.get('issue_expectations', []):
            truth = expectation['truth']
            if unit['applicability'] != 'applicable' or unit['adjudication_status'] != 'adjudicated':
                truth = 'unresolved'
            expectations[(uid, expectation['type'])] = truth
    hit = set()
    for prediction in predictions:
        pid = prediction['prediction_id']
        decision = matching.get(pid, {})
        target = (decision.get('reference_check_id'), prediction['type'])
        outcome = 'pending'
        faithful = decision.get('proposition_faithful') is True and all(
            check_matching.get(oid, {}).get('proposition_faithful') is not False
            for oid in prediction.get('observations', []))
        if decision.get('decision') == 'rejected':
            outcome = 'fp'
        elif decision.get('decision') in {'matched', 'duplicate'} and expectations.get(target) in {'positive', 'negative'}:
            if decision['decision'] == 'duplicate':
                outcome = 'duplicate_fp'
            elif expectations[target] == 'positive' and faithful:
                outcome = 'tp'
                hit.add(target)
            else:
                outcome = 'fp'
        output.append({**plan, **prediction, 'reference_check_id': decision.get('reference_check_id'),
                       'outcome': outcome, 'matching_decision': decision.get('decision', 'unresolved')})
    for target, truth in expectations.items():
        if truth == 'positive' and target not in hit:
            output.append({**plan, 'prediction_id': None, 'reference_check_id': target[0],
                           'type': target[1], 'outcome': 'fn'})
        elif truth == 'negative' and not any(r.get('reference_check_id') == target[0] and r['type'] == target[1]
                                             and r['outcome'] in {'fp', 'duplicate_fp'} for r in output):
            undecided = any(r['type'] == target[1] and r['outcome'] == 'pending' for r in output)
            output.append({**plan, 'prediction_id': None, 'reference_check_id': target[0],
                           'type': target[1], 'outcome': 'negative_pending' if undecided else 'tn'})
    return output, sum(v == 'unresolved' for v in expectations.values())


def summarize(checks, issues, runs):
    summaries = []
    for method, split in sorted({(r['method'], r['split']) for r in runs}):
        selected = [r for r in checks if r['method'] == method and r['split'] == split]
        scored = [r for r in selected if r['scorable']]
        done = [r for r in scored if r['state'] == 'completed']
        actual = [r for r in runs if r['method'] == method and r['split'] == split]
        by_label = {}
        for label in sorted({r['label'] for r in selected}):
            rows = [r for r in scored if r['label'] == label]
            confusion = Counter((r['reference_value'], r['prediction_value'] if r['state'] == 'completed'
                                 else '__' + r['state'] + '__') for r in rows)
            by_label[label] = {'accuracy': fraction(sum(r['correct'] for r in rows), len(rows)),
                'confusion': [{'reference': k[0], 'prediction': k[1], 'count': v} for k, v in sorted(confusion.items())]}
        by_issue = {}
        types = {r['type'] for r in issues if r['method'] == method and r['split'] == split}
        for kind in sorted(types):
            counts = Counter(r['outcome'] for r in issues if r['method'] == method and r['split'] == split and r['type'] == kind)
            tp, fp, pending = counts['tp'], counts['fp'] + counts['duplicate_fp'], counts['pending']
            by_issue[kind] = {**dict(counts), 'precision_adjudicated': fraction(tp, tp + fp),
                'precision_bounds': [fraction(tp, tp + fp + pending)['value'],
                                     fraction(tp + pending, tp + fp + pending)['value']],
                'recall_lower_bound': fraction(tp, tp + counts['fn'])}
        unknown = sum(r['cost']['unknown_usage_calls'] for r in actual)
        known = sum((Decimal(r['cost']['known_cost']) for r in actual), Decimal(0))
        answered = [r for r in done if not r['business_abstention']]
        must_abstain = [r for r in scored if r['reference_requires_abstention']]
        summaries.append({'method': method, 'split': split, 'planned_runs': len(actual),
            'run_status_counts': dict(Counter(r['run_status'] for r in actual)),
            'reference_units': len(selected), 'scorable_units': len(scored),
            'unresolved_reference_units': sum(r['adjudication_status'] == 'unresolved' or r['applicability'] == 'unresolved' for r in selected),
            'not_applicable_units': sum(r['applicability'] == 'not_applicable' for r in selected),
            'output_completion': fraction(len(done), len(scored)),
            'execution_completion': fraction(sum(r['execution_state'] == 'completed' for r in scored), len(scored)),
            'accuracy': fraction(sum(r['correct'] for r in scored), len(scored)),
            'determinate_coverage': fraction(len(answered), len(scored)),
            'accuracy_when_determinate': fraction(sum(r['correct'] for r in answered), len(answered)),
            'correct_abstention': fraction(sum(r['correct'] for r in must_abstain), len(must_abstain)),
            'false_certainty': fraction(sum(r['state'] == 'completed' and not r['business_abstention'] for r in must_abstain), len(must_abstain)),
            'evidence_supported_lower_bound': fraction(sum(r['evidence_supported'] for r in scored), len(scored)),
            'evidence_support_bounds': [fraction(sum(r['evidence_supported'] for r in scored), len(scored))['value'],
                fraction(sum(r['evidence_support_status'] in {'supported', 'unknown'} for r in scored), len(scored))['value']],
            'pending_evidence_support': sum(r['evidence_support_status'] == 'unknown' for r in scored),
            'by_label': by_label, 'by_issue': by_issue, 'known_cost': str(known),
            'total_cost': None if unknown else str(known), 'unknown_usage_calls': unknown,
            'pending_matching': sum(r['pending_matching'] for r in actual),
            'duplicate_checks': sum(r['duplicate_checks'] for r in actual),
            'issue_domain_unavailable_units': sum(r['issue_domain_unavailable_units'] for r in actual)})
    return summaries


def score_evaluation(manifest_path, index_path, ledger_path=None):
    report = {'contract_version': 'evaluation-score-1', 'status': 'blocked', 'errors': [],
              'check_scores': [], 'issue_scores': [], 'runs': [], 'summaries': [], 'observations': []}
    try:
        freeze = validate_manifest(manifest_path)
        if not freeze['frozen_valid']:
            raise ValueError('manifest_not_frozen_valid')
        manifest, sha = read_object(manifest_path)
        base, ibase = Path(manifest_path).parent, Path(index_path).parent
        index, index_sha = read_object(index_path)
        if index.get('contract_version') != 'evaluation-runs-1' or index.get('manifest_sha256') != sha:
            raise ValueError('runs_index_not_bound_to_manifest')
        if manifest['retry_policy']['max_retries'] != 0:
            raise ValueError('scorer_v1_requires_zero_retries')
        indexed = unique_rows(index.get('runs'), 'planned_run_id')
        plan_ids = {r['planned_run_id'] for r in freeze['planned_runs']}
        if set(indexed) - plan_ids:
            raise ValueError('unplanned_run')
        ledgers, ledger_sha = {}, None
        if ledger_path:
            ledger, ledger_sha = read_object(ledger_path)
            if ledger.get('contract_version') != 'evaluation-matching-1' or ledger.get('manifest_sha256') != sha:
                raise ValueError('ledger_not_bound_to_manifest')
            ledgers = unique_rows(ledger.get('runs'), 'planned_run_id')
            if set(ledgers) - plan_ids:
                raise ValueError('unplanned_ledger_run')
        cases = {r['case_id']: r for r in manifest['cases']}
        refs = {r['case_id']: r for r in manifest['references']}
        pricing = read_ref(base, manifest['pricing'])
        schema = read_ref(base, manifest['artifacts']['schema'])
        implementation = {'scorer': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'adapter': hashlib.sha256((Path(__file__).parents[1] / 'aml_qc/scoring_inputs.py').read_bytes()).hexdigest()}
        if manifest['artifacts']['scorer']['sha256'] != implementation['scorer']:
            raise ValueError('scorer_differs_from_frozen_implementation')
        if implementation['adapter'] not in {ref['sha256'] for ref in manifest['artifacts']['tools']}:
            raise ValueError('adapter_not_frozen')
        report.update(manifest_sha256=sha, runs_index_sha256=index_sha, matching_ledger_sha256=ledger_sha,
                      implementation_sha256=implementation, currency=pricing['currency'])
        report['matching_template'] = {'contract_version': 'evaluation-matching-1', 'manifest_sha256': sha, 'runs': []}
        for plan in freeze['planned_runs']:
            plan = {k: v for k, v in plan.items() if k != 'status'}
            pid, cid = plan['planned_run_id'], plan['case_id']
            case = read_ref(base, cases[cid])
            if case.get('schema') is not None and case['schema'] != schema:
                raise ValueError('case_schema_differs_from_frozen_schema')
            reference = read_ref(base, refs[cid])
            units = unique_rows(reference['check_units'], 'reference_check_id')
            for unit in units.values():
                if (unit['adjudication_status'] == 'adjudicated' and unit['applicability'] == 'applicable'
                        and unit['reference_value'] not in LABEL_VALUES.get(unit['label'], [])):
                    raise ValueError('reference_value_not_scoreable')
                for expectation in unique_rows(unit.get('issue_expectations', []), 'type').values():
                    if expectation.get('truth') not in {'positive', 'negative', 'unresolved'}:
                        raise ValueError('invalid_issue_expectation')
            row, raw, raw_sha = indexed.get(pid), None, None
            state, errors, obs, predictions = 'not_run', [], [], []
            cost = call_cost([], pricing, attempted=False)
            if row is not None:
                cost = call_cost([], pricing, attempted=True)
                try:
                    method = manifest['methods'][plan['method']]
                    if row.get('case_sha256') != cases[cid]['sha256'] or row.get('method_config_sha256') != digest(method):
                        raise ValueError('run_case_or_method_binding_mismatch')
                    raw = read_ref(ibase, row.get('raw_result'))
                    raw_sha = row['raw_result']['sha256']
                    calls = raw.get('call_records') if plan['method'] == 'B0' else raw.get('model_requests')
                    stats = raw.get('stats', {})
                    if not isinstance(stats, dict):
                        raise ValueError('invalid_call_stats')
                    cost = call_cost(calls, pricing, attempted=True, declared_calls=stats.get('model_calls'),
                                     usage_complete=stats.get('usage_complete'))
                    verify_call_configuration(calls, method)
                    execution = raw.get('execution', {})
                    if (not isinstance(execution, dict) or raw.get('case_id') != cid or raw.get('mode') != plan['method'].lower()
                            or execution.get('model') != method['model']
                            or any(execution.get(k) != v for k, v in method['generation'].items())):
                        raise ValueError('raw_case_method_or_generation_mismatch')
                    if plan['method'] != 'B0':
                        if raw.get('strategy') != 'full':
                            raise ValueError('quality_comparison_requires_full_run')
                        if not isinstance(raw.get('snapshot'), dict) or raw['snapshot'].get('sources') != sources_for(case, case.get('schema') or schema, execution):
                            raise ValueError('raw_snapshot_does_not_match_case')
                    else:
                        from aml_qc.baseline import PROMPT_PATH, incomplete_targets, parse_output, raw_case_input
                        baseline_hash = hashlib.sha256((Path(__file__).parents[1] / 'aml_qc/baseline.py').read_bytes()).hexdigest()
                        if (method['runner']['sha256'] != baseline_hash or execution.get('implementation_hash') != baseline_hash
                                or execution.get('prompt_file_sha256') != hashlib.sha256(PROMPT_PATH.read_bytes()).hexdigest()
                                or execution.get('prompt_file_sha256') not in {r['sha256'] for r in manifest['artifacts']['prompts']}):
                            raise ValueError('b0_runner_or_prompt_not_frozen')
                        if raw.get('input_projection') != raw_case_input({**case, 'schema': schema}):
                            raise ValueError('b0_input_does_not_match_case')
                        if raw.get('parsed_output') is not None:
                            if parse_output(raw.get('raw_response')) != raw['parsed_output']:
                                raise ValueError('b0_parse_not_bound_to_raw_response')
                            expected_state = 'partial' if incomplete_targets(raw['input_projection'], raw['parsed_output']) else 'completed'
                            if raw.get('run_status') != expected_state:
                                raise ValueError('b0_status_inconsistent_with_parsed_output')
                        for call in calls if isinstance(calls, list) else []:
                            if isinstance(call, dict) and 'response' in call and call['response'] != raw.get('raw_response'):
                                raise ValueError('b0_call_response_mismatch')
                    state = raw.get('run_status')
                    if state not in {'completed', 'partial', 'failed'}:
                        raise ValueError('invalid_run_status')
                    obs = normalize_observations(raw, plan['method'], case)
                    predictions = normalize_issue_predictions(raw, plan['method'])
                except (ValueError, TypeError, KeyError, OSError, UnicodeError) as error:
                    errors.append(str(error) if isinstance(error, ValueError) else 'raw_result_unreadable')
                    state, obs, predictions = 'invalid', [], []
                    if raw is None:
                        cost = call_cost([], pricing, attempted=True)
            cm, im = {}, {}
            if pid in ledgers:
                try:
                    entry = ledgers[pid]
                    if raw_sha is None or entry.get('raw_sha256') != raw_sha or entry.get('reference_sha256') != refs[cid]['sha256']:
                        raise ValueError('ledger_raw_or_reference_hash_mismatch')
                    cm = decisions(entry.get('checks'), obs, units)
                    im = decisions(entry.get('issues'), predictions, units, issue=True)
                except (ValueError, TypeError, KeyError) as error:
                    errors.append(str(error))
                    cm, im = {}, {}
            check_scores = score_checks(plan, units, obs, cm, case)
            issue_scores, unresolved_issues = score_issues(plan, units, predictions, im, cm)
            pending = sum(cm.get(r['observation_id'], {}).get('decision', 'unresolved') == 'unresolved' for r in obs)
            pending += sum(r['outcome'] == 'pending' for r in issue_scores)
            report['check_scores'].extend(check_scores)
            report['issue_scores'].extend(issue_scores)
            report['observations'].extend({**plan, **r} for r in obs)
            if raw_sha and (obs or predictions):
                unsigned = dict(reviewer_id=None, signed_at=None, reason=None, method_blinded=None, decision='unresolved')
                report['matching_template']['runs'].append({'planned_run_id': pid, 'raw_sha256': raw_sha,
                    'reference_sha256': refs[cid]['sha256'],
                    'checks': [dict(observation_id=r['observation_id'], **unsigned) for r in obs],
                    'issues': [dict(prediction_id=r['prediction_id'], **unsigned) for r in predictions]})
            report['runs'].append({**plan, 'run_status': state, 'raw_sha256': raw_sha, 'errors': errors,
                'raw_observation_count': len(obs), 'raw_issue_count': len(predictions), 'cost': cost,
                'pending_matching': pending, 'duplicate_checks': sum(r['decision'] == 'duplicate' for r in cm.values()),
                'unresolved_issue_expectations': unresolved_issues,
                'issue_domain_unavailable_units': sum('issue_expectations' not in u for u in units.values())})
        report['summaries'] = summarize(report['check_scores'], report['issue_scores'], report['runs'])
        report['status'] = 'scoring_pending' if (any(r['pending_matching'] or r['errors'] for r in report['runs'])
            or any(r['pending_evidence_support'] for r in report['summaries'])) else 'scored'
        report['limitations'] = ['Quality is conditional on independent reference and matching declarations; identities are not authenticated.',
            'No automatic semantic matching or semantic proof. Citation resolution and frozen evidence-set matching are separate.',
            'Missing issue domains and unresolved references are not negative examples. Pending matching yields lower-bound recall.',
            'One attempt per planned run only. Token costs use saved usage and frozen prices, not reconciled billing.',
            'No claim of matched actual cost, paid-budget enforcement, model quality or award readiness is made by this tool.']
    except (ValueError, TypeError, KeyError, OSError, UnicodeError) as error:
        report['errors'].append(str(error) if isinstance(error, ValueError) else 'scoring_input_unreadable')
        report['check_scores'], report['issue_scores'], report['runs'], report['summaries'], report['observations'] = [], [], [], [], []
        report.pop('matching_template', None)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('runs', type=Path)
    parser.add_argument('--ledger', type=Path)
    args = parser.parse_args()
    report = score_evaluation(args.manifest, args.runs, args.ledger)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return int(report['status'] == 'blocked')


if __name__ == '__main__':
    raise SystemExit(main())
