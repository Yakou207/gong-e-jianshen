"""Hand-written scoring oracles; these are NOT human AML quality references."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import pytest

from aml_qc.depgraph import digest, sources_for
from scripts.score_evaluation import call_cost, score_evaluation
from scripts.validate_evaluation import EXPOSURES, validate_manifest


ROOT = Path(__file__).resolve().parents[1]
STAMP = '2026-10-03T14:00:00+08:00'
EVIDENCE = {'type': 'document_span', 'document_id': 'narrative', 'revision': '1', 'span': [0, 4]}


def write(root, name, value):
    raw = json.dumps(value, ensure_ascii=False, indent=2).encode()
    (root / name).write_bytes(raw)
    return {'path': name, 'sha256': hashlib.sha256(raw).hexdigest()}


def source(root, name):
    path = ROOT / name
    return {'path': os.path.relpath(path, root), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def declaration(**extra):
    return dict(reviewer_id='fictional-fixture-reviewer', signed_at=STAMP,
                reason='Hand-authored oracle, no actual review claimed.', method_blinded=True, **extra)


@pytest.fixture
def experiment(tmp_path):
    case = {'case_id': 'fixture', 'subject_account_id': 'a', 'coverage_start': '2026-09-01T00:00:00+08:00',
        'coverage_end': '2026-09-02T00:00:00+08:00', 'documents': [
            {'document_id': 'narrative', 'revision': '1', 'text': '向甲一笔；向乙两笔'}],
        'transactions': [], 'materials': [], 'material_links': [], 'coverage': [],
        'review_scope': {'target_labels': ['count']}}
    schema = {'labels': {'count': {'allowed_values': ['supported', 'contradicted', 'insufficient_evidence']}}}
    people = [dict(person_id='fictional-' + p, signed_at=STAMP, exposure=dict.fromkeys(EXPOSURES, False)) for p in ('A', 'B')]
    units = [dict(reference_check_id='claim-' + str(n), label='count', object_scope={'account_id': 'a'},
        applicability='applicable', adjudication_status='adjudicated', reference_value='supported',
        reason='Hand-written expected label for scoring mechanics only.', evidence_sets=[[EVIDENCE]],
        issue_expectations=[{'type': 'claim_error', 'truth': 'negative'}]) for n in (1, 2)]
    runner = source(tmp_path, 'aml_qc/workflow.py')
    method = {'model': 'fixture-model', 'generation': {'temperature': 0, 'thinking': 'disabled', 'max_output_tokens': 4096},
        'runner': runner, 'budget': {'max_calls': 6, 'max_output_tokens': 4096, 'total_token_budget': 40000, 'currency_limit': '1'}}
    pricing = {'currency': 'CNY', 'effective_at': STAMP, 'source_url': 'https://example.invalid/fixture',
        'unit_tokens': 1000, 'rates': {'input_cache_hit': '0.1', 'input_cache_miss': '1', 'output': '2'}}
    manifest = {'contract_version': 'evaluation-freeze-1', 'experiment_id': 'scoring-fixture-only', 'status': 'frozen',
        'frozen_at': STAMP, 'approved_by': [people[0]], 'repeat_count': 1,
        'retry_policy': {'max_retries': 0, 'retryable_errors': []}, 'currency': 'CNY', 'total_currency_budget': '3',
        'pricing': write(tmp_path, 'price.json', pricing), 'methods': {m: deepcopy(method) for m in ('B0', 'Fixed', 'Agent')},
        'artifacts': {k: runner for k in ('spec', 'generator', 'dependency_lock')}}
    manifest['methods']['B0']['budget']['max_calls'] = 1
    manifest['artifacts'].update(schema=write(tmp_path, 'schema.json', schema),
        scorer=source(tmp_path, 'scripts/score_evaluation.py'), tools=[source(tmp_path, 'aml_qc/scoring_inputs.py')],
        prompts=[runner], family_grouping=write(tmp_path, 'grouping.json', {'reviewed_by': [people[0]], 'cases': [
            {'case_id': 'fixture', 'economic_group': 'fixture-family', 'split': 'test', 'reason': 'Artificial fixture.'}]}))
    reference = {'case_id': 'fixture', 'reviewers': people, 'check_units': units}
    return {'root': tmp_path, 'case': case, 'schema': schema, 'manifest': manifest, 'reference': reference,
            'pricing': pricing, 'raw': None, 'checks': [], 'issues': []}


def raw_run(experiment, *, count=2, state='completed', values=None):
    e = experiment
    claims = [dict(claim_id='local-' + str(n), kind='count', operator='exact', value=n,
                   source=deepcopy(EVIDENCE)) for n in range(1, count + 1)]
    raw = dict(case_id='fixture', mode='agent', strategy='full', run_status=state,
        execution={'model': 'fixture-model', **e['manifest']['methods']['Agent']['generation']},
        features=[], claims=claims, claim_results=[dict(claim_id=c['claim_id'], result=(values or ['supported'] * count)[i],
            execution_status='completed', evidence=[deepcopy(EVIDENCE)]) for i, c in enumerate(claims)],
        material_results=[], semantic_results=[], issues=[],
        model_requests=[{'request_hash': 'identical-request', 'usage': {
            'prompt_tokens': 300, 'prompt_cache_hit_tokens': 100, 'prompt_cache_miss_tokens': 200,
            'completion_tokens': 30, 'total_tokens': 330}}])
    raw['snapshot'] = {'sources': sources_for(e['case'], e['schema'], raw['execution'])}
    e['raw'] = raw
    e['checks'] = [declaration(observation_id='/claim_results/' + str(i), decision='matched',
        reference_check_id='claim-' + str(i + 1), proposition_faithful=True, evidence_support='supported') for i in range(count)]
    return raw


def run(experiment, *, ledger=True, corrupt=None):
    e, root = experiment, experiment['root']
    manifest = e['manifest']
    cref = write(root, 'case.json', e['case'])
    manifest['cases'] = [dict(case_id='fixture', family_id='fixture-family', split='test',
                             required_check_ids=[u['reference_check_id'] for u in e['reference']['check_units']], **cref)]
    e['reference']['case_sha256'] = cref['sha256']
    rref = write(root, 'reference.json', e['reference'])
    manifest['references'] = [dict(case_id='fixture', **rref)]
    units = e['reference']['check_units']
    manifest['reference_counts'] = {k: sum(u['adjudication_status'] == k for u in units) for k in ('adjudicated', 'unresolved')}
    mref = write(root, 'manifest.json', manifest)
    freeze = validate_manifest(root / 'manifest.json')
    assert freeze['frozen_valid'], freeze['issues']
    method = e.get('method', 'Agent')
    plan = next(p for p in freeze['planned_runs'] if p['method'] == method)
    index = dict(contract_version='evaluation-runs-1', manifest_sha256=mref['sha256'], runs=[])
    book = dict(contract_version='evaluation-matching-1', manifest_sha256=mref['sha256'], runs=[])
    if e['raw'] is not None:
        rawref = write(root, 'raw.json', e['raw'])
        index['runs'].append(dict(planned_run_id=plan['planned_run_id'], case_sha256=cref['sha256'],
            method_config_sha256=digest(manifest['methods'][method]), raw_result=rawref))
        book['runs'].append(dict(planned_run_id=plan['planned_run_id'], raw_sha256=rawref['sha256'],
            reference_sha256=rref['sha256'], checks=e['checks'], issues=e['issues']))
    if corrupt:
        corrupt(index, book)
    write(root, 'runs.json', index)
    write(root, 'ledger.json', book)
    return score_evaluation(root / 'manifest.json', root / 'runs.json', root / 'ledger.json' if ledger else None)


def agent(report):
    assert report['status'] != 'blocked', report['errors']
    return next(r for r in report['summaries'] if r['method'] == 'Agent')


def test_same_label_omission_is_a_miss_not_perfect_existing_output_accuracy(experiment):
    raw_run(experiment, count=1)
    result = agent(run(experiment))
    assert result['accuracy'] == {'numerator': 1, 'denominator': 2, 'value': .5}
    assert result['output_completion']['value'] == .5


@pytest.mark.parametrize('state', ['failed', 'partial'])
def test_partial_and_failed_runs_keep_valid_outputs_and_missing_units(experiment, state):
    raw_run(experiment, count=1, state=state)
    result = agent(run(experiment))
    assert result['accuracy']['value'] == .5
    assert result['run_status_counts'] == {state: 1}


def test_whole_failure_and_absent_plan_keep_reference_denominators_and_unknown_cost(experiment):
    raw = raw_run(experiment, count=0, state='failed')
    raw['model_requests'] = [{'status': 'failed', 'usage': None}]
    report = run(experiment)
    result = agent(report)
    assert result['accuracy']['denominator'] == 2 and result['accuracy']['numerator'] == 0
    assert result['total_cost'] is None and result['unknown_usage_calls'] == 1
    absent = next(r for r in report['summaries'] if r['method'] == 'B0')
    assert absent['accuracy']['denominator'] == 2 and absent['run_status_counts'] == {'not_run': 1}


def test_identity_unresolved_can_be_correct_business_abstention_but_not_successful_query(experiment):
    experiment['reference']['check_units'][0]['reference_value'] = 'insufficient_evidence'
    raw = raw_run(experiment, values=['insufficient_evidence', 'supported'])
    raw['claim_results'][0]['execution_status'] = 'identity_unresolved'
    result = agent(run(experiment))
    assert result['accuracy']['value'] == 1
    assert result['correct_abstention']['value'] == 1
    assert result['execution_completion']['value'] == .5


def test_real_citations_do_not_make_false_certainty_correct_or_supported(experiment):
    for unit in experiment['reference']['check_units']:
        unit['reference_value'] = 'insufficient_evidence'
    raw_run(experiment)
    report = run(experiment)
    result = agent(report)
    assert result['false_certainty']['value'] == 1 and result['accuracy']['value'] == 0
    assert result['evidence_supported_lower_bound']['value'] == 0
    assert all(r['citation_counts']['valid'] == 1 for r in report['check_scores'] if r['method'] == 'Agent')


def test_bad_extraction_does_not_pass_because_final_label_matches(experiment):
    raw = raw_run(experiment)
    raw['claims'][0]['value'] = 99
    experiment['checks'][0]['proposition_faithful'] = False
    assert agent(run(experiment))['accuracy']['value'] == .5


def add_issues(e, count=1):
    e['reference']['check_units'][0]['issue_expectations'][0]['truth'] = 'positive'
    e['raw']['issues'] = [dict(issue_id='i-' + str(i), type='claim_error', target_id='claim:local-1', evidence=[EVIDENCE]) for i in range(count)]
    e['issues'] = [declaration(prediction_id='/issues/' + str(i), decision='matched' if i == 0 else 'duplicate',
        reference_check_id='claim-1', proposition_faithful=True) for i in range(count)]


def test_duplicate_predictions_count_once_for_recall_and_charge_precision(experiment):
    raw_run(experiment)
    add_issues(experiment, 4)
    result = agent(run(experiment))['by_issue']['claim_error']
    assert result['tp'] == 1 and result['duplicate_fp'] == 3
    assert result['precision_adjudicated']['value'] == .25 and result['recall_lower_bound']['value'] == 1


def test_false_proposition_cannot_get_issue_tp_through_second_view(experiment):
    raw_run(experiment)
    add_issues(experiment)
    experiment['checks'][0]['proposition_faithful'] = False
    result = agent(run(experiment))['by_issue']['claim_error']
    assert result['fp'] == 1 and result['fn'] == 1 and result.get('tp', 0) == 0


def test_new_issue_stays_pending_and_widens_precision_bounds(experiment):
    raw_run(experiment)
    add_issues(experiment, 2)
    experiment['issues'][1] = declaration(prediction_id='/issues/1', decision='new_issue')
    report = run(experiment)
    result = agent(report)['by_issue']['claim_error']
    assert report['status'] == 'scoring_pending'
    assert result['precision_bounds'] == [.5, 1] and result['recall_lower_bound']['value'] == 1


def test_alternative_evidence_sets_are_or_of_and_and_no_citation_means_no_support(experiment):
    raw = raw_run(experiment)
    missing = {'type': 'transactions', 'transaction_ids': ['missing']}
    for u in experiment['reference']['check_units']:
        u['evidence_sets'] = [[EVIDENCE, missing], [EVIDENCE]]
    assert agent(run(experiment))['evidence_supported_lower_bound']['value'] == 1
    experiment['reference']['check_units'][0]['evidence_sets'] = [[EVIDENCE, missing]]
    raw['claim_results'][1]['evidence'] = []
    assert agent(run(experiment))['evidence_supported_lower_bound']['value'] == 0


def test_missing_matching_and_stale_hash_never_grant_tp(experiment):
    raw_run(experiment)
    report = run(experiment, ledger=False)
    assert report['status'] == 'scoring_pending' and agent(report)['accuracy']['value'] == 0
    report = run(experiment, corrupt=lambda i, b: b['runs'][0].update(reference_sha256='0' * 64))
    assert agent(report)['accuracy']['value'] == 0
    assert 'ledger_raw_or_reference_hash_mismatch' in report['runs'][-1]['errors']


def test_changed_reference_rescores_same_bytes_and_unresolved_reference_is_separate(experiment):
    raw_run(experiment)
    first = run(experiment)
    assert agent(first)['accuracy']['value'] == 1
    experiment['reference']['check_units'][0]['reference_value'] = 'contradicted'
    second = run(experiment)
    assert agent(second)['accuracy']['value'] == .5
    assert first['runs'][-1]['raw_sha256'] == second['runs'][-1]['raw_sha256']
    experiment['reference']['check_units'][0].update(reference_value=None, adjudication_status='unresolved')
    third = agent(run(experiment))
    assert third['accuracy']['denominator'] == 1 and third['unresolved_reference_units'] == 1


@pytest.mark.parametrize('problem', ['case', 'generation', 'human', 'source', 'incremental', 'file_hash'])
def test_invalid_runs_are_preserved_in_denominator(experiment, problem):
    raw = raw_run(experiment)
    if problem == 'case':
        raw['case_id'] = 'other-case'
    elif problem == 'generation':
        raw['execution']['temperature'] = 1
    elif problem == 'human':
        raw['claims'][0]['origin'] = 'human_reviewed'
    elif problem == 'source':
        raw['snapshot']['sources']['source:documents']['value'][0]['text'] = 'different'
    elif problem == 'incremental':
        raw['strategy'] = 'incremental'
    corrupt = (lambda i, b: i['runs'][0]['raw_result'].update(sha256='0' * 64)) if problem == 'file_hash' else None
    result = agent(run(experiment, corrupt=corrupt))
    assert result['run_status_counts'] == {'invalid': 1} and result['accuracy']['denominator'] == 2
    assert result['accuracy']['numerator'] == 0


def test_duplicate_primary_cannot_select_better_later_output(experiment):
    raw = raw_run(experiment)
    raw['claim_results'].append(deepcopy(raw['claim_results'][0]))
    experiment['checks'].append(declaration(observation_id='/claim_results/2', decision='matched',
        reference_check_id='claim-1', proposition_faithful=True, evidence_support='supported'))
    report = run(experiment)
    assert 'cannot_skip_first_structural_duplicate' in report['runs'][-1]['errors']
    assert agent(report)['accuracy']['value'] == 0


def test_no_issue_reference_is_unavailable_not_implicitly_negative(experiment):
    raw_run(experiment)
    add_issues(experiment)
    for unit in experiment['reference']['check_units']:
        del unit['issue_expectations']
    result = agent(run(experiment))
    assert result['issue_domain_unavailable_units'] == 2
    assert result['by_issue']['claim_error']['pending'] == 1
    assert result['by_issue']['claim_error']['recall_lower_bound']['value'] is None


def test_exact_costs_count_identical_requests_separately_without_double_cache(experiment):
    costs = []
    for hit, miss, output in [(20, 80, 10), (50, 150, 20), (100, 200, 30)]:
        call = {'request_hash': 'same', 'usage': {'prompt_cache_hit_tokens': hit, 'prompt_cache_miss_tokens': miss,
                'prompt_tokens': hit + miss, 'completion_tokens': output}}
        costs.append(call_cost([call], experiment['pricing'], attempted=True)['total_cost'])
    assert costs == ['0.102', '0.195', '0.27']
    double = call_cost([call, call], experiment['pricing'], attempted=True)
    assert double['total_cost'] == '0.54' and double['recorded_calls'] == 2
    unknown = call_cost([call, {'status': 'failed'}], experiment['pricing'], attempted=True)
    assert unknown['total_cost'] is None and unknown['known_cost'] == '0.27'


def test_retry_capable_protocol_is_explicitly_blocked_not_best_of_selected(experiment):
    experiment['manifest']['retry_policy']['max_retries'] = 1
    result = run(experiment)
    assert result['status'] == 'blocked' and result['errors'] == ['scorer_v1_requires_zero_retries']


def test_failed_and_not_run_positive_expectations_still_count_fn(experiment):
    raw_run(experiment, count=0, state='failed')
    for unit in experiment['reference']['check_units']:
        unit['issue_expectations'][0]['truth'] = 'positive'
    report = run(experiment)
    for summary in report['summaries']:
        metric = summary['by_issue']['claim_error']
        assert metric['fn'] == 2 and metric['recall_lower_bound']['value'] == 0


def test_feature_met_is_never_implicitly_a_qc_error(experiment):
    raw = raw_run(experiment, count=0)
    unit = experiment['reference']['check_units'][0]
    unit.update(label='F1', reference_value='met', issue_expectations=[])
    experiment['reference']['check_units'] = [unit]
    experiment['case']['review_scope']['target_labels'] = ['F1']
    experiment['schema']['labels'] = {'F1': {'allowed_values': ['met', 'not_met', 'undeterminable']}}
    experiment['manifest']['artifacts']['schema'] = write(experiment['root'], 'schema.json', experiment['schema'])
    raw['features'] = [dict(feature_code='F1', result='met', execution_status='completed', evidence=[EVIDENCE])]
    raw['snapshot']['sources'] = sources_for(experiment['case'], experiment['schema'], raw['execution'])
    experiment['checks'] = [declaration(observation_id='/features/0', reference_check_id='claim-1',
        decision='matched', proposition_faithful=True, evidence_support='supported')]
    summary = agent(run(experiment))
    assert summary['accuracy']['value'] == 1 and summary['by_issue'] == {}


def test_unchecked_complex_citations_remain_pending_not_verified_support(experiment):
    raw = raw_run(experiment)
    raw['claim_results'][0]['evidence'].append({'type': 'tool_result', 'node': 'unverified'})
    report = run(experiment)
    assert report['status'] == 'scoring_pending'
    assert agent(report)['pending_evidence_support'] == 1
    assert agent(report)['evidence_supported_lower_bound']['value'] == .5


@pytest.mark.parametrize('corruption', [None, 'raw_response', 'parsed_output', 'call_response'])
def test_b0_saved_output_uses_same_reference_and_preserves_one_call(experiment, corruption):
    from aml_qc.baseline import run_baseline
    from aml_qc.schema import load_schema
    e = experiment
    e['case'] = json.loads((ROOT / 'data/synthetic/seed-01.json').read_text())
    e['case'].update(case_id='fixture', task_mode='annotation_only', alert=None)
    e['case']['review_scope'] = {'target_labels': ['count']}
    e['schema'] = load_schema()
    e['manifest']['artifacts']['schema'] = write(e['root'], 'schema.json', e['schema'])
    e['manifest']['methods']['B0']['runner'] = source(e['root'], 'aml_qc/baseline.py')
    e['manifest']['artifacts']['prompts'] = [source(e['root'], 'config/prompts/b0-v1/direct.txt')]
    text = next(d['text'] for d in e['case']['documents'] if d['document_id'] == 'narrative')
    evidence = {**EVIDENCE, 'quote': text[:4]}
    for unit in e['reference']['check_units']:
        unit['evidence_sets'] = [[evidence]]
        unit['object_scope']['account_id'] = e['case']['subject_account_id']
    response = {'checks': [dict(check_id='c-' + str(i), label='count', status='completed', value='supported',
        object_scope={'account_id': e['case']['subject_account_id']},
        anchor={k: v for k, v in evidence.items() if k != 'type'} | {'claim': {'operator': 'exact', 'value': i}},
        reason='Artificial fixture answer.', evidence=[evidence]) for i in (1, 2)],
        'coverage': [{'label': 'count', 'status': 'completed', 'reason': 'Artificial fixture.'}], 'issues': [], 'unfinished': []}
    class Model:
        model = 'fixture-model'
        def __init__(self):
            self.calls = []
        def complete(self, messages):
            self.calls.append({'usage': {'prompt_tokens': 100, 'completion_tokens': 10, 'total_tokens': 110,
                'prompt_cache_hit_tokens': 20, 'prompt_cache_miss_tokens': 80}})
            return {'role': 'assistant', 'content': json.dumps(response)}
    model = Model()
    e['raw'] = run_baseline(e['case'], model=model)
    assert e['raw']['parsed_output'] is not None, e['raw']['validation_errors']
    e['method'] = 'B0'
    e['checks'] = [declaration(observation_id='/parsed_output/checks/' + str(i), reference_check_id='claim-' + str(i + 1),
        decision='matched', proposition_faithful=True, evidence_support='supported') for i in range(2)]
    if corruption == 'raw_response':
        e['raw']['raw_response']['content'] = 'not JSON'
    elif corruption == 'parsed_output':
        e['raw']['parsed_output']['checks'][0]['value'] = 'contradicted'
    elif corruption == 'call_response':
        e['raw']['call_records'][0]['response'] = {'role': 'assistant', 'content': '{}'}
    report = run(e)
    summary = next(r for r in report['summaries'] if r['method'] == 'B0')
    if corruption:
        assert summary['accuracy']['value'] == 0 and summary['run_status_counts'] == {'invalid': 1}
        return
    assert report['status'] == 'scored', report
    assert summary['accuracy']['value'] == 1 and summary['total_cost'] == '0.102'
    assert len(model.calls) == 1


def test_manual_faithful_declaration_cannot_override_known_direction_conflict(experiment):
    raw = raw_run(experiment)
    for unit in experiment['reference']['check_units']:
        unit['object_scope']['direction'] = 'out'
    for claim in raw['claims']:
        claim['direction'] = 'in'
    report = run(experiment)
    assert agent(report)['accuracy']['value'] == 0
    assert 'explicit_scope_conflict' in report['runs'][-1]['errors']


def test_marking_bad_first_duplicate_unrelated_cannot_pick_better_later_answer(experiment):
    raw = raw_run(experiment, values=['contradicted', 'supported'])
    raw['claim_results'].append({**deepcopy(raw['claim_results'][0]), 'result': 'supported'})
    experiment['checks'][0] = declaration(observation_id='/claim_results/0', decision='unrelated')
    experiment['checks'].append(declaration(observation_id='/claim_results/2', decision='matched',
        reference_check_id='claim-1', proposition_faithful=True, evidence_support='supported'))
    report = run(experiment)
    assert 'cannot_skip_first_structural_duplicate' in report['runs'][-1]['errors']
    assert agent(report)['accuracy']['value'] == 0


def test_declared_call_count_and_usage_incompleteness_cannot_hide_missing_bill(experiment):
    raw = raw_run(experiment)
    raw['stats'] = {'model_calls': 2, 'usage_complete': False}
    result = agent(run(experiment))
    assert result['total_cost'] is None and result['known_cost'] == '0.27' and result['unknown_usage_calls'] == 1


@pytest.mark.parametrize('field,value', [('execution', []), ('snapshot', None), ('mode', None)])
def test_malformed_raw_shape_becomes_invalid_without_dropping_plan(experiment, field, value):
    raw_run(experiment)[field] = value
    result = agent(run(experiment))
    assert result['run_status_counts'] == {'invalid': 1} and result['accuracy']['denominator'] == 2


@pytest.mark.parametrize('change', ['returned_model', 'request_model', 'request_temperature'])
def test_recorded_model_or_request_contradiction_invalidates_run(experiment, change):
    raw = raw_run(experiment)
    call = raw['model_requests'][0]
    if change == 'returned_model':
        call['model_returned'] = 'different-model'
    else:
        call['request'] = {'model': 'different-model' if change == 'request_model' else 'fixture-model', 'temperature': 1}
    result = agent(run(experiment))
    assert result['run_status_counts'] == {'invalid': 1} and result['accuracy']['numerator'] == 0


@pytest.mark.parametrize('usage', [None, dict(prompt_tokens=0, prompt_cache_hit_tokens=0,
    prompt_cache_miss_tokens=0, completion_tokens=0, total_tokens=0)])
def test_explicit_not_sent_invocation_costs_zero_without_disappearing(experiment, usage):
    record = {'dispatch_status': 'not_sent', 'budget_event_id': 'budget-denied', 'usage': usage,
              'status': 'failed', 'response': None}
    result = call_cost([record], experiment['pricing'], attempted=True, declared_calls=1,
                      usage_complete=usage is not None)
    assert result['recorded_calls'] == 1 and result['dispatched_calls'] == 0
    assert result['total_cost'] == '0' and result['unknown_usage_calls'] == 0


def test_not_sent_missing_usage_does_not_hide_sent_usage_or_missing_invocation(experiment):
    sent = {'dispatch_status': 'sent', 'usage': {'prompt_tokens': 300, 'prompt_cache_hit_tokens': 100,
        'prompt_cache_miss_tokens': 200, 'completion_tokens': 30, 'total_tokens': 330}}
    denied = {'dispatch_status': 'not_sent', 'usage': None, 'status': 'failed'}
    complete = call_cost([sent, denied], experiment['pricing'], attempted=True, declared_calls=2, usage_complete=False)
    assert complete['total_cost'] == '0.27' and complete['unknown_usage_calls'] == 0
    assert complete['recorded_calls'] == 2 and complete['dispatched_calls'] == 1
    for calls, declared in [([{'dispatch_status': 'sent', 'usage': None}, denied], 2), ([sent, denied], 3)]:
        incomplete = call_cost(calls, experiment['pricing'], attempted=True, declared_calls=declared, usage_complete=False)
        assert incomplete['total_cost'] is None and incomplete['unknown_usage_calls'] == 1


@pytest.mark.parametrize('contradiction', ['usage', 'response'])
def test_not_sent_contradiction_cannot_silently_make_response_free(experiment, contradiction):
    record = {'dispatch_status': 'not_sent', 'usage': None, 'status': 'failed'}
    if contradiction == 'usage':
        record['usage'] = dict(prompt_tokens=1, prompt_cache_hit_tokens=0,
                               prompt_cache_miss_tokens=1, completion_tokens=0)
    else:
        record.update(status='completed', response={'role': 'assistant', 'content': '{}'})
    result = call_cost([record], experiment['pricing'], attempted=True, declared_calls=1)
    assert result['total_cost'] is None and result['unknown_usage_calls'] == 1


def test_legacy_usage_incomplete_stays_unknown_without_explicit_not_sent(experiment):
    usage = dict(prompt_tokens=0, prompt_cache_hit_tokens=0, prompt_cache_miss_tokens=0, completion_tokens=0)
    result = call_cost([{'usage': usage}], experiment['pricing'], attempted=True, declared_calls=1, usage_complete=False)
    assert result['total_cost'] is None and result['unknown_usage_calls'] == 1
