"""Independent adversarial oracles for human extraction, never model quality."""
from copy import deepcopy
from pathlib import Path

import pytest

from aml_qc.claim_edits import (
    assess_fidelity, fidelity_context_hash, merge_claim_amendments,
    normalize_proposed_claim,
)
from aml_qc import core
from aml_qc.ingest import load_case
from aml_qc.depgraph import business_result
from aml_qc.llm import FrozenModel
from aml_qc.store import Store
from aml_qc.workflow import run_review
from test_model_safety import MODEL, ScriptedModel, json_message


def case(text):
    package = load_case(Path(__file__).resolve().parents[1] / 'data/synthetic/seed-01.json')
    package['task_mode'] = 'annotation_only'
    package['alert'] = None
    package['review_scope']['target_labels'] = ['count', 'amount_sum']
    next(d for d in package['documents'] if d['document_id'] == 'narrative')['text'] = text
    return package


def claim(package, value=1, *, quote=None, kind='count', **fields):
    text = next(d['text'] for d in package['documents'] if d['document_id'] == 'narrative')
    return normalize_proposed_claim(package, {
        'kind': kind, 'operator': 'exact', 'value': value,
        'quote': text if quote is None else quote, 'direction': 'out',
        'counterparty_ref': '乙公司', 'unit': '元' if kind == 'amount_sum' else '次',
        **fields,
    }, 'independent-proposed-claim')


@pytest.mark.parametrize('text,value,kind,fields', [
    ('检查期间仅向乙公司支付99次。', 1, 'count', {}),
    ('检查期间仅向乙公司支付99次。', 99, 'count', {'operator': 'at_least'}),
    ('检查期间向乙公司至少支付99次。', 99, 'count', {'operator': 'exact'}),
    ('检查期间向乙公司至多支付99次。', 99, 'count', {'operator': 'exact'}),
    ('检查期间仅向乙公司支付99次。', 99, 'count', {'operator': 'exists'}),
    ('检查期间仅向乙公司支付99次。', 99, 'count', {'operator': 'none'}),
    ('检查期间向乙公司支付99次，每笔金额为1050元。', 1, 'count', {}),
    ('检查期间向乙公司合计支付99万元。', '99', 'amount_sum', {}),
    ('检查期间向乙公司合计支付1.5万元。', '1.5', 'amount_sum', {}),
    ('检查期间向乙公司合计支付99元。', '99', 'amount_sum', {'direction': 'in'}),
])
def test_explicit_text_cannot_be_weakened_to_match_a_different_transaction_result(text, value, kind, fields):
    package = case(text)
    proposed = claim(package, value, kind=kind, **fields)
    assert assess_fidelity(package, proposed)['blocking_errors'], (
        'A faithful extraction must preserve the explicit proposition, '
        'regardless of whether another proposition would match the ledger'
    )


def test_substring_quote_cannot_erase_explicit_month_or_numeric_qualifier():
    package = case('本月仅向乙公司支付99次。')
    proposed = claim(package, 1, quote='支付99次', operator='at_least',
                     start='2026-09-01T00:00:00+08:00', end='2026-09-02T00:00:00+08:00')
    errors = assess_fidelity(package, proposed)['blocking_errors']
    assert any('数值' in item for item in errors)
    assert any('限定词' in item for item in errors)
    assert any('本月' in item for item in errors)


@pytest.mark.parametrize('text,value,kind', [
    ('检查期间仅向乙公司支付1次。', 1, 'count'),
    ('检查期间向乙公司合计支付99万元。', '990000', 'amount_sum'),
    ('检查期间向乙公司合计支付1.5万元。', '15000', 'amount_sum'),
])
def test_faithful_explicit_value_is_not_rejected_as_a_business_error(text, value, kind):
    package = case(text)
    assessment = assess_fidelity(package, claim(package, value, kind=kind))
    assert not assessment['blocking_errors']
    assert assessment['notes'], 'Mechanical success must not claim semantic or real-person verification'


def amendment(package, old, proposed=None, operation='replace', **fields):
    return {'amendment_id': 'amendment-one', 'operation': operation,
            'target_claim_id': old['claim_id'], 'original_claim': deepcopy(old),
            'proposed_claim': deepcopy(proposed), 'context_hash': fidelity_context_hash(package), **fields}


def test_replacement_deduplicates_a_later_machine_extraction_that_is_already_correct():
    package = case('检查期间仅向乙公司支付1次。')
    proposed = claim(package)
    wrong = {**deepcopy(proposed), 'claim_id': 'machine-wrong', 'value': 99}
    corrected_machine = {**deepcopy(proposed), 'claim_id': 'machine-correct'}
    package['claim_amendments'] = [amendment(package, wrong, proposed)]
    machine_inputs = deepcopy([corrected_machine])
    merged = merge_claim_amendments(package, machine_inputs)
    assert len(merged['claims']) == 1 and merged['claims'][0]['value'] == 1
    assert merged['claims'][0]['origin'] == 'human_reviewed'
    assert merged['amendments'][0]['status'] == 'applied'
    assert machine_inputs == [corrected_machine], 'Merge must not rewrite saved machine predictions'


def test_changed_document_with_same_revision_cannot_reuse_an_old_approved_extraction():
    package = case('检查期间仅向乙公司支付1次。')
    proposed = claim(package)
    old = {**deepcopy(proposed), 'claim_id': 'machine-wrong', 'value': 99}
    package['claim_amendments'] = [amendment(package, old, proposed)]
    next(d for d in package['documents'] if d['document_id'] == 'narrative')['text'] = '检查期间仅向乙公司支付2次。'
    merged = merge_claim_amendments(package, [old])
    assert merged['amendments'][0]['status'] == 'needs_review'
    assert merged['claims'] == [old]
    assert not merged['superseded_machine_claims']


def test_retirement_does_not_claim_success_when_its_machine_target_has_changed():
    package = case('检查期间仅向乙公司支付1次。')
    old = {**claim(package), 'claim_id': 'machine-old'}
    new = {**deepcopy(old), 'claim_id': 'machine-new', 'value': 2}
    package['claim_amendments'] = [amendment(package, old, operation='retire')]
    merged = merge_claim_amendments(package, [new])
    assert merged['claims'] == [new]
    assert merged['amendments'][0]['status'] == 'needs_review'
    assert not merged['superseded_machine_claims']


def test_revoking_a_replacement_restores_current_machine_claim_without_deleting_history():
    package = case('检查期间仅向乙公司支付1次。')
    proposed = claim(package)
    old = {**deepcopy(proposed), 'claim_id': 'machine-wrong', 'value': 99}
    approved = amendment(package, old, proposed)
    revoked = {**amendment(package, old, operation='revoke'), 'amendment_id': 'amendment-two',
               'supersedes_amendment_id': approved['amendment_id']}
    package['claim_amendments'] = [approved, revoked]
    frozen = deepcopy(package['claim_amendments'])
    merged = merge_claim_amendments(package, [old])
    assert merged['claims'] == [old]
    assert [item['status'] for item in merged['amendments']] == ['superseded', 'revoked']
    assert package['claim_amendments'] == frozen


def machine_run(package, value):
    text = next(d['text'] for d in package['documents'] if d['document_id'] == 'narrative')
    model = ScriptedModel([json_message({'claims': [{
        'kind': 'count', 'operator': 'exact', 'value': value, 'unit': '次',
        'quote': text, 'direction': 'out', 'counterparty_ref': '乙公司',
    }], 'unresolved': []})])
    return run_review(package, provider='frozen', model=model)


def approve_replacement(store, state, value=1):
    package = state['package']
    text = next(d['text'] for d in package['documents'] if d['document_id'] == 'narrative')
    state = store.propose_claim(package['case_id'], operation='replace',
        target_claim_id=state['latest_run']['claims'][0]['claim_id'],
        proposed_claim={'kind': 'count', 'operator': 'exact', 'value': value, 'unit': '次',
                        'quote': text, 'direction': 'out', 'counterparty_ref': '乙公司'},
        actor='proposer-a', reason='只更正误抽，保留被检原文', snapshot_id=state['latest_run']['snapshot_id'])
    proposal = state['claim_proposals'][-1]
    return store.review_claim_proposal(package['case_id'], proposal['proposal_id'],
        action='approve', actor='reviewer-b', reason='独立对照原文核对结构忠实性，不替代流水核验',
        fidelity='faithful', snapshot_id=state['latest_run']['snapshot_id'],
        expected_event_id=proposal['expected_event_id'])


def test_actual_original_99_cannot_be_approved_as_1_even_when_preview_is_supported(tmp_path):
    package = case('检查期间仅向乙公司支付99次。')
    package['review_scope']['target_labels'] = ['count']
    store = Store(tmp_path / 'literal-99.sqlite3')
    initial = store.create(package)
    state = store.save_run(package['case_id'], initial['source_hash'], machine_run(initial['package'], 99))
    assert state['latest_run']['claim_results'][0]['result'] == 'contradicted'
    with pytest.raises(ValueError, match='忠实性'):
        approve_replacement(store, state)
    current = store.get(package['case_id'])
    proposal = current['claim_proposals'][-1]
    assert proposal['preview_verification']['result'] == 'supported'
    assert proposal['status'] == 'pending'
    assert not current['package'].get('claim_amendments')
    assert current['latest_run']['claims'][0]['value'] == 99
    assert not store.export(package['case_id'])['deliverable']['passed']


def test_approved_misextract_replacement_cannot_inherit_old_material_claim_binding(tmp_path):
    package = case('检查期间仅向乙公司支付1次。')
    package['review_scope']['target_labels'] = ['count', 'material_relation']
    old_id = machine_run(package, 99)['claims'][0]['claim_id']
    package['material_links'][0]['claim_or_issue_id'] = old_id
    store = Store(tmp_path / 'old-claim-link.sqlite3')
    initial = store.create(package)
    state = store.save_run(package['case_id'], initial['source_hash'], machine_run(initial['package'], 99))
    assert state['latest_run']['material_results'][0]['result'] == 'corresponds'
    approved = approve_replacement(store, state)
    fresh = store.save_run(package['case_id'], approved['source_hash'], machine_run(approved['package'], 99))
    assert fresh['latest_run']['claim_results'][0]['result'] == 'supported'
    link = fresh['latest_run']['material_results'][0]
    assert link['result'] != 'corresponds', 'Old Claim correspondence must be re-bound after proposition correction'
    assert fresh['open_items'] and not store.export(package['case_id'])['deliverable']['passed']


@pytest.mark.parametrize('kind', ['counterparty', 'time_range'])
@pytest.mark.parametrize('operation', ['add', 'replace'])
def test_other_claim_kinds_enter_real_workflow_only_after_independent_approval(tmp_path, kind, operation):
    text = ('检查期间仅向乙公司付款。' if kind == 'counterparty'
            else '检查期间的付款均发生在2026年9月1日。')
    package = case(text)
    package['review_scope']['target_labels'] = [kind]
    value = ('乙公司' if kind == 'counterparty' else
             {'start': '2026-09-01T00:00:00+08:00', 'end': '2026-09-02T00:00:00+08:00'})
    correct = {'kind': kind, 'operator': 'only', 'value': value, 'quote': text, 'direction': 'out'}
    wrong = {**deepcopy(correct), 'value': ('甲公司' if kind == 'counterparty' else
             {'start': '2026-09-03T00:00:00+08:00', 'end': '2026-09-04T00:00:00+08:00'})}
    output = [] if operation == 'add' else [wrong]
    store = Store(tmp_path / (kind + '-' + operation + '.sqlite3'))
    state = store.create(package)

    def execute(package, previous=None, strategy='full', frozen=None):
        model = frozen or ScriptedModel([json_message({'claims': output, 'unresolved': []})])
        return run_review(package, provider='frozen', strategy=strategy, previous=previous, model=model), model

    baseline, _ = execute(state['package'])
    state = store.save_run(package['case_id'], state['source_hash'], baseline)
    old_snapshot = deepcopy(state['latest_run']['snapshot'])
    old_machine = deepcopy(state['latest_run']['machine_claims'])
    target = old_machine[0]['claim_id'] if operation == 'replace' else None
    pending = store.propose_claim(package['case_id'], operation=operation, target_claim_id=target,
        proposed_claim=correct, actor='first-reviewer', reason='根据当前原文提出结构提议',
        snapshot_id=state['latest_run']['snapshot_id'])
    proposal = pending['claim_proposals'][-1]
    assert proposal['fidelity_assessment']['notes']
    assert pending['latest_run']['machine_claims'] == old_machine
    assert not pending['package'].get('claim_amendments') and not pending['can_pass']
    review_args = dict(action='approve', fidelity='faithful', reason='逐项对照原文核对，交易判断交给重算',
                       snapshot_id=pending['latest_run']['snapshot_id'], expected_event_id=proposal['expected_event_id'])
    with pytest.raises(ValueError, match='另一人员'):
        store.review_claim_proposal(package['case_id'], proposal['proposal_id'], actor=' FIRST-REVIEWER ', **review_args)
    approved = store.review_claim_proposal(package['case_id'], proposal['proposal_id'], actor='second-reviewer', **review_args)
    assert approved['stale'] and not approved['can_pass']
    full, recording = execute(approved['package'])
    incremental, _ = execute(approved['package'], previous=old_snapshot, strategy='incremental',
                             frozen=FrozenModel(recording.request_products, MODEL))
    assert business_result(full) == business_result(incremental)
    assert full['stats']['reused'] == 0
    assert full['machine_claims'] == old_machine
    effective, = full['claims']
    assert effective['kind'] == kind and effective['origin'] == 'human_reviewed'
    assert full['claim_results'][0]['result'] == 'supported'
    assert full['claim_results'][0]['execution_status'] == 'completed'
    assert 'source:claim_amendments' in incremental['stats']['changed_sources']
    fresh = store.save_run(package['case_id'], approved['source_hash'], full)
    assert fresh['annotations'][0]['candidate_value'] == 'supported'
    assert not store.export(package['case_id'])['deliverable']['passed'], 'Source approval is not final task approval'


@pytest.mark.parametrize('text,kind,value', [
    ('向甲公司支付3次，向乙公司支付4次，均为货款。', 'count', 3),
    ('不是支付99次，而是支付1次。', 'count', 1),
    ('商品每笔单价99元，订单数量待核对。', 'amount_sum', '99'),
])
def test_ambiguous_fidelity_is_explicitly_left_to_reviewers_not_declared_verified(text, kind, value):
    package = case(text)
    assessment = assess_fidelity(package, claim(package, value, kind=kind))
    assert not assessment['blocking_errors']
    assert len(assessment['notes']) >= 2
    assert '语义' in assessment['notes'][0]


def test_unknown_material_target_is_unresolved_even_when_all_material_fields_correspond():
    package = case('检查期间仅向乙公司支付1次。')
    package['review_scope']['target_labels'] = ['count', 'material_relation']
    package['material_links'][0]['claim_or_issue_id'] = 'claim-that-never-existed'
    result = machine_run(package, 1)
    material = result['material_results'][0]
    assert material['field_result'] == 'corresponds'
    assert material['result'] == 'insufficient' and material['binding_status'] == 'needs_review'
    assert any(i['kind'] == 'material_insufficient' for i in result['open_items'])


def test_human_add_does_not_keep_applying_after_its_label_is_removed_from_task_scope():
    package = case('检查期间仅向乙公司支付1次。')
    proposed = claim(package)
    record = amendment(package, proposed, proposed, operation='add')
    record.update(target_claim_id=None, original_claim=None)
    package['claim_amendments'] = [record]
    package['review_scope']['target_labels'] = ['amount_sum']
    merged = merge_claim_amendments(package, [])
    assert merged['claims'] == [], 'Approved additions must not escape the current task scope'
    assert merged['amendments'][0]['status'] == 'needs_review'


def test_superseding_an_old_addition_applies_only_the_new_frozen_structured_claim():
    package = case('检查期间向乙公司支付1次。向甲公司支付2次。')
    first = claim(package, 1, quote='检查期间向乙公司支付1次。')
    old = amendment(package, first, first, operation='add')
    old.update(target_claim_id=None, original_claim=None)
    replacement = claim(package, 2, quote='向甲公司支付2次。', counterparty_ref='甲公司')
    new = {**amendment(package, replacement, replacement, operation='add'),
           'amendment_id': 'amendment-two', 'supersedes_amendment_id': old['amendment_id'],
           'target_claim_id': None, 'original_claim': None}
    package['claim_amendments'] = [old, new]
    merged = merge_claim_amendments(package, [])
    assert len(merged['claims']) == 1 and merged['claims'][0]['value'] == 2
    assert [row['status'] for row in merged['amendments']] == ['superseded', 'applied']
    assert package['claim_amendments'][0]['proposed_claim'] == first


@pytest.mark.parametrize('kind', ['counterparty', 'time_range', 'count'])
@pytest.mark.parametrize('ambiguous_preamble', ['', '原说明按不同业务分别记载，'])
def test_explicit_review_scope_cannot_be_narrowed_to_filter_out_a_counterexample(kind, ambiguous_preamble):
    text = ('检查期间仅向乙公司付款。' if kind == 'counterparty' else
            '检查期间的付款均发生在2026年9月1日。' if kind == 'time_range' else
            '检查期间仅向乙公司支付1次。')
    text = ambiguous_preamble + text
    package = case(text)
    package['review_scope']['target_labels'] = [kind]
    extra = deepcopy(next(t for t in package['transactions'] if t['direction'] == 'out'))
    extra.update(transaction_id='independent-counterexample', timestamp='2026-09-03T15:00:00+08:00')
    if kind == 'counterparty':
        extra['counterparty_token'] = 'different-visible-counterparty'
    package['transactions'].append(extra)
    payload = {'kind': kind, 'operator': 'only' if kind != 'count' else 'exact', 'quote': text, 'direction': 'out'}
    if kind == 'counterparty':
        payload.update(value='乙公司', counterparty_ref='乙公司')
    else:
        bounds = {'start': '2026-09-01T00:00:00+08:00', 'end': '2026-09-02T00:00:00+08:00'}
        payload.update(**bounds, value=bounds if kind == 'time_range' else 1)
    try:
        narrowed = normalize_proposed_claim(package, payload, 'narrowed-claim')
    except ValueError:
        return  # A structural guard may reject unsafe narrowing before assessment.
    assert core.verify_claim(package, narrowed)['result'] == 'supported', 'This fixture exposes the misleading narrow query'
    assert assess_fidelity(package, narrowed)['blocking_errors'], (
        'Explicit whole-review propositions must not be weakened by filtering away visible counterexamples'
    )


def test_second_identical_human_add_cannot_silently_replace_the_first_reviewed_record():
    package = case('检查期间向乙公司支付1次。')
    proposed = claim(package)
    first = amendment(package, proposed, proposed, operation='add')
    first.update(target_claim_id=None, original_claim=None)
    second = {**deepcopy(first), 'amendment_id': 'amendment-two'}
    second['proposed_claim']['claim_id'] = 'second-human-claim'
    package['claim_amendments'] = [first, second]
    merged = merge_claim_amendments(package, [])
    assert len(merged['claims']) == 1
    assert merged['claims'][0]['amendment_id'] == first['amendment_id']
    assert [row['status'] for row in merged['amendments']] == ['applied', 'needs_review']


def test_two_independent_counterparty_counts_in_one_sentence_do_not_conflict():
    package = case('检查期间向乙公司支付1次，向甲公司支付2次。')
    first = claim(package, 1, counterparty_ref='乙公司')
    second = {**claim(package, 2, counterparty_ref='甲公司'), 'claim_id': 'second-human-claim'}
    records = [amendment(package, first, first, operation='add'),
               {**amendment(package, second, second, operation='add'), 'amendment_id': 'amendment-two'}]
    for row in records:
        row.update(target_claim_id=None, original_claim=None)
    package['claim_amendments'] = records
    merged = merge_claim_amendments(package, [])
    assert len(merged['claims']) == 2
    assert {row['counterparty_ref']: row['value'] for row in merged['claims']} == {'乙公司': 1, '甲公司': 2}
    assert all(row['status'] == 'applied' for row in merged['amendments'])
