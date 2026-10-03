"""Extraction provenance and downstream review invariants, with offline fixtures."""
from copy import deepcopy

from aml_qc.claim_edits import assess_fidelity, fidelity_context_hash, merge_claim_amendments
from aml_qc.llm import FrozenModel
from aml_qc.depgraph import business_result
from aml_qc.store import Store
from aml_qc.workflow import run_review
from test_claim_edit_safety import case, claim, amendment, approve_replacement
from test_model_safety import MODEL, ScriptedModel, json_message, semantic_response


def test_fidelity_context_ignores_collection_order_and_transactions_but_not_text():
    package = case('检查期间向乙公司支付一次。')
    reordered = deepcopy(package)
    for key in ('documents', 'counterparties', 'entity_mappings'):
        reordered[key].reverse()
    reordered['transactions'] = []
    reordered['coverage'] = []
    assert fidelity_context_hash(package) == fidelity_context_hash(reordered)
    reordered['documents'][0]['text'] += '补充文字'
    assert fidelity_context_hash(package) != fidelity_context_hash(reordered)


def test_two_active_additions_cannot_silently_replace_one_another():
    package = case('检查期间向乙公司支付一次。')
    proposed = claim(package)
    first = amendment(package, proposed, proposed, operation='add', target_claim_id=None, original_claim=None)
    second = {**deepcopy(first), 'amendment_id': 'second', 'proposed_claim': {**proposed, 'claim_id': 'second-claim'}}
    package['claim_amendments'] = [first, second]
    result = merge_claim_amendments(package, [])
    assert len(result['claims']) == 1
    assert result['claims'][0]['amendment_id'] == first['amendment_id']
    assert result['amendments'][1]['status'] == 'needs_review'


def test_ambiguous_wording_cannot_skip_structural_query_scope_guard():
    package = case('检查期间分别向乙公司支付一次和两次。')
    proposed = claim(package, start=package['coverage_start'], end='2026-09-02T00:00:00+08:00')
    assessment = assess_fidelity(package, proposed)
    assert assessment['blocking_errors']
    assert any('语境' in note for note in assessment['notes'])


def test_semantic_and_agent_receive_effective_claims_after_independent_review(tmp_path):
    package = case('检查期间仅向乙公司支付一次。')
    package['review_scope']['target_labels'] = ['count', 'alert_response']
    package['alert'] = {'alert_id': 'test-alert', 'revision': 1, 'focuses': [{'focus_id': 'count-focus', 'text': '支付次数是否得到说明？'}]}
    quote = next(d['text'] for d in package['documents'] if d['document_id'] == 'narrative')
    raw = {'kind': 'count', 'operator': 'exact', 'value': 99, 'quote': quote, 'direction': 'out', 'counterparty_ref': '乙公司'}

    def model():
        return ScriptedModel([json_message({'claims': [raw], 'unresolved': []}),
                              json_message(semantic_response(package)), json_message(semantic_response(package))])

    store = Store(tmp_path / 'downstream.sqlite3')
    state = store.create(package)
    package = state['package']
    original = run_review(package, provider='frozen', mode='agent', model=model())
    state = store.save_run(package['case_id'], state['source_hash'], original)
    approved = approve_replacement(store, state)
    package = approved['package']
    proposed = package['claim_amendments'][0]['proposed_claim']
    scripted = model()
    full = run_review(package, provider='frozen', mode='agent', model=scripted)
    assert full['machine_claims'][0]['value'] == 99
    assert full['claims'][0]['value'] == 1
    assert full['claim_results'][0]['result'] == 'supported'
    for request in scripted.calls[1:]:
        payload = request['request']['messages'][1]['content']
        assert 'human_reviewed' in payload and proposed['claim_id'] in payload
        assert '"value":99' not in payload
    inc = run_review(package, provider='frozen', mode='agent', strategy='incremental', previous=original['snapshot'],
                     model=FrozenModel(scripted.request_products, MODEL))
    assert business_result(inc) == business_result(full)
