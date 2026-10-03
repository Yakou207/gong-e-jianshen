"""Mechanism checks use the actual local workflow, never a live model/API.

These validate cache/dependency correctness, not model quality or Agent ability.
The independent full path is always called without the previous snapshot.
"""
from copy import deepcopy
from pathlib import Path

import pytest

from aml_qc import core
from aml_qc.depgraph import affected_nodes, business_result
from aml_qc.ingest import load_case
from aml_qc.workflow import run_review

DATA = Path(__file__).resolve().parents[1] / 'data' / 'synthetic'


def case(number=2):
    return load_case(DATA / f'seed-{number:02d}.json')


def test_partial_coverage_keeps_two_feature_obligations_individually_addressable():
    value = case(1)
    value['coverage'][0]['status'] = 'partial'
    result = run_review(value)
    problems = [item for item in result['issues'] if item['type'] == 'insufficient_coverage']
    assert len(problems) == 2
    assert len({item['issue_id'] for item in problems}) == 2
    checks = {item['check_id'] for item in result['required_checks'] if item['label'] in {'F1', 'F2'}}
    assert {item['target_id'] for item in problems} == checks
    open_items = {item['item_id']: item for item in result['open_items']}
    assert all(open_items[item['issue_id']]['target_id'] == item['target_id'] for item in problems)


def test_explicit_response_target_without_focus_is_required_even_annotation_only():
    value = case()
    value['task_mode'] = 'annotation_only'
    value['alert'] = None
    value['review_scope'] = {'target_labels': ['alert_response']}
    result = run_review(value)
    assert any(c['check_id'] == 'alert' and c['status'] != 'completed' for c in result['required_checks'])
    assert any(i['type'] == 'missing_alert' for i in result['issues'])
    assert result['run_status'] == 'partial'


@pytest.mark.parametrize('text', ['', '  \n  '])
def test_empty_narrative_cannot_be_a_semantic_candidate(text):
    value = case()
    next(d for d in value['documents'] if d['document_id'] == 'narrative')['text'] = text
    result = run_review(value)
    assert result['semantic_results'] == []
    assert any(c['check_id'] == 'narrative' and c['status'] == 'pending' for c in result['required_checks'])
    assert any(i['type'] == 'missing_narrative' for i in result['issues'])
    assert result['run_status'] == 'partial'


def compare(old_case, new_case):
    old = run_review(deepcopy(old_case), mode='fixed', provider='local', strategy='full')
    frozen_previous = deepcopy(old['snapshot'])
    incremental = run_review(deepcopy(new_case), mode='fixed', provider='local', strategy='incremental', previous=old['snapshot'])
    independent_full = run_review(deepcopy(new_case), mode='fixed', provider='local', strategy='full')
    assert business_result(incremental) == business_result(independent_full)
    assert incremental['snapshot']['nodes'] == independent_full['snapshot']['nodes']
    assert incremental['snapshot']['candidate_ids'] == independent_full['snapshot']['candidate_ids']
    assert old['snapshot'] == frozen_previous, '重查不能改写历史快照'
    assert independent_full['stats']['reused'] == 0
    assert independent_full['stats']['recomputed'] == len(independent_full['snapshot']['nodes'])
    assert incremental['stats']['recomputed'] + incremental['stats']['reused'] == len(incremental['snapshot']['nodes'])
    changed, affected = affected_nodes(old['snapshot'], incremental['snapshot']['sources'])
    assert incremental['stats']['changed_sources'] == changed
    assert incremental['stats']['potentially_affected'] == len(affected)
    return old, incremental, independent_full, set(changed), set(affected)


@pytest.mark.parametrize('number', range(1, 7))
def test_six_seed_packages_unchanged_incremental_equals_independent_full(number):
    original = case(number)
    _, result, _, changed, affected = compare(original, deepcopy(original))
    assert result['stats']['recomputed'] == 0
    assert result['stats']['reused'] == len(result['snapshot']['nodes'])
    assert not changed and not affected
    assert result['agent_verified'] is False
    assert result['run_status'] == 'partial', '离线流程不能冒充语义检查和抽取完整性已验收'


@pytest.mark.parametrize('number', range(1, 7))
def test_package_revision_alone_does_not_relabel_transaction_evidence(number):
    original = case(number); updated = deepcopy(original)
    updated['data_version'] = '2'
    _, result, _, changed, affected = compare(original, updated)
    assert not changed and not affected
    assert result['stats']['recomputed'] == 0
    assert result['stats']['reused'] > 0


@pytest.mark.parametrize('number', range(1, 7))
def test_document_only_edit_preserves_transactions_and_reuses_features(number):
    original = case(number); updated = deepcopy(original)
    updated['data_version'] = '2'
    updated['documents'][0]['text'] = '补充说明：' + updated['documents'][0]['text']
    updated['documents'][0]['revision'] = '2'
    old, result, _, changed, affected = compare(original, updated)
    assert {'source:documents', 'source:documents:narrative'} <= changed
    assert 'extraction' in affected and 'features' not in affected
    assert result['snapshot']['nodes']['features'] == old['snapshot']['nodes']['features']
    assert result['stats']['reused'] >= 1
    for claim in result['claims']:
        assert claim['source']['revision'] == '2'
        assert core.validate_span(updated, claim['source'], claim['text'])


def test_transaction_correction_retracts_contradiction_without_reextracting():
    original = case(2); updated = deepcopy(original)
    outgoing = [t for t in updated['transactions'] if t['direction'] == 'out']
    removed = {t['transaction_id'] for t in outgoing[1:]}
    updated['transactions'] = [t for t in updated['transactions'] if t['transaction_id'] not in removed]
    updated['data_version'] = '2'
    old, result, _, changed, affected = compare(original, updated)
    assert 'source:transactions' in changed
    assert 'features' in affected and 'extraction' not in affected
    assert all(key in affected for key in old['snapshot']['nodes'] if key.startswith('claim:'))
    assert old['claim_results'][0]['result'] == 'contradicted'
    assert result['claim_results'][0]['result'] == 'supported'
    assert result['stats']['reused'] >= 1
    old_errors = {i['issue_id'] for i in old['issues'] if i['type'] == 'claim_error'}
    assert old_errors and old_errors.isdisjoint(result['snapshot']['candidate_ids'])
    assert result['stats']['revoked'] >= len(old_errors)


def test_empty_query_then_first_transaction_is_not_lost():
    updated = case(1); original = deepcopy(updated)
    original['transactions'] = [t for t in original['transactions'] if t['direction'] != 'out']
    updated['data_version'] = '2'
    old, result, _, changed, affected = compare(original, updated)
    old_query = next(e for e in old['claim_results'][0]['evidence'] if e['type'] == 'query_scope')
    assert old_query['transaction_ids'] == []
    assert old_query['scope']['transaction_set_version'].startswith('sha256:')
    assert old['claim_results'][0]['result'] == 'contradicted'
    assert result['claim_results'][0]['result'] == 'supported'
    assert 'source:transactions' in changed
    assert all(key in affected for key in old['snapshot']['nodes'] if key.startswith('claim:'))
    assert result['stats']['reused'] >= 1


def test_deletion_of_unique_evidence_invalidates_claim_and_material():
    original = case(1); updated = deepcopy(original)
    updated['transactions'] = [t for t in updated['transactions'] if t['direction'] != 'out']
    updated['data_version'] = '2'
    old, result, _, _, affected = compare(original, updated)
    assert old['claim_results'][0]['result'] == 'supported'
    assert result['claim_results'][0]['result'] == 'contradicted'
    assert result['material_results'][0]['result'] == 'insufficient'
    assert 'materials' in affected
    assert result['stats']['reused'] >= 1


def test_add_missing_material_updates_only_related_deterministic_nodes():
    original = case(3); updated = deepcopy(original)
    updated['materials'] = case(1)['materials']
    updated['data_version'] = '2'
    old, result, _, changed, affected = compare(original, updated)
    assert 'source:materials' in changed and 'materials' in affected
    assert 'features' not in affected and 'extraction' not in affected
    assert old['material_results'][0]['result'] == 'insufficient'
    assert result['material_results'][0]['result'] == 'corresponds'
    assert result['stats']['reused'] >= 2


def test_link_only_change_selects_installment_template():
    original = case(5); updated = deepcopy(original)
    updated['material_links'][0]['relation_template'] = 'contract_installments'
    updated['data_version'] = '2'
    old, result, _, changed, affected = compare(original, updated)
    assert changed == {'source:material_links'}
    assert affected == {'materials'}
    assert old['material_results'][0]['result'] == 'pending_judgement'
    assert result['material_results'][0]['result'] == 'corresponds'
    assert result['stats']['recomputed'] == 1
    assert result['stats']['reused'] >= 2


def test_identity_mapping_change_with_same_revision_is_not_reused():
    original = case(1)
    original['counterparties'].append({'counterparty_token':'supplier-other','display_name_masked':'乙公司','type':'business'})
    updated = deepcopy(original)
    updated['entity_mappings'][0]['target_token'] = 'supplier-other'
    updated['data_version'] = '2'
    old, result, _, changed, affected = compare(original, updated)
    assert changed == {'source:entities'}
    assert 'features' not in affected and 'extraction' not in affected
    assert all(key in affected for key in old['snapshot']['nodes'] if key.startswith('claim:'))
    assert old['claim_results'][0]['result'] == 'supported'
    assert result['claim_results'][0]['result'] == 'contradicted'
    assert old['claim_results'][0]['identity']['basis'] != result['claim_results'][0]['identity']['basis']
    assert result['stats']['reused'] >= 2


def test_schema_migration_uses_new_threshold_and_bound_versions():
    original = case(2); updated = deepcopy(original)
    updated['schema'] = core.load_schema()
    updated['schema']['schema_version'] = 'S1.1-test'
    updated['schema']['features']['F1']['minimum_days'] = 4
    updated['schema_version'] = 'S1.1-test'
    for link in updated['material_links']:
        link['schema_version'] = 'S1.1-test'
    updated['data_version'] = '2'
    old, result, _, changed, affected = compare(original, updated)
    assert 'source:schema' in changed and 'source:metadata' in changed
    assert affected == set(old['snapshot']['nodes'])
    assert next(f for f in old['features'] if f['feature_code'] == 'F1')['result'] == 'met'
    assert next(f for f in result['features'] if f['feature_code'] == 'F1')['result'] == 'not_met'
    assert result['stats']['reused'] == 0
    assert all(f['schema_version'] == 'S1.1-test' for f in result['features'])


def test_coverage_correction_recomputes_business_checks():
    original = case(2); original['coverage'][0]['status'] = 'partial'
    updated = deepcopy(original); updated['coverage'][0]['status'] = 'full'
    updated['coverage'][0]['revision'] = '2'; updated['data_version'] = '2'
    old, result, _, changed, affected = compare(original, updated)
    assert changed == {'source:coverage'}
    assert 'features' in affected and 'extraction' not in affected
    assert next(f for f in old['features'] if f['feature_code'] == 'F1')['result'] == 'undeterminable'
    assert next(f for f in result['features'] if f['feature_code'] == 'F1')['result'] == 'met'
    assert result['stats']['reused'] >= 1


def test_transaction_input_order_does_not_change_content_revision():
    original = case(2); updated = deepcopy(original)
    updated['transactions'].reverse(); updated['data_version'] = '2'
    _, result, _, changed, affected = compare(original, updated)
    assert not changed and not affected
    assert result['stats']['recomputed'] == 0


def test_material_input_order_does_not_change_content_evidence():
    original = case(1)
    extra = deepcopy(original['materials'][0]); extra['material_id'] = 'contract-extra'
    original['materials'].append(extra)
    updated = deepcopy(original); updated['materials'].reverse(); updated['data_version'] = '2'
    _, result, _, changed, affected = compare(original, updated)
    assert not changed and not affected
    assert result['stats']['recomputed'] == 0
