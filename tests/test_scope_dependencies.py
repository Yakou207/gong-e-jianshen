"""Scope changes compare independent business results, without live models."""
from copy import deepcopy

import pytest

from aml_qc.depgraph import business_result
from aml_qc.workflow import run_review
from test_workflow import case


def compare(old_case, changed_case):
    old = run_review(old_case)
    snapshot = deepcopy(old['snapshot'])
    incremental = run_review(changed_case, strategy='incremental', previous=snapshot)
    full = run_review(changed_case, strategy='full')
    assert business_result(incremental) == business_result(full)
    assert incremental['snapshot']['nodes'] == full['snapshot']['nodes']
    assert old['snapshot'] == snapshot
    assert full['stats']['reused'] == 0
    return old, incremental, full


@pytest.mark.parametrize('include_count', [False, True])
def test_label_scope_removal_and_addition_recompute_effective_claims(include_count):
    original = case(2)
    original['task_mode'] = 'annotation_only'
    original['alert'] = None
    original['review_scope'] = {'target_labels': ['amount_sum'] if include_count else ['count']}
    changed = deepcopy(original)
    changed['review_scope']['target_labels'] = ['count'] if include_count else ['amount_sum']
    old, incremental, full = compare(original, changed)
    assert bool(old['claims']) is not include_count
    assert bool(full['claims']) is include_count
    assert incremental['machine_claims'] == full['machine_claims']
    assert {row['kind'] for row in full['claims']} == ({'count'} if include_count else set())
    assert any(row['type'] == 'claim_error' for row in full['issues']) is include_count


@pytest.mark.parametrize('change', ['rename', 'remove'])
def test_material_binding_changes_when_original_focus_is_renamed_or_removed(change):
    original = case(1)
    changed = deepcopy(original)
    if change == 'rename':
        changed['alert']['focuses'][0]['focus_id'] = 'replacement-focus'
    else:
        changed['alert']['focuses'] = []
        changed['alert']['original_focus'] = ''
    changed['alert']['revision'] = '2'
    old, incremental, full = compare(original, changed)
    assert old['material_results'][0]['result'] == 'corresponds'
    assert full['material_results'][0]['result'] == 'insufficient'
    assert full['material_results'][0]['binding_status'] == 'needs_review'
    assert incremental['material_results'] == full['material_results']
    assert any(row['type'] == 'material_insufficient' for row in full['issues'])


@pytest.mark.parametrize('change', ['rename', 'remove', 'add'])
def test_material_binding_tracks_upgraded_focus_scope_changes(change):
    original = case(1)
    original['material_links'][0]['claim_or_issue_id'] = 'upgraded-focus'
    focus = {'focus_id': 'upgraded-focus', 'text': '明确升级为必需核查事项。'}
    original['review_scope']['upgraded_leads'] = [] if change == 'add' else [focus]
    changed = deepcopy(original)
    if change == 'rename':
        changed['review_scope']['upgraded_leads'][0]['focus_id'] = 'replacement-upgraded-focus'
    elif change == 'remove':
        changed['review_scope']['upgraded_leads'] = []
    else:
        changed['review_scope']['upgraded_leads'] = [focus]
    old, incremental, full = compare(original, changed)
    expected = 'corresponds' if change == 'add' else 'insufficient'
    assert old['material_results'][0]['result'] != expected
    assert full['material_results'][0]['result'] == expected
    assert incremental['material_results'] == full['material_results']
    assert any(row['type'] == 'material_insufficient' for row in full['issues']) is (change != 'add')
