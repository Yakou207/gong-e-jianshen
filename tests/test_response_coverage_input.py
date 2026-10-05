"""Response task projection mechanics; no model quality or human-reference claim."""
from copy import deepcopy

from aml_qc import workflow
from test_model_safety import case


def projection(value, checks=None):
    return workflow.semantic_input(value, checks or {})['response_coverage_input']


def test_response_input_does_not_mix_material_or_transaction_findings():
    original=case();changed=deepcopy(original)
    changed['transactions'][0]['amount']='98765.43'
    changed['materials'][0]['amount']='54321.00'
    changed['coverage'][0]['status']='partial'
    changed['case_id']='different-evidence-variant'
    assert projection(original,{'claim_results':[]})==projection(changed,{'claim_results':[{'claim_id':'other','execution_status':'completed'}]})
    assert 'transactions' not in projection(changed) and 'materials' not in projection(changed) and 'checks' not in projection(changed)


def test_original_narrative_and_required_focus_change_response_task():
    original=case();changed=deepcopy(original)
    next(d for d in changed['documents'] if d['document_id']=='narrative')['text']='新的原文，仍需真实模型判断。'
    assert projection(original)!=projection(changed)
    changed=deepcopy(original);changed['alert']['original_focus']='新的需回应事项。';changed['alert'].pop('focuses',None)
    assert projection(original)!=projection(changed)


def test_task_scope_change_is_not_frozen_out_of_response_judgment():
    original=case();changed=deepcopy(original)
    changed['review_scope']['target_labels']=['alert_response']
    assert projection(original)!=projection(changed)
    changed=deepcopy(original);changed['coverage_end']='2026-10-01T00:00:00+08:00'
    assert projection(original)!=projection(changed)


def test_response_label_definition_tracks_effective_case_schema():
    original=case();changed=deepcopy(original)
    changed['schema']=workflow.default_schema()
    changed['schema']['labels']['alert_response']['definition']='不同任务规范下的回应定义。'
    assert projection(original)!=projection(changed)


def test_authored_requirements_are_isolated_copied_and_change_response_input():
    original = case()
    changed = deepcopy(original)
    changed['alert']['focuses'][0]['response_requirements'] = [
        {'kind': 'verification_result', 'text': '报告规定范围的核对结果。'}]
    projected = projection(changed)
    assert projected != projection(original)
    assert projected['focuses'][0]['response_requirements'] == changed['alert']['focuses'][0]['response_requirements']
    projected['focuses'][0]['response_requirements'][0]['text'] = 'caller changed projection'
    assert changed['alert']['focuses'][0]['response_requirements'][0]['text'] == '报告规定范围的核对结果。'
    changed['materials'] = []
    assert projection(changed)['focuses'][0]['response_requirements'][0]['text'] == '报告规定范围的核对结果。'
