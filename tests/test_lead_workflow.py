"""Real workflow paths with deterministic model fixtures, not live-model quality."""
from copy import deepcopy
from pathlib import Path

import pytest

from aml_qc.annotations import validate_evidence
from aml_qc.depgraph import business_result, digest
from aml_qc.ingest import load_case
from aml_qc.leads import lead_context_hash
from aml_qc.llm import FrozenModel
from aml_qc.workflow import run_review
from test_model_safety import MODEL, ScriptedModel, json_message, response_message, semantic_response, tool_message


SUGGESTION = {"observation": "客户资料只给出了经营类别，具体结算安排仍需人工判断。",
              "question": "是否需要补充核实经营类别与结算安排的关系？", "basis_refs": ["document:kyc"]}


def lead_case():
    value = load_case(Path(__file__).resolve().parents[1] / 'data/synthetic/seed-01.json')
    value['review_scope']['target_labels'] = ['F1', 'F2', 'alert_response']
    return value


def lead_model(value, suggestion=None, *, mode='fixed', tool=None):
    answer = semantic_response(value)
    answer['leads'] = [deepcopy(SUGGESTION if suggestion is None else suggestion)]
    responses = [json_message({'claims': [], 'unresolved': []}), response_message(value)]
    if mode == 'agent':
        if tool:
            responses.append(tool)
        responses.append({"role": "assistant", "content": "取证结束。"})
    responses.append(json_message({'gaps':answer['gaps'],'leads':answer['leads']}))
    return ScriptedModel(responses)


def test_workflow_produces_bound_notice_without_creating_required_focus():
    value = lead_case()
    result = run_review(value, provider='frozen', model=lead_model(value))
    lead, = result['lead_candidates']
    issue, = result['issues']
    assert issue['type'] == 'new_lead' and issue['severity'] == 'notice'
    assert issue['lead_id'] == lead['lead_id'] and result['open_items'] == []
    assert {f['focus_id'] for f in result['semantic_results']} == {'focus-1'}
    assert lead['context_hash'] == lead_context_hash(value)
    assert lead['execution_fingerprint'] == digest(result['execution'])
    assert lead['model_draft']['status'] == 'unverified_model_suggestion' and lead['novelty_status'] == 'unconfirmed'
    kyc = next(d for d in value['documents'] if d['document_id'] == 'kyc')
    evidence, = lead['evidence']
    assert evidence['document_id'] == 'kyc' and evidence['revision'] == kyc['revision']
    assert kyc['text'][slice(*evidence['span'])] == evidence['text']
    assert result['run_status'] == 'completed'


@pytest.mark.parametrize('change', ['unknown_basis', 'empty_refs', 'duplicate_refs', 'whitespace', 'invented_revision', 'original_focus'])
def test_bad_lead_payload_leaves_required_review_failed_not_optional(change):
    value = lead_case(); suggestion = deepcopy(SUGGESTION)
    if change == 'unknown_basis': suggestion['basis_refs'] = ['document:absent']
    elif change == 'empty_refs': suggestion['basis_refs'] = []
    elif change == 'duplicate_refs': suggestion['basis_refs'] *= 2
    elif change == 'whitespace': suggestion['question'] = '   '
    elif change == 'invented_revision': suggestion['revision'] = 'invented'
    else: suggestion['question'] = value['alert']['focuses'][0]['text']
    result = run_review(value, provider='frozen', model=lead_model(value, suggestion))
    assert result['lead_candidates'] == []
    assert any(c['check_id'] == 'semantic' and c['status'] == 'failed' for c in result['required_checks'])
    assert any(i['type'] == 'execution_failed' for i in result['issues'])
    assert result['model_requests'][-1]['status'] == 'frozen'


def test_agent_must_actually_receive_successful_query_before_citing_its_result():
    value = lead_case(); args = {'direction': 'out'}
    ref = 'tool:query_transactions:' + digest(args)[:20]
    suggestion = {**SUGGESTION, 'basis_refs': [ref]}
    early = run_review(value, provider='frozen', model=lead_model(value, suggestion))
    assert early['lead_candidates'] == []
    model = lead_model(value, suggestion, mode='agent', tool=tool_message('query_transactions', args))
    actual = run_review(value, provider='frozen', mode='agent', model=model)
    lead, = actual['lead_candidates']
    proof = next(e for e in lead['evidence'] if e['type'] == 'tool_result')
    assert proof['result_ref'] == ref and proof['result']['metrics']['count'] == 1
    assert proof['content_hash'] == digest(proof['result'])
    assert actual['stats']['adaptive_tool_calls'] == 1
    messages = [r['request']['messages'] for r in model.calls]
    assert any(ref in item.get('content', '') for message in messages for item in message if item.get('role') == 'tool')


def test_missing_tool_document_cannot_be_used_as_observed_evidence():
    value = lead_case(); args = {'document_id': 'absent'}
    suggestion = {**SUGGESTION, 'basis_refs': ['tool:read_document:' + digest(args)[:20]]}
    result = run_review(value, provider='frozen', mode='agent',
                        model=lead_model(value, suggestion, mode='agent', tool=tool_message('read_document', args)))
    assert result['lead_candidates'] == [] and result['run_status'] == 'partial'


def test_changed_observation_creates_distinct_candidate_not_a_silent_rewrite():
    value = lead_case()
    first = run_review(value, provider='frozen', model=lead_model(value))['lead_candidates'][0]
    second = run_review(value, provider='frozen', model=lead_model(value, {**SUGGESTION, 'observation': '不同的模型观察草稿。'}))['lead_candidates'][0]
    assert first['lead_id'] != second['lead_id'] and first['basis_fingerprint'] == second['basis_fingerprint']


def test_changed_read_document_rebinds_lead_and_full_incremental_business_agrees():
    old = lead_case(); new = deepcopy(old)
    next(d for d in new['documents'] if d['document_id'] == 'kyc')['text'] += ' 新增经营周期资料。'
    products = {}
    for value in (old, new):
        model = lead_model(value, mode='agent')
        run_review(value, provider='frozen', mode='agent', model=model)
        products.update(model.request_products)
    before = run_review(old, provider='frozen', mode='agent', model=FrozenModel(products, MODEL))
    inc = run_review(new, provider='frozen', mode='agent', strategy='incremental', previous=before['snapshot'], model=FrozenModel(products, MODEL))
    full = run_review(new, provider='frozen', mode='agent', model=FrozenModel(products, MODEL))
    assert business_result(inc) == business_result(full) and full['stats']['reused'] == 0
    prior, current = before['lead_candidates'][0], inc['lead_candidates'][0]
    assert prior['lead_id'] == current['lead_id']
    assert prior['basis_fingerprint'] != current['basis_fingerprint'] and prior['context_hash'] != current['context_hash']
    assert current['evidence'][0]['text'].endswith('新增经营周期资料。')


def test_upgraded_focus_reference_binds_scope_not_original_alert():
    value = lead_case()
    focus = {'focus_id': 'lead-focus', 'text': '人工决定核查的问题', 'lead_id': 'lead-id', 'origin_issue_id': 'original-issue'}
    value['review_scope']['upgraded_leads'] = [focus]
    result = run_review(value, provider='frozen', model=lead_model(value))
    row = next(r for r in result['semantic_results'] if r['focus_id'] == 'lead-focus')
    ref = next(e for e in row['evidence'] if e['type'] == 'upgraded_focus')
    assert ref['content_hash'] == digest(focus) and 'revision' not in ref
    assert not any(e.get('type') == 'alert_focus' for e in row['evidence'])
    assert validate_evidence(value, [ref], [ref]) == [ref]
    changed = deepcopy(value); changed['review_scope']['upgraded_leads'][0]['text'] += '（已修订）'
    with pytest.raises(ValueError): validate_evidence(changed, [ref], [ref])
