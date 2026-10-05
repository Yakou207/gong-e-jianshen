from copy import deepcopy
from pathlib import Path

import pytest

from aml_qc.core import check_coverage, check_materials, compute_features, query_transactions, resolve_entity, verify_claim
from aml_qc.depgraph import Evaluator, sources_for
from aml_qc.ingest import load_case

DATA = Path(__file__).resolve().parents[1] / 'data' / 'synthetic'


def sample(number=2):
    return load_case(DATA / f'seed-{number:02d}.json')


def claim(case, kind='count', value=1, **kwargs):
    doc = case['documents'][0]
    return {'claim_id':'claim-1', 'kind':kind, 'operator':'exact', 'value':value, 'direction':'out',
            'counterparty_ref':'乙公司', 'source':{'document_id':doc['document_id'],'revision':doc['revision'],'span':[0,len(doc['text'])]},
            'text':doc['text'], **kwargs}


def features(case):
    return {r['feature_code']:r for r in compute_features(case)}


def test_only_one_versus_three_complete_and_partial():
    case = sample()
    assert verify_claim(case, claim(case))['result'] == 'contradicted'
    case['coverage'][0]['status'] = 'partial'
    assert verify_claim(case, claim(case))['result'] == 'contradicted'
    case['transactions'] = [r for r in case['transactions'] if r['direction'] == 'in'] + [next(r for r in case['transactions'] if r['direction'] == 'out')]
    assert verify_claim(case, claim(case))['result'] == 'insufficient_evidence'
    assert verify_claim(case, claim(case, operator='at_least'))['result'] == 'supported'
    case['transactions'] = [r for r in case['transactions'] if r['direction'] == 'in']
    assert verify_claim(case, claim(case, operator='none', value=0))['result'] == 'insufficient_evidence'
    case['coverage'][0]['status'] = 'full'
    assert verify_claim(case, claim(case, operator='none', value=0))['result'] == 'supported'


def test_partial_ratio_and_third_outgoing_account_counterexamples():
    case = sample()
    assert features(case)['F1']['result'] == 'met'
    assert features(case)['F2']['result'] == 'met'
    case['coverage'][0]['status'] = 'partial'
    assert features(case)['F1']['result'] == 'undeterminable'
    assert features(case)['F2']['result'] == 'undeterminable'
    extra = deepcopy(case['transactions'][0]); extra.update(transaction_id='large-in', amount='9000.00')
    case['transactions'].append(extra)
    case['coverage'][0]['status'] = 'full'
    assert features(case)['F1']['result'] == 'not_met'
    assert features(case)['F2']['result'] == 'not_met'
    case = sample()
    for index, row in enumerate([r for r in case['transactions'] if r['direction'] == 'out']):
        row['counterparty_token'] = f'out-{index}'
    assert features(case)['F2']['result'] == 'not_met'


def test_case_partial_but_specific_window_full():
    case = sample()
    case['coverage'][0]['status'] = 'partial'
    full = deepcopy(case['coverage'][0]); full.update(status='full',end='2026-09-04T00:00:00+08:00')
    case['coverage'].append(full)
    assert features(case)['F1']['result'] == 'met'
    assert features(case)['F2']['result'] == 'undeterminable'
    assert check_coverage(case, end='2026-09-04T00:00:00+08:00')['status'] == 'full'


def test_explicit_missing_range_overrides_broad_full_declaration():
    case = sample()
    case['coverage'][0]['missing_ranges'] = [{'start':'2026-09-01T12:00:00+08:00','end':'2026-09-01T13:00:00+08:00','fields':['amount']}]
    assert features(case)['F1']['windows'][0]['result'] == 'undeterminable'
    assert check_coverage(case, fields=['timestamp'])['status'] == 'full'


def test_zero_denominator_empty_queries_have_scope_and_coverage():
    case = sample(); case['transactions'] = []
    result = features(case)
    assert result['F1']['result'] == 'not_met'
    assert result['F2']['result'] == 'not_met'
    assert result['F1']['windows'][0]['metrics']['out_in_ratio'] is None
    query = query_transactions(case)
    assert query['scope']['transaction_set_hash']
    assert query['coverage']['status'] == 'full'
    case['coverage'][0]['status'] = 'partial'
    assert features(case)['F1']['result'] == 'undeterminable'


def test_half_open_and_midnight_boundaries():
    case = sample()
    row = deepcopy(case['transactions'][0]); row.update(transaction_id='midnight',timestamp='2026-09-02T00:00:00+08:00')
    case['transactions'].append(row)
    first = query_transactions(case, {'end':'2026-09-02T00:00:00+08:00'})
    second = query_transactions(case, {'start':'2026-09-02T00:00:00+08:00','end':'2026-09-03T00:00:00+08:00'})
    assert 'midnight' not in first['transaction_ids']
    assert 'midnight' in second['transaction_ids']
    case['coverage_start'] = '2026-09-01T08:00:00+08:00'
    case['coverage_end'] = '2026-09-08T10:00:00+08:00'
    f = features(case)
    assert f['F1']['windows'][0]['complete_duration'] is False
    assert f['F1']['windows'][0]['result'] == 'undeterminable'
    assert f['F1']['windows'][-1]['complete_duration'] is False
    assert f['F2']['windows'][0]['end'] == '2026-09-08T08:00:00+08:00'
    assert f['F2']['windows'][-1]['result'] == 'undeterminable'


def test_less_than_three_days_and_no_applicable_window():
    case = sample(); case['coverage_end'] = '2026-09-03T00:00:00+08:00'
    assert features(case)['F1']['result'] == 'not_met'
    assert features(case)['F2']['result'] == 'undeterminable'
    case['coverage_end'] = case['coverage_start']
    assert features(case)['F1']['result'] == 'undeterminable'


def test_precise_identity_and_same_revision_mapping_change():
    case = sample(4)
    assert resolve_entity(case, '乙公司')['execution_status'] == 'identity_unresolved'
    assert resolve_entity(case, {'credit_code':'SYNTHETIC-B'})['counterparty_token'] == 'supplier-b'
    assert verify_claim(case, claim(case))['execution_status'] == 'identity_unresolved'
    mapping = {'mapping_id':'map','source_ref':'乙公司','target_token':'supplier-b','confirmed':True,'revision':'1'}
    case['entity_mappings'] = [mapping]
    first = resolve_entity(case, '乙公司')
    mapping['target_token'] = 'supplier-b2'
    second = resolve_entity(case, '乙公司')
    assert first['basis'] != second['basis']
    assert verify_claim(case, claim(case))['result'] == 'contradicted'


def test_amount_counterparty_and_time_claims():
    case = sample()
    assert verify_claim(case, claim(case, 'amount_sum', '1050.00'))['result'] == 'supported'
    assert verify_claim(case, claim(case, 'amount_sum', '1000.00'))['result'] == 'contradicted'
    assert verify_claim(case, claim(case, 'counterparty', ['supplier-b'], counterparty_ref=None))['result'] == 'supported'
    outside = {'start':'2026-09-01T00:00:00+08:00','end':'2026-09-03T00:00:00+08:00'}
    assert verify_claim(case, claim(case, 'time_range', outside))['result'] == 'contradicted'
    inside = {'start':'2026-09-01T00:00:00+08:00','end':'2026-09-04T00:00:00+08:00'}
    assert verify_claim(case, claim(case, 'time_range', inside))['result'] == 'supported'
    case['coverage'][0]['status'] = 'partial'
    assert verify_claim(case, claim(case, 'time_range', inside))['result'] == 'insufficient_evidence'


def test_changed_document_span_is_not_accepted():
    case = sample(); extracted = claim(case)
    case['documents'][0]['text'] = '新增：' + case['documents'][0]['text']
    assert verify_claim(case, extracted)['execution_status'] == 'extraction_failed'
    case = sample(); extracted = claim(case); case['documents'][0]['revision'] = '2'
    assert verify_claim(case, extracted)['execution_status'] == 'extraction_failed'


def test_unreliable_fields_cannot_supply_counterexample():
    case = sample(); case['coverage'][0].update(status='partial', reliable=False)
    assert verify_claim(case, claim(case))['result'] == 'insufficient_evidence'


def test_installment_material_not_forced_equal_to_contract():
    case = sample(5)
    assert check_materials(case)[0]['result'] == 'pending_judgement'
    case['material_links'][0]['relation_template'] = 'contract_installments'
    assert check_materials(case)[0]['result'] == 'corresponds'
    case['material_links'][0]['relation_template'] = 'single_purchase_payment'
    assert check_materials(case)[0]['result'] == 'mismatch'


def test_missing_material_addition_and_link_only_change():
    case = sample(3)
    assert check_materials(case)[0]['result'] == 'insufficient'
    case['materials'] = sample(1)['materials']
    assert check_materials(case)[0]['result'] == 'corresponds'
    case['material_links'][0]['revision'] = '2'
    assert check_materials(case)[0]['result'] == 'insufficient'
    case = sample(2)
    assert check_materials(case)[0]['result'] == 'corresponds'
    case['material_links'][0]['relation_template'] = 'single_purchase_payment'
    assert check_materials(case)[0]['result'] == 'mismatch'


def test_material_subject_period_and_missing_fields():
    case = sample(1); case['materials'][0]['subject']['account_id'] = 'another-account'
    assert check_materials(case)[0]['result'] == 'mismatch'
    case = sample(1); case['materials'][0]['period']['start'] = '2026-09-02T00:00:00+08:00'
    assert check_materials(case)[0]['result'] == 'mismatch'
    case = sample(1); del case['materials'][0]['amount']
    assert check_materials(case)[0]['result'] == 'insufficient'


def test_duplicate_transaction_not_double_counted_and_conflict_rejected():
    case = sample(1); row = deepcopy(case['transactions'][-1]); case['transactions'].append(row)
    assert query_transactions(case, {'direction':'out'})['metrics']['amount_sum_cents'] == 105000
    row['amount'] = '999.00'
    with pytest.raises(ValueError, match='冲突'):
        query_transactions(case)


def test_coverage_with_no_complete_day_is_undeterminable():
    case = sample(); case['coverage_start'] = '2026-09-01T08:00:00+08:00'; case['coverage_end'] = '2026-09-01T16:00:00+08:00'
    assert features(case)['F1']['result'] == 'undeterminable'


def test_count_coverage_does_not_require_amount_field():
    case = sample(1); case['coverage'][0]['fields'].remove('amount')
    assert verify_claim(case, claim(case))['result'] == 'supported'
    assert verify_claim(case, claim(case,'amount_sum','1050.00'))['result'] == 'insufficient_evidence'


def test_absent_material_references_set_and_all_material_paths_resolve():
    case = sample(3)
    assert any(e['type']=='material_set' for e in check_materials(case)[0]['evidence'])
    case = sample(1)
    evidence = next(e for e in check_materials(case)[0]['evidence'] if e['type']=='material')
    for path in evidence['field_paths']:
        value = case['materials'][0]
        for component in path.split('.'):
            value = value[component]


def test_unresolved_identity_has_unknown_observed_count_not_zero():
    case = sample(4)
    result = verify_claim(case, claim(case))
    assert result['execution_status'] == 'identity_unresolved'
    assert result['result'] == 'insufficient_evidence'
    assert result['observed'] is None
    assert '空返回不代表零笔交易' in result['reason']
    # The statement is unresolved despite an available outgoing transaction.
    assert any(row['direction'] == 'out' for row in case['transactions'])
    direct = query_transactions(case, {'counterparty_token': 'unrecognized-name'})
    assert direct['execution_status'] == 'identity_unresolved'
    assert direct['metrics'] is None


def test_query_rejects_reversed_interval():
    case = sample()
    with pytest.raises(ValueError, match='非空半开区间'):
        query_transactions(case, {'start': case['coverage_end'], 'end': case['coverage_start']})


def test_transaction_id_query_returns_exact_record_and_bound_scope():
    case = sample()
    row = next(r for r in case['transactions'] if r['direction'] == 'out')
    result = query_transactions(case, {'transaction_id': row['transaction_id']})
    assert result['rows'] == [row]
    assert result['transaction_ids'] == [row['transaction_id']]
    assert result['metrics']['count'] == 1
    assert result['scope']['transaction_id'] == row['transaction_id']
    assert result['scope']['account_id'] == case['subject_account_id']
    assert result['scope']['start'] == case['coverage_start']
    assert result['scope']['end'] == case['coverage_end']
    assert 'transaction_id' not in query_transactions(case)['scope']


def test_missing_transaction_id_keeps_exact_empty_scope_and_distinct_fingerprint():
    first = query_transactions(sample(), {'transaction_id': 'missing-first'})
    second = query_transactions(sample(), {'transaction_id': 'missing-second'})
    for result in (first, second):
        assert result['rows'] == result['transaction_ids'] == []
        assert result['execution_status'] == 'completed'
        assert result['metrics']['count'] == 0 and result['coverage']['status'] == 'full'
    assert first['scope']['transaction_id'] == 'missing-first'
    assert second['scope']['transaction_id'] == 'missing-second'
    assert first['query_id'] != second['query_id']


def test_transaction_id_lookup_exposes_actual_counterparty_and_intersects_filter():
    case = sample()
    row = next(r for r in case['transactions'] if r['direction'] == 'out')
    row['counterparty_token'] = 'payer-00'
    actual = query_transactions(case, {'transaction_id': row['transaction_id']})
    assert actual['rows'] == [row] and actual['metrics']['counterparty_tokens'] == ['payer-00']
    filtered = query_transactions(case, {'transaction_id': row['transaction_id'], 'counterparty_token': 'supplier-b'})
    assert filtered['rows'] == [] and filtered['metrics']['count'] == 0
    assert filtered['scope']['transaction_id'] == row['transaction_id']
    assert filtered['scope']['counterparty_token'] == 'supplier-b'
    assert actual['query_id'] != filtered['query_id']


@pytest.mark.parametrize('direction, expected_count', [('out', 1), ('in', 0)])
def test_transaction_id_does_not_override_direction(direction, expected_count):
    case = sample()
    row = next(r for r in case['transactions'] if r['direction'] == 'out')
    result = query_transactions(case, {'transaction_id': row['transaction_id'], 'direction': direction})
    assert result['metrics']['count'] == expected_count
    assert result['scope']['direction'] == direction


@pytest.mark.parametrize('window', [
    {'start': '2026-09-02T00:00:00+08:00'},
    {'end': '2026-09-01T15:00:00+08:00'},
])
def test_transaction_id_does_not_expand_requested_half_open_window(window):
    case = sample()
    result = query_transactions(case, {'transaction_id': 'seed-02-out-00', **window})
    assert result['rows'] == [] and result['metrics']['count'] == 0
    assert all(result['scope'][key] == value for key, value in window.items())


def test_transaction_id_does_not_expand_case_account_or_default_period():
    case = sample()
    row = next(r for r in case['transactions'] if r['direction'] == 'out')
    with pytest.raises(ValueError, match='本案账户'):
        query_transactions(case, {'transaction_id': row['transaction_id'], 'account_id': 'another-account'})
    row['timestamp'] = case['coverage_end']
    assert query_transactions(case, {'transaction_id': row['transaction_id']})['rows'] == []
    row['timestamp'] = case['coverage_start']
    row['account_id'] = 'another-account'
    assert query_transactions(case, {'transaction_id': row['transaction_id']})['rows'] == []


@pytest.mark.parametrize('transaction_id', [None, '', '  ', 1, True, [], {}])
def test_transaction_id_requires_nonempty_string(transaction_id):
    with pytest.raises(ValueError, match='transaction_id'):
        query_transactions(sample(), {'transaction_id': transaction_id})


def test_transaction_id_deleted_record_incremental_matches_independent_full_empty_result():
    case = sample()
    transaction_id = next(r['transaction_id'] for r in case['transactions'] if r['direction'] == 'out')
    query = {'transaction_id': transaction_id}
    dependencies = ['source:metadata', 'source:transactions', 'source:coverage', 'source:entities', 'source:execution']
    original = Evaluator(sources_for(case, {}, {'fixture': 'exact-transaction-query'}))
    before = original.evaluate('lookup', 'query_transactions', query, dependencies,
                               lambda: query_transactions(case, query))
    old_snapshot = deepcopy(original.snapshot())
    changed = deepcopy(case)
    changed['transactions'] = [r for r in changed['transactions'] if r['transaction_id'] != transaction_id]
    results = []
    for strategy in ('incremental', 'full'):
        evaluator = Evaluator(sources_for(changed, {}, {'fixture': 'exact-transaction-query'}),
                              old_snapshot if strategy == 'incremental' else None, strategy)
        result = evaluator.evaluate('lookup', 'query_transactions', query, dependencies,
                                    lambda: query_transactions(changed, query))
        assert evaluator.reused == 0 and evaluator.recomputed == 1
        assert result['rows'] == [] and result['metrics']['count'] == 0
        assert result['scope']['transaction_id'] == transaction_id
        assert result['scope']['transaction_set_hash'] != before['scope']['transaction_set_hash']
        results.append(result)
    assert results[0] == results[1]
    assert original.snapshot() == old_snapshot
