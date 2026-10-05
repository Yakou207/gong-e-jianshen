"""Claim scope mechanisms; supplied candidates do not test live extraction quality."""
from copy import deepcopy
from pathlib import Path

import pytest

from aml_qc.contracts import ExtractionOutput
from aml_qc.core import query_transactions, resolve_entity, verify_claim
from aml_qc.ingest import load_case
from aml_qc.workflow import normalize_claims


DATA = Path(__file__).resolve().parents[1] / 'data' / 'synthetic'
ROLE_TEXT = '本期两笔向供货账户的付款为设备采购合同第1、第2期款。'


def scoped_case(text=ROLE_TEXT, token='supplier-account'):
    case = load_case(DATA / 'seed-01.json')
    case['documents'] = [{'document_id': 'narrative', 'revision': '1', 'text': text}]
    case['entity_mappings'] = []
    case['counterparties'] = [{'counterparty_token': token, 'type': 'payment_account'}]
    row = case['transactions'][0]
    case['transactions'] = [
        {**row, 'transaction_id': 'payment-1', 'direction': 'out',
         'amount': '3000.00', 'counterparty_token': token},
        {**row, 'transaction_id': 'payment-2', 'direction': 'out',
         'amount': '2000.00', 'counterparty_token': token},
    ]
    return case


def normalized_count(case, reference):
    candidate = {'kind': 'count', 'operator': 'exact', 'value': 2, 'unit': '笔',
                 'direction': 'out', 'counterparty_ref': reference,
                 'start': case['coverage_start'], 'end': case['coverage_end'],
                 'quote': case['documents'][0]['text']}
    result = normalize_claims(case, {'claims': [candidate], 'unresolved': []})
    assert result['unresolved'] == []
    assert len(result['claims']) == 1
    return result['claims'][0]


def test_role_limited_count_survives_but_single_visible_token_does_not_resolve_it():
    case = scoped_case()
    claim = normalized_count(case, '供货账户')
    assert (claim['kind'], claim['value'], claim['direction'], claim['counterparty_ref']) == (
        'count', 2, 'out', '供货账户')
    assert claim['text'] == ROLE_TEXT
    assert claim['source']['span'] == [0, len(ROLE_TEXT)]
    assert resolve_entity(case, '供货账户')['execution_status'] == 'identity_unresolved'
    result = verify_claim(case, claim)
    assert result['execution_status'] == 'identity_unresolved'
    assert result['result'] == 'insufficient_evidence'
    assert result['observed'] is None
    scope = next(row for row in result['evidence'] if row['type'] == 'query_scope')
    assert scope['scope']['counterparty_ref'] == '供货账户'
    assert scope['transaction_ids'] == []


def test_confirmed_role_mapping_checks_its_object_without_unrelated_third_payment():
    case = scoped_case()
    case['entity_mappings'] = [{'mapping_id': 'confirmed-role', 'source_ref': '供货账户',
                               'target_token': 'supplier-account', 'confirmed': True, 'revision': '1'}]
    unrelated = deepcopy(case['transactions'][0])
    unrelated.update(transaction_id='unrelated-payment', counterparty_token='other-account')
    case['transactions'].append(unrelated)
    assert query_transactions(case, {'direction': 'out'})['metrics']['count'] == 3
    result = verify_claim(case, normalized_count(case, '供货账户'))
    assert result['execution_status'] == 'completed'
    assert result['result'] == 'supported'
    assert result['comparison']['actual'] == 2
    scope = next(row for row in result['evidence'] if row['type'] == 'query_scope')
    assert scope['transaction_ids'] == ['payment-1', 'payment-2']
    assert scope['scope']['counterparty_token'] == 'supplier-account'


def test_valid_empty_extraction_for_role_relationship_does_not_invent_facts():
    case = scoped_case('收款方为原付款方，相关返还记录见材料。')
    supplied = ExtractionOutput.model_validate({'claims': [], 'unresolved': []})
    before = deepcopy(case)
    assert normalize_claims(case, supplied.model_dump()) == {'claims': [], 'unresolved': []}
    assert case == before


@pytest.mark.parametrize('reference,token,needs_mapping', [
    ('供货账户有限公司', 'named-supplier-account', True),
    ('acct-供货-17', 'acct-供货-17', False),
])
def test_explicit_name_or_token_containing_role_word_is_not_deleted(reference, token, needs_mapping):
    case = scoped_case(f'本期向{reference}付款两笔。', token)
    if needs_mapping:
        case['entity_mappings'] = [{'mapping_id': 'confirmed-name', 'source_ref': reference,
                                   'target_token': token, 'confirmed': True, 'revision': '1'}]
    claim = normalized_count(case, reference)
    assert claim['counterparty_ref'] == reference
    assert claim['value'] == 2
    result = verify_claim(case, claim)
    assert result['execution_status'] == 'completed'
    assert result['result'] == 'supported'
    assert result['identity']['counterparty_token'] == token
