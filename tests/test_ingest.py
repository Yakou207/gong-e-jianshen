from copy import deepcopy
import json
from pathlib import Path

import pytest

from aml_qc.ingest import amount_cents, load_case, validate_case

DATA = Path(__file__).resolve().parents[1] / 'data' / 'synthetic'


def test_all_synthetic_seed_packages_valid_and_without_answers():
    files = sorted(DATA.glob('seed-*.json'))
    assert len(files) == 6
    for path in files:
        case = load_case(path)
        assert validate_case(case)['valid']
        assert not case.get('claims')
        assert case['profile']['data_origin'] == 'synthetic'


@pytest.mark.parametrize('value,expected',[('0',0),('0.01',1),('1050.00',105000),('9999999999999999.99',999999999999999999)])
def test_exact_decimal_amounts(value, expected):
    assert amount_cents(value) == expected


@pytest.mark.parametrize('value',[0.1,True,'-1','NaN','Infinity','0.001','no amount'])
def test_invalid_or_inexact_amounts(value):
    with pytest.raises(ValueError):
        amount_cents(value)


def test_conflicting_duplicate_rejected_identical_warned(tmp_path):
    case = load_case(DATA / 'seed-01.json')
    row = deepcopy(case['transactions'][0]); case['transactions'].append(row)
    assert validate_case(case)['valid']
    assert validate_case(case)['warnings']
    path=tmp_path/'case.json'; path.write_text(json.dumps(case))
    assert len(load_case(path)['transactions']) == len(case['transactions'])-1
    row['amount'] = '9.00'
    assert not validate_case(case)['valid']
    path.write_text(json.dumps(case))
    with pytest.raises(ValueError,match='重复'):
        load_case(path)


def test_half_open_coverage_and_foreign_account_enforced():
    case = load_case(DATA / 'seed-01.json')
    case['transactions'][0]['timestamp'] = case['coverage_end']
    assert not validate_case(case)['valid']
    case = load_case(DATA / 'seed-01.json'); case['transactions'][0]['account_id'] = 'another-account'
    assert not validate_case(case)['valid']


def test_hidden_answers_rejected_even_nested():
    case = load_case(DATA / 'seed-01.json'); case['profile']['ground_truth'] = {'count':1}
    assert not validate_case(case)['valid']


def test_missing_alert_is_valid_input_but_incomplete_task():
    case = load_case(DATA / 'seed-01.json'); del case['alert']
    result = validate_case(case)
    assert result['valid']
    assert any('原预警' in warning for warning in result['warnings'])


def test_invalid_collection_is_validation_failure():
    case = load_case(DATA / 'seed-01.json'); case['transactions'] = None
    assert not validate_case(case)['valid']
