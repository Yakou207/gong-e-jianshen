from copy import deepcopy
from pathlib import Path

import pytest

from aml_qc.core import compute_features, load_schema
from aml_qc.ingest import load_case, validate_case
from aml_qc.schema import validate_schema

DATA = Path(__file__).resolve().parents[1] / 'data/synthetic/seed-01.json'


def test_frozen_default_schema_valid():
    assert validate_schema(load_schema()) == []


@pytest.mark.parametrize('code,key,value', [
    ('F2','window_days',0), ('F2','window_days',-1), ('F2','window_days',True),
    ('F1','minimum_days',0), ('F1','ratio_denominator',0), ('F2','ratio_numerator',-1),
    ('F2','minimum_in_counterparties',0), ('F2','maximum_out_counterparties',0),
    ('F2','maximum_out_counterparties',0.5),
])
def test_invalid_feature_parameters_rejected_before_window_iteration(code,key,value):
    schema=load_schema(); schema['features'][code][key]=value
    assert validate_schema(schema)
    with pytest.raises(ValueError):
        compute_features(load_case(DATA),schema)


def test_counterparty_bounds_and_template_relations_are_implemented_only():
    schema=load_schema(); schema['features']['F2']['minimum_out_counterparties']=3
    assert validate_schema(schema)
    for field,value in [('amount_relation','approximate'),('period_relation','overlap'),('amount_tolerance_cents',-1),('transaction_count',0)]:
        schema=load_schema(); schema['material_templates']['single_purchase_payment'][field]=value
        assert validate_schema(schema)


@pytest.mark.parametrize('targets',[None,[],['unknown'],['F1','F1'],'F1',[None]])
def test_empty_unknown_duplicate_or_invalid_target_labels_rejected(targets):
    case=load_case(DATA); case['review_scope']['target_labels']=targets
    assert not validate_case(case)['valid']


def test_annotation_only_needs_explicit_scope_but_alert_review_has_default():
    case=load_case(DATA); del case['review_scope']
    assert validate_case(case)['valid']
    case['task_mode']='annotation_only'
    assert not validate_case(case)['valid']
    case['review_scope']={'target_labels':['count']}
    assert validate_case(case)['valid']


def test_embedded_schema_and_case_version_must_match():
    case=load_case(DATA); case['schema']=load_schema(); case['schema_version']='S2.0'
    assert not validate_case(case)['valid']
    case['schema']['schema_version']='S2.0'
    assert validate_case(case)['valid']
