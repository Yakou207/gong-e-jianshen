"""Reference preparation mechanics; no answers, personnel or model calls."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from aml_qc import llm
from scripts.prepare_reference_workpack import prepare_reference_workpack


def write_json(path, value):
    raw = (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Reference preparation must not read credentials or call models')
    monkeypatch.setattr(llm, 'settings', forbidden)
    monkeypatch.setattr(llm.httpx, 'post', forbidden)


@pytest.fixture
def inputs(tmp_path):
    # Hand-written input syntax fixture, not a generated AML business reference.
    case = {'case_id': 'manual-case-1', 'case_family': 'manual-family', 'task_mode': 'annotation_only',
            'subject_account_id': 'manual-account', 'profile': {'data_origin': 'synthetic'},
            'data_version': '1', 'coverage_start': '2026-09-01T00:00:00+08:00',
            'coverage_end': '2026-09-02T00:00:00+08:00', 'currency': 'CNY',
            'timezone': 'Asia/Shanghai', 'schema_version': 'S1.0', 'transactions': [],
            'counterparties': [], 'entity_mappings': [], 'coverage': [],
            'documents': [{'document_id': 'narrative', 'revision': '1', 'source': 'synthetic',
                           'text': '手写输入格式机制夹具，不是业务参考答案。'}],
            'materials': [], 'material_links': [], 'review_scope': {'target_labels': ['count']}}
    case_path = tmp_path / 'cases' / 'input.json'
    row = {'case_id': case['case_id'], 'path': 'cases/input.json', 'sha256': write_json(case_path, case)}
    index = tmp_path / 'case-index.json'
    write_json(index, {'cases': [row]})
    return index, case_path, case, row, tmp_path / 'workpack'


def test_two_reviewer_packets_have_only_visible_cases_and_empty_current_contract(inputs):
    index, case_path, case, row, output = inputs
    original = case_path.read_bytes()
    result = prepare_reference_workpack(case_index=index, output_dir=output)
    manifest = json.loads((output / 'preparation-manifest.json').read_text())
    assert result['case_count'] == 1 and manifest['status'] == 'preparation_only_not_references'
    prepared = manifest['cases'][0]
    assert prepared['case_sha256'] == row['sha256']
    assert prepared['visible_case_sha256'] != row['sha256']
    assert manifest['projection'] == 'baseline.raw_case_input closed visible-field whitelist'
    for reviewer in ('reviewer-1', 'reviewer-2'):
        visible = json.loads((output / reviewer / 'cases' / prepared['filename']).read_text())
        assert visible['documents'] == case['documents'] and visible['case_id'] == case['case_id']
        assert 'case_family' not in visible and 'split' not in visible
        reference = json.loads((output / reviewer / 'references' / prepared['filename']).read_text())
        assert reference['case_sha256'] == row['sha256'] and reference['check_units'] == []
        assert reference['visible_case_sha256'] == prepared['visible_case_sha256']
        assert reference['visible_case_sha256'] == hashlib.sha256(
            (output / reviewer / 'cases' / prepared['filename']).read_bytes()).hexdigest()
        assert reference['reviewers'] == [{'person_id': None, 'signed_at': None, 'exposure': None, 'authorship_bias_disclosure': None}]
    final = json.loads((output / 'final-references' / prepared['filename']).read_text())
    assert set(final) == {'case_id', 'case_sha256', 'visible_case_sha256', 'reviewers', 'check_units'}
    assert final['visible_case_sha256'] == prepared['visible_case_sha256']
    assert len(final['reviewers']) == 2 and all(v is None for p in final['reviewers'] for v in p.values())
    adjudication = json.loads((output / 'adjudication' / prepared['filename']).read_text())
    assert adjudication['decisions'] == [] and adjudication['status'] is None
    assert adjudication['visible_case_sha256'] == prepared['visible_case_sha256']
    assert adjudication['adjudicator'] == {'person_id': None, 'signed_at': None, 'exposure': None, 'authorship_bias_disclosure': None}
    exposure = json.loads((output / 'exposure-declaration.template.json').read_text())
    assert exposure['saw_generator_private'] is None and all(v is None for v in exposure.values())
    assert json.loads((output / 'reference-protocol.template.json').read_text())['reference_protocol'] is None
    generic = json.loads((output / 'reference-record.template.json').read_text())
    assert generic['case_sha256'] is None and generic['visible_case_sha256'] is None
    assert case_path.read_bytes() == original
    assert not list(output.rglob('*.md')) and not list(output.rglob('README*'))


def test_only_explicit_check_ids_are_inserted_without_labels_answers_or_declarations(inputs):
    index, _, _, row, output = inputs
    row['required_check_ids'] = ['manual-unit-one', 'manual-unit-two']
    write_json(index, {'cases': [row]})
    prepare_reference_workpack(case_index=index, output_dir=output)
    final = json.loads(next((output / 'final-references').glob('*.json')).read_text())
    assert [u['reference_check_id'] for u in final['check_units']] == row['required_check_ids']
    for unit in final['check_units']:
        assert unit['evidence_sets'] == []
        assert all(v is None for k, v in unit.items() if k not in {'reference_check_id', 'evidence_sets'})


@pytest.mark.parametrize('mutation', ['root_private', 'nested_model', 'nested_reference', 'scalar_smuggling', 'invalid_currency'])
def test_nonvisible_or_invalid_case_fails_before_output_creation(inputs, mutation):
    index, case_path, case, row, output = inputs
    if mutation == 'root_private': case['private'] = {'truth': 'do-not-copy'}
    elif mutation == 'nested_model': case['documents'][0]['model_output'] = 'do-not-copy'
    elif mutation == 'nested_reference': case['profile']['reference_value'] = 'do-not-copy'
    elif mutation == 'scalar_smuggling': case['documents'][0]['text'] = {'private': 'do-not-copy'}
    else: case['currency'] = 'USD'
    row['sha256'] = write_json(case_path, case)
    write_json(index, {'cases': [row]})
    with pytest.raises(ValueError): prepare_reference_workpack(case_index=index, output_dir=output)
    assert not output.exists()


@pytest.mark.parametrize('mutation', ['hash', 'case_id', 'duplicate', 'empty', 'condition', 'two_paths'])
def test_invalid_explicit_index_or_case_binding_fails_without_outputs(inputs, mutation):
    index, _, _, row, output = inputs
    rows = [row]
    if mutation == 'hash': row['sha256'] = '0' * 64
    elif mutation == 'case_id': row['case_id'] = 'wrong-case'
    elif mutation == 'duplicate': rows.append(deepcopy(row))
    elif mutation == 'empty': rows = []
    elif mutation == 'condition': row['condition'] = 'hidden-variant'
    else: row['file'] = row['path']
    write_json(index, {'cases': rows})
    with pytest.raises(ValueError): prepare_reference_workpack(case_index=index, output_dir=output)
    assert not output.exists()


def test_explicit_assembly_file_alias_is_supported_without_directory_search(inputs):
    index, _, _, row, output = inputs
    row['file'] = row.pop('path')
    write_json(index, {'cases': [row]})
    assert prepare_reference_workpack(case_index=index, output_dir=output)['case_count'] == 1


def test_visible_material_transaction_and_order_ids_are_not_silently_dropped(inputs):
    index, case_path, case, row, output = inputs
    case['materials'] = [{'material_id': 'manual-material', 'revision': '1', 'material_type': 'synthetic_voucher',
                          'source': 'synthetic', 'text': '手写字段投影机制夹具。', 'order_id': 'manual-order',
                          'original_payment_transaction_id': 'manual-original', 'refund_transaction_id': 'manual-return'}]
    row['sha256'] = write_json(case_path, case)
    write_json(index, {'cases': [row]})
    prepare_reference_workpack(case_index=index, output_dir=output)
    visible = json.loads(next((output / 'reviewer-1' / 'cases').glob('*.json')).read_text())
    assert visible['materials'] == case['materials']


@pytest.mark.parametrize('raw', ['{"cases":[],"cases":[]}', '{"cases":[],"other":NaN}'])
def test_ambiguous_or_nonfinite_json_is_rejected_without_output(inputs, raw):
    index, _, _, _, output = inputs
    index.write_text(raw)
    with pytest.raises(ValueError): prepare_reference_workpack(case_index=index, output_dir=output)
    assert not output.exists()


@pytest.mark.parametrize('blocked', ['private', '.env', 'symlink', 'references', 'model-products'])
def test_private_reference_model_and_credential_paths_are_rejected_before_read(inputs, monkeypatch, blocked):
    index, case_path, _, row, output = inputs
    if blocked == 'symlink':
        target = index.parent / '.env'; target.write_text('not-a-real-secret')
        path = index.parent / 'alias.json'; path.symlink_to(target)
    elif blocked == '.env':
        path = index.parent / '.env'; path.write_text('not-a-real-secret')
    else:
        path = index.parent / blocked / 'case.json'; path.parent.mkdir(); path.write_bytes(case_path.read_bytes())
    row['path'] = str(path.relative_to(index.parent))
    write_json(index, {'cases': [row]})
    read_bytes = Path.read_bytes
    def guarded_read(self):
        assert self.resolve() != path.resolve(), 'Forbidden input was read'
        return read_bytes(self)
    monkeypatch.setattr(Path, 'read_bytes', guarded_read)
    with pytest.raises(ValueError): prepare_reference_workpack(case_index=index, output_dir=output)
    assert not output.exists()


def test_existing_output_is_never_overwritten(inputs):
    index, _, _, _, output = inputs
    output.mkdir(); marker = output / 'keep.json'; marker.write_text('original')
    with pytest.raises(ValueError, match='exists'): prepare_reference_workpack(case_index=index, output_dir=output)
    assert marker.read_text() == 'original' and list(output.iterdir()) == [marker]


def test_cli_requires_explicit_index_and_rejects_incomplete_reference_templates(inputs):
    index, _, _, _, output = inputs
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'prepare_reference_workpack.py'
    completed = subprocess.run([sys.executable, str(script), '--case-index', str(index), '--output-dir', str(output)],
                               capture_output=True, text=True)
    assert completed.returncode == 0 and json.loads(completed.stdout)['case_count'] == 1
    from scripts.validate_evaluation import FreezeAudit
    final = json.loads(next((output / 'final-references').glob('*.json')).read_text())
    audit = FreezeAudit(index)
    audit.people(final['reviewers'], 'template.reviewers', minimum=2)
    assert {'person_id_missing', 'person_timestamp_missing', 'exposure_declaration_missing'} <= {i['code'] for i in audit.issues}
