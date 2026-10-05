"""Copy explicit visible cases and blank reference forms; never create answers."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile

from aml_qc.baseline import RAW_FIELDS, raw_case_input
from aml_qc.ingest import validate_case
from scripts.validate_evaluation import EXPOSURES


def require(condition, message):
    if not condition:
        raise ValueError(message)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def safe_input(path):
    path = Path(path).absolute()
    blocked = {'private', 'hidden', 'reference', 'references', 'model-products', 'model_products',
               'model-outputs', 'model_outputs', 'run', 'runs', 'source-archives'}
    for candidate in (path, path.resolve()):
        parts = candidate.parts
        # macOS resolves temporary storage under /private/var; this mount is
        # unrelated to a project's hidden generator/private directory.
        if parts[1:3] in (('private', 'var'), ('private', 'tmp')):
            parts = parts[2:]
        require(not any(part.lower() in blocked or part.lower().startswith('.env') for part in parts),
                'private, reference, model or credential input path forbidden')
    return path


def read_record(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def reject_constant(_value):
        raise ValueError('non-finite JSON number')
    raw = safe_input(path).read_bytes()
    return json.loads(raw, object_pairs_hook=unique, parse_constant=reject_constant), sha256(raw)


def closed_visible(value, fields, path='case'):
    if isinstance(fields, dict):
        require(isinstance(value, dict), 'visible object required: ' + path)
        require(not set(value) - set(fields), 'non-visible fields: ' + path)
        for key, item in value.items():
            closed_visible(item, fields[key], path + '.' + key)
    elif isinstance(fields, list):
        require(isinstance(value, list), 'visible array required: ' + path)
        for item in value:
            closed_visible(item, fields[0], path + '[]')
    else:
        require(not isinstance(value, (dict, list)), 'non-scalar visible field: ' + path)


def person_template():
    return {'person_id': None, 'signed_at': None, 'exposure': None, 'authorship_bias_disclosure': None}


def unit_template(check_id=None):
    return {'reference_check_id': check_id, 'label': None, 'object_scope': None,
            'applicability': None, 'adjudication_status': None, 'reference_value': None,
            'reason': None, 'evidence_sets': []}


def reference_template(case_id=None, case_sha256=None, visible_case_sha256=None, check_ids=(), reviewer_count=2):
    return {'case_id': case_id, 'case_sha256': case_sha256, 'visible_case_sha256': visible_case_sha256,
            'reviewers': [person_template() for _ in range(reviewer_count)],
            'check_units': [unit_template(check_id) for check_id in check_ids]}


def prepare_reference_workpack(*, case_index, output_dir):
    output = Path(output_dir).absolute()
    require(not output.exists() and not output.is_symlink(), 'output already exists; never overwrite')
    index, index_hash = read_record(case_index)
    require(isinstance(index, dict) and not set(index) - {'cases', 'contract_version'}, 'invalid visible case index fields')
    require(index.get('contract_version', 'case-index-1') == 'case-index-1', 'unsupported case index contract')
    entries = index.get('cases')
    require(isinstance(entries, list) and bool(entries), 'explicit nonempty cases list required')
    prepared, seen = [], set()
    for entry in entries:
        require(isinstance(entry, dict) and not set(entry) - {'case_id', 'path', 'file', 'sha256', 'required_check_ids'},
                'invalid visible case index entry fields')
        case_id, expected_hash = entry.get('case_id'), entry.get('sha256')
        require(isinstance(case_id, str) and bool(case_id.strip()) and case_id not in seen, 'missing or duplicate case ID')
        seen.add(case_id)
        require(isinstance(expected_hash, str) and re.fullmatch('[0-9a-f]{64}', expected_hash), 'explicit SHA-256 required')
        require(('path' in entry) != ('file' in entry), 'exactly one explicit path or file required')
        name = entry.get('path', entry.get('file'))
        require(isinstance(name, str) and bool(name.strip()) and not Path(name).is_absolute(), 'relative explicit case path required')
        source = Path(case_index).absolute().parent / name
        case, observed_hash = read_record(source)
        require(observed_hash == expected_hash, 'case SHA-256 mismatch')
        require(isinstance(case, dict) and case.get('case_id') == case_id, 'case ID binding mismatch')
        closed_visible(case, {**RAW_FIELDS, 'case_family': None, 'title': None})
        validation = validate_case(case)
        require(validation['valid'], 'invalid case: ' + '; '.join(validation['errors']))
        visible_raw = json_bytes(raw_case_input(case))
        check_ids = entry.get('required_check_ids', [])
        require(isinstance(check_ids, list) and all(isinstance(uid, str) and bool(uid.strip()) for uid in check_ids)
                and len(check_ids) == len(set(check_ids)), 'invalid explicit check IDs')
        prepared.append({'case_id': case_id, 'case_sha256': observed_hash,
                         'visible_case_sha256': sha256(visible_raw), 'visible_raw': visible_raw,
                         'filename': 'case-' + sha256(case_id.encode())[:24] + '.json',
                         'check_ids': check_ids})

    output.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix='.reference-workpack-', dir=output.parent))
    reserved = False
    try:
        files = {}
        def write(name, value):
            raw = value if isinstance(value, bytes) else json_bytes(value)
            target = staged / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            files[name] = sha256(raw)
        write('reference-record.template.json', reference_template())
        write('check-unit.template.json', unit_template())
        write('exposure-declaration.template.json', dict.fromkeys(EXPOSURES))
        write('reference-protocol.template.json', {'reference_protocol': None})
        for case in prepared:
            cid, case_hash, filename = case['case_id'], case['case_sha256'], case['filename']
            visible_hash = case['visible_case_sha256']
            for reviewer in ('reviewer-1', 'reviewer-2'):
                write(reviewer + '/cases/' + filename, case['visible_raw'])
                write(reviewer + '/references/' + filename,
                      reference_template(cid, case_hash, visible_hash, case['check_ids'], reviewer_count=1))
            write('final-references/' + filename, reference_template(cid, case_hash, visible_hash, case['check_ids']))
            write('adjudication/' + filename, {'case_id': cid, 'case_sha256': case_hash, 'visible_case_sha256': visible_hash,
                  'reference_1_sha256': None, 'reference_2_sha256': None, 'status': None,
                  'adjudicator': person_template(), 'decisions': [], 'reason': None})
        manifest = {'contract_version': 'reference-workpack-preparation-1', 'status': 'preparation_only_not_references',
                    'reference_contract': 'evaluation-freeze-1', 'reference_protocol': None,
                    'case_index_sha256': index_hash, 'case_count': len(prepared),
                    'projection': 'baseline.raw_case_input closed visible-field whitelist',
                    'case_binding': 'Reference case_sha256 binds original indexed package bytes; visible_case_sha256 binds projected reviewer copy.',
                    'cases': [{k: deepcopy(v) for k, v in case.items() if k != 'visible_raw'} for case in prepared],
                    'files_sha256': files,
                    'steps': [
                        'Distribute only each reviewer directory and blank templates; directories alone do not establish access isolation.',
                        'Assign actual people and protocol; disclose authorship bias where required. Null exposure is no declaration, not false.',
                        'Reviewers enter independent initial units, evidence and declarations, then seal their separate files before discussion.',
                        'Record disagreements and actual adjudication; bind final references to original case hashes. Undecided answers stay unresolved.',
                        'Supply reviewed family grouping, complete reference coverage and frozen experiment settings separately; these forms cannot establish acceptance.',
                    ]}
        write('preparation-manifest.json', manifest)
        # Reserve the destination exclusively, then atomically replace only our
        # empty reservation with the complete staged directory.
        output.mkdir(exist_ok=False)
        reserved = True
        staged.rename(output)
        reserved = False
    finally:
        if staged.exists():
            shutil.rmtree(staged)
        if reserved:
            output.rmdir()
    return {'output_dir': str(output), 'case_count': len(prepared), 'status': manifest['status']}


def main():
    parser = argparse.ArgumentParser(description='Prepare visible cases and empty independent reference forms; no answers or declarations.')
    parser.add_argument('--case-index', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    try:
        print(json.dumps(prepare_reference_workpack(**vars(parser.parse_args())), ensure_ascii=False))
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(2, f'Reference preparation rejected: {error}\n')


if __name__ == '__main__':
    main()
