"""Score saved held-out runs (B0 / Fixed / Agent) against generator-injected truth.

Reads only saved raw outputs and the separate truth folder after all runs finished.
Every planned attempt stays in the denominator: missing, failed or partial runs are
counted as such, never dropped. Truth is generator-injected, not a human reference.
"""
import argparse
from collections import Counter, defaultdict
from decimal import Decimal
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'eval/agent-development-2026-10-03/run'
CLAIM_KINDS = ('count', 'amount_sum', 'counterparty', 'time_range')
ISSUE_TYPES = ('claim_error', 'material_mismatch', 'focus_not_addressed')
METHODS = ('b0', 'fixed', 'agent')


def aggregate(values):
    """One value per claim kind: any contradiction wins, then abstention, then support."""
    values = set(values)
    for value in ('contradicted', 'insufficient_evidence', 'supported'):
        if value in values:
            return value
    return 'not_extracted'


def system_view(raw):
    kinds = {c['claim_id']: c['kind'] for c in raw.get('claims', [])}
    by_kind = defaultdict(list)
    for row in raw.get('claim_results', []):
        by_kind[kinds.get(row.get('claim_id'))].append(row.get('result'))
    issue_types = {i['type'] for i in raw.get('issues', [])}
    return {'claims': {k: aggregate(by_kind.get(k, [])) for k in CLAIM_KINDS},
            'material': aggregate_material([m.get('result') for m in raw.get('material_results', [])]),
            'focuses': {s['focus_id']: s.get('status') for s in raw.get('semantic_results', [])},
            'features': {f.get('feature_code', f.get('feature')): f.get('result') for f in raw.get('features', [])},
            'issue_types': sorted(t for t in issue_types if t in ISSUE_TYPES),
            'return_for_revision': raw.get('qc_recommendation') == '建议退回修订',
            'run_status': raw.get('run_status')}


def aggregate_material(values):
    for value in ('mismatch', 'insufficient', 'pending_judgement', 'corresponds'):
        if value in values:
            return value
    return 'not_checked'


def b0_view(raw):
    out = raw.get('parsed_output') or {'checks': []}
    checks = [c for c in out['checks'] if c.get('status') == 'completed']
    by_kind = defaultdict(list)
    for c in checks:
        if c['label'] in CLAIM_KINDS:
            by_kind[c['label']].append(c['value'])
    focuses = {c['anchor'].get('focus_id'): c['value'] for c in checks if c['label'] == 'alert_response'}
    features = {c['label']: c['value'] for c in checks if c['label'] in ('F1', 'F2')}
    material = aggregate_material([c['value'] for c in checks if c['label'] == 'material_relation'])
    claims = {k: aggregate(by_kind.get(k, [])) for k in CLAIM_KINDS}
    issue_types = set()
    if 'contradicted' in claims.values():
        issue_types.add('claim_error')
    if material == 'mismatch':
        issue_types.add('material_mismatch')
    if 'not_addressed' in focuses.values():
        issue_types.add('focus_not_addressed')
    return {'claims': claims, 'material': material, 'focuses': focuses, 'features': features,
            'issue_types': sorted(issue_types), 'return_for_revision': bool(issue_types),
            'run_status': raw.get('run_status')}


def usage_cost(raw, rates):
    calls = raw.get('model_requests', [])
    tokens_in = tokens_out = 0
    cost = Decimal(0)
    complete = True
    for call in calls:
        usage = call.get('usage')
        settlement = call.get('budget_settlement') or {}
        if not isinstance(usage, dict) or settlement.get('status') != 'settled':
            complete = False
            continue
        tokens_in += usage['prompt_tokens']
        tokens_out += usage['completion_tokens']
        cost += Decimal(settlement['cost'])
    return {'model_calls': len(calls), 'input_tokens': tokens_in, 'output_tokens': tokens_out,
            'cost_cny': cost, 'usage_complete': complete}


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else (0.0 if precision == 0 or recall == 0 else None)
    return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': precision, 'recall': recall, 'f1': f1}


def score(bench, suffix):
    manifest = json.loads((bench / 'manifest.json').read_text())
    rows, per_method = [], {m: defaultdict(Counter) for m in METHODS}
    totals = {m: {'cost_cny': Decimal(0), 'input_tokens': 0, 'output_tokens': 0, 'model_calls': 0, 'duration_ms': 0.0,
                  'run_status': Counter(), 'usage_complete': True, 'attempts': 0} for m in METHODS}
    for case in manifest['cases']:
        truth = json.loads((bench / 'truth' / (case['case_id'] + '.json')).read_text())
        for method in METHODS:
            path = RUN / f"{case['case_id']}-{method}-{suffix}" / 'raw.json'
            t = totals[method]
            t['attempts'] += 1
            if not path.exists():
                view = None
                t['run_status']['not_run_or_interrupted'] += 1
            else:
                raw = json.loads(path.read_text())
                view = b0_view(raw) if method == 'b0' else system_view(raw)
                t['run_status'][view['run_status']] += 1
                u = usage_cost(raw, None)
                for key in ('input_tokens', 'output_tokens', 'model_calls'):
                    t[key] += u[key]
                t['cost_cny'] += u['cost_cny']
                t['usage_complete'] &= u['usage_complete']
                t['duration_ms'] += (raw.get('stats') or {}).get('duration_ms') or 0
            c = per_method[method]
            expected_types = set(truth['expected_issue_types'])
            predicted_types = set(view['issue_types']) if view else set()
            for kind in ISSUE_TYPES:
                e, p = kind in expected_types, kind in predicted_types
                c['issue:' + kind]['tp' if e and p else 'fp' if p else 'fn' if e else 'tn'] += 1
            e, p = truth['expected_return_for_revision'], bool(view and view['return_for_revision'])
            c['case_return']['tp' if e and p else 'fp' if p else 'fn' if e else 'tn'] += 1
            for kind, expected in truth['claims'].items():
                got = view['claims'][kind] if view else 'not_run'
                c['claim_value']['correct' if got == expected else 'wrong'] += 1
                c['claim_value:' + kind]['correct' if got == expected else 'wrong'] += 1
                if expected == 'insufficient_evidence':
                    c['abstention']['correct' if got == expected else 'false_contradiction' if got == 'contradicted'
                                    else 'false_support' if got == 'supported' else 'other'] += 1
                if got == 'not_extracted':
                    c['claim_not_extracted']['count'] += 1
            got = view['material'] if view else 'not_run'
            c['material']['correct' if got == truth['material_relation'] else 'wrong'] += 1
            for focus_id, expected in truth['focuses'].items():
                got = view['focuses'].get(focus_id) if view else 'not_run'
                c['focus']['correct' if got == expected else 'wrong'] += 1
            for name, expected in truth['features'].items():
                got = view['features'].get(name) if view else 'not_run'
                c['feature']['correct' if got == expected else 'wrong'] += 1
            rows.append({'case_id': case['case_id'], 'profile': truth['profile'], 'method': method,
                         'expected_issue_types': sorted(expected_types), 'predicted_issue_types': sorted(predicted_types),
                         'claims_expected': truth['claims'], 'claims_predicted': view['claims'] if view else None,
                         'material_expected': truth['material_relation'], 'material_predicted': view['material'] if view else None,
                         'focus_expected': truth['focuses'], 'focus_predicted': view['focuses'] if view else None,
                         'features_expected': truth['features'], 'features_predicted': view['features'] if view else None,
                         'run_status': view['run_status'] if view else 'not_run'})
    summary = {}
    for method in METHODS:
        c, t = per_method[method], totals[method]
        acc = lambda key: {'correct': c[key]['correct'], 'total': c[key]['correct'] + c[key]['wrong'],
                           'rate': c[key]['correct'] / (c[key]['correct'] + c[key]['wrong']) if c[key]['correct'] + c[key]['wrong'] else None}
        summary[method] = {
            'attempts': t['attempts'], 'run_status': dict(t['run_status']),
            'issue_detection': {k: prf(c['issue:' + k]['tp'], c['issue:' + k]['fp'], c['issue:' + k]['fn']) for k in ISSUE_TYPES},
            'case_return_for_revision': {**prf(c['case_return']['tp'], c['case_return']['fp'], c['case_return']['fn']),
                                         'tn': c['case_return']['tn']},
            'claim_value_accuracy': acc('claim_value'),
            'claim_value_accuracy_by_kind': {k: acc('claim_value:' + k) for k in CLAIM_KINDS if c['claim_value:' + k]},
            'claim_not_extracted': c['claim_not_extracted']['count'],
            'partial_coverage_abstention': dict(c['abstention']),
            'material_accuracy': acc('material'), 'focus_accuracy': acc('focus'), 'feature_accuracy': acc('feature'),
            'cost_cny_total': str(t['cost_cny']), 'cost_cny_per_case': str((t['cost_cny'] / t['attempts']).quantize(Decimal('0.000001'))),
            'model_calls_total': t['model_calls'], 'input_tokens_total': t['input_tokens'], 'output_tokens_total': t['output_tokens'],
            'mean_duration_s': round(t['duration_ms'] / 1000 / t['attempts'], 1), 'usage_complete': t['usage_complete']}
    return {'contract': 'heldout-injected-defect-score-1', 'benchmark_manifest': str((bench / 'manifest.json').resolve().relative_to(ROOT)),
            'truth_kind': manifest['truth_kind'], 'cases': len(manifest['cases']),
            'families': len({c['family'] for c in manifest['cases']}), 'attempt_suffix': suffix,
            'formal_human_reference': False, 'summary': summary, 'rows': rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bench', type=Path, default=ROOT / 'eval/heldout-v1')
    parser.add_argument('--suffix', default='hv1')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = score(args.bench, args.suffix)
    text = json.dumps(result, ensure_ascii=False, indent=1, default=str)
    if args.output:
        args.output.write_text(text + '\n')
    print(json.dumps(result['summary'], ensure_ascii=False, indent=1, default=str))
