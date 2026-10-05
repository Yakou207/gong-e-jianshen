"""Version-bound label candidates and append-only human adjudication helpers."""
from copy import deepcopy
from datetime import timedelta

from . import core
from .depgraph import digest
from .ingest import parse_time, transaction_cents
from .schema import LABEL_VALUES, load_schema

CLAIM_LABELS = {'count', 'amount_sum', 'counterparty', 'time_range'}
UNCERTAIN = {'undeterminable', 'insufficient_evidence', 'insufficient', 'pending_judgement'}
PROBLEM_VALUES = {'contradicted', 'mismatch', 'not_addressed'}
DECISION_ACTIONS = {'confirm', 'reconfirm', 'revise', 'abstain', 'dispute', 'reject', 'not_applicable'}


def run_schema(run, package):
    return (run.get('snapshot', {}).get('sources', {}).get('source:schema', {}).get('value')
            or package.get('schema') or load_schema())


def material_judgement_inputs_complete(package, material, link, schema):
    """A human may judge an untemplated relation, but cannot supply absent facts."""
    if not material or link.get('schema_version', schema['schema_version']) != schema['schema_version']:
        return False
    try:
        subject, counterparty, period = material['subject'], material['counterparty'], material['period']
        if (subject.get('account_id') != package['subject_account_id'] or not isinstance(subject.get('role'), str) or not subject['role'].strip()
                or not isinstance(counterparty.get('role'), str) or not counterparty['role'].strip() or material.get('currency') != 'CNY'):
            return False
        if core.resolve_entity(package, counterparty)['execution_status'] != 'completed':
            return False
        if parse_time(period['start']) >= parse_time(period['end']):
            return False
        transaction_cents(material)
        selected = [row for row in package.get('transactions', []) if row['transaction_id'] in link.get('transaction_ids', [])]
        fields = ['transaction_id', 'account_id', 'direction', 'amount', 'currency', 'timestamp', 'counterparty_token']
        for row in selected:
            start = parse_time(row['timestamp'])
            coverage = core.check_coverage(package, start.isoformat(), (start + timedelta(microseconds=1)).isoformat(), fields)
            if not coverage['reliable']:
                return False
    except (KeyError, ValueError, TypeError, AttributeError):
        return False
    return True


def build_annotations(run, package):
    """Every result is a label candidate; issues remain a separate business list."""
    if not run:
        return []
    schema = run_schema(run, package)
    schema_version = run.get('schema_version') or schema['schema_version']
    claims = {claim['claim_id']: claim for claim in run.get('claims', [])}
    links = {link['link_id']: link for link in package.get('material_links', [])}
    records = []
    def add(kind, label, target, result, value, **extra):
        definition = schema.get('labels', {}).get(label, {})
        obj = deepcopy(result.get('object') or {'account_id': package['subject_account_id'],
                       'start': package['coverage_start'], 'end': package['coverage_end']})
        identity = 'annotation-' + digest([run.get('case_id'), label, target, obj])[:24]
        records.append({'annotation_id': identity, 'kind': kind, 'label': label, 'target_id': target,
            'object': obj,
            'candidate_value': value, 'value': value, 'allowed_values': definition.get('allowed_values', LABEL_VALUES[label]),
            'required': result.get('required', True), 'execution_status': result.get('execution_status', 'completed'),
            'evidence': deepcopy(result.get('evidence', [])), 'coverage': deepcopy(result.get('coverage')),
            'source_hash': run.get('source_hash'), 'schema_version': schema_version,
            'snapshot_id': run.get('snapshot_id'), 'run_id': run.get('run_id'),
            'origin': 'machine_candidate', 'reason': result.get('reason'), **extra})
    for item in run.get('features', []):
        add('feature', item['feature_code'], 'feature:' + item['feature_code'], item, item['result'])
    for item in run.get('claim_results', []):
        claim = claims.get(item['claim_id'])
        if claim and claim.get('kind') in CLAIM_LABELS:
            query = next((ref['scope'] for ref in item.get('evidence', []) if ref.get('type') == 'query_scope'), {})
            obj = {key: query.get(key, claim.get(key, package.get('coverage_' + key) if key in {'start', 'end'} else None))
                   for key in ('account_id', 'start', 'end', 'direction', 'counterparty_ref', 'counterparty_token')}
            obj['account_id'] = obj['account_id'] or package['subject_account_id']
            add('claim', claim['kind'], 'claim:' + item['claim_id'], {**item, 'object': obj}, item['result'], claim=deepcopy(claim))
            records[-1]['origin'] = claim.get('origin', 'machine_candidate')
            records[-1]['amendment_id'] = claim.get('amendment_id')
    for item in run.get('material_results', []):
        link = links.get(item.get('link_id'), {})
        material = next((m for m in package.get('materials', []) if m['material_id'] == link.get('material_id')
                         and m['revision'] == link.get('revision', link.get('material_revision'))), None)
        transaction_ids = {r['transaction_id'] for r in package.get('transactions', [])}
        can_judge = bool(item['result'] == 'pending_judgement' and item.get('binding_status') != 'needs_review' and material_judgement_inputs_complete(package, material, link, schema)
                         and link.get('transaction_ids') and set(link['transaction_ids']) <= transaction_ids)
        obj = {'account_id': package['subject_account_id'], 'material_id': link.get('material_id'),
               'transaction_ids': sorted(link.get('transaction_ids', [])), 'relation_template': link.get('relation_template')}
        add('material', 'material_relation', 'materials:' + str(item.get('link_id')), {**item, 'object': obj}, item['result'],
            human_judgement_allowed=can_judge, material_link=deepcopy(link))
        if can_judge:
            records[-1]['evidence'].append({'type': 'material', 'material_id': material['material_id'],
                                           'revision': material['revision'], 'field_paths': [k for k in ('subject', 'counterparty', 'period', 'amount', 'amount_cents', 'currency') if k in material]})
    for item in run.get('semantic_results', []):
        add('semantic', 'alert_response', 'semantic:' + item['focus_id'], item, item['status'])
    targets = package.get('review_scope', {}).get('target_labels', list(schema['labels']))
    present = {record['label'] for record in records}
    doc = next((d for d in package.get('documents', []) if d['document_id'] == 'narrative'), None)
    references = ([{'type': 'document_span', 'document_id': doc['document_id'], 'revision': doc['revision'],
                    'span': [0, len(doc['text'])]}] if doc and doc['text'] else [])
    for label in targets:
        if label in present:
            continue
        evidence = deepcopy(references)
        if label == 'material_relation':
            evidence.append({'type': 'material_set', 'content_hash': digest(sorted(package.get('materials', []), key=lambda m: m['material_id'])),
                             'members': [{'material_id': m['material_id'], 'revision': m['revision']} for m in package.get('materials', [])]})
        add('scope_review', label, 'scope:' + label, {'evidence': evidence,
            'reason': '本次未产生该目标的候选；须复核是否漏抽或明确记录不适用'}, None, coverage_status='no_candidate')
    return records


def _view(event, annotation):
    if not event:
        return {'status': 'candidate', 'final_value': None, 'event_id': None, 'valid': False, 'adjudicated': False}
    action = event.get('action')
    status = {'confirm': 'confirmed', 'reconfirm': 'confirmed', 'revise': 'revised',
              'abstain': 'abstained', 'dispute': 'disputed', 'reject': 'rejected', 'not_applicable': 'not_applicable'}.get(action, 'candidate')
    execution = event.get('verification', {}).get('execution_status', annotation['execution_status'])
    adjudicated = action in {'confirm', 'reconfirm', 'revise', 'not_applicable'}
    value = event.get('final_value') if adjudicated else None
    return {'status': status, 'final_value': value, 'event_id': event.get('event_id'),
            'valid': adjudicated and execution == 'completed' and ((value in annotation['allowed_values'] and value not in UNCERTAIN)
                      or (action == 'not_applicable' and annotation['kind'] == 'scope_review')),
            'adjudicated': adjudicated, 'actor': event.get('actor'), 'reason': event.get('reason'),
            'at': event.get('created_at'), 'evidence': deepcopy(event.get('evidence', [])),
            'previous_value': event.get('previous_value'), 'origin': event.get('origin'),
            'verification': deepcopy(event.get('verification')), 'claim': deepcopy(event.get('revised_claim')),
            'object': deepcopy(event.get('reviewed_object', annotation['object'])),
            'proposed_value': event.get('proposed_value'), 'correction_status': event.get('correction_status'),
            'applicability': event.get('applicability'),
            'snapshot_id': event.get('snapshot_id'), 'source_hash': event.get('source_hash')}


def apply_reviews(annotations, events, *, stale=False, integrity=True):
    output = deepcopy(annotations)
    for item in output:
        decisions = [e for e in events if e.get('target_id') == item['annotation_id'] and e.get('annotation_id') == item['annotation_id']]
        current = [e for e in decisions if e.get('snapshot_id') == item['snapshot_id'] and e.get('source_hash') == item['source_hash']]
        previous = [e for e in decisions if e not in current]
        item['prior_review'] = _view(previous[-1], item) if previous else None
        item['review'] = _view(current[-1], item) if current else _view(None, item)
        if stale or not integrity:
            if current:
                item['prior_review'] = _view(current[-1], item)
            item['review'] = {'status': 'needs_review', 'final_value': None, 'event_id': None, 'valid': False, 'adjudicated': False}
        elif not current and previous:
            item['review']['status'] = 'needs_review'
        can_revise = item['kind'] in {'claim', 'semantic'} or item.get('human_judgement_allowed', False)
        item['editable'] = can_revise and not stale and integrity
        actions = ['abstain', 'dispute', 'reject']
        if item['execution_status'] == 'completed':
            actions.insert(0, 'reconfirm' if item['prior_review'] and not current else 'confirm')
        if can_revise:
            actions.insert(1, 'revise')
        if item['kind'] == 'scope_review':
            actions = ['not_applicable', 'abstain', 'dispute']
        item['allowed_actions'] = actions if not stale and integrity else []
        item['editable_claim'] = deepcopy(item['review'].get('claim') or item.get('claim'))
    return output


def annotation_pending(annotations):
    pending = []
    for item in annotations:
        if not item['required']:
            continue
        review = item['review']
        if not review.get('adjudicated'):
            reason = {'needs_review': '来源或运行改变后需重新裁决', 'abstained': '必需标签弃标，仍待补证或判断',
                      'disputed': '标签裁决存在争议', 'rejected': '候选被驳回，仍需有效修订或重查'}.get(review['status'], '必需标签尚未逐项裁决')
        elif not review.get('valid'):
            reason = ('人工修订改变了待核验命题；工具结果只针对新命题，须补正来源重查，不能自动解除原问题'
                      if review.get('correction_status') else '已记录人工判断，但执行或资料仍不足，不能作为完成的核验')
        elif review['final_value'] in PROBLEM_VALUES:
            reason = '已确认存在质检问题，需补正、补证或有依据的修订后才能通过'
        else:
            continue
        pending.append({'annotation_id': item['annotation_id'], 'label': item['label'], 'target_id': item['target_id'], 'reason': reason})
    return pending


def resolved_targets(annotations):
    """Human judgements may resolve their own issue; execution failures stay open."""
    resolved = {}
    for item in annotations:
        review = item['review']
        if review.get('valid') and review['final_value'] in {'supported', 'addressed', 'corresponds'}:
            resolved[item['target_id']] = item
    return resolved


def _current_structured_reference(package, ref):
    kind = ref['type']
    materials = sorted(package.get('materials', []), key=lambda m: m['material_id'])
    transaction_hash = digest(sorted(package.get('transactions', []), key=lambda r: r['transaction_id']))
    if kind == 'query_scope':
        if ref.get('source') == 'materials':
            return ref == {'type': kind, 'source': 'materials',
                           'material_ids': [m['material_id'] for m in package.get('materials', [])],
                           'material_links': package.get('material_links', [])}
        scope = ref['scope']
        query = {k: scope[k] for k in ('start', 'end', 'account_id', 'direction', 'counterparty_ref')}
        if 'transaction_id' in scope:
            query['transaction_id'] = scope['transaction_id']
        query['fields'] = ref['coverage']['fields']
        if sorted(query['fields']) != scope['fields']:
            return False
        current = core.query_transactions(package, query)
        return ref == {'type': kind, **{k: current[k] for k in ('scope', 'coverage', 'transaction_ids')}}
    if kind == 'material_set':
        return (ref.get('content_hash') == digest(materials)
                and sorted(ref['members'], key=lambda m: m['material_id'])
                == [{'material_id': m['material_id'], 'revision': m['revision']} for m in materials])
    if kind == 'material_link':
        return any(ref == {'type': kind, 'link_id': link.get('link_id'), 'content_hash': digest(link),
                           'relation_template': link.get('relation_template')}
                   for link in package.get('material_links', []))
    if kind == 'transaction_set':
        return any(ref == {'type': kind, 'content_hash': transaction_hash,
                           'transaction_set_version': 'sha256:' + transaction_hash,
                           'requested_transaction_ids': link.get('transaction_ids', [])}
                   for link in package.get('material_links', []))
    if kind == 'alert_focus':
        alert = package.get('alert') or {}
        focuses = alert.get('focuses') or ([{'focus_id': alert.get('alert_id', 'original-focus')}]
                                          if alert.get('original_focus') else [])
        return any(ref == {'type': kind, 'focus_id': focus['focus_id'], 'revision': alert.get('revision')}
                   for focus in focuses)
    return False


def validate_evidence(package, evidence, known_evidence):
    if not isinstance(evidence, list) or any(not isinstance(ref, dict) for ref in evidence):
        raise ValueError('证据必须为引用对象数组')
    known = {digest(ref) for ref in known_evidence}
    transactions = {r['transaction_id']: r for r in package.get('transactions', [])}
    transaction_version = 'sha256:' + digest(sorted(package.get('transactions', []), key=lambda r: r['transaction_id']))
    for ref in evidence:
        is_known = digest(ref) in known
        kind = ref.get('type')
        if kind == 'document_span' and not set(ref) - {'type', 'document_id', 'revision', 'span', 'text'} and core.validate_span(package, ref, ref.get('text')):
            continue
        if kind in {'transaction', 'transactions'}:
            ids = [ref.get('transaction_id')] if kind == 'transaction' else ref.get('transaction_ids')
            if (is_known or not set(ref) - {'type', 'transaction_id', 'transaction_ids'}) and isinstance(ids, list) and ids and all(isinstance(tid, str) and tid in transactions for tid in ids):
                fields = ref.get('transaction_fields', {})
                if (ref.get('transaction_set_version', transaction_version) == transaction_version
                        and isinstance(fields, dict) and all(tid in ids and isinstance(paths, list) and paths
                            and all(isinstance(path, str) and path in transactions[tid] for path in paths)
                            for tid, paths in fields.items())):
                    continue
        if kind == 'material':
            material = next((m for m in package.get('materials', []) if m['material_id'] == ref.get('material_id') and m['revision'] == ref.get('revision')), None)
            paths = ref.get('field_paths')
            if (is_known or not set(ref) - {'type', 'material_id', 'revision', 'field_paths'}) and material and ref.get('content_hash', digest(material)) == digest(material) and isinstance(paths, list) and paths and all(isinstance(path, str) for path in paths):
                try:
                    for path in paths:
                        value = material
                        for part in path.split('.'):
                            value = value[part]
                    continue
                except (KeyError, TypeError):
                    pass
        if is_known and kind == 'upgraded_focus':
            focus = next((item for item in package.get('review_scope', {}).get('upgraded_leads', [])
                          if item.get('focus_id') == ref.get('focus_id')), None)
            if focus and ref.get('content_hash') == digest(focus):
                continue
        if is_known and kind in {'query_scope', 'material_set', 'material_link', 'transaction_set', 'alert_focus'}:
            # Exact server provenance is necessary, but its source must still resolve.
            try:
                if _current_structured_reference(package, ref):
                    continue
            except (KeyError, ValueError, TypeError, AttributeError):
                pass
        raise ValueError('人工证据不能解析到当前快照，或不是允许的引用类型')
    return deepcopy(list({digest(ref): ref for ref in evidence}.values()))


def prepare_decision(annotation, package, *, action, new_value=None, claim_patch=None, evidence=None, known_evidence=(), previous_event_id=None):
    action = 'revise' if action == 'amend' else action
    if action not in annotation['allowed_actions']:
        raise ValueError('当前标签不允许该动作；确定性结论须修正输入后重新核验')
    if action == 'reconfirm':
        prior = annotation.get('prior_review')
        if not prior or previous_event_id != prior.get('event_id'):
            raise ValueError('重新确认必须明确引用待复核的旧人工事件')
    if action != 'revise' and (new_value is not None or claim_patch is not None):
        raise ValueError('只有修订动作可以提交新值或陈述修订')
    proof = validate_evidence(package, annotation['evidence'] if evidence is None else evidence, known_evidence)
    if action in {'confirm', 'reconfirm', 'revise', 'not_applicable'} and not proof:
        raise ValueError('确认或修订标签必须保留当前有效证据')
    if action in {'confirm', 'reconfirm', 'revise'} and annotation['kind'] == 'semantic':
        doc = next((d for d in package.get('documents', []) if d['document_id'] == 'narrative'), None)
        if not doc or not doc['text'].strip() or not any(ref.get('type') == 'document_span' and ref.get('document_id') == 'narrative' for ref in proof):
            raise ValueError('回应标签必须引用非空的当前理由，其他证据不能替代理由内容')
    if action == 'revise' and annotation['kind'] == 'material':
        link = annotation['material_link']
        material_proof = any(ref.get('type') == 'material' and ref.get('material_id') == link['material_id'] for ref in proof)
        selected = set(link.get('transaction_ids', []))
        transaction_proof = any(selected <= set(ref.get('transaction_ids', ref.get('requested_transaction_ids', [])))
                                for ref in proof if ref.get('type') in {'transactions', 'transaction_set', 'query_scope'})
        if not material_proof or not transaction_proof:
            raise ValueError('材料人工关系判断须引用对应材料字段和所选交易集合')
    prior_value = annotation['review'].get('final_value')
    if prior_value is None and annotation.get('prior_review'):
        prior_value = annotation['prior_review'].get('final_value')
    decision = {'action': action, 'annotation_id': annotation['annotation_id'], 'label': annotation['label'],
                'candidate_value': annotation['candidate_value'], 'previous_value': prior_value,
                'final_value': None, 'evidence': proof, 'origin': 'human_confirmation', 'previous_event_id': previous_event_id}
    if action in {'confirm', 'reconfirm'}:
        decision['final_value'] = annotation['candidate_value']
    elif action == 'not_applicable':
        decision.update(applicability='not_applicable', origin='human_scope_review')
    elif action == 'revise' and annotation['kind'] == 'claim':
        if new_value is not None or not isinstance(claim_patch, dict) or not claim_patch:
            raise ValueError('事实修订须提交非空claim_patch，结论由确定性工具重核，不能直接指定new_value')
        allowed = {'quote', 'source', 'text', 'counterparty_ref', 'counterparty_token', 'direction', 'operator', 'value', 'value_cents',
                   'unit', 'currency', 'account_id', 'start', 'end'}
        if set(claim_patch) - allowed or {'value', 'value_cents'} <= set(claim_patch):
            raise ValueError('陈述修订字段不被允许，或同时提交了元和分金额')
        revised = deepcopy(annotation.get('editable_claim') or annotation['claim'])
        for key, value in claim_patch.items():
            if key == 'quote':
                continue
            if value is None and key in {'counterparty_ref', 'counterparty_token', 'direction', 'start', 'end'}:
                revised.pop(key, None)
            else:
                revised[key] = deepcopy(value)
        if 'value' in claim_patch:
            revised.pop('value_cents', None)
        if 'value_cents' in claim_patch:
            revised.pop('value', None)
        if 'quote' in claim_patch:
            quote = claim_patch['quote']
            doc = next((d for d in package.get('documents', []) if d['document_id'] == 'narrative'), None)
            if not isinstance(quote, str) or not quote or not doc or doc['text'].count(quote) != 1:
                raise ValueError('修订原文必须能在当前理由中唯一定位')
            start = doc['text'].index(quote)
            revised.update(text=quote, source={'document_id': 'narrative', 'revision': doc['revision'], 'span': [start, start + len(quote)]})
        if revised.get('source', {}).get('document_id') != 'narrative' or not core.validate_span(package, revised.get('source'), revised.get('text')):
            raise ValueError('陈述修订的来源和原文跨度无效')
        verified = core.verify_claim(package, revised, package.get('schema'))
        query = next((ref['scope'] for ref in verified.get('evidence', []) if ref.get('type') == 'query_scope'), None)
        decision.update(final_value=verified['result'], verification=verified, revised_claim=revised,
                        claim_patch=deepcopy(claim_patch), origin='human_extraction_reverified',
                        reviewed_object={k:query.get(k) for k in ('account_id', 'start', 'end', 'direction', 'counterparty_ref', 'counterparty_token')} if query else None)
        proposition_fields = {'counterparty_ref', 'counterparty_token', 'direction', 'operator', 'value', 'value_cents',
                              'unit', 'currency', 'account_id', 'start', 'end'}
        original = annotation['claim']
        if revised.get('text') != original.get('text') or any(revised.get(key) != original.get(key) for key in proposition_fields):
            decision.update(final_value=None, proposed_value=verified['result'],
                            origin='human_extraction_correction_pending', correction_status='pending_source_review')
        decision['evidence'] = list({digest(ref): ref for ref in proof + verified.get('evidence', [])}.values())
    elif action == 'revise':
        if claim_patch is not None or new_value not in annotation['allowed_values'] or new_value in UNCERTAIN - {'insufficient'}:
            raise ValueError('请选择当前标签允许的明确修订值')
        decision.update(final_value=new_value, origin='human_judgement')
    return decision
