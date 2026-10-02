"""Frozen cross-case rule migration previews; never execute review or model calls."""
from copy import deepcopy
from uuid import uuid4

import jsonpatch
from jsonpointer import resolve_pointer

from .depgraph import affected_nodes, canonical, digest, sources_for
from .schema import LABEL_VALUES, load_schema, require_schema


def validate_target(schema):
    """Only complete settings understood by this implementation may be migrated."""
    require_schema(schema)
    baseline = load_schema()
    if set(schema) != set(baseline):
        raise ValueError('目标规范须完整，且不能含未实现的顶层配置')
    if schema['calculator_version'] != baseline['calculator_version'] or digest(schema['claims']) != digest(baseline['claims']):
        raise ValueError('目标规范改变了当前计算器未支持的版本或事实核验语义')
    if schema['schema_version'] != schema['schema_version'].strip() or len(schema['schema_version']) > 80:
        raise ValueError('规范版本须为不带首尾空白的短标识')
    for field in ('purpose', 'material_boundary'):
        if not isinstance(schema[field], str) or not schema[field].strip():
            raise ValueError(f'目标规范缺少{field}')
    if set(schema['features']) != {'F1', 'F2'}:
        raise ValueError('当前实现仅支持完整F1/F2特征规范')
    for code, rule in schema['features'].items():
        if set(rule) != set(baseline['features'][code]):
            raise ValueError(f'{code}须完整且不能含未实现参数')
    if set(schema['labels']) != set(LABEL_VALUES):
        raise ValueError('目标规范须保留完整的八类标签定义')
    for label, definition in schema['labels'].items():
        if set(definition) != set(baseline['labels'][label]):
            raise ValueError(f'标签{label}的定义字段不完整或未实现')
        for key, value in definition.items():
            if key in {'required_data', 'allowed_values'}:
                if not isinstance(value, list) or not value or any(not isinstance(x, str) or not x.strip() for x in value):
                    raise ValueError(f'标签{label}.{key}须为非空文本数组')
            elif not isinstance(value, str) or not value.strip():
                raise ValueError(f'标签{label}.{key}须为非空文本')
    required = {'subject_role', 'counterparty_role', 'direction', 'amount_relation', 'period_relation', 'amount_tolerance_cents'}
    for name, template in schema['material_templates'].items():
        if not isinstance(name, str) or not name or not required <= set(template) or set(template) - required - {'transaction_count'}:
            raise ValueError('材料模板须完整，且不能含未实现字段')
    return deepcopy(schema)


def schema_diff(before, after):
    # Canonical JSON prevents dictionary insertion order from changing the patch.
    import json
    before, after = json.loads(canonical(before)), json.loads(canonical(after))
    operations = jsonpatch.make_patch(before, after).patch
    if digest(jsonpatch.JsonPatch(operations).apply(before)) != digest(after):
        raise ValueError('规范差异无法重建目标规范')
    rows = []
    cursor = deepcopy(before)
    absent = object()
    for operation in operations:
        path = operation['path']
        old = resolve_pointer(cursor, path, default=absent)
        following = jsonpatch.JsonPatch([operation]).apply(cursor)
        new = resolve_pointer(following, path, default=absent)
        rows.append({'op': operation['op'], 'path': path,
                     'before': None if old is absent else old, 'after': None if new is absent else new,
                     'before_exists': old is not absent, 'after_exists': new is not absent,
                     **({'from': operation['from']} if 'from' in operation else {})})
        cursor = following
    return operations, rows


def changed_rules(changes):
    codes = set()
    for change in changes:
        parts = change['path'].split('/')
        if len(parts) > 2 and parts[1] == 'features':
            codes.add(parts[2])
        elif len(parts) > 2 and parts[1] == 'labels':
            codes.add(parts[2])
        elif len(parts) > 1 and parts[1] == 'material_templates':
            codes.add('material_relation')
        elif len(parts) > 1 and parts[1] == 'claims':
            codes.update({'count', 'amount_sum', 'counterparty', 'time_range'})
        elif parts[1:] != ['schema_version']:
            codes.update(LABEL_VALUES)
    return sorted(codes)


def state_anchor(state):
    run = state['latest_run'] or {}
    events = state['review_events']
    return {'source_hash': state['source_hash'], 'run_id': run.get('run_id'), 'snapshot_id': run.get('snapshot_id'),
            'event_hash': events[-1]['event_hash'] if events else None,
            'schema_hash': digest(state['package']['schema']), 'schema_version': state['package']['schema_version']}


def review_records(state):
    run = state['latest_run'] or {}
    return [{k: event.get(k) for k in ('event_id', 'target_id', 'annotation_id', 'action', 'actor', 'snapshot_id', 'source_hash')}
            for event in state['review_events'] if run and event.get('snapshot_id') == run.get('snapshot_id')
            and event.get('action') not in {'source_changed', 'needs_review', 'schema_migrated'}]


def migrated_package(package, target):
    result = deepcopy(package)
    old_version = package['schema_version']
    target_version = target['schema_version']
    result['schema'], result['schema_version'] = deepcopy(target), target_version
    version = str(package['data_version'])
    result['data_version'] = str(int(version) + 1) if version.isdigit() else 'migration-' + digest([package, target])[:16]
    link_changes, warnings = [], []
    for link in result.get('material_links', []):
        old_binding = link.get('schema_version')
        if old_binding == old_version:
            link['schema_version'] = target_version
            link_changes.append({'link_id': link['link_id'], 'field': 'schema_version', 'before': old_binding,
                                 'after': target_version, 'action': 'migrate_matching_binding'})
        elif old_binding is None:
            # A formerly implicit default must not silently switch meaning.
            link['schema_version'] = old_version
            link_changes.append({'link_id': link['link_id'], 'field': 'schema_version', 'before': None,
                                 'after': old_version, 'action': 'pin_previous_implicit_binding'})
            warnings.append({'link_id': link['link_id'], 'code': 'implicit_binding_pinned',
                             'message': '原关系未显式绑定规范，保留其旧口径并明确标记；重查后须修订该关系再核验'})
        else:
            warnings.append({'link_id': link['link_id'], 'code': 'existing_version_conflict',
                             'schema_version': old_binding, 'message': '已有规范冲突保留，不随本次迁移自动修复'})
    return result, link_changes, warnings


def migration_case(state, target, selected, implementation_hash):
    package = state['package']
    proposed, link_changes, warnings = migrated_package(package, target)
    _, changes = schema_diff(package['schema'], target)
    run = state['latest_run'] or {}
    snapshot = run.get('snapshot') or {}
    full = not snapshot.get('sources') or not snapshot.get('nodes') or state['stale']
    execution = deepcopy(run.get('execution') or {})
    execution['implementation_hash'] = implementation_hash
    current_sources = sources_for(proposed, target, execution)
    changed, affected = affected_nodes(snapshot, current_sources)
    if full:
        affected = sorted(set(snapshot.get('nodes', {})) | {'features', 'extraction', 'claims', 'materials', 'semantic_or_agent_stage', 'candidate_assembly'})
    same = digest(package['schema']) == digest(target)
    return {'case_id': package['case_id'], 'title': package.get('title', package['case_id']),
            'current_schema_version': package['schema_version'], 'current_schema_hash': digest(package['schema']),
            'current_source_hash': state['source_hash'], 'current_run_id': run.get('run_id'),
            'current_snapshot_id': run.get('snapshot_id'), 'current_event_hash': state_anchor(state)['event_hash'],
            'case_anchor': state_anchor(state), 'selected': selected, 'selectable': not same and package['schema_version'] != target['schema_version'],
            'changes': changes, 'affected_rule_codes': changed_rules(changes), 'affected_nodes': affected,
            'changed_sources': changed, 'requires_full_recheck': full,
            'impact_mode': 'full_case' if full else 'conservative_dependency_graph',
            'review_records': review_records(state), 'review_policy': 'all_current_snapshot_human_records',
            'material_link_changes': link_changes, 'material_link_warnings': warnings,
            'projected_source_hash': digest(proposed), 'status': 'pending' if selected else 'not_selected'}


def catalogue(store, db):
    import json
    schemas = {}
    for schema in [load_schema()] + [json.loads(row[0])['schema'] for row in db.execute('SELECT package FROM sources')]:
        schemas.setdefault(digest(schema), {'schema_hash': digest(schema), 'schema_version': schema['schema_version'],
                                          'schema': schema, 'case_ids': []})
    for row in db.execute('SELECT source_hash,case_id FROM cases'):
        source = json.loads(db.execute('SELECT package FROM sources WHERE source_hash=?', (row['source_hash'],)).fetchone()[0])
        schemas[digest(source['schema'])]['case_ids'].append(row['case_id'])
    for row in db.execute('SELECT preview_id FROM migration_previews'):
        preview, _ = read_preview(db,row[0])
        schema = preview['target_schema']
        schemas.setdefault(digest(schema), {'schema_hash': digest(schema), 'schema_version': schema['schema_version'],
                                          'schema': schema, 'case_ids': []})
    return {'schemas': sorted(schemas.values(), key=lambda s:(s['schema_version'],s['schema_hash'])), 'default_schema':load_schema()}


def check_version_binding(schemas, target):
    target_hash = digest(target)
    if any(row['schema_version'] == target['schema_version'] and row['schema_hash'] != target_hash for row in schemas):
        raise ValueError('目标规范版本已绑定不同内容；请使用新的规范版本，不可静默改口径')


def create_preview(store, *, base_schema_hash, target_schema, selected_case_ids, reason, actor, implementation_hash):
    if not reason.strip() or not actor.strip():
        raise ValueError('规范迁移预览必须记录原因和人员')
    target = validate_target(target_schema)
    if not isinstance(selected_case_ids, list) or any(not isinstance(x,str) or not x for x in selected_case_ids) or len(set(selected_case_ids)) != len(selected_case_ids):
        raise ValueError('迁移范围须为明确且不重复的案件ID数组')
    with store.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        schemas = catalogue(store, db)['schemas']
        base = next((s['schema'] for s in schemas if s['schema_hash']==base_schema_hash), None)
        if base is None:
            raise ValueError('预览基准规范不存在或内容指纹无效')
        if target['schema_version'] == base['schema_version']:
            raise ValueError('规范内容迁移必须使用新版本标识')
        check_version_binding(schemas,target)
        operations, changes = schema_diff(base,target)
        rules = changed_rules(changes)
        states = [store.get(row[0]) for row in db.execute('SELECT case_id FROM cases ORDER BY case_id')]
        related = [s for s in states if s['package']['schema_version']==base['schema_version']
                   or set(s['package'].get('review_scope',{}).get('target_labels',LABEL_VALUES)) & set(rules)]
        if set(selected_case_ids) - {s['package']['case_id'] for s in related}:
            raise ValueError('选择的案件不在本次相关任务清单中')
        cases = [migration_case(s,target,s['package']['case_id'] in selected_case_ids,implementation_hash) for s in related]
        for state, row in zip(related,cases):
            if row['selected'] and (not row['selectable'] or not state['audit_integrity']['valid']):
                raise ValueError('所选案件已采用目标规范、版本冲突或审计链无效')
            if row['selected']:
                store.validate(migrated_package(state['package'],target)[0])
        from .store import now
        preview = {'preview_id':uuid4().hex,'created_at':now(),'actor':actor,'reason':reason,
                   'base_schema_hash':base_schema_hash,'base_schema_version':base['schema_version'],
                   'base_schema':base,'target_schema':target,'target_schema_hash':digest(target),
                   'selected_case_ids':list(selected_case_ids),'changes':changes,'patch':operations,
                   'affected_rule_codes':rules,'cases':cases,'implementation_hash':implementation_hash,
                   'scope_policy':'仅逐案执行所选范围；未选或未执行案件保留原规范；不自动重查或调用模型'}
        preview_hash=digest(preview)
        db.execute('INSERT INTO migration_previews VALUES (?,?,?,?)',(preview['preview_id'],canonical(preview),preview_hash,preview['created_at']))
    return get_preview(store,preview['preview_id'],implementation_hash)


def read_preview(db, preview_id):
    import json
    row=db.execute('SELECT payload,payload_hash FROM migration_previews WHERE preview_id=?',(preview_id,)).fetchone()
    if not row:
        raise KeyError('规范迁移预览不存在')
    preview=json.loads(row['payload'])
    if digest(preview)!=row['payload_hash']:
        raise ValueError('迁移预览审计内容校验失败')
    return preview,row['payload_hash']


def read_receipts(db, preview_id=None, case_id=None):
    import json
    receipts=[]
    for row in db.execute('SELECT payload,receipt_hash FROM migration_receipts ORDER BY created_at'):
        receipt=json.loads(row['payload'])
        if (preview_id is not None and receipt['preview_id']!=preview_id) or (case_id is not None and receipt['case_id']!=case_id):
            continue
        if digest(receipt)!=row['receipt_hash']:
            raise ValueError('规范迁移回执审计内容校验失败')
        receipts.append({**receipt,'receipt_hash':row['receipt_hash']})
    return receipts


def get_preview(store, preview_id, implementation_hash):
    with store.connect() as db:
        preview,preview_hash=read_preview(db,preview_id)
        receipts=read_receipts(db,preview_id=preview_id)
    applied={r['case_id'] for r in receipts}
    view=deepcopy(preview)
    for row in view['cases']:
        current=store.get(row['case_id'])
        row['anchor_valid']=(state_anchor(current)==row['case_anchor'] and implementation_hash==preview['implementation_hash'])
        if row['case_id'] in applied:
            row['status']='applied'
        elif not row['anchor_valid']:
            row['status']='stale'
        row['can_apply']=row['selected'] and row['selectable'] and row['anchor_valid'] and row['case_id'] not in applied
    return {**view,'preview_hash':preview_hash,'receipts':receipts}


def apply_case(store, preview_id, case_id, *, preview_hash, reason, actor, implementation_hash):
    if not reason.strip() or not actor.strip():
        raise ValueError('逐案迁移必须记录原因和人员')
    from .store import now
    with store.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        preview,stored_hash=read_preview(db,preview_id)
        if stored_hash!=preview_hash:
            raise ValueError('预览内容指纹不匹配，请重新预览')
        if db.execute('SELECT 1 FROM migration_receipts WHERE preview_id=? AND case_id=?',(preview_id,case_id)).fetchone():
            raise ValueError('该预览中的案件已迁移，不可重复执行；请读取原回执')
        row=next((r for r in preview['cases'] if r['case_id']==case_id and r['selected'] and r['selectable']),None)
        if row is None:
            raise ValueError('该案件未包含在已确认的迁移范围')
        state=store.get(case_id)
        if implementation_hash!=preview['implementation_hash'] or state_anchor(state)!=row['case_anchor']:
            raise ValueError('预览已过期：来源、运行、人工记录或实现版本已改变，请重新预览')
        if not state['audit_integrity']['valid']:
            raise ValueError('案件审计链无效，不能迁移规范')
        target=validate_target(preview['target_schema'])
        if digest(target)!=preview['target_schema_hash']:
            raise ValueError('预览目标规范指纹无效')
        check_version_binding(catalogue(store,db)['schemas'],target)
        package,links,warnings=migrated_package(state['package'],target)
        if digest(package)!=row['projected_source_hash']:
            raise ValueError('迁移投影与冻结预览不一致，请重新预览')
        store.validate(package)
        context={'preview_id':preview_id,'preview_hash':preview_hash,'from_schema_version':state['package']['schema_version'],
                 'target_schema_version':target['schema_version'],'target_schema_hash':digest(target)}
        store._write_source_change(db,state,package,reason,actor,context)
        event=store._event(db,case_id,{'action':'schema_migrated','target_id':'task','actor':actor,'reason':reason,
            **context,'previous_source_hash':state['source_hash'],'source_hash':digest(package),
            'previous_run_id':row['current_run_id'],'previous_snapshot_id':row['current_snapshot_id'],
            'review_records':row['review_records'],'material_link_changes':links,'material_link_warnings':warnings})
        receipt={'receipt_id':uuid4().hex,'preview_id':preview_id,'preview_hash':preview_hash,'case_id':case_id,
                 'applied_at':now(),'actor':actor,'reason':reason,'event_id':event['event_id'],
                 'previous_source_hash':state['source_hash'],'source_hash':digest(package),
                 'previous_schema_version':state['package']['schema_version'],'schema_version':target['schema_version'],
                 'target_schema_hash':digest(target),'previous_run_id':row['current_run_id'],
                 'previous_snapshot_id':row['current_snapshot_id'],'review_records':row['review_records'],
                 'changed_sources':row['changed_sources'],'affected_nodes':row['affected_nodes'],
                 'requires_full_recheck':row['requires_full_recheck'],'material_link_changes':links,
                 'material_link_warnings':warnings,'run_started':False}
        db.execute('INSERT INTO migration_receipts VALUES (?,?,?,?,?)',
                   (preview_id,case_id,canonical(receipt),digest(receipt),receipt['applied_at']))
    return {'receipt':{**receipt,'receipt_hash':digest(receipt)},'case':store.get(case_id),
            'preview':get_preview(store,preview_id,implementation_hash)}
