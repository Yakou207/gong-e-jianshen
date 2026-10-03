"""Human extraction proposals remain separate from machine predictions and labels."""
from copy import deepcopy
from decimal import Decimal
import re

from . import core
from .contracts import ClaimOutput
from .depgraph import digest
from .ingest import parse_time

CLAIM_KINDS = {'count', 'amount_sum', 'counterparty', 'time_range'}


def fidelity_context_hash(case):
    # Transaction/coverage changes require recalculation, not a new interpretation
    # of unchanged text. Query period, identities and specification do matter.
    context = {key: case.get(key) for key in (
        'case_id', 'subject_account_id', 'currency', 'timezone', 'coverage_start',
        'coverage_end', 'documents', 'entity_mappings', 'counterparties', 'schema')}
    for name in ('documents', 'entity_mappings', 'counterparties'):
        context[name] = sorted(context[name] or [], key=digest)
    context['claim_targets'] = sorted(set(case.get('review_scope', {}).get('target_labels', CLAIM_KINDS)) & CLAIM_KINDS)
    return digest(context)


def claim_signature(claim):
    value = {key: value for key, value in claim.items()
             if key not in {'claim_id', 'origin', 'amendment_id', 'replaces_claim_id', 'unit'}}
    value.setdefault('operator', 'exact')
    if value.get('kind') == 'amount_sum' and 'value' in value:
        value['value'] = str(Decimal(value['value']).normalize())
    return digest(value)


def _proposition_scope(claim):
    return {key: claim.get(key) for key in ('kind', 'source', 'direction', 'counterparty_ref', 'start', 'end')}


def normalize_proposed_claim(case, payload, claim_id):
    if not isinstance(payload, dict):
        raise ValueError('人工提议必须提供完整事实结构')
    allowed = {'kind', 'operator', 'value', 'value_cents', 'quote', 'text', 'source',
               'unit', 'currency', 'account_id', 'start', 'end', 'direction',
               'counterparty_ref', 'counterparty_token'}
    if set(payload) - allowed:
        raise ValueError('人工提议包含不允许的字段')
    if payload.get('kind') not in set(case.get('review_scope', {}).get('target_labels', CLAIM_KINDS)) & CLAIM_KINDS:
        raise ValueError('事实类型必须属于本次检查范围')
    if payload.get('account_id', case['subject_account_id']) != case['subject_account_id'] or payload.get('currency', 'CNY') != 'CNY':
        raise ValueError('人工事实对象须为本案账户，币种须为CNY')
    if 'counterparty_ref' in payload and 'counterparty_token' in payload:
        raise ValueError('查询对手只能提供一种明确引用')
    if 'value' in payload and 'value_cents' in payload:
        raise ValueError('不能同时提供元和分金额')
    data = {k: deepcopy(v) for k, v in payload.items() if k in ClaimOutput.model_fields and v is not None}
    if 'counterparty_token' in payload:
        data['counterparty_ref'] = payload['counterparty_token']
    if 'value_cents' in payload:
        cents = payload['value_cents']
        if payload.get('kind') != 'amount_sum' or type(cents) is not int or cents < 0:
            raise ValueError('金额分须为非负整数')
        data['value'] = str(Decimal(cents) / 100)
    quote = payload.get('text', payload.get('quote'))
    if 'quote' in payload and 'text' in payload and payload['quote'] != payload['text']:
        raise ValueError('原文与引用字段不一致')
    if not isinstance(quote, str) or not quote.strip():
        raise ValueError('必须引用当前理由中的非空原文')
    doc = next((d for d in case.get('documents', []) if d['document_id'] == 'narrative'), None)
    source = deepcopy(payload.get('source'))
    if source is None:
        if not doc or doc['text'].count(quote) != 1:
            raise ValueError('原文重复或不存在，请明确指定当前修订和跨度')
        start = doc['text'].index(quote)
        source = {'document_id': 'narrative', 'revision': doc['revision'], 'span': [start, start + len(quote)]}
    if not isinstance(source, dict) or set(source) != {'document_id', 'revision', 'span'} or source['document_id'] != 'narrative' or not core.validate_span(case, source, quote):
        raise ValueError('提议原文不能解析到当前理由修订与跨度')
    data['quote'] = quote
    try:
        normalized = ClaimOutput.model_validate(data).model_dump(exclude_none=True)
    except ValueError as exc:
        raise ValueError('人工事实结构无效：' + str(exc).splitlines()[0]) from exc
    unit = normalized.get('unit')
    permitted_units = {'count': {'次', '笔', 'transactions'}, 'amount_sum': {'元', 'CNY'}}
    if normalized['kind'] in permitted_units and unit is not None and unit not in permitted_units[normalized['kind']]:
        raise ValueError('次数单位须为次/笔，金额输入统一为元；万元须先换算')
    result = {k: v for k, v in normalized.items() if k not in {'quote', 'document_id'}}
    result.update(claim_id=claim_id, text=quote, source=source)
    return result


def _number(text):
    if re.fullmatch(r'\d+(?:\.\d+)?', text):
        return Decimal(text)
    if match := re.fullmatch(r'(\d+(?:\.\d+)?)([十百千万])', text):
        return Decimal(match[1]) * {'十': 10, '百': 100, '千': 1000, '万': 10000}[match[2]]
    digits = dict(zip('零〇一二两三四五六七八九', (0, 0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9)))
    total = section = digit = 0
    for char in text:
        if char in digits:
            digit = digits[char]
        elif char in '十百千':
            section += (digit or 1) * {'十': 10, '百': 100, '千': 1000}[char]
            digit = 0
        elif char == '万':
            total += (section + digit or 1) * 10000
            section = digit = 0
        else:
            return None
    return Decimal(total + section + digit) if text else None


def assess_fidelity(case, claim):
    """Reject only explicit text/structure conflicts; this is not a semantic judge."""
    errors, notes = [], ['原文忠实性仍须由另一操作人核对；机械校验不证明语义正确或身份真实']
    doc = next(d for d in case['documents'] if d['document_id'] == 'narrative')
    text, (start, end) = doc['text'], claim['source']['span']
    boundaries = '。；;\n'
    left = max((text.rfind(char, 0, start) for char in boundaries), default=-1) + 1
    right = min((pos for char in boundaries if (pos := text.find(char, end)) >= 0), default=len(text))
    sentence = text[left:right]
    ambiguous = bool(re.search(r'不是|而是|分别', sentence) or (claim['kind'] == 'amount_sum' and re.search(r'每笔|单价', sentence)))
    if ambiguous:
        notes.append('存在对比、分项或单价语境，不能用单一数值规则判断忠实性')
    numbers = r'[零〇一二两三四五六七八九十百千万\d]+(?:\.\d+)?'
    pattern = rf'({numbers})\s*(?:次|笔)' if claim['kind'] == 'count' else rf'({numbers})\s*(万)?元' if claim['kind'] == 'amount_sum' else None
    matches = list(re.finditer(pattern, sentence)) if pattern and not ambiguous else []
    if len(matches) == 1:
        match = matches[0]
        number = _number(match[1])
        if number is not None:
            if claim['kind'] == 'amount_sum' and match[2]:
                number *= 10000
            if claim.get('operator', 'exact') in {'exists', 'none'}:
                errors.append('原文包含明确数量，不能降为仅存在或不存在的命题')
            elif Decimal(claim['value']) != number:
                errors.append('提议数值与所在完整句的明确次数/金额不一致')
            prefix = sentence[:match.start()]
            operator = ('at_least' if re.search(r'至少|不少于|不低于', prefix) else
                        'at_most' if re.search(r'至多|最多|不超过|不多于|不高于', prefix) else 'exact')
            if claim.get('operator', 'exact') != operator:
                errors.append('提议限定词与原文明确的数量关系不一致')
    elif pattern:
        notes.append('数值语境无法由单一明确次数/金额定位，须人工核对')
    directions = set()
    if re.search(r'支付|付款|转出|付给', sentence): directions.add('out')
    if re.search(r'收到|收款|转入|入账', sentence): directions.add('in')
    if len(directions) == 1 and claim.get('direction') != next(iter(directions)):
        errors.append('提议方向与原文明确的收付方向不一致')
    if claim['kind'] == 'counterparty' and claim.get('operator', 'exact') in {'exact', 'only'} and claim.get('counterparty_ref'):
        errors.append('对手集合的仅有或精确陈述不能先按对手过滤，否则会排除反例')
    if '检查期间' in sentence:
        if parse_time(claim.get('start', case['coverage_start'])) != parse_time(case['coverage_start']) or parse_time(claim.get('end', case['coverage_end'])) != parse_time(case['coverage_end']):
            errors.append('原文明确检查期间，查询范围须保持本案完整检查期间')
    if '本月' in sentence:
        anchor = parse_time(case['coverage_start']).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        after = anchor.replace(year=anchor.year + 1, month=1) if anchor.month == 12 else anchor.replace(month=anchor.month + 1)
        if parse_time(claim.get('start', case['coverage_start'])) != anchor or parse_time(claim.get('end', case['coverage_end'])) != after:
            errors.append('原文明确本月，查询范围不能缩为其他期间')
    return {'blocking_errors': errors, 'notes': notes, 'context_text': sentence}


def merge_claim_amendments(case, machine_claims):
    claims = deepcopy(machine_claims)
    records = case.get('claim_amendments', [])
    superseded = {r['supersedes_amendment_id'] for r in records if r.get('supersedes_amendment_id')}
    views, removed = [], []
    for record in records:
        view = {**deepcopy(record), 'status': 'applied', 'allowed_actions': ['supersede', 'revoke']}
        if record['amendment_id'] in superseded:
            view.update(status='superseded', allowed_actions=[])
        elif record['operation'] == 'revoke':
            view.update(status='revoked', allowed_actions=[])
        else:
            operation, proposed = record['operation'], record.get('proposed_claim')
            target = next((c for c in claims if c['claim_id'] == record.get('target_claim_id')), None)
            equivalent = next((c for c in claims if proposed and claim_signature(c) == claim_signature(proposed)), None)
            conflict = next((c for c in claims if proposed and _proposition_scope(c) == _proposition_scope(proposed) and c is not target and c is not equivalent), None)
            reason = None
            if record.get('context_hash') != fidelity_context_hash(case):
                reason = '原文、身份、检查期间或规范已变化，人工抽取须重新核对'
            elif target and claim_signature(target) != claim_signature(record['original_claim']):
                reason = '待替代的当前机器候选已变化'
            elif operation == 'replace' and target is None and equivalent is None:
                reason = '原机器候选已变化，须明确重新选择替代对象'
            elif operation == 'retire' and target is None:
                reason = '原废弃对象已不在当前机器候选中，须重新核对或撤销该处置'
            elif equivalent and equivalent.get('origin') == 'human_reviewed':
                reason = '相同事实已有另一条有效人工修订，须明确替代该修订，不能静默覆盖'
            elif conflict and operation in {'add', 'replace'}:
                reason = '当前同一原文存在不同结构候选，须明确核对冲突'
            if reason:
                view.update(status='needs_review', reason=reason)
            else:
                replacing = [c for c in (target, equivalent) if c is not None]
                if operation == 'retire':
                    replacing = [target] if target else []
                for old in replacing:
                    if old in claims:
                        claims.remove(old)
                        removed.append({'claim': deepcopy(old), 'amendment_id': record['amendment_id'], 'operation': operation})
                if operation in {'add', 'replace'}:
                    claims.append({**deepcopy(proposed), 'origin': 'human_reviewed',
                                   'amendment_id': record['amendment_id'], 'replaces_claim_id': record.get('target_claim_id')})
        views.append(view)
    return {'claims': claims, 'amendments': views, 'superseded_machine_claims': removed}
