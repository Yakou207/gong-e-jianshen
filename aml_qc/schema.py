"""Validate the frozen rule set before deterministic calculations can execute."""
import json
from pathlib import Path

LABEL_VALUES = {
    'F1': ['met', 'not_met', 'undeterminable'],
    'F2': ['met', 'not_met', 'undeterminable'],
    'count': ['supported', 'contradicted', 'insufficient_evidence'],
    'amount_sum': ['supported', 'contradicted', 'insufficient_evidence'],
    'counterparty': ['supported', 'contradicted', 'insufficient_evidence'],
    'time_range': ['supported', 'contradicted', 'insufficient_evidence'],
    'alert_response': ['addressed', 'not_addressed', 'pending_judgement'],
    'material_relation': ['corresponds', 'mismatch', 'insufficient', 'pending_judgement'],
}


def load_schema():
    return json.loads((Path(__file__).resolve().parents[1] / 'config/schema/S1.0.json').read_text(encoding='utf-8'))


def validate_schema(schema):
    errors = []
    if not isinstance(schema, dict):
        return ['Schema必须为对象']
    if not isinstance(schema.get('schema_version'), str) or not schema['schema_version'].strip():
        errors.append('Schema必须声明schema_version')
    features = schema.get('features')
    if not isinstance(features, dict):
        return errors + ['Schema.features必须包含F1和F2']
    def integer(rule, key, minimum, path):
        value = rule.get(key)
        if type(value) is not int or value < minimum:
            errors.append(f'{path}.{key}须为不小于{minimum}的整数')
            return None
        return value
    for code in ('F1', 'F2'):
        rule = features.get(code)
        if not isinstance(rule, dict):
            errors.append(f'Schema.features.{code}必须为对象')
            continue
        integer(rule, 'ratio_numerator', 0, code)
        integer(rule, 'ratio_denominator', 1, code)
        if code == 'F1':
            integer(rule, 'minimum_days', 1, code)
            if rule.get('window') != 'Asia/Shanghai_calendar_day':
                errors.append('F1仅支持Asia/Shanghai自然日固定窗口')
        else:
            integer(rule, 'window_days', 1, code)
            integer(rule, 'minimum_in_counterparties', 1, code)
            minimum = integer(rule, 'minimum_out_counterparties', 1, code)
            maximum = integer(rule, 'maximum_out_counterparties', 1, code)
            if minimum is not None and maximum is not None and maximum < minimum:
                errors.append('F2转出对手最大值不能小于最小值')
            if rule.get('anchor') != 'coverage_start':
                errors.append('F2仅支持coverage_start对齐的固定窗口')
    templates = schema.get('material_templates')
    if not isinstance(templates, dict):
        errors.append('Schema.material_templates必须为对象')
    else:
        for name, template in templates.items():
            if not isinstance(template, dict):
                errors.append(f'材料模板{name}必须为对象')
                continue
            for role in ('subject_role', 'counterparty_role'):
                if not isinstance(template.get(role), str) or not template[role].strip():
                    errors.append(f'材料模板{name}缺少{role}')
            if template.get('direction') not in ('in', 'out'):
                errors.append(f'材料模板{name}方向须为in/out')
            if template.get('amount_relation') not in ('equal', 'sum_at_most'):
                errors.append(f'材料模板{name}金额关系未实现')
            if template.get('period_relation') != 'transactions_within_material':
                errors.append(f'材料模板{name}期间关系未实现')
            integer({'amount_tolerance_cents': template.get('amount_tolerance_cents', 0)}, 'amount_tolerance_cents', 0, name)
            if 'transaction_count' in template:
                integer(template, 'transaction_count', 1, name)
    labels = schema.get('labels')
    if not isinstance(labels, dict) or not labels:
        errors.append('Schema.labels必须为非空对象')
    else:
        for name, definition in labels.items():
            if name not in LABEL_VALUES:
                errors.append(f'标签{name}尚无对应的核验实现')
            elif not isinstance(definition, dict) or definition.get('allowed_values') != LABEL_VALUES[name]:
                errors.append(f'标签{name}允许值与当前实现不一致')
    return errors


def require_schema(schema):
    errors = validate_schema(schema)
    if errors:
        raise ValueError('; '.join(errors))
    return schema
