"""Offline holiday/tariff evidence mechanism fixtures, not provider fee receipts."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib

import pytest

from aml_qc import llm
from scripts import run_evaluation as runner


@pytest.fixture(autouse=True)
def no_provider_or_credentials(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('holiday guard checks must not read credentials or dispatch')
    monkeypatch.setattr(llm, 'settings', forbidden)
    monkeypatch.setattr(llm.httpx, 'post', forbidden)
    monkeypatch.setattr(llm.httpx, 'stream', forbidden)


@pytest.fixture
def pricing(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    captures = {}
    for name, url, text in [
        ('government', 'https://www.gov.cn/gongbao/2025/issue_12406/202511/content_7048922.html',
         '<h1>国务院办公厅关于2026年部分节假日安排的通知</h1>'
         '<p>国庆节：10月1日（周四）至7日（周三）放假调休，共7天。9月20日、10月10日上班。</p>'),
        ('pricing', 'https://api-docs.deepseek.com/zh-cn/quick_start/pricing/',
         '<table><tr><th>模型</th><th>deepseek-flash<sup>(1)</sup></th><th>deepseek-v4-pro</th></tr>'
         '<tr><td>百万tokens输入（缓存命中）</td><td>空闲时段</td><td>0.02元</td><td>0.15元</td></tr>'
         '<tr><td>高峰时段</td><td>0.04元</td><td>0.30元</td></tr>'
         '<tr><td>百万tokens输入（缓存未命中）</td><td>空闲时段</td><td>1元</td><td>4.5元</td></tr>'
         '<tr><td>百万tokens输出</td><td>空闲时段</td><td>4元</td><td>13.5元</td></tr></table>'
         '<p>北京时间周一至周五（不含中国法定节假日）9:00 - 12:00、14:00 - 18:00为高峰时段；'
         '其余时段，包括周末及中国法定节假日全天均为空闲时段。</p>')]:
        path = tmp_path / (name + '.html'); path.write_text(text)
        captures[name + '_capture'] = {'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'url': url}
    return {'currency': 'CNY', 'unit_tokens': 1000000, 'period': 'off_peak',
        'source_url': 'https://api-docs.deepseek.com/zh-cn/quick_start/pricing/',
        'rates': {'input_cache_hit': '0.02', 'input_cache_miss': '1', 'output': '4'},
        'valid_until': '2026-10-06T00:00:00+00:00',
        'holiday_verification': {'contract': 'official-holiday-tariff-evidence-1', 'year': 2026,
            'dates': ['2026-10-05'], 'model': 'deepseek-flash', **captures}}


def at(monkeypatch, stamp):
    monkeypatch.setattr(runner, 'now', lambda: datetime.fromisoformat(stamp).astimezone(timezone.utc))


@pytest.mark.parametrize('stamp', ['2026-10-05T08:59:30+08:00', '2026-10-05T09:00:00+08:00',
    '2026-10-05T11:59:30+08:00', '2026-10-05T15:00:00+08:00'])
def test_verified_official_holiday_accepts_actual_peak_clock_and_transition(pricing, monkeypatch, stamp):
    at(monkeypatch, stamp)
    before = deepcopy(pricing)
    runner.verify_pricing_window(pricing, live=True)
    assert pricing == before


@pytest.mark.parametrize('capture', ['government_capture', 'pricing_capture'])
@pytest.mark.parametrize('alteration', ['missing_hash', 'tampered_bytes', 'nonofficial_url', 'unsafe_path'])
def test_both_official_captures_are_checked_each_time(pricing, monkeypatch, capture, alteration):
    at(monkeypatch, '2026-10-05T09:00:00+08:00')
    evidence = pricing['holiday_verification'][capture]
    if alteration == 'missing_hash':
        del evidence['sha256']
    elif alteration == 'tampered_bytes':
        (runner.ROOT / evidence['path']).write_text('changed source bytes')
    elif alteration == 'nonofficial_url':
        evidence['url'] = 'https://example.invalid/not-official'
    else:
        evidence['path'] = '../outside.html'
    with pytest.raises(ValueError, match='holiday'):
        runner.verify_pricing_window(pricing, live=True)


@pytest.mark.parametrize('field,value', [('dates', ['2026-10-10']), ('dates', ['2026-09-29']),
    ('year', 2025), ('model', 'deepseek-v4-pro')])
def test_declared_holiday_must_match_original_notice_year_dates_and_model(pricing, monkeypatch, field, value):
    at(monkeypatch, '2026-10-05T09:00:00+08:00')
    pricing['holiday_verification'][field] = value
    with pytest.raises(ValueError, match='holiday'):
        runner.verify_pricing_window(pricing, live=True)


@pytest.mark.parametrize('alteration', ['wrong_rate', 'wrong_period', 'no_holiday_clause', 'notice_not_a_calendar'])
def test_calendar_alone_cannot_authorize_an_unverified_tariff(pricing, monkeypatch, alteration):
    at(monkeypatch, '2026-10-05T09:00:00+08:00')
    if alteration == 'wrong_rate':
        pricing['rates']['output'] = '8'
    elif alteration == 'wrong_period':
        pricing['period'] = 'peak'
    else:
        name = 'pricing_capture' if alteration == 'no_holiday_clause' else 'government_capture'
        source = pricing['holiday_verification'][name]; path = runner.ROOT / source['path']
        text = path.read_text()
        if alteration == 'no_holiday_clause':
            text = text.replace('中国法定节假日全天均为空闲时段', '请另行查询假期价格')
        else:
            text = '<p>2026年10月5日有通知，但没有官方放假安排。</p>'
        path.write_text(text); source['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='holiday'):
        runner.verify_pricing_window(pricing, live=True)


def test_new_proof_does_not_authorize_unverified_day_or_expiry(pricing, monkeypatch):
    pricing['valid_until'] = '2026-10-07T00:00:00+00:00'
    at(monkeypatch, '2026-10-06T08:59:30+08:00')
    with pytest.raises(ValueError, match='pricing_transition_within_request_timeout'):
        runner.verify_pricing_window(pricing, live=True)
    at(monkeypatch, '2026-10-06T09:00:00+08:00')
    with pytest.raises(ValueError, match='weekday_peak_requires_verified_holiday_calendar'):
        runner.verify_pricing_window(pricing, live=True)
    pricing['valid_until'] = '2026-10-06T00:00:00+00:00'
    at(monkeypatch, '2026-10-06T07:59:00+08:00')
    with pytest.raises(ValueError, match='pricing_expired_or_too_near_expiry'):
        runner.verify_pricing_window(pricing, live=True)


def test_boolean_holiday_claim_cannot_bypass_old_guard(pricing, monkeypatch):
    pricing['holiday_verification'] = True
    at(monkeypatch, '2026-10-05T09:00:00+08:00')
    with pytest.raises(ValueError, match='holiday'):
        runner.verify_pricing_window(pricing, live=True)


@pytest.mark.parametrize('name', ['government_capture', 'pricing_capture'])
def test_each_dispatch_rechecks_capture_bytes_even_outside_peak_clock(pricing, monkeypatch, name):
    at(monkeypatch, '2026-10-05T08:00:00+08:00')
    runner.verify_pricing_window(pricing, live=True)
    source = pricing['holiday_verification'][name]; path = runner.ROOT / source['path']
    path.write_bytes(path.read_bytes() + b'changed after previous verification')
    with pytest.raises(ValueError, match='holiday source capture changed'):
        runner.verify_pricing_window(pricing, live=True)


def test_malformed_short_official_price_row_stays_rejected(pricing, monkeypatch):
    at(monkeypatch, '2026-10-05T09:00:00+08:00')
    source = pricing['holiday_verification']['pricing_capture']; path = runner.ROOT / source['path']
    text = path.read_text().replace('<tr><td>百万tokens输出</td><td>空闲时段</td><td>4元</td><td>13.5元</td></tr>',
                                    '<tr><td>空闲时段</td></tr>')
    path.write_text(text); source['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='holiday_evidence_invalid'):
        runner.verify_pricing_window(pricing, live=True)
