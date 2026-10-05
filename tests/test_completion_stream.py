"""Offline SSE transport failures must preserve fees and never become success."""
from contextlib import contextmanager
from copy import deepcopy
import json

import httpx
import pytest

from aml_qc import llm
from aml_qc.evaluation_budget import BudgetLedger, BudgetedModel
from aml_qc.llm import DeepSeek, ModelError
from test_evaluation_budget import BUDGET, MESSAGES, PRICING, usage


class Fragments(httpx.SyncByteStream):
    def __init__(self, text, interrupt=False):
        self.data, self.interrupt = text.encode(), interrupt

    def __iter__(self):
        for start in range(0, len(self.data), 7):
            yield self.data[start:start + 7]
        if self.interrupt:
            raise httpx.ReadTimeout('Offline interrupted stream')


def event(delta=None, *, finish=None, receipt=None, choices=True):
    return 'data: ' + json.dumps({'model': 'deepseek-flash', 'usage': receipt,
        'choices': [{'index': 0, 'delta': delta or {}, 'finish_reason': finish}] if choices else []},
        ensure_ascii=False) + '\n\n'


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(llm, 'settings', lambda: {'DEEPSEEK_API_KEY': 'offline-key',
        'DEEPSEEK_BASE_URL': 'https://model.invalid', 'DEEPSEEK_MODEL': 'deepseek-flash'})
    monkeypatch.setattr(llm.httpx, 'post', lambda *a, **k: pytest.fail('Final calls must stream'))
    monkeypatch.setattr(llm.httpx, 'stream', lambda *a, **k: pytest.fail('No live streaming calls'))


def transport(monkeypatch, text, *, interrupt=False, code=200):
    sent = []
    @contextmanager
    def stream(method, url, **kwargs):
        assert method == 'POST' and kwargs['timeout'] == llm.REQUEST_TIMEOUT_SECONDS
        sent.append(deepcopy(kwargs['json']))
        response = httpx.Response(code, headers={'content-type': 'text/event-stream'},
                                  stream=Fragments(text, interrupt))
        try:
            yield response
        finally:
            response.close()
    monkeypatch.setattr(llm.httpx, 'stream', stream)
    return sent


def model():
    base, events = DeepSeek(), []
    ledger = BudgetLedger('30', PRICING)
    return base, BudgetedModel(base, ledger, 'stream-fixture', BUDGET, events.append), ledger, events


@pytest.mark.parametrize('usage_only', [False, True])
def test_fragmented_unicode_keepalives_and_terminal_usage_are_audited(monkeypatch, usage_only):
    text = ': keep-alive\n\n' + event({'reasoning_content': 'HIDDEN-REASONING'})
    text += event({'content': '{"答'}) + event({'content': '案":"已核验"}'})
    text += event(finish='stop', receipt=None if usage_only else usage())
    if usage_only:
        text += event(receipt=usage(), choices=False)
    sent = transport(monkeypatch, text + 'data: [DONE]\n\n')
    base, governed, ledger, events = model()
    result = governed.complete(MESSAGES)
    assert json.loads(result['content']) == {'答案': '已核验'}
    assert sent[0]['stream'] is True and sent[0]['stream_options'] == {'include_usage': True}
    assert base.calls[0]['response_transport'] == 'sse'
    assert base.calls[0]['stream_keepalives'] == 1
    assert governed.calls[0]['usage'] == base.calls[0]['usage'] == usage()
    assert ledger.spent > 0 and ledger.held == 0 and not ledger.stopped
    assert len(sent) == 1
    saved = json.dumps([base.calls, governed.calls, events, result])
    assert 'HIDDEN-' not in saved and 'reasoning_content' not in saved and 'offline-key' not in saved


@pytest.mark.parametrize('fault', ['missing_done', 'missing_finish', 'aborted', 'length',
                                    'read_after_usage', 'invalid_choices'])
def test_failed_stream_with_known_usage_settles_once_and_remains_failed(monkeypatch, fault):
    finish = None if fault == 'missing_finish' else 'aborted' if fault == 'aborted' else 'length' if fault == 'length' else 'stop'
    text = event({'content': '{}'}) + event(finish=finish, receipt=usage())
    if fault == 'invalid_choices':
        text = 'data: ' + json.dumps({'usage': usage(), 'choices': 'bad'}) + '\n\n'
    if fault not in {'missing_done', 'read_after_usage', 'invalid_choices'}:
        text += 'data: [DONE]\n\n'
    sent = transport(monkeypatch, text, interrupt=fault == 'read_after_usage')
    base, governed, ledger, events = model()
    with pytest.raises(ModelError):
        governed.complete(MESSAGES)
    assert len(sent) == 1 and base.calls[0]['status'] == governed.calls[0]['status'] == 'failed'
    assert base.calls[0]['usage'] == governed.calls[0]['usage'] == usage()
    assert ledger.spent > 0 and ledger.held == 0 and not ledger.stopped
    assert events[-1]['settlement']['status'] == 'settled'


@pytest.mark.parametrize('fault', ['read_before_usage', 'invalid_json', 'no_terminal_usage'])
def test_unknown_stream_usage_stops_the_original_ledger_without_retry(monkeypatch, fault):
    text = event({'reasoning_content': 'HIDDEN-REASONING'})
    text += 'data: bad-json\n\n' if fault == 'invalid_json' else event({'content': '{}'})
    if fault == 'no_terminal_usage':
        text += event(finish='stop') + 'data: [DONE]\n\n'
    sent = transport(monkeypatch, text, interrupt=fault == 'read_before_usage')
    base, governed, ledger, events = model()
    with pytest.raises(ModelError):
        governed.complete(MESSAGES)
    if fault == 'no_terminal_usage':
        assert base.calls[0]['status'] == 'completed'
        assert governed.calls[0]['status'] == 'failed'
    assert len(sent) == 1 and ledger.stopped and ledger.held > 0 and ledger.spent == 0
    assert events[-1]['settlement']['status'] == 'unknown'
    assert 'HIDDEN-' not in json.dumps([base.calls, governed.calls, events])


def test_http_failure_does_not_read_a_completion_or_repeat_the_request(monkeypatch):
    sent = transport(monkeypatch, '', code=503)
    base, governed, ledger, _ = model()
    with pytest.raises(ModelError) as failure:
        governed.complete(MESSAGES)
    assert 'HTTP 503' in str(failure.value.__cause__)
    assert len(sent) == 1 and ledger.stopped and base.calls[0]['status'] == 'failed'
