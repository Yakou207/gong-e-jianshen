"""Explicit conservative budget consumption preserves unknown provider receipts."""
from copy import deepcopy
from decimal import Decimal

import httpx
import pytest

from aml_qc.depgraph import digest
from aml_qc.evaluation_budget import BudgetError, BudgetLedger, BudgetedModel, TOKEN_RESERVATION
from aml_qc.llm import generation_request
from test_evaluation_budget import BUDGET, FakeModel, MESSAGES, PRICING


REVIEW_SHA = 'b' * 64
IDLE_PRICING = deepcopy(PRICING)
IDLE_PRICING['rates'].update(input_cache_miss='1', output='4')


class TimeoutModel:
    model = 'deepseek-flash'

    def __init__(self):
        self.calls, self.invocations = [], 0

    def complete(self, messages, tools=None):
        self.invocations += 1
        request = generation_request(self.model, messages, tools)
        self.calls.append({'request': request, 'request_hash': digest(request),
                           'status': 'failed', 'usage': None, 'duration_ms': 60000})
        raise httpx.ReadTimeout('Offline mechanism fixture')


def stopped_unknown(*, total='20', budget=None, known_first=True):
    ledger, events = BudgetLedger(total, IDLE_PRICING), []
    if known_first:
        BudgetedModel(FakeModel(), ledger, 'known', BUDGET, events.append).complete(MESSAGES)
    base = TimeoutModel()
    model = BudgetedModel(base, ledger, 'unknown', budget or BUDGET, events.append)
    with pytest.raises(BudgetError, match='budget_charge_unresolved:incomplete_usage'):
        model.complete(MESSAGES)
    return ledger, events, model


def recover(ledger, events, sink=None):
    return ledger.assume_unknown_upper_bound(
        digest(ledger.snapshot()), REVIEW_SHA, ledger.snapshot()['pending_call_ids'], sink or events.append)


def test_explicit_upper_bound_consumption_preserves_actual_cost_usage_and_old_snapshot():
    ledger, events, model = stopped_unknown()
    before, original = ledger.snapshot(), deepcopy(events)
    assert before['spent'] == '0.000244' and before['held'] == '2.62144'
    assert not {'conservative_spent', 'budget_consumed', 'conservative_call_ids'} & set(before)
    call_id = before['pending_call_ids'][0]

    def sink(event):
        assert ledger.snapshot() == before
        events.append(event)

    event = recover(ledger, events, sink)
    after = ledger.snapshot()
    assert events[:len(original)] == original
    assert [e['event'] for e in events[-2:]] == [
        'budget_unknown_charge_proposed', 'budget_unknown_charge_committed']
    assert event['proposal_hash'] == digest(events[-2])
    assert event['after_snapshot_hash'] == digest(after)
    assert event['calls'][0]['receipt_hash'] == digest(original[-1]['record'])
    assert event['calls'][0]['request_hash'] == original[-2]['request_hash']
    assert event['calls'][0]['reservation'] == original[-2]['reservation']
    assert after['status'] == 'ready' and after['pending_call_ids'] == [] and after['held'] == '0'
    assert after['spent'] == before['spent'] and after['total'] == before['total'] == '20'
    assert after['conservative_spent'] == '2.62144' and after['budget_consumed'] == '2.621684'
    assert after['remaining'] == before['remaining']
    assert after['conservative_call_ids'] == [call_id]
    run = after['runs']['unknown']
    assert run['calls'] == 1 and run['spent'] == '0' and run['tokens_spent'] == 0
    assert run['conservative_spent'] == '2.62144' and run['conservative_tokens'] == TOKEN_RESERVATION
    assert model.calls[-1]['usage'] is None and model.calls[-1]['status'] == 'failed'
    assert original[-1]['settlement'] == {
        'status': 'unknown', 'cost': None, 'tokens': None, 'reason': 'incomplete_usage'}
    assert BudgetLedger('20', IDLE_PRICING).restore(events).snapshot() == after
    with pytest.raises(BudgetError, match='not_eligible'):
        ledger.assume_unknown_upper_bound(digest(after), REVIEW_SHA, [call_id], events.append)
    assert ledger.snapshot() == after


@pytest.mark.parametrize('failure_at', [1, 2])
def test_append_then_raise_keeps_stop_and_replay_never_releases_held_money(failure_at):
    ledger, events, _ = stopped_unknown()
    before, count = ledger.snapshot(), 0

    def sink(event):
        nonlocal count
        count += 1
        events.append(event)
        if count == failure_at:
            raise OSError('Offline fsync failure after append')

    with pytest.raises(BudgetError, match='not_persisted'):
        recover(ledger, events, sink)
    assert ledger.snapshot() == before
    assert BudgetLedger('20', IDLE_PRICING).restore(events).snapshot() == before
    if failure_at == 2:
        assert events[-1]['event'] == 'budget_unknown_charge_audit_failed'


@pytest.mark.parametrize('mutation', ['no_proposal', 'review', 'snapshot', 'request', 'reservation', 'duplicate'])
def test_replay_rejects_missing_proposal_or_tampered_binding(mutation):
    ledger, events, _ = stopped_unknown()
    recover(ledger, events)
    if mutation == 'no_proposal':
        del events[-2]
    elif mutation == 'review':
        events[-1]['review_sha256'] = 'c' * 64
    elif mutation == 'snapshot':
        events[-1]['before_snapshot_hash'] = 'c' * 64
    elif mutation == 'request':
        events[-1]['calls'][0]['request_hash'] = 'c' * 64
    elif mutation == 'reservation':
        events[-1]['calls'][0]['reservation']['currency_amount'] = '0'
    else:
        events.append(deepcopy(events[-1]))
    replayed = BudgetLedger('20', IDLE_PRICING)
    with pytest.raises(ValueError):
        replayed.restore(events)
    assert replayed.stopped
    assert replayed.held + replayed.conservative_spent == Decimal('2.62144')
    assert replayed.spent == Decimal('0.000244')


@pytest.mark.parametrize('bad', ['unfinished', 'audit_failed', 'legacy', 'known_usage',
                               'multiple_provider_records', 'request_mismatch', 'other_held'])
def test_upper_bound_recovery_rejects_ineligible_sent_receipts_and_other_stops(bad):
    ledger, events, _ = stopped_unknown(known_first=False)
    if bad == 'unfinished':
        events.pop()
    elif bad == 'audit_failed':
        events.append({'event': 'call_audit_failed', 'call_id': events[-1]['call_id']})
    elif bad == 'legacy':
        events[0]['budget']['max_output_tokens'] = 4096
        events[0]['reservation'] = {'currency_amount': '1.06496', 'token_count': 1048576 + 4096}
    elif bad == 'known_usage':
        ledger, events, _ = stopped_unknown(known_first=False)
        events[-1]['record']['usage'] = {'prompt_tokens': 1, 'prompt_cache_hit_tokens': 1,
                                         'prompt_cache_miss_tokens': 0, 'completion_tokens': 0, 'total_tokens': 1}
    elif bad == 'multiple_provider_records':
        events[-1]['record']['provider_records'] *= 2
    elif bad == 'request_mismatch':
        events[-1]['record']['provider_records'][0]['request_hash'] = 'c' * 64
    else:
        pending = deepcopy(events[0])
        pending.update(call_id='another-pending-call', run_id='another-pending-run')
        events.insert(0, pending)
    replayed = BudgetLedger('20', IDLE_PRICING)
    if bad == 'known_usage':
        with pytest.raises(ValueError, match='settlement mismatch'):
            replayed.restore(events)
        return
    replayed.restore(events)
    before = replayed.snapshot()
    with pytest.raises(BudgetError, match='not_eligible|durable_sent_unknown'):
        recover(replayed, [])
    assert replayed.snapshot() == before


def test_unknown_recovery_cannot_clear_a_provider_hard_bound_stop():
    ledger = BudgetLedger('20', IDLE_PRICING)
    ledger.reserve('hard-bound', BUDGET, 'hard-call')
    receipt = {'prompt_tokens': 0, 'prompt_cache_hit_tokens': 0, 'prompt_cache_miss_tokens': 0,
               'completion_tokens': 393217, 'total_tokens': 393217}
    ledger.commit('hard-call', ledger.settlement('hard-call', receipt), usage=receipt)
    before = ledger.snapshot()
    with pytest.raises(BudgetError, match='not_eligible'):
        ledger.assume_unknown_upper_bound(digest(before), REVIEW_SHA, ['hard-call'], [].append)
    assert ledger.snapshot() == before and ledger.stop_reason == 'provider_bound_exceeded'


@pytest.mark.parametrize('limit', ['global', 'method_currency', 'method_tokens'])
def test_conservative_consumption_still_blocks_dispatch_at_all_budget_limits(limit):
    budget = deepcopy(BUDGET)
    total = '5' if limit == 'global' else '20'
    if limit == 'method_currency':
        budget['currency_limit'] = '3'
    if limit == 'method_tokens':
        budget['total_token_budget'] = TOKEN_RESERVATION
    ledger, events, _ = stopped_unknown(total=total, budget=budget, known_first=False)
    recover(ledger, events)
    base = FakeModel()
    model = BudgetedModel(base, ledger, 'new' if limit == 'global' else 'unknown', budget, events.append)
    expected = 'experiment_currency' if limit == 'global' else limit.replace('tokens', 'token')
    with pytest.raises(BudgetError, match=expected):
        model.complete(MESSAGES)
    assert base.invocations == 0 and model.calls[-1]['dispatch_status'] == 'not_sent'
    assert ledger.spent == 0 and ledger.conservative_spent == Decimal('2.62144')
    assert ledger.total == Decimal(total)


def test_missing_confirmation_and_wrong_snapshot_cannot_append_recovery_events():
    ledger, events, _ = stopped_unknown()
    before, count = ledger.snapshot(), len(events)
    with pytest.raises(ValueError, match='SHA256'):
        ledger.assume_unknown_upper_bound(digest(before), '', before['pending_call_ids'], events.append)
    with pytest.raises(ValueError, match='snapshot mismatch'):
        ledger.assume_unknown_upper_bound('c' * 64, REVIEW_SHA, before['pending_call_ids'], events.append)
    with pytest.raises(BudgetError, match='not_eligible'):
        ledger.assume_unknown_upper_bound(digest(before), REVIEW_SHA, [], events.append)
    assert len(events) == count and ledger.snapshot() == before


@pytest.mark.parametrize('partial', [
    {'completion_tokens': 393217}, {'prompt_tokens': 1048577}, {'total_tokens': 1048577},
    {'prompt_cache_miss_tokens': 1048577},
    {'prompt_cache_hit_tokens': 600000, 'prompt_cache_miss_tokens': 600000},
    {'prompt_tokens': 1048576, 'completion_tokens': 1},
    {'completion_tokens': -1}, {'completion_tokens': True}, {'completion_tokens': '10'},
])
def test_partial_usage_cannot_hide_known_hard_overrun_or_invalid_token_values(partial):
    _, events, _ = stopped_unknown(known_first=False)
    events[-1]['record']['usage'] = deepcopy(partial)
    events[-1]['record']['provider_records'][0]['usage'] = deepcopy(partial)
    ledger = BudgetLedger('20', IDLE_PRICING).restore(events)
    before = ledger.snapshot()
    with pytest.raises(BudgetError, match='partial_usage'):
        recover(ledger, events)
    assert ledger.snapshot() == before and ledger.conservative_spent == 0


def test_valid_partial_usage_stays_partial_after_conservative_consumption():
    _, events, _ = stopped_unknown(known_first=False)
    partial = {'prompt_tokens': 10, 'completion_tokens': 20, 'total_tokens': 30}
    events[-1]['record']['usage'] = deepcopy(partial)
    events[-1]['record']['provider_records'][0]['usage'] = deepcopy(partial)
    original = deepcopy(events)
    ledger = BudgetLedger('20', IDLE_PRICING).restore(events)
    recover(ledger, events)
    assert ledger.spent == 0 and ledger.conservative_spent == Decimal('2.62144')
    assert ledger.snapshot()['runs']['unknown']['tokens_spent'] == 0
    assert events[:len(original)] == original
    assert events[1]['record']['usage'] == partial
    assert events[1]['settlement']['cost'] is None


def test_zero_price_consumption_still_discloses_unknown_calls_and_conservative_tokens():
    pricing = deepcopy(IDLE_PRICING)
    pricing['rates'] = {key: '0' for key in pricing['rates']}
    ledger, events = BudgetLedger('20', pricing), []
    budget = BUDGET | {'total_token_budget': TOKEN_RESERVATION}
    model = BudgetedModel(TimeoutModel(), ledger, 'unknown', budget, events.append)
    with pytest.raises(BudgetError, match='incomplete_usage'):
        model.complete(MESSAGES)
    before = ledger.snapshot()
    assert 'conservative_call_ids' not in before
    event = recover(ledger, events)
    after = ledger.snapshot()
    assert after['conservative_spent'] == after['budget_consumed'] == '0'
    assert after['conservative_call_ids'] == before['pending_call_ids']
    assert after['runs']['unknown']['conservative_tokens'] == TOKEN_RESERVATION
    assert after['runs']['unknown']['tokens_spent'] == 0
    assert event['after_snapshot_hash'] == digest(after)
    assert BudgetLedger('20', pricing).restore(events).snapshot() == after
    with pytest.raises(BudgetError, match='method_token'):
        BudgetedModel(FakeModel(), ledger, 'unknown', budget, events.append).complete(MESSAGES)


def test_new_calls_and_second_unknown_charge_replay_without_losing_prior_consumption():
    ledger, events, _ = stopped_unknown()
    recover(ledger, events)
    BudgetedModel(FakeModel(), ledger, 'after-recovery', BUDGET, events.append).complete(MESSAGES)
    with pytest.raises(BudgetError, match='incomplete_usage'):
        BudgetedModel(TimeoutModel(), ledger, 'second-unknown', BUDGET, events.append).complete(MESSAGES)
    before = ledger.snapshot()
    recover(ledger, events)
    after = ledger.snapshot()
    assert ledger.spent == Decimal('0.000488')
    assert after['conservative_spent'] == '5.24288' and len(after['conservative_call_ids']) == 2
    assert after['remaining'] == before['remaining']
    assert BudgetLedger('20', IDLE_PRICING).restore(events).snapshot() == after
