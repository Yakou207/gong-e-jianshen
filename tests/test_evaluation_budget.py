"""Budget mechanics with synthetic model receipts only; no provider requests."""
from copy import deepcopy
from decimal import Decimal

import pytest

from aml_qc.depgraph import digest
from aml_qc.evaluation_budget import (BudgetError, BudgetLedger, BudgetedModel,
                                      CONTEXT_BOUND, OUTPUT_BOUND, TOKEN_RESERVATION)
from aml_qc.llm import GENERATION, ModelError, generation_request


PRICING = {"currency": "CNY", "effective_at": "2026-09-10T12:00:00+08:00",
           "source_url": "https://api-docs.deepseek.com/zh-cn/quick_start/pricing/", "unit_tokens": 1000000,
           "rates": {"input_cache_hit": "0.04", "input_cache_miss": "2", "output": "8"}}
BUDGET = {"max_calls": 6, "max_output_tokens": GENERATION["max_output_tokens"], "total_token_budget": TOKEN_RESERVATION * 6,
          "currency_limit": "20"}
MESSAGES = [{"role": "user", "content": "Synthetic fixture; no real case or API call."}]


def usage(hit=100, miss=200, output=10):
    return {"prompt_tokens": hit + miss, "prompt_cache_hit_tokens": hit,
            "prompt_cache_miss_tokens": miss, "completion_tokens": output,
            "total_tokens": hit + miss + output}


class FakeModel:
    model = "deepseek-flash"

    def __init__(self, receipt=None, fail=False, hook=None):
        self.calls, self.invocations = [], 0
        self.config = {"DEEPSEEK_API_KEY": "SECRET-NOT-FOR-LOGS"}
        self.receipt = usage() if receipt is None else receipt
        self.fail, self.hook = fail, hook

    def complete(self, messages, tools=None):
        self.invocations += 1
        if self.hook:
            self.hook()
        record = {"request_hash": "same-provider-hash", "status": "failed" if self.fail else "completed",
                  "usage": deepcopy(self.receipt), "model_returned": self.model, "finish_reason": "length" if self.fail else "stop",
                  "response": {"role": "assistant", "content": "{}", "reasoning_content": "HIDDEN-REASONING"},
                  "Authorization": "SECRET-HEADER", "config": self.config, "duration_ms": 12}
        self.calls.append(record)
        if self.fail:
            raise ModelError("fixture failure")
        return record["response"]


def wrapper(base=None, *, total="30", budget=None, sink=None, ledger=None, run_id="run-1"):
    ledger = ledger or BudgetLedger(total, PRICING)
    events = []
    model = BudgetedModel(base or FakeModel(), ledger, run_id, budget or BUDGET, sink or events.append)
    return model, ledger, events


def test_exact_worst_case_reservation_and_peak_prices():
    ledger = BudgetLedger("6", PRICING)
    assert ledger.reservation_amount == Decimal("5.24288")
    assert TOKEN_RESERVATION == 1441792
    assert ledger.pricing_hash == digest(PRICING)
    ledger.reserve("run", BUDGET, "call")
    assert ledger.held == Decimal("5.24288")
    assert ledger.remaining == Decimal("0.75712")
    assert ledger.snapshot()["runs"]["run"]["tokens_held"] == TOKEN_RESERVATION


def test_uses_larger_input_rate_even_if_hit_price_is_higher():
    pricing = deepcopy(PRICING)
    pricing["rates"].update(input_cache_hit="3", input_cache_miss="2")
    assert BudgetLedger("10", pricing).reservation_amount == Decimal("6.291456")


def test_persist_reservation_before_dispatch_and_finish_before_release():
    events, ledger = [], BudgetLedger("30", PRICING)
    def sink(event):
        assert ledger.held == ledger.reservation_amount
        if event["event"] == "call_reserved":
            assert base.invocations == 0
        else:
            assert base.invocations == 1
        events.append(event)
    base = FakeModel(hook=lambda: events[0]["event"] == "call_reserved")
    model = BudgetedModel(base, ledger, "run", BUDGET, sink)
    assert model.complete(MESSAGES)["content"] == "{}"
    assert [e["event"] for e in events] == ["call_reserved", "call_finished"]
    assert ledger.spent == Decimal("0.000484")
    assert ledger.held == 0
    assert ledger.snapshot()["runs"]["run"]["tokens_spent"] == 310
    assert model.calls[0]["dispatch_status"] == "sent"
    assert model.calls[0]["budget_event_id"] == events[0]["call_id"]


@pytest.mark.parametrize("change,total,reason", [
    ({"currency_limit": "2"}, "30", "method_currency"),
    ({"total_token_budget": TOKEN_RESERVATION - 1}, "30", "method_token"),
    ({}, "2", "experiment_currency"),
])
def test_gate_denials_never_dispatch_or_charge(change, total, reason):
    model, ledger, events = wrapper(total=total, budget=BUDGET | change)
    with pytest.raises(BudgetError, match=reason):
        model.complete(MESSAGES)
    assert model.base.invocations == 0
    assert ledger.spent == ledger.held == 0
    assert model.calls[0]["dispatch_status"] == "not_sent"
    assert model.calls[0]["usage"] is None
    assert events[0]["event"] == "call_blocked"


def test_each_invocation_is_counted_despite_identical_hash():
    model, ledger, events = wrapper(budget=BUDGET | {"max_calls": 2})
    model.complete(MESSAGES)
    model.complete(MESSAGES)
    with pytest.raises(BudgetError, match="method_call"):
        model.complete(MESSAGES)
    assert model.base.invocations == 2
    assert len(model.calls) == 3
    assert len({r["call_id"] for r in model.calls}) == 3
    assert model.calls[0]["request_hash"] == model.calls[1]["request_hash"]
    assert ledger.spent == Decimal("0.000968")
    assert ledger.snapshot()["runs"]["run-1"]["calls"] == 2


def test_global_ledger_covers_multiple_runs_in_flight():
    ledger = BudgetLedger("10", PRICING)
    ledger.reserve("other-run", BUDGET, "other-call")
    model, _, events = wrapper(ledger=ledger)
    with pytest.raises(BudgetError, match="experiment_currency"):
        model.complete(MESSAGES)
    assert model.base.invocations == 0
    assert ledger.held == ledger.reservation_amount
    assert events[0]["record"]["dispatch_status"] == "not_sent"


def test_used_tokens_plus_next_reservation_must_fit_run_budget():
    model, ledger, _ = wrapper(budget=BUDGET | {"total_token_budget": TOKEN_RESERVATION})
    model.complete(MESSAGES)
    with pytest.raises(BudgetError, match="method_token"):
        model.complete(MESSAGES)
    assert model.base.invocations == 1


@pytest.mark.parametrize("bad", [
    {}, {"prompt_tokens": 300}, usage() | {"total_tokens": None},
    usage() | {"completion_tokens": True}, usage() | {"prompt_tokens": 400},
    usage() | {"total_tokens": 311}, usage() | {"prompt_cache_hit_tokens": -1},
])
def test_unknown_usage_is_sticky_and_reservation_remains(bad):
    model, ledger, events = wrapper(FakeModel(receipt=bad))
    with pytest.raises(BudgetError, match="^budget_charge_unresolved:incomplete_usage$"):
        model.complete(MESSAGES)
    assert ledger.stopped and ledger.held == ledger.reservation_amount
    assert ledger.spent == 0
    assert events[-1]["settlement"]["status"] == "unknown"
    with pytest.raises(BudgetError, match="stopped"):
        model.complete(MESSAGES)
    assert model.base.invocations == 1
    assert model.calls[-1]["dispatch_status"] == "not_sent"


@pytest.mark.parametrize("receipt", [usage(output=OUTPUT_BOUND + 1), usage(hit=0, miss=CONTEXT_BOUND, output=1),
                                     usage(hit=0, miss=CONTEXT_BOUND * 3, output=0)])
def test_exceeding_documented_bounds_stops_and_records_actual_known_cost(receipt):
    model, ledger, events = wrapper(FakeModel(receipt=receipt))
    with pytest.raises(BudgetError, match="^provider_bound_exceeded$"):
        model.complete(MESSAGES)
    assert ledger.stopped and ledger.held == 0
    assert ledger.spent == Decimal(events[-1]["settlement"]["cost"])
    assert events[-1]["settlement"]["status"] == "overrun"
    assert model.calls[-1]["status"] == "failed"
    assert model.calls[-1]["usage"] == receipt


def test_provider_failure_with_receipt_is_billed_and_raw_is_retained():
    model, ledger, _ = wrapper(FakeModel(fail=True))
    with pytest.raises(ModelError, match="fixture failure"):
        model.complete(MESSAGES)
    assert ledger.spent == Decimal("0.000484")
    assert not ledger.stopped and ledger.held == 0
    assert model.calls[0]["response"]["content"] == "{}"
    assert model.calls[0]["finish_reason"] == "length"
    assert model.calls[0]["status"] == "failed"


def test_timeout_without_provider_record_retains_reservation():
    class Timeout(FakeModel):
        def complete(self, messages, tools=None):
            self.invocations += 1
            raise TimeoutError("unknown delivery")
    model, ledger, _ = wrapper(Timeout())
    with pytest.raises(BudgetError, match="unresolved"):
        model.complete(MESSAGES)
    assert ledger.stopped and ledger.held == ledger.reservation_amount
    assert model.calls[0]["dispatch_status"] == "sent"


def test_reservation_sink_failure_never_calls_provider_and_releases_known_unsent():
    events = []
    def sink(event):
        if event["event"] == "call_reserved":
            raise OSError("fixture disk full")
        events.append(event)
    model, ledger, _ = wrapper(sink=sink)
    with pytest.raises(BudgetError, match="reservation_not_persisted"):
        model.complete(MESSAGES)
    assert model.base.invocations == 0
    assert ledger.stopped and ledger.spent == ledger.held == 0
    assert events[-1]["event"] == "call_blocked"
    assert events[-1]["record"]["dispatch_status"] == "not_sent"


def test_finish_sink_failure_stops_and_holds_even_known_usage():
    events = []
    def sink(event):
        if event["event"] == "call_finished":
            raise OSError("fixture disk full")
        events.append(event)
    model, ledger, _ = wrapper(sink=sink)
    with pytest.raises(BudgetError, match="finish_not_persisted"):
        model.complete(MESSAGES)
    assert model.base.invocations == 1
    assert ledger.stopped and ledger.held == ledger.reservation_amount
    assert ledger.spent == 0
    assert events[-1]["event"] == "call_audit_failed"
    assert model.calls[0]["audit_status"] == "persistence_failed"


def test_finish_written_but_fsync_failed_cannot_release_after_restore():
    events = []
    def sink(event):
        events.append(event)
        if event["event"] == "call_finished":
            raise OSError("fixture fsync failed after append")
    model, ledger, _ = wrapper(sink=sink)
    with pytest.raises(BudgetError, match="finish_not_persisted"):
        model.complete(MESSAGES)
    restored = BudgetLedger("30", PRICING).restore(events)
    assert restored.stopped and restored.held == restored.reservation_amount
    assert restored.spent == 0


def test_observed_overspend_is_not_clipped_to_approved_budget():
    model, ledger, events = wrapper(FakeModel(receipt=usage(hit=0, miss=10000000, output=0)), total="6")
    with pytest.raises(BudgetError, match="provider_bound"):
        model.complete(MESSAGES)
    assert ledger.spent == Decimal("20") and ledger.remaining == Decimal("-14")
    assert ledger.stopped and ledger.held == 0
    restored = BudgetLedger("6", PRICING).restore(events)
    assert restored.snapshot() == ledger.snapshot()


def test_global_exhaustion_stops_other_runs_but_per_run_exhaustion_does_not():
    model, ledger, _ = wrapper(total="2")
    with pytest.raises(BudgetError):
        model.complete(MESSAGES)
    assert ledger.stopped
    other, _, _ = wrapper(ledger=ledger, run_id="other")
    with pytest.raises(BudgetError, match="stopped"):
        other.complete(MESSAGES)
    assert other.base.invocations == 0
    small, good, _ = wrapper(budget=BUDGET | {"currency_limit": "1"})
    with pytest.raises(BudgetError):
        small.complete(MESSAGES)
    assert not good.stopped
    other, _, _ = wrapper(ledger=good, run_id="other")
    other.complete(MESSAGES)
    assert other.base.invocations == 1


@pytest.mark.parametrize("config", [None, "absent"])
def test_models_without_config_keep_workflow_compatible(config):
    base = FakeModel()
    if config == "absent":
        del base.config
    else:
        base.config = None
    model, _, _ = wrapper(base)
    assert model.config == {}


def test_total_sink_failure_is_fail_closed():
    def fail(event):
        raise OSError("fixture unavailable storage")
    model, ledger, _ = wrapper(sink=fail)
    with pytest.raises(BudgetError):
        model.complete(MESSAGES)
    with pytest.raises(BudgetError):
        model.complete(MESSAGES)
    assert model.base.invocations == 0
    assert ledger.stopped and ledger.held == 0


def test_private_provider_metadata_and_reasoning_are_not_logged():
    model, ledger, events = wrapper()
    model.complete(MESSAGES)
    text = repr(events) + repr(model.calls)
    assert "SECRET" not in text and "HIDDEN-REASONING" not in text
    assert model.config is model.base.config
    assert model.model == model.base.model


@pytest.mark.parametrize("use_tools", [False, True])
def test_actual_provider_payload_is_preserved_with_its_own_hash(use_tools):
    class PayloadModel(FakeModel):
        def complete(self, messages, tools=None):
            result = super().complete(messages, tools)
            payload = generation_request(self.model, messages, tools)
            self.calls[-1].update(request=payload, request_hash=digest(payload))
            return result
    model, _, events = wrapper(PayloadModel())
    tools = [{"type": "function", "function": {"name": "fixture"}}] if use_tools else None
    model.complete(MESSAGES, tools=tools)
    provider = model.calls[0]["provider_records"][0]
    assert provider["request"] == model.base.calls[0]["request"]
    assert digest(provider["request"]) == provider["request_hash"]
    assert provider["request_hash"] == model.calls[0]["request_hash"]
    assert events[-1]["record"]["provider_records"][0]["request"] == provider["request"]
    model.base.calls[0]["request"]["messages"][0]["content"] = "changed later"
    assert provider["request"]["messages"] == MESSAGES


def test_provider_request_projection_drops_transport_and_secret_metadata():
    class ContaminatedRecord(FakeModel):
        def complete(self, messages, tools=None):
            result = super().complete(messages, tools)
            self.calls[-1]["request"] = {**generation_request(self.model, messages, tools),
                "headers": {"Authorization": "SECRET-HEADER"},
                "auth": "SECRET-AUTH", "key": "SECRET-KEY", "api_key": "SECRET-API-KEY",
                "DEEPSEEK_API_KEY": "SECRET-ENV-KEY", "config": self.config}
            return result
    model, _, events = wrapper(ContaminatedRecord())
    model.complete(MESSAGES)
    payload = model.calls[0]["provider_records"][0]["request"]
    assert set(payload) == set(generation_request(model.model, MESSAGES))
    assert "SECRET" not in repr(events) + repr(model.calls)


def test_caller_mutation_during_sink_does_not_change_dispatched_request():
    messages, tools = deepcopy(MESSAGES), [{"type": "function", "function": {"name": "fixture"}}]
    captured = []
    class Capture(FakeModel):
        def complete(self, incoming, tools=None):
            captured.append((deepcopy(incoming), deepcopy(tools)))
            return super().complete(incoming, tools)
    def sink(event):
        messages[0]["content"] = "changed after audit"
        tools[0]["function"]["name"] = "changed"
        event.get("request", {}).clear()
    model, _, _ = wrapper(Capture(), sink=sink)
    model.complete(messages, tools)
    assert captured[0][0] == MESSAGES
    assert captured[0][1][0]["function"]["name"] == "fixture"


def test_hidden_multiple_provider_calls_cannot_be_settled_as_one():
    class Multiple(FakeModel):
        def complete(self, messages, tools=None):
            result = super().complete(messages, tools)
            super().complete(messages, tools)
            return result
    model, ledger, _ = wrapper(Multiple())
    with pytest.raises(BudgetError, match="unresolved"):
        model.complete(MESSAGES)
    assert ledger.stopped and ledger.held == ledger.reservation_amount
    assert len(model.calls[0]["provider_records"]) == 2


def test_restore_success_preserves_spent_and_call_limits():
    model, ledger, events = wrapper(budget=BUDGET | {"max_calls": 1})
    model.complete(MESSAGES)
    restored = BudgetLedger("30", PRICING).restore(events)
    assert restored.snapshot() == ledger.snapshot()
    next_model, _, _ = wrapper(ledger=restored, budget=BUDGET | {"max_calls": 1})
    with pytest.raises(BudgetError, match="method_call"):
        next_model.complete(MESSAGES)
    assert next_model.base.invocations == 0


def test_restore_dangling_reservation_is_unknown_not_free():
    model, ledger, events = wrapper()
    model.complete(MESSAGES)
    restored = BudgetLedger("30", PRICING).restore(events[:1])
    assert restored.stopped and restored.held == restored.reservation_amount
    assert restored.spent == 0


def test_restore_completed_blocked_request_does_not_charge():
    model, _, events = wrapper(total="1")
    with pytest.raises(BudgetError):
        model.complete(MESSAGES)
    restored = BudgetLedger("1", PRICING).restore(events)
    assert restored.spent == restored.held == 0
    assert restored.stopped and restored.stop_reason == "experiment_currency_budget_exhausted"


def test_restore_unknown_finish_and_subsequent_blocked():
    model, ledger, events = wrapper(FakeModel(receipt={}))
    for _ in range(2):
        with pytest.raises(BudgetError):
            model.complete(MESSAGES)
    restored = BudgetLedger("30", PRICING).restore(events)
    assert restored.snapshot() == ledger.snapshot()


@pytest.mark.parametrize("mutation", ["price", "reservation", "settlement", "duplicate", "run", "record_id"])
def test_restore_rejects_mismatched_or_duplicate_charge_events(mutation):
    model, _, events = wrapper()
    model.complete(MESSAGES)
    events = deepcopy(events)
    if mutation == "price": events[0]["pricing_hash"] = "wrong"
    if mutation == "reservation": events[0]["reservation"]["currency_amount"] = "0"
    if mutation == "settlement": events[1]["settlement"]["cost"] = "0"
    if mutation == "duplicate": events.append(deepcopy(events[-1]))
    if mutation == "run": events[1]["run_id"] = "other-run"
    if mutation == "record_id": events[1]["record"]["budget_event_id"] = "other-call"
    restored = BudgetLedger("30", PRICING)
    with pytest.raises(ValueError):
        restored.restore(events)
    assert restored.stopped


def test_restore_invalid_event_object_marks_ledger_stopped():
    ledger = BudgetLedger("30", PRICING)
    with pytest.raises(ValueError):
        ledger.restore([None])
    assert ledger.stopped


def test_restore_even_zero_price_unfinished_call_is_unknown():
    pricing = deepcopy(PRICING)
    pricing["rates"] = dict.fromkeys(pricing["rates"], "0")
    ledger = BudgetLedger("30", pricing)
    model, _, events = wrapper(ledger=ledger)
    model.complete(MESSAGES)
    restored = BudgetLedger("30", pricing).restore(events[:1])
    assert restored.stopped and restored.held == 0
    assert restored.snapshot()["pending_call_ids"]


@pytest.mark.parametrize("amount", ["NaN", "Infinity", "-1", "0", 1.0, True])
def test_invalid_global_amount_is_rejected(amount):
    with pytest.raises(ValueError):
        BudgetLedger(amount, PRICING)


@pytest.mark.parametrize("field,value", [("max_calls", True), ("max_calls", 0), ("max_output_tokens", OUTPUT_BOUND + 1),
                                         ("total_token_budget", -1), ("currency_limit", "NaN")])
def test_invalid_method_budgets_rejected_before_any_provider_call(field, value):
    with pytest.raises(ValueError):
        wrapper(budget=BUDGET | {field: value})


def test_existing_run_budget_cannot_be_rewritten():
    model, ledger, _ = wrapper()
    model.complete(MESSAGES)
    next_model, _, _ = wrapper(ledger=ledger, budget=BUDGET | {"max_calls": 99})
    with pytest.raises(BudgetError, match="budget changed"):
        next_model.complete(MESSAGES)
    assert next_model.base.invocations == 0
