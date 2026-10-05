"""Offline receipt replay, conservative reservations, and explicit recovery."""
from copy import deepcopy
from decimal import Decimal

import pytest

from aml_qc.depgraph import digest
from aml_qc.evaluation_budget import (BudgetError, BudgetLedger, CONTEXT_BOUND,
                                      LEGACY_OUTPUT_BOUND, OUTPUT_BOUND, RESERVATION_POLICY, TOKEN_RESERVATION)
from aml_qc.llm import GENERATION, ModelError
from test_evaluation_budget import BUDGET, FakeModel, MESSAGES, PRICING, usage, wrapper


REVIEW_SHA = "a" * 64
LEGACY_TOKENS = CONTEXT_BOUND + LEGACY_OUTPUT_BOUND
LEGACY_AMOUNT = Decimal("2.12992")
LEGACY_BUDGET = BUDGET | {"max_output_tokens": LEGACY_OUTPUT_BOUND, "total_token_budget": LEGACY_TOKENS * 6}


def legacy_events(*receipts):
    events = []
    for number, receipt in enumerate(receipts):
        call_id, run_id = f"old-call-{number}", f"old-run-{number // 6}"
        cost = (Decimal(receipt["prompt_cache_hit_tokens"]) * Decimal("0.04")
                + Decimal(receipt["prompt_cache_miss_tokens"]) * 2
                + Decimal(receipt["completion_tokens"]) * 8) / 1000000
        overrun = (receipt["completion_tokens"] > LEGACY_OUTPUT_BOUND
                   or receipt["total_tokens"] > CONTEXT_BOUND or cost > LEGACY_AMOUNT)
        settlement = {"status": "overrun" if overrun else "settled", "cost": str(cost), "tokens": receipt["total_tokens"]}
        if overrun:
            settlement["reason"] = "provider_bound_exceeded"
        events.append({"event": "call_reserved", "run_id": run_id, "call_id": call_id,
                       "budget": deepcopy(LEGACY_BUDGET), "pricing_hash": digest(PRICING),
                       "reservation": {"currency_amount": str(LEGACY_AMOUNT), "token_count": LEGACY_TOKENS}})
        events.append({"event": "call_finished", "run_id": run_id, "call_id": call_id,
                       "record": {"call_id": call_id, "budget_event_id": call_id, "dispatch_status": "sent",
                                  "status": "failed" if overrun else "completed", "usage": deepcopy(receipt)},
                       "settlement": settlement})
    return events


def stopped_legacy(receipt=None):
    events = legacy_events(usage(), usage(output=4097) if receipt is None else receipt)
    ledger = BudgetLedger("20", PRICING).restore(events)
    return ledger, events


def test_current_reservation_has_provider_policy_and_idle_price_financial_bound():
    pricing = deepcopy(PRICING)
    pricing["rates"].update(input_cache_miss="1", output="4")
    ledger = BudgetLedger("20", pricing)
    reservation = ledger.reserve("new", BUDGET, "call")
    assert ledger.reservation_amount == ledger.held == Decimal("2.62144")
    assert reservation == {"currency_amount": "2.62144", "token_count": TOKEN_RESERVATION,
                           "policy": RESERVATION_POLICY, "context_bound": CONTEXT_BOUND, "output_bound": 393216}
    assert TOKEN_RESERVATION == 1441792
    assert set(ledger.snapshot()) == {"currency", "pricing_hash", "total", "spent", "held", "remaining", "status",
                                      "stop_reason", "runs", "pending_call_ids"}
    with pytest.raises(TypeError):
        ledger.reserve("new", BUDGET, "cheap", legacy=True)


@pytest.mark.parametrize("calls", [2, 322])
def test_legacy_replay_keeps_original_reservation_spend_failure_and_stop(calls):
    events = legacy_events(*([usage()] * (calls - 1)), usage(output=4097))
    original = deepcopy(events)
    ledger = BudgetLedger("20", PRICING).restore(events)
    snapshot = ledger.snapshot()
    expected_spent = Decimal("0.000484") * (calls - 1) + Decimal("0.03318")
    assert ledger.spent == expected_spent and ledger.held == 0
    assert ledger.stop_reason == "provider_bound_exceeded"
    assert snapshot["total"] == "20" and snapshot["remaining"] == str(Decimal("20") - expected_spent)
    assert sum(run["calls"] for run in snapshot["runs"].values()) == calls
    assert sum(run["tokens_spent"] for run in snapshot["runs"].values()) == 310 * (calls - 1) + 4397
    assert not snapshot["pending_call_ids"] and events == original
    assert events[-1]["record"]["status"] == "failed" and events[-1]["settlement"]["status"] == "overrun"
    with pytest.raises(BudgetError, match="stopped"):
        ledger.reserve("new", BUDGET, "new-call")
    assert ledger.snapshot() == snapshot


def test_revalidation_is_persisted_before_clear_and_replays_without_dropping_old_cost():
    ledger, events = stopped_legacy()
    before = ledger.snapshot()
    def durable_sink(event):
        assert ledger.snapshot() == before
        events.append(event)
    event = ledger.revalidate_bounds(digest(before), REVIEW_SHA, durable_sink)
    assert [e["event"] for e in events[-2:]] == ["budget_bounds_revalidation_proposed", "budget_bounds_revalidated"]
    assert event["event"] == "budget_bounds_revalidated"
    assert event["proposal_hash"] == digest(events[-2])
    assert event["review_sha256"] == REVIEW_SHA and event["before_snapshot_hash"] == digest(before)
    assert event["revalidated_call_ids"] == ["old-call-1"]
    after = ledger.snapshot()
    assert event["after_snapshot_hash"] == digest(after)
    assert after == before | {"status": "ready", "stop_reason": None}
    restored = BudgetLedger("20", PRICING).restore(events)
    assert restored.snapshot() == after
    model, _, added = wrapper(ledger=restored, run_id="new-after-recovery")
    model.complete(MESSAGES)
    assert added[0]["reservation"]["output_bound"] == OUTPUT_BOUND
    assert added[0]["reservation"]["policy"] == RESERVATION_POLICY
    assert restored.spent == ledger.spent + Decimal("0.000484") and restored.total == Decimal("20")
    replayed = BudgetLedger("20", PRICING).restore(events + added)
    assert replayed.snapshot() == restored.snapshot()
    assert events[3]["record"]["status"] == "failed" and events[3]["settlement"]["status"] == "overrun"


def test_legacy_unfinished_reservation_keeps_old_amount_and_blocks_recovery():
    events = legacy_events(usage())[:1]
    ledger = BudgetLedger("20", PRICING).restore(events)
    before = ledger.snapshot()
    assert ledger.held == LEGACY_AMOUNT and ledger.reservation_amount == Decimal("5.24288")
    assert before["pending_call_ids"] == ["old-call-0"]
    assert before["runs"]["old-run-0"]["tokens_held"] == LEGACY_TOKENS
    assert ledger.stop_reason == "unresolved_reservation"
    with pytest.raises(BudgetError, match="not_eligible"):
        ledger.revalidate_bounds(digest(before), REVIEW_SHA, [].append)
    assert ledger.snapshot() == before


def test_revalidation_sink_failure_leaves_original_stop_and_money():
    ledger, _ = stopped_legacy()
    before = ledger.snapshot()
    def fsync_failure(event):
        raise OSError("offline fsync failure")
    with pytest.raises(BudgetError, match="not_persisted"):
        ledger.revalidate_bounds(digest(before), REVIEW_SHA, fsync_failure)
    assert ledger.snapshot() == before


def test_proposal_only_never_clears_original_stop_on_restore():
    ledger, events = stopped_legacy()
    before = ledger.snapshot()
    ledger.revalidate_bounds(digest(before), REVIEW_SHA, events.append)
    assert BudgetLedger("20", PRICING).restore(events[:-1]).snapshot() == before


def test_commit_without_a_proposal_is_rejected():
    ledger, events = stopped_legacy()
    ledger.revalidate_bounds(digest(ledger.snapshot()), REVIEW_SHA, events.append)
    del events[-2]
    restored = BudgetLedger("20", PRICING)
    with pytest.raises(ValueError, match="matching proposal"):
        restored.restore(events)
    assert restored.stop_reason == "provider_bound_exceeded"


def test_first_sink_append_then_fsync_failure_replays_as_stopped():
    ledger, events = stopped_legacy()
    before, emitted = ledger.snapshot(), []
    def append_then_fail(event):
        events.append(event)
        emitted.append(event)
        raise OSError("offline fsync failure after proposal append")
    with pytest.raises(BudgetError, match="not_persisted"):
        ledger.revalidate_bounds(digest(before), REVIEW_SHA, append_then_fail)
    assert len(emitted) == 1 and emitted[0]["event"] == "budget_bounds_revalidation_proposed"
    assert ledger.snapshot() == before
    assert BudgetLedger("20", PRICING).restore(events).snapshot() == before


@pytest.mark.parametrize("append_commit,marker_raises", [(True, False), (False, False), (True, True)])
def test_second_sink_failure_marks_commit_uncertain_and_preserves_stop_in_replay(append_commit, marker_raises):
    ledger, events = stopped_legacy()
    before, emitted = ledger.snapshot(), []
    def durable_sink(event):
        emitted.append(event)
        if event["event"] != "budget_bounds_revalidated" or append_commit:
            events.append(event)
        if event["event"] == "budget_bounds_revalidated":
            raise OSError("offline fsync failure at commit")
        if marker_raises and event["event"] == "budget_bounds_revalidation_audit_failed":
            raise OSError("offline fsync failure after failure marker append")
    with pytest.raises(BudgetError, match="commit_not_persisted"):
        ledger.revalidate_bounds(digest(before), REVIEW_SHA, durable_sink)
    assert [e["event"] for e in emitted] == ["budget_bounds_revalidation_proposed", "budget_bounds_revalidated",
                                            "budget_bounds_revalidation_audit_failed"]
    assert emitted[-1]["reason"] == "commit_not_persisted"
    assert emitted[-1]["proposal_hash"] == digest(emitted[0])
    assert ledger.snapshot() == before
    assert BudgetLedger("20", PRICING).restore(events).snapshot() == before


@pytest.mark.parametrize("field,value", [("review_sha256", "b" * 64), ("before_snapshot_hash", "b" * 64),
                                         ("revalidated_call_ids", []), ("output_bound", 4096)])
def test_proposal_binding_tampering_is_rejected_before_commit(field, value):
    ledger, events = stopped_legacy()
    ledger.revalidate_bounds(digest(ledger.snapshot()), REVIEW_SHA, events.append)
    events[-2][field] = value
    restored = BudgetLedger("20", PRICING)
    with pytest.raises((ValueError, BudgetError)):
        restored.restore(events)
    assert restored.stopped


def test_revalidation_hash_uses_canonical_proposal_without_journal_envelope():
    ledger, events = stopped_legacy()
    ledger.revalidate_bounds(digest(ledger.snapshot()), REVIEW_SHA, events.append)
    events[-2].update(sequence=5, recorded_at="offline-envelope")
    events[-1].update(sequence=6, recorded_at="offline-envelope")
    assert BudgetLedger("20", PRICING).restore(events).snapshot() == ledger.snapshot()


@pytest.mark.parametrize("snapshot_hash,review_sha", [("wrong", REVIEW_SHA), ("b" * 64, REVIEW_SHA),
                                                     ("", REVIEW_SHA), (None, ""), (None, "not-a-sha"), (None, None)])
def test_revalidation_requires_exact_snapshot_and_valid_review_hash(snapshot_hash, review_sha):
    ledger, _ = stopped_legacy()
    before, emitted = ledger.snapshot(), []
    with pytest.raises((ValueError, BudgetError)):
        ledger.revalidate_bounds(digest(before) if snapshot_hash is None else snapshot_hash, review_sha, emitted.append)
    assert not emitted and ledger.snapshot() == before


@pytest.mark.parametrize("receipt", [usage(output=OUTPUT_BOUND + 1), usage(hit=0, miss=CONTEXT_BOUND, output=1),
                                      usage(hit=0, miss=10000000, output=0), usage(output=270000)])
def test_revalidation_rejects_provider_or_original_currency_bound_overrun(receipt):
    ledger, _ = stopped_legacy(receipt)
    before, emitted = ledger.snapshot(), []
    with pytest.raises(BudgetError, match="hard_bound"):
        ledger.revalidate_bounds(digest(before), REVIEW_SHA, emitted.append)
    assert ledger.snapshot() == before and not emitted


@pytest.mark.parametrize("kind", ["unknown", "pending", "zero_price_pending", "other_stop", "manual_stop"])
def test_revalidation_never_clears_unknown_pending_or_other_stops(kind):
    if kind == "unknown":
        model, ledger, _ = wrapper(FakeModel(receipt={}))
        with pytest.raises(BudgetError):
            model.complete(MESSAGES)
    elif kind in {"pending", "zero_price_pending"}:
        pricing = deepcopy(PRICING)
        if kind == "zero_price_pending":
            pricing["rates"] = dict.fromkeys(pricing["rates"], "0")
        ledger = BudgetLedger("20", pricing)
        ledger.reserve("pending", BUDGET, "pending-call")
        ledger.stop("provider_bound_exceeded")
    else:
        ledger = BudgetLedger("20", PRICING)
        ledger.stop("event_sink_failed" if kind == "other_stop" else "provider_bound_exceeded")
    before, emitted = ledger.snapshot(), []
    with pytest.raises(BudgetError):
        ledger.revalidate_bounds(digest(before), REVIEW_SHA, emitted.append)
    assert ledger.snapshot() == before and not emitted


def test_revalidation_is_not_repeatable_live_or_in_replay():
    ledger, events = stopped_legacy()
    event = ledger.revalidate_bounds(digest(ledger.snapshot()), REVIEW_SHA, events.append)
    with pytest.raises(BudgetError, match="not_eligible"):
        ledger.revalidate_bounds(digest(ledger.snapshot()), REVIEW_SHA, events.append)
    restored = BudgetLedger("20", PRICING)
    with pytest.raises((ValueError, BudgetError)):
        restored.restore(events + [event])
    assert restored.stopped


@pytest.mark.parametrize("field,value", [("before_snapshot_hash", "b" * 64), ("after_snapshot_hash", "b" * 64),
                                         ("review_sha256", ""), ("revalidated_call_ids", []), ("policy", "legacy"),
                                         ("context_bound", 99999999), ("output_bound", 99999999), ("proposal_hash", "b" * 64)])
def test_restore_rejects_tampered_revalidation_bindings(field, value):
    ledger, events = stopped_legacy()
    ledger.revalidate_bounds(digest(ledger.snapshot()), REVIEW_SHA, events.append)
    events[-1][field] = value
    restored = BudgetLedger("20", PRICING)
    with pytest.raises((ValueError, BudgetError)):
        restored.restore(events)
    assert restored.stopped and restored.spent == Decimal("0.033664")


@pytest.mark.parametrize("change", [{"policy": "legacy"}, {"output_bound": 4096}, {"context_bound": 2}])
def test_restore_rejects_new_reservation_policy_tampering(change):
    model, _, events = wrapper()
    model.complete(MESSAGES)
    events[0]["reservation"].update(change)
    with pytest.raises(ValueError, match="reservation mismatch"):
        BudgetLedger("30", PRICING).restore(events)


@pytest.mark.parametrize("finish_length", [False, True])
def test_request_limit_overrun_is_business_failure_but_known_cost_is_settled(finish_length):
    receipt = usage(output=GENERATION["max_output_tokens"] + 1)
    model, ledger, events = wrapper(FakeModel(receipt=receipt, fail=finish_length))
    with pytest.raises(ModelError):
        model.complete(MESSAGES)
    assert not ledger.stopped and ledger.held == 0 and ledger.spent > 0
    assert model.calls[0]["status"] == "failed" and model.calls[0]["usage"] == receipt
    assert events[-1]["settlement"]["status"] == "settled"
    assert BudgetLedger("30", PRICING).restore(events).snapshot() == ledger.snapshot()


def test_request_above_method_cap_blocks_before_reservation_or_dispatch():
    model, ledger, events = wrapper(budget=BUDGET | {"max_output_tokens": 1})
    with pytest.raises(BudgetError, match="request_output_exceeds_method_budget"):
        model.complete(MESSAGES)
    assert model.base.invocations == 0 and ledger.spent == ledger.held == 0 and not ledger.stopped
    assert events[0]["event"] == "call_blocked" and not ledger.snapshot()["runs"]


def test_explicit_stage_is_dispatched_and_budget_hash_matches_its_request():
    stages = []
    class Staged(FakeModel):
        def complete(self, messages, tools=None, stage=None):
            stages.append(stage)
            return super().complete(messages, tools)
    model, _, events = wrapper(Staged())
    model.complete(MESSAGES, stage="extraction")
    assert stages == ["extraction"]
    request = events[0]["request"]
    assert request["thinking"] == {"type": "disabled"} and request["max_tokens"] == 4096
    assert events[0]["request_hash"] == model.calls[0]["request_hash"] == digest(request)


def test_tool_request_exceeding_4096_by_one_settles_cost_without_global_stop():
    model, ledger, events = wrapper(FakeModel(receipt=usage(output=4097)))
    tools = [{"type": "function", "function": {"name": "fixture"}}]
    with pytest.raises(ModelError, match="当次请求上限"):
        model.complete(MESSAGES, tools=tools)
    assert events[0]["request"]["max_tokens"] == 4096
    assert events[-1]["settlement"]["status"] == "settled"
    assert ledger.spent == Decimal("0.03318") and ledger.held == 0 and not ledger.stopped


def test_current_hard_output_overrun_cannot_be_recovered_as_legacy():
    model, ledger, _ = wrapper(FakeModel(receipt=usage(output=OUTPUT_BOUND + 1)))
    with pytest.raises(BudgetError, match="provider_bound"):
        model.complete(MESSAGES)
    with pytest.raises(BudgetError, match="complete_legacy_receipt"):
        ledger.revalidate_bounds(digest(ledger.snapshot()), REVIEW_SHA, [].append)
    assert ledger.stopped and ledger.held == 0
