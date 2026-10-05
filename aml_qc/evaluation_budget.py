"""Conservative, pre-dispatch evaluation reservations; no token estimator.

The frozen prices must bound the provider's rates throughout the experiment.
The caller validates that price validity and durably writes each event. This
ledger covers only requests routed through it, not other users of an API key.
``sent`` means dispatched to base.complete, not proof of network delivery.
The base must make at most one provider request per complete (no hidden retry).
"""
from __future__ import annotations

from copy import deepcopy
from decimal import Decimal, InvalidOperation
import re
from threading import RLock
from uuid import uuid4

from .depgraph import digest
from .llm import GENERATION, ModelError, generation_request


CONTEXT_BOUND = 1048576
OUTPUT_BOUND = 393216
LEGACY_OUTPUT_BOUND = 4096
RESERVATION_POLICY = "provider-bounds-1"
TOKEN_RESERVATION = CONTEXT_BOUND + OUTPUT_BOUND
USAGE_FIELDS = ("prompt_tokens", "prompt_cache_hit_tokens", "prompt_cache_miss_tokens",
                "completion_tokens", "total_tokens")


class BudgetError(ModelError):
    pass


def _decimal(value, *, zero=False):
    if not isinstance(value, str):
        raise ValueError("currency amounts must be decimal strings")
    try:
        number = Decimal(value)
    except InvalidOperation:
        raise ValueError("invalid currency amount") from None
    if not number.is_finite() or number < 0 or (not zero and number == 0):
        raise ValueError("invalid currency amount")
    return number


def _budget(value):
    if not isinstance(value, dict):
        raise ValueError("invalid method budget")
    for key in ("max_calls", "max_output_tokens", "total_token_budget"):
        if type(value.get(key)) is not int or value[key] <= 0:
            raise ValueError("method budget requires positive integer " + key)
    if value["max_output_tokens"] > OUTPUT_BOUND:
        raise ValueError("method output bound exceeds provider output bound")
    _decimal(value.get("currency_limit"))
    return {key: deepcopy(value[key]) for key in
            ("max_calls", "max_output_tokens", "total_token_budget", "currency_limit")}


class BudgetLedger:
    def __init__(self, total_currency_budget, pricing):
        self.total = _decimal(total_currency_budget)
        if not isinstance(pricing, dict) or not isinstance(pricing.get("currency"), str) or not pricing["currency"]:
            raise ValueError("pricing currency is required")
        if type(pricing.get("unit_tokens")) is not int or pricing["unit_tokens"] <= 0:
            raise ValueError("invalid pricing unit")
        rates = pricing.get("rates")
        if not isinstance(rates, dict):
            raise ValueError("missing pricing rates")
        self.rates = {k: _decimal(rates.get(k), zero=True)
                      for k in ("input_cache_hit", "input_cache_miss", "output")}
        self.pricing = deepcopy(pricing)
        self.pricing_hash = digest(pricing)
        self.currency = pricing["currency"]
        self.unit_tokens = pricing["unit_tokens"]
        self.reservation_amount = (CONTEXT_BOUND * max(self.rates["input_cache_hit"], self.rates["input_cache_miss"])
                                   + OUTPUT_BOUND * self.rates["output"]) / self.unit_tokens
        self._runs = {}
        self._reservations = {}
        self._lock = RLock()
        self.stop_reason = None
        self._audit_failed = False

    @property
    def stopped(self):
        return self.stop_reason is not None

    @property
    def status(self):
        return "stopped" if self.stopped else "ready"

    @property
    def spent(self):
        return sum((r["spent"] for r in self._runs.values()), Decimal(0))

    @property
    def held(self):
        return sum((r["amount"] for r in self._reservations.values() if r["state"] == "held"), Decimal(0))

    @property
    def conservative_spent(self):
        return sum((r["amount"] for r in self._reservations.values()
                    if r["state"] == "conservative_upper_bound"), Decimal(0))

    @property
    def remaining(self):
        return self.total - self.spent - self.conservative_spent - self.held

    def stop(self, reason):
        with self._lock:
            if reason == "event_sink_failed":
                self._audit_failed = True
            if self.stop_reason is None:
                self.stop_reason = reason

    def _run_held(self, run_id):
        held = [r for r in self._reservations.values() if r["run_id"] == run_id and r["state"] == "held"]
        return sum((r["amount"] for r in held), Decimal(0)), sum(r["tokens"] for r in held)

    def _run_conservative(self, run_id):
        charged = [r for r in self._reservations.values()
                   if r["run_id"] == run_id and r["state"] == "conservative_upper_bound"]
        return sum((r["amount"] for r in charged), Decimal(0)), sum(r["tokens"] for r in charged)

    def reserve(self, run_id, budget, call_id, *, request_hash=None):
        return self._reserve(run_id, budget, call_id, legacy=False, request_hash=request_hash)

    def _reserve(self, run_id, budget, call_id, *, legacy, request_hash=None):
        with self._lock:
            budget = _budget(budget)
            if request_hash is not None and (not isinstance(request_hash, str)
                                            or not re.fullmatch(r"[0-9a-f]{64}", request_hash)):
                raise ValueError("valid request SHA256 is required")
            if legacy and budget["max_output_tokens"] != LEGACY_OUTPUT_BOUND:
                raise ValueError("legacy output bound must be 4096")
            if not isinstance(run_id, str) or not run_id or not isinstance(call_id, str) or not call_id:
                raise ValueError("run_id and call_id are required")
            if self.stopped:
                raise BudgetError("budget_ledger_stopped:" + self.stop_reason)
            if call_id in self._reservations:
                raise ValueError("duplicate call_id")
            run = self._runs.get(run_id, {"budget": budget, "calls": 0, "spent": Decimal(0), "tokens": 0})
            if run["budget"] != budget:
                raise ValueError("run budget changed")
            output_bound = LEGACY_OUTPUT_BOUND if legacy else OUTPUT_BOUND
            tokens = CONTEXT_BOUND + output_bound
            amount = (CONTEXT_BOUND * max(self.rates["input_cache_hit"], self.rates["input_cache_miss"])
                      + output_bound * self.rates["output"]) / self.unit_tokens
            held, token_held = self._run_held(run_id)
            conservative, conservative_tokens = self._run_conservative(run_id)
            if run["calls"] >= budget["max_calls"]:
                raise BudgetError("method_call_budget_exhausted")
            if run["tokens"] + conservative_tokens + token_held + tokens > budget["total_token_budget"]:
                raise BudgetError("method_token_budget_exhausted")
            if run["spent"] + conservative + held + amount > _decimal(budget["currency_limit"]):
                raise BudgetError("method_currency_budget_exhausted")
            if amount > self.remaining:
                self.stop("experiment_currency_budget_exhausted")
                raise BudgetError("experiment_currency_budget_exhausted")
            self._runs[run_id] = run
            self._reservations[call_id] = {"run_id": run_id, "amount": amount, "tokens": tokens, "state": "held",
                                          "legacy": legacy, "context_bound": CONTEXT_BOUND, "output_bound": output_bound,
                                          "request_hash": request_hash}
            run["calls"] += 1
            result = {"currency_amount": str(amount), "token_count": tokens}
            if not legacy:
                result.update(policy=RESERVATION_POLICY, context_bound=CONTEXT_BOUND, output_bound=OUTPUT_BOUND)
            return result

    def cancel_not_sent(self, call_id):
        with self._lock:
            reservation = self._reservations[call_id]
            if reservation["state"] != "held":
                raise ValueError("reservation already resolved")
            reservation["state"] = "cancelled"
            self._runs[reservation["run_id"]]["calls"] -= 1

    def settlement(self, call_id, usage):
        """Calculate only. The caller must persist this before commit releases funds."""
        reservation = self._reservations[call_id]
        if reservation["state"] != "held":
            raise ValueError("reservation already resolved")
        if (not isinstance(usage, dict) or any(type(usage.get(k)) is not int or usage[k] < 0 for k in USAGE_FIELDS)
                or usage["prompt_cache_hit_tokens"] + usage["prompt_cache_miss_tokens"] != usage["prompt_tokens"]
                or usage["prompt_tokens"] + usage["completion_tokens"] != usage["total_tokens"]):
            return {"status": "unknown", "cost": None, "tokens": None, "reason": "incomplete_usage"}
        cost = (usage["prompt_cache_hit_tokens"] * self.rates["input_cache_hit"]
                + usage["prompt_cache_miss_tokens"] * self.rates["input_cache_miss"]
                + usage["completion_tokens"] * self.rates["output"]) / self.unit_tokens
        if (cost > reservation["amount"] or usage["total_tokens"] > reservation["context_bound"]
                or usage["completion_tokens"] > reservation["output_bound"]):
            return {"status": "overrun", "cost": str(cost), "tokens": usage["total_tokens"], "reason": "provider_bound_exceeded"}
        return {"status": "settled", "cost": str(cost), "tokens": usage["total_tokens"]}

    def commit(self, call_id, settlement, *, usage=None, record=None):
        with self._lock:
            reservation = self._reservations[call_id]
            if reservation["state"] != "held":
                raise ValueError("reservation already resolved")
            if settlement["status"] == "unknown":
                reservation.update(usage=deepcopy(usage), settlement=deepcopy(settlement), record=deepcopy(record))
                self.stop(settlement.get("reason", "unknown_charge"))
                return
            if settlement["status"] not in {"settled", "overrun"}:
                raise ValueError("invalid settlement status")
            cost = _decimal(settlement["cost"], zero=True)
            tokens = settlement["tokens"]
            if type(tokens) is not int or tokens < 0:
                raise ValueError("invalid settlement")
            if settlement["status"] == "settled" and (cost > reservation["amount"] or tokens > reservation["tokens"]):
                raise ValueError("invalid settlement")
            reservation["state"] = settlement["status"]
            reservation.update(usage=deepcopy(usage), settlement=deepcopy(settlement))
            run = self._runs[reservation["run_id"]]
            run["spent"] += cost
            run["tokens"] += tokens
            if settlement["status"] == "overrun":
                self.stop(settlement.get("reason", "provider_bound_exceeded"))

    def _revalidation_event(self, before_snapshot_hash, review_sha256):
        if (not isinstance(review_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", review_sha256)
                or not isinstance(before_snapshot_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", before_snapshot_hash)):
            raise ValueError("valid snapshot and review SHA256 are required")
        snapshot = self.snapshot()
        if digest(snapshot) != before_snapshot_hash:
            raise ValueError("revalidation snapshot mismatch")
        if self.stop_reason != "provider_bound_exceeded" or snapshot["pending_call_ids"] or self.held:
            raise BudgetError("bounds_revalidation_not_eligible")
        overruns = {key: r for key, r in self._reservations.items() if r["state"] == "overrun"}
        if not overruns:
            raise BudgetError("bounds_revalidation_requires_old_overrun")
        for reservation in overruns.values():
            receipt, settlement = reservation.get("usage"), reservation.get("settlement")
            if (not reservation["legacy"] or not isinstance(receipt, dict)
                    or any(type(receipt.get(k)) is not int or receipt[k] < 0 for k in USAGE_FIELDS)
                    or receipt["prompt_cache_hit_tokens"] + receipt["prompt_cache_miss_tokens"] != receipt["prompt_tokens"]
                    or receipt["prompt_tokens"] + receipt["completion_tokens"] != receipt["total_tokens"]):
                raise BudgetError("bounds_revalidation_requires_complete_legacy_receipt")
            cost = (receipt["prompt_cache_hit_tokens"] * self.rates["input_cache_hit"]
                    + receipt["prompt_cache_miss_tokens"] * self.rates["input_cache_miss"]
                    + receipt["completion_tokens"] * self.rates["output"]) / self.unit_tokens
            if (receipt["total_tokens"] > CONTEXT_BOUND or receipt["completion_tokens"] > OUTPUT_BOUND
                    or cost > reservation["amount"] or not isinstance(settlement, dict)
                    or settlement.get("status") != "overrun" or settlement.get("cost") != str(cost)
                    or settlement.get("tokens") != receipt["total_tokens"]
                    or receipt["completion_tokens"] <= LEGACY_OUTPUT_BOUND
                    or settlement.get("reason") != "provider_bound_exceeded"):
                raise BudgetError("bounds_revalidation_hard_bound_or_charge_mismatch")
        after = deepcopy(snapshot)
        after.update(status="ready", stop_reason=None)
        return {"event": "budget_bounds_revalidation_proposed", "before_snapshot_hash": before_snapshot_hash,
                "after_snapshot_hash": digest(after), "review_sha256": review_sha256,
                "policy": RESERVATION_POLICY, "context_bound": CONTEXT_BOUND, "output_bound": OUTPUT_BOUND,
                "revalidated_call_ids": sorted(overruns)}

    def revalidate_bounds(self, before_snapshot_hash, review_sha256, event_sink):
        """The caller validates the review and fsyncs proposal, then commit."""
        if not callable(event_sink):
            raise ValueError("a durable event sink is required")
        with self._lock:
            proposal = self._revalidation_event(before_snapshot_hash, review_sha256)
            try:
                event_sink(deepcopy(proposal))
            except Exception as error:
                raise BudgetError("bounds_revalidation_not_persisted") from error
            event = proposal | {"event": "budget_bounds_revalidated", "proposal_hash": digest(proposal)}
            try:
                event_sink(deepcopy(event))
            except Exception as error:
                try:
                    event_sink(deepcopy(event | {"event": "budget_bounds_revalidation_audit_failed",
                                                "reason": "commit_not_persisted"}))
                except Exception:
                    pass
                raise BudgetError("bounds_revalidation_commit_not_persisted") from error
            self.stop_reason = None
            return event

    def _unknown_charge_event(self, before_snapshot_hash, review_sha256, call_ids):
        if (not isinstance(review_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", review_sha256)
                or not isinstance(before_snapshot_hash, str) or not re.fullmatch(r"[0-9a-f]{64}", before_snapshot_hash)):
            raise ValueError("valid snapshot and review SHA256 are required")
        before = self.snapshot()
        if digest(before) != before_snapshot_hash:
            raise ValueError("unknown charge snapshot mismatch")
        held_ids = sorted(key for key, value in self._reservations.items() if value["state"] == "held")
        if (not isinstance(call_ids, list) or not call_ids
                or any(not isinstance(key, str) or not key for key in call_ids)
                or len(set(call_ids)) != len(call_ids) or sorted(call_ids) != held_ids
                or self.stop_reason != "incomplete_usage" or self._audit_failed):
            raise BudgetError("unknown_charge_not_eligible")
        calls = []
        for call_id in held_ids:
            reservation = self._reservations[call_id]
            record, settlement = reservation.get("record"), reservation.get("settlement")
            usage = reservation.get("usage")
            if usage is not None and (not isinstance(usage, dict)
                    or any(value is not None and (type(value) is not int or value < 0)
                           for key, value in usage.items() if key in USAGE_FIELDS)):
                raise BudgetError("unknown_charge_invalid_partial_usage")
            known = {key: value for key, value in (usage or {}).items()
                     if key in USAGE_FIELDS and value is not None}
            hit, miss, output = (known.get(key, 0) for key in
                                 ("prompt_cache_hit_tokens", "prompt_cache_miss_tokens", "completion_tokens"))
            prompt = max(hit + miss, known.get("prompt_tokens", 0))
            lower_cost = (hit * self.rates["input_cache_hit"] + miss * self.rates["input_cache_miss"]
                          + (prompt - hit - miss) * min(self.rates["input_cache_hit"], self.rates["input_cache_miss"])
                          + output * self.rates["output"]) / self.unit_tokens
            if (max(prompt + output, known.get("total_tokens", 0)) > reservation["context_bound"]
                    or output > reservation["output_bound"] or lower_cost > reservation["amount"]):
                raise BudgetError("unknown_charge_partial_usage_exceeds_bound")
            if (reservation["legacy"] or reservation["context_bound"] != CONTEXT_BOUND
                    or reservation["output_bound"] != OUTPUT_BOUND
                    or not isinstance(record, dict) or record.get("call_id") != call_id
                    or record.get("budget_event_id") != call_id or record.get("dispatch_status") != "sent"
                    or record.get("status") != "failed" or record.get("audit_status") == "persistence_failed"
                    or not isinstance(record.get("request"), dict)
                    or type(record["request"].get("max_tokens")) is not int
                    or not 0 < record["request"]["max_tokens"] <= self._runs[reservation["run_id"]]["budget"]["max_output_tokens"]
                    or record.get("request_hash") != digest(record["request"])
                    or record.get("request_hash") != reservation.get("request_hash")
                    or not isinstance(record.get("provider_records"), list) or len(record["provider_records"]) != 1
                    or not isinstance(record["provider_records"][0], dict)
                    or record["provider_records"][0].get("request_hash") != record["request_hash"]
                    or record["provider_records"][0].get("request") != record["request"]
                    or record["provider_records"][0].get("usage") != record.get("usage")
                    or settlement != {"status": "unknown", "cost": None, "tokens": None, "reason": "incomplete_usage"}
                    or record.get("budget_settlement") != settlement
                    or self.settlement(call_id, record.get("usage")) != settlement
                    or record.get("usage") != reservation.get("usage")):
                raise BudgetError("unknown_charge_requires_durable_sent_unknown_receipt")
            calls.append({"call_id": call_id, "run_id": reservation["run_id"],
                          "request_hash": record["request_hash"], "receipt_hash": digest(record),
                          "reservation": {"currency_amount": str(reservation["amount"]),
                                          "token_count": reservation["tokens"], "policy": RESERVATION_POLICY,
                                          "context_bound": CONTEXT_BOUND, "output_bound": OUTPUT_BOUND}})
        after = deepcopy(before)
        conservative = self.conservative_spent + self.held
        charged_ids = sorted(key for key, value in self._reservations.items()
                             if value["state"] == "conservative_upper_bound" or key in held_ids)
        after.update(status="ready", stop_reason=None, held="0", pending_call_ids=[])
        after.update(conservative_spent=str(conservative), budget_consumed=str(self.spent + conservative),
                     conservative_call_ids=charged_ids)
        for run_id, run in after["runs"].items():
            amount, tokens = self._run_conservative(run_id)
            held_amount, held_tokens = self._run_held(run_id)
            amount, tokens = amount + held_amount, tokens + held_tokens
            run.update(held="0", tokens_held=0)
            run_charged_ids = [key for key in charged_ids if self._reservations[key]["run_id"] == run_id]
            if run_charged_ids:
                run.update(conservative_spent=str(amount),
                           budget_consumed=str(self._runs[run_id]["spent"] + amount),
                           conservative_tokens=tokens,
                           conservative_call_ids=run_charged_ids)
        return {"event": "budget_unknown_charge_proposed", "before_snapshot_hash": before_snapshot_hash,
                "after_snapshot_hash": digest(after), "review_sha256": review_sha256,
                "pricing_hash": self.pricing_hash, "policy": RESERVATION_POLICY,
                "context_bound": CONTEXT_BOUND, "output_bound": OUTPUT_BOUND,
                "call_ids": held_ids, "calls": calls}

    def assume_unknown_upper_bound(self, before_snapshot_hash, review_sha256, call_ids, event_sink):
        """Consume reserved ceilings, never invent actual cost or provider usage."""
        if not callable(event_sink):
            raise ValueError("a durable event sink is required")
        with self._lock:
            proposal = self._unknown_charge_event(before_snapshot_hash, review_sha256, call_ids)
            try:
                event_sink(deepcopy(proposal))
            except Exception as error:
                raise BudgetError("unknown_charge_proposal_not_persisted") from error
            event = proposal | {"event": "budget_unknown_charge_committed", "proposal_hash": digest(proposal)}
            try:
                event_sink(deepcopy(event))
            except Exception as error:
                try:
                    event_sink(deepcopy(event | {"event": "budget_unknown_charge_audit_failed",
                                                "reason": "commit_not_persisted"}))
                except Exception:
                    pass
                self.stop("event_sink_failed")
                raise BudgetError("unknown_charge_commit_not_persisted") from error
            for call_id in proposal["call_ids"]:
                self._reservations[call_id]["state"] = "conservative_upper_bound"
            self.stop_reason = None
            return event

    def snapshot(self):
        with self._lock:
            result = {"currency": self.currency, "pricing_hash": self.pricing_hash, "total": str(self.total),
                    "spent": str(self.spent), "held": str(self.held), "remaining": str(self.remaining),
                    "status": self.status, "stop_reason": self.stop_reason,
                    "runs": {key: {"budget": deepcopy(run["budget"]), "calls": run["calls"],
                                   "spent": str(run["spent"]), "held": str(self._run_held(key)[0]),
                                   "tokens_spent": run["tokens"], "tokens_held": self._run_held(key)[1]}
                             for key, run in self._runs.items()},
                    "pending_call_ids": [key for key, value in self._reservations.items() if value["state"] == "held"]}
            charged_ids = sorted(key for key, value in self._reservations.items()
                                 if value["state"] == "conservative_upper_bound")
            if charged_ids:
                result.update(conservative_spent=str(self.conservative_spent),
                              budget_consumed=str(self.spent + self.conservative_spent),
                              conservative_call_ids=charged_ids)
                for run_id, run in result["runs"].items():
                    amount, tokens = self._run_conservative(run_id)
                    run_charged_ids = [key for key in charged_ids if self._reservations[key]["run_id"] == run_id]
                    if run_charged_ids:
                        run.update(conservative_spent=str(amount),
                                   budget_consumed=str(self._runs[run_id]["spent"] + amount),
                                   conservative_tokens=tokens,
                                   conservative_call_ids=run_charged_ids)
            return result

    def restore(self, events):
        """Replay an ordered, complete durable event log into a fresh ledger.

        No ledger may resume past a reservation without a durable finish. The
        caller separately binds the log to the experiment and validates bytes.
        """
        if self._runs or self._reservations or self.stopped:
            raise ValueError("restore requires a fresh ledger")
        finished = set()
        proposals, revalidated = {}, set()
        unknown_proposals, unknown_committed = {}, set()
        try:
            events = list(events)
            if any(not isinstance(e, dict) for e in events):
                raise ValueError("invalid event record")
            # A sink may append bytes and still report an fsync error. A later
            # durable audit failure prevents that apparent finish releasing money.
            uncertain_finishes = {e.get("call_id") for e in events if e.get("event") == "call_audit_failed"}
            uncertain_revalidations = {e.get("proposal_hash") for e in events
                                      if e.get("event") == "budget_bounds_revalidation_audit_failed"}
            uncertain_unknown_charges = {e.get("proposal_hash") for e in events
                                         if e.get("event") == "budget_unknown_charge_audit_failed"}
            for event in events:
                kind, call_id = event.get("event"), event.get("call_id")
                if kind == "call_reserved":
                    if event.get("pricing_hash") != self.pricing_hash:
                        raise ValueError("event pricing mismatch")
                    legacy = "policy" not in event.get("reservation", {})
                    reservation = self._reserve(event["run_id"], event["budget"], call_id, legacy=legacy,
                                                request_hash=event.get("request_hash"))
                    if reservation != event.get("reservation"):
                        raise ValueError("event reservation mismatch")
                elif kind == "call_finished":
                    if call_id in finished or call_id not in self._reservations:
                        raise ValueError("unmatched or duplicate call_finished")
                    if self._reservations[call_id]["run_id"] != event.get("run_id"):
                        raise ValueError("event run mismatch")
                    record = event["record"]
                    if record.get("call_id") != call_id or record.get("budget_event_id") != call_id:
                        raise ValueError("record call identity mismatch")
                    if record.get("dispatch_status") != "sent":
                        raise ValueError("call_finished requires sent record")
                    settlement = self.settlement(call_id, record.get("usage"))
                    if settlement != event.get("settlement"):
                        raise ValueError("event settlement mismatch")
                    if call_id not in uncertain_finishes:
                        self.commit(call_id, settlement, usage=record.get("usage"), record=record)
                    finished.add(call_id)
                elif kind == "call_blocked":
                    if event.get("record", {}).get("dispatch_status") != "not_sent":
                        raise ValueError("call_blocked requires not_sent record")
                    if call_id in self._reservations:
                        self.cancel_not_sent(call_id)
                    if event.get("stop_reason"):
                        self.stop(event["stop_reason"])
                elif kind == "call_audit_failed":
                    self.stop("event_sink_failed")
                elif kind == "budget_bounds_revalidation_proposed":
                    expected = self._revalidation_event(event.get("before_snapshot_hash"), event.get("review_sha256"))
                    if any(event.get(key) != value for key, value in expected.items()):
                        raise ValueError("bounds revalidation proposal mismatch")
                    proposal_hash = digest(expected)
                    if proposal_hash in proposals:
                        raise ValueError("duplicate bounds revalidation proposal")
                    proposals[proposal_hash] = expected
                elif kind in {"budget_bounds_revalidated", "budget_bounds_revalidation_audit_failed"}:
                    proposal_hash = event.get("proposal_hash")
                    if proposal_hash not in proposals:
                        raise ValueError("bounds revalidation requires matching proposal")
                    expected = proposals[proposal_hash] | {"event": kind, "proposal_hash": proposal_hash}
                    if kind == "budget_bounds_revalidation_audit_failed":
                        expected["reason"] = "commit_not_persisted"
                    if any(event.get(key) != value for key, value in expected.items()):
                        raise ValueError("bounds revalidation event mismatch")
                    if kind == "budget_bounds_revalidated":
                        if proposal_hash in revalidated:
                            raise ValueError("duplicate bounds revalidation commit")
                        if self._revalidation_event(event["before_snapshot_hash"], event["review_sha256"]) != proposals[proposal_hash]:
                            raise ValueError("bounds revalidation state changed")
                        revalidated.add(proposal_hash)
                        if proposal_hash not in uncertain_revalidations:
                            self.stop_reason = None
                elif kind == "budget_unknown_charge_proposed":
                    expected = self._unknown_charge_event(event.get("before_snapshot_hash"),
                                                         event.get("review_sha256"), event.get("call_ids"))
                    if any(event.get(key) != value for key, value in expected.items()):
                        raise ValueError("unknown charge proposal mismatch")
                    proposal_hash = digest(expected)
                    if proposal_hash in unknown_proposals:
                        raise ValueError("duplicate unknown charge proposal")
                    unknown_proposals[proposal_hash] = expected
                elif kind in {"budget_unknown_charge_committed", "budget_unknown_charge_audit_failed"}:
                    proposal_hash = event.get("proposal_hash")
                    if proposal_hash not in unknown_proposals:
                        raise ValueError("unknown charge requires matching proposal")
                    expected = unknown_proposals[proposal_hash] | {"event": kind, "proposal_hash": proposal_hash}
                    if kind == "budget_unknown_charge_audit_failed":
                        expected["reason"] = "commit_not_persisted"
                    if any(event.get(key) != value for key, value in expected.items()):
                        raise ValueError("unknown charge event mismatch")
                    if kind == "budget_unknown_charge_audit_failed":
                        self.stop("event_sink_failed")
                    if kind == "budget_unknown_charge_committed":
                        if proposal_hash in unknown_committed:
                            raise ValueError("duplicate unknown charge commit")
                        if self._unknown_charge_event(event["before_snapshot_hash"], event["review_sha256"],
                                                      event["call_ids"]) != unknown_proposals[proposal_hash]:
                            raise ValueError("unknown charge state changed")
                        unknown_committed.add(proposal_hash)
                        if proposal_hash not in uncertain_unknown_charges:
                            for charged_id in event["call_ids"]:
                                self._reservations[charged_id]["state"] = "conservative_upper_bound"
                            self.stop_reason = None
            if self.held:
                self.stop("unresolved_reservation")
            # Zero-cost rates still require every uncertain request to be resolved.
            if any(r["state"] == "held" for r in self._reservations.values()):
                self.stop("unresolved_reservation")
        except Exception:
            self.stop("invalid_event_log")
            raise
        return self


def _safe_response(message):
    return {k: deepcopy(message[k]) for k in ("role", "content", "tool_calls") if k in message} if isinstance(message, dict) else None


def _provider_record(record):
    if not isinstance(record, dict):
        return {}
    value = {k: deepcopy(record[k]) for k in ("request_hash", "status", "duration_ms", "model_returned",
                                             "system_fingerprint", "finish_reason", "generation", "stage") if k in record}
    if isinstance(record.get("request"), dict):
        value["request"] = {k: deepcopy(record["request"][k]) for k in
                            ("model", "messages", "max_tokens", "temperature", "thinking", "reasoning_effort", "tools",
                             "tool_choice", "response_format", "stream", "stream_options") if k in record["request"]}
    usage = record.get("usage")
    value["usage"] = {k: deepcopy(usage[k]) for k in USAGE_FIELDS if k in usage} if isinstance(usage, dict) else None
    if isinstance(record.get("response"), dict):
        value["response"] = _safe_response(record["response"])
    return value


class BudgetedModel:
    def __init__(self, base, ledger, run_id, budget, event_sink):
        self.base, self.ledger, self.run_id = base, ledger, run_id
        self.budget = _budget(budget)
        if not callable(event_sink):
            raise ValueError("a durable event sink is required")
        self.event_sink = event_sink
        self.model = base.model
        self.config = getattr(base, "config", None) or {}
        self.max_calls = self.budget["max_calls"]
        self.calls = []

    def _emit(self, event):
        self.event_sink(deepcopy(event))

    def _blocked(self, record, reason):
        record.update(status="blocked", dispatch_status="not_sent", error_type=reason)
        try:
            self._emit({"event": "call_blocked", "run_id": self.run_id, "call_id": record["call_id"],
                        "record": record, "stop_reason": self.ledger.stop_reason})
        except Exception:
            self.ledger.stop("event_sink_failed")

    def complete(self, messages, tools=None, stage=None):
        # The payload is frozen before reserving; mutable caller data cannot
        # silently change the request after its audit event has been persisted.
        request = (generation_request(self.model, messages, tools, stage=stage) if stage is not None
                   else generation_request(self.model, messages, tools))
        messages, tools = request["messages"], request.get("tools")
        call_id = str(uuid4())
        record = {"call_id": call_id, "budget_event_id": call_id, "request_hash": digest(request),
                  "request": deepcopy(request), "status": "blocked", "dispatch_status": "not_sent",
                  "generation": deepcopy(GENERATION), "usage": None, "provider_records": [],
                  "stage": stage if stage is not None else "tool_review" if tools else "final"}
        self.calls.append(record)
        try:
            if request["max_tokens"] > self.budget["max_output_tokens"]:
                raise BudgetError("request_output_exceeds_method_budget")
            reservation = self.ledger.reserve(self.run_id, self.budget, call_id,
                                              request_hash=record["request_hash"])
        except (BudgetError, ValueError) as error:
            self._blocked(record, str(error))
            raise BudgetError(str(error)) from error
        try:
            self._emit({"event": "call_reserved", "run_id": self.run_id, "call_id": call_id,
                        "budget": self.budget, "pricing_hash": self.ledger.pricing_hash,
                        "request_hash": record["request_hash"], "request": request, "reservation": reservation,
                        "stage": record["stage"]})
        except Exception as error:
            self.ledger.cancel_not_sent(call_id)
            self.ledger.stop("event_sink_failed")
            self._blocked(record, "reservation_not_persisted")
            raise BudgetError("reservation_not_persisted") from error
        record.update(status="started", dispatch_status="sent")
        prior = len(getattr(self.base, "calls", []))
        result, failure = None, None
        try:
            kwargs = {"tools": tools} if tools is not None else {}
            if stage is not None:
                kwargs["stage"] = stage
            result = self.base.complete(messages, **kwargs)
            record.update(status="completed", response=_safe_response(result))
        except Exception as error:
            failure = error
            record.update(status="failed", error_type=type(error).__name__)
        records = getattr(self.base, "calls", [])[prior:]
        record["provider_records"] = [_provider_record(r) for r in records]
        if len(records) == 1:
            safe = record["provider_records"][0]
            for key in ("usage", "duration_ms", "model_returned", "system_fingerprint", "finish_reason"):
                if key in safe:
                    record[key] = deepcopy(safe[key])
            if record.get("response") is None and "response" in safe:
                record["response"] = deepcopy(safe["response"])
        if failure is None and (record.get("finish_reason") == "length"
                or (isinstance(record.get("usage"), dict) and type(record["usage"].get("completion_tokens")) is int
                    and record["usage"]["completion_tokens"] > request["max_tokens"])):
            failure = ModelError("模型输出被截断或超过当次请求上限，不能用于业务通过")
            record.update(status="failed", error_type="ModelError")
        settlement = self.ledger.settlement(call_id, record["usage"])
        record["budget_settlement"] = deepcopy(settlement)
        if settlement["status"] != "settled":
            record["status"] = "failed"
        try:
            self._emit({"event": "call_finished", "run_id": self.run_id, "call_id": call_id,
                        "record": record, "settlement": settlement, "ledger_before_settlement": self.ledger.snapshot()})
        except Exception as error:
            self.ledger.stop("event_sink_failed")
            record.update(status="failed", audit_status="persistence_failed")
            try:
                self._emit({"event": "call_audit_failed", "run_id": self.run_id, "call_id": call_id,
                            "dispatch_status": "sent", "reason": "finish_not_persisted"})
            except Exception:
                pass
            raise BudgetError("finish_not_persisted") from error
        self.ledger.commit(call_id, settlement, usage=record["usage"], record=record)
        if settlement["status"] == "overrun":
            raise BudgetError(settlement["reason"]) from failure
        if settlement["status"] != "settled":
            raise BudgetError("budget_charge_unresolved:" + settlement["reason"]) from failure
        if failure is not None:
            raise failure
        return _safe_response(result)
