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
from threading import RLock
from uuid import uuid4

from .depgraph import digest
from .llm import ModelError


CONTEXT_BOUND = 1048576
OUTPUT_BOUND = 4096
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
    if value["max_output_tokens"] != OUTPUT_BOUND:
        raise ValueError("evaluation output bound must be 4096")
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
    def remaining(self):
        return self.total - self.spent - self.held

    def stop(self, reason):
        with self._lock:
            if self.stop_reason is None:
                self.stop_reason = reason

    def _run_held(self, run_id):
        held = [r for r in self._reservations.values() if r["run_id"] == run_id and r["state"] == "held"]
        return sum((r["amount"] for r in held), Decimal(0)), sum(r["tokens"] for r in held)

    def reserve(self, run_id, budget, call_id):
        with self._lock:
            budget = _budget(budget)
            if not isinstance(run_id, str) or not run_id or not isinstance(call_id, str) or not call_id:
                raise ValueError("run_id and call_id are required")
            if self.stopped:
                raise BudgetError("budget_ledger_stopped:" + self.stop_reason)
            if call_id in self._reservations:
                raise ValueError("duplicate call_id")
            run = self._runs.get(run_id, {"budget": budget, "calls": 0, "spent": Decimal(0), "tokens": 0})
            if run["budget"] != budget:
                raise ValueError("run budget changed")
            held, token_held = self._run_held(run_id)
            if run["calls"] >= budget["max_calls"]:
                raise BudgetError("method_call_budget_exhausted")
            if run["tokens"] + token_held + TOKEN_RESERVATION > budget["total_token_budget"]:
                raise BudgetError("method_token_budget_exhausted")
            if run["spent"] + held + self.reservation_amount > _decimal(budget["currency_limit"]):
                raise BudgetError("method_currency_budget_exhausted")
            if self.reservation_amount > self.remaining:
                self.stop("experiment_currency_budget_exhausted")
                raise BudgetError("experiment_currency_budget_exhausted")
            self._runs[run_id] = run
            self._reservations[call_id] = {"run_id": run_id, "amount": self.reservation_amount,
                                           "tokens": TOKEN_RESERVATION, "state": "held"}
            run["calls"] += 1
            return {"currency_amount": str(self.reservation_amount), "token_count": TOKEN_RESERVATION}

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
        if (cost > reservation["amount"] or usage["total_tokens"] > CONTEXT_BOUND
                or usage["completion_tokens"] > OUTPUT_BOUND):
            return {"status": "overrun", "cost": str(cost), "tokens": usage["total_tokens"], "reason": "provider_bound_exceeded"}
        return {"status": "settled", "cost": str(cost), "tokens": usage["total_tokens"]}

    def commit(self, call_id, settlement):
        with self._lock:
            reservation = self._reservations[call_id]
            if reservation["state"] != "held":
                raise ValueError("reservation already resolved")
            if settlement["status"] == "unknown":
                self.stop(settlement.get("reason", "unknown_charge"))
                return
            if settlement["status"] not in {"settled", "overrun"}:
                raise ValueError("invalid settlement status")
            cost = _decimal(settlement["cost"], zero=True)
            tokens = settlement["tokens"]
            if type(tokens) is not int or tokens < 0:
                raise ValueError("invalid settlement")
            if settlement["status"] == "settled" and (cost > reservation["amount"] or tokens > TOKEN_RESERVATION):
                raise ValueError("invalid settlement")
            reservation["state"] = settlement["status"]
            run = self._runs[reservation["run_id"]]
            run["spent"] += cost
            run["tokens"] += tokens
            if settlement["status"] == "overrun":
                self.stop(settlement.get("reason", "provider_bound_exceeded"))

    def snapshot(self):
        with self._lock:
            return {"currency": self.currency, "pricing_hash": self.pricing_hash, "total": str(self.total),
                    "spent": str(self.spent), "held": str(self.held), "remaining": str(self.remaining),
                    "status": self.status, "stop_reason": self.stop_reason,
                    "runs": {key: {"budget": deepcopy(run["budget"]), "calls": run["calls"],
                                   "spent": str(run["spent"]), "held": str(self._run_held(key)[0]),
                                   "tokens_spent": run["tokens"], "tokens_held": self._run_held(key)[1]}
                             for key, run in self._runs.items()},
                    "pending_call_ids": [key for key, value in self._reservations.items() if value["state"] == "held"]}

    def restore(self, events):
        """Replay an ordered, complete durable event log into a fresh ledger.

        No ledger may resume past a reservation without a durable finish. The
        caller separately binds the log to the experiment and validates bytes.
        """
        if self._runs or self._reservations or self.stopped:
            raise ValueError("restore requires a fresh ledger")
        finished = set()
        try:
            events = list(events)
            if any(not isinstance(e, dict) for e in events):
                raise ValueError("invalid event record")
            # A sink may append bytes and still report an fsync error. A later
            # durable audit failure prevents that apparent finish releasing money.
            uncertain_finishes = {e.get("call_id") for e in events if e.get("event") == "call_audit_failed"}
            for event in events:
                kind, call_id = event.get("event"), event.get("call_id")
                if kind == "call_reserved":
                    if event.get("pricing_hash") != self.pricing_hash:
                        raise ValueError("event pricing mismatch")
                    reservation = self.reserve(event["run_id"], event["budget"], call_id)
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
                        self.commit(call_id, settlement)
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
                                             "system_fingerprint", "finish_reason") if k in record}
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

    def complete(self, messages, tools=None):
        # The payload is frozen before reserving; mutable caller data cannot
        # silently change the request after its audit event has been persisted.
        messages, tools = deepcopy(messages), deepcopy(tools)
        request = {"model": self.model, "messages": messages, "tools": tools, "max_tokens": OUTPUT_BOUND,
                   "temperature": 0, "thinking": {"type": "disabled"}}
        call_id = str(uuid4())
        record = {"call_id": call_id, "budget_event_id": call_id, "request_hash": digest(request),
                  "request": deepcopy(request), "status": "blocked", "dispatch_status": "not_sent",
                  "usage": None, "provider_records": []}
        self.calls.append(record)
        try:
            reservation = self.ledger.reserve(self.run_id, self.budget, call_id)
        except (BudgetError, ValueError) as error:
            self._blocked(record, str(error))
            raise BudgetError(str(error)) from error
        try:
            self._emit({"event": "call_reserved", "run_id": self.run_id, "call_id": call_id,
                        "budget": self.budget, "pricing_hash": self.ledger.pricing_hash,
                        "request_hash": record["request_hash"], "request": request, "reservation": reservation})
        except Exception as error:
            self.ledger.cancel_not_sent(call_id)
            self.ledger.stop("event_sink_failed")
            self._blocked(record, "reservation_not_persisted")
            raise BudgetError("reservation_not_persisted") from error
        record.update(status="started", dispatch_status="sent")
        prior = len(getattr(self.base, "calls", []))
        result, failure = None, None
        try:
            result = self.base.complete(messages, tools=tools) if tools is not None else self.base.complete(messages)
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
        self.ledger.commit(call_id, settlement)
        if settlement["status"] != "settled":
            raise BudgetError("budget_charge_unresolved:" + settlement["reason"]) from failure
        if failure is not None:
            raise failure
        return result
