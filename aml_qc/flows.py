"""Deterministic fund-flow profile for one visible synthetic account (facts only, no thresholds)."""
from collections import defaultdict

from . import core
from .ingest import transaction_cents

VERSION = "flows-1.0"


def _yuan(cents):
    return f"{cents // 100}.{cents % 100:02d}"


def flow_profile(case, start=None, end=None, top=5):
    """Totals, counterparty concentration, pass-through and timing facts for a half-open window.

    Every figure is computed in integer cents from the visible rows and carries the coverage of the
    same window: a partial window describes only what is visible, never the complete account.
    """
    start = start or case["coverage_start"]
    end = end or case["coverage_end"]
    query = core.query_transactions(case, {"start": start, "end": end})
    rows = query["rows"]
    names = {c["counterparty_token"]: c.get("display_name_masked") or c["counterparty_token"]
             for c in case.get("counterparties", [])}
    by_direction = {}
    for direction in ("in", "out"):
        picked = [r for r in rows if r["direction"] == direction]
        per_party = defaultdict(lambda: {"count": 0, "amount_cents": 0, "transaction_ids": []})
        for r in picked:
            party = per_party[r["counterparty_token"]]
            party["count"] += 1
            party["amount_cents"] += transaction_cents(r)
            party["transaction_ids"].append(r["transaction_id"])
        total = sum(p["amount_cents"] for p in per_party.values())
        ranked = sorted(per_party.items(), key=lambda kv: (-kv[1]["amount_cents"], kv[0]))
        by_direction[direction] = {
            "count": len(picked), "amount_cents": total, "amount": _yuan(total),
            "distinct_counterparties": len(per_party),
            "top_counterparties": [{"counterparty_token": token, "display_name": names.get(token, token),
                                    "count": p["count"], "amount": _yuan(p["amount_cents"]),
                                    "share_percent": round(p["amount_cents"] * 100 / total, 1) if total else None,
                                    "transaction_ids": p["transaction_ids"]}
                                   for token, p in ranked[:top]]}
    days = defaultdict(lambda: {"in": 0, "out": 0})
    for r in rows:
        days[core._time(r["timestamp"]).date().isoformat()][r["direction"]] += transaction_cents(r)
    same_day = sorted(day for day, v in days.items() if v["in"] and v["out"])
    night = [r["transaction_id"] for r in rows if core._time(r["timestamp"]).hour < 6]
    tin, tout = by_direction["in"]["amount_cents"], by_direction["out"]["amount_cents"]
    return {"version": VERSION, "scope": query["scope"], "coverage": query["coverage"],
            "inflow": by_direction["in"], "outflow": by_direction["out"],
            "out_to_in_percent": round(tout * 100 / tin, 1) if tin else None,
            "days_with_both_directions": same_day,
            "night_transaction_ids_00_06": night,
            "transaction_ids": query["transaction_ids"],
            "note": "仅为可见流水的事实统计，不是可疑判定；覆盖不完整时只描述已见部分。"}
