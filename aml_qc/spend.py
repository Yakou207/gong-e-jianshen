"""Workbench spend guard: every real model call is logged; calls stop before a configured cap is reached.

Costs use the official peak-hour tariff (the higher of the two) so the log never under-states spend.
A call whose usage is unknown is charged at its reservation. This is an estimate, not an invoice.
"""
from datetime import datetime, timezone
from decimal import Decimal
import json
import os
from pathlib import Path
from threading import Lock

from .llm import GENERATION, ModelError

ROOT = Path(__file__).resolve().parents[1]
RATES_PER_MILLION = {"input_cache_hit": Decimal("0.04"), "input_cache_miss": Decimal("2"), "output": Decimal("8")}
INPUT_RESERVATION_TOKENS = 200_000
_LOCK = Lock()


def cap():
    return Decimal(os.getenv("AML_QC_SPEND_CAP_CNY", "3"))


def log_path():
    return Path(os.getenv("AML_QC_SPEND_LOG", str(ROOT / "runtime/workbench-spend.jsonl")))


def spent(path=None):
    path = Path(path or log_path())
    if not path.exists():
        return Decimal(0)
    return sum((Decimal(json.loads(line)["cost_cny"]) for line in path.read_text().splitlines() if line.strip()), Decimal(0))


def cost(usage):
    if not isinstance(usage, dict) or not all(isinstance(usage.get(k), int) for k in ("prompt_tokens", "completion_tokens")):
        return None
    hit = int(usage.get("prompt_cache_hit_tokens") or 0)
    miss = usage["prompt_tokens"] - hit
    return (hit * RATES_PER_MILLION["input_cache_hit"] + miss * RATES_PER_MILLION["input_cache_miss"]
            + usage["completion_tokens"] * RATES_PER_MILLION["output"]) / Decimal(1_000_000)


def reservation(stage):
    output = GENERATION["stage_max_output_tokens"][stage]
    return (INPUT_RESERVATION_TOKENS * RATES_PER_MILLION["input_cache_miss"] + output * RATES_PER_MILLION["output"]) / Decimal(1_000_000)


class SpendCappedModel:
    """Wraps a model client: refuses a call that could cross the cap, logs every call's cost."""

    def __init__(self, base, purpose, path=None, limit=None):
        self.base, self.purpose = base, purpose
        self.path, self.limit = Path(path or log_path()), limit if limit is not None else cap()
        self.model = base.model
        self.calls = base.calls

    def complete(self, messages, tools=None, stage=None):
        stage = stage or ("tool_review" if tools else "final")
        with _LOCK:
            if spent(self.path) + reservation(stage) > self.limit:
                raise ModelError(f"工作台累计 API 费用已接近上限 {self.limit} 元，本次调用未发送")
        before = len(self.base.calls)
        try:
            return self.base.complete(messages, tools=tools, stage=stage)
        finally:
            record = self.base.calls[before] if len(self.base.calls) > before else {}
            amount = cost(record.get("usage"))
            entry = {"at": datetime.now(timezone.utc).isoformat(), "purpose": self.purpose, "stage": stage,
                     "status": record.get("status", "not_sent"), "usage": record.get("usage"),
                     "cost_cny": str(amount if amount is not None else (reservation(stage) if record else Decimal(0))),
                     "estimated": amount is None}
            with _LOCK:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a") as handle:
                    handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
