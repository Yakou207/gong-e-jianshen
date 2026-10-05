"""Spec §7.3: volatile run accounting must not enter business-input fingerprints."""
import json
from pathlib import Path

from aml_qc.llm import ModelError
from aml_qc.workflow import run_review

ROOT = Path(__file__).resolve().parents[1]


class OfflineFailingModel:
    """Every model step fails; deterministic nodes still compute and fingerprint."""
    model = "deepseek-flash"
    config = {}

    def __init__(self, ledger_spent, max_calls=6):
        self.calls = []
        self.execution_budget_spec = {"method_budget": {"max_calls": max_calls, "max_output_tokens": 16384},
                                      "pricing_hash": "p", "initial_shared_ledger": {"spent": ledger_spent, "remaining": "1"}}

    def complete(self, messages, tools=None, stage=None):
        raise ModelError("offline")


def case():
    return json.loads((ROOT / "data/synthetic/seed-02.json").read_text())


def test_ledger_snapshot_does_not_invalidate_deterministic_nodes():
    first = run_review(case(), mode="fixed", provider="frozen", model=OfflineFailingModel("1.00"))
    second = run_review(case(), mode="fixed", provider="frozen", strategy="incremental",
                        previous=first["snapshot"], model=OfflineFailingModel("2.00"))
    assert second["stats"]["reused"] > 0
    assert "source:execution" not in second["stats"]["changed_sources"]
    # The full accounting snapshot is still recorded with the run for audit.
    assert second["execution"]["evaluation_budget"]["initial_shared_ledger"]["spent"] == "2.00"


def test_business_budget_change_still_invalidates():
    first = run_review(case(), mode="fixed", provider="frozen", model=OfflineFailingModel("1.00", max_calls=6))
    second = run_review(case(), mode="fixed", provider="frozen", strategy="incremental",
                        previous=first["snapshot"], model=OfflineFailingModel("1.00", max_calls=5))
    assert "source:execution" in second["stats"]["changed_sources"]
    assert second["stats"]["reused"] == 0
