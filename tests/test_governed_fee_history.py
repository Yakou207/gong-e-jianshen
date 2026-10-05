"""API regression tests for saved fee receipts; no live transport or real charges."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from aml_qc import llm
from aml_qc.ingest import load_case
from aml_qc.workflow import run_review

ROOT = Path(__file__).resolve().parents[1]
with TemporaryDirectory() as folder:
    with patch.object(llm, "settings", return_value={"AML_QC_DB": str(Path(folder) / "import.sqlite3"),
                      "DEEPSEEK_API_KEY": "", "DEEPSEEK_MODEL": "offline-qa"}):
        from aml_qc import api
api.settings = llm.settings


@pytest.fixture
def session(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("History mechanism tests must not read credentials or call a provider")
    monkeypatch.setattr(llm, "settings", forbidden)
    monkeypatch.setattr(llm.httpx, "post", forbidden)
    monkeypatch.setattr(llm.httpx, "stream", forbidden)
    app = api.create_app(tmp_path / "history.sqlite3", seed=False)
    value = load_case(ROOT / "data/synthetic/seed-01.json")
    app.state.store.create(value)
    with TestClient(app) as client:
        yield client, app.state.store


def save(store, *, attempt=None, budget=None):
    state = store.get("seed-01")
    result = run_review(state["package"], provider="local", mode="fixed")
    if attempt:
        result["governed_development"] = {"attempt_id": attempt, "budget": deepcopy(budget),
            "phase_path": "synthetic-not-a-real-phase", "must_not_project": "synthetic-extra-marker"}
    return store.save_run("seed-01", state["source_hash"], result)["latest_run"]


def budget(**changes):
    return {"currency": "CNY", "spent": "1.25", "held": "0", "remaining": "18.75",
            "total": "20", "status": "ready", "stop_reason": None,
            "runs": {"synthetic-method": {"must_not_project": "synthetic-budget-extra"}}, **changes}


def history(client):
    result = client.get("/api/cases/seed-01")
    assert result.status_code == 200
    return result.json()["governed_fee_history"]


def test_no_paid_receipt_returns_empty_projection(session):
    client, store = session
    assert history(client) == []
    save(store)
    assert history(client) == []


def test_governed_receipt_is_retained_after_local_recheck_without_full_raw_projection(session):
    client, store = session
    paid = save(store, attempt="synthetic-receipt-1", budget=budget())
    local = save(store)
    state = client.get("/api/cases/seed-01").json()
    assert state["latest_run"]["run_id"] == local["run_id"]
    assert "governed_development" not in state["latest_run"]
    receipts = history(client)
    assert len(receipts) == 1
    receipt = receipts[0]
    assert set(receipt) == {"run_id", "source_hash", "created_at", "mode", "run_status", "governed_development"}
    assert receipt["run_id"] == paid["run_id"] and receipt["source_hash"] == paid["source_hash"]
    assert receipt["created_at"] == paid["created_at"]
    assert set(receipt["governed_development"]) == {"budget", "attempt_id"}
    assert receipt["governed_development"]["attempt_id"] == "synthetic-receipt-1"
    assert set(receipt["governed_development"]["budget"]) == {
        "currency", "spent", "held", "remaining", "total", "status", "stop_reason"}
    assert "snapshot" not in receipt and "model_requests" not in receipt
    original = client.get("/api/cases/seed-01/export").json()["historical_runs"][0]
    assert original["governed_development"] == paid["governed_development"]
    receipt["governed_development"]["budget"]["spent"] = "999"
    assert history(client)[0]["governed_development"]["budget"]["spent"] == "1.25"


def test_unknown_usage_and_failure_values_are_preserved_without_summing_cumulatives(session):
    client, store = session
    first = save(store, attempt="synthetic-known-1", budget=budget())
    second_budget = budget(spent="1.25", conservative_spent="2.00", held="2.62144",
        remaining="14.12856", status="stopped", stop_reason="incomplete_usage")
    second = save(store, attempt="synthetic-unknown-2", budget=second_budget)
    save(store)
    receipts = history(client)
    assert [row["run_id"] for row in receipts] == [first["run_id"], second["run_id"]]
    old, unknown = [row["governed_development"]["budget"] for row in receipts]
    assert old["spent"] == "1.25" and "conservative_spent" not in old
    assert unknown["spent"] == "1.25" and unknown["conservative_spent"] == "2.00"
    assert unknown["held"] == "2.62144" and unknown["remaining"] == "14.12856"
    assert unknown["status"] == "stopped" and unknown["stop_reason"] == "incomplete_usage"
    assert unknown["total"] == "20"


def test_source_change_keeps_receipt_bound_to_original_source_and_never_passes_old_result(session):
    client, store = session
    saved = save(store, attempt="synthetic-source-1", budget=budget())
    package = deepcopy(store.get("seed-01")["package"])
    package["documents"][0]["text"] += " 隔离QA来源变化。"
    store.change_source("seed-01", package, "隔离QA费用历史来源绑定")
    state = client.get("/api/cases/seed-01").json()
    assert state["stale"] and not state["can_pass"]
    receipt = history(client)[0]
    assert receipt["source_hash"] == saved["source_hash"] != state["source_hash"]
    assert receipt["run_id"] == saved["run_id"]
    save(store)
    assert history(client)[0] == receipt
