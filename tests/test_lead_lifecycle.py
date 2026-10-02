"""Offline workflow-to-store lead lifecycle; fixtures do not prove model quality."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from aml_qc import llm
from aml_qc.annotations import validate_evidence
from aml_qc.depgraph import business_result
from aml_qc.ingest import load_case
from aml_qc.llm import FrozenModel
from aml_qc.store import Store
from aml_qc.workflow import run_review
from test_model_safety import MODEL, ScriptedModel, json_message, semantic_response
from test_store import close_manual


DATA = Path(__file__).resolve().parents[1] / "data" / "synthetic"
QUESTION = "经营资料中的结算安排是否有对应补充说明？"


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    monkeypatch.setattr(llm, "settings", lambda: {"DEEPSEEK_API_KEY": "fixture",
        "DEEPSEEK_MODEL": MODEL, "DEEPSEEK_BASE_URL": "https://model.invalid"})
    monkeypatch.setattr(llm.httpx, "post", lambda *a, **kw: pytest.fail("No live model calls allowed"))


@pytest.fixture
def store(tmp_path):
    result = Store(tmp_path / "leads.sqlite3")
    package = load_case(DATA / "seed-01.json")
    package["review_scope"]["target_labels"] = ["F1", "F2", "alert_response"]
    result.create(package)
    return result


def run_leads(store, *, include=True, pending_upgraded=False, strategy="full", previous=None, products=None):
    state = store.get("seed-01")
    semantic = semantic_response(state["package"])
    semantic["leads"] = [{"question": QUESTION, "observation": "经营资料包含结算安排，建议人工核对说明范围。",
                          "basis_refs": ["document:kyc"]}] if include else []
    if pending_upgraded:
        upgraded = {f["focus_id"] for f in state["package"]["review_scope"].get("upgraded_leads", [])}
        for focus in semantic["focuses"]:
            if focus["focus_id"] in upgraded:
                focus.update(status="not_addressed", quote="", reason="离线机制夹具：理由尚未回应该新增事项")
    model = ScriptedModel([json_message({"claims": [], "unresolved": []}), json_message(semantic)])
    if strategy == "incremental":
        model = FrozenModel(products, MODEL)
    result = run_review(state["package"], provider="frozen", model=model,
                        strategy=strategy, previous=previous)
    if products is not None and strategy == "full":
        products.update(model.request_products)
    return store.save_run("seed-01", state["source_hash"], result)


def act(store, state, action, **kwargs):
    lead = state["lead_dispositions"][0]
    return store.review("seed-01", action=action, target_id=lead["lead_id"],
        reason="人工核对当前观察与所列证据后作出本次处置", actor="synthetic-reviewer",
        snapshot_id=state["latest_run"]["snapshot_id"], expected_event_id=lead["expected_event_id"], **kwargs)


def confirm_task(store):
    state = close_manual(store, "seed-01")
    assert state["can_pass"]
    return store.review("seed-01", action="confirm", target_id="task", reason="逐项确认当前限定范围")


def test_actual_workflow_notice_is_nonblocking_and_exported_after_pass(store):
    state = run_leads(store)
    candidate = state["latest_run"]["lead_candidates"][0]
    issue = next(i for i in state["latest_run"]["issues"] if i["type"] == "new_lead")
    assert issue["lead_id"] == candidate["lead_id"]
    assert issue["severity"] == "notice"
    assert all(i["item_id"] != issue["issue_id"] for i in state["open_items"])
    assert candidate["evidence"][0]["document_id"] == "kyc"
    confirm_task(store)
    exported = store.export("seed-01")
    assert exported["deliverable"]["passed"]
    assert exported["deliverable"]["lead_dispositions"][0]["status"] == "candidate"
    assert exported["lead_dispositions"][0]["candidate"] == candidate


def test_unupgraded_close_invalidates_old_reviews_and_survives_no_new_candidate(store):
    state = run_leads(store)
    passed = confirm_task(store)
    old_run = deepcopy(passed["latest_run"])
    state = act(store, passed, "close_lead")
    assert state["stale"] and not state["can_pass"]
    assert state["lead_dispositions"][0]["status"] == "closed"
    assert not state["package"]["review_scope"]["upgraded_leads"]
    invalidated = next(e for e in reversed(state["review_events"]) if e["action"] == "needs_review")
    assert any(r["target_id"] == "task" for r in invalidated["review_records"])
    record = state["package"]["review_scope"]["lead_dispositions"][0]
    assert record["candidate"] == old_run["lead_candidates"][0]
    assert record["snapshot_id"] == old_run["snapshot_id"]
    assert record["actor"] == "synthetic-reviewer" and record["evidence"]
    state = run_leads(store, include=False)
    assert state["lead_dispositions"][0]["current_valid"]
    assert not state["lead_dispositions"][0]["candidate_current"]
    confirm_task(store)
    exported = store.export("seed-01")
    assert exported["deliverable"]["passed"]
    assert exported["deliverable"]["lead_dispositions"][0]["history"] == [record]
    assert exported["historical_runs"][0]["snapshot_id"] == old_run["snapshot_id"]


def test_upgrade_uses_candidate_question_requires_response_then_can_close(store):
    state = run_leads(store)
    old_source = state["source_hash"]
    upgraded = act(store, state, "upgrade_lead")
    focus = upgraded["package"]["review_scope"]["upgraded_leads"][0]
    assert focus["text"] == QUESTION
    assert focus["text"] != upgraded["package"]["review_scope"]["lead_dispositions"][0]["reason"]
    assert upgraded["stale"] and upgraded["source_hash"] != old_source
    refreshed = run_leads(store, include=False, pending_upgraded=True)
    assert refreshed["lead_dispositions"][0]["status"] == "upgraded"
    assert refreshed["lead_dispositions"][0]["allowed_actions"] == ["close_lead"]
    assert any(c["focus_id"] == focus["focus_id"] and c["required"] for c in refreshed["latest_run"]["semantic_results"])
    assert any(i["target_id"] == "semantic:" + focus["focus_id"] for i in refreshed["open_items"])
    assert not refreshed["can_pass"]
    closed = act(store, refreshed, "close_lead")
    assert closed["stale"] and not closed["package"]["review_scope"]["upgraded_leads"]
    history = closed["lead_dispositions"][0]["history"]
    assert [r["status"] for r in history] == ["upgraded", "closed"]
    assert history[1]["previous_event_id"] == history[0]["event_id"]
    final = run_leads(store, include=False)
    assert all(c["focus_id"] != focus["focus_id"] for c in final["latest_run"]["semantic_results"])
    assert final["lead_dispositions"][0]["history"] == history


def test_business_change_revokes_closed_disposition_without_erasing_it(store):
    state = act(store, run_leads(store), "close_lead")
    state = run_leads(store)
    assert not any(i["type"] == "new_lead" for i in state["latest_run"]["issues"])
    package = deepcopy(state["package"])
    package["profile"]["business_type"] += "，资料待复核"
    store.change_source("seed-01", package, "补充业务背景")
    state = run_leads(store)
    lead = state["lead_dispositions"][0]
    assert lead["status"] == "candidate" and not lead["current_valid"]
    assert lead["history"][0]["status"] == "closed"
    assert any(i["type"] == "new_lead" for i in state["latest_run"]["issues"])
    reclosed = act(store, state, "close_lead")
    assert len(reclosed["lead_dispositions"][0]["history"]) == 2


def test_retracted_unhandled_observation_remains_readonly_in_export(store):
    state = run_leads(store)
    original = deepcopy(state["latest_run"]["lead_candidates"][0])
    state = run_leads(store, include=False)
    record = state["lead_dispositions"][0]
    assert record["candidate"] == original
    assert record["candidate_current"] is False and record["allowed_actions"] == []
    assert store.export("seed-01")["lead_dispositions"][0]["candidate"] == original


@pytest.mark.parametrize("key", ["snapshot_id", "expected_event_id"])
def test_lead_action_rejects_old_optimistic_anchor(store, key):
    state = run_leads(store)
    request = dict(action="close_lead", target_id=state["lead_dispositions"][0]["lead_id"],
        reason="并发测试", snapshot_id=state["latest_run"]["snapshot_id"], expected_event_id=None)
    request[key] = "outdated-id"
    with pytest.raises(ValueError, match="快照|其他人员"):
        store.review("seed-01", **request)
    assert store.get("seed-01")["source_hash"] == state["source_hash"]
    assert not store.get("seed-01")["review_events"]


@pytest.mark.parametrize("mutation", ["event", "run", "source"])
def test_lead_transaction_rechecks_all_anchors_before_writing(store, monkeypatch, mutation):
    state = run_leads(store)
    other = Store(store.path)
    if mutation == "event":
        annotation = state["annotations"][0]
        other.review("seed-01", action="confirm", target_id=annotation["annotation_id"], reason="另一人员抢先复核",
            snapshot_id=state["latest_run"]["snapshot_id"], expected_event_id=None)
    elif mutation == "run":
        run_leads(other)
    else:
        package = deepcopy(state["package"])
        package["profile"]["business_type"] += "新增业务说明"
        other.change_source("seed-01", package, "并发资料更正")
    concurrently_updated = other.get("seed-01")
    monkeypatch.setattr(store, "get", lambda _: deepcopy(state))
    with pytest.raises(ValueError, match="已改变"):
        act(store, state, "close_lead")
    fresh = other.get("seed-01")
    assert fresh["source_hash"] == concurrently_updated["source_hash"]
    assert fresh["review_events"] == concurrently_updated["review_events"]
    assert fresh["latest_run"]["run_id"] == concurrently_updated["latest_run"]["run_id"]


def test_import_and_plain_source_cannot_forge_dispositions_or_upgrade_scope(store, tmp_path):
    closed = act(store, run_leads(store), "close_lead")
    with pytest.raises(ValueError, match="导入不可携带"):
        Store(tmp_path / "forged.sqlite3").create(closed["package"])
    for field in ("lead_dispositions", "upgraded_leads"):
        package = deepcopy(closed["package"])
        package["review_scope"][field] = [] if field == "lead_dispositions" else [{"focus_id": "forged", "text": "伪造升级"}]
        with pytest.raises(ValueError, match="只能通过"):
            store.change_source("seed-01", package, "禁止绕过裁决入口")


@pytest.mark.parametrize("action", ["upgrade_lead", "close_lead"])
def test_lead_scope_change_matches_independent_full_and_incremental_business_results(store, action):
    before = run_leads(store)
    if action == "close_lead":
        act(store, before, "upgrade_lead")
        before = run_leads(store, pending_upgraded=True)
    old_snapshot = deepcopy(before["latest_run"]["snapshot"])
    scope = act(store, before, action)["package"]["review_scope"]
    products = {}
    full = run_leads(store, pending_upgraded=True, products=products)["latest_run"]
    incremental = run_leads(store, pending_upgraded=True, strategy="incremental",
                            previous=before["latest_run"]["snapshot"], products=products)["latest_run"]
    assert business_result(full) == business_result(incremental)
    assert full["lead_candidates"] == incremental["lead_candidates"]
    assert "source:review_scope" in incremental["stats"]["changed_sources"]
    assert before["latest_run"]["snapshot"] == old_snapshot
    if action == "close_lead":
        focus_id = scope["lead_dispositions"][-1]["focus_id"]
        assert not scope["upgraded_leads"]
        assert all(row["focus_id"] != focus_id for row in full["semantic_results"])
        assert all(row["target_id"] != "semantic:" + focus_id for row in full["open_items"])
    else:
        assert any(row["type"] == "upgraded_lead" and row["required"] for row in full["semantic_results"])


def test_scope_upgrade_human_recheck_list_covers_independent_expected_set(store, record_property):
    run_leads(store)
    before = confirm_task(store)
    semantic_targets = {a["annotation_id"] for a in before["annotations"] if a["kind"] == "semantic"}
    # Expected affected human records are defined from the change's meaning,
    # independently of graph traversal or the emitted invalidation list.
    expected = {e["event_id"] for e in before["review_events"]
                if e.get("snapshot_id") == before["latest_run"]["snapshot_id"]
                and e.get("target_id") in semantic_targets | {"task"}}
    assert len(expected) == 2
    after = act(store, before, "upgrade_lead")
    change = next(e for e in reversed(after["review_events"]) if e["action"] == "needs_review")
    actual = {r["event_id"] for r in change["review_records"]}
    assert expected <= actual
    assert len(actual - expected) == 2, "Current conservative policy also re-lists unchanged F1/F2 decisions"
    record_property("mechanism", json.dumps({"mutation": "upgrade_lead", "expected_record_ids": sorted(expected),
        "listed_record_ids": sorted(actual), "missed": len(expected - actual), "extra": len(actual - expected)}, ensure_ascii=False))


def test_changed_model_configuration_cannot_reuse_old_closed_disposition(store, monkeypatch):
    act(store, run_leads(store), "close_lead")
    monkeypatch.setattr(ScriptedModel, "model", "different-offline-model")
    historical = run_leads(store, include=False)
    assert historical["lead_dispositions"][0]["current_valid"] is False
    assert historical["lead_dispositions"][0]["allowed_actions"] == []
    current = run_leads(store)
    assert current["lead_dispositions"][0]["status"] == "candidate"
    assert any(i["type"] == "new_lead" for i in current["latest_run"]["issues"])
    assert "close_lead" in current["lead_dispositions"][0]["allowed_actions"]


def test_closing_historical_upgrade_separates_origin_from_current_decision_evidence(store):
    first = run_leads(store)
    upgraded = act(store, first, "upgrade_lead")
    package = deepcopy(upgraded["package"])
    kyc = next(d for d in package["documents"] if d["document_id"] == "kyc")
    kyc["text"] += "经营资料已补充更新。"
    store.change_source("seed-01", package, "修订经营资料")
    current = run_leads(store, include=False, pending_upgraded=True)
    closed = act(store, current, "close_lead")
    record = closed["lead_dispositions"][0]["disposition"]
    assert record["source_hash"] == current["source_hash"]
    assert record["origin_source_hash"] == first["source_hash"]
    assert record["origin_snapshot_id"] == first["latest_run"]["snapshot_id"]
    assert record["origin_evidence"] == first["latest_run"]["lead_candidates"][0]["evidence"]
    assert record["evidence"][0]["type"] == "upgraded_focus"
    known = [ref for annotation in current["annotations"] for ref in annotation["evidence"]]
    assert validate_evidence(current["package"], record["evidence"], known_evidence=known)
    assert record["candidate"] == first["latest_run"]["lead_candidates"][0]


def test_existing_legacy_upgrade_can_close_without_fabricating_missing_origin(store):
    before = store.get("seed-01")
    package = deepcopy(before["package"])
    package["review_scope"]["upgraded_leads"] = [{"focus_id": "legacy-focus", "text": "旧版人工升级事项",
                                                "origin_issue_id": "legacy-issue"}]
    package["data_version"] = "2"
    # Simulate a database already written by the old application, not a newly
    # imported package. Public import/source endpoints reject this bypass.
    with store.connect() as db:
        db.execute("BEGIN IMMEDIATE")
        store._write_source_change(db, before, package, "旧版持久数据兼容夹具", "fixture", None)
    current = run_leads(store, include=False, pending_upgraded=True)
    lead = current["lead_dispositions"][0]
    assert lead["status"] == "upgraded" and lead["candidate"]["legacy"]
    closed = store.review("seed-01", action="close_lead", target_id="legacy-issue", reason="对旧版升级事项显式关闭",
        snapshot_id=current["latest_run"]["snapshot_id"], expected_event_id=None)
    assert not closed["package"]["review_scope"]["upgraded_leads"]
    record = closed["lead_dispositions"][0]["disposition"]
    assert record["origin_source_hash"] is None
    assert record["origin_evidence"] == []
    assert record["evidence"][0]["focus_id"] == "legacy-focus"
