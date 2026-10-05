"""Two-person claim edits over real workflow snapshots, using offline model fixtures."""
from copy import deepcopy
from pathlib import Path

import pytest

from test_store import bound_review

from aml_qc import llm
from aml_qc.depgraph import business_result
from aml_qc.ingest import load_case
from aml_qc.llm import FrozenModel
from aml_qc.store import Store
from aml_qc.workflow import run_review
from test_model_safety import MODEL, ScriptedModel, json_message


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    monkeypatch.setattr(llm, "settings", lambda: {"DEEPSEEK_API_KEY": "fixture",
        "DEEPSEEK_MODEL": MODEL, "DEEPSEEK_BASE_URL": "https://model.invalid"})
    monkeypatch.setattr(llm.httpx, "post", lambda *a, **kw: pytest.fail("No live model calls allowed"))


@pytest.fixture
def store(tmp_path):
    package = load_case(Path(__file__).resolve().parents[1] / "data/synthetic/seed-01.json")
    package["task_mode"], package["alert"] = "annotation_only", None
    package["review_scope"]["target_labels"] = ["count"]
    next(d for d in package["documents"] if d["document_id"] == "narrative")["text"] = "检查期间仅向乙公司支付一次。"
    result = Store(tmp_path / "claims.sqlite3")
    result.create(package)
    return result


def run_raw(store, value=99, *, extracted=True, strategy="full", previous=None, products=None):
    state = store.get("seed-01")
    claims = [{"kind": "count", "operator": "exact", "value": value, "direction": "out",
               "counterparty_ref": "乙公司", "quote": "检查期间仅向乙公司支付一次"}] if extracted else []
    model = ScriptedModel([json_message({"claims": claims, "unresolved": []})])
    if strategy == "incremental":
        model = FrozenModel(products, MODEL)
    result = run_review(state["package"], provider="frozen", model=model, strategy=strategy, previous=previous)
    if products is not None and strategy == "full":
        products.update(model.request_products)
    return store.save_run("seed-01", state["source_hash"], result)


def propose(store, state, *, operation="replace", value=1, **kwargs):
    target = state["latest_run"]["machine_claims"][0] if state["latest_run"]["machine_claims"] else None
    payload = {"kind": "count", "operator": "exact", "value": value, "direction": "out",
               "counterparty_ref": "乙公司", "quote": "检查期间仅向乙公司支付一次"}
    request = {"operation": operation, "reason": "核对原文，机器抽取需更正", "actor": "author",
               "snapshot_id": state["latest_run"]["snapshot_id"]}
    if operation in {"replace", "retire"}:
        request["target_claim_id"] = target["claim_id"]
    if operation in {"replace", "add"}:
        request["proposed_claim"] = payload
    request.update(kwargs)
    return store.propose_claim("seed-01", **request)


def review(store, state, *, action="approve", actor="reviewer", fidelity="faithful", **kwargs):
    proposal = state["claim_proposals"][-1]
    request = {"action": action, "reason": "逐项对照原句、对象与期间后作出裁决", "actor": actor,
        "snapshot_id": state["latest_run"]["snapshot_id"], "expected_event_id": proposal["expected_event_id"],
        "fidelity": fidelity}
    request.update(kwargs)
    return store.review_claim_proposal("seed-01", proposal["proposal_id"], **request)


def confirm_current(store, state):
    for annotation in state["annotations"]:
        store.review("seed-01", action="confirm", target_id=annotation["annotation_id"], reason="核对当前核验结果",
            snapshot_id=state["latest_run"]["snapshot_id"], expected_event_id=annotation["review"]["event_id"])
    return bound_review(store, "seed-01", action="confirm", target_id="task", reason="限定范围检查已逐项完成")


def test_proposal_cannot_clear_machine_issue_until_review_and_recalculation(store):
    original = run_raw(store)
    original_run = deepcopy(original["latest_run"])
    pending = propose(store, original)
    proposal = pending["claim_proposals"][0]
    assert proposal["status"] == "pending" and proposal["current"]
    assert proposal["preview_verification"]["result"] == "supported"
    assert pending["source_hash"] == original["source_hash"]
    assert pending["latest_run"]["claim_results"][0]["result"] == "contradicted"
    assert any(item["kind"] == "claim_proposal_pending" for item in pending["open_items"])
    assert not pending["can_pass"]
    approved = review(store, pending)
    assert approved["stale"] and not approved["can_pass"]
    amendment = approved["package"]["claim_amendments"][0]
    assert amendment["proposer"] == "author" and amendment["reviewer"] == "reviewer"
    assert amendment["original_claim"] == original_run["machine_claims"][0]
    assert amendment["original_claim"]["value"] == 99 and amendment["proposed_claim"]["value"] == 1
    assert any(e["action"] == "needs_review" for e in approved["review_events"])
    refreshed = run_raw(store)
    assert refreshed["latest_run"]["machine_claims"][0]["value"] == 99
    assert refreshed["latest_run"]["claims"][0]["value"] == 1
    assert refreshed["latest_run"]["claim_results"][0]["result"] == "supported"
    assert refreshed["claim_amendments"][0]["status"] == "applied"
    passed = confirm_current(store, refreshed)
    assert passed["review_status"] == "本次质检范围内通过"
    exported = store.export("seed-01")
    assert exported["deliverable"]["claim_proposals"][0]["status"] == "approved"
    delivered = exported["deliverable"]["annotations"][0]
    assert delivered["candidate_origin"] == "human_reviewed"
    assert delivered["origin"] == delivered["review"]["origin"] == "human_confirmation"
    assert delivered["candidate_value"] == "supported"
    assert delivered["machine_candidate_value"] is None and delivered["machine_object"] is None
    assert exported["historical_runs"][0]["claims"][0]["value"] == 99
    assert original_run["snapshot_id"] == exported["historical_runs"][0]["snapshot_id"]


def test_current_pending_proposal_revokes_pass_even_when_preview_is_supported(store):
    state = confirm_current(store, run_raw(store, value=1))
    assert state["can_pass"]
    pending = propose(store, state)
    assert not pending["can_pass"]
    assert not store.export("seed-01")["deliverable"]["passed"]


@pytest.mark.parametrize("action,actor", [("approve", "author"), ("approve", " AUTHOR "), ("reject", "author"), ("withdraw", "reviewer")])
def test_same_person_cannot_approve_or_reject_and_others_cannot_withdraw(store, action, actor):
    pending = propose(store, run_raw(store))
    with pytest.raises(ValueError, match="另一人员"):
        review(store, pending, action=action, actor=actor, fidelity="faithful" if action == "approve" else None)
    assert store.get("seed-01")["claim_proposals"][0]["status"] == "pending"
    assert not store.get("seed-01")["package"].get("claim_amendments")


@pytest.mark.parametrize("action,actor,status", [("reject", "reviewer", "rejected"), ("withdraw", "author", "withdrawn")])
def test_rejection_and_withdrawal_leave_original_claim_unchanged(store, action, actor, status):
    original = run_raw(store)
    pending = propose(store, original)
    closed = review(store, pending, action=action, actor=actor, fidelity=None)
    assert closed["claim_proposals"][0]["status"] == status
    assert closed["source_hash"] == original["source_hash"]
    assert closed["latest_run"]["claims"] == original["latest_run"]["claims"]
    assert any(i["kind"] == "claim_error" for i in closed["open_items"])
    assert not any(i["kind"] == "claim_proposal_pending" for i in closed["open_items"])


def test_fidelity_and_explicit_approval_are_separate_from_numerical_preview(store):
    pending = propose(store, run_raw(store), value=2)
    assert pending["claim_proposals"][0]["fidelity_assessment"]["blocking_errors"]
    with pytest.raises(ValueError, match="忠实性.*阻断"):
        review(store, pending)
    with pytest.raises(ValueError, match="明确选择"):
        review(store, pending, fidelity=None)
    assert not store.get("seed-01")["package"].get("claim_amendments")


def test_omitted_claim_can_be_added_and_recalculated_from_same_text(store):
    missing = run_raw(store, extracted=False)
    assert missing["annotations"][0]["kind"] == "scope_review"
    approved = review(store, propose(store, missing, operation="add"))
    assert approved["stale"]
    refreshed = run_raw(store, extracted=False)
    assert refreshed["latest_run"]["machine_claims"] == []
    assert len(refreshed["latest_run"]["claims"]) == 1
    assert refreshed["latest_run"]["claims"][0]["origin"] == "human_reviewed"
    assert refreshed["latest_run"]["claim_results"][0]["result"] == "supported"
    assert refreshed["annotations"][0]["kind"] == "claim"


@pytest.mark.parametrize("mutation", ["source", "run"])
def test_obsolete_proposal_is_readonly_history_until_author_withdraws(store, mutation):
    pending = propose(store, run_raw(store))
    if mutation == "source":
        package = deepcopy(pending["package"])
        package["profile"]["business_type"] += "追加业务背景"
        store.change_source("seed-01", package, "当前资料已改变")
    current = run_raw(store)
    assert current["claim_proposals"][0]["status"] == "stale"
    assert current["claim_proposals"][0]["allowed_actions"] == ["withdraw"]
    assert not any(i["kind"] == "claim_proposal_pending" for i in current["open_items"])
    with pytest.raises(ValueError, match="过期提议"):
        review(store, current)
    withdrawn = review(store, current, action="withdraw", actor="author", fidelity=None)
    assert withdrawn["claim_proposals"][0]["status"] == "withdrawn"


@pytest.mark.parametrize("field", ["snapshot_id", "expected_event_id"])
def test_review_rejects_outdated_client_anchor(store, field):
    pending = propose(store, run_raw(store))
    with pytest.raises(ValueError, match="快照|其他人员"):
        review(store, pending, **{field: "old-value"})
    assert not store.get("seed-01")["package"].get("claim_amendments")


@pytest.mark.parametrize("mutation", ["source", "run", "event"])
def test_approval_transaction_rechecks_source_run_and_event_tail(store, monkeypatch, mutation):
    pending = propose(store, run_raw(store))
    other = Store(store.path)
    if mutation == "source":
        package = deepcopy(pending["package"])
        package["profile"]["business_type"] += "并发更新"
        other.change_source("seed-01", package, "并发来源修订")
    elif mutation == "run":
        run_raw(other)
    else:
        review(other, pending, action="reject", fidelity=None)
    before = other.get("seed-01")
    monkeypatch.setattr(store, "get", lambda _: deepcopy(pending))
    with pytest.raises(ValueError, match="已改变"):
        review(store, pending)
    after = other.get("seed-01")
    assert before["source_hash"] == after["source_hash"]
    assert before["review_events"] == after["review_events"]
    assert not after["package"].get("claim_amendments")


def test_approved_amendment_can_be_revoked_only_by_new_independent_review(store):
    approved = review(store, propose(store, run_raw(store)))
    amendment_id = approved["package"]["claim_amendments"][0]["amendment_id"]
    current = run_raw(store)
    pending = propose(store, current, operation="revoke", supersedes_amendment_id=amendment_id)
    assert pending["latest_run"]["claims"][0]["value"] == 1
    revoked = review(store, pending, fidelity=None)
    assert revoked["stale"]
    refreshed = run_raw(store)
    assert [a["status"] for a in refreshed["claim_amendments"]] == ["superseded", "revoked"]
    assert refreshed["latest_run"]["claims"][0]["value"] == 99
    assert any(i["kind"] == "claim_error" for i in refreshed["open_items"])
    with pytest.raises(ValueError, match="尚未被替代"):
        propose(store, refreshed, operation="revoke", supersedes_amendment_id=amendment_id)


def test_plain_import_and_source_change_cannot_inject_or_erase_approved_amendments(store, tmp_path):
    approved = review(store, propose(store, run_raw(store)))
    with pytest.raises(ValueError, match="导入不可携带"):
        Store(tmp_path / "forged.sqlite3").create(approved["package"])
    package = deepcopy(approved["package"])
    package["claim_amendments"] = []
    with pytest.raises(ValueError, match="普通来源修订不能修改"):
        store.change_source("seed-01", package, "禁止删除修订记录")


@pytest.mark.parametrize("fidelity", ["not_a_claim", "duplicate"])
def test_retirement_cannot_erase_a_real_nonduplicate_assertion(store, fidelity):
    pending = propose(store, run_raw(store, value=1), operation="retire")
    with pytest.raises(ValueError, match="不支持|另一条"):
        review(store, pending, fidelity=fidelity)
    assert store.get("seed-01")["latest_run"]["claims"][0]["value"] == 1


@pytest.mark.parametrize("machine_now_correct", [False, True])
def test_superseding_active_human_claim_targets_current_machine_ancestor(store, machine_now_correct):
    first = review(store, propose(store, run_raw(store)))
    first_id = first["package"]["claim_amendments"][0]["amendment_id"]
    current = run_raw(store, value=1 if machine_now_correct else 99)
    human_id = current["latest_run"]["claims"][0]["claim_id"]
    with pytest.raises(ValueError, match="明确替代"):
        propose(store, current, target_claim_id=human_id)
    pending = propose(store, current, target_claim_id=human_id, supersedes_amendment_id=first_id)
    assert pending["claim_proposals"][-1]["target_claim_id"] == current["latest_run"]["machine_claims"][0]["claim_id"]
    review(store, pending)
    refreshed = run_raw(store, value=1 if machine_now_correct else 99)
    assert [a["status"] for a in refreshed["claim_amendments"]] == ["superseded", "applied"]
    assert len(refreshed["latest_run"]["claims"]) == 1
    assert refreshed["latest_run"]["claims"][0]["value"] == 1
    assert refreshed["latest_run"]["claims"][0]["claim_id"] != human_id


def test_unresolved_amendment_cannot_be_rejected_or_closed_as_an_ordinary_issue(store):
    approved = review(store, propose(store, run_raw(store)))
    package = deepcopy(approved["package"])
    next(d for d in package["documents"] if d["document_id"] == "narrative")["text"] += "补充事项仍待核对。"
    store.change_source("seed-01", package, "文本新增使忠实性审核需重新核对")
    current = run_raw(store, value=1)
    amendment = current["claim_amendments"][0]
    assert amendment["status"] == "needs_review"
    issue = next(i for i in current["latest_run"]["issues"] if i["type"] == "manual_claim_review")
    with pytest.raises(ValueError, match="不能通过否决"):
        bound_review(store, "seed-01", action="reject", target_id=issue["issue_id"], reason="不能跳过重新审核")
    with pytest.raises(ValueError, match="不可手动关闭"):
        bound_review(store, "seed-01", action="close_item", target_id=issue["issue_id"], reason="不能手动清掉修订失效", resolution="addressed")
    with pytest.raises(ValueError, match="不能通过"):
        confirm_current(store, current)
    pending = propose(store, store.get("seed-01"), supersedes_amendment_id=amendment["amendment_id"])
    review(store, pending)
    refreshed = run_raw(store, value=1)
    assert [a["status"] for a in refreshed["claim_amendments"]] == ["superseded", "applied"]
    assert not any(i["kind"] == "manual_claim_review" for i in refreshed["open_items"])


@pytest.mark.parametrize("operation", ["replace", "add", "revoke", "supersede"])
def test_approved_amendment_change_matches_independent_full_and_incremental(store, operation):
    before = run_raw(store, extracted=operation != "add")
    if operation in {"revoke", "supersede"}:
        approved = review(store, propose(store, before))
        amendment_id = approved["package"]["claim_amendments"][0]["amendment_id"]
        before = run_raw(store)
        pending = propose(store, before, operation="revoke" if operation == "revoke" else "replace",
                          supersedes_amendment_id=amendment_id)
    else:
        pending = propose(store, before, operation=operation)
    original_snapshot = deepcopy(before["latest_run"]["snapshot"])
    review(store, pending, fidelity=None if operation == "revoke" else "faithful")
    products = {}
    full = run_raw(store, extracted=operation != "add", products=products)["latest_run"]
    incremental = run_raw(store, extracted=operation != "add", strategy="incremental",
                          previous=original_snapshot, products=products)["latest_run"]
    assert business_result(full) == business_result(incremental)
    assert before["latest_run"]["snapshot"] == original_snapshot
    assert "source:claim_amendments" in incremental["stats"]["changed_sources"]
    assert full["claims"][0]["value"] == (99 if operation == "revoke" else 1)
    assert full["claim_results"][0]["result"] == ("contradicted" if operation == "revoke" else "supported")


def test_claim_amendment_bad_import_shape_reports_validation_instead_of_crashing(store, tmp_path):
    package = deepcopy(store.get("seed-01")["package"])
    package["claim_amendments"] = [{"amendment_id": []}]
    with pytest.raises(ValueError, match="人工陈述"):
        Store(tmp_path / "invalid.sqlite3").create(package)


def test_http_proposal_and_review_contracts_reject_extra_fields_and_preserve_anchors(store, monkeypatch):
    from fastapi.testclient import TestClient
    from test_api import api
    # API tests share a locally guarded import; this test never starts a model.
    original = run_raw(store)
    app = api.create_app(store.path, seed=False)
    with TestClient(app) as client:
        body = {"operation": "replace", "target_claim_id": original["latest_run"]["machine_claims"][0]["claim_id"],
            "proposed_claim": {"kind": "count", "operator": "exact", "value": 1, "direction": "out",
                               "counterparty_ref": "乙公司", "quote": "检查期间仅向乙公司支付一次"},
            "actor": "author", "reason": "HTTP提议夹具", "snapshot_id": original["latest_run"]["snapshot_id"]}
        assert client.post("/api/cases/seed-01/claim-proposals", json={**body, "preapproved": True}).status_code == 422
        response = client.post("/api/cases/seed-01/claim-proposals", json=body)
        assert response.status_code == 200, response.text
        proposal = response.json()["claim_proposals"][-1]
        endpoint = f"/api/cases/seed-01/claim-proposals/{proposal['proposal_id']}/review"
        decision = {"action": "approve", "actor": "reviewer", "reason": "HTTP审核夹具", "fidelity": "faithful",
            "snapshot_id": original["latest_run"]["snapshot_id"], "expected_event_id": proposal["expected_event_id"]}
        assert client.post(endpoint, json={**decision, "proposed_claim": {}}).status_code == 422
        assert client.post(endpoint, json={**decision, "actor": "author"}).status_code == 409
        response = client.post(endpoint, json=decision)
        assert response.status_code == 200, response.text
        assert response.json()["stale"]
        assert client.get("/api/cases/seed-01/export").json()["claim_proposals"][-1]["status"] == "approved"
