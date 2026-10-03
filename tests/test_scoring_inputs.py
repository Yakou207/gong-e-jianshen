"""Raw-input adapters and reference resolution, without model or truth generation."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from aml_qc.scoring_inputs import normalize_issue_predictions, normalize_observations, resolve_evidence


def case():
    return {"case_id": "example", "subject_account_id": "acct", "coverage_start": "2026-01-01", "coverage_end": "2026-02-01",
        "documents": [{"document_id": "doc", "revision": "1", "text": "😀向乙支付一次"}],
        "transactions": [{"transaction_id": "tx", "account_id": "acct", "amount": "1.00"}],
        "materials": [{"material_id": "mat", "revision": "1", "subject": {"account_id": "acct"}, "amount": None}],
        "material_links": [{"link_id": "link", "material_id": "mat", "revision": "1", "transaction_ids": ["tx"], "relation_template": "single_purchase_payment"}],
        "alert": {"revision": "1", "focuses": [{"focus_id": "focus", "text": "解释付款"}]},
        "coverage": [{"coverage_id": "cov", "revision": "1", "status": "full", "fields": ["account_id"]}]}


def document_ref():
    return {"type": "document_span", "document_id": "doc", "revision": "1", "span": [1, 7], "text": "向乙支付一次"}


def engine():
    ref = document_ref()
    return {"case_id": "example", "claim_amendments": [],
        "features": [{"feature_code": "F1", "result": "not_met", "execution_status": "completed", "evidence": []}],
        "claims": [{"claim_id": "model-local-id", "kind": "count", "operator": "exact", "value": 99,
                    "direction": "out", "counterparty_ref": "乙", "source": {k: ref[k] for k in ("document_id", "revision", "span")}, "text": ref["text"]}],
        "claim_results": [{"claim_id": "model-local-id", "result": "contradicted", "execution_status": "completed", "evidence": [ref,
                          {"type": "query_scope", "scope": {"start": "2026-01-01"}, "opaque_metadata": [False, 0]}]}],
        "material_results": [{"link_id": "link", "result": "insufficient", "execution_status": "completed", "evidence": []}],
        "semantic_results": [{"focus_id": "focus", "status": "pending_judgement", "execution_status": "completed", "evidence": []}],
        "issues": [{"issue_id": "i", "type": "claim_error", "target_id": "claim:model-local-id", "evidence": [ref]}]}


def b0():
    ref = document_ref()
    return {"case_id": "example", "run_status": "partial", "parsed_output": {
        "checks": [{"check_id": "local", "label": "count", "status": "completed", "value": "contradicted",
            "object_scope": {"account_id": "acct", "counterparty_tokens": ["乙"]},
            "anchor": {"document_id": "doc", "revision": "1", "span": [1, 7], "quote": "向乙支付一次",
                       "claim": {"operator": "exact", "value": 99, "unit": "次"}}, "evidence": [ref]}],
        "coverage": [{"label": "amount_sum", "status": "no_candidate"}], "unfinished": [],
        "issues": [{"issue_id": "i", "type": "claim_error", "check_ids": ["local"], "evidence": [ref]}]}}


def test_engine_adapter_preserves_wrong_proposition_and_all_evidence_without_fixing_it():
    raw, frozen = engine(), case()
    before = deepcopy(raw)
    rows = normalize_observations(raw, "Fixed", frozen)
    assert [r["observation_id"] for r in rows] == ["/features/0", "/claim_results/0", "/material_results/0", "/semantic_results/0"]
    claim = rows[1]
    assert claim["proposition"]["value"] == 99 and claim["prediction_value"] == "contradicted"
    assert "claim_id" not in claim["proposition"]
    assert claim["evidence"] == raw["claim_results"][0]["evidence"]
    assert claim["matching_anchor"] == raw["claims"][0]["source"]
    claim["evidence"][0]["span"][0] = 42
    assert raw == before


def test_b0_adapter_keeps_unfinished_null_and_does_not_invent_no_candidate():
    raw = b0()
    raw["parsed_output"]["checks"][0].update(status="unfinished", value=None)
    row, = normalize_observations(raw, "B0", case())
    assert row["observation_id"] == "/parsed_output/checks/0"
    assert row["execution_state"] == "unfinished" and row["prediction_value"] is None
    assert row["proposition"] == {"operator": "exact", "value": 99, "unit": "次"}
    assert row["matching_anchor"]["quote"] == "向乙支付一次" and "claim" not in row["matching_anchor"]


def test_issue_views_link_only_actual_local_observations_and_never_add_checks():
    raw = engine()
    raw["issues"].extend([
        {"issue_id": "m", "type": "material_insufficient", "target_id": "materials:link", "evidence": []},
        {"issue_id": "s", "type": "manual_focus", "target_id": "semantic:focus", "evidence": []},
        {"issue_id": "g", "type": "unsupported_explanation", "target_id": "gap:opaque", "evidence": []}])
    rows = normalize_issue_predictions(raw, "Agent")
    assert [r["observations"] for r in rows] == [["/claim_results/0"], ["/material_results/0"], ["/semantic_results/0"], []]
    assert len(normalize_observations(raw, "Agent", case())) == 4
    assert rows[0]["local_check_refs"] == ["claim:model-local-id"]
    direct, = normalize_issue_predictions(b0(), "B0")
    assert direct["prediction_id"] == "/parsed_output/issues/0"
    assert direct["observations"] == ["/parsed_output/checks/0"]


def test_duplicate_content_remains_separate_predictions_but_local_join_ids_must_be_unique():
    raw = engine()
    raw["claim_results"].append(deepcopy(raw["claim_results"][0]))
    rows = normalize_observations(raw, "Fixed", case())
    assert {r["raw_pointer"] for r in rows if r["label"] == "count"} == {"/claim_results/0", "/claim_results/1"}
    raw["claims"].append(deepcopy(raw["claims"][0]))
    with pytest.raises(ValueError, match="duplicate local claim_id"):
        normalize_observations(raw, "Fixed", case())
    direct = b0()
    direct["parsed_output"]["checks"].append(deepcopy(direct["parsed_output"]["checks"][0]))
    with pytest.raises(ValueError, match="duplicate B0"):
        normalize_observations(direct, "B0", case())


@pytest.mark.parametrize("mutate", [lambda r: r.update(claim_results=None), lambda r: r.update(features=["bad"]),
    lambda r: r["claim_results"][0].update(claim_id="absent"), lambda r: r["claim_results"][0].update(evidence="bad"),
    lambda r: r["claim_results"][0].update(result="made_up"), lambda r: r["features"][0].update(feature_code={}),
    lambda r: r["claims"][0].update(source={}), lambda r: r["material_results"][0].update(link_id="absent")])
def test_malformed_raw_structures_raise_instead_of_disappearing(mutate):
    raw = engine(); mutate(raw)
    with pytest.raises(ValueError):
        normalize_observations(raw, "Fixed", case())


@pytest.mark.parametrize("where", ["top", "nested_origin", "deliverable", "case"])
def test_human_interventions_are_not_scored_as_autonomous_output(where):
    raw, frozen = engine(), case()
    if where == "top": raw["claim_amendments"] = [{"amendment_id": "human"}]
    elif where == "nested_origin": raw["claims"][0]["origin"] = "human_reviewed"
    elif where == "deliverable": raw["deliverable"] = {}
    else: frozen["claim_amendments"] = [{"amendment_id": "human"}]
    with pytest.raises(ValueError):
        normalize_observations(raw, "Fixed", frozen)


def test_failed_b0_null_output_is_preserved_as_absence_but_success_null_is_invalid():
    raw = {"case_id": "example", "parsed_output": None, "run_status": "failed"}
    assert normalize_observations(raw, "B0", case()) == normalize_issue_predictions(raw, "B0") == []
    raw["run_status"] = "completed"
    with pytest.raises(ValueError): normalize_observations(raw, "B0", case())


def test_invalid_or_duplicate_issue_local_refs_fail():
    raw = b0(); raw["parsed_output"]["issues"][0]["check_ids"] = ["absent"]
    with pytest.raises(ValueError): normalize_issue_predictions(raw, "B0")
    raw = engine(); raw["issues"].append(deepcopy(raw["issues"][0]))
    with pytest.raises(ValueError): normalize_issue_predictions(raw, "Fixed")


@pytest.mark.parametrize("patch", [{"revision": "2"}, {"span": [0, 6]}, {"span": [-1, 5]},
                                  {"span": [1, 99]}, {"span": [True, 7]}, {"quote": "other"}, {"document_id": []}])
def test_document_reference_requires_exact_revision_unicode_span_and_optional_quote(patch):
    ref = document_ref()
    assert resolve_evidence(ref, case()) is True
    ref.update(patch)
    assert resolve_evidence(ref, case()) is False


@pytest.mark.parametrize("ref", [
    {"type": "transaction", "transaction_id": "tx", "field_paths": ["account_id"]},
    {"type": "transactions", "transaction_ids": ["tx"], "fields": ["amount"]},
    {"type": "material", "material_id": "mat", "revision": "1"},
    {"type": "material_fields", "material_id": "mat", "revision": "1", "field_paths": ["subject.account_id", "amount"]},
    {"type": "alert_focus", "focus_id": "focus", "revision": "1", "quote": "解释付款"},
    {"type": "coverage", "coverage_id": "cov", "revision": "1", "status": "full"},
])
def test_simple_source_references_resolve_without_business_verification(ref):
    assert resolve_evidence(ref, case()) is True


@pytest.mark.parametrize("ref", [
    {"type": "transactions", "transaction_ids": []}, {"type": "transactions", "transaction_ids": ["tx", "tx"]},
    {"type": "transactions", "transaction_ids": ["tx"], "fields": ["not_present"]},
    {"type": "transactions", "transaction_ids": ["tx"], "transaction_set_version": "sha256:old"},
    {"type": "material", "material_id": "mat", "revision": "2"},
    {"type": "material_fields", "material_id": "mat", "revision": "1", "field_paths": ["absent"]},
    {"type": "alert_focus", "focus_id": "absent", "revision": "1"},
    {"type": "coverage", "coverage_id": "cov", "revision": "1", "status": "partial"},
])
def test_missing_stale_or_invented_evidence_is_invalid(ref):
    assert resolve_evidence(ref, case()) is False


def test_unsupported_complex_evidence_stays_unknown_and_bad_shapes_raise():
    for kind in ("query_scope", "tool_result", "material_set", "upgraded_focus", "entity_mapping"):
        assert resolve_evidence({"type": kind, "claimed_result": "supported"}, case()) is None
    with pytest.raises(ValueError): resolve_evidence([], case())
    frozen = case(); frozen["transactions"].append(deepcopy(frozen["transactions"][0]))
    with pytest.raises(ValueError): resolve_evidence({"type": "transactions", "transaction_ids": ["tx"]}, frozen)


def test_unverified_extra_binding_is_unknown_and_boolean_revision_is_not_integer_one():
    assert resolve_evidence({**document_ref(), "unverified_source_hash": "invented"}, case()) is None
    assert resolve_evidence({"type": "transactions", "transaction_ids": ["tx"], "content_hash": "invented"}, case()) is None
    frozen = case(); frozen["documents"][0]["revision"] = 1
    assert resolve_evidence({**document_ref(), "revision": True}, frozen) is False
    assert resolve_evidence({**document_ref(), "revision": 1}, frozen) is True


def test_optional_material_and_alert_content_hashes_cannot_be_silently_ignored():
    frozen = case()
    for kind, row, identity in (("material", frozen["materials"][0], "material_id"),
                                ("alert_focus", frozen["alert"]["focuses"][0], "focus_id")):
        digest = hashlib.sha256(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        ref = {"type": kind, identity: row[identity], "revision": "1", "content_hash": digest}
        assert resolve_evidence(ref, frozen) is True
        ref["content_hash"] = "bad"
        assert resolve_evidence(ref, frozen) is False


def test_actual_local_workflow_raw_results_use_same_adapter_without_private_fixtures():
    from aml_qc.ingest import load_case
    from aml_qc.workflow import run_review
    frozen = load_case(Path(__file__).resolve().parents[1] / "data/synthetic/seed-01.json")
    raw = run_review(frozen, provider="local")
    observations = normalize_observations(raw, "Fixed", frozen)
    assert len(observations) == sum(len(raw[k]) for k in ("features", "claim_results", "material_results", "semantic_results"))
    pointers = {r["observation_id"] for r in observations}
    assert all(set(issue["observations"]) <= pointers for issue in normalize_issue_predictions(raw, "Fixed"))
