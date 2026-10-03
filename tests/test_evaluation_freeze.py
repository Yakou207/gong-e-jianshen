"""Hand-authored validator fixtures, not independent AML quality references."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.validate_evaluation import EXPOSURES, validate_manifest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/validate_evaluation.py"
STAMP = "2026-10-03T14:00:00+08:00"


def write_ref(root, name, value):
    path = root / name
    raw = json.dumps(value, ensure_ascii=False, indent=2).encode()
    path.write_bytes(raw)
    return {"path": name, "sha256": hashlib.sha256(raw).hexdigest()}


@pytest.fixture
def frozen(tmp_path):
    ref = write_ref(tmp_path, "source.json", {"purpose": "placeholder implementation source for file validation only"})
    people = [{"person_id": person, "signed_at": STAMP, "exposure": dict.fromkeys(EXPOSURES, False)}
              for person in ("reviewer-A", "reviewer-B")]
    cases, references = [], []
    for number in (1, 2):
        cid = f"case-{number}"
        case = {"case_id": cid, "family_id": f"family-{number}", "split": "test",
                **write_ref(tmp_path, f"{cid}.json", {"case_id": cid, "task_mode": "annotation_only",
                    "review_scope": {"target_labels": ["count", "material_relation"]}, "material_links": []}),
                "required_check_ids": ["known", "uncertain"]}
        cases.append(case)
        units = [{"reference_check_id": "known", "label": "count", "object_scope": {"account": "synthetic-account"},
                  "applicability": "applicable", "adjudication_status": "adjudicated", "reference_value": False,
                  "reason": "Hand-written negative reference for validator tests.",
                  "evidence_sets": [[{"document_id": "narrative", "span": [0, 3]}],
                                    [{"transaction_id": "one"}, {"transaction_id": "two"}]]},
                 {"reference_check_id": "uncertain", "label": "material_relation", "object_scope": {"material": "missing"},
                  "applicability": "applicable", "adjudication_status": "unresolved", "reference_value": None,
                  "reason": "Reviewers could not decide from available material.", "evidence_sets": []}]
        record = {"case_id": cid, "case_sha256": case["sha256"], "reviewers": people, "check_units": units}
        references.append({"case_id": cid, **write_ref(tmp_path, f"{cid}-reference.json", record)})
    method = {"model": "frozen-model-id", "generation": {"temperature": 0, "thinking": "disabled"},
              "runner": ref, "budget": {"max_calls": 6, "max_output_tokens": 4096,
                                        "total_token_budget": 40000, "currency_limit": "1.00"}}
    pricing = {"currency": "CNY", "effective_at": STAMP, "source_url": "https://example.invalid/fixture-price",
               "unit_tokens": 1000000, "rates": {"input_cache_hit": "0.1", "input_cache_miss": "1", "output": "2"}}
    manifest = {"contract_version": "evaluation-freeze-1", "experiment_id": "handwritten-validator-fixture",
                "status": "frozen", "frozen_at": STAMP, "approved_by": [people[0]], "cases": cases,
                "references": references, "reference_counts": {"adjudicated": 2, "unresolved": 2},
                "artifacts": {k: ref for k in ("spec", "schema", "generator", "dependency_lock", "scorer")},
                "methods": {name: deepcopy(method) for name in ("B0", "Fixed", "Agent")}, "repeat_count": 2,
                "retry_policy": {"max_retries": 0, "retryable_errors": []},
                "pricing": write_ref(tmp_path, "pricing.json", pricing), "currency": "CNY", "total_currency_budget": "12.00"}
    manifest["artifacts"].update(prompts=[ref], tools=[ref])
    manifest["artifacts"]["schema"] = write_ref(tmp_path, "schema.json", {"labels": {"count": {}, "material_relation": {}}})
    manifest["artifacts"]["family_grouping"] = write_ref(tmp_path, "grouping.json", {"reviewed_by": [people[0]],
        "cases": [{"case_id": c["case_id"], "economic_group": c["family_id"], "split": c["split"],
                   "reason": "Separate hypothetical sources in this hand-written validator fixture."} for c in cases]})
    manifest["methods"]["B0"]["budget"]["max_calls"] = 1
    return tmp_path / "manifest.json", manifest


def check(frozen):
    path, manifest = frozen
    path.write_text(json.dumps(manifest, ensure_ascii=False))
    return validate_manifest(path)


def codes(report):
    return {issue["code"] for issue in report["issues"]}


def change_reference(frozen, mutate, index=0):
    path, manifest = frozen
    ref = manifest["references"][index]
    value = json.loads((path.parent / ref["path"]).read_text())
    mutate(value)
    ref.update(write_ref(path.parent, ref["path"], value))


def test_complete_freeze_expands_all_runs_and_preserves_unresolved_and_alternative_evidence(frozen):
    report = check(frozen)
    assert report["frozen_valid"] and not report["issues"]
    assert report["reference_counts_observed"] == {"adjudicated": 2, "unresolved": 2}
    expected = {(f"case-{case}", method, repeat) for case in (1, 2)
                for method in ("B0", "Fixed", "Agent") for repeat in (1, 2)}
    assert {(r["case_id"], r["method"], r["repeat"]) for r in report["planned_runs"]} == expected
    assert report["planned_run_count"] == len(expected) == 12
    assert len({r["planned_run_id"] for r in report["planned_runs"]}) == 12
    assert all(r["status"] == "not_run" for r in report["planned_runs"])
    assert "B0 has no built-in runner" in report["limitations"][1]
    assert "do not authenticate" in report["limitations"][0]


@pytest.mark.parametrize("target", ["case", "reference", "artifact", "pricing"])
def test_changed_exact_file_bytes_block_freeze(frozen, target):
    path, manifest = frozen
    ref = {"case": manifest["cases"][0], "reference": manifest["references"][0],
           "artifact": manifest["artifacts"]["scorer"], "pricing": manifest["pricing"]}[target]
    file = path.parent / ref["path"]
    file.write_bytes(file.read_bytes() + b"\n")
    report = check(frozen)
    assert not report["frozen_valid"] and "file_hash_mismatch" in codes(report)


def test_one_family_cannot_cross_split(frozen):
    cases = frozen[1]["cases"]
    cases[1].update(family_id=cases[0]["family_id"], split="development")
    assert "family_crosses_split" in codes(check(frozen))


def test_family_names_cannot_override_separately_reviewed_grouping(frozen):
    frozen[1]["cases"][1].update(family_id="renamed-to-hide-relationship", split="development")
    assert "family_grouping_mismatch" in codes(check(frozen))


def use_product_seeds(frozen, *, shrink_targets):
    path, manifest = frozen
    manifest["artifacts"]["schema"] = write_ref(path.parent, "schema.json", json.loads(
        (SCRIPT.parent.parent / "config/schema/S1.0.json").read_text()))
    grouping = json.loads((path.parent / "grouping.json").read_text())
    for index, number in enumerate((1, 3)):
        package = json.loads((SCRIPT.parent.parent / f"data/synthetic/seed-{number:02d}.json").read_text())
        cid = package["case_id"]
        labels = ["F1"] if shrink_targets else package["review_scope"]["target_labels"]
        case = manifest["cases"][index]
        case.update(case_id=cid, family_id="legacy-development", split="development", required_check_ids=labels,
                    **write_ref(path.parent, case["path"], package))
        grouping["cases"][index].update(case_id=cid, economic_group="legacy-development", split="development")
        ref = manifest["references"][index]
        record = json.loads((path.parent / ref["path"]).read_text())
        record.update(case_id=cid, case_sha256=case["sha256"], check_units=[
            {"reference_check_id": label, "label": label, "object_scope": {"account": "merchant-001"},
             "applicability": "applicable", "adjudication_status": "unresolved", "reference_value": None,
             "reason": "Unresolved hand-written fixture; not a real human evaluation.", "evidence_sets": []} for label in labels])
        ref.update(case_id=cid, **write_ref(path.parent, ref["path"], record))
    manifest["artifacts"]["family_grouping"] = write_ref(path.parent, "grouping.json", grouping)
    manifest["reference_counts"] = {"adjudicated": 0, "unresolved": sum(len(c["required_check_ids"]) for c in manifest["cases"])}


def test_actual_product_target_scope_cannot_be_shrunk_to_only_f1(frozen):
    use_product_seeds(frozen, shrink_targets=True)
    report = check(frozen)
    assert not report["frozen_valid"]
    assert {"reference_target_coverage", "reference_focus_coverage", "reference_material_link_coverage"} <= codes(report)


def test_all_labels_alone_do_not_cover_each_alert_and_material_object(frozen):
    use_product_seeds(frozen, shrink_targets=False)
    report = check(frozen)
    assert "reference_target_coverage" not in codes(report)
    assert {"reference_focus_coverage", "reference_material_link_coverage"} <= codes(report)
    for index in (0, 1):
        def add_objects(record):
            for unit in record["check_units"]:
                if unit["label"] == "alert_response":
                    unit["object_scope"]["focus_id"] = "focus-1"
                elif unit["label"] == "material_relation":
                    unit["object_scope"]["material_link_id"] = "purchase-link"
        change_reference(frozen, add_objects, index)
    assert check(frozen)["frozen_valid"]


def test_known_legacy_seeds_cannot_be_declared_independent_test_families(frozen):
    use_product_seeds(frozen, shrink_targets=False)
    manifest = frozen[1]
    manifest["cases"][1].update(family_id="claimed-new-family", split="test")
    path = frozen[0].parent / "grouping.json"
    grouping = json.loads(path.read_text())
    grouping["cases"][1].update(economic_group="claimed-new-family", split="test")
    manifest["artifacts"]["family_grouping"] = write_ref(path.parent, path.name, grouping)
    assert {"known_seed_not_development", "known_seed_family_split"} <= codes(check(frozen))


def test_total_budget_and_currency_cover_every_planned_run(frozen):
    report = check(frozen)
    assert report["planned_currency_ceiling"] == "12.00" and not report["execution_budget_enforced"]
    frozen[1]["total_currency_budget"] = "11.99"
    assert "planned_budget_exceeds_total" in codes(check(frozen))
    frozen[1]["currency"] = "USD"
    assert "currency_mismatch" in codes(check(frozen))


@pytest.mark.parametrize("change,expected", [("empty", "reference_coverage_empty"),
                                           ("case_missing", "reference_case_coverage"),
                                           ("unit_missing", "reference_check_coverage"),
                                           ("unit_empty", "reference_units_empty")])
def test_reference_coverage_cannot_be_empty_or_drop_cases_or_units(frozen, change, expected):
    if change == "empty":
        frozen[1]["references"] = []
    elif change == "case_missing":
        frozen[1]["references"].pop()
    else:
        change_reference(frozen, lambda r: r.update(check_units=[] if change == "unit_empty" else r["check_units"][:1]))
    assert expected in codes(check(frozen))


def test_unresolved_count_cannot_be_hidden_or_promoted_to_answer(frozen):
    frozen[1]["reference_counts"]["unresolved"] = 0
    assert "reference_count_mismatch" in codes(check(frozen))
    frozen[1]["reference_counts"]["unresolved"] = 2
    change_reference(frozen, lambda r: r["check_units"][1].update(reference_value="supported"))
    assert "unresolved_has_value" in codes(check(frozen))


def test_fixed_agent_budget_must_match_while_single_read_b0_can_differ(frozen):
    assert check(frozen)["frozen_valid"]
    frozen[1]["methods"]["Agent"]["budget"]["max_calls"] = 7
    assert "method_budget_differs" in codes(check(frozen))


def test_missing_b0_runner_is_explicit_and_does_not_remove_b0_plan(frozen):
    del frozen[1]["methods"]["B0"]["runner"]
    report = check(frozen)
    assert any(i["location"] == "methods.B0.runner" for i in report["issues"])
    assert len([r for r in report["planned_runs"] if r["method"] == "B0"]) == 4
    del frozen[1]["methods"]["B0"]
    assert "required_methods" in codes(check(frozen))


@pytest.mark.parametrize("change,expected", [("case", "duplicate_case"), ("unit", "duplicate_reference_unit"),
                                           ("plan_duplicate", "duplicate_planned_run"), ("plan_missing", "planned_run_coverage")])
def test_duplicate_and_incomplete_declarations_fail(frozen, change, expected):
    if change == "case":
        frozen[1]["cases"].append(deepcopy(frozen[1]["cases"][0]))
    elif change == "unit":
        change_reference(frozen, lambda r: r["check_units"].append(deepcopy(r["check_units"][0])))
    else:
        planned = check(frozen)["planned_runs"]
        frozen[1]["planned_runs"] = planned + [planned[0]] if change == "plan_duplicate" else planned[:-1]
    assert expected in codes(check(frozen))


@pytest.mark.parametrize("exposure", list(EXPOSURES))
def test_declared_answer_exposure_blocks_blind_freeze(frozen, exposure):
    change_reference(frozen, lambda r: r["reviewers"][1]["exposure"].update({exposure: True}))
    report = check(frozen)
    assert "reference_exposure" in codes(report) and not report["blind_reference_declared_clear"]
    assert exposure in report["personnel_declarations"][1]["declared_exposures"]


def test_same_person_under_case_and_spacing_changes_is_not_two_reviewers(frozen):
    change_reference(frozen, lambda r: r["reviewers"][1].update(person_id=" REVIEWER-a "))
    assert "duplicate_person" in codes(check(frozen))


@pytest.mark.parametrize("symlink", [False, True])
def test_env_paths_are_not_read_even_through_alias(frozen, monkeypatch, symlink):
    path, manifest = frozen
    secret = path.parent / ".env"
    secret.write_text("not a real secret; this file must never be read")
    alias = path.parent / "innocent.txt"
    if symlink:
        alias.symlink_to(secret)
    manifest["artifacts"]["scorer"] = {"path": alias.name if symlink else secret.name, "sha256": "0" * 64}
    original = Path.read_bytes
    def guarded(file):
        assert file.resolve() != secret, "attempted to read credential file"
        return original(file)
    monkeypatch.setattr(Path, "read_bytes", guarded)
    assert "credential_file_forbidden" in codes(check(frozen))


def test_absolute_paths_rejected_but_explicit_parent_relative_paths_supported(frozen):
    path, manifest = frozen
    manifest["artifacts"]["scorer"] = {**manifest["artifacts"]["scorer"], "path": str(path.parent / "source.json")}
    assert "absolute_path_forbidden" in codes(check(frozen))
    manifest["artifacts"]["scorer"]["path"] = "../" + path.parent.name + "/source.json"
    assert check(frozen)["frozen_valid"]


def test_cli_draft_lists_gaps_and_formal_mode_fails_without_writing_sources(tmp_path):
    path = tmp_path / "old-template.json"
    path.write_text('{"status":"not_frozen","case_files":[],"reference_files":[]}')
    before = path.read_bytes()
    draft = subprocess.run([sys.executable, str(SCRIPT), str(path)], capture_output=True, text=True)
    formal = subprocess.run([sys.executable, str(SCRIPT), str(path), "--require-frozen"], capture_output=True, text=True)
    assert draft.returncode == 0 and formal.returncode == 1
    report = json.loads(draft.stdout)
    assert {"contract_version", "case_coverage_empty", "reference_coverage_empty", "required_methods"} <= codes(report)
    assert not report["frozen_valid"] and report["planned_runs"] == []
    assert path.read_bytes() == before and list(tmp_path.iterdir()) == [path]


def test_cli_complete_draft_is_not_a_frozen_manifest(frozen):
    path, manifest = frozen
    manifest["status"] = "draft"
    report = check(frozen)
    assert report["ready_to_freeze"] and not report["frozen_valid"]
    result = subprocess.run([sys.executable, str(SCRIPT), str(path), "--require-frozen"], capture_output=True, text=True)
    assert result.returncode == 1


@pytest.mark.parametrize("field,value", [("status", {}), ("cases", 7), ("repeat_count", True),
                                        ("methods", []), ("references", {})])
def test_malformed_top_level_fields_report_errors_instead_of_crashing(frozen, field, value):
    frozen[1][field] = value
    assert not check(frozen)["frozen_valid"]


def test_duplicate_json_keys_and_nonfinite_values_are_rejected(tmp_path):
    path = tmp_path / "bad.json"
    for text in ('{"status":"draft","status":"frozen"}', '{"repeat_count":NaN}'):
        path.write_text(text)
        assert "manifest_unreadable" in codes(validate_manifest(path))
