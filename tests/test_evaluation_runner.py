"""Hand-written runner mechanism fixtures, not independent human AML gold labels.

Every model is a local test double. These tests do not read credentials or call
a provider, and the fictional reviewer declarations establish no real identity.
"""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import fcntl
import hashlib
import json
import os
from pathlib import Path

import pytest

from aml_qc import llm
from aml_qc.baseline import raw_case_input
from aml_qc.depgraph import digest
from aml_qc.llm import GENERATION, generation_request
from scripts.validate_evaluation import EXPOSURES, validate_manifest


ROOT = Path(__file__).resolve().parents[1]
MODEL = "offline-runner-fixture"
IMPLEMENTATIONS = ["aml_qc/" + name + ".py" for name in (
    "core", "ingest", "schema", "contracts", "llm", "depgraph", "workflow", "claim_edits",
    "leads", "baseline", "scoring_inputs", "evaluation_budget")] + [
    "scripts/run_evaluation.py", "scripts/score_evaluation.py", "scripts/validate_evaluation.py"]


def write_json(path, value):
    raw = json.dumps(value, ensure_ascii=False, indent=2).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def file_ref(base, path):
    return {"path": os.path.relpath(path, base), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def json_ref(base, name, value):
    path = base / name
    write_json(path, value)
    return file_ref(base, path)


@pytest.fixture(autouse=True)
def forbid_credentials_and_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Offline runner tests must not read credentials or invoke a paid API")
    monkeypatch.setattr(llm, "settings", forbidden)
    monkeypatch.setattr(llm.httpx, "post", forbidden)


@pytest.mark.parametrize("stamp", [
    "2026-10-03T10:00:00+08:00",  # Weekend morning and afternoon are off-peak.
    "2026-10-04T16:00:00+08:00",
    "2026-09-29T08:56:59+08:00",  # The 180-second guard stays before 09:00.
    "2026-09-29T12:00:00+08:00",
    "2026-09-29T13:56:59+08:00",
    "2026-09-29T18:00:00+08:00",
    "2026-09-28T23:59:30+08:00",  # Crossing midnight does not change this tariff.
])
def test_live_pricing_accepts_unambiguous_off_peak_intervals(monkeypatch, stamp):
    from scripts import run_evaluation as runner
    current = datetime.fromisoformat(stamp)
    monkeypatch.setattr(runner, "now", lambda: current.astimezone(timezone.utc))
    pricing = {"period": "off_peak", "valid_until": (current + timedelta(days=1)).isoformat()}
    runner.verify_pricing_window(pricing, live=True)


@pytest.mark.parametrize("stamp", [
    "2026-10-01T10:00:00+08:00",  # A holiday must not be inferred to be peak from its weekday.
    "2026-09-29T15:00:00+08:00",  # Ordinary weekdays also require verified calendar evidence.
])
@pytest.mark.parametrize("period", ["peak", "off_peak"])
def test_live_weekday_peak_hours_are_unknown_without_verified_holiday_calendar(monkeypatch, stamp, period):
    from scripts import run_evaluation as runner
    current = datetime.fromisoformat(stamp)
    monkeypatch.setattr(runner, "now", lambda: current.astimezone(timezone.utc))
    pricing = {"period": period, "valid_until": (current + timedelta(days=1)).isoformat()}
    with pytest.raises(ValueError, match="weekday_peak_requires_verified_holiday_calendar"):
        runner.verify_pricing_window(pricing, live=True)


@pytest.mark.parametrize("stamp", [
    "2026-09-29T08:59:00+08:00",
    "2026-09-29T08:57:30+08:00",
    "2026-09-29T13:59:30+08:00",
    "2026-10-01T08:59:30+08:00",
])
def test_live_call_cannot_cross_into_unknown_weekday_tariff(monkeypatch, stamp):
    from scripts import run_evaluation as runner
    current = datetime.fromisoformat(stamp)
    monkeypatch.setattr(runner, "now", lambda: current.astimezone(timezone.utc))
    pricing = {"period": "off_peak", "valid_until": (current + timedelta(days=1)).isoformat()}
    with pytest.raises(ValueError, match="pricing_transition_within_request_timeout"):
        runner.verify_pricing_window(pricing, live=True)


def test_weekend_cannot_use_frozen_peak_price(monkeypatch):
    from scripts import run_evaluation as runner
    current = datetime.fromisoformat("2026-10-03T10:00:00+08:00")
    monkeypatch.setattr(runner, "now", lambda: current)
    pricing = {"period": "peak", "valid_until": (current + timedelta(days=1)).isoformat()}
    with pytest.raises(ValueError, match="frozen_pricing_period_is_not_current"):
        runner.verify_pricing_window(pricing, live=True)


def test_offline_verification_does_not_require_holiday_calendar(monkeypatch):
    from scripts import run_evaluation as runner
    current = datetime.fromisoformat("2026-10-01T10:00:00+08:00")
    monkeypatch.setattr(runner, "now", lambda: current)
    pricing = {"period": "peak", "valid_until": (current + timedelta(days=1)).isoformat()}
    runner.verify_pricing_window(pricing, live=False)


@pytest.mark.parametrize("live", [False, True])
def test_request_must_finish_before_frozen_pricing_expiry(monkeypatch, live):
    from scripts import run_evaluation as runner
    current = datetime.fromisoformat("2026-10-03T10:00:00+08:00")
    monkeypatch.setattr(runner, "now", lambda: current)
    pricing = {"period": "off_peak", "valid_until": (current + timedelta(seconds=60)).isoformat()}
    with pytest.raises(ValueError, match="pricing_expired_or_too_near_expiry"):
        runner.verify_pricing_window(pricing, live=live)


@pytest.fixture
def experiment(tmp_path):
    stamp = datetime.now(timezone.utc) - timedelta(minutes=1)
    people = [{"person_id": "fictional-fixture-" + name, "signed_at": stamp.isoformat(),
               "exposure": dict.fromkeys(EXPOSURES, False)} for name in ("A", "B")]
    case = {"case_id": "runner-mechanism", "case_family": "handwritten-mechanism-family",
        "task_mode": "annotation_only", "subject_account_id": "fixture-account", "data_version": "1",
        "coverage_start": "2026-09-01T00:00:00+08:00", "coverage_end": "2026-09-09T00:00:00+08:00",
        "currency": "CNY", "timezone": "Asia/Shanghai", "schema_version": "S1.0",
        "review_scope": {"target_labels": ["F1"]}, "transactions": [], "counterparties": [],
        "coverage": [{"coverage_id": "fixture-coverage", "revision": "1", "account_id": "fixture-account",
            "source": "transactions", "start": "2026-09-01T00:00:00+08:00", "end": "2026-09-09T00:00:00+08:00",
            "status": "unknown", "fields": ["account_id", "direction", "timestamp", "amount"]}],
        "materials": [], "material_links": [], "entity_mappings": [],
        "documents": [{"document_id": "narrative", "revision": "1", "source": "synthetic",
                       "text": "仅用于离线持久化机制测试，无业务真值。"}]}
    cref = json_ref(tmp_path, "case.json", case)
    reference = {"case_id": case["case_id"], "case_sha256": cref["sha256"],
        "visible_case_sha256": hashlib.sha256((json.dumps(raw_case_input(case), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()).hexdigest(),
        "reviewers": people,
        "check_units": [{"reference_check_id": "f1", "label": "F1", "object_scope": {"account_id": "fixture-account"},
            "applicability": "applicable", "adjudication_status": "adjudicated", "reference_value": "undeterminable",
            "reason": "Hand-written persistence oracle; not a real independent AML judgment.",
            "evidence_sets": [[{"type": "coverage", "coverage_id": "fixture-coverage", "revision": "1"}]],
            "issue_expectations": [{"type": "insufficient_coverage", "truth": "positive"}]}]}
    placeholder = json_ref(tmp_path, "mechanism-only.json", {"purpose": "hand-written test specification; no actual quality claim"})
    method = {"model": MODEL, "generation": deepcopy(GENERATION),
        "runner": file_ref(tmp_path, ROOT / "aml_qc/workflow.py"),
        "budget": {"max_calls": 6, "max_output_tokens": GENERATION["max_output_tokens"], "total_token_budget": 10_000_000, "currency_limit": "10"}}
    pricing = {"currency": "CNY", "effective_at": stamp.isoformat(),
        "valid_until": (stamp + timedelta(days=1)).isoformat(), "source_url": "https://example.invalid/offline-mechanism-price",
        "unit_tokens": 1_000_000, "rates": {"input_cache_hit": "0.1", "input_cache_miss": "1", "output": "2"}}
    manifest = {"contract_version": "evaluation-freeze-1", "experiment_id": "offline-runner-mechanism-only",
        "execution_directory": "run-output",
        "status": "frozen", "frozen_at": stamp.isoformat(), "approved_by": [people[0]],
        "cases": [{"case_id": case["case_id"], "family_id": case["case_family"], "split": "test",
                   "required_check_ids": ["f1"], **cref}],
        "references": [{"case_id": case["case_id"], **json_ref(tmp_path, "reference.json", reference)}],
        "reference_counts": {"adjudicated": 1, "unresolved": 0}, "repeat_count": 1,
        "retry_policy": {"max_retries": 0, "retryable_errors": []}, "currency": "CNY", "total_currency_budget": "30",
        "pricing": json_ref(tmp_path, "pricing.json", pricing),
        "methods": {name: deepcopy(method) for name in ("B0", "Fixed", "Agent")},
        "artifacts": {"spec": placeholder, "generator": placeholder,
            "schema": file_ref(tmp_path, ROOT / "config/schema/S1.0.json"),
            "dependency_lock": file_ref(tmp_path, ROOT / "uv.lock"),
            "scorer": file_ref(tmp_path, ROOT / "scripts/score_evaluation.py"),
            "tools": [file_ref(tmp_path, ROOT / name) for name in IMPLEMENTATIONS],
            "prompts": [file_ref(tmp_path, path) for path in sorted((ROOT / "config/prompts/v3.1").glob("*.txt"))]
                       + [file_ref(tmp_path, ROOT / "config/prompts/b0-v1/direct.txt")],
            "family_grouping": json_ref(tmp_path, "grouping.json", {"reviewed_by": [people[0]], "cases": [
                {"case_id": case["case_id"], "economic_group": case["case_family"], "split": "test",
                 "reason": "Single hand-written hypothetical mechanism fixture."}]})}}
    manifest["methods"]["B0"]["runner"] = file_ref(tmp_path, ROOT / "aml_qc/baseline.py")
    manifest["methods"]["B0"]["budget"]["max_calls"] = 1
    path = tmp_path / "manifest.json"
    write_json(path, manifest)
    frozen = validate_manifest(path)
    assert frozen["frozen_valid"], frozen["issues"]
    return {"path": path, "manifest": manifest, "pricing": pricing, "case": case,
            "output": tmp_path / "run-output", "plan": frozen["planned_runs"]}


class OfflineModel:
    def __init__(self, method, calls, *, unknown_usage=False, malformed=False, unfinished=False):
        self.model, self.calls, self.method = MODEL, [], method
        self.observed_calls = calls
        self.unknown_usage, self.malformed, self.unfinished = unknown_usage, malformed, unfinished

    def complete(self, messages, tools=None, stage=None):
        self.observed_calls.append(self.method)
        if self.method == "B0":
            output = {"checks": [{"check_id": "local-f1", "label": "F1",
                "status": "unfinished" if self.unfinished else "completed",
                "value": None if self.unfinished else "undeterminable", "object_scope": {}, "anchor": {},
                "reason": "Hand-written mechanism answer, not model quality evidence.", "evidence": []}],
                "coverage": [{"label": "F1", "status": "unfinished" if self.unfinished else "completed", "reason": "Fixture only."}],
                "issues": [], "unfinished": []}
        else:
            output = {"claims": [], "unresolved": []}
        message = {"role": "assistant", "content": "not-json" if self.malformed else json.dumps(output)}
        request = generation_request(self.model, deepcopy(messages), deepcopy(tools), stage=stage)
        self.calls.append({"request": request, "request_hash": digest(request), "status": "completed",
            "model_returned": MODEL, "finish_reason": "stop", "response": deepcopy(message),
            "usage": None if self.unknown_usage else {"prompt_tokens": 20, "prompt_cache_hit_tokens": 0,
                "prompt_cache_miss_tokens": 20, "completion_tokens": 10, "total_tokens": 30}})
        return message


def model_factory(factories, calls, **options):
    def create(plan, method):
        factories.append(plan["method"])
        return OfflineModel(plan["method"], calls, **options)
    return create


def run(experiment, *, execute=True, factory=None):
    from scripts.run_evaluation import run_evaluation
    return run_evaluation(experiment["path"], experiment["output"], execute=execute, model_factory=factory)


def read_index(experiment):
    return json.loads((experiment["output"] / "runs.json").read_text())


def assert_blocked_without_calls(experiment, report, factories, calls):
    assert report["errors"]
    assert factories == calls == []
    assert not experiment["output"].exists()


def test_preflight_retains_full_plan_without_factory_credentials_or_output_writes(experiment):
    factories, calls = [], []
    report = run(experiment, execute=False, factory=model_factory(factories, calls))
    assert not report["errors"], report
    assert {p["planned_run_id"] for p in report["planned_runs"]} == {p["planned_run_id"] for p in experiment["plan"]}
    assert {p["method"] for p in report["planned_runs"]} == {"B0", "Fixed", "Agent"}
    assert factories == calls == [] and not experiment["output"].exists()


@pytest.mark.parametrize("binding", ["default_differs", "embedded_differs", "embedded_matches"])
def test_effective_case_schema_must_match_frozen_content_before_dispatch(experiment, binding):
    from aml_qc.schema import validate_schema
    base, manifest = experiment["path"].parent, experiment["manifest"]
    alternate = json.loads((ROOT / "config/schema/S1.0.json").read_text())
    assert alternate["features"]["F1"]["minimum_days"] == 3
    alternate["features"]["F1"]["minimum_days"] = 4
    assert alternate["schema_version"] == experiment["case"]["schema_version"]
    assert validate_schema(alternate) == []
    if binding != "embedded_differs":
        manifest["artifacts"]["schema"] = json_ref(base, "alternate-schema.json", alternate)
    if binding != "default_differs":
        experiment["case"]["schema"] = deepcopy(alternate)
        manifest["cases"][0].update(json_ref(base, "case.json", experiment["case"]))
        reference = json.loads((base / manifest["references"][0]["path"]).read_text())
        reference["case_sha256"] = manifest["cases"][0]["sha256"]
        reference["visible_case_sha256"] = hashlib.sha256((json.dumps(raw_case_input(experiment["case"]), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()).hexdigest()
        manifest["references"][0].update(json_ref(base, "reference.json", reference))
    write_json(experiment["path"], manifest)
    # File hashes and the schema version alone cannot bind the rules executed.
    frozen = validate_manifest(experiment["path"])
    assert frozen["frozen_valid"], frozen["issues"]
    factories, calls = [], []
    preflight = run(experiment, execute=False, factory=model_factory(factories, calls))
    assert factories == calls == [] and not experiment["output"].exists()
    if binding != "embedded_matches":
        assert preflight["status"] == "blocked", preflight
        assert "case_schema_does_not_match_frozen_schema" in preflight["errors"]
        dispatched = run(experiment, factory=model_factory(factories, calls))
        assert dispatched["status"] == "blocked", dispatched
        assert_blocked_without_calls(experiment, dispatched, factories, calls)
        return
    assert preflight["status"] == "planned" and not preflight["errors"], preflight
    completed = run(experiment, factory=model_factory(factories, calls))
    assert completed["status"] == "completed" and not completed["errors"], completed
    assert factories == calls == ["B0", "Fixed", "Agent"]
    methods = {plan["planned_run_id"]: plan["method"] for plan in frozen["planned_runs"]}
    for row in read_index(experiment)["runs"]:
        raw = json.loads((experiment["output"] / row["raw_result"]["path"]).read_text())
        effective = (raw["input_projection"]["schema"] if methods[row["planned_run_id"]] == "B0"
                     else raw["snapshot"]["sources"]["source:schema"]["value"])
        assert effective == alternate


def test_b0_projection_must_preserve_frozen_custom_template_before_dispatch(experiment):
    from aml_qc.baseline import raw_case_input
    from aml_qc.schema import validate_schema
    base, manifest = experiment["path"].parent, experiment["manifest"]
    schema = json.loads((ROOT / "config/schema/S1.0.json").read_text())
    schema["material_templates"]["custom_purchase"] = deepcopy(schema["material_templates"]["single_purchase_payment"])
    assert validate_schema(schema) == []
    experiment["case"]["schema"] = deepcopy(schema)
    manifest["artifacts"]["schema"] = json_ref(base, "custom-schema.json", schema)
    manifest["cases"][0].update(json_ref(base, "case.json", experiment["case"]))
    reference = json.loads((base / manifest["references"][0]["path"]).read_text())
    reference["case_sha256"] = manifest["cases"][0]["sha256"]
    reference["visible_case_sha256"] = hashlib.sha256((json.dumps(raw_case_input(experiment["case"]), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()).hexdigest()
    manifest["references"][0].update(json_ref(base, "reference.json", reference))
    write_json(experiment["path"], manifest)
    frozen = validate_manifest(experiment["path"])
    assert frozen["frozen_valid"], frozen["issues"]
    assert experiment["case"]["schema"] == schema
    assert "custom_purchase" not in raw_case_input(experiment["case"])["schema"]["material_templates"]
    factories, calls = [], []
    for execute in (False, True):
        report = run(experiment, execute=execute, factory=model_factory(factories, calls))
        assert report["status"] == "blocked", report
        assert "b0_projected_schema_does_not_match_frozen_schema" in report["errors"]
        assert_blocked_without_calls(experiment, report, factories, calls)


@pytest.mark.parametrize("mutation", ["draft", "retry", "generation", "calls", "artifact", "prompt", "expired"])
def test_invalid_freeze_or_execution_contract_fails_before_model_construction(experiment, mutation):
    manifest = experiment["manifest"]
    if mutation == "draft": manifest["status"] = "draft"
    elif mutation == "retry": manifest["retry_policy"]["max_retries"] = 1
    elif mutation == "generation":
        for method in manifest["methods"].values(): method["generation"]["temperature"] = 0.5
    elif mutation == "calls": manifest["methods"]["B0"]["budget"]["max_calls"] = 2
    elif mutation == "artifact": manifest["artifacts"]["tools"] = [r for r in manifest["artifacts"]["tools"] if not r["path"].endswith("llm.py")]
    elif mutation == "prompt": manifest["artifacts"]["prompts"].pop()
    else:
        experiment["pricing"]["valid_until"] = "2020-01-01T00:00:00+00:00"
        manifest["pricing"] = json_ref(experiment["path"].parent, "pricing.json", experiment["pricing"])
    write_json(experiment["path"], manifest)
    factories, calls = [], []
    report = run(experiment, factory=model_factory(factories, calls))
    assert_blocked_without_calls(experiment, report, factories, calls)


def test_serial_outputs_bind_original_plan_and_second_invocation_never_repeats_calls(experiment):
    factories, calls = [], []
    first = run(experiment, factory=model_factory(factories, calls))
    assert not first["errors"], first
    assert factories == calls == ["B0", "Fixed", "Agent"]
    plan = json.loads((experiment["output"] / "plan.json").read_text())
    assert plan["contract_version"] == "evaluation-execution-plan-1" and plan["simulation_only"] is True
    assert plan["planned_runs"] == experiment["plan"]
    index = read_index(experiment)
    assert index["contract_version"] == "evaluation-runs-1" and len(index["runs"]) == 3
    assert index["manifest_sha256"] == hashlib.sha256(experiment["path"].read_bytes()).hexdigest()
    snapshots = {}
    by_id = {p["planned_run_id"]: p for p in experiment["plan"]}
    for row in index["runs"]:
        raw_path = experiment["output"] / row["raw_result"]["path"]
        data = raw_path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == row["raw_result"]["sha256"]
        raw = json.loads(data)
        assert raw["evaluation_execution"] == {"planned_run_id": row["planned_run_id"],
            "manifest_sha256": index["manifest_sha256"], "case_sha256": row["case_sha256"],
            "method_config_sha256": row["method_config_sha256"]}
        assert row["method_config_sha256"] == digest(experiment["manifest"]["methods"][by_id[row["planned_run_id"]]["method"]])
        assert (raw_path.parent / "started.json").is_file()
        snapshots[raw_path] = data
    again = run(experiment, factory=model_factory(factories, calls))
    assert not again["errors"], again
    assert factories == calls == ["B0", "Fixed", "Agent"]
    assert all(path.read_bytes() == data for path, data in snapshots.items())


def test_committed_raw_recovers_missing_index_without_model_calls(experiment):
    factories, calls = [], []
    assert not run(experiment, factory=model_factory(factories, calls))["errors"]
    original = read_index(experiment)
    write_json(experiment["output"] / "runs.json", {**original, "runs": []})
    factories.clear(); calls.clear()
    recovered = run(experiment, factory=model_factory(factories, calls))
    assert not recovered["errors"], recovered
    assert factories == calls == []
    assert read_index(experiment) == original


def test_started_without_raw_is_not_retried_and_becomes_indexed_unknown_attempt(experiment):
    factories, calls = [], []
    def crash_after_permanent_start(plan, method):
        factories.append(plan["method"])
        assert (experiment["output"] / "run" / plan["planned_run_id"] / "started.json").is_file()
        raise KeyboardInterrupt("simulated process interruption before outcome persistence")
    try:
        run(experiment, factory=crash_after_permanent_start)
    except KeyboardInterrupt:
        pass
    assert factories == ["B0"]
    factories.clear()
    recovered = run(experiment, factory=model_factory(factories, calls))
    assert recovered["status"] == "stopped"
    assert factories == calls == []
    index = read_index(experiment)
    assert len(index["runs"]) == 1
    assert index["runs"][0]["planned_run_id"] == experiment["plan"][0]["planned_run_id"]
    failed = json.loads((experiment["output"] / index["runs"][0]["raw_result"]["path"]).read_text())
    assert failed["runner_failure"] and failed["run_status"] == "failed"
    from scripts.score_evaluation import score_evaluation
    score = score_evaluation(experiment["path"], experiment["output"] / "runs.json")
    assert len(score["runs"]) == 3
    assert score["runs"][0]["cost"]["total_cost"] is None
    assert sum(r["run_status"] == "not_run" for r in score["runs"]) == 2


def test_unknown_usage_stops_remaining_plan_and_stays_unknown_in_saved_scoring(experiment):
    factories, calls = [], []
    result = run(experiment, factory=model_factory(factories, calls, unknown_usage=True))
    assert result["status"] == "stopped"
    assert factories == calls == ["B0"]
    assert len(read_index(experiment)["runs"]) == 1
    from scripts.score_evaluation import score_evaluation
    scored = score_evaluation(experiment["path"], experiment["output"] / "runs.json")
    assert len(scored["runs"]) == 3
    assert scored["runs"][0]["cost"]["total_cost"] is None
    assert sum(r["run_status"] == "not_run" for r in scored["runs"]) == 2
    factories.clear(); calls.clear()
    run(experiment, factory=model_factory(factories, calls))
    assert factories == calls == []


@pytest.mark.parametrize("options,expected", [({"malformed": True}, "failed"), ({"unfinished": True}, "partial")])
def test_b0_failure_and_unfinished_raw_outputs_are_saved_without_repair(experiment, options, expected):
    factories, calls = [], []
    run(experiment, factory=model_factory(factories, calls, **options))
    row = read_index(experiment)["runs"][0]
    raw = json.loads((experiment["output"] / row["raw_result"]["path"]).read_text())
    assert raw["run_status"] == expected
    if options.get("malformed"):
        assert raw["raw_response"]["content"] == "not-json" and raw["parsed_output"] is None
    else:
        assert raw["parsed_output"]["checks"][0]["value"] is None
    assert calls.count("B0") == 1


def test_kernel_lock_prevents_a_second_writer_from_constructing_a_model(experiment):
    experiment["output"].mkdir()
    factories, calls = [], []
    with (experiment["output"] / ".runner.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        report = run(experiment, factory=model_factory(factories, calls))
        assert report["errors"]
        assert factories == calls == []
    assert not (experiment["output"] / "runs.json").exists()


def test_manifest_drift_does_not_reuse_prior_directory_or_invoke_models(experiment):
    factories, calls = [], []
    assert not run(experiment, factory=model_factory(factories, calls))["errors"]
    previous = (experiment["output"] / "runs.json").read_bytes()
    experiment["manifest"]["experiment_id"] += "-changed"
    write_json(experiment["path"], experiment["manifest"])
    factories.clear(); calls.clear()
    report = run(experiment, factory=model_factory(factories, calls))
    assert report["errors"] and factories == calls == []
    assert (experiment["output"] / "runs.json").read_bytes() == previous


def test_truncated_journal_cannot_turn_an_existing_attempt_into_a_retry(experiment):
    factories, calls = [], []
    assert not run(experiment, factory=model_factory(factories, calls))["errors"]
    with (experiment["output"] / "journal.jsonl").open("ab") as handle:
        handle.write(b'{"event":"call_started"')
    factories.clear(); calls.clear()
    report = run(experiment, factory=model_factory(factories, calls))
    assert report["errors"] and factories == calls == []


def test_same_manifest_cannot_be_dispatched_into_another_output_directory(experiment):
    experiment["output"] = experiment["path"].parent / "unbound-second-copy"
    factories, calls = [], []
    report = run(experiment, factory=model_factory(factories, calls))
    assert_blocked_without_calls(experiment, report, factories, calls)


def test_recovery_rejects_raw_with_wrong_plan_binding_instead_of_relabeling_it(experiment):
    factories, calls = [], []
    assert not run(experiment, factory=model_factory(factories, calls))["errors"]
    original_index = (experiment["output"] / "runs.json").read_bytes()
    row = read_index(experiment)["runs"][0]
    path = experiment["output"] / row["raw_result"]["path"]
    raw = json.loads(path.read_text())
    raw["evaluation_execution"]["planned_run_id"] = "not-the-planned-run"
    write_json(path, raw)
    factories.clear(); calls.clear()
    report = run(experiment, factory=model_factory(factories, calls))
    assert report["errors"] and factories == calls == []
    assert (experiment["output"] / "runs.json").read_bytes() == original_index


@pytest.mark.parametrize("keep_index", [True, False])
def test_changed_raw_bytes_cannot_be_blessed_with_a_new_hash_during_recovery(experiment, keep_index):
    factories, calls = [], []
    assert not run(experiment, factory=model_factory(factories, calls))["errors"]
    index = read_index(experiment)
    path = experiment["output"] / index["runs"][0]["raw_result"]["path"]
    raw = json.loads(path.read_text())
    raw["raw_response"]["content"] = "tampered-after-commit"
    write_json(path, raw)
    if not keep_index:
        write_json(experiment["output"] / "runs.json", {**index, "runs": []})
    previous = (experiment["output"] / "runs.json").read_bytes()
    factories.clear(); calls.clear()
    report = run(experiment, factory=model_factory(factories, calls))
    assert report["errors"] and factories == calls == []
    assert (experiment["output"] / "runs.json").read_bytes() == previous


def test_expired_pricing_does_not_prevent_read_only_recovery_of_committed_results(experiment, monkeypatch):
    factories, calls = [], []
    assert not run(experiment, factory=model_factory(factories, calls))["errors"]
    index = read_index(experiment)
    write_json(experiment["output"] / "runs.json", {**index, "runs": []})
    from scripts import run_evaluation as runner
    after_expiry = datetime.fromisoformat(experiment["pricing"]["valid_until"]) + timedelta(days=1)
    monkeypatch.setattr(runner, "now", lambda: after_expiry)
    factories.clear(); calls.clear()
    report = run(experiment, factory=model_factory(factories, calls))
    assert not report["errors"], report
    assert factories == calls == [] and read_index(experiment) == index


def test_interrupted_dispatch_is_already_indexed_before_recovery_and_never_repeated(experiment):
    factories, calls = [], []
    class InterruptedModel(OfflineModel):
        def complete(self, messages, tools=None, stage=None):
            calls.append(self.method)
            raise KeyboardInterrupt("simulated loss after request may have been sent")
    def factory(plan, method):
        factories.append(plan["method"])
        return InterruptedModel(plan["method"], calls)
    with pytest.raises(KeyboardInterrupt):
        run(experiment, factory=factory)
    assert factories == calls == ["B0"]
    index = read_index(experiment)
    assert len(index["runs"]) == 1
    from scripts.score_evaluation import score_evaluation
    before = score_evaluation(experiment["path"], experiment["output"] / "runs.json")
    assert before["runs"][0]["cost"]["total_cost"] is None
    assert sum(r["run_status"] == "not_run" for r in before["runs"]) == 2
    factories.clear(); calls.clear()
    recovered = run(experiment, factory=model_factory(factories, calls))
    assert recovered["status"] == "stopped" and factories == calls == []
    assert len(read_index(experiment)["runs"]) == 1


@pytest.mark.parametrize("when", ["before_dispatch", "after_dispatch"])
def test_manifest_changed_in_running_process_cannot_dispatch_under_original_binding(experiment, when):
    factories, calls = [], []
    original_sha = hashlib.sha256(experiment["path"].read_bytes()).hexdigest()
    def change_manifest():
        # The changed manifest is still a valid freeze; invalid-JSON or budget
        # validation must not accidentally be what prevents the next dispatch.
        experiment["manifest"]["total_currency_budget"] = "31"
        write_json(experiment["path"], experiment["manifest"])
        assert validate_manifest(experiment["path"])["frozen_valid"]
    class ChangingModel(OfflineModel):
        def complete(self, messages, tools=None, stage=None):
            result = super().complete(messages, tools, stage=stage)
            change_manifest()
            return result
    def factory(plan, method):
        factories.append(plan["method"])
        if when == "before_dispatch":
            change_manifest()
        return ChangingModel(plan["method"], calls)
    report = run(experiment, factory=factory)
    assert report["status"] in {"blocked", "stopped"}
    assert factories == ["B0"]
    assert calls == ([] if when == "before_dispatch" else ["B0"])
    index = read_index(experiment)
    assert index["manifest_sha256"] == original_sha and len(index["runs"]) == 1
    assert json.loads((experiment["output"] / "plan.json").read_text())["manifest_sha256"] == original_sha


def rewrite_journal(experiment, transform):
    path = experiment["output"] / "journal.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows = transform(rows)
    # Maintain ordinary framing/sequence validity so the test specifically
    # requires cross-checking the call ledger against the committed raw data.
    for number, row in enumerate(rows, 1):
        row["sequence"] = number
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


@pytest.mark.parametrize("mutation", ["remove_all_calls", "remove_finished", "zero_finished_usage"])
def test_recovery_cross_checks_call_journal_against_raw_and_cannot_erase_cost(experiment, mutation):
    factories, calls = [], []
    assert not run(experiment, factory=model_factory(factories, calls))["errors"]
    original_index = (experiment["output"] / "runs.json").read_bytes()
    def mutate(rows):
        if mutation == "remove_all_calls":
            return [r for r in rows if not r["event"].startswith("call_")]
        target = next(r for r in rows if r["event"] == "call_finished")
        if mutation == "remove_finished":
            return [r for r in rows if r is not target]
        target["record"]["usage"] = {"prompt_tokens": 0, "prompt_cache_hit_tokens": 0,
            "prompt_cache_miss_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        target["settlement"] = {"status": "settled", "cost": "0", "tokens": 0}
        target["record"]["budget_settlement"] = deepcopy(target["settlement"])
        return rows
    rewrite_journal(experiment, mutate)
    factories.clear(); calls.clear()
    report = run(experiment, factory=model_factory(factories, calls))
    assert report["status"] == "blocked" and report["errors"]
    assert factories == calls == []
    assert (experiment["output"] / "runs.json").read_bytes() == original_index
    assert "budget" not in report or Decimal(report["budget"]["spent"]) > 0
    from scripts.score_evaluation import score_evaluation
    saved = score_evaluation(experiment["path"], experiment["output"] / "runs.json")
    assert all(Decimal(r["cost"]["total_cost"]) > 0 for r in saved["runs"])


@pytest.mark.parametrize("field,value", [("max_calls", 2), ("currency_limit", "99"), ("total_token_budget", 99_000_000)])
def test_recovery_reservation_budget_must_equal_frozen_method_budget(experiment, field, value):
    factories, calls = [], []
    assert not run(experiment, factory=model_factory(factories, calls))["errors"]
    original_index = (experiment["output"] / "runs.json").read_bytes()
    def mutate(rows):
        reserved = next(r for r in rows if r["event"] == "call_reserved")
        reserved["budget"][field] = value
        return rows
    rewrite_journal(experiment, mutate)
    factories.clear(); calls.clear()
    report = run(experiment, factory=model_factory(factories, calls))
    assert report["status"] == "blocked" and report["errors"]
    assert factories == calls == []
    assert (experiment["output"] / "runs.json").read_bytes() == original_index


def test_known_not_sent_budget_blocks_recover_normally_without_new_dispatch_or_unknown_cost(experiment):
    for method in experiment["manifest"]["methods"].values():
        method["budget"]["total_token_budget"] = 1
    write_json(experiment["path"], experiment["manifest"])
    factories, calls = [], []
    first = run(experiment, factory=model_factory(factories, calls))
    assert not first["errors"], first
    assert first["status"] == "completed" and factories == ["B0", "Fixed", "Agent"] and calls == []
    index = read_index(experiment)
    assert len(index["runs"]) == 3
    from scripts.score_evaluation import score_evaluation
    saved = score_evaluation(experiment["path"], experiment["output"] / "runs.json")
    assert len(saved["runs"]) == 3 and all(r["cost"]["total_cost"] == "0" for r in saved["runs"])
    assert all(r["cost"]["unknown_usage_calls"] == 0 for r in saved["runs"])
    factories.clear()
    restored = run(experiment, factory=model_factory(factories, calls))
    assert not restored["errors"] and restored["status"] == "completed", restored
    assert factories == calls == [] and read_index(experiment) == index
    assert Decimal(restored["budget"]["spent"]) == Decimal(restored["budget"]["held"]) == 0


@pytest.mark.parametrize("interrupted_method", ["B0", "Fixed"])
def test_repeated_interrupted_recovery_preserves_reservation_and_rejects_deleted_call_history(experiment, interrupted_method):
    factories, calls = [], []
    class InterruptedModel(OfflineModel):
        def complete(self, messages, tools=None, stage=None):
            calls.append(self.method)
            raise KeyboardInterrupt("possibly sent; no durable provider response")
    def factory(plan, method):
        factories.append(plan["method"])
        implementation = InterruptedModel if plan["method"] == interrupted_method else OfflineModel
        return implementation(plan["method"], calls)
    with pytest.raises(KeyboardInterrupt):
        run(experiment, factory=factory)
    expected = ["B0"] if interrupted_method == "B0" else ["B0", "Fixed"]
    assert factories == calls == expected
    factories.clear(); calls.clear()
    recovered = run(experiment, factory=model_factory(factories, calls))
    assert recovered["status"] == "stopped" and not recovered["errors"]
    assert factories == calls == []
    held = recovered["budget"]
    assert Decimal(held["held"]) > 0 and len(held["pending_call_ids"]) == 1
    assert (Decimal(held["spent"]) > 0) is (interrupted_method == "Fixed")
    index = read_index(experiment)
    assert len(index["runs"]) == len(expected)
    failure_path = experiment["output"] / index["runs"][-1]["raw_result"]["path"]
    failure_bytes = failure_path.read_bytes()
    failure = json.loads(failure_bytes)
    assert failure_path.name == "interrupted.json" and failure["runner_failure"]
    records = failure["call_records" if interrupted_method == "B0" else "model_requests"]
    assert len(records) == 1 and records[0]["dispatch_status"] == "possibly_sent"
    assert records[0]["usage"] is None
    again = run(experiment, factory=model_factory(factories, calls))
    assert again["status"] == "stopped" and not again["errors"]
    assert again["budget"] == held and factories == calls == []
    assert read_index(experiment) == index and failure_path.read_bytes() == failure_bytes
    prior_index = (experiment["output"] / "runs.json").read_bytes()
    rewrite_journal(experiment, lambda rows: [r for r in rows if not r["event"].startswith("call_")])
    refused = run(experiment, factory=model_factory(factories, calls))
    assert refused["status"] == "blocked" and refused["errors"]
    assert factories == calls == []
    assert (experiment["output"] / "runs.json").read_bytes() == prior_index
    assert failure_path.read_bytes() == failure_bytes
