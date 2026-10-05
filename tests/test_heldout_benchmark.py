import json
import subprocess
import sys
from pathlib import Path

from aml_qc import core
from aml_qc.ingest import validate_case
from scripts import score_heldout

ROOT = Path(__file__).resolve().parents[1]


def generate(tmp_path, name="bench"):
    out = tmp_path / name
    subprocess.run([sys.executable, str(ROOT / "scripts/generate_heldout.py"), str(out)], check=True, capture_output=True)
    return out


def test_generator_is_deterministic_and_keeps_truth_out_of_packages(tmp_path):
    a, b = generate(tmp_path, "a"), generate(tmp_path, "b")
    manifest = json.loads((a / "manifest.json").read_text())
    assert len(manifest["cases"]) == 32 and len({c["family"] for c in manifest["cases"]}) == 8
    for case in manifest["cases"]:
        body = (a / "cases" / f"{case['case_id']}.json").read_text()
        assert body == (b / "cases" / f"{case['case_id']}.json").read_text()
        for leaked in ("expected_issue_types", "injected", "\"profile\": \"", "truth"):
            assert leaked not in body.replace('"profile": {', '')


def test_independent_truth_agrees_with_core_and_cases_validate(tmp_path):
    out = generate(tmp_path)
    schema = json.loads((ROOT / "config/schema/S1.0.json").read_text())
    for path in sorted((out / "cases").glob("*.json")):
        case = json.loads(path.read_text())
        truth = json.loads((out / "truth" / path.name).read_text())
        assert validate_case(case)["valid"]
        features = {f["feature_code"]: f["result"] for f in core.compute_features(case, schema)}
        assert features == truth["features"]
        assert [m["result"] for m in core.check_materials(case, schema=schema)] == [truth["material_relation"]]


def test_missing_runs_stay_in_the_denominator(tmp_path, monkeypatch):
    out = generate(tmp_path)
    monkeypatch.setattr(score_heldout, "RUN", tmp_path / "no-runs")
    monkeypatch.setattr(score_heldout, "ROOT", tmp_path)
    result = score_heldout.score(out, "none")
    fixed = result["summary"]["fixed"]
    assert fixed["attempts"] == 32 and fixed["run_status"] == {"not_run_or_interrupted": 32}
    detection = fixed["case_return_for_revision"]
    assert detection["tp"] == 0 and detection["fn"] > 0 and detection["recall"] == 0


def test_claim_aggregation_prefers_contradiction_then_abstention():
    assert score_heldout.aggregate(["supported", "contradicted"]) == "contradicted"
    assert score_heldout.aggregate(["supported", "insufficient_evidence"]) == "insufficient_evidence"
    assert score_heldout.aggregate([]) == "not_extracted"
