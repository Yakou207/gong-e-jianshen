"""Score saved AI 研判 runs against the designed recommendation of each held-out case.

Every planned case stays in the denominator; a run without a valid draft counts as "no_draft".
"""
import argparse
from collections import Counter
from decimal import Decimal
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ("report_suspicious", "insufficient_evidence", "exclude")


def score(bench, run_dir, suffix, prefix=""):
    manifest = json.loads((bench / "manifest.json").read_text())
    confusion, rows = Counter(), []
    totals = Counter()
    cost = Decimal(0)
    tools = Counter()
    for case in manifest["cases"]:
        truth = json.loads((bench / "truth" / f"{case['case_id']}.json").read_text())
        path = run_dir / f"{prefix}{case['case_id']}-investigate-{suffix}" / "raw.json"
        raw = json.loads(path.read_text()) if path.exists() else None
        predicted = (raw or {}).get("verdict", {}) or {}
        got = predicted.get("recommendation", "no_draft")
        expected = truth["expected_recommendation"]
        confusion[(expected, got)] += 1
        totals["cases"] += 1
        if raw:
            totals["evidence_errors"] += bool(raw["errors"]) and raw.get("verdict") is not None
            totals["number_warnings"] += bool(raw["warnings"])
            totals["policy_flags"] += bool(raw["policy_flags"])
            totals["model_calls"] += len(raw.get("model_requests", []))
            totals["tool_calls"] += len(raw.get("trace", []))
            totals["tool_failures"] += sum(t["status"] != "completed" for t in raw.get("trace", []))
            totals["duration_ms"] += raw.get("stats", {}).get("duration_ms", 0)
            tools.update(t["tool"] for t in raw.get("trace", []))
            for call in raw.get("model_requests", []):
                settlement = call.get("budget_settlement") or {}
                if settlement.get("status") == "settled":
                    cost += Decimal(settlement["cost"])
                    totals["input_tokens"] += call["usage"]["prompt_tokens"]
                    totals["output_tokens"] += call["usage"]["completion_tokens"]
        rows.append({"case_id": case["case_id"], "expected": expected, "predicted": got,
                     "status": (raw or {}).get("status", "not_run"), "errors": (raw or {}).get("errors"),
                     "warnings": (raw or {}).get("warnings"), "policy_flags": (raw or {}).get("policy_flags"),
                     "summary": predicted.get("summary")})
    correct = sum(v for (e, g), v in confusion.items() if e == g)
    per_class = {c: {"correct": confusion[(c, c)], "total": sum(v for (e, _), v in confusion.items() if e == c)} for c in CLASSES}
    return {"contract": "investigation-score-1", "truth_kind": manifest["truth_kind"], "cases": totals["cases"],
            "accuracy": {"correct": correct, "total": totals["cases"], "rate": correct / totals["cases"]},
            "per_class_recall": per_class,
            "confusion": {f"{e}->{g}": v for (e, g), v in sorted(confusion.items())},
            "suspicious_missed_as_exclude": confusion[("report_suspicious", "exclude")],
            "exclude_flagged_suspicious": confusion[("exclude", "report_suspicious")],
            "drafts_with_evidence_errors": totals["evidence_errors"], "drafts_with_number_warnings": totals["number_warnings"],
            "drafts_with_policy_flags": totals["policy_flags"],
            "model_calls": totals["model_calls"], "tool_calls": totals["tool_calls"], "tool_failures": totals["tool_failures"],
            "tool_usage": dict(tools), "input_tokens": totals["input_tokens"], "output_tokens": totals["output_tokens"],
            "cost_cny": str(cost), "cost_cny_per_case": str((cost / totals["cases"]).quantize(Decimal("0.0001"))),
            "mean_duration_s": round(totals["duration_ms"] / 1000 / totals["cases"], 1), "rows": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", type=Path, default=ROOT / "eval/inv-v1")
    parser.add_argument("--runs", type=Path, default=ROOT / "eval/phase2-2026-10-06/run")
    parser.add_argument("--suffix", default="p2")
    parser.add_argument("--prefix", default="")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = score(args.bench, args.runs, args.suffix, args.prefix)
    if args.output:
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, ensure_ascii=False, indent=1))
