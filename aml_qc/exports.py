"""Portable audit JSON, JSONL and CSV views of a frozen Store export."""
import csv
import json
from pathlib import Path


def export_bundle(payload, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    run = payload["candidates_and_open_items"]["run"] or {}
    rows = []
    for item in payload.get("annotations", []):
        review = item["review"]
        rows.append({**item, "task_id": run.get("case_id"), "collection": item["kind"],
                     "value": item["candidate_value"], "final_value": review.get("final_value"),
                     "applicability": review.get("applicability"),
                     "status": "stale" if payload["candidates_and_open_items"]["stale"] else review["status"]})
    collections = {"candidates": rows, "review_events": payload["review_events"],
                   "deliverable": payload["deliverable"].get("annotations", []),
                   "issues": run.get("issues", []), "open_items": payload["candidates_and_open_items"]["open_items"],
                   "annotation_pending": payload.get("annotation_pending", [])}
    for name, items in collections.items():
        (directory / (name + ".jsonl")).write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items))
    fields = ["annotation_id", "task_id", "collection", "label", "value", "final_value", "applicability", "status", "snapshot_id", "source_hash", "schema_version"]
    with (directory / "candidates.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        # Text only: avoid spreadsheet formula evaluation in imported fields.
        for row in rows:
            writer.writerow({k: "'" + str(v) if isinstance(v, str) and v.startswith(("=", "+", "-", "@")) else v for k, v in row.items()})
    (directory / "README.txt").write_text(
        "工e鉴审合成演示导出。audit.json是完整来源、快照、状态和人工记录。\n"
        "candidates保留机器value及人工final_value，不能直接当作质量评分；deliverable仅在任务通过时放入当前有效人工标签和范围裁决。\n"
        "不适用的范围裁决保留空标签值，并以applicability明确标识；issues是独立问题清单，不与标签混淆。\n"
        "通过条件、未决项、过期状态和历史来源以audit.json为准。旧人工记录不沿用为当前裁决。\n"
        "统计分母：此包的候选行数=%d；人工事件=%d；当前未决项=%d。不是准确率/召回率分母。\n"
        "不作真实银行合规或客户无风险结论；哈希链不是防篡改存证。\n" % (len(rows), len(payload["review_events"]), len(collections["open_items"])))
    return directory
