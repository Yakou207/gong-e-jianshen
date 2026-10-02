"""Local append-only snapshots and human decisions. Not a production audit system."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from uuid import uuid4

from .annotations import annotation_pending, apply_reviews, build_annotations, prepare_decision, resolved_targets
from .core import check_coverage
from .depgraph import canonical, digest
from .ingest import validate_case
from . import migrations
from .workflow import IMPLEMENTATION_HASH, default_schema


def now():
    return datetime.now(timezone.utc).isoformat()


def audit_integrity(events):
    previous = "GENESIS"
    for index, event in enumerate(events):
        payload = {k: v for k, v in event.items() if k not in {"previous_hash", "event_hash"}}
        if event["previous_hash"] != previous or event["event_hash"] != digest({"previous_hash": previous, "payload": payload}):
            return {"valid": False, "checked_events": index, "broken_event_id": event.get("event_id")}
        previous = event["event_hash"]
    return {"valid": True, "checked_events": len(events), "last_hash": previous}


class Store:
    def __init__(self, path):
        self.path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
              CREATE TABLE IF NOT EXISTS cases (case_id TEXT PRIMARY KEY, source_hash TEXT NOT NULL,
                current_run TEXT, created_at TEXT NOT NULL);
              CREATE TABLE IF NOT EXISTS sources (source_hash TEXT PRIMARY KEY, case_id TEXT NOT NULL,
                package TEXT NOT NULL, reason TEXT NOT NULL, created_at TEXT NOT NULL);
              CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, case_id TEXT NOT NULL,
                source_hash TEXT NOT NULL, result TEXT NOT NULL, created_at TEXT NOT NULL);
              CREATE TABLE IF NOT EXISTS events (seq INTEGER PRIMARY KEY AUTOINCREMENT,
                case_id TEXT NOT NULL, payload TEXT NOT NULL, previous_hash TEXT NOT NULL, event_hash TEXT NOT NULL);
              CREATE TABLE IF NOT EXISTS migration_previews (preview_id TEXT PRIMARY KEY,
                payload TEXT NOT NULL, payload_hash TEXT NOT NULL, created_at TEXT NOT NULL);
              CREATE TABLE IF NOT EXISTS migration_receipts (preview_id TEXT NOT NULL, case_id TEXT NOT NULL,
                payload TEXT NOT NULL, receipt_hash TEXT NOT NULL, created_at TEXT NOT NULL,
                PRIMARY KEY(preview_id,case_id));
            """)

    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        return db

    def create(self, package, reason="导入合成案例"):
        package = deepcopy(package)
        package.setdefault("schema", default_schema())
        self.validate(package)
        package["transactions"] = list({r["transaction_id"]: r for r in package.get("transactions", [])}.values())
        source_hash = digest(package)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            migrations.check_version_binding(migrations.catalogue(self,db)["schemas"],package["schema"])
            if db.execute("SELECT 1 FROM cases WHERE case_id=?", (package["case_id"],)).fetchone():
                raise ValueError("案例ID已存在；修改来源请使用新版本接口")
            db.execute("INSERT INTO sources VALUES (?,?,?,?,?)", (source_hash, package["case_id"], canonical(package), reason, now()))
            db.execute("INSERT INTO cases VALUES (?,?,NULL,?)", (package["case_id"], source_hash, now()))
        return self.get(package["case_id"])

    @staticmethod
    def validate(package):
        validation = validate_case(package)
        if not validation["valid"]:
            raise ValueError("; ".join(validation["errors"]))
        if package.get("profile", {}).get("data_origin") != "synthetic":
            raise ValueError("演示工作台仅接受profile.data_origin=synthetic的合成资料")
        if not isinstance(package.get("schema"), dict) or package["schema"].get("schema_version") != package["schema_version"]:
            raise ValueError("Schema内容版本须与案件schema_version一致")

    def _event(self, db, case_id, payload):
        previous = db.execute("SELECT event_hash FROM events WHERE case_id=? ORDER BY seq DESC LIMIT 1", (case_id,)).fetchone()
        previous_hash = previous[0] if previous else "GENESIS"
        payload = {"event_id": uuid4().hex, "created_at": now(), **payload}
        event_hash = digest({"previous_hash": previous_hash, "payload": payload})
        db.execute("INSERT INTO events (case_id,payload,previous_hash,event_hash) VALUES (?,?,?,?)",
                   (case_id, canonical(payload), previous_hash, event_hash))
        return payload

    def list(self):
        with self.connect() as db:
            ids = [r[0] for r in db.execute("SELECT case_id FROM cases ORDER BY case_id")]
        rows = []
        for case_id in ids:
            item = self.get(case_id)
            run = item["latest_run"] or {}
            rows.append({"case_id": case_id, "title": item["package"].get("title", case_id),
                         "task_mode": item["package"]["task_mode"], "data_version": item["package"]["data_version"],
                         "run_status": "stale" if item["stale"] else run.get("run_status", "not_run"),
                         "qc_recommendation": run.get("qc_recommendation", "未运行"),
                         "review_status": item["review_status"], "stale": item["stale"],
                         "schema_version": item["package"]["schema_version"], "schema_hash": digest(item["package"]["schema"]),
                         "coverage_summary": check_coverage(item["package"])["status"]})
        return rows

    def get(self, case_id):
        with self.connect() as db:
            row = db.execute("SELECT * FROM cases WHERE case_id=?", (case_id,)).fetchone()
            if not row:
                raise KeyError("案例不存在")
            package = json.loads(db.execute("SELECT package FROM sources WHERE source_hash=?", (row["source_hash"],)).fetchone()[0])
            run_row = db.execute("SELECT result FROM runs WHERE run_id=?", (row["current_run"],)).fetchone()
            run = json.loads(run_row[0]) if run_row else None
            run_package = json.loads(db.execute("SELECT package FROM sources WHERE source_hash=?", (run["source_hash"],)).fetchone()[0]) if run else package
            events = [{**json.loads(e["payload"]), "previous_hash": e["previous_hash"], "event_hash": e["event_hash"]}
                      for e in db.execute("SELECT * FROM events WHERE case_id=? ORDER BY seq", (case_id,))]
            history = [{"run_id": r["run_id"], "source_hash": r["source_hash"], "created_at": r["created_at"]}
                       for r in db.execute("SELECT run_id,source_hash,created_at FROM runs WHERE case_id=? ORDER BY created_at DESC", (case_id,))]
        engine_changed = bool(run and run.get("execution") and
                              run["execution"].get("implementation_hash") != IMPLEMENTATION_HASH)
        stale = bool(run and (run["source_hash"] != row["source_hash"] or engine_changed))
        integrity = audit_integrity(events)
        annotations = apply_reviews((run or {}).get("annotations") or build_annotations(run, run_package), events,
                                    stale=stale, integrity=integrity["valid"])
        label_pending = annotation_pending(annotations)
        resolved = resolved_targets(annotations)
        current_events = [e for e in events if run and e.get("snapshot_id") == run["snapshot_id"] and not stale]
        last = {e.get("target_id"): e for e in current_events if e.get("action") != "source_changed"}
        source_task_events = [e for e in events if e.get("target_id") == "task" and e.get("source_hash") == row["source_hash"]
                              and e.get("action") in {"confirm", "reconfirm", "dispute", "request_correction", "close_item"}]
        source_task = source_task_events[-1] if source_task_events else {}
        open_items = []
        for item in (run or {}).get("open_items", []):
            if item["target_id"] in resolved and item.get("kind") in {"claim_error", "claim_unresolved", "manual_focus", "focus_not_addressed", "manual_material"}:
                continue
            decision = last.get(item["item_id"], {})
            if decision.get("action") == "reject" or (decision.get("action") == "close_item" and decision.get("resolution") in {"addressed", "corresponds", "not_applicable"}):
                continue
            open_items.append(item)
        if source_task.get("action") in {"dispute", "request_correction"}:
            open_items.append({"item_id": "task-review", "target_id": "task", "kind": source_task["action"],
                               "title": "任务争议待裁决" if source_task["action"] == "dispute" else "任务补正待处理",
                               "reason": source_task["reason"], "event_id": source_task["event_id"]})
        if not integrity["valid"]:
            open_items.append({"item_id": "audit-integrity", "target_id": "task", "kind": "audit_invalid",
                               "title": "审计链一致性检查失败", "reason": "当前记录须核查，不能作为已通过结果导出"})
        pending_checks = []
        for check in (run or {}).get("required_checks", []):
            if check["status"] == "completed":
                continue
            if check["check_id"] in resolved and (resolved[check["check_id"]]["review"].get("verification") or {}).get("execution_status") == "completed":
                continue
            semantic_labels = [a for a in annotations if a["kind"] == "semantic"]
            if check["check_id"] == "semantic" and check["status"] == "pending_judgement" and semantic_labels and all(a["review"].get("valid") for a in semantic_labels):
                continue
            related = [i for i in (run or {}).get("open_items", [])
                       if i["target_id"] == check["check_id"] or i["target_id"].startswith(check["check_id"] + ":")]
            manually_closed = related and all(last.get(i["item_id"], {}).get("action") == "close_item" and
                last[i["item_id"]].get("resolution") in {"addressed", "corresponds", "not_applicable"} for i in related)
            if check["status"] != "pending_judgement" or not manually_closed:
                pending_checks.append(check)
        disputes = [e for e in last.values() if e.get("action") == "dispute"]
        if source_task.get("action") == "dispute" and source_task not in disputes:
            disputes.append(source_task)
        task = last.get("task", {})
        final_confirmation_current = bool(task and current_events and current_events[-1].get("event_id") == task.get("event_id"))
        can_pass = bool(run and not stale and not open_items and not pending_checks and not disputes and not label_pending
                        and run.get("required_checks") and run.get("run_status") in {"completed", "partial"})
        status = "needs_review" if stale else "本次质检范围内通过" if can_pass and final_confirmation_current and task.get("action") in {"confirm", "reconfirm"} else "disputed" if disputes else "待人工复核"
        if run:
            run["review_status"] = status
        return {"package": package, "source_hash": row["source_hash"], "latest_run": run, "stale": stale,
                "review_events": events, "open_items": open_items, "pending_checks": pending_checks,
                "review_status": status, "can_pass": can_pass, "history": history, "audit_integrity": integrity,
                "engine_changed": engine_changed, "annotations": annotations,
                "annotation_pending": label_pending, "annotation_pending_count": len(label_pending)}

    def change_source(self, case_id, package, reason, *, actor="operator", context=None):
        if not reason.strip() or not actor.strip():
            raise ValueError("修改来源必须记录原因和人员")
        old = self.get(case_id)
        package = deepcopy(package)
        package.setdefault("schema", old["package"]["schema"])
        if package.get("schema_version") != old["package"]["schema_version"] or digest(package["schema"]) != digest(old["package"]["schema"]):
            raise ValueError("Schema规范内容或版本变化须先生成迁移预览并逐案执行，不能通过普通来源修订绕过")
        if package.get("case_id") != case_id:
            raise ValueError("不可修改案件ID")
        if package == old["package"]:
            raise ValueError("来源没有变化")
        package["data_version"] = str(int(old["package"]["data_version"]) + 1) if str(old["package"]["data_version"]).isdigit() else uuid4().hex[:12]
        # A text edit cannot silently retain the old span revision.
        for collection, field in [("documents", "document_id"), ("materials", "material_id"), ("entity_mappings", "mapping_id")]:
            prior = {i[field]: i for i in old["package"].get(collection, [])}
            for item in package.get(collection, []):
                before = prior.get(item[field])
                if before and before != item and before.get("revision") == item.get("revision"):
                    item["revision"] = str(item.get("revision", "1")) + "." + digest(item)[:8]
        if package.get("alert") and package["alert"] != old["package"].get("alert"):
            package["alert"]["revision"] = digest(package["alert"])[:12]
        self.validate(package)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._write_source_change(db,old,package,reason,actor,context)
        return self.get(case_id)

    def _write_source_change(self, db, old, package, reason, actor, context):
        """Append a validated source change inside the caller's transaction."""
        case_id, source_hash = package["case_id"], digest(package)
        current = db.execute("SELECT source_hash FROM cases WHERE case_id=?", (case_id,)).fetchone()[0]
        if current != old["source_hash"]:
            raise ValueError("来源已被其他操作更新，请刷新")
        db.execute("INSERT INTO sources VALUES (?,?,?,?,?)", (source_hash, case_id, canonical(package), reason, now()))
        db.execute("UPDATE cases SET source_hash=? WHERE case_id=?", (source_hash, case_id))
        self._event(db, case_id, {"action": "source_changed", "target_id": "task", "reason": reason,
                    "source_hash": source_hash, "previous_source_hash": current, "actor": actor, "context": context})
        if old["latest_run"]:
            old_snapshot = old["latest_run"]["snapshot_id"]
            records = migrations.review_records(old)
            self._event(db, case_id, {"action": "needs_review", "target_id": "task", "reason": "来源改变，旧结论不沿用；首版保守要求全部人工裁决重新确认",
                        "invalidated_snapshot_id": old_snapshot, "source_hash": source_hash,
                        "previous_source_hash": current, "review_records": records, "actor": "system"})

    def schemas(self):
        with self.connect() as db:
            return migrations.catalogue(self,db)

    def preview_migration(self, **kwargs):
        return migrations.create_preview(self,implementation_hash=IMPLEMENTATION_HASH,**kwargs)

    def migration_preview(self, preview_id):
        return migrations.get_preview(self,preview_id,IMPLEMENTATION_HASH)

    def migrate_case(self, preview_id, case_id, **kwargs):
        return migrations.apply_case(self,preview_id,case_id,implementation_hash=IMPLEMENTATION_HASH,**kwargs)

    def save_run(self, case_id, source_hash, result):
        result = deepcopy(result)
        if result.get("case_id") != case_id:
            raise ValueError("运行结果不属于当前案件")
        result.update(run_id=uuid4().hex, snapshot_id=uuid4().hex, source_hash=source_hash, created_at=now())
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            source = db.execute("SELECT package FROM sources WHERE source_hash=? AND case_id=?", (source_hash, case_id)).fetchone()
            if not source:
                raise ValueError("运行来源不属于当前案件")
            result["annotations"] = build_annotations(result, json.loads(source[0]))
            db.execute("INSERT INTO runs VALUES (?,?,?,?,?)", (result["run_id"], case_id, source_hash, canonical(result), result["created_at"]))
            current = db.execute("SELECT source_hash FROM cases WHERE case_id=?", (case_id,)).fetchone()[0]
            if current == source_hash:
                db.execute("UPDATE cases SET current_run=? WHERE case_id=?", (result["run_id"], case_id))
            else:
                raise ValueError("运行时来源已改变，本次结果不作为当前结论，请重新运行")
        return self.get(case_id)

    def review(self, case_id, *, action, target_id, reason, actor="reviewer", resolution=None,
               new_value=None, claim_patch=None, evidence=None, snapshot_id=None, expected_event_id=None, previous_event_id=None):
        if not reason.strip() or not actor.strip():
            raise ValueError("人工裁决必须记录人员和依据")
        state = self.get(case_id)
        run = state["latest_run"]
        if not run or state["stale"]:
            raise ValueError("请先对当前资料运行质检，旧快照不可裁决")
        annotation = next((a for a in state["annotations"] if a["annotation_id"] == target_id), None)
        if annotation:
            if snapshot_id != run["snapshot_id"]:
                raise ValueError("标签裁决必须绑定当前快照，请刷新后重试")
            if expected_event_id != annotation["review"]["event_id"]:
                raise ValueError("标签已被其他人员裁决，请刷新后重试")
            decision = prepare_decision(annotation, state["package"], action=action, new_value=new_value,
                claim_patch=claim_patch, evidence=evidence, previous_event_id=previous_event_id,
                known_evidence=[ref for a in state["annotations"] for ref in a["evidence"]])
            with self.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                current = db.execute("SELECT source_hash,current_run FROM cases WHERE case_id=?", (case_id,)).fetchone()
                latest = db.execute("SELECT event_hash FROM events WHERE case_id=? ORDER BY seq DESC LIMIT 1", (case_id,)).fetchone()
                if current[0] != state["source_hash"] or current[1] != run["run_id"] or (latest[0] if latest else None) != (state["review_events"][-1]["event_hash"] if state["review_events"] else None):
                    raise ValueError("来源、运行或人工记录已改变，请刷新后重新裁决")
                self._event(db, case_id, {**decision, "target_id": target_id, "subject_target_id": annotation["target_id"],
                    "reason": reason, "actor": actor, "snapshot_id": snapshot_id, "run_id": run["run_id"],
                    "source_hash": state["source_hash"], "schema_version": annotation["schema_version"],
                    "object": annotation["object"]})
            return self.get(case_id)
        if new_value is not None or claim_patch is not None or evidence is not None:
            raise ValueError("标签修订与证据字段只能提交给当前标签对象")
        allowed = {"confirm", "reject", "request_correction", "dispute", "reconfirm", "close_item", "upgrade_lead", "close_lead"}
        if action not in allowed:
            raise ValueError("未知人工动作")
        issues = {i["issue_id"]: i for i in run.get("issues", [])}
        existing_leads = state["package"].get("review_scope", {}).get("upgraded_leads", [])
        def refers_to_lead(lead):
            return (target_id in {lead.get("origin_issue_id"), lead.get("focus_id"), "semantic:" + str(lead.get("focus_id"))}
                    or issues.get(target_id, {}).get("target_id") == "semantic:" + str(lead.get("focus_id")))
        closing_existing_lead = action == "close_lead" and any(refers_to_lead(lead) for lead in existing_leads)
        if target_id != "task" and target_id not in issues and not closing_existing_lead:
            raise ValueError("人工裁决对象不属于当前快照")
        if action in {"upgrade_lead", "close_lead"}:
            if target_id == "task":
                raise ValueError("线索操作必须指定具体线索")
            package = deepcopy(state["package"])
            leads = package.setdefault("review_scope", {}).setdefault("upgraded_leads", [])
            if action == "upgrade_lead":
                if issues[target_id].get("type") != "new_lead":
                    raise ValueError("只有新增线索候选可以升级为应回应事项")
                if any(lead.get("origin_issue_id") == target_id for lead in leads):
                    raise ValueError("该线索已经升级")
                leads.append({"focus_id": "lead-" + digest([target_id, reason])[:12], "text": reason, "origin_issue_id": target_id})
            else:
                kept = [lead for lead in leads if not refers_to_lead(lead)]
                if len(kept) == len(leads):
                    raise ValueError("当前对象没有已升级线索")
                package["review_scope"]["upgraded_leads"] = kept
            return self.change_source(case_id, package, action + ": " + reason, actor=actor,
                                      context={"review_action": action, "target_id": target_id,
                                               "snapshot_id": run["snapshot_id"], "source_hash": state["source_hash"]})
        if target_id == "task":
            if action not in {"confirm", "reconfirm", "dispute", "request_correction", "close_item"}:
                raise ValueError("该动作不适用于整个任务")
            if action == "close_item":
                if not any(item.get("item_id") == "task-review" for item in state["open_items"]):
                    raise ValueError("没有待处理的任务级补正或争议")
                if resolution not in {"addressed", "not_applicable"}:
                    raise ValueError("任务补正或争议须明确裁决为已处理或不适用")
            if action in {"confirm", "reconfirm"} and not state["can_pass"]:
                raise ValueError("存在未完成检查、开放事项或争议，不能通过")
        else:
            issue = issues[target_id]
            if action == "close_item":
                if issue["type"] not in {"manual_focus", "manual_material", "manual_extraction"}:
                    raise ValueError("该事项需要更正来源并重查，不可手动关闭")
                permitted = {"manual_focus": {"addressed", "not_addressed"}, "manual_material": {"corresponds", "mismatch", "not_applicable"}, "manual_extraction": {"addressed"}}
                if resolution not in permitted[issue["type"]]:
                    raise ValueError("请选择该事项对应的明确判断结果")
            if action == "reject" and issue["type"] in {"execution_failed", "insufficient_coverage", "material_insufficient", "claim_unresolved", "missing_alert", "manual_extraction", "manual_focus", "manual_material"}:
                raise ValueError("未完成的检查不能通过否决候选来绕过")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            current = db.execute("SELECT source_hash,current_run FROM cases WHERE case_id=?", (case_id,)).fetchone()
            latest = db.execute("SELECT event_hash FROM events WHERE case_id=? ORDER BY seq DESC LIMIT 1", (case_id,)).fetchone()
            if current[0] != state["source_hash"] or current[1] != run["run_id"] or (latest[0] if latest else None) != (state["review_events"][-1]["event_hash"] if state["review_events"] else None):
                raise ValueError("资料、运行或人工记录已改变，请刷新后重新裁决")
            self._event(db, case_id, {"action": action, "target_id": target_id, "reason": reason, "actor": actor,
                        "resolution": resolution, "snapshot_id": run["snapshot_id"], "source_hash": state["source_hash"],
                        "evidence": issues.get(target_id, {}).get("evidence", [])})
        return self.get(case_id)

    def export(self, case_id):
        state = self.get(case_id)
        run = state["latest_run"]
        current_events = [e for e in state["review_events"] if run and not state["stale"] and state["audit_integrity"]["valid"]
                          and e.get("snapshot_id") == run["snapshot_id"]]
        decisions = {e.get("target_id"): e for e in current_events}
        confirmed = [i for i in (run or {}).get("issues", []) if decisions.get(i["issue_id"], {}).get("action") in {"confirm", "reconfirm"}]
        final_annotations = [{**a, "machine_candidate_value": a["candidate_value"], "machine_object": a["object"],
                              "value": a["review"]["final_value"], "final_value": a["review"]["final_value"],
                              "object": a["review"].get("object") or a["object"]}
                             for a in state["annotations"] if a["review"].get("valid")]
        with self.connect() as db:
            historical_sources = [{"source_hash": row["source_hash"], "package": json.loads(row["package"]),
                                   "reason": row["reason"], "created_at": row["created_at"]}
                                  for row in db.execute("SELECT * FROM sources WHERE case_id=? ORDER BY created_at", (case_id,))]
            historical_runs = [json.loads(row["result"]) for row in
                               db.execute("SELECT result FROM runs WHERE case_id=? ORDER BY created_at", (case_id,))]
            migration_previews = []
            for row in db.execute("SELECT preview_id FROM migration_previews"):
                preview, preview_hash = migrations.read_preview(db,row["preview_id"])
                if case_id in preview["selected_case_ids"]:
                    migration_previews.append({**preview,"preview_hash":preview_hash})
            migration_receipts = migrations.read_receipts(db,case_id=case_id)
        return {"format": "gong-e-qc-audit-1", "exported_at": now(), "scope": "合成单账户演示；不证明客户无风险或生产合规",
                "package": state["package"], "source_hash": state["source_hash"],
                "deliverable": {"snapshot_id": run["snapshot_id"] if run and not state["stale"] else None,
                                "review_status": state["review_status"],
                                "confirmed_issues": confirmed if state["review_status"] == "本次质检范围内通过" else [],
                                "annotations": final_annotations
                                    if state["review_status"] == "本次质检范围内通过" else [],
                                "passed": state["review_status"] == "本次质检范围内通过"},
                "candidates_and_open_items": {"run": run, "open_items": state["open_items"], "stale": state["stale"],
                                              "current_confirmed_issues": confirmed},
                "annotations": state["annotations"], "annotation_pending": state["annotation_pending"],
                "review_events": state["review_events"], "history": state["history"],
                "historical_sources": historical_sources, "historical_runs": historical_runs,
                "schema_migrations": {"previews":migration_previews,"receipts":migration_receipts},
                "historical_scope": "冻结来源和运行仅供证据回溯，包含历史及可能已过期版本，不计入当前可交付集合。",
                "audit_integrity": state["audit_integrity"],
                "integrity_note": "哈希链用于内部一致性检查，不是防篡改存证。历史人工事件不作为当前有效裁决导出。"}
