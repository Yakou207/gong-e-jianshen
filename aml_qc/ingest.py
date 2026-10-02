"""Read synthetic case snapshots without treating their text as instructions."""
from __future__ import annotations

import csv
import json
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

from .schema import LABEL_VALUES, load_schema, validate_schema


def amount_cents(value):
    """Convert a decimal yuan string/integer to exact integer fen; never round."""
    if isinstance(value, (bool, float)):
        raise ValueError("金额须使用十进制字符串或整数，不能使用浮点数")
    try:
        amount = Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError("金额格式无效") from None
    if not amount.is_finite() or amount < 0 or amount * 100 != (amount * 100).to_integral_value():
        raise ValueError("金额须为非负数且最多两位小数")
    return int(amount * 100)


def transaction_cents(row):
    if "amount_cents" in row:
        cents = row["amount_cents"]
        if isinstance(cents, bool) or not isinstance(cents, int) or cents < 0:
            raise ValueError("amount_cents须为非负整数")
        if "amount" in row and amount_cents(row["amount"]) != cents:
            raise ValueError("amount与amount_cents冲突")
        return cents
    return amount_cents(row["amount"])


def parse_time(value):
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("时间必须包含时区偏移")
    return dt


def _read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_case(path):
    """Load one JSON snapshot, or an unpacked case.json/CSV/document directory.

    Hidden answers are not enumerated. Invalid cases raise ValueError containing
    only validation diagnostics, never silently discarding conflicting IDs.
    """
    path = Path(path)
    if path.is_file():
        case = json.loads(path.read_text(encoding="utf-8"))
    else:
        case = json.loads((path / "case.json").read_text(encoding="utf-8"))
        for key in ("transactions", "counterparties"):
            file = path / f"{key}.csv"
            if file.exists():
                case[key] = _read_csv(file)
        alert = path / "alert.json"
        if alert.exists():
            case["alert"] = json.loads(alert.read_text(encoding="utf-8"))
        documents = {d["document_id"]: d for d in case.get("documents", [])}
        for name in ("narrative", "kyc"):
            file = path / f"{name}.txt"
            if file.exists():
                prior = documents.get(name, {})
                documents[name] = {"document_id": name, "revision": prior.get("revision", "1"),
                                   "source": prior.get("source", "synthetic"),
                                   "text": file.read_text(encoding="utf-8")}
        case["documents"] = list(documents.values())
        material_dir = path / "materials"
        if material_dir.exists():
            case["materials"] = [json.loads(p.read_text(encoding="utf-8"))
                                 for p in sorted(material_dir.glob("*.json"))]
    result = validate_case(case)
    if not result["valid"]:
        raise ValueError("; ".join(result["errors"]))
    # Byte-identical duplicate IDs represent the same transaction, not extra money.
    case["transactions"] = list({r["transaction_id"]: r for r in case.get("transactions", [])}.values())
    return case


def validate_case(case):
    errors, warnings = [], []
    if not isinstance(case, dict):
        return {"valid": False, "errors": ["案件必须为JSON对象"], "warnings": []}
    collections = ("transactions", "counterparties", "coverage", "documents", "claims", "materials", "material_links", "entity_mappings")
    for key in collections:
        if key in case and (not isinstance(case[key], list) or any(not isinstance(row, dict) for row in case[key])):
            errors.append(f"{key}必须为JSON对象数组")
    if errors:
        return {"valid": False, "errors": errors, "warnings": warnings}
    required = ("case_id", "case_family", "task_mode", "subject_account_id", "data_version",
                "coverage_start", "coverage_end", "currency", "timezone", "schema_version")
    for name in required:
        if not case.get(name):
            errors.append(f"缺少案件字段:{name}")
    if case.get("task_mode") not in ("alert_review", "annotation_only"):
        errors.append("未知task_mode")
    schema = case.get("schema", load_schema())
    schema_errors = validate_schema(schema)
    errors.extend(schema_errors)
    if isinstance(schema, dict) and schema.get("schema_version") != case.get("schema_version"):
        errors.append("Schema内容版本须与案件schema_version一致")
    scope = case.get("review_scope", {})
    if not isinstance(scope, dict):
        errors.append("review_scope必须为对象")
    else:
        targets = scope.get("target_labels")
        if "target_labels" not in scope:
            if case.get("task_mode") == "annotation_only":
                errors.append("annotation_only必须显式指定非空target_labels")
            elif not schema_errors and set(LABEL_VALUES) - set(schema['labels']):
                errors.append("默认目标标签须全部在当前规范中实现，或显式指定target_labels")
        elif not isinstance(targets, list) or not targets or any(not isinstance(name, str) for name in targets):
            errors.append("target_labels必须为非空标签名称数组")
        elif len(targets) != len(set(targets)):
            errors.append("target_labels不允许重复")
        elif not schema_errors:
            for name in targets:
                if name not in schema["labels"] or name not in LABEL_VALUES:
                    errors.append(f"目标标签未在当前规范中实现:{name}")
        for field in ("upgraded_leads", "lead_dispositions"):
            records = scope.get(field, [])
            if not isinstance(records, list) or any(not isinstance(r, dict) for r in records):
                errors.append(f"review_scope.{field}必须为对象数组")
                continue
            if field == "upgraded_leads":
                ids = []
                for record in records:
                    if any(not isinstance(record.get(key), str) or not record[key].strip() for key in ("focus_id", "text")):
                        errors.append("升级线索须有focus_id和具体需回应事项text")
                    else:
                        ids.append(record["focus_id"])
                if len(ids) != len(set(ids)):
                    errors.append("升级线索focus_id不可重复")
            else:
                ids = []
                for record in records:
                    required_fields = ("event_id", "lead_id", "context_hash", "actor", "reason", "snapshot_id", "run_id", "source_hash", "created_at")
                    if any(not isinstance(record.get(key), str) or not record[key].strip() for key in required_fields):
                        errors.append("线索处置须保留完整人员、依据、版本和事件绑定")
                    else:
                        ids.append(record["event_id"])
                    if record.get("status") not in {"upgraded", "closed"} or not isinstance(record.get("candidate"), dict):
                        errors.append("线索处置状态或原候选格式无效")
                    elif record["candidate"].get("lead_id") != record.get("lead_id"):
                        errors.append("线索处置与原候选标识不一致")
                if len(ids) != len(set(ids)):
                    errors.append("线索处置事件不可重复")
    if case.get("currency") != "CNY":
        errors.append("首版仅支持CNY")
    if case.get("timezone") != "Asia/Shanghai":
        errors.append("首版时间窗口固定Asia/Shanghai")
    bounds = None
    try:
        start, end = parse_time(case["coverage_start"]), parse_time(case["coverage_end"])
        if start >= end:
            raise ValueError("案件范围必须为非空半开区间")
        bounds = start, end
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        errors.append(f"案件时间范围无效:{exc}")
    if case.get("task_mode") == "alert_review" and not case.get("alert"):
        warnings.append("缺少原预警，相关必需检查不能完成")
    if not case.get("coverage"):
        warnings.append("未声明字段及窗口覆盖，精确/否定判断将不可判定")
    forbidden = {"ground_truth", "hidden_truth", "reference_answers", "expected_results"}
    def inspect_keys(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in forbidden:
                    errors.append(f"任务包禁止包含评测参考字段:{key}")
                inspect_keys(item)
        elif isinstance(value, list):
            for item in value:
                inspect_keys(item)
    inspect_keys(case)
    seen = {}
    for index, row in enumerate(case.get("transactions", [])):
        prefix = f"transactions[{index}]"
        try:
            for name in ("transaction_id", "account_id", "direction", "currency", "timestamp", "counterparty_token"):
                if not row.get(name):
                    raise ValueError(f"缺少{name}")
            if row["account_id"] != case.get("subject_account_id"):
                raise ValueError("不属于本案单一账户范围")
            if row["direction"] not in ("in", "out"):
                raise ValueError("direction须为in/out")
            if row["currency"] != "CNY":
                raise ValueError("币种不一致")
            transaction_cents(row)
            timestamp = parse_time(row["timestamp"])
            if bounds and not bounds[0] <= timestamp < bounds[1]:
                raise ValueError("交易时间在案件半开区间之外")
            tid = row["transaction_id"]
            if tid in seen and seen[tid] != row:
                raise ValueError(f"冲突的重复交易ID:{tid}")
            if tid in seen:
                warnings.append(f"相同重复交易ID将去重:{tid}")
            seen[tid] = row
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            errors.append(f"{prefix}:{exc}")
    for key, id_key in (("documents", "document_id"), ("materials", "material_id"),
                        ("counterparties", "counterparty_token"), ("claims", "claim_id"),
                        ("material_links", "link_id"), ("entity_mappings", "mapping_id")):
        identities = set()
        for index, row in enumerate(case.get(key, [])):
            identity = row.get(id_key)
            if not identity or identity in identities:
                errors.append(f"{key}[{index}]:缺少或重复{id_key}")
            identities.add(identity)
    for index, row in enumerate(case.get("coverage", [])):
        try:
            if row.get("status") not in ("full", "partial", "unknown"):
                raise ValueError("status须为full/partial/unknown")
            if row.get("source", "transactions") != "transactions":
                continue
            if row.get("account_id") != case.get("subject_account_id") or not row.get("fields"):
                raise ValueError("必须声明账户与字段")
            start, end = parse_time(row["start"]), parse_time(row["end"])
            if start >= end:
                raise ValueError("覆盖范围为空")
            for gap in row.get("missing_ranges", []):
                if not start <= parse_time(gap["start"]) < parse_time(gap["end"]) <= end:
                    raise ValueError("缺失范围不在声明范围内")
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            errors.append(f"coverage[{index}]:{exc}")
    for doc in case.get("documents", []):
        if not doc.get("revision") or not isinstance(doc.get("text"), str):
            errors.append(f"文档{doc.get('document_id')}:必须有revision和text")
    return {"valid": not errors, "errors": errors, "warnings": warnings}
