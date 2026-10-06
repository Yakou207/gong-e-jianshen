"""Build a case package from what an analyst actually has: a transaction CSV plus alert, KYC and reason text.

CSV columns (header row required): transaction_id, direction, amount, timestamp, counterparty_token,
optional counterparty_name, memo. direction accepts in/out or 转入/转出; amounts are yuan with up to two decimals.
"""
import csv
import io
import re

from .ingest import validate_case
from .workflow import default_schema

DIRECTIONS = {"in": "in", "out": "out", "转入": "in", "转出": "out", "收入": "in", "支出": "out"}
REQUIRED = ("transaction_id", "direction", "amount", "timestamp", "counterparty_token")
FIELDS = ["transaction_id", "account_id", "direction", "amount", "currency", "timestamp", "counterparty_token"]


def _amount(raw, line):
    text = str(raw).replace(",", "").strip()
    if not re.fullmatch(r"\d+(\.\d{1,2})?", text):
        raise ValueError(f"第 {line} 行金额无效：{raw}")
    whole, _, frac = text.partition(".")
    return f"{int(whole)}.{(frac + '00')[:2]}"


def _timestamp(raw, line):
    text = str(raw).strip().replace(" ", "T")
    if not re.search(r"[+-]\d{2}:\d{2}$|Z$", text):
        text += "+08:00"
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?([+-]\d{2}:\d{2}|Z)", text):
        raise ValueError(f"第 {line} 行时间无效：{raw}（示例 2026-09-01 09:30:00）")
    return text


def build_case(form):
    """form: case_id, account_id, business_type, coverage_start, coverage_end, coverage_status (full|partial),
    transactions_csv, focuses (list of text), kyc_text, narrative_text, synthetic_confirmed (must be True)."""
    if form.get("synthetic_confirmed") is not True:
        raise ValueError("演示工作台只接受合成或已脱敏的演示数据，请勾选确认")
    case_id = str(form.get("case_id", "")).strip()
    account = str(form.get("account_id", "")).strip()
    if not re.fullmatch(r"[A-Za-z0-9_\-]{3,40}", case_id) or not account:
        raise ValueError("案件编号须为 3–40 位字母、数字、下划线或连字符，并填写被检账户")
    reader = csv.DictReader(io.StringIO(form.get("transactions_csv", "").lstrip("﻿")))
    if not reader.fieldnames or set(REQUIRED) - {h.strip() for h in reader.fieldnames}:
        raise ValueError("流水 CSV 表头须包含：" + "、".join(REQUIRED))
    rows, parties = [], {}
    for line, raw in enumerate(reader, start=2):
        raw = {k.strip(): (v or "").strip() for k, v in raw.items() if k}
        direction = DIRECTIONS.get(raw["direction"])
        if direction is None:
            raise ValueError(f"第 {line} 行方向须为 in/out 或 转入/转出")
        token = raw["counterparty_token"]
        if not token:
            raise ValueError(f"第 {line} 行缺少对手账户标识")
        name = raw.get("counterparty_name") or token
        parties.setdefault(token, {"counterparty_token": token, "display_name_masked": name, "type": "payment_account"})
        rows.append({"transaction_id": raw["transaction_id"], "account_id": account, "direction": direction,
                     "amount": _amount(raw["amount"], line), "currency": "CNY",
                     "timestamp": _timestamp(raw["timestamp"], line), "channel": "upload",
                     "counterparty_token": token, "counterparty_name_masked": name, "memo": raw.get("memo", "")})
    if not rows:
        raise ValueError("流水 CSV 没有数据行")
    start = _timestamp(form.get("coverage_start") or min(r["timestamp"] for r in rows)[:10] + " 00:00:00", "检查期起点")
    end = _timestamp(form["coverage_end"], "检查期终点") if form.get("coverage_end") else None
    if end is None:
        raise ValueError("请填写检查期终点（不含当日，如 2026-10-01 00:00:00）")
    status = form.get("coverage_status", "full")
    if status not in {"full", "partial"}:
        raise ValueError("流水完整性须为 full 或 partial")
    focuses = [t.strip() for t in form.get("focuses", []) if isinstance(t, str) and t.strip()]
    if not focuses:
        raise ValueError("请至少填写一条预警关注点")
    documents = [{"document_id": "kyc", "revision": "1", "source": "uploaded_profile",
                  "text": form.get("kyc_text", "").strip() or "未提供客户资料。"}]
    if form.get("narrative_text", "").strip():
        documents.append({"document_id": "narrative", "revision": "1", "source": "uploaded_statement",
                          "text": form["narrative_text"].strip()})
    case = {"case_id": case_id, "case_family": "uploaded", "task_mode": "alert_review", "subject_account_id": account,
            "profile": {"business_type": form.get("business_type", "").strip() or "未填写", "data_origin": "synthetic"},
            "data_version": "1", "coverage_start": start, "coverage_end": end, "currency": "CNY",
            "timezone": "Asia/Shanghai", "schema_version": "S1.0",
            "coverage": [{"coverage_id": "uploaded", "source": "transactions", "account_id": account, "fields": FIELDS,
                          "start": start, "end": end, "status": status, "revision": "1"}],
            "counterparties": list(parties.values()), "transactions": rows, "materials": [], "material_links": [],
            "entity_mappings": [],
            "alert": {"alert_id": "alert-" + case_id, "revision": "1",
                      "focuses": [{"focus_id": f"focus-{i}", "text": t} for i, t in enumerate(focuses, 1)],
                      "original_focus": focuses[0], "trigger_features": [], "subject_account_id": account,
                      "start": start, "end": end, "rule_source": "人工录入预警", "rule_version": "S1.0"},
            "documents": documents, "schema": default_schema(),
            "review_scope": {"target_labels": ["F1", "F2", "count", "amount_sum", "counterparty", "time_range", "alert_response"]}}
    validation = validate_case(case)
    if not validation["valid"]:
        raise ValueError("；".join(validation["errors"]))
    return case
