"""Build a case package from what an analyst actually has: a transaction CSV plus alert, KYC and reason text.

CSV columns (header row required): transaction_id, direction, amount, timestamp, counterparty_token, optional
counterparty_name, memo — or the usual Chinese bank-export headers (交易流水号, 借贷标志, 交易金额, 交易时间, 对方账号,
对方户名, 摘要). direction accepts in/out, 转入/转出, 收入/支出, 贷/借; amounts are yuan with up to two decimals.
"""
import csv
from datetime import datetime
import io
import re

from .ingest import validate_case
from .workflow import default_schema

DIRECTIONS = {"in": "in", "out": "out", "转入": "in", "转出": "out", "收入": "in", "支出": "out", "收": "in", "支": "out",
              "付": "out", "贷": "in", "借": "out", "c": "in", "d": "out", "credit": "in", "debit": "out"}
REQUIRED = ("transaction_id", "direction", "amount", "timestamp", "counterparty_token")
FIELDS = ["transaction_id", "account_id", "direction", "amount", "currency", "timestamp", "counterparty_token"]
HEADERS = {
    "transaction_id": ("交易流水号", "流水号", "交易编号", "交易序号"),
    "direction": ("借贷标志", "收付标志", "收支方向", "交易方向", "方向"),
    "amount": ("交易金额", "金额", "发生额"),
    "timestamp": ("交易时间", "交易日期时间", "时间"),
    "counterparty_token": ("对方账号", "对手账号", "对方账户", "对手账户"),
    "counterparty_name": ("对方户名", "对手户名", "对方名称", "对手名称"),
    "memo": ("摘要", "备注", "用途", "附言"),
}
ALIASES = {alias: field for field, names in HEADERS.items() for alias in names}
MAX_ROWS = 50000
MAX_REPORTED = 20


def _amount(raw, line):
    text = re.sub(r"[,\s¥￥元]", "", str(raw))
    if not re.fullmatch(r"\d+(\.\d{1,2})?", text):
        raise ValueError(f"第 {line} 行金额无效：{raw}（填写不带符号的元金额，最多两位小数）")
    whole, _, frac = text.partition(".")
    return f"{int(whole)}.{(frac + '00')[:2]}"


def _timestamp(raw, line):
    text = str(raw).strip()
    zone = re.search(r"([+-]\d{2}:\d{2}|Z)$", text)
    body = text[:zone.start()].strip() if zone else text
    body = re.sub(r"[年月]", "-", body).replace("日", " ").replace("秒", "")
    body = re.sub(r"[时分]", ":", body).strip().rstrip(":")
    m = (re.fullmatch(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})(?:[ T]+(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?)?", body)
         or re.fullmatch(r"(\d{4})(\d{2})(\d{2})(?:[ T]?(\d{2})(\d{2})(\d{2})?)?", body))
    if not m:
        raise ValueError(f"第 {line} 行时间无效：{raw}（示例 2026-09-01 09:30:00）")
    y, mo, d, h, mi, sec = (int(x) if x else 0 for x in m.groups())
    try:
        datetime(y, mo, d, h, mi, sec)
    except ValueError:
        raise ValueError(f"第 {line} 行时间无效：{raw}") from None
    return f"{y:04d}-{mo:02d}-{d:02d}T{h:02d}:{mi:02d}:{sec:02d}" + (zone.group(1) if zone else "+08:00")


def build_case(form):
    """form: case_id, account_id, business_type, coverage_start, coverage_end, coverage_status (full|partial),
    transactions_csv, focuses (list of text), kyc_text, narrative_text, synthetic_confirmed (must be True)."""
    if form.get("synthetic_confirmed") is not True:
        raise ValueError("演示工作台只接受合成或已脱敏的演示数据，请勾选确认")
    case_id = str(form.get("case_id", "")).strip()
    account = str(form.get("account_id", "")).strip()
    if not re.fullmatch(r"[A-Za-z0-9_\-]{3,40}", case_id) or not account:
        raise ValueError("案件编号须为 3–40 位字母、数字、下划线或连字符，并填写被检账户")
    reader = csv.DictReader(io.StringIO(form.get("transactions_csv", "").lstrip("\ufeff")))
    header = {h: ALIASES.get(h.strip(), h.strip()) for h in reader.fieldnames or [] if h}
    if set(REQUIRED) - set(header.values()):
        missing = [f"{f}（或 {HEADERS[f][0]}）" for f in REQUIRED if f not in header.values()]
        raise ValueError("流水 CSV 表头缺少：" + "、".join(missing))
    rows, parties, problems, seen = [], {}, [], {}
    for line, raw in enumerate(reader, start=2):
        if line - 1 > MAX_ROWS:
            raise ValueError(f"单个案件最多 {MAX_ROWS} 行流水，请按检查期拆分")
        raw = {header[k]: (v or "").strip() for k, v in raw.items() if k in header}
        if not any(raw.values()):
            continue
        try:
            direction = DIRECTIONS.get(raw["direction"].lower())
            if direction is None:
                raise ValueError(f"第 {line} 行方向无效：{raw['direction']}（转入/转出、收入/支出、贷/借 或 in/out）")
            token, tx_id = raw["counterparty_token"], raw["transaction_id"]
            if not token or not tx_id:
                raise ValueError(f"第 {line} 行缺少交易编号或对手账户标识")
            if tx_id in seen:
                raise ValueError(f"第 {line} 行交易编号 {tx_id} 与第 {seen[tx_id]} 行重复")
            seen[tx_id] = line
            amount, timestamp = _amount(raw["amount"], line), _timestamp(raw["timestamp"], line)
        except ValueError as exc:
            problems.append(str(exc))
            continue
        name = raw.get("counterparty_name") or token
        parties.setdefault(token, {"counterparty_token": token, "display_name_masked": name, "type": "payment_account"})
        rows.append({"transaction_id": tx_id, "account_id": account, "direction": direction, "amount": amount, "currency": "CNY",
                     "timestamp": timestamp, "channel": "upload", "counterparty_token": token,
                     "counterparty_name_masked": name, "memo": raw.get("memo", "")})
    if problems:
        more = f"；另有 {len(problems) - MAX_REPORTED} 行同类问题" if len(problems) > MAX_REPORTED else ""
        raise ValueError(f"流水有 {len(problems)} 行无法导入：" + "；".join(problems[:MAX_REPORTED]) + more)
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
