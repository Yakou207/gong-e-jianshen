"""Deterministic, read-only checks for a single visible synthetic account."""
from __future__ import annotations

import hashlib
import json
from datetime import timedelta
from zoneinfo import ZoneInfo

from .ingest import amount_cents, parse_time, transaction_cents
from .schema import load_schema, require_schema

CALCULATOR_VERSION = "core-1.0"
TRANSACTION_FIELDS = ["transaction_id", "account_id", "direction", "amount", "currency", "timestamp", "counterparty_token"]


def _hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _time(value):
    return parse_time(value).astimezone(ZoneInfo("Asia/Shanghai"))


def _iso(value):
    return value.isoformat()


def _unique_transactions(case):
    seen = {}
    for row in case.get("transactions", []):
        key = row["transaction_id"]
        if key in seen and seen[key] != row:
            raise ValueError(f"冲突的重复交易ID:{key}")
        seen[key] = row
    return sorted(seen.values(), key=lambda r: (r["timestamp"], r["transaction_id"]))


def _transaction_set_hash(case):
    # A document/package revision must not relabel unchanged transaction evidence.
    return _hash(sorted(case.get("transactions", []), key=lambda row: row["transaction_id"]))


def resolve_entity(case, entity):
    """Only explicit IDs or confirmed mappings resolve an account; never names."""
    if entity is None:
        return {"execution_status": "completed", "counterparty_token": None, "basis": "all_visible_counterparties"}
    counterparties = case.get("counterparties", [])
    tokens = {r["counterparty_token"] for r in counterparties} | {r["counterparty_token"] for r in case.get("transactions", [])}
    targets, basis = set(), []
    if isinstance(entity, dict):
        if entity.get("counterparty_token") in tokens:
            targets.add(entity["counterparty_token"])
            basis.append("explicit_counterparty_token")
        for field in ("credit_code", "account_no_masked"):
            # A masked number is not unique identification by itself.
            if field == "account_no_masked":
                continue
            if entity.get(field):
                for row in counterparties:
                    if row.get(field) == entity[field]:
                        targets.add(row["counterparty_token"])
                        basis.append(f"explicit_{field}")
        reference = entity.get("mapping_id", entity.get("source_ref"))
    else:
        reference = entity
        if entity in tokens:
            targets.add(entity)
            basis.append("explicit_counterparty_token")
    for mapping in case.get("entity_mappings", []):
        if mapping.get("confirmed") is True and reference is not None and reference in (mapping.get("mapping_id"), mapping.get("source_ref")):
            if mapping.get("target_token") in tokens:
                targets.add(mapping["target_token"])
                basis.append({"mapping_id": mapping["mapping_id"], "revision": mapping.get("revision"), "content_hash": _hash(mapping)})
    if len(targets) != 1:
        return {"execution_status": "identity_unresolved", "counterparty_token": None,
                "reason": "对象缺少明确标识或人工确认映射，或存在相互冲突的映射", "candidates": sorted(targets)}
    return {"execution_status": "completed", "counterparty_token": next(iter(targets)), "basis": basis}


def check_coverage(case, start=None, end=None, fields=None):
    start, end = _time(start or case["coverage_start"]), _time(end or case["coverage_end"])
    fields = fields or TRANSACTION_FIELDS
    if start >= end:
        raise ValueError("查询范围必须为非空半开区间")
    records = [r for r in case.get("coverage", []) if r.get("source", "transactions") == "transactions"
               and r.get("account_id") == case["subject_account_id"]
               and _time(r["start"]) < end and _time(r["end"]) > start]
    gaps, unreliable = [], []
    for field in fields:
        intervals, missing = [], []
        for record in records:
            aliases = set(record.get("fields", []))
            if "amount_cents" in aliases:
                aliases.add("amount")
            if field not in aliases and "*" not in aliases:
                continue
            a, b = max(start, _time(record["start"])), min(end, _time(record["end"]))
            if record.get("reliable", True) is False:
                unreliable.append(field)
                missing.append((a, b))
            if record["status"] == "full" and record.get("reliable", True):
                intervals.append((a, b))
            for gap in record.get("missing_ranges", []):
                if not gap.get("fields") or field in gap["fields"] or "*" in gap["fields"]:
                    ga, gb = max(start, _time(gap["start"])), min(end, _time(gap["end"]))
                    if ga < gb:
                        missing.append((ga, gb))
        cursor = start
        for a, b in sorted(intervals):
            if a > cursor:
                gaps.append({"field": field, "start": _iso(cursor), "end": _iso(a)})
            cursor = max(cursor, b)
        if cursor < end:
            gaps.append({"field": field, "start": _iso(cursor), "end": _iso(end)})
        for a, b in missing:
            gaps.append({"field": field, "start": _iso(a), "end": _iso(b)})
    in_case = _time(case["coverage_start"]) <= start < end <= _time(case["coverage_end"])
    return {"status": "full" if not gaps and in_case else "partial" if records else "unknown",
            "reliable": not unreliable, "unreliable_fields": sorted(set(unreliable)),
            "start": _iso(start), "end": _iso(end), "fields": fields, "missing_ranges": gaps,
            "within_case": in_case, "records": records}


def query_transactions(case, query=None):
    query = query or {}
    if "transaction_id" in query and (not isinstance(query["transaction_id"], str) or not query["transaction_id"].strip()):
        raise ValueError("transaction_id须为非空字符串")
    start, end = _time(query.get("start", case["coverage_start"])), _time(query.get("end", case["coverage_end"]))
    if start >= end:
        raise ValueError("查询范围必须为非空半开区间")
    entity = resolve_entity(case, query.get("counterparty_ref", query.get("counterparty_token")))
    scope = {"case_id": case["case_id"], "account_id": case["subject_account_id"], "start": _iso(start), "end": _iso(end),
             "direction": query.get("direction"), "counterparty_ref": query.get("counterparty_ref", query.get("counterparty_token")),
             "counterparty_token": entity["counterparty_token"], "transaction_set_version": "sha256:" + _transaction_set_hash(case),
             "transaction_set_hash": _transaction_set_hash(case), "fields": sorted(query.get("fields") or TRANSACTION_FIELDS)}
    if "transaction_id" in query:
        scope["transaction_id"] = query["transaction_id"]
    coverage = check_coverage(case, _iso(start), _iso(end), query.get("fields"))
    if query.get("account_id", case["subject_account_id"]) != case["subject_account_id"]:
        raise ValueError("工具只能查询本案账户")
    if query.get("direction") not in (None, "in", "out"):
        raise ValueError("direction须为in/out")
    rows = []
    if entity["execution_status"] == "completed":
        rows = [r for r in _unique_transactions(case) if r["account_id"] == case["subject_account_id"]
                and start <= _time(r["timestamp"]) < end
                and ("transaction_id" not in query or r["transaction_id"] == query["transaction_id"])
                and (not query.get("direction") or r["direction"] == query["direction"])
                and (entity["counterparty_token"] is None or r["counterparty_token"] == entity["counterparty_token"])]
    return {"execution_status": entity["execution_status"], "rows": rows, "transaction_ids": [r["transaction_id"] for r in rows],
            "scope": scope, "query_id": f"query:{_hash(scope)[:20]}", "coverage": coverage, "identity": entity,
            "metrics": {"count": len(rows), "amount_sum_cents": sum(transaction_cents(r) for r in rows),
                        "counterparty_tokens": sorted({r["counterparty_token"] for r in rows})}
                       if entity["execution_status"] == "completed" else None}


def compute_features(case, schema=None):
    schema = require_schema(schema or load_schema())
    begin, end = _time(case["coverage_start"]), _time(case["coverage_end"])
    results = []
    for code in ("F1", "F2"):
        params, windows = schema["features"][code], []
        cursor = begin
        while cursor < end:
            if code == "F1":
                anchor = cursor.replace(hour=0, minute=0, second=0, microsecond=0)
                natural_end = anchor + timedelta(days=1)
                complete_duration = cursor == anchor and natural_end <= end
            else:
                natural_end = cursor + timedelta(days=params["window_days"])
                complete_duration = natural_end <= end
            stop = min(end, natural_end)
            query = query_transactions(case, {"start": _iso(cursor), "end": _iso(stop)})
            incoming = [r for r in query["rows"] if r["direction"] == "in"]
            outgoing = [r for r in query["rows"] if r["direction"] == "out"]
            in_sum, out_sum = sum(map(transaction_cents, incoming)), sum(map(transaction_cents, outgoing))
            in_tokens, out_tokens = len({r["counterparty_token"] for r in incoming}), len({r["counterparty_token"] for r in outgoing})
            metrics = {"in_amount_cents": in_sum, "out_amount_cents": out_sum, "in_counterparty_count": in_tokens,
                       "out_counterparty_count": out_tokens, "in_count": len(incoming), "out_count": len(outgoing),
                       "out_in_ratio": {"numerator": out_sum, "denominator": in_sum} if in_sum else None}
            status = "undeterminable"
            if complete_duration and query["coverage"]["status"] == "full":
                met = in_sum > 0 and out_sum > 0 and out_sum * params["ratio_denominator"] >= in_sum * params["ratio_numerator"]
                if code == "F2":
                    met = met and in_tokens >= params["minimum_in_counterparties"] and params["minimum_out_counterparties"] <= out_tokens <= params["maximum_out_counterparties"]
                status = "met" if met else "not_met"
            windows.append({"start": _iso(cursor), "end": _iso(stop), "complete_duration": complete_duration, "result": status,
                            "metrics": metrics, "transaction_ids": query["transaction_ids"], "query_scope": query["scope"], "coverage": query["coverage"]})
            cursor = stop
        met_count = sum(w["result"] == "met" for w in windows)
        unknown_count = sum(w["result"] == "undeterminable" for w in windows)
        threshold = params["minimum_days"] if code == "F1" else 1
        applicable = any(w["complete_duration"] for w in windows)
        status = "undeterminable" if not applicable else "met" if met_count >= threshold else "not_met" if met_count + unknown_count < threshold else "undeterminable"
        logical_key = f"feature:{case['case_id']}:{code}:{_iso(begin)}:{_iso(end)}"
        result = {"logical_key": logical_key, "feature_code": code, "execution_status": "completed", "result": status,
                  "parameters": params, "schema_version": schema["schema_version"], "calculator_version": CALCULATOR_VERSION,
                  "windows": windows, "metrics": {"met_windows": met_count, "undeterminable_windows": unknown_count, "total_windows": len(windows)},
                  "evidence": [{"type": "query_scope", "scope": w["query_scope"], "coverage": w["coverage"], "transaction_ids": w["transaction_ids"]} for w in windows]}
        revision = _hash(result)
        result.update({"revision": revision, "feature_result_id": f"{logical_key}:{revision[:12]}"})
        results.append(result)
    return results


def validate_span(case, source, text=None):
    if not isinstance(source, dict):
        return False
    doc = next((d for d in case.get("documents", []) if d["document_id"] == source.get("document_id") and d["revision"] == source.get("revision")), None)
    span = source.get("span")
    if isinstance(span, dict):
        span = [span.get("start"), span.get("end")]
    if doc is None or not isinstance(span, (list, tuple)) or len(span) != 2:
        return False
    a, b = span
    return (type(a) is int and type(b) is int and 0 <= a < b <= len(doc["text"])
            and (text is None or doc["text"][a:b] == text))


def _numeric_decision(actual, expected, operator, full):
    if operator in ("exact", "only"):
        return "contradicted" if actual > expected or (full and actual != expected) else "supported" if full else "insufficient_evidence"
    if operator == "at_least":
        return "supported" if actual >= expected else "contradicted" if full else "insufficient_evidence"
    if operator == "at_most":
        return "contradicted" if actual > expected else "supported" if full else "insufficient_evidence"
    if operator == "exists":
        return "supported" if actual > 0 else "contradicted" if full else "insufficient_evidence"
    if operator == "none":
        return "contradicted" if actual > 0 else "supported" if full else "insufficient_evidence"
    raise ValueError(f"未知限定词:{operator}")


def verify_claim(case, claim, schema=None):
    schema = require_schema(schema or load_schema())
    result = {"claim_id": claim.get("claim_id"), "execution_status": "completed", "result": "insufficient_evidence",
              "schema_version": schema["schema_version"], "evidence": []}
    if not validate_span(case, claim.get("source"), claim.get("text", claim.get("original_text"))):
        return {**result, "execution_status": "extraction_failed", "reason": "原文文档修订或跨度不匹配"}
    try:
        if claim.get("currency", "CNY") != "CNY" or claim.get("account_id", case["subject_account_id"]) != case["subject_account_id"]:
            raise ValueError("陈述对象或币种不在本案范围")
        required_fields = ["transaction_id", "account_id", "timestamp"]
        if claim.get("direction"):
            required_fields.append("direction")
        if claim.get("counterparty_ref") or claim.get("counterparty_token") or claim.get("kind") == "counterparty":
            required_fields.append("counterparty_token")
        if claim.get("kind") == "amount_sum":
            required_fields.extend(["amount", "currency"])
        query_args = {k: claim[k] for k in ("start", "end", "direction", "counterparty_ref", "counterparty_token", "account_id") if k in claim}
        query = query_transactions(case, {**query_args, "fields": required_fields})
        result.update({"execution_status": query["execution_status"], "coverage": query["coverage"], "observed": query["metrics"], "identity": query["identity"],
                       "evidence": [{"type": "document_span", **claim["source"]}, {"type": "query_scope", "scope": query["scope"], "coverage": query["coverage"], "transaction_ids": query["transaction_ids"]}]})
        if query["execution_status"] != "completed":
            result["observed"] = None
            result["reason"] = "对象或查询未完成，空返回不代表零笔交易"
            return result
        if not query["coverage"]["reliable"]:
            return {**result, "reason": "用于判断的字段可靠性未满足要求"}
        full = query["coverage"]["status"] == "full"
        kind, operator = claim["kind"], claim.get("operator", "exact")
        if kind in ("count", "amount_sum"):
            if operator in ("none", "exists"):
                expected = 0
            elif kind == "count":
                expected = claim["value"]
                if type(expected) is not int or expected < 0:
                    raise ValueError("次数须为非负整数")
            elif "value_cents" in claim:
                expected = claim["value_cents"]
                if type(expected) is not int or expected < 0:
                    raise ValueError("金额分须为非负整数")
            else:
                expected = amount_cents(claim["value"])
            actual = query["metrics"]["count" if kind == "count" else "amount_sum_cents"]
            result["result"] = _numeric_decision(actual, expected, operator, full)
            result["comparison"] = {"actual": actual, "expected": expected, "operator": operator, "unit": "transactions" if kind == "count" else "CNY_fen"}
        elif kind == "counterparty":
            values = claim["value"] if isinstance(claim["value"], list) else [claim["value"]]
            entities = [resolve_entity(case, value) for value in values]
            if any(e["execution_status"] != "completed" or e["counterparty_token"] is None for e in entities):
                return {**result, "execution_status": "identity_unresolved", "reason": "陈述对手账户未能精确对齐"}
            expected, actual = {e["counterparty_token"] for e in entities}, set(query["metrics"]["counterparty_tokens"])
            if operator in ("exact", "only"):
                contradicted = bool(actual - expected) or (full and operator == "exact" and actual != expected)
                result["result"] = "contradicted" if contradicted else "supported" if full else "insufficient_evidence"
            elif operator in ("exists", "none"):
                observed = bool(actual & expected)
                result["result"] = _numeric_decision(int(observed), 0, operator, full)
            else:
                raise ValueError("对手核验仅支持exact/only/exists/none")
            result["comparison"] = {"actual": sorted(actual), "expected": sorted(expected), "operator": operator}
        elif kind == "time_range":
            a, b = _time(claim["value"]["start"]), _time(claim["value"]["end"])
            if a >= b:
                raise ValueError("陈述时间范围为空")
            inside = [r for r in query["rows"] if a <= _time(r["timestamp"]) < b]
            if operator in ("exact", "only"):
                result["result"] = "contradicted" if len(inside) != len(query["rows"]) else "supported" if full else "insufficient_evidence"
            elif operator in ("exists", "none"):
                result["result"] = _numeric_decision(len(inside), 0, operator, full)
            else:
                raise ValueError("时间范围核验仅支持exact/only/exists/none")
            result["comparison"] = {"inside_count": len(inside), "outside_count": len(query["rows"]) - len(inside), "operator": operator,
                                    "asserted_start": _iso(a), "asserted_end": _iso(b), "relation": "transaction_timestamps_within_interval"}
        else:
            raise ValueError(f"未知事实类型:{kind}")
    except (ValueError, TypeError, KeyError, AttributeError) as exc:
        result.update({"execution_status": "failed", "result": "insufficient_evidence", "reason": str(exc)})
    return result


def check_materials(case, links=None, schema=None):
    schema, results = require_schema(schema or load_schema()), []
    transactions = {r["transaction_id"]: r for r in _unique_transactions(case)}
    material_set = sorted(case.get("materials", []), key=lambda material: material["material_id"])
    materials = {(m["material_id"], m["revision"]): m for m in material_set}
    for link in case.get("material_links", []) if links is None else links:
        result = {"link_id": link.get("link_id"), "claim_or_issue_id": link.get("claim_or_issue_id"), "material_id": link.get("material_id"),
                  "execution_status": "completed", "result": "insufficient", "checks": [], "evidence": [], "schema_version": schema["schema_version"],
                  "boundary": schema.get("material_boundary")}
        result["evidence"] = [
            {"type": "material_set", "content_hash": _hash(material_set),
             "members": [{"material_id": m["material_id"], "revision": m["revision"]} for m in material_set]},
            {"type": "material_link", "link_id": link.get("link_id"), "content_hash": _hash(link),
             "relation_template": link.get("relation_template")},
            {"type": "transaction_set", "content_hash": _transaction_set_hash(case), "transaction_set_version": "sha256:" + _transaction_set_hash(case),
             "requested_transaction_ids": link.get("transaction_ids", [])}]

        template = schema["material_templates"].get(link.get("relation_template"))
        material = materials.get((link.get("material_id"), link.get("revision", link.get("material_revision"))))
        selected = [transactions[tid] for tid in dict.fromkeys(link.get("transaction_ids", [])) if tid in transactions]
        if not template:
            result.update({"result": "pending_judgement", "reason": "尚未选择适用的材料关系模板"})
        elif not material:
            result["reason"] = "未提供所引用修订的材料"
        elif not selected or len(selected) != len(set(link.get("transaction_ids", []))):
            result["reason"] = "拟支持的交易集合为空或引用缺失"
        elif link.get("schema_version", schema["schema_version"]) != schema["schema_version"]:
            result.update({"result": "pending_judgement", "reason": "材料关系绑定的规范版本不一致"})
        else:
            checks = result["checks"]
            def check(name, actual, expected, matched):
                checks.append({"field": name, "actual": actual, "expected": expected, "result": "missing" if matched is None else "match" if matched else "mismatch"})
            try:
                subject, counterparty = material.get("subject", {}), material.get("counterparty", {})
                check("subject.account_id", subject.get("account_id"), case["subject_account_id"], subject.get("account_id") == case["subject_account_id"] if subject.get("account_id") else None)
                check("subject.role", subject.get("role"), template["subject_role"], subject.get("role") == template["subject_role"] if subject.get("role") else None)
                check("counterparty.role", counterparty.get("role"), template["counterparty_role"], counterparty.get("role") == template["counterparty_role"] if counterparty.get("role") else None)
                entity = resolve_entity(case, counterparty)
                tokens = sorted({r["counterparty_token"] for r in selected})
                check("counterparty.counterparty_token", tokens, entity["counterparty_token"], tokens == [entity["counterparty_token"]] if entity["execution_status"] == "completed" else None)
                check("direction", sorted({r["direction"] for r in selected}), template["direction"], all(r["direction"] == template["direction"] for r in selected))
                check("currency", material.get("currency"), "CNY", material.get("currency") == "CNY" if material.get("currency") else None)
                if template.get("transaction_count"):
                    check("transaction_count", len(selected), template["transaction_count"], len(selected) == template["transaction_count"])
                period = material.get("period", {})
                if period.get("start") and period.get("end"):
                    a, b = _time(period["start"]), _time(period["end"])
                    if a >= b:
                        raise ValueError("材料期间必须为非空半开区间")
                    check("period", [r["timestamp"] for r in selected], period, all(a <= _time(r["timestamp"]) < b for r in selected))
                else:
                    check("period", None, "明确的材料适用期间", None)
                actual = sum(transaction_cents(r) for r in selected)
                if "amount" in material or "amount_cents" in material:
                    expected = transaction_cents(material)
                    tolerance = template.get("amount_tolerance_cents", 0)
                    matched = abs(actual - expected) <= tolerance if template["amount_relation"] == "equal" else actual <= expected + tolerance
                    check("amount_cents", actual, {"value": expected, "relation": template["amount_relation"], "tolerance_cents": tolerance}, matched)
                else:
                    check("amount_cents", actual, None, None)
                result["result"] = "mismatch" if any(c["result"] == "mismatch" for c in checks) else "insufficient" if any(c["result"] == "missing" for c in checks) else "corresponds"
                material_paths = []
                for name in ("subject.account_id", "subject.role", "counterparty.role", "counterparty.counterparty_token", "counterparty.credit_code", "period.start", "period.end", "amount", "amount_cents", "currency"):
                    cursor = material
                    for component in name.split("."):
                        cursor = cursor.get(component) if isinstance(cursor, dict) else None
                    if cursor is not None:
                        material_paths.append(name)
                result["evidence"].append({"type": "material", "material_id": material["material_id"], "revision": material["revision"],
                                           "field_paths": material_paths, "content_hash": _hash(material)})
                result["evidence"].append({"type": "transactions", "transaction_ids": [r["transaction_id"] for r in selected],
                                           "transaction_fields": {r["transaction_id"]: [f for f in ("account_id", "direction", "currency", "counterparty_token", "timestamp", "amount", "amount_cents") if f in r] for r in selected},
                                           "transaction_set_version": "sha256:" + _transaction_set_hash(case)})
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                result.update({"execution_status": "failed", "result": "insufficient", "reason": str(exc)})
        results.append(result)
    return results
