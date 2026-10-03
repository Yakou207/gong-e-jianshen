"""Read raw model results for offline scoring; never compute business answers.

Pointers identify rows only within an outer hash-bound raw artifact. Evidence
resolution checks source existence/version, not whether a claim is supported.
"""
from copy import deepcopy
import hashlib
import json

from .schema import LABEL_VALUES


CLAIM_LABELS = {"count", "amount_sum", "counterparty", "time_range"}
SCOPE_FIELDS = ("account_id", "start", "end", "direction", "counterparty_ref", "counterparty_token")
REFERENCE_FIELDS = {
    "document_span": {"document_id", "revision", "span", "quote", "text", "content_hash"},
    "transaction": {"transaction_id", "revision", "content_hash", "field_paths", "fields", "transaction_fields", "transaction_set_version"},
    "transactions": {"transaction_ids", "fields", "transaction_fields", "transaction_set_version"},
    "material": {"material_id", "revision", "field_paths", "content_hash"},
    "material_fields": {"material_id", "revision", "field_paths", "content_hash"},
    "alert_focus": {"focus_id", "revision", "content_hash", "quote", "text"},
    "coverage": {"coverage_id", "revision", "content_hash", "account_id", "source", "start", "end", "status", "fields"},
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _rows(value, name):
    _require(isinstance(value, list) and all(isinstance(row, dict) for row in value), name + " must be a list of objects")
    return value


def _evidence(value):
    rows = _rows(value, "evidence")
    _require(all(_text(row.get("type")) for row in rows), "Evidence type is required")
    return deepcopy(rows)


def _claim_anchor(value):
    _require(isinstance(value, dict) and _text(value.get("document_id"))
             and (isinstance(value.get("revision"), str) or type(value.get("revision")) is int)
             and isinstance(value.get("span"), list) and len(value["span"]) == 2
             and all(type(n) is int for n in value["span"]), "Claim needs a document revision and two integer span offsets")


def _guard(raw, method):
    _require(isinstance(raw, dict), "Raw result must be an object")
    _require(method in ("B0", "Fixed", "Agent"), "Unknown evaluation method")
    def inspect(value):
        if isinstance(value, dict):
            _require("deliverable" not in value, "Use original model output, not a deliverable export")
            _require(value.get("origin") != "human_reviewed", "Human-reviewed results cannot be scored as autonomous predictions")
            if "claim_amendments" in value:
                _require(value["claim_amendments"] == [], "Human amendments are not autonomous input")
            for child in value.values():
                inspect(child)
        elif isinstance(value, list):
            for child in value:
                inspect(child)
    inspect(raw)


def _engine(raw):
    rows = {key: _rows(raw.get(key), key) for key in
            ("features", "claims", "claim_results", "material_results", "semantic_results")}
    claims = {}
    for claim in rows["claims"]:
        cid = claim.get("claim_id")
        _require(_text(cid) and cid not in claims, "Missing or duplicate local claim_id")
        _require(_text(claim.get("kind")) and claim["kind"] in CLAIM_LABELS, "Unknown Claim kind")
        _claim_anchor(claim.get("source"))
        _require(_text(claim.get("operator")) and "value" in claim, "Claim proposition is incomplete")
        claims[cid] = claim
    for result in rows["claim_results"]:
        cid = result.get("claim_id")
        _require(_text(cid) and cid in claims, "Claim result has no unique local Claim")
    return rows, claims


def _observation(pointer, label, value, state, scope, anchor, proposition, evidence):
    _require(_text(label) and label in LABEL_VALUES, "Unknown prediction label")
    _require((isinstance(value, str) and value in LABEL_VALUES[label]) or (state == "unfinished" and value is None), "Invalid prediction value")
    _require(isinstance(state, str) and state in {"completed", "unfinished", "failed", "extraction_failed", "identity_unresolved", "pending", "not_run"},
             "Unknown or missing execution state")
    _require(all(isinstance(item, dict) for item in (scope, anchor, proposition)), "Scope, anchor, and proposition must be objects")
    return {"observation_id": pointer, "raw_pointer": pointer, "label": label,
        "prediction_value": value, "execution_state": state, "object_scope": deepcopy(scope),
        "matching_anchor": deepcopy(anchor), "proposition": deepcopy(proposition), "evidence": _evidence(evidence)}


def normalize_observations(raw, method, case):
    _guard(raw, method)
    _require(isinstance(case, dict), "Case must be an object")
    _guard(case, method)
    _require(_text(raw.get("case_id")) and raw["case_id"] == case.get("case_id"), "Raw result case_id differs from frozen case")
    if method == "B0":
        return _b0_observations(raw)
    rows, claims = _engine(raw)
    base = {"account_id": case.get("subject_account_id"), "start": case.get("coverage_start"), "end": case.get("coverage_end")}
    links = {}
    for link in _rows(case.get("material_links", []), "material_links"):
        lid = link.get("link_id")
        _require(_text(lid) and lid not in links, "Missing or duplicate MaterialLink ID")
        links[lid] = link
    output = []
    for index, row in enumerate(rows["features"]):
        label = row.get("feature_code")
        _require(_text(label) and label in {"F1", "F2"}, "Invalid feature code")
        output.append(_observation(f"/features/{index}", label, row.get("result"), row.get("execution_status"),
            row.get("object", base), {}, {"parameters": deepcopy(row.get("parameters", {}))}, row.get("evidence")))
    for index, row in enumerate(rows["claim_results"]):
        claim = claims[row["claim_id"]]
        scope = {**base, **{key: deepcopy(claim[key]) for key in SCOPE_FIELDS if key in claim}}
        proposition = {key: deepcopy(value) for key, value in claim.items()
                       if key not in {"claim_id", "source", "text", "origin", *SCOPE_FIELDS}}
        output.append(_observation(f"/claim_results/{index}", claim["kind"], row.get("result"), row.get("execution_status"),
            scope, claim["source"], proposition, row.get("evidence")))
    for index, row in enumerate(rows["material_results"]):
        lid = row.get("link_id")
        _require(_text(lid) and lid in links, "Material result has no frozen MaterialLink")
        link = links[lid]
        scope = {**base, "material_link_id": lid, "material_id": link.get("material_id"),
                 "transaction_ids": deepcopy(link.get("transaction_ids")), "relation_template": link.get("relation_template")}
        output.append(_observation(f"/material_results/{index}", "material_relation", row.get("result"), row.get("execution_status"),
            scope, {"material_link_id": lid}, {}, row.get("evidence")))
    for index, row in enumerate(rows["semantic_results"]):
        focus = row.get("focus_id")
        _require(_text(focus), "Semantic result focus_id is required")
        scope = row.get("object", base)
        _require(isinstance(scope, dict), "Semantic object scope must be an object")
        output.append(_observation(f"/semantic_results/{index}", "alert_response", row.get("status"), row.get("execution_status"),
            {**scope, "focus_id": focus}, {"focus_id": focus}, {}, row.get("evidence")))
    return output


def _b0_parsed(raw):
    _require("parsed_output" in raw, "B0 parsed_output is required")
    parsed = raw["parsed_output"]
    if parsed is None:
        _require(raw.get("run_status") == "failed", "Absent B0 parsed output requires failed execution")
        return None
    _require(isinstance(parsed, dict), "B0 parsed_output must be an object")
    checks = _rows(parsed.get("checks"), "B0 checks")
    seen = set()
    for row in checks:
        cid = row.get("check_id")
        _require(_text(cid) and cid not in seen, "Missing or duplicate B0 check_id")
        seen.add(cid)
    return parsed


def _b0_observations(raw):
    parsed = _b0_parsed(raw)
    if parsed is None:
        return []
    output = []
    for index, row in enumerate(parsed["checks"]):
        anchor = row.get("anchor")
        _require(isinstance(anchor, dict), "B0 anchor must be an object")
        proposition = anchor.get("claim")
        _require(proposition is None or isinstance(proposition, dict), "B0 Claim proposition must be an object")
        _require(row.get("status") in ("completed", "unfinished"), "Invalid B0 check status")
        _require(row.get("status") != "unfinished" or row.get("value") is None, "Unfinished B0 value must stay null")
        if _text(row.get("label")) and row["label"] in CLAIM_LABELS:
            _require(isinstance(proposition, dict) and bool(proposition), "B0 Claim needs its original structured proposition")
            _claim_anchor(anchor)
        output.append(_observation(f"/parsed_output/checks/{index}", row.get("label"), row.get("value"), row.get("status"),
            row.get("object_scope"), {key: deepcopy(value) for key, value in anchor.items() if key != "claim"},
            proposition or {}, row.get("evidence")))
    return output


def normalize_issue_predictions(raw, method):
    """Keep issue rows separate; local refs are not reference-answer IDs."""
    _guard(raw, method)
    targets = {}
    if method == "B0":
        parsed = _b0_parsed(raw)
        if parsed is None:
            return []
        _b0_observations(raw)
        targets = {row["check_id"]: [f"/parsed_output/checks/{i}"] for i, row in enumerate(parsed["checks"])}
        issues, prefix = _rows(parsed.get("issues"), "B0 issues"), "/parsed_output/issues/"
    else:
        rows, _ = _engine(raw)
        for collection, field, stem in (("features", "feature_code", "feature:"), ("claim_results", "claim_id", "claim:"),
                                       ("material_results", "link_id", "materials:"), ("semantic_results", "focus_id", "semantic:")):
            for index, row in enumerate(rows[collection]):
                key = row.get(field)
                _require(_text(key), "Missing issue-target local identifier")
                targets.setdefault(stem + key, []).append(f"/{collection}/{index}")
        issues, prefix = _rows(raw.get("issues"), "issues"), "/issues/"
    output, ids = [], set()
    for index, issue in enumerate(issues):
        iid, kind = issue.get("issue_id"), issue.get("type")
        _require(_text(iid) and iid not in ids and _text(kind), "Missing/duplicate local issue ID or issue type")
        ids.add(iid)
        if method == "B0":
            refs = issue.get("check_ids")
            _require(isinstance(refs, list) and all(_text(value) and value in targets for value in refs), "B0 issue has invalid local check references")
        else:
            _require(_text(issue.get("target_id")), "Issue target_id is required")
            refs = [issue["target_id"]]
        pointer = prefix + str(index)
        output.append({"prediction_id": pointer, "raw_pointer": pointer, "type": kind,
            "local_check_refs": deepcopy(refs), "observations": list(dict.fromkeys(obs for ref in refs for obs in targets.get(ref, []))),
            "evidence": _evidence(issue.get("evidence"))})
    return output


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _unique(case, collection, identifier):
    indexed = {}
    for row in _rows(case.get(collection, []), collection):
        key = row.get(identifier)
        _require(_text(key) and key not in indexed, "Missing or duplicate source " + identifier)
        indexed[key] = row
    return indexed


def _fields_exist(row, paths):
    if not isinstance(paths, list) or not paths or any(not _text(path) for path in paths):
        return False
    for path in paths:
        value = row
        for part in path.split("."):
            if not isinstance(value, dict) or part not in value:
                return False
            value = value[part]
    return True


def _record_matches(ref, record):
    if "revision" in ref and not _revision_equal(ref["revision"], record.get("revision")):
        return False
    if "content_hash" in ref and ref["content_hash"] != _digest(record):
        return False
    return True


def _revision_equal(left, right):
    return type(left) in (str, int) and type(left) is type(right) and left == right


def _lookup(indexed, value):
    return indexed.get(value) if _text(value) else None


def resolve_evidence(ref, case):
    """True=source resolves, False=invalid, None=unsupported; no entailment claim."""
    _require(isinstance(ref, dict) and _text(ref.get("type")), "Evidence must be a typed object")
    _require(isinstance(case, dict), "Case must be an object")
    kind = ref["type"]
    if kind not in REFERENCE_FIELDS or set(ref) - REFERENCE_FIELDS[kind] - {"type"}:
        return None
    if kind == "document_span":
        doc = _lookup(_unique(case, "documents", "document_id"), ref.get("document_id"))
        span = ref.get("span")
        if not doc or "revision" not in ref or not _record_matches(ref, doc) or not isinstance(doc.get("text"), str):
            return False
        if not isinstance(span, list) or len(span) != 2 or any(type(n) is not int for n in span) or not 0 <= span[0] < span[1] <= len(doc["text"]):
            return False
        quoted = doc["text"][span[0]:span[1]]
        return all(isinstance(ref[key], str) and ref[key] == quoted for key in ("quote", "text") if key in ref)
    if kind in {"transaction", "transactions"}:
        tx = _unique(case, "transactions", "transaction_id")
        ids = [ref.get("transaction_id")] if kind == "transaction" else ref.get("transaction_ids")
        if not isinstance(ids, list) or not ids or any(not _text(x) for x in ids) or len(ids) != len(set(ids)) or any(x not in tx for x in ids):
            return False
        if "transaction_set_version" in ref and ref["transaction_set_version"] != "sha256:" + _digest(sorted(tx.values(), key=lambda r: r["transaction_id"])):
            return False
        if kind == "transaction" and (not _record_matches(ref, tx[ids[0]]) or
                ("field_paths" in ref and not _fields_exist(tx[ids[0]], ref["field_paths"]))):
            return False
        if "transaction_fields" in ref:
            fields = ref["transaction_fields"]
            if not isinstance(fields, dict) or set(fields) != set(ids) or any(not _fields_exist(tx[key], value) for key, value in fields.items()):
                return False
        if "fields" in ref and any(not _fields_exist(tx[key], ref["fields"]) for key in ids):
            return False
        return True
    if kind in {"material", "material_fields"}:
        material = _lookup(_unique(case, "materials", "material_id"), ref.get("material_id"))
        if not material or "revision" not in ref or not _record_matches(ref, material):
            return False
        return _fields_exist(material, ref.get("field_paths")) if kind == "material_fields" or "field_paths" in ref else True
    if kind == "alert_focus":
        alert = case.get("alert")
        if not isinstance(alert, dict) or not _revision_equal(ref.get("revision"), alert.get("revision")):
            return False
        focuses = _rows(alert.get("focuses", []), "alert.focuses")
        if not focuses and _text(alert.get("original_focus")):
            focuses = [{"focus_id": alert.get("alert_id", "original-focus"), "text": alert["original_focus"]}]
        matches = [row for row in focuses if row.get("focus_id") == ref.get("focus_id")]
        return len(matches) == 1 and ("content_hash" not in ref or ref["content_hash"] == _digest(matches[0])) and all(
            isinstance(ref[k], str) and ref[k] == matches[0].get("text") for k in ("text", "quote") if k in ref)
    if kind == "coverage":
        row = _lookup(_unique(case, "coverage", "coverage_id"), ref.get("coverage_id"))
        if not row or "revision" not in ref or not _record_matches(ref, row):
            return False
        return all(ref[key] == row.get(key) for key in ("account_id", "source", "start", "end", "status", "fields") if key in ref)
    return None
