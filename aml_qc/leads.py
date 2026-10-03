"""Bind model suggestions to inputs actually visible in the current review.

Binding a suggestion to evidence does not verify its meaning or novelty.
Human dispositions are stored separately from immutable machine candidates.
"""
from copy import deepcopy

from .depgraph import digest, sources_for
from .llm import ModelError
from .schema import load_schema


def lead_context_hash(case):
    value = deepcopy(case)
    for collection, key in (("transactions", "transaction_id"), ("documents", "document_id"), ("materials", "material_id")):
        value[collection] = sorted({r[key]: r for r in value.get(collection, [])}.values(), key=lambda r: r[key])
    scope = value.setdefault("review_scope", {})
    scope.pop("upgraded_leads", None)
    scope.pop("lead_dispositions", None)
    sources = sources_for(value, value.get("schema") or load_schema(), {})
    return digest({key: row["value"] for key, row in sources.items() if key != "source:execution"})


def lead_basis(case, checks, tool_trace=()):
    """Only checked results, supplied documents, and successful read tools qualify."""
    basis = {}

    def add(ref, kind, value, evidence):
        basis[ref] = {"basis_ref": ref, "kind": kind, "content_hash": digest(value),
                      "evidence": deepcopy(evidence)}

    for document in case.get("documents", []):
        if document.get("text"):
            add("document:" + document["document_id"], "document", document,
                [{"type": "document_span", "document_id": document["document_id"],
                  "revision": document["revision"], "span": [0, len(document["text"])], "text": document["text"]}])
    for collection, identity, prefix in (("features", "feature_code", "feature"),
                                         ("claim_results", "claim_id", "claim"),
                                         ("material_results", "link_id", "material")):
        for result in checks.get(collection, []):
            if result.get(identity) and result.get("execution_status", "completed") == "completed" and result.get("evidence"):
                add(prefix + ":" + result[identity], prefix, result, result["evidence"])
    for entry in tool_trace:
        result = entry.get("result", {})
        if entry["tool"] == "read_schema":
            # Normative definitions are not observations about this case.
            continue
        if entry.get("status") != "completed" or not entry.get("result_ref") or result.get("execution_status", "completed") != "completed":
            continue
        if (entry["tool"] == "read_document" and not result.get("document")) or (entry["tool"] == "read_material" and not result.get("material")):
            continue
        ref = entry["result_ref"]
        evidence = [{"type": "tool_result", "result_ref": ref, "tool": entry["tool"],
                     "arguments": deepcopy(entry["arguments"]), "content_hash": digest(result), "result": deepcopy(result)}]
        if entry["tool"] == "query_transactions":
            evidence.append({"type": "query_scope", "scope": result["scope"], "coverage": result["coverage"],
                             "transaction_ids": result["transaction_ids"]})
        elif entry["tool"] == "compute_features":
            for feature in result["features"]:
                evidence.extend(deepcopy(feature["evidence"]))
        add(ref, "read_tool", {"tool": entry["tool"], "arguments": entry["arguments"], "result": result}, evidence)
    return basis


def normalize_leads(case, suggestions, basis, execution=None):
    candidates = []
    for suggestion in suggestions:
        question, observation = suggestion["question"].strip(), suggestion["observation"].strip()
        refs = suggestion["basis_refs"]
        if not question or not observation or len(set(refs)) != len(refs) or any(ref not in basis for ref in refs):
            raise ModelError("新线索缺少明确问题或引用了未提供、失败或不存在的当前依据")
        # Exact duplicates of existing obligations must not become optional notices.
        alert = case.get("alert") or {}
        existing = [f.get("text", "").strip() for f in alert.get("focuses", [])]
        existing += [alert.get("original_focus", "").strip()]
        lead_id = "lead-" + digest({"question": question, "observation": observation, "basis_refs": sorted(refs)})[:24]
        repeated_upgrade = any(row.get("text", "").strip() == question and row.get("lead_id") != lead_id
                               for row in case.get("review_scope", {}).get("upgraded_leads", []))
        if question in existing or repeated_upgrade:
            raise ModelError("原预警关注点不能包装成可选新线索")
        records = [{k: basis[ref][k] for k in ("basis_ref", "kind", "content_hash")} for ref in sorted(refs)]
        evidence = [item for ref in sorted(refs) for item in basis[ref]["evidence"]]
        evidence = list({digest(item): item for item in evidence}.values())
        candidate = {"lead_id": lead_id, "question": question, "observation": observation,
                     "basis_refs": sorted(refs), "basis_records": records, "basis_fingerprint": digest(records),
                     "context_hash": lead_context_hash(case), "execution_fingerprint": digest(execution or {}), "evidence": evidence,
                     "origin": "model_candidate", "evidence_binding_valid": True, "novelty_status": "unconfirmed",
                     "model_draft": {"observation": suggestion["observation"], "question": suggestion["question"],
                                     "status": "unverified_model_suggestion"}}
        if any(row["lead_id"] == lead_id for row in candidates):
            raise ModelError("模型重复提出同一新线索")
        candidates.append(candidate)
    return candidates


def lead_has_disposition(case, candidate):
    """An upgraded obligation stays mandatory; a closed notice is context-bound."""
    scope = case.get("review_scope", {})
    if any(row.get("lead_id") == candidate["lead_id"] for row in scope.get("upgraded_leads", [])):
        return True
    record = next((row for row in reversed(scope.get("lead_dispositions", []))
                   if row.get("lead_id") == candidate["lead_id"]), None)
    return bool(record and record.get("status") == "closed" and record.get("context_hash") == candidate["context_hash"]
                and record.get("basis_fingerprint") == candidate["basis_fingerprint"]
                and record.get("execution_fingerprint") == candidate["execution_fingerprint"])
