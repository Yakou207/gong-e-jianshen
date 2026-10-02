"""Content-addressed, conservative snapshot dependency evaluation.

Edges point from a derived node to its dependencies. Full runs never consult
old nodes. Empty collections are real sources, so additions invalidate queries.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from time import perf_counter


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return sha256(canonical(value).encode()).hexdigest()


def sources_for(case, schema, execution):
    collections = {"transactions", "documents", "materials", "material_links", "entity_mappings",
                   "counterparties", "coverage", "alert", "review_scope", "schema", "claims"}
    meta = {k: v for k, v in case.items() if k not in collections | {"data_version", "title"}}
    sources = {
        "metadata": meta, "coverage": case.get("coverage", []),
        "transactions": sorted(case.get("transactions", []), key=lambda x: x["transaction_id"]),
        "documents": sorted(case.get("documents", []), key=lambda x: x["document_id"]),
        "materials": sorted(case.get("materials", []), key=lambda x: x["material_id"]),
        "material_links": case.get("material_links", []),
        "entities": {"mappings": case.get("entity_mappings", []), "counterparties": case.get("counterparties", [])},
        "alert": case.get("alert"), "review_scope": case.get("review_scope", {}),
        "schema": schema, "execution": execution,
    }
    for collection, id_field in [("documents", "document_id"), ("materials", "material_id")]:
        for item in case.get(collection, []):
            sources[f"{collection}:{item[id_field]}"] = item
    return {f"source:{key}": {"hash": digest(value), "value": value} for key, value in sources.items()}


def affected_nodes(old_snapshot, current_sources):
    old_sources = old_snapshot.get("sources", {})
    changed = {key for key in old_sources.keys() | current_sources.keys()
               if old_sources.get(key, {}).get("hash") != current_sources.get(key, {}).get("hash")}
    affected = set(changed)
    while True:
        found = {key for key, node in old_snapshot.get("nodes", {}).items()
                 if set(node["dependencies"]) & affected}
        new = found - affected
        if not new:
            break
        affected.update(new)
    return sorted(changed), sorted(affected - changed)


class Evaluator:
    def __init__(self, sources, previous=None, strategy="full"):
        if strategy not in {"full", "incremental"}:
            raise ValueError("未知重查策略")
        self.sources = sources
        self.nodes = {}
        self.trace = []
        self.strategy = strategy
        # Independent full evaluation gets neither old graph nor old cache.
        self.previous = previous if strategy == "incremental" and previous else {}
        self.changed, self.affected = affected_nodes(self.previous, sources)
        self.recomputed = self.reused = 0

    def evaluate(self, key, kind, parameters, dependencies, compute):
        dependencies = sorted(set(dependencies))
        refs = []
        for dep in dependencies:
            item = self.sources.get(dep) or self.nodes.get(dep)
            if item is None:
                raise ValueError(f"未登记的依赖:{dep}")
            refs.append([dep, item["hash"]])
        fingerprint = digest({"implementation": "engine-1", "kind": kind,
                              "parameters": parameters, "dependencies": refs})
        old = self.previous.get("nodes", {}).get(key)
        start = perf_counter()
        if old and old["fingerprint"] == fingerprint:
            result = deepcopy(old["result"])
            status = "reused"
            self.reused += 1
        else:
            result = compute()
            status = "computed"
            self.recomputed += 1
        self.nodes[key] = {"kind": kind, "parameters": parameters, "dependencies": dependencies,
                           "fingerprint": fingerprint, "hash": digest(result), "result": result}
        self.trace.append({"tool": kind, "arguments": parameters, "result_ref": key,
                           "result": result, "status": status,
                           "duration_ms": round((perf_counter() - start) * 1000, 3),
                           "purpose": "按已登记来源核验；复用必须满足完整输入指纹一致"})
        return result

    def snapshot(self):
        return {"sources": self.sources, "nodes": self.nodes}

    def stats(self):
        return {"potentially_affected": len(self.affected), "recomputed": self.recomputed,
                "reused": self.reused, "changed_sources": self.changed}


def business_result(run):
    """Fields compared by independent full/incremental mechanism tests."""
    return {k: run.get(k) for k in ["features", "claims", "claim_results", "material_results", "semantic_results", "lead_candidates",
                                  "issues", "open_items", "required_checks", "qc_recommendation", "run_status"]}
