"""Fixed and bounded model-driven evidence review over the same read-only tools."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import re
from time import perf_counter
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from . import core
from .claim_edits import merge_claim_amendments
from .depgraph import Evaluator, canonical, digest, sources_for
from .contracts import ClaimOutput, ExtractionOutput, SemanticOutput, contract_schemas
from .ingest import validate_case
from .llm import DeepSeek, ModelError, json_answer
from .leads import lead_basis, lead_has_disposition, normalize_leads

ROOT = Path(__file__).resolve().parents[1]
VERSION = "workflow-2.0"
PROMPT_VERSION = "claims-response-2.6"
PROMPT_ROOT = ROOT / "config/prompts/v2.6"
FIXED_POLICY = {"version": "fixed-visible-scope-1", "steps": ["check_coverage", "query_transactions"],
                "scope": "entire visible case interval, all directions and counterparties",
                "applies_when": "model semantic review is required"}
BASE_SYSTEM = (PROMPT_ROOT / "base.txt").read_text().strip()
EXTRACT_SYSTEM = BASE_SYSTEM + "\n" + (PROMPT_ROOT / "extraction.txt").read_text().strip()
SEMANTIC_SYSTEM = BASE_SYSTEM + "\n" + (PROMPT_ROOT / "semantic.txt").read_text().strip()
AGENT_SYSTEM = BASE_SYSTEM + "\n" + (PROMPT_ROOT / "agent.txt").read_text().strip() + "\n最终答复的格式与判断边界：\n" + (PROMPT_ROOT / "semantic.txt").read_text().strip()
IMPLEMENTATION_HASH = digest({name: (ROOT / "aml_qc" / name).read_text() for name in
                              ["core.py", "ingest.py", "schema.py", "depgraph.py", "workflow.py", "llm.py", "contracts.py",
                               "annotations.py", "store.py", "exports.py", "api.py", "migrations.py", "leads.py", "claim_edits.py"]})


def default_schema():
    return json.loads((ROOT / "config/schema/S1.0.json").read_text())


def narrative(case):
    return next((d for d in case.get("documents", []) if d["document_id"] == "narrative"),
                {"document_id": "narrative", "revision": "missing", "text": ""})


def doc_evidence(case, quote=""):
    if not isinstance(quote, str):
        raise ModelError("模型引用必须为文本")
    doc = narrative(case)
    text = doc["text"]
    if quote and text.count(quote) != 1:
        raise ModelError("模型引用不能唯一定位到理由原文")
    start = text.find(quote) if quote else 0
    return {"type": "document_span", "document_id": doc["document_id"], "revision": doc["revision"],
            "span": [start, start + len(quote) if quote else len(text)], "text": quote or text}


def extract_local(case):
    """Transparent rule-only fallback, with a mandatory human extraction check."""
    doc = narrative(case)
    text = doc["text"]
    claims = []
    for match in re.finditer(r"仅向([^，。；]+?)支付([一二三四五六七八九十\d]+)次", text):
        raw = match.group(2)
        number = int(raw) if raw.isdigit() else {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
                                               "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}.get(raw)
        start, end = case["coverage_start"], case["coverage_end"]
        if "本月" in text[:match.start() + 1]:
            anchor = datetime.fromisoformat(start).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            after = anchor.replace(year=anchor.year + 1, month=1) if anchor.month == 12 else anchor.replace(month=anchor.month + 1)
            start, end = anchor.isoformat(), after.isoformat()
        if number is not None:
            claims.append({"kind": "count", "operator": "exact", "value": number, "unit": "次",
                           "counterparty_ref": match.group(1), "direction": "out", "start": start, "end": end,
                           "document_id": doc["document_id"], "quote": match.group(0)})
    return {"claims": claims, "unresolved": ["离线规则抽取不覆盖全部自由文本，须人工核对遗漏事实"]}


def normalize_claims(case, response):
    claims = []
    try:
        parsed = ExtractionOutput.model_validate(response)
    except ValidationError as exc:
        raise ModelError("事实抽取响应不符合结构契约") from exc
    failures = list(parsed.unresolved)
    for item in parsed.claims:
        try:
            item = ClaimOutput.model_validate(item).model_dump(exclude_none=True)
            quote = item["quote"]
            evidence = doc_evidence(case, quote)
            claim = {k: v for k, v in item.items() if k not in {"quote", "document_id"}}
            claim.update(text=quote, source={k: evidence[k] for k in ["document_id", "revision", "span"]})
            claim["claim_id"] = "claim-" + digest({k: v for k, v in claim.items() if k != "source"})[:16]
            claims.append(claim)
        except (ModelError, ValidationError, TypeError, KeyError, AttributeError, ValueError) as exc:
            failures.append("事实候选未通过结构或引用校验：" + str(exc).splitlines()[0])
    # Duplicate model outputs must not count as independent checks.
    claims = list({c["claim_id"]: c for c in claims}.values())
    return {"claims": claims, "unresolved": failures}


def focuses_for(case):
    alert = case.get("alert") or {}
    focuses = list(alert.get("focuses", []))
    if not focuses and alert.get("original_focus"):
        focuses = [{"focus_id": alert.get("alert_id", "original-focus"), "text": alert["original_focus"]}]
    focuses += case.get("review_scope", {}).get("upgraded_leads", [])
    return focuses


def focus_evidence(case, focus):
    upgraded = next((row for row in case.get("review_scope", {}).get("upgraded_leads", [])
                     if row["focus_id"] == focus["focus_id"]), None)
    if upgraded:
        return {"type": "upgraded_focus", "focus_id": focus["focus_id"], "lead_id": upgraded.get("lead_id"),
                "origin_issue_id": upgraded.get("origin_issue_id"), "content_hash": digest(upgraded)}
    return {"type": "alert_focus", "focus_id": focus["focus_id"], "revision": (case.get("alert") or {}).get("revision")}


def tool_definitions():
    specs = [
        ("query_transactions", "查询本案指定范围流水并精确汇总；空集也保留查询范围。",
         {"direction": {"type": "string", "enum": ["in", "out"]}, "counterparty_token": {"type": "string"},
          "start": {"type": "string"}, "end": {"type": "string"}}),
        ("read_material", "读取目录中的一份材料；缺失也返回目录范围。", {"material_id": {"type": "string"}}),
        ("read_document", "读取当前案件文档。", {"document_id": {"type": "string"}}),
        ("resolve_entity", "用明确标识或已确认映射核实对象，不按名称相似合并。", {"entity": {"type": "string"}}),
        ("check_coverage", "查看来源覆盖声明及缺失范围。", {}),
    ]
    return [{"type": "function", "function": {"name": name, "description": desc,
             "parameters": {"type": "object", "properties": props, "additionalProperties": False}}}
            for name, desc, props in specs]


def execute_tool(case, schema, evaluator, name, args):
    definitions = {d["function"]["name"]: d["function"] for d in tool_definitions()}
    if name not in definitions or not isinstance(args, dict):
        raise ValueError("未知工具或参数格式")
    allowed = definitions[name]["parameters"]["properties"]
    if set(args) - set(allowed) or any(not isinstance(v, str) for v in args.values()):
        raise ValueError("工具参数超出允许范围")
    deps = ["source:metadata", "source:execution", "source:schema"]
    if name == "query_transactions":
        deps += ["source:transactions", "source:coverage", "source:entities"]
        def compute():
            return core.query_transactions(case, args)
    elif name == "read_material":
        deps += ["source:materials"]
        def compute():
            return {"material": next((m for m in case.get("materials", []) if m["material_id"] == args.get("material_id")), None),
                    "scope": {"material_ids": [m["material_id"] for m in case.get("materials", [])], "source": "current_snapshot"}}
    elif name == "read_document":
        deps += ["source:documents"]
        def compute():
            return {"document": next((d for d in case.get("documents", []) if d["document_id"] == args.get("document_id")), None)}
    elif name == "resolve_entity":
        deps += ["source:entities", "source:transactions"]
        def compute():
            return core.resolve_entity(case, args.get("entity", ""))
    else:
        deps += ["source:coverage"]
        def compute():
            return {"coverage": case.get("coverage", []), "scope": [case["coverage_start"], case["coverage_end"]]}
    key = "tool:" + name + ":" + digest(args)[:20]
    return evaluator.evaluate(key, name, args, deps, compute)


def semantic_input(case, checks):
    # Keep inspectable summaries in the initial context. Detailed rows remain
    # available through the same read-only tools and the stored evidence graph.
    def compact(value):
        if isinstance(value, dict):
            return {k: compact(v) for k, v in value.items() if k not in {"evidence", "windows", "coverage", "checks"}}
        if isinstance(value, list):
            return [compact(v) for v in value]
        return value
    return {"task_mode": case["task_mode"], "focuses": focuses_for(case), "documents": case.get("documents", []),
            "materials": case.get("materials", []), "material_links": case.get("material_links", []),
            "checks": compact(checks),
            "lead_basis_catalog": [{k: row[k] for k in ("basis_ref", "kind", "content_hash")} for row in lead_basis(case, checks).values()],
            "scope": {"account": case["subject_account_id"], "start": case["coverage_start"], "end": case["coverage_end"]}}


def normalize_semantic(case, response, checks=None, tool_trace=(), execution=None):
    try:
        response = SemanticOutput.model_validate(response).model_dump()
    except ValidationError as exc:
        raise ModelError("语义候选不符合结构契约") from exc
    checks = checks or {}
    expected = {f["focus_id"] for f in focuses_for(case)}
    items = response.get("focuses", [])
    if not isinstance(items, list) or any(not isinstance(f, dict) or not isinstance(f.get("focus_id"), str) for f in items) or {f.get("focus_id") for f in items} != expected or len(items) != len(expected):
        raise ModelError("回应判断没有完整覆盖必需关注点")
    for focus in items:
        if focus.get("status") not in {"addressed", "not_addressed", "pending_judgement"}:
            raise ModelError("非法回应判断状态")
        if focus["status"] == "addressed" and not focus.get("quote"):
            raise ModelError("回应判断缺少原文")
        focus["evidence"] = [doc_evidence(case, focus.get("quote", "")), focus_evidence(case, focus)]
    gaps = response.get("gaps", [])
    if not isinstance(gaps, list):
        raise ModelError("材料缺口格式错误")
    claim_results = {c["claim_id"]: c for c in checks.get("claim_results", [])}
    material_results = {m["link_id"]: m for m in checks.get("material_results", [])}
    links = {m["link_id"]: m for m in case.get("material_links", [])}
    for gap in gaps:
        cited = doc_evidence(case, gap["quote"])
        kind, ref = gap["basis_kind"], gap["basis_ref"]
        evidence = [cited]
        # Model prose remains in the raw trace and model_draft; user-facing
        # assertions come from validated categories and frozen results.
        draft = gap.pop("reason")
        if kind == "identity_unresolved":
            related = claim_results.get(ref)
            if not related or related.get("execution_status") != "identity_unresolved":
                raise ModelError("身份缺口没有绑定未对齐的事实候选")
            spans = [e for e in related.get("evidence", []) if e.get("type") == "document_span"]
            if not any(e.get("document_id") == cited["document_id"] and e.get("revision") == cited["revision"] and
                       e["span"][0] < cited["span"][1] and cited["span"][0] < e["span"][1] for e in spans):
                raise ModelError("身份缺口与所引用事实不是同一原文对象")
            gap.update(title="理由中的对象身份待核实", reason="原文名称未完成明确映射；已给定账户标识的材料字段对应不等于该名称身份已确认。")
            evidence += related.get("evidence", [])
        elif kind in {"missing_linked_material", "material_mismatch"}:
            related, link = material_results.get(ref), links.get(ref)
            if not related or not link:
                raise ModelError("材料缺口未绑定本次材料关系检查")
            if kind == "material_mismatch":
                if related["result"] != "mismatch":
                    raise ModelError("模型材料不匹配判断与确定性结果冲突")
                gap.update(title="已关联材料字段不对应", reason="指定关系" + ref + "的确定性字段核验结果为mismatch，需补正或复核。")
            else:
                revision = str(link.get("material_revision", link.get("revision", "")))
                if any(m["material_id"] == link["material_id"] and str(m["revision"]) == revision for m in case.get("materials", [])):
                    raise ModelError("模型声称缺失的已关联材料实际存在")
                gap.update(title="缺少已关联材料版本", reason="关系" + ref + "引用的材料及修订在当前材料集合中不存在，需补证。")
            evidence += related.get("evidence", [])
        else:
            if ref != "narrative":
                raise ModelError("解释支持建议须绑定被检理由")
            gap.update(title="业务解释支持程度需人工核实", reason="该原文解释的支持充分性尚待人工判断；此建议不改变确定性事实或材料字段对应结果。")
            evidence.append({"type": "query_scope", "source": "materials", "material_ids": [m["material_id"] for m in case.get("materials", [])],
                             "material_links": case.get("material_links", [])})
        gap["model_draft"] = {"reason": draft, "requested_material": gap["requested_material"], "status": "unverified_model_suggestion"}
        gap["evidence"] = evidence
    gaps = list({(g["basis_kind"], g["basis_ref"], g["quote"]): g for g in gaps}.values())
    return {"focuses": items, "gaps": gaps,
            "leads": normalize_leads(case, response["leads"], lead_basis(case, checks, tool_trace), execution)}


def model_stage(case, schema, evaluator, model, checks, mode, attempt_trace=None, fixed_trace=()):
    data = semantic_input(case, checks)
    if mode == "fixed":
        # Cache bookkeeping is not model input: a fresh and reused read with
        # identical facts must produce the same complete semantic request.
        data["fixed_tool_results"] = [{k: row[k] for k in ("tool", "arguments", "result", "status", "result_ref")}
                                      for row in fixed_trace]
        data["lead_basis_catalog"] = [{k: row[k] for k in ("basis_ref", "kind", "content_hash")}
                                      for row in lead_basis(case, checks, fixed_trace).values()]
    execution = evaluator.sources["source:execution"]["value"]
    base = [{"role": "system", "content": SEMANTIC_SYSTEM}, {"role": "user", "content": canonical(data)}]
    initial = normalize_semantic(case, json_answer(model.complete(base)), checks, fixed_trace, execution=execution)
    if mode == "fixed":
        return {"semantic": initial, "agent_trace": [], "adaptive_rounds": 0}
    messages = [{"role": "system", "content": AGENT_SYSTEM},
                {"role": "user", "content": canonical({"case": data, "initial_candidates": initial,
                    "available_material_ids": [m["material_id"] for m in case.get("materials", [])]})}]
    trace = attempt_trace if attempt_trace is not None else []
    rounds = 0
    for round_index in range(3):
        message = model.complete(messages, tools=tool_definitions())
        messages.append(message)
        calls = message.get("tool_calls", [])
        if not isinstance(calls, list) or any(not isinstance(c, dict) or not isinstance(c.get("id"), str) or
                not isinstance(c.get("function"), dict) or not isinstance(c["function"].get("name"), str) or
                not isinstance(c["function"].get("arguments"), str) for c in calls):
            raise ModelError("模型工具调用格式无效")
        if not calls:
            return {"semantic": normalize_semantic(case, json_answer(message), checks, trace, execution), "agent_trace": trace, "adaptive_rounds": rounds}
        rounds += 1
        if len(calls) > 2:
            raise ModelError("模型单轮工具调用超出冻结预算，未执行本轮")
        for call in calls:
            function = call.get("function", {})
            tool = function.get("name", "")
            start = perf_counter()
            args = {}
            try:
                args = json.loads(function.get("arguments", "{}"))
                result = execute_tool(case, schema, evaluator, tool, args)
                status = "completed"
            except (ValueError, KeyError, TypeError) as exc:
                result = {"error": str(exc), "execution_status": "failed"}
                status = "failed"
            trace.append({"tool": tool, "arguments": args, "result": result, "status": status,
                          "result_ref": "tool:" + tool + ":" + digest(args)[:20] if status == "completed" else None,
                          "raw_arguments": function.get("arguments"), "round": round_index + 1,
                          "duration_ms": round((perf_counter() - start) * 1000, 3), "purpose": "模型根据已见返回结果选择的追加核查"})
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "content": canonical({**result, "lead_basis_ref": trace[-1]["result_ref"]})})
    messages.append({"role": "user", "content": "追加工具预算已耗尽。请返回JSON最终候选；没有核实的事项保持pending_judgement。"})
    return {"semantic": normalize_semantic(case, json_answer(model.complete(messages)), checks, trace, execution), "agent_trace": trace, "adaptive_rounds": rounds}


class FlowState(TypedDict):
    result: dict


def run_review(case, *, mode="fixed", provider="local", strategy="full", previous=None, model=None):
    """Public entry point. Returned snapshot is internal, immutable storage input."""
    if mode not in {"fixed", "agent"} or provider not in {"local", "deepseek", "frozen"}:
        raise ValueError("未知执行方式")
    if mode == "agent" and provider == "local":
        raise ValueError("离线规则流程不是真实Agent，请选择DeepSeek")
    validation = validate_case(case)
    if not validation["valid"]:
        raise ValueError("; ".join(validation["errors"]))
    case = deepcopy(case)
    for collection, key in [("documents", "document_id"), ("materials", "material_id"), ("transactions", "transaction_id")]:
        case[collection] = sorted({i[key]: i for i in case.get(collection, [])}.values(), key=lambda i: i[key])
    schema = case.get("schema") or default_schema()
    model = model or (DeepSeek(max_calls=6) if provider == "deepseek" else None)
    execution = {"workflow": VERSION, "prompt": PROMPT_VERSION, "mode": mode, "provider": provider,
                 "implementation_hash": IMPLEMENTATION_HASH,
                 "dependency_lock_hash": digest((ROOT / "uv.lock").read_text()),
                 "model_endpoint": getattr(model, "config", {}).get("DEEPSEEK_BASE_URL"),
                 "model": getattr(model, "model", "rule-parser"), "max_rounds": 3, "tools_per_round": 2,
                 "max_model_calls": 6, "max_output_tokens": 4096, "temperature": 0, "thinking": "disabled",
                 "tools": tool_definitions(), "extraction_prompt": EXTRACT_SYSTEM, "semantic_prompt": SEMANTIC_SYSTEM,
                 "agent_prompt": AGENT_SYSTEM, "response_contracts": contract_schemas()}
    if mode == "fixed" and provider != "local":
        execution["fixed_policy"] = deepcopy(FIXED_POLICY)
    if getattr(model, "execution_budget_spec", None) is not None:
        execution["evaluation_budget"] = deepcopy(model.execution_budget_spec)
    evaluator = Evaluator(sources_for(case, schema, execution), previous, strategy)
    started = perf_counter()
    result = {"case_id": case["case_id"], "mode": mode, "provider": provider, "strategy": strategy,
              "features": [], "claims": [], "machine_claims": [], "claim_amendments": [], "superseded_machine_claims": [],
              "claim_results": [], "material_results": [], "semantic_results": [], "lead_candidates": [], "issues": [],
              "schema_version": schema["schema_version"],
              "open_items": [], "required_checks": [], "warnings": list(validation["warnings"]),
              "agent_verified": False, "execution": execution}
    all_dependencies = sorted(evaluator.sources)

    def issue(kind, title, reason, target, evidence, blocking=True):
        item = {"issue_id": "issue-" + digest([kind, target])[:16], "type": kind, "title": title,
                "description": reason, "target_id": target, "evidence": evidence,
                "severity": "blocking" if blocking else "notice", "status": "candidate"}
        result["issues"].append(item)
        if blocking:
            result["open_items"].append({"item_id": item["issue_id"], "kind": kind, "title": title,
                                         "reason": reason, "target_id": target})

    def check(check_id, label, status="completed"):
        result["required_checks"].append({"check_id": check_id, "label": label, "status": status})

    def calculate(_):
        labels = case.get("review_scope", {}).get("target_labels", ["F1", "F2", "count", "amount_sum", "counterparty", "time_range", "material_relation", "alert_response"])
        features = evaluator.evaluate("features", "compute_features", {},
            ["source:metadata", "source:transactions", "source:coverage", "source:schema", "source:execution"],
            lambda: core.compute_features(case, schema))
        result["features"] = [f for f in features if f.get("feature_code", f.get("feature")) in labels]
        for feature in result["features"]:
            name = feature.get("feature_code", feature.get("feature"))
            check("feature:" + name, name)
            if feature["result"] == "undeterminable":
                issue("insufficient_coverage", name + "资料覆盖不足", "无法完成所要求窗口的确定判断", "feature:" + name, feature.get("evidence", []))
        try:
            extraction = evaluator.evaluate("extraction", "extract_claims", {},
                ["source:documents", "source:metadata", "source:schema", "source:execution"],
                lambda: normalize_claims(case, extract_local(case) if provider == "local" else json_answer(model.complete([
                    {"role": "system", "content": EXTRACT_SYSTEM},
                    {"role": "user", "content": canonical({"document": narrative(case), "account": case["subject_account_id"],
                      "start": case["coverage_start"], "end": case["coverage_end"]})}]))))
            result["machine_claims"] = [c for c in extraction["claims"] if c["kind"] in labels]
            effective = evaluator.evaluate("effective_claims", "apply_human_claims", {},
                ["extraction", "source:claim_amendments", "source:documents", "source:entities", "source:metadata", "source:schema", "source:execution"],
                lambda: merge_claim_amendments(case, result["machine_claims"]))
            result["claims"] = effective["claims"]
            result["claim_amendments"] = effective["amendments"]
            result["superseded_machine_claims"] = effective["superseded_machine_claims"]
            for amendment in effective["amendments"]:
                if amendment["status"] == "needs_review":
                    target = "human_claim:" + amendment["amendment_id"]
                    check(target, "人工抽取忠实性待重新核对", "pending_judgement")
                    issue("manual_claim_review", "人工抽取需重新审核", amendment["reason"], target, [doc_evidence(case)])
            check("extraction", "文本事实抽取", "pending_judgement" if extraction["unresolved"] else "completed")
            if extraction["unresolved"]:
                issue("manual_extraction", "需人工核对事实抽取", "; ".join(map(str, extraction["unresolved"])), "extraction", [doc_evidence(case)])
        except ModelError as exc:
            result["claim_amendments"] = [{**deepcopy(r), "status": "needs_review", "allowed_actions": ["supersede", "revoke"],
                "reason": "抽取失败，未将人工提议视为已生效结果"} for r in case.get("claim_amendments", [])]
            check("extraction", "文本事实抽取", "failed")
            issue("execution_failed", "事实抽取失败", str(exc), "extraction", [])
        for claim in result["claims"]:
            target = "claim:" + claim["claim_id"]
            verified = evaluator.evaluate(target, "verify_claim", claim,
                ["effective_claims", "source:transactions", "source:coverage", "source:entities", "source:metadata", "source:schema", "source:execution"],
                lambda c=claim: core.verify_claim(case, c, schema))
            result["claim_results"].append(verified)
            status = verified.get("execution_status", "completed")
            check(target, claim["text"], status)
            if verified["result"] != "supported":
                issue("claim_error" if verified["result"] == "contradicted" else "claim_unresolved", "事实矛盾" if verified["result"] == "contradicted" else "事实待核实",
                      verified.get("reason", verified["result"]), target, verified.get("evidence", []))
        if "material_relation" in labels:
            def check_bound_materials():
                rows = core.check_materials(case, schema=schema)
                valid_targets = {c['claim_id'] for c in result['claims']} | {f['focus_id'] for f in focuses_for(case)}
                valid_targets |= {'claim:' + c['claim_id'] for c in result['claims']}
                changed_targets = {r['claim']['claim_id'] for r in result['superseded_machine_claims']}
                changed_targets |= {'claim:' + target for target in list(changed_targets)}
                for row in rows:
                    target = row.get('claim_or_issue_id')
                    if target in changed_targets or (target and target not in valid_targets):
                        row.update(field_result=row['result'], result='insufficient', binding_status='needs_review',
                                   reason='材料关系所指陈述已替换、撤销或不存在，须明确修订支持对象后重查')
                return rows
            material_results = evaluator.evaluate("materials", "check_materials", {},
                ["source:materials", "source:material_links", "source:transactions", "source:coverage", "source:entities", "source:metadata", "source:schema", "source:execution", "source:claim_amendments"]
                    + (["effective_claims"] if "effective_claims" in evaluator.nodes else []),
                check_bound_materials)
            result["material_results"] = material_results
            check("materials", "材料支持关系", "failed" if any(m.get("execution_status") == "failed" for m in material_results) else "completed")
            for material in material_results:
                if material["result"] != "corresponds":
                    target = "materials:" + str(material.get("link_id", material.get("material_id", "unknown")))
                    kind = "material_mismatch" if material["result"] == "mismatch" else "manual_material" if material["result"] == "pending_judgement" else "material_insufficient"
                    issue(kind, "材料关系" + ("不对应" if kind == "material_mismatch" else "待补证或判断"),
                          material.get("reason", material["result"]), target, material.get("evidence", []))
        return {"result": result}

    def respond(_):
        targets = case.get("review_scope", {}).get("target_labels", list(schema["labels"]))
        requires_focus = case["task_mode"] == "alert_review" or "alert_response" in targets or bool(focuses_for(case))
        if requires_focus and (not focuses_for(case) or (not case.get("alert") and case["task_mode"] == "alert_review")):
            check("alert", "原预警与回应范围", "pending")
            issue("missing_alert", "缺少原预警", "相关必需检查未完成", "semantic", [])
        elif requires_focus and not narrative(case)["text"].strip():
            check("narrative", "甄别理由全文", "pending")
            issue("missing_narrative", "缺少甄别理由", "空理由无法完成回应核验，须补正来源后重查", "semantic", [])
        elif requires_focus:
            if provider == "local":
                check("semantic", "原预警回应核验", "pending_judgement")
                for focus in focuses_for(case):
                    upgraded = {f["focus_id"] for f in case.get("review_scope", {}).get("upgraded_leads", [])}
                    result["semantic_results"].append({"focus_id": focus["focus_id"], "status": "pending_judgement",
                        "type": "upgraded_lead" if focus["focus_id"] in upgraded else "alert_focus",
                        "execution_status": "completed", "required": True, "reason": "离线演示不作语义判断",
                        "evidence": [doc_evidence(case), focus_evidence(case, focus)],
                        "object": {"account_id": case["subject_account_id"], "start": case["coverage_start"], "end": case["coverage_end"]}})
                    issue("manual_focus", "待人工判断是否回应关注点", focus["text"], "semantic:" + focus["focus_id"],
                          [doc_evidence(case), focus_evidence(case, focus)])
            else:
                attempt_trace = []
                try:
                    checks = {k: result[k] for k in ["features", "claims", "claim_results", "material_results", "issues"]}
                    fixed_trace = []
                    if mode == "fixed":
                        result["fixed_policy_reads"] = fixed_trace
                        for name in FIXED_POLICY["steps"]:
                            args = {} if name == "check_coverage" else {"start": case["coverage_start"], "end": case["coverage_end"]}
                            try:
                                value = execute_tool(case, schema, evaluator, name, args)
                            except (ValueError, KeyError, TypeError) as exc:
                                fixed_trace.append({"tool": name, "arguments": args, "status": "failed",
                                                    "result_ref": None, "error": str(exc)})
                                raise ModelError("固定取证未完成：" + str(exc)) from exc
                            fixed_trace.append({"tool": name, "arguments": args, "result": value, "status": "completed",
                                                "result_ref": "tool:" + name + ":" + digest(args)[:20],
                                                "cache_status": evaluator.trace[-1]["status"]})
                    stage = evaluator.evaluate("agent_stage", "agent_stage" if mode == "agent" else "semantic_review", {}, all_dependencies + list(evaluator.nodes),
                        lambda: model_stage(case, schema, evaluator, model, checks, mode, attempt_trace, fixed_trace))
                    stage_reused = evaluator.trace[-1]["status"] == "reused"
                    if stage_reused:
                        for entry in stage["agent_trace"]:
                            if entry["status"] == "completed":
                                execute_tool(case, schema, evaluator, entry["tool"], entry["arguments"])
                    result["agent_trace"] = [{**t, "status": "reused", "original_status": t["status"],
                        "reused_from": "prior_snapshot"} for t in stage["agent_trace"]] if stage_reused else stage["agent_trace"]
                    result["adaptive_rounds"] = 0 if stage_reused else stage["adaptive_rounds"]
                    result["replayed_adaptive_rounds"] = stage["adaptive_rounds"] if stage_reused else 0
                    result["agent_acceptance"] = "需跨任务人工审查路径与工具结果依赖；调用轮数不等于验收通过"
                    check("semantic", "原预警回应核验")
                    for focus in stage["semantic"]["focuses"]:
                        upgraded = {f["focus_id"] for f in case.get("review_scope", {}).get("upgraded_leads", [])}
                        result["semantic_results"].append({**focus, "type": "upgraded_lead" if focus["focus_id"] in upgraded else "alert_focus",
                            "execution_status": "completed", "required": True,
                            "object": {"account_id": case["subject_account_id"], "start": case["coverage_start"], "end": case["coverage_end"]}})
                        if focus["status"] != "addressed":
                            issue("focus_not_addressed" if focus["status"] == "not_addressed" else "manual_focus",
                                  "关注点未回应" if focus["status"] == "not_addressed" else "回应需人工判断", focus.get("reason", ""), "semantic:" + focus["focus_id"], focus["evidence"])
                    for gap in stage["semantic"]["gaps"]:
                        issue("unsupported_explanation", gap.get("title", "解释支持不足"), gap.get("reason", ""),
                              "gap:" + digest([gap["basis_kind"], gap["basis_ref"], gap["quote"]])[:16], gap["evidence"])
                        result["issues"][-1]["model_draft"] = gap["model_draft"]
                    result["lead_candidates"] = stage["semantic"]["leads"]
                    for lead in result["lead_candidates"]:
                        if lead_has_disposition(case, lead):
                            continue
                        issue("new_lead", "新增观察建议，待人工确认范围", lead["question"],
                              "lead:" + lead["lead_id"], lead["evidence"], blocking=False)
                        result["issues"][-1].update(lead_id=lead["lead_id"], model_draft=lead["model_draft"],
                                                    novelty_status="unconfirmed")
                except ModelError as exc:
                    result["agent_trace"] = attempt_trace
                    result["adaptive_rounds"] = len({t["round"] for t in attempt_trace})
                    check("semantic", "原预警回应核验", "failed")
                    issue("execution_failed", "模型核验未完成", str(exc), "agent_stage", [])
        result["run_status"] = "partial" if any(c["status"] != "completed" for c in result["required_checks"]) else "completed"
        result["qc_recommendation"] = ("建议退回修订" if any(i["type"] in {"claim_error", "material_mismatch", "focus_not_addressed"} for i in result["issues"])
            else "建议补证" if any(i["type"] in {"unsupported_explanation", "material_insufficient", "insufficient_coverage", "claim_unresolved"} for i in result["issues"])
            else "待人工判断" if result["open_items"] else "未发现确定问题")
        return {"result": result}

    graph = StateGraph(FlowState)
    graph.add_node("deterministic_checks", calculate)
    graph.add_node("bounded_review", respond)
    graph.add_edge(START, "deterministic_checks")
    graph.add_edge("deterministic_checks", "bounded_review")
    graph.add_edge("bounded_review", END)
    graph.compile().invoke({"result": result})
    agent_trace = result.pop("agent_trace", [])
    result["trace"] = evaluator.trace + agent_trace
    result["model_requests"] = list(getattr(model, "calls", []))
    usage_complete = all(isinstance(r.get("usage"), dict) and all(isinstance(r["usage"].get(k), int)
        for k in ["prompt_tokens", "completion_tokens"]) for r in result["model_requests"])
    result["stats"] = {**evaluator.stats(), "tool_calls": sum(t["status"] == "computed" and t["tool"] not in {"extract_claims", "apply_human_claims", "semantic_review", "agent_stage"} for t in evaluator.trace),
        "adaptive_tool_calls": sum(t["status"] != "reused" for t in agent_trace),
        "replayed_tool_calls": sum(t["status"] == "reused" for t in agent_trace),
        "adaptive_tools_completed": sum(t["status"] == "completed" for t in agent_trace),
        "adaptive_tools_failed": sum(t["status"] == "failed" for t in agent_trace),
        "model_calls": len(result["model_requests"]),
        "usage_complete": usage_complete,
        "input_tokens": sum((r.get("usage") or {}).get("prompt_tokens", 0) for r in result["model_requests"]) if usage_complete else None,
        "output_tokens": sum((r.get("usage") or {}).get("completion_tokens", 0) for r in result["model_requests"]) if usage_complete else None,
        "duration_ms": round((perf_counter() - started) * 1000, 3),
        "revoked": len(set((previous or {}).get("candidate_ids", [])) - {i["issue_id"] for i in result["issues"]})}
    result["snapshot"] = evaluator.snapshot()
    result["snapshot"]["candidate_ids"] = [i["issue_id"] for i in result["issues"]]
    result["warnings"].append("仅适用于当前合成案例及可见单账户范围；质检通过不代表客户无洗钱风险。")
    if provider == "local":
        result["warnings"].append("离线规则流程不是Agent；语义判断和抽取完整性须人工裁决。")
    return result
