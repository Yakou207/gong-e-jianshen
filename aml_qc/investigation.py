"""AI 研判 Agent: investigates one alert with read-only tools and drafts a recommendation for human review.

The model chooses which tool to call next from what earlier calls returned; every number comes from a
deterministic tool. The draft is checked after the fact: evidence must point at real successful calls,
figures must appear in tool returns, and "exclude" is blocked when the window is not fully covered.
A human adopts, revises or rejects the draft; the system never acts on it.
"""
from copy import deepcopy
import json
from pathlib import Path
import re
from time import perf_counter
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from . import core
from .depgraph import Evaluator, canonical, digest, sources_for
from .flows import flow_profile
from .ingest import transaction_cents
from .llm import ModelError, json_answer
from .workflow import default_schema, execute_tool, tool_definitions

ROOT = Path(__file__).resolve().parents[1]
VERSION = "investigation-1.4"
PROMPT_ROOT = ROOT / "config/prompts/inv-1.2"
AGENT_PROMPT = (PROMPT_ROOT / "agent.txt").read_text().strip()
VERDICT_PROMPT = (PROMPT_ROOT / "verdict.txt").read_text().strip()
CHAT_PROMPT = (PROMPT_ROOT / "chat.txt").read_text().strip()
MAX_ROUNDS, TOOLS_PER_ROUND, MAX_MODEL_CALLS = 4, 3, 6
CHAT_ROUNDS = 3
RECOMMENDATIONS = {"report_suspicious": "建议上报可疑交易", "exclude": "建议排除",
                   "insufficient_evidence": "证据不足，建议补充尽调"}
INVESTIGATION_TOOLS = ("query_transactions", "flow_profile", "compute_features", "check_coverage",
                       "resolve_entity", "read_material", "read_document")


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Evidence(Strict):
    ref: str = Field(min_length=1)
    transaction_ids: list[str] = []


class Finding(Strict):
    finding: str = Field(min_length=1, max_length=400)
    severity: Literal["high", "medium", "low"]
    evidence: list[Evidence] = Field(min_length=1)


class FocusAnswer(Strict):
    focus_id: str = Field(min_length=1)
    status: Literal["explained", "unexplained", "needs_information"]
    answer: str = Field(min_length=1, max_length=600)
    evidence: list[Evidence] = Field(min_length=1)


class Verdict(Strict):
    recommendation: Literal["report_suspicious", "exclude", "insufficient_evidence"]
    summary: str = Field(min_length=1, max_length=600)
    risk_findings: list[Finding]
    mitigating_findings: list[Finding]
    focus_answers: list[FocusAnswer]
    information_requests: list[str]
    draft_opinion: str = Field(min_length=1, max_length=1200)


class ChatAnswer(Strict):
    answer: str = Field(min_length=1, max_length=1200)
    citations: list[str] = []


def tools():
    defs = [d for d in tool_definitions() if d["function"]["name"] in INVESTIGATION_TOOLS]
    defs.append({"type": "function", "function": {
        "name": "flow_profile",
        "description": "统计时间窗内收付合计、对手集中度（前5名及占比）、同日收付日期和00:00-06:00交易；只描述可见流水。",
        "parameters": {"type": "object", "properties": {"start": {"type": "string"}, "end": {"type": "string"}},
                       "additionalProperties": False}}})
    return defs


def _evaluator(case, schema):
    execution = {"component": VERSION, "tools": tools(), "agent_prompt": AGENT_PROMPT, "verdict_prompt": VERDICT_PROMPT}
    return Evaluator(sources_for(case, schema, execution), None, "full")


def run_tool(case, schema, evaluator, name, args):
    """Read-only, current-case tools; the analyst's narrative is withheld so the investigation stays independent."""
    if name not in INVESTIGATION_TOOLS or not isinstance(args, dict):
        raise ValueError("未知工具或参数格式")
    if name == "read_document" and args.get("document_id") != "kyc":
        raise ValueError("研判阶段只提供客户资料（kyc），经办理由不在本阶段读取")
    if name == "flow_profile":
        if set(args) - {"start", "end"} or any(not isinstance(v, str) for v in args.values()):
            raise ValueError("工具参数超出允许范围")
        return evaluator.evaluate("tool:flow_profile:" + digest(args)[:20], "flow_profile", args,
                                  ["source:transactions", "source:coverage", "source:entities", "source:metadata"],
                                  lambda: flow_profile(case, args.get("start"), args.get("end")))
    result = execute_tool(case, schema, evaluator, name, args)
    if name == "read_material":
        # The material's declared links (which payments it is meant to support) belong to the same read.
        result = {**deepcopy(result), "material_links": [
            {k: link.get(k) for k in ("link_id", "transaction_ids", "relation_template", "claim_or_issue_id")}
            for link in case.get("material_links", []) if link.get("material_id") == args.get("material_id")]}
    return result


def _yuan(cents):
    return f"{cents // 100}.{cents % 100:02d}"


def summarize(name, result):
    """One plain line per tool return, for the console; derived only from the returned data."""
    if result.get("execution_status") == "failed" or "error" in result:
        return "调用失败：" + str(result.get("error", "未完成"))
    if name == "query_transactions":
        m = result.get("metrics") or {}
        rows = result.get("rows", [])
        cin = sum(transaction_cents(r) for r in rows if r.get("direction") == "in")
        cout = sum(transaction_cents(r) for r in rows if r.get("direction") == "out")
        return (f"返回 {m.get('count', len(rows))} 笔（转入 {_yuan(cin)} 元，转出 {_yuan(cout)} 元），"
                f"对手 {len(m.get('counterparty_tokens', []))} 个，覆盖 {result.get('coverage', {}).get('status', '未知')}")
    if name == "flow_profile":
        top = result["outflow"]["top_counterparties"][:1]
        lead = f"；最大转出对手 {top[0]['display_name']} 占 {top[0]['share_percent']}%" if top else ""
        return (f"转入 {result['inflow']['count']} 笔 {result['inflow']['amount']} 元 / {result['inflow']['distinct_counterparties']} 个对手，"
                f"转出 {result['outflow']['count']} 笔 {result['outflow']['amount']} 元{lead}，覆盖 {result['coverage']['status']}")
    if name == "compute_features":
        return "；".join(f"{f['feature_code']} {f['result']}" for f in result.get("features", []))
    if name == "check_coverage":
        return "覆盖声明 " + "、".join(f"{c.get('start', '')[:10]}~{c.get('end', '')[:10]} {c.get('status')}" for c in result.get("coverage", []))
    if name == "read_material":
        m = result.get("material") or {}
        return f"材料 {m.get('material_id', '未找到')}：{m.get('material_type', '')} 金额 {m.get('amount', '—')} 元" if m else "目录中没有该材料"
    if name == "resolve_entity":
        return f"对象核实：{result.get('execution_status')} {result.get('counterparty_token') or ''}"
    if name == "read_document":
        return "已读取客户资料"
    return "已返回"


def context(case, schema):
    """What the agent sees up front: alert, customer profile, catalog and deterministic rule results — no narrative."""
    alert = case.get("alert") or {}
    kyc = next((d for d in case.get("documents", []) if d["document_id"] == "kyc"), None)
    return {"case_id": case["case_id"], "account": case["subject_account_id"], "profile": case.get("profile", {}),
            "period": {"start": case["coverage_start"], "end": case["coverage_end"], "timezone": case.get("timezone")},
            "alert": {"alert_id": alert.get("alert_id"), "trigger_features": alert.get("trigger_features", []),
                      "focuses": [{"focus_id": f["focus_id"], "text": f["text"]} for f in alert.get("focuses", [])]},
            "customer_profile": kyc["text"] if kyc else None,
            "materials_catalog": [{"material_id": m["material_id"], "material_type": m.get("material_type")}
                                  for m in case.get("materials", [])],
            "rule_results": rule_results(case, schema)}


def rule_results(case, schema):
    """Stage ② (system checks): deterministic only."""
    features = core.compute_features(case, schema)
    profile = flow_profile(case)
    return {"features": [{"feature_code": f["feature_code"], "result": f["result"]} for f in features],
            "coverage": core.check_coverage(case)["status"],
            "flow_profile": {k: profile[k] for k in ("inflow", "outflow", "out_to_in_percent", "days_with_both_directions")}}


def _returned_ids(value, found=None):
    found = set() if found is None else found
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "transaction_id" and isinstance(item, str):
                found.add(item)
            elif key == "transaction_ids" and isinstance(item, list):
                found.update(x for x in item if isinstance(x, str))
            else:
                _returned_ids(item, found)
    elif isinstance(value, list):
        for item in value:
            _returned_ids(item, found)
    return found


YUAN, COUNT, PERCENT = "元", "笔", "%"
UNIT_KIND = {"元": YUAN, "万元": YUAN, "笔": COUNT, "个": COUNT, "次": COUNT, "%": PERCENT}
AMOUNT = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(万元|元|笔|个|次|%)")
DECIMAL = re.compile(r"-?\d+\.\d{1,2}")


def _figure(number, unit):
    """(kind, value) for a figure written in text; 万元 is converted to 元."""
    value = float(number.replace(",", "")) * (10000 if unit == "万元" else 1)
    return UNIT_KIND[unit], value


def _figures(value, found=None, key=""):
    """Typed figures in tool returns: yuan, counts and percentages are kept apart so one cannot vouch for another.

    Yuan: decimal strings and *amount* fields, and *_cents integers converted to yuan. Counts: other integers and list
    lengths. Percent: *percent* fields and numerator/denominator pairs. Figures quoted inside returned text count by unit.
    """
    found = {YUAN: set(), COUNT: set(), PERCENT: set()} if found is None else found
    name = key.lower()
    if isinstance(value, dict):
        num, den = value.get("numerator", value.get("ratio_numerator")), value.get("denominator", value.get("ratio_denominator"))
        if isinstance(num, (int, float)) and isinstance(den, (int, float)) and den:
            found[PERCENT].add(round(num / den * 100, 1))
        for k, item in value.items():
            _figures(item, found, k)
    elif isinstance(value, list):
        found[COUNT].add(float(len(value)))
        for item in value:
            _figures(item, found, key)
    elif isinstance(value, bool):
        pass
    elif isinstance(value, (int, float)):
        if name.endswith("_cents"):
            found[YUAN].add(round(value / 100, 2))
        elif "percent" in name:
            found[PERCENT].add(round(float(value), 1))
        elif "amount" in name or "numerator" in name or "denominator" in name:
            pass  # bare amounts in cents are only meaningful through *_cents or ratios
        elif isinstance(value, int):
            found[COUNT].add(float(value))
    elif isinstance(value, str):
        if DECIMAL.fullmatch(value) and ("amount" in name or name in ("", "value", "total")):
            found[YUAN].add(round(float(value), 2))
        elif "percent" in name and re.fullmatch(r"-?\d+(\.\d+)?", value):
            found[PERCENT].add(round(float(value), 1))
        else:
            for number, unit in AMOUNT.findall(value):
                kind, figure = _figure(number, unit)
                found[kind].add(round(figure, 2 if kind == YUAN else 1))
    return found


def _traceable(number, unit, found):
    kind, figure = _figure(number, unit)
    if kind == PERCENT:  # written percentages may be rounded to whole or one-decimal figures
        return any(abs(figure - p) <= (0.05 if figure != int(figure) else 0.5) for p in found[PERCENT])
    if unit == "万元":  # 万元 figures are roundings: "116万元" stands for any amount that rounds to it
        places = len(number.split(".")[1]) if "." in number else 0
        return any(round(v / 10000, places) == round(figure / 10000, places) for v in found[YUAN])
    return round(figure, 2) in found[kind] if kind == YUAN else float(figure) in found[kind]


def unverified_figures(texts, found):
    """Figures in the given texts whose unit-matched value appears in no tool return or given context."""
    return sorted({number + unit for text in texts for number, unit in AMOUNT.findall(text) if not _traceable(number, unit, found)})


def check_verdict(case, verdict, trace, given=None):
    """Return (errors, warnings, policy_flags). Errors mean the draft cannot be shown as a valid draft.

    Figures may come from tool returns or from the context the agent was given up front (alert, customer profile,
    deterministic rule results); each must match in unit."""
    calls = {t["ref"]: t for t in trace if t["status"] == "completed"}
    errors, warnings, flags = [], [], []

    def refs(items, where):
        for item in items:
            for ev in item.evidence:
                call = calls.get(ev.ref)
                if call is None:
                    errors.append(f"{where}引用了不存在或失败的工具返回 {ev.ref}")
                    continue
                missing = set(ev.transaction_ids) - _returned_ids(call["result"])
                if missing:
                    errors.append(f"{where}引用了该调用未返回的交易：{', '.join(sorted(missing))}")
    refs(verdict.risk_findings, "可疑特征")
    refs(verdict.mitigating_findings, "合理解释")
    refs(verdict.focus_answers, "关注点回答")
    focuses = {f["focus_id"] for f in (case.get("alert") or {}).get("focuses", [])}
    answered = [a.focus_id for a in verdict.focus_answers]
    if sorted(answered) != sorted(focuses):
        errors.append("关注点回答须与预警关注点一一对应：应为 " + "、".join(sorted(focuses)))
    known = _figures(given) if given else None
    for call in calls.values():
        known = _figures(call["result"], known)
    known = known or _figures({})
    texts = [verdict.summary, verdict.draft_opinion] + [f.finding for f in verdict.risk_findings + verdict.mitigating_findings] \
        + [a.answer for a in verdict.focus_answers]
    unverified = unverified_figures(texts, known)
    if unverified:
        warnings.append("以下数字未在工具返回中找到，请人工核对：" + "、".join(unverified))
    coverage = core.check_coverage(case)
    if verdict.recommendation == "exclude" and coverage["status"] != "full":
        flags.append("检查期间流水覆盖不完整，“建议排除”不能成立：请补齐资料后重新研判，或改为证据不足。")
    if verdict.recommendation == "exclude" and any(f.severity == "high" for f in verdict.risk_findings):
        flags.append("草稿列出了高风险特征却建议排除，请人工重点复核。")
    return errors, warnings, flags


def _parse_calls(message):
    calls = message.get("tool_calls") or []
    if not isinstance(calls, list) or any(not isinstance(c, dict) or not isinstance(c.get("function"), dict) for c in calls):
        raise ModelError("模型工具调用格式无效")
    return calls


def _tool_round(case, schema, evaluator, calls, round_index, trace, messages, emit, prefix="T", offset=0):
    for index, call in enumerate(calls):
        name = call["function"].get("name", "")
        ref = f"{prefix}{offset + len(trace) + 1}"
        started = perf_counter()
        try:
            args = json.loads(call["function"].get("arguments") or "{}")
        except ValueError:
            args = None
        emit({"type": "tool_call", "id": call.get("id"), "ref": ref, "tool": name, "arguments": args, "round": round_index})
        try:
            if index >= TOOLS_PER_ROUND:
                raise ValueError(f"每轮最多执行 {TOOLS_PER_ROUND} 个工具，此调用未执行")
            if not isinstance(args, dict):
                raise ValueError("工具参数不是 JSON 对象")
            result, status = run_tool(case, schema, evaluator, name, args), "completed"
        except (ValueError, KeyError, TypeError) as exc:
            result, status = {"error": str(exc), "execution_status": "failed"}, "failed"
        record = {"ref": ref, "tool_call_id": call.get("id"), "tool": name, "arguments": args, "result": result, "status": status,
                  "round": round_index, "duration_ms": round((perf_counter() - started) * 1000, 2)}
        trace.append(record)
        emit({"type": "tool_result", "id": call.get("id"), "ref": ref, "tool": name, "status": status,
              "summary": summarize(name, result), "result": result})
        messages.append({"role": "tool", "tool_call_id": call.get("id"),
                         "content": canonical({"ref": ref, "status": status, "result": result})})


def ref_catalog(trace):
    """The only evidence references a draft may cite: system-assigned refs of successful tool returns."""
    return [{"ref": t["ref"], "tool": t["tool"], "arguments": t["arguments"], "summary": summarize(t["tool"], t["result"])}
            for t in trace if t["status"] == "completed"]


def investigate(case, model, emit=lambda event: None, schema=None):
    """Run one investigation; returns the session record (events, trace, verdict, checks, usage)."""
    schema = schema or case.get("schema") or default_schema()
    evaluator = _evaluator(case, schema)
    events, trace = [], []

    def push(event):
        event = {"seq": len(events) + 1, **event}
        events.append(event)
        emit(deepcopy(event))
    session = {"session_id": uuid4().hex, "kind": "investigation", "version": VERSION, "case_id": case["case_id"],
               "model": getattr(model, "model", None), "events": events, "trace": trace,
               "verdict": None, "errors": [], "warnings": [], "policy_flags": [], "status": "failed"}
    started = perf_counter()
    try:
        ctx = context(case, schema)
        push({"type": "stage", "text": "读取预警、客户资料与系统规则检验结果"})
        push({"type": "context", "rule_results": ctx["rule_results"], "focuses": ctx["alert"]["focuses"]})
        messages = [{"role": "system", "content": AGENT_PROMPT}, {"role": "user", "content": canonical(ctx)}]
        for round_index in range(1, MAX_ROUNDS + 1):
            if len(getattr(model, "calls", [])) >= MAX_MODEL_CALLS - 1:
                push({"type": "stage", "text": "调查轮次预算已用完，进入结论"})
                break
            message = model.complete(messages, tools=tools(), stage="tool_review")
            messages.append({k: v for k, v in message.items() if k in ("role", "content", "tool_calls")})
            if message.get("content"):
                push({"type": "assistant", "text": message["content"].strip(), "round": round_index})
            calls = _parse_calls(message)
            if not calls:
                break
            _tool_round(case, schema, evaluator, calls, round_index, trace, messages, push)
        push({"type": "stage", "text": "根据实际工具返回生成研判草稿"})
        final = [{"role": "system", "content": AGENT_PROMPT + "\n\n" + VERDICT_PROMPT}] + messages[1:] + \
            [{"role": "user", "content": canonical({"instruction": "调查结束。请按输出结构给出研判草稿 JSON；evidence.ref 只能取自下列可引用的工具返回。",
                                                   "citable_refs": ref_catalog(trace)})}]
        raw = json_answer(model.complete(final, stage="final"))
        try:
            verdict = Verdict.model_validate(raw)
        except ValidationError as exc:
            raise ModelError("研判草稿不符合输出结构：" + "; ".join(e["msg"] for e in exc.errors()[:3])) from None
        errors, warnings, flags = check_verdict(case, verdict, trace, ctx)
        session.update(verdict=verdict.model_dump(), errors=errors, warnings=warnings, policy_flags=flags,
                       status="needs_attention" if errors else "completed")
        push({"type": "verdict", "verdict": session["verdict"], "label": RECOMMENDATIONS[verdict.recommendation],
              "errors": errors, "warnings": warnings, "policy_flags": flags})
    except ModelError as exc:
        session["errors"].append(str(exc))
        push({"type": "error", "text": str(exc)})
    session["stats"] = {"model_calls": len(getattr(model, "calls", [])),
                        "tool_calls": len(trace), "rounds": len({t["round"] for t in trace}),
                        "duration_ms": round((perf_counter() - started) * 1000, 1),
                        "usage": usage(getattr(model, "calls", []))}
    return session


def chat(case, history, question, model, emit=lambda event: None, schema=None):
    """Answer one follow-up question; history = earlier session records of this case (read-only)."""
    schema = schema or case.get("schema") or default_schema()
    evaluator = _evaluator(case, schema)
    events, trace = [], []

    def push(event):
        event = {"seq": len(events) + 1, **event}
        events.append(event)
        emit(deepcopy(event))
    turn = {"session_id": uuid4().hex, "kind": "chat", "version": VERSION, "case_id": case["case_id"],
            "question": question, "events": events, "trace": trace, "answer": None, "errors": [], "status": "failed"}
    started = perf_counter()
    try:
        latest_trace = next((r.get("trace", []) for r in reversed(history) if r.get("kind") == "investigation"), [])
        known = ref_catalog(latest_trace)
        previous = [{"question": r.get("question"), "answer": (r.get("answer") or {}).get("answer")}
                    for r in history if r.get("kind") == "chat" and r.get("answer")]
        latest = next((r["verdict"] for r in reversed(history) if r.get("kind") == "investigation" and r.get("verdict")), None)
        messages = [{"role": "system", "content": CHAT_PROMPT},
                    {"role": "user", "content": canonical({"case": context(case, schema), "earlier_tool_results": known,
                                                          "earlier_questions": previous, "latest_draft": latest,
                                                          "question": question})}]
        push({"type": "user", "text": question})
        for round_index in range(1, CHAT_ROUNDS + 1):
            message = model.complete(messages, tools=tools(), stage="tool_review")
            messages.append({k: v for k, v in message.items() if k in ("role", "content", "tool_calls")})
            calls = _parse_calls(message)
            if not calls:
                break
            if message.get("content"):
                push({"type": "assistant", "text": message["content"].strip(), "round": round_index})
            _tool_round(case, schema, evaluator, calls, round_index, trace, messages, push, prefix="Q")
        final = messages + [{"role": "user", "content": canonical({"instruction": "请按规定输出 JSON 回答；citations 只能取自下列 ref。",
                                                                   "citable_refs": known + ref_catalog(trace)})}]
        raw = json_answer(model.complete(final, stage="final"))
        try:
            answer = ChatAnswer.model_validate(raw)
        except ValidationError:
            raise ModelError("回答不符合输出结构") from None
        valid = {t["ref"] for t in trace if t["status"] == "completed"} | {k["ref"] for k in known}
        turn["answer"] = answer.model_dump()
        turn["errors"] = [f"引用了不存在的工具返回 {c}" for c in answer.citations if c not in valid]
        turn["status"] = "completed"
        push({"type": "answer", "text": answer.answer, "citations": answer.citations, "errors": turn["errors"]})
    except ModelError as exc:
        turn["errors"].append(str(exc))
        push({"type": "error", "text": str(exc)})
    turn["stats"] = {"model_calls": len(getattr(model, "calls", [])), "tool_calls": len(trace),
                     "duration_ms": round((perf_counter() - started) * 1000, 1),
                     "usage": usage(getattr(model, "calls", []))}
    return turn


def usage(calls):
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "prompt_cache_hit_tokens": 0, "complete": True}
    for call in calls:
        u = call.get("usage")
        if not isinstance(u, dict):
            totals["complete"] = False
            continue
        for key in ("prompt_tokens", "completion_tokens", "prompt_cache_hit_tokens"):
            totals[key] += int(u.get(key) or 0)
    return totals

