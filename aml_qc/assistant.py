"""AML 助手: the workbench's side-panel agent.

It helps a user operate the workbench (read the built-in help, list cases, open a case or stage,
propose an action that the user must confirm) and answers questions about the open case with the
same read-only tools as the investigation agent. It never writes data: navigation is returned as
UI actions for the page to perform, and anything that costs money or records a decision stays a
button the user presses.
"""
from copy import deepcopy
import json
from pathlib import Path
import re
from time import perf_counter
from uuid import uuid4

from pydantic import ValidationError

from . import investigation as inv
from .depgraph import canonical
from .llm import ModelError, json_answer
from .workflow import default_schema

ROOT = Path(__file__).resolve().parents[1]
VERSION = "assistant-1.0"
SYSTEM_PROMPT = (ROOT / "config/prompts/assistant-1.0/system.txt").read_text().strip()
HELP = (ROOT / "config/help/workbench.txt").read_text()
HELP_SECTIONS = {m.group(1).strip(): m.group(2).strip() for m in re.finditer(r"^## (.+?)\n(.*?)(?=^## |\Z)", HELP, re.S | re.M)}
STAGES = {"materials": "① 案件资料", "rules": "② 规则检验", "ai": "③ AI 研判", "qc": "④ 理由质检", "review": "⑤ 复核归档"}
ACTIONS = {"start_investigation": "开始 AI 研判（调用大模型，产生费用）", "replay_investigation": "回放上次研判（免费）"}
MAX_ROUNDS, TOOLS_PER_ROUND, MAX_MODEL_CALLS = 4, 3, 6
CASE_TOOLS = ("query_transactions", "flow_profile", "compute_features", "check_coverage", "resolve_entity", "read_material", "read_document")


def _fn(name, description, properties=None, required=()):
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": properties or {}, "required": list(required), "additionalProperties": False}}}


def tools(case_open):
    defs = [
        _fn("read_help", "读取工作台使用说明的某一节。", {"topic": {"type": "string", "enum": list(HELP_SECTIONS)}}, ["topic"]),
        _fn("list_cases", "列出案件及其 AI 研判结论、质检状态与人工处理状态；可按 AI 结论筛选。",
            {"ai_recommendation": {"type": "string", "enum": ["report_suspicious", "exclude", "insufficient_evidence", "none"]}}),
        _fn("case_status", "查看案件的研判草稿摘要、质检问题与未决事项；不填 case_id 即当前打开的案件。", {"case_id": {"type": "string"}}),
        _fn("open_case", "在工作台中打开案件，可指定阶段。", {"case_id": {"type": "string"},
            "stage": {"type": "string", "enum": list(STAGES)}}, ["case_id"]),
        _fn("go_to_stage", "切换当前案件的阶段。", {"stage": {"type": "string", "enum": list(STAGES)}}, ["stage"]),
        _fn("propose_action", "向用户建议一个需要其确认才执行的操作。", {"action": {"type": "string", "enum": list(ACTIONS)},
            "reason": {"type": "string"}}, ["action", "reason"]),
    ]
    if case_open:
        defs += [d for d in inv.tools() if d["function"]["name"] in CASE_TOOLS]
    return defs


class Workbench:
    """What the assistant may read from the workbench; supplied by the API layer."""

    def __init__(self, list_cases, case_status, package):
        self.list_cases, self.case_status, self.package = list_cases, case_status, package


def _ui(name, args, workbench, context):
    """Validate a UI action server-side; the page performs it."""
    case_id = args.get("case_id") or context.get("case_id")
    if name in ("open_case",) and case_id not in {c["case_id"] for c in workbench.list_cases()}:
        raise ValueError("案件不存在：" + str(case_id))
    if name == "go_to_stage" and not context.get("case_id"):
        raise ValueError("当前没有打开的案件，请先 open_case")
    if name == "propose_action" and not context.get("case_id"):
        raise ValueError("请先打开案件，再建议操作")
    stage = args.get("stage")
    if stage is not None and stage not in STAGES:
        raise ValueError("未知阶段")
    action = {"open_case": {"action": "open_case", "case_id": case_id, "stage": stage or "ai"},
              "go_to_stage": {"action": "go_to_stage", "stage": stage},
              "propose_action": {"action": "propose", "proposal": args.get("action"), "label": ACTIONS.get(args.get("action")),
                                 "reason": args.get("reason", ""), "case_id": context.get("case_id")}}[name]
    if name == "propose_action" and args.get("action") not in ACTIONS:
        raise ValueError("未知操作")
    return action


def run_tool(name, args, workbench, context, case_state):
    if not isinstance(args, dict):
        raise ValueError("工具参数不是 JSON 对象")
    if name == "read_help":
        if args.get("topic") not in HELP_SECTIONS:
            raise ValueError("可读主题：" + "、".join(HELP_SECTIONS))
        return {"topic": args["topic"], "text": HELP_SECTIONS[args["topic"]]}, None
    if name == "list_cases":
        wanted = args.get("ai_recommendation")
        rows = [c for c in workbench.list_cases()
                if wanted is None or (c.get("ai_recommendation") or "none") == wanted]
        return {"count": len(rows), "cases": rows[:40]}, None
    if name == "case_status":
        target = args.get("case_id") or context.get("case_id")
        if not target:
            raise ValueError("请给出 case_id，或先打开案件")
        return workbench.case_status(target), None
    if name in ("open_case", "go_to_stage", "propose_action"):
        action = _ui(name, args, workbench, context)
        return {"status": "ok", "ui_action": action}, action
    if name in CASE_TOOLS:
        if case_state is None:
            raise ValueError("当前没有打开的案件，案件工具不可用；可先 open_case")
        case, schema, evaluator = case_state
        if name == "read_document":
            document = next((d for d in case.get("documents", []) if d["document_id"] == args.get("document_id")), None)
            if document is None:
                raise ValueError("可读文档：" + "、".join(d["document_id"] for d in case.get("documents", [])))
            return {"document": document}, None
        return inv.run_tool(case, schema, evaluator, name, args), None
    raise ValueError("未知工具")


def converse(message, history, context, workbench, model, emit=lambda event: None):
    """One assistant turn. history: earlier [{"role": "user"|"assistant", "text": ...}] of this panel."""
    events, trace = [], []

    def push(event):
        event = {"seq": len(events) + 1, **event}
        events.append(event)
        emit(deepcopy(event))
    turn = {"turn_id": uuid4().hex, "version": VERSION, "context": deepcopy(context), "message": message,
            "events": events, "trace": trace, "answer": None, "ui_actions": [], "errors": [], "warnings": [], "status": "failed"}
    started = perf_counter()
    case_id = context.get("case_id")
    case_state = None
    snapshot = {"view": context.get("view"), "stage": context.get("stage"), "case": None,
                "cases_overview": {"count": len(workbench.list_cases())}}
    try:
        if case_id:
            package = workbench.package(case_id)
            schema = package.get("schema") or default_schema()
            case_state = (package, schema, inv._evaluator(package, schema))
            snapshot["case"] = workbench.case_status(case_id)
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": canonical({"workbench_state": snapshot, "stages": STAGES})}]
        for item in history[-8:]:
            if item.get("role") in ("user", "assistant") and isinstance(item.get("text"), str):
                messages.append({"role": item["role"], "content": item["text"][:1500]})
        where = f"{case_id} · {STAGES.get(context.get('stage'), '')}" if case_id else "案件列表"
        messages.append({"role": "user", "content": f"[当前页面：{where}] {message}"})
        push({"type": "user", "text": message})
        defs = tools(case_state is not None)
        for round_index in range(1, MAX_ROUNDS + 1):
            if len(getattr(model, "calls", [])) >= MAX_MODEL_CALLS - 1:
                break
            reply = model.complete(messages, tools=defs, stage="tool_review")
            messages.append({k: v for k, v in reply.items() if k in ("role", "content", "tool_calls")})
            calls = reply.get("tool_calls") or []
            if not calls:
                break
            if reply.get("content"):
                push({"type": "assistant", "text": reply["content"].strip(), "round": round_index})
            for index, call in enumerate(calls):
                name = (call.get("function") or {}).get("name", "")
                ref = f"A{len(trace) + 1}"
                try:
                    args = json.loads((call.get("function") or {}).get("arguments") or "{}")
                except ValueError:
                    args = None
                push({"type": "tool_call", "ref": ref, "tool": name, "arguments": args, "round": round_index})
                try:
                    if index >= TOOLS_PER_ROUND:
                        raise ValueError(f"每轮最多执行 {TOOLS_PER_ROUND} 个工具，此调用未执行")
                    result, action = run_tool(name, args, workbench, context, case_state)
                    status = "completed"
                except (ValueError, KeyError, TypeError) as exc:
                    result, action, status = {"error": str(exc), "execution_status": "failed"}, None, "failed"
                trace.append({"ref": ref, "tool": name, "arguments": args, "result": result, "status": status})
                push({"type": "tool_result", "ref": ref, "tool": name, "status": status,
                      "summary": summarize(name, result), "result": result})
                if action:
                    turn["ui_actions"].append(action)
                    push({"type": "ui_action", **action})
                    if action["action"] in ("open_case", "go_to_stage"):
                        context = {**context, **({"case_id": action["case_id"]} if action.get("case_id") else {}),
                                   "stage": action.get("stage")}
                        if action["action"] == "open_case" and action["case_id"] != case_id:
                            case_id = action["case_id"]
                            package = workbench.package(case_id)
                            schema = package.get("schema") or default_schema()
                            case_state = (package, schema, inv._evaluator(package, schema))
                            defs = tools(True)
                messages.append({"role": "tool", "tool_call_id": call.get("id"),
                                 "content": canonical({"ref": ref, "status": status, "result": result})})
        citable = [{"ref": t["ref"], "tool": t["tool"], "summary": summarize(t["tool"], t["result"])} for t in trace if t["status"] == "completed"]
        final = messages + [{"role": "user", "content": canonical({"instruction": "请按规定输出 JSON 回答；citations 只能取自 citable_refs。",
                                                                   "citable_refs": citable})}]
        raw = json_answer(model.complete(final, stage="final"))
        try:
            answer = inv.ChatAnswer.model_validate(raw)
        except ValidationError:
            raise ModelError("回答不符合输出结构") from None
        valid = {c["ref"] for c in citable}
        turn["errors"] = [f"引用了不存在的工具返回 {c}" for c in answer.citations if c not in valid]
        known = set()
        for t in trace:
            if t["status"] == "completed":
                inv._numbers_in(t["result"], known)
        inv._numbers_in(snapshot, known)
        # Figures quoted inside returned text (case summaries, help) also count as coming from the system.
        for text in re.findall(r'"([^"]*)"', canonical([t["result"] for t in trace if t["status"] == "completed"] + [snapshot])):
            known.update(inv._norm(n) for n in re.findall(r"\d[\d,]*(?:\.\d+)?", text))
        unverified = sorted({m.group(1) + m.group(2) for m in inv.AMOUNT.finditer(answer.answer)
                             if m.group(2) != "%" and inv._norm(m.group(1)) not in known})
        if unverified:
            turn["warnings"].append("以下数字未在工具返回中找到，请核对：" + "、".join(unverified))
        turn.update(answer=answer.model_dump(), status="completed")
        push({"type": "answer", "text": answer.answer, "citations": answer.citations,
              "errors": turn["errors"], "warnings": turn["warnings"]})
    except ModelError as exc:
        turn["errors"].append(str(exc))
        push({"type": "error", "text": str(exc)})
    turn["stats"] = {"model_calls": len(getattr(model, "calls", [])), "tool_calls": len(trace),
                     "duration_ms": round((perf_counter() - started) * 1000, 1), "usage": inv.usage(getattr(model, "calls", []))}
    return turn


def summarize(name, result):
    if result.get("execution_status") == "failed" or "error" in result:
        return "未执行：" + str(result.get("error"))
    if name == "read_help":
        return f"已读取说明「{result['topic']}」"
    if name == "list_cases":
        return f"共 {result['count']} 个案件"
    if name == "case_status":
        verdict = result.get("ai_recommendation")
        return f"{result.get('case_id')}：AI 研判 {inv.RECOMMENDATIONS.get(verdict, '未研判')}，质检问题 {len(result.get('qc_issues', []))} 条"
    if "ui_action" in result:
        a = result["ui_action"]
        return {"open_case": f"打开 {a.get('case_id')} · {STAGES.get(a.get('stage'), '')}", "go_to_stage": f"切换到 {STAGES.get(a.get('stage'), '')}",
                "propose": f"建议：{a.get('label')}（待你确认）"}[a["action"]]
    return inv.summarize(name, result)
