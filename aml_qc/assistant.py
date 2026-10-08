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
from .str_draft import MEASURES
from .depgraph import canonical
from .llm import ModelError, json_answer
from .workflow import default_schema

ROOT = Path(__file__).resolve().parents[1]
VERSION = "assistant-1.2"
SYSTEM_PROMPT = (ROOT / "config/prompts/assistant-1.1/system.txt").read_text().strip()
HELP = (ROOT / "config/help/workbench.txt").read_text()
HELP_SECTIONS = {m.group(1).strip(): m.group(2).strip() for m in re.finditer(r"^## (.+?)\n(.*?)(?=^## |\Z)", HELP, re.S | re.M)}
STAGES = {"materials": "① 案件资料", "rules": "② 规则检验", "ai": "③ AI 研判", "qc": "④ 理由质检", "review": "⑤ 复核归档"}
ACTIONS = {"start_investigation": "开始 AI 研判（调用大模型，产生费用）", "replay_investigation": "回放上次研判（免费）",
           "select_measures": "在报告初稿中勾选建议措施", "prefill_decision": "预填人工决定表单（不提交）"}
QUEUE_FILTERS = {"all": "全部预警", "deep": "优先深挖", "evidence": "补充尽调", "close": "可快速关闭"}
SOURCE_TABS = {"narrative": "理由与预警", "transactions": "流水", "materials": "材料", "coverage": "覆盖 / 映射"}
DECISIONS = {"adopt": "采纳 AI 草稿", "revise": "修改后采纳", "reject": "驳回 AI 草稿"}
UI_TOOLS = ("open_case", "go_to_stage", "go_to_list", "show_source", "highlight_transactions", "open_report_draft", "open_intake",
            "propose_action")
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
        _fn("go_to_list", "回到案件列表，可按处理优先级筛选（all 全部、deep 优先深挖、evidence 补充尽调、close 可快速关闭）。",
            {"filter": {"type": "string", "enum": list(QUEUE_FILTERS)}}),
        _fn("show_source", "在当前案件的 ① 案件资料中打开某个页签（narrative 理由与预警、transactions 流水、materials 材料、coverage 覆盖）。",
            {"tab": {"type": "string", "enum": list(SOURCE_TABS)}}, ["tab"]),
        _fn("highlight_transactions", "在当前案件的流水中高亮并定位指定交易；交易编号须来自工具返回。",
            {"transaction_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 50}}, ["transaction_ids"]),
        _fn("open_report_draft", "打开当前案件 ⑤ 复核归档中的可疑交易报告初稿预览。"),
        _fn("open_intake", "打开“上传资料新建案件”对话框。"),
        _fn("propose_action", "向用户建议一个需要其确认才执行的操作：start_investigation 开始研判（产生费用）、replay_investigation 免费回放、"
            "select_measures 在报告初稿中勾选措施（填 measures）、prefill_decision 预填人工决定表单但不提交（填 decision，revise 时填 recommendation，可填 reason_draft）。",
            {"action": {"type": "string", "enum": list(ACTIONS)}, "reason": {"type": "string"},
             "measures": {"type": "array", "items": {"type": "string", "enum": list(MEASURES)}},
             "decision": {"type": "string", "enum": list(DECISIONS)},
             "recommendation": {"type": "string", "enum": list(inv.RECOMMENDATIONS)},
             "reason_draft": {"type": "string", "maxLength": 300}}, ["action", "reason"]),
    ]
    if case_open:
        defs += [d for d in inv.tools() if d["function"]["name"] in CASE_TOOLS]
    return defs


class Workbench:
    """What the assistant may read from the workbench; supplied by the API layer."""

    def __init__(self, list_cases, case_status, package):
        self.list_cases, self.case_status, self.package = list_cases, case_status, package


def _ui(name, args, workbench, context, case=None):
    """Validate a UI action server-side; the page performs it. Nothing here submits a decision or changes data."""
    case_id = args.get("case_id") or context.get("case_id")
    if name == "open_case" and case_id not in {c["case_id"] for c in workbench.list_cases()}:
        raise ValueError("案件不存在：" + str(case_id))
    if name in ("go_to_stage", "show_source", "highlight_transactions", "open_report_draft", "propose_action") and not context.get("case_id"):
        raise ValueError("当前没有打开的案件，请先 open_case")
    stage = args.get("stage")
    if stage is not None and stage not in STAGES:
        raise ValueError("未知阶段")
    if name == "open_case":
        return {"action": "open_case", "case_id": case_id, "stage": stage or "ai"}
    if name == "go_to_stage":
        return {"action": "go_to_stage", "stage": stage}
    if name == "go_to_list":
        wanted = args.get("filter") or "all"
        if wanted not in QUEUE_FILTERS:
            raise ValueError("可选筛选：" + "、".join(QUEUE_FILTERS))
        return {"action": "go_to_list", "filter": wanted}
    if name == "show_source":
        if args.get("tab") not in SOURCE_TABS:
            raise ValueError("可选页签：" + "、".join(SOURCE_TABS))
        return {"action": "show_source", "tab": args["tab"]}
    if name == "highlight_transactions":
        wanted = args.get("transaction_ids")
        if not isinstance(wanted, list) or not wanted or len(wanted) > 50:
            raise ValueError("请给出 1–50 个交易编号")
        known = {t["transaction_id"] for t in (case or {}).get("transactions", [])}
        unknown = [t for t in wanted if t not in known]
        if unknown:
            raise ValueError("本案没有这些交易：" + "、".join(map(str, unknown[:5])))
        return {"action": "highlight_transactions", "transaction_ids": list(dict.fromkeys(wanted))}
    status = workbench.case_status(context["case_id"]) if context.get("case_id") else {}
    has_draft = bool(status.get("ai_recommendation")) and not status.get("ai_draft_stale")
    if name == "open_report_draft":
        if not has_draft:
            raise ValueError("本案还没有当前有效的 AI 研判草稿，无法汇编报告初稿")
        return {"action": "open_report_draft"}
    if name == "open_intake":
        return {"action": "open_intake"}
    proposal = args.get("action")
    if proposal not in ACTIONS:
        raise ValueError("未知操作")
    action = {"action": "propose", "proposal": proposal, "label": ACTIONS[proposal], "reason": args.get("reason", ""),
              "case_id": context.get("case_id")}
    if proposal == "select_measures":
        measures = args.get("measures") or []
        if not measures or any(m not in MEASURES for m in measures):
            raise ValueError("请从以下措施中选择：" + "、".join(MEASURES))
        if not has_draft:
            raise ValueError("本案还没有当前有效的 AI 研判草稿")
        action["measures"] = list(dict.fromkeys(measures))
    if proposal == "prefill_decision":
        decision, recommendation = args.get("decision"), args.get("recommendation")
        if decision not in DECISIONS:
            raise ValueError("decision 须为 adopt、revise 或 reject")
        if decision == "revise" and recommendation not in inv.RECOMMENDATIONS:
            raise ValueError("修改后采纳须给出最终结论 recommendation")
        if not has_draft:
            raise ValueError("本案还没有当前有效的 AI 研判草稿")
        action.update(decision=decision, recommendation=recommendation if decision == "revise" else None,
                      reason_draft=str(args.get("reason_draft") or "")[:300])
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
    if name in UI_TOOLS:
        action = _ui(name, args, workbench, context, case_state[0] if case_state else None)
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
                    elif action["action"] == "go_to_list":
                        context, case_id, case_state, defs = {**context, "view": "tasks", "case_id": None, "stage": None}, None, None, tools(False)
                    elif action["action"] in ("show_source", "highlight_transactions"):
                        context = {**context, "stage": "materials"}
                    elif action["action"] == "open_report_draft":
                        context = {**context, "stage": "review"}
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
        # Typed by unit: yuan only vouches for yuan, counts for counts, percentages for percentages. Figures quoted
        # inside returned text (case summaries, help) count by the unit written next to them.
        known = inv._figures([t["result"] for t in trace if t["status"] == "completed"] + [snapshot])
        unverified = inv.unverified_figures([answer.answer], known)
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
                "go_to_list": f"回到案件列表 · {QUEUE_FILTERS.get(a.get('filter'), '')}", "show_source": f"打开资料页签「{SOURCE_TABS.get(a.get('tab'), '')}」",
                "highlight_transactions": f"高亮 {len(a.get('transaction_ids', []))} 笔交易", "open_report_draft": "打开报告初稿预览",
                "open_intake": "打开上传对话框", "propose": f"建议：{a.get('label')}（待你确认）"}[a["action"]]
    return inv.summarize(name, result)
