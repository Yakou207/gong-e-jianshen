"""Routes for stage ① intake, ② system checks, ③ AI 研判 (streamed) and ④ the human decision on the AI draft."""
import json
import queue
import threading
import time
from typing import Literal

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from . import assistant, core, investigation
from .flows import flow_profile
from .intake import build_case
from .llm import DeepSeek, settings
from .spend import SpendCappedModel, cap, spent
from .workflow import default_schema

_RUN_LOCK = threading.Lock()
REPLAY_DELAY_SECONDS = 0.6


class IntakeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str
    account_id: str
    business_type: str = ""
    coverage_start: str = ""
    coverage_end: str
    coverage_status: Literal["full", "partial"] = "full"
    transactions_csv: str = Field(min_length=1, max_length=2_000_000)
    focuses: list[str]
    kyc_text: str = ""
    narrative_text: str = ""
    synthetic_confirmed: bool


class InvestigateInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: Literal["deepseek", "replay"] = "deepseek"
    session_id: str | None = None


class ChatInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question: str = Field(min_length=1, max_length=500)


class AssistantInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str = Field(min_length=1, max_length=800)
    history: list[dict] = Field(default_factory=list, max_length=20)
    context: dict = Field(default_factory=dict)


class VerdictReviewInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: str
    action: Literal["adopt", "revise", "reject"]
    reason: str = Field(min_length=1, max_length=5000)
    actor: str = Field(default="甄别人员", min_length=1, max_length=100)
    recommendation: Literal["report_suspicious", "exclude", "insufficient_evidence"] | None = None
    opinion: str | None = Field(default=None, max_length=2000)


def _sse(payload):
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


def _stream(work):
    """Run work(emit) in a thread and forward each emitted event to the browser as it happens."""
    events = queue.Queue()

    def worker():
        try:
            record = work(lambda event: events.put(("event", event)))
            events.put(("done", record))
        except Exception as exc:  # surfaced to the console, never turned into a result
            events.put(("fail", f"{type(exc).__name__}: {exc}"))

    threading.Thread(target=worker, daemon=True).start()

    def generate():
        while True:
            kind, value = events.get()
            if kind == "event":
                yield _sse(value)
            elif kind == "done":
                yield _sse({"type": "done", "session_id": value.get("session_id") if value else None,
                            "status": value.get("status") if value else None,
                            "answer": (value.get("answer") or {}).get("answer") if value else None})
                return
            else:
                yield _sse({"type": "error", "text": value})
                yield _sse({"type": "done", "status": "failed"})
                return
    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def _model(purpose):
    values = settings()
    if not values["DEEPSEEK_API_KEY"]:
        raise HTTPException(409, "未配置 DeepSeek API Key：可选择“回放已保存的研判”，或在 .env 中配置后重启")
    return SpendCappedModel(DeepSeek(max_calls=investigation.MAX_MODEL_CALLS), purpose)


ISSUE_LABEL = {"claim_error": "事实矛盾", "claim_unresolved": "事实待核实", "material_mismatch": "材料不对应",
               "focus_not_addressed": "关注点未回应", "unsupported_explanation": "解释支持不足", "insufficient_coverage": "资料覆盖不足",
               "material_insufficient": "材料不足", "manual_focus": "回应待人工判断", "execution_failed": "执行失败"}


def discrepancy(run, issue, package):
    """Plain-language stated-vs-observed text for a fact issue, from the deterministic comparison only."""
    target = str(issue.get("target_id", ""))
    if not target.startswith("claim:"):
        return None
    result = next((r for r in run.get("claim_results", []) if "claim:" + r["claim_id"] == target), None)
    claim = next((c for c in run.get("claims", []) if "claim:" + c["claim_id"] == target), {})
    c = (result or {}).get("comparison") or {}
    names = {p["counterparty_token"]: p.get("display_name_masked", p["counterparty_token"]) for p in package.get("counterparties", [])}
    scope = "完整流水" if ((result or {}).get("coverage") or {}).get("status") == "full" else "现有可见流水"
    quote = f"理由原文“{claim.get('text', '')}”："
    if c.get("unit") == "transactions":
        return f"{quote}陈述 {c['expected']} 笔，{scope}核得 {c['actual']} 笔"
    if c.get("unit") == "CNY_fen":
        return f"{quote}陈述合计 {c['expected'] / 100:.2f} 元，{scope}核得 {c['actual'] / 100:.2f} 元"
    if isinstance(c.get("expected"), list):
        show = lambda tokens: "、".join(names.get(t, t) for t in tokens) or "无"
        return f"{quote}陈述对手 {show(c['expected'])}，{scope}实际对手 {show(c['actual'])}"
    if "inside_count" in c:
        return f"{quote}陈述期间内 {c['inside_count']} 笔、期间外 {c['outside_count']} 笔"
    return None


def register(app, store):
    @app.get("/api/agent/status")
    def agent_status():
        return {"deepseek_configured": bool(settings()["DEEPSEEK_API_KEY"]), "spent_cny": str(spent()),
                "cap_cny": str(cap()), "version": investigation.VERSION,
                "limits": {"rounds": investigation.MAX_ROUNDS, "tools_per_round": investigation.TOOLS_PER_ROUND,
                           "model_calls": investigation.MAX_MODEL_CALLS}}

    @app.post("/api/intake")
    def intake(body: IntakeInput):
        try:
            return store.create(build_case(body.model_dump()), reason="上传资料创建案件")
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None

    @app.get("/api/cases/{case_id}/rules")
    def rules(case_id: str):
        package = store.get(case_id)["package"]
        schema = package.get("schema") or default_schema()
        return {"features": core.compute_features(package, schema), "flow_profile": flow_profile(package),
                "coverage": core.check_coverage(package), "schema_version": schema["schema_version"],
                "note": "本页结果全部由确定性代码按整数分计算；F1/F2 阈值为演示口径，不是监管标准。"}

    @app.get("/api/cases/{case_id}/sessions")
    def sessions(case_id: str):
        return {"sessions": store.sessions(case_id)}

    @app.post("/api/cases/{case_id}/investigate")
    def investigate(case_id: str, body: InvestigateInput):
        state = store.get(case_id)
        if body.provider == "replay":
            saved = [s for s in store.sessions(case_id) if s["kind"] == "investigation" and s.get("verdict")
                     and (body.session_id in (None, s["session_id"]))]
            if not saved:
                raise HTTPException(404, "本案还没有可回放的研判记录")
            source = saved[-1]

            def replay(emit):
                emit({"type": "stage", "text": f"回放 {source['created_at'][:16].replace('T', ' ')} 的真实研判记录（不调用模型）",
                      "replay": True})
                for event in source["events"]:
                    time.sleep(REPLAY_DELAY_SECONDS)
                    emit({**event, "replay": True})
                return {"session_id": source["session_id"], "status": source["status"]}
            return _stream(replay)
        model = _model("investigate:" + case_id)
        if not _RUN_LOCK.acquire(blocking=False):
            raise HTTPException(409, "已有研判在运行，请稍候")

        def run(emit):
            try:
                session = investigation.investigate(state["package"], model, emit=emit)
                store.save_session(case_id, state["source_hash"], session)
                return session
            finally:
                _RUN_LOCK.release()
        return _stream(run)

    @app.post("/api/cases/{case_id}/chat")
    def chat(case_id: str, body: ChatInput):
        state = store.get(case_id)
        history = [s for s in store.sessions(case_id) if not s["stale"]]
        model = _model("chat:" + case_id)
        if not _RUN_LOCK.acquire(blocking=False):
            raise HTTPException(409, "已有研判在运行，请稍候")

        def run(emit):
            try:
                turn = investigation.chat(state["package"], history, body.question, model, emit=emit)
                store.save_session(case_id, state["source_hash"], turn)
                return turn
            finally:
                _RUN_LOCK.release()
        return _stream(run)

    def case_status(case_id):
        state = store.get(case_id)
        agent = state["agent"]
        latest = agent["latest_investigation"] or {}
        verdict = latest.get("verdict") or {}
        run = state["latest_run"] or {}
        return {"case_id": case_id, "business_type": state["package"].get("profile", {}).get("business_type"),
                "alert_focuses": [f["text"] for f in (state["package"].get("alert") or {}).get("focuses", [])],
                "has_narrative": any(d["document_id"] == "narrative" for d in state["package"].get("documents", [])),
                "ai_recommendation": verdict.get("recommendation"), "ai_summary": verdict.get("summary"),
                "ai_draft_stale": latest.get("stale"), "ai_information_requests": verdict.get("information_requests"),
                "human_decision": (agent["latest_decision"] or {}).get("action"),
                "qc_run_status": "stale" if state["stale"] else run.get("run_status", "not_run"),
                "qc_recommendation": run.get("qc_recommendation"),
                "qc_issues": [{"type": ISSUE_LABEL.get(i["type"], i["type"]), "title": i.get("title"),
                               "detail": discrepancy(run, i, state["package"]) or str(i.get("description", ""))[:160]}
                              for i in run.get("issues", []) if i["type"] != "new_lead"],
                "open_items": len(state["open_items"]), "review_status": state["review_status"]}

    def case_rows():
        return [{k: row.get(k) for k in ("case_id", "ai_recommendation", "human_decision", "run_status", "qc_recommendation",
                                         "review_status", "coverage_summary")} for row in store.list()]

    workbench = assistant.Workbench(case_rows, case_status, lambda case_id: store.get(case_id)["package"])

    @app.get("/api/help")
    def help_sections():
        return {"sections": [{"topic": k, "text": v} for k, v in assistant.HELP_SECTIONS.items()]}

    @app.post("/api/assistant")
    def assistant_turn(body: AssistantInput):
        model = _model("assistant")
        if not _RUN_LOCK.acquire(blocking=False):
            raise HTTPException(409, "已有 Agent 在运行，请稍候")

        def run(emit):
            try:
                return assistant.converse(body.message, body.history, body.context, workbench, model, emit=emit)
            finally:
                _RUN_LOCK.release()
        return _stream(run)

    @app.post("/api/cases/{case_id}/verdict-review")
    def verdict_review(case_id: str, body: VerdictReviewInput):
        try:
            store.review_verdict(case_id, **body.model_dump())
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        return store.get(case_id)
