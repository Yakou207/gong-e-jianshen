"""Routes for stage ① intake, ② system checks, ③ AI 研判 (streamed) and ④ the human decision on the AI draft."""
import json
import queue
import threading
import time
from typing import Literal

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from . import core, investigation
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
                            "status": value.get("status") if value else None})
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

    @app.post("/api/cases/{case_id}/verdict-review")
    def verdict_review(case_id: str, body: VerdictReviewInput):
        try:
            store.review_verdict(case_id, **body.model_dump())
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        return store.get(case_id)
