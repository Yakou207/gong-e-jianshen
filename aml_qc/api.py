"""Loopback-only demonstration API; no bank integration or autonomous actions."""
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock
from typing import Literal
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .depgraph import canonical
from .ingest import load_case
from .llm import settings
from .store import Store
from .workflow import run_review

ROOT = Path(__file__).resolve().parents[1]


class PackageInput(BaseModel):
    package: dict
    reason: str = ""


class RunInput(BaseModel):
    mode: Literal["fixed", "agent"] = "fixed"
    provider: Literal["local", "deepseek"] = "local"
    strategy: Literal["full", "incremental"] = "full"


class ReviewInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: str
    target_id: str
    reason: str = Field(min_length=1, max_length=5000)
    actor: str = Field(default="reviewer", min_length=1, max_length=100)
    resolution: str | None = None
    new_value: str | None = None
    claim_patch: dict | None = None
    evidence: list[dict] | None = None
    snapshot_id: str | None = None
    expected_event_id: str | None = None
    previous_event_id: str | None = None


class MigrationPreviewInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    base_schema_hash: str
    target_schema: dict
    selected_case_ids: list[str]
    reason: str = Field(min_length=1, max_length=5000)
    actor: str = Field(default="reviewer", min_length=1, max_length=100)


class ClaimProposalInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operation: Literal["replace", "add", "retire", "revoke"]
    target_claim_id: str | None = None
    proposed_claim: dict | None = None
    supersedes_amendment_id: str | None = None
    reason: str = Field(min_length=1, max_length=5000)
    actor: str = Field(min_length=1, max_length=100)
    snapshot_id: str


class ClaimProposalReviewInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["approve", "reject", "withdraw"]
    actor: str = Field(min_length=1, max_length=100)
    reason: str = Field(min_length=1, max_length=5000)
    snapshot_id: str
    expected_event_id: str
    fidelity: Literal["faithful", "not_a_claim", "duplicate"] | None = None


class MigrationApplyInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    preview_hash: str
    reason: str = Field(min_length=1, max_length=5000)
    actor: str = Field(default="reviewer", min_length=1, max_length=100)


def create_app(db_path=None, seed=True):
    store = Store(db_path or settings()["AML_QC_DB"])
    run_lock = Lock()

    @asynccontextmanager
    async def lifespan(app):
        if seed:
            existing = {r["case_id"] for r in store.list()}
            for path in sorted((ROOT / "data/synthetic").glob("seed-*.json")):
                package = load_case(path)
                if package["case_id"] not in existing:
                    store.create(package)
        yield

    app = FastAPI(title="工e鉴审 · 合成演示", lifespan=lifespan)
    app.state.store = store
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])

    @app.middleware("http")
    async def local_boundary(request: Request, call_next):
        origin = request.headers.get("origin")
        if request.method != "GET" and origin:
            try:
                parsed = urlparse(origin)
            except ValueError:
                return JSONResponse({"detail": "仅允许同源本地工作台操作"}, status_code=403)
            if parsed.netloc != request.headers.get("host") or parsed.scheme != request.url.scheme:
                return JSONResponse({"detail": "仅允许同源本地工作台操作"}, status_code=403)
        if request.method == "POST" and not request.headers.get("content-type", "").startswith("application/json"):
            return JSONResponse({"detail": "仅接受JSON请求"}, status_code=415)
        try:
            declared_length = int(request.headers.get("content-length", "0"))
            if declared_length < 0:
                raise ValueError
        except ValueError:
            return JSONResponse({"detail": "Content-Length格式无效"}, status_code=400)
        if declared_length > 5_000_000 or (request.method == "POST" and len(await request.body()) > 5_000_000):
            return JSONResponse({"detail": "演示导入上限5MB"}, status_code=413)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = "default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'"
        return response

    @app.exception_handler(ValueError)
    async def value_error(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(KeyError)
    async def missing(request, exc):
        return JSONResponse({"detail": "案例或所需字段不存在"}, status_code=404)

    @app.get("/api/config")
    def config():
        values = settings()
        return {"deepseek_configured": bool(values["DEEPSEEK_API_KEY"]), "model": values["DEEPSEEK_MODEL"],
                "max_model_calls": 6, "max_output_tokens_per_call": 4096, "synthetic_only": True,
                "version": "0.1.0", "review_roles": ["reviewer", "reviewer-2"]}

    @app.get("/api/cases")
    def cases():
        return {"cases": store.list()}

    @app.get("/api/schemas")
    def schemas():
        return store.schemas()

    @app.post("/api/migrations/preview")
    def migration_preview(body: MigrationPreviewInput):
        return store.preview_migration(**body.model_dump())

    @app.get("/api/migrations/{preview_id}")
    def get_migration_preview(preview_id: str):
        return store.migration_preview(preview_id)

    @app.post("/api/migrations/{preview_id}/cases/{case_id}")
    def apply_migration(preview_id: str, case_id: str, body: MigrationApplyInput):
        return store.migrate_case(preview_id,case_id,**body.model_dump())

    @app.post("/api/cases")
    def create_case(body: PackageInput):
        return store.create(body.package, reason=body.reason.strip() or "导入合成案例")

    @app.get("/api/cases/{case_id}")
    def get_case(case_id: str):
        return store.get(case_id)

    @app.post("/api/cases/{case_id}/source")
    def source(case_id: str, body: PackageInput):
        return store.change_source(case_id, body.package, body.reason)

    @app.post("/api/cases/{case_id}/run")
    def run(case_id: str, body: RunInput):
        if not run_lock.acquire(blocking=False):
            raise HTTPException(409, "工作台已有运行进行中，请等待完成")
        try:
            state = store.get(case_id)
            previous = (state["latest_run"] or {}).get("snapshot") if body.strategy == "incremental" else None
            if body.provider == "local" and body.mode == "agent":
                raise ValueError("离线规则不是Agent，请选择真实DeepSeek")
            try:
                result = run_review(state["package"], **body.model_dump(), previous=previous)
            except Exception as exc:
                # A failed rerun must replace the current status, never inherit a
                # previous pass. Diagnostics deliberately omit raw HTTP objects.
                result = {"case_id": case_id, **body.model_dump(), "run_status": "failed",
                          "qc_recommendation": "运行失败，需重查", "issues": [], "open_items": [],
                          "required_checks": [{"check_id": "run", "label": "本次运行", "status": "failed"}],
                          "warnings": ["运行失败：" + type(exc).__name__], "trace": [], "stats": {}, "snapshot": {}}
            return store.save_run(case_id, state["source_hash"], result)
        finally:
            run_lock.release()

    @app.post("/api/cases/{case_id}/reviews")
    def review(case_id: str, body: ReviewInput):
        return store.review(case_id, **body.model_dump())

    @app.post("/api/cases/{case_id}/claim-proposals")
    def propose_claim(case_id: str, body: ClaimProposalInput):
        return store.propose_claim(case_id, **body.model_dump())

    @app.post("/api/cases/{case_id}/claim-proposals/{proposal_id}/review")
    def review_claim_proposal(case_id: str, proposal_id: str, body: ClaimProposalReviewInput):
        return store.review_claim_proposal(case_id, proposal_id, **body.model_dump())

    @app.get("/api/cases/{case_id}/export")
    def export(case_id: str):
        return Response(canonical(store.export(case_id)), media_type="application/json",
                        headers={"Content-Disposition": 'attachment; filename="gong-e-qc-audit.json"'})

    @app.get("/")
    def index():
        return FileResponse(ROOT / "web/index.html")

    app.mount("/", StaticFiles(directory=ROOT / "web"), name="web")
    return app


app = create_app()
