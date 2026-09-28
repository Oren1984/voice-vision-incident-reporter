"""FastAPI application: pages, forms, media, health and metrics.

Run locally:  python -m incident_reporter serve   (binds 127.0.0.1 by default)
"""

from __future__ import annotations

import logging
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import __version__, audit
from .concepts import BY_ID
from .config import Settings
from .db import Database
from .labels import DISPLAY, LABELS, pct
from .media import AUDIO_TYPES, INCIDENT_ID_RE, MediaError, MediaStore, read_limited
from .observability import Metrics, request_id_var, setup_logging
from .security import LoginThrottle, User, authenticate, csrf_ok, end_session, session_user, start_session
from .workflow import IncidentService, WorkflowError

HERE = Path(__file__).parent
COOKIE = "vvir_session"
log = logging.getLogger("vvir")

CSP = ("default-src 'self'; img-src 'self' blob: data:; media-src 'self' blob:; style-src 'self'; script-src 'self'; "
       "object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'")


def highlight(text: str, mentions: list[dict]) -> list[dict]:
    """Split text into plain / highlighted pieces for safe (autoescaped) rendering."""
    out, pos = [], 0
    for m in sorted(mentions, key=lambda m: m["start"]):
        if m["start"] < pos:
            continue
        if m["start"] > pos:
            out.append({"text": text[pos:m["start"]]})
        out.append({"text": text[m["start"]:m["end"]], "concept": m["concept"], "negated": m.get("negated", False),
                    "label": BY_ID[m["concept"]].label if m["concept"] in BY_ID else m["concept"]})
        pos = m["end"]
    if pos < len(text):
        out.append({"text": text[pos:]})
    return out


def create_app(settings: Settings | None = None, models=None, log_to_file: bool = True) -> FastAPI:
    settings = settings or Settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(settings.log_path if log_to_file else None)
    db = Database(settings.db_path)
    store = MediaStore(settings.media_dir)
    metrics = Metrics()
    if models is None:
        from .models import ModelSuite

        models = ModelSuite(settings)
    service = IncidentService(db, store, models, settings, metrics)
    throttle = LoginThrottle()
    templates = Jinja2Templates(directory=str(HERE / "templates"))
    templates.env.globals.update(DISPLAY=DISPLAY, LABELS=LABELS, BY_ID=BY_ID, version=__version__)
    templates.env.filters["pct"] = pct

    @asynccontextmanager
    async def lifespan(_app):
        yield
        db.close()

    app = FastAPI(title="Voice + Vision Incident Reporter", version=__version__, docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)
    app.state.settings, app.state.db, app.state.service, app.state.metrics, app.state.models = settings, db, service, metrics, models
    app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")
    max_body = settings.limits.audio_max_bytes + settings.limits.image_max_bytes + 256 * 1024

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        rid = secrets.token_hex(6)
        token = request_id_var.set(rid)
        t0 = time.perf_counter()
        try:
            cl = request.headers.get("content-length")
            if cl and cl.isdigit() and int(cl) > max_body:
                metrics.inc("http.413")
                response = JSONResponse({"error": "Request too large."}, status_code=413)
            else:
                response = await call_next(request)
        except Exception:
            metrics.inc("http.500")
            log.exception("unhandled_error", extra={"path": request.url.path})
            response = HTMLResponse("Internal error. The request id is " + rid, status_code=500)
        dt = time.perf_counter() - t0
        route = request.scope.get("route")
        name = getattr(route, "path", "unmatched")
        metrics.observe(f"http {request.method} {name}", dt)
        metrics.inc(f"http.status.{response.status_code}")
        if not request.url.path.startswith("/static"):
            log.info("http_request", extra={"method": request.method, "route": name, "status": response.status_code, "duration_ms": round(1000 * dt, 1)})
        response.headers["X-Request-ID"] = rid
        response.headers["Content-Security-Policy"] = CSP
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Cache-Control"] = response.headers.get("Cache-Control", "no-store")
        request_id_var.reset(token)
        return response

    # ------------------------------------------------------------ helpers
    def current(request: Request) -> tuple[User, str] | None:
        return session_user(db, request.cookies.get(COOKIE))

    def page(request: Request, name: str, ctx: dict, status: int = 200) -> HTMLResponse:
        sess = current(request)
        base = {"user": sess[0] if sess else None, "csrf": sess[1] if sess else "", "flash": request.query_params.get("msg")}
        return templates.TemplateResponse(request, name, {**base, **ctx}, status_code=status)

    def need_user(request: Request) -> tuple[User, str]:
        s = current(request)
        if s is None:
            raise _Redirect("/login")
        return s

    async def form_with_csrf(request: Request, csrf: str, **kw):
        form = await request.form(max_files=2, max_fields=60, **kw)
        if not csrf_ok(csrf, form.get("csrf")):
            metrics.inc("security.csrf_rejected")
            raise WorkflowError("Your session form token is invalid. Reload the page and try again.", 403)
        return form

    class _Redirect(Exception):
        def __init__(self, url: str) -> None:
            self.url = url

    @app.exception_handler(_Redirect)
    async def _redir(request: Request, exc: _Redirect):
        return RedirectResponse(exc.url, status_code=303)

    @app.exception_handler(WorkflowError)
    async def _wf(request: Request, exc: WorkflowError):
        metrics.inc("workflow.refused")
        if request.headers.get("accept", "").startswith("application/json"):
            return JSONResponse({"error": str(exc)}, status_code=exc.status)
        return page(request, "error.html", {"message": str(exc), "back": request.headers.get("referer") and "javascript:history.back()"}, exc.status)

    # ------------------------------------------------------------ auth
    @app.get("/login", response_class=HTMLResponse)
    async def login_form(request: Request):
        return page(request, "login.html", {"error": None})

    @app.post("/login")
    async def login(request: Request):
        form = await request.form(max_files=0, max_fields=5)
        username = str(form.get("username", ""))[:64].strip().lower()
        password = str(form.get("password", ""))[:256]
        if throttle.blocked(username):
            metrics.inc("auth.locked")
            return page(request, "login.html", {"error": "Too many attempts. Wait five minutes and try again."}, 429)
        user = await run_in_threadpool(authenticate, db, username, password)
        if user is None:
            throttle.fail(username)
            metrics.inc("auth.failed")
            audit.record(db, username or "(blank)", "login_failed", request_id=request_id_var.get())
            return page(request, "login.html", {"error": "Wrong username or password."}, 401)
        throttle.reset(username)
        token, _ = start_session(db, user, settings.session_hours)
        audit.record(db, user.username, "login", request_id=request_id_var.get())
        resp = RedirectResponse("/", status_code=303)
        resp.set_cookie(COOKIE, token, httponly=True, samesite="strict", secure=settings.secure_cookies, max_age=int(settings.session_hours * 3600))
        return resp

    @app.post("/logout")
    async def logout(request: Request):
        user, csrf = need_user(request)
        await form_with_csrf(request, csrf)
        end_session(db, request.cookies.get(COOKIE))
        audit.record(db, user.username, "logout", request_id=request_id_var.get())
        resp = RedirectResponse("/login", status_code=303)
        resp.delete_cookie(COOKIE)
        return resp

    # ------------------------------------------------------------ pages
    @app.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request):
        user, _ = need_user(request)
        return page(request, "dashboard.html", {"incidents": service.list(user)})

    @app.get("/incidents/new", response_class=HTMLResponse)
    async def new_form(request: Request):
        need_user(request)
        return page(request, "new.html", {"limits": settings.limits, "error": None, "audio_types": sorted(AUDIO_TYPES)})

    @app.post("/incidents")
    async def create(request: Request):
        user, csrf = need_user(request)
        try:
            form = await form_with_csrf(request, csrf, max_part_size=64 * 1024)
            audio_f, image_f = form.get("audio"), form.get("image")
            audio = read_limited(audio_f.file, settings.limits.audio_max_bytes, "audio") if getattr(audio_f, "filename", None) else None
            image = read_limited(image_f.file, settings.limits.image_max_bytes, "image") if getattr(image_f, "filename", None) else None
            incident_id = await run_in_threadpool(service.create, user, audio, image)
        except MediaError as e:
            return page(request, "new.html", {"limits": settings.limits, "error": str(e), "audio_types": sorted(AUDIO_TYPES)}, 400)
        return RedirectResponse(f"/incidents/{incident_id}", status_code=303)

    def _valid_id(incident_id: str) -> str:
        if not INCIDENT_ID_RE.match(incident_id):
            raise WorkflowError("Incident not found.", 404)
        return incident_id

    @app.get("/incidents/{incident_id}", response_class=HTMLResponse)
    async def detail(request: Request, incident_id: str):
        user, _ = need_user(request)
        inc = service.get(user, _valid_id(incident_id))
        a = inc.get("analysis") or {}
        text_ev = a.get("text_evidence")
        current_text = inc["transcript_corrected"] or inc["transcript_original"] or ""
        return page(request, "incident.html", {
            "inc": inc, "a": a, "draft": inc.get("draft") or {},
            "pieces": highlight(current_text, text_ev["mentions"]) if text_ev else [],
            "current_text": current_text,
            "events": audit.events(db, incident_id),
            "revisions": service.revisions(user, incident_id),
            "record": service.approved_record(user, incident_id),
            "four_eyes_block": settings.four_eyes and inc["created_by"] == user.id,
        })

    @app.get("/incidents/{incident_id}/media/{kind}")
    async def media(request: Request, incident_id: str, kind: str):
        user, _ = need_user(request)
        inc = service.get(user, _valid_id(incident_id))
        name = inc["audio_file"] if kind == "audio" else inc["image_file"] if kind == "image" else None
        if not name:
            raise WorkflowError("Not found.", 404)
        mime = "image/jpeg" if kind == "image" else AUDIO_TYPES[name.rsplit(".", 1)[1]]
        return FileResponse(store.path(incident_id, name), media_type=mime, headers={"Cache-Control": "private, no-store", "Content-Disposition": "inline"})

    @app.get("/incidents/{incident_id}/record.json")
    async def record_json(request: Request, incident_id: str):
        user, _ = need_user(request)
        rec = service.approved_record(user, _valid_id(incident_id))
        if rec is None:
            raise WorkflowError("No approved record for this incident.", 404)
        return JSONResponse(rec)

    # ------------------------------------------------------------ actions
    def _int(v) -> int:
        try:
            return int(v)
        except (TypeError, ValueError):
            raise WorkflowError("Missing version. Reload the page.", 400)

    @app.post("/incidents/{incident_id}/transcript")
    async def correct(request: Request, incident_id: str):
        user, csrf = need_user(request)
        form = await form_with_csrf(request, csrf)
        await run_in_threadpool(service.correct_transcript, user, _valid_id(incident_id), str(form.get("transcript", "")), _int(form.get("version")))
        return RedirectResponse(f"/incidents/{incident_id}?msg=Transcript+saved+and+analysis+updated#transcript", status_code=303)

    @app.post("/incidents/{incident_id}/submit")
    async def submit(request: Request, incident_id: str):
        user, csrf = need_user(request)
        form = await form_with_csrf(request, csrf)
        service.submit(user, _valid_id(incident_id), _int(form.get("version")))
        return RedirectResponse(f"/incidents/{incident_id}?msg=Submitted+for+review", status_code=303)

    @app.post("/incidents/{incident_id}/draft")
    async def edit(request: Request, incident_id: str):
        user, csrf = need_user(request)
        form = await form_with_csrf(request, csrf)
        answers = {k[len("answer_"):]: str(v) for k, v in form.items() if k.startswith("answer_")}
        service.edit_draft(user, _valid_id(incident_id), _int(form.get("version")), form.get("category") or None,
                           form.get("summary"), form.get("notes"), answers)
        return RedirectResponse(f"/incidents/{incident_id}?msg=Draft+saved#review", status_code=303)

    @app.post("/incidents/{incident_id}/approve")
    async def approve(request: Request, incident_id: str):
        user, csrf = need_user(request)
        form = await form_with_csrf(request, csrf)
        service.approve(user, _valid_id(incident_id), _int(form.get("version")), form.get("confirm") == "yes", form.get("acknowledge_open") == "yes")
        return RedirectResponse(f"/incidents/{incident_id}?msg=Approved", status_code=303)

    @app.post("/incidents/{incident_id}/reject")
    async def reject(request: Request, incident_id: str):
        user, csrf = need_user(request)
        form = await form_with_csrf(request, csrf)
        service.reject(user, _valid_id(incident_id), _int(form.get("version")), str(form.get("reason", "")))
        return RedirectResponse(f"/incidents/{incident_id}?msg=Rejected", status_code=303)

    # ------------------------------------------------------------ ops
    @app.get("/healthz")
    async def healthz():
        try:
            db.one("SELECT 1")
            db_ok = True
        except Exception:
            db_ok = False
        info = models.info() if hasattr(models, "info") else {}
        return JSONResponse({"status": "ok" if db_ok else "degraded", "db": db_ok, "models_loaded": info.get("loaded", {}),
                             "heads": info.get("heads"), "version": __version__}, status_code=200 if db_ok else 503)

    def _ops_data() -> dict:
        return {"metrics": metrics.snapshot(), "stats": service.stats(), "audit_chain": audit.verify(db),
                "models": models.info() if hasattr(models, "info") else {}}

    @app.get("/ops", response_class=HTMLResponse)
    async def ops(request: Request):
        user, _ = need_user(request)
        if not user.can_review:
            raise WorkflowError("Only reviewers and admins can view operations.", 403)
        return page(request, "ops.html", {**_ops_data(), "recent": audit.events(db, limit=40)})

    @app.get("/api/metrics")
    async def api_metrics(request: Request):
        user, _ = need_user(request)
        if not user.can_review:
            raise WorkflowError("Forbidden.", 403)
        return JSONResponse(_ops_data())

    return app
