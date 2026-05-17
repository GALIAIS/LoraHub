"""LoraHub HTTP API.

Run with `lorahub serve` (preferred) or `uvicorn lorahub.api.app:app`.
The API surface is intentionally small for v0.2 — list/create/cancel jobs,
read config schema, stream events. Auth is out of scope; bind to localhost.

All API routes live under `/api`. The site root and `/{spa-path}` are reserved
for the React frontend (mounted from `web/dist` when present).

Per-domain endpoint logic lives under `lorahub.api.routers.*`. This module
keeps only FastAPI plumbing, websocket endpoints (FastAPI APIRouter has had
historical caveats with WS routes), the SPA fallback, and the singletons
that tests monkeypatch on this module:

    monkeypatch.setattr(app_mod, "_settings_store", ...)
    monkeypatch.setattr(app_mod, "_build_bootstrap_runner", ...)
    monkeypatch.setattr(app_mod, "_bootstrap_session", None)

Routers reference these names dynamically through `app_module.<name>` rather
than importing them at module import time — that keeps the patches effective
without requiring TestClient rebuild between tests.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from lorahub import __version__
from lorahub.api import scheduler as sched
from lorahub.api import state
from lorahub.api.bootstrap_session import (
    _BootstrapSession,
    default_build_bootstrap_runner,
)
from lorahub.api.ai_store import (
    AIStore,
    default_ai_store_path,
)
from lorahub.api.helpers import _resolve_web_dist
from lorahub.api.jobs_helpers import _job_events
from lorahub.api.session_store import SessionStore, default_session_store_path
from lorahub.api.settings import SettingsStore
from lorahub.api.state import JobState
from lorahub.api.sweep_store import SweepStore, default_sweep_store_path
from lorahub.core.events import EventType, TrainingEvent

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Test hooks
#
# Module-level so tests can monkeypatch isolated values via
# `monkeypatch.setattr(app, "<name>", ...)` before issuing requests. Routers
# resolve these dynamically as `app_module.<name>` per-request, so patches
# take effect without needing a fresh TestClient.
# --------------------------------------------------------------------------- #
_settings_store: SettingsStore = SettingsStore()
_build_bootstrap_runner = default_build_bootstrap_runner
_bootstrap_session: _BootstrapSession | None = None
# Persistence stores. The lifespan hook lazily points these at on-disk
# SQLite files; tests monkeypatch them to in-memory or per-test paths.
_sweep_store: SweepStore | None = None
_session_store: SessionStore | None = None
_ai_store: AIStore | None = None


@asynccontextmanager
async def _lifespan(_app: FastAPI):  # type: ignore[no-untyped-def]
    from lorahub.api.store import JobStore, default_store_path

    if state.registry.store is None:
        store_path = default_store_path()
        store = JobStore(store_path)
        state.registry = state.JobRegistry(store=store)
        orphans = store.mark_orphans_interrupted()
        if orphans > 0:
            log.info(
                "marked %d orphaned job(s) from a previous session as interrupted",
                orphans,
            )
        loaded = state.registry.load_persisted()
        if loaded > 0:
            log.info("loaded %d job record(s) from %s", loaded, store_path)

    # Sibling stores: sweeps and sessions. Each gets its own SQLite file
    # so a corrupt or aggressively-locked DB on one side doesn't take
    # the rest of the API offline.
    global _sweep_store, _session_store, _ai_store  # noqa: PLW0603
    if _sweep_store is None:
        _sweep_store = SweepStore(default_sweep_store_path())
    if _session_store is None:
        _session_store = SessionStore(default_session_store_path())
    if _ai_store is None:
        _ai_store = AIStore(default_ai_store_path())
        # Seed empty routes for the LoraHub task ids so the Settings UI
        # has something to render on a fresh install. Each row carries
        # `enabled=True` + null provider/model — the user picks them in
        # the routes panel. We don't overwrite existing rows.
        from lorahub.api.ai_store import AIRoute  # noqa: PLC0415

        _LORAHUB_TASKS: tuple[tuple[str, str], ...] = (
            ("global.default", "未单独配置的任务都走这里"),
            ("tagging.assist", "VLM 给图补充 wd14 不擅长的描述"),
            ("caption.rewrite", "把 wd14 标签改写为自然语言或统一格式"),
            ("dataset.analyze", "对扫描结果做诊断 — caption 长度、tag 分布"),
            ("training.diagnose", "解读 loss / grad_norm 曲线给优化建议"),
            ("error.diagnose", "训练或安装失败时给修复建议"),
        )
        for task_id, hint in _LORAHUB_TASKS:
            if _ai_store.get_route(task_id) is None:
                _ai_store.upsert_route(
                    AIRoute(
                        task_id=task_id,
                        provider_id=None,
                        model_id=None,
                        system_prompt="",
                        enabled=True,
                    )
                )
                log.debug("seeded empty AI route for %s (%s)", task_id, hint)

    # Auto-resume: replay interrupted jobs that have a usable checkpoint.
    # Done before scheduler.start() so resumed work lands at the head of
    # the queue. Per-job `metadata.auto_resume` overrides the global
    # default in either direction; sweep children are always declined.
    try:
        _settings = _settings_store.load()
        from lorahub.api.jobs_helpers import (  # noqa: PLC0415
            _attempt_auto_resume,
            _requeue_pending_jobs,
        )

        resumed = _attempt_auto_resume(
            max_attempts=max(1, int(_settings.auto_resume_max_attempts)),
            global_default=bool(_settings.auto_resume_interrupted),
        )
        if resumed > 0:
            log.info("auto-resumed %d interrupted job(s) on startup", resumed)
        # Queued jobs (never started before the previous shutdown) need
        # their scheduler closure rebuilt; rows alone aren't enough.
        requeued = _requeue_pending_jobs()
        if requeued > 0:
            log.info("re-enqueued %d pending job(s) on startup", requeued)
    except Exception:  # noqa: BLE001
        log.exception("auto-resume hook failed; continuing startup")

    # Resize the module-level scheduler from persisted Settings before
    # workers start. We reach for the *current* `_settings_store` symbol
    # rather than a captured reference so test monkeypatches still apply.
    try:
        desired = max(1, int(_settings_store.load().max_concurrent_jobs))
    except Exception:  # noqa: BLE001
        desired = 1
    if desired != sched.scheduler.concurrency:
        log.info(
            "scheduler concurrency: %d -> %d (from settings.max_concurrent_jobs)",
            sched.scheduler.concurrency,
            desired,
        )
        # The default scheduler hasn't been start()-ed yet, but stop() is a
        # safe no-op when no workers exist, so it's harmless to call here.
        sched.scheduler.stop(timeout=2.0)
        sched.scheduler = sched.JobScheduler(
            concurrency=desired,
            available_slots=list(range(desired)),
        )
    sched.scheduler.start()
    yield
    sched.scheduler.stop(timeout=2.0)


app = FastAPI(
    title="LoraHub",
    version=__version__,
    description="HTTP API for the LoraHub training workbench.",
    lifespan=_lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Import routers AFTER the test-hook attributes above are bound, so the
# circular `routers -> app_module._foo` reference always sees populated names.
from lorahub.api.routers import all_routers  # noqa: E402

for _r in all_routers:
    app.include_router(_r)


@app.websocket("/api/jobs/{job_id}/stream")
async def stream_events(ws: WebSocket, job_id: str) -> None:
    job = state.registry.get(job_id)
    if job is None:
        await ws.close(code=4404)
        return
    await ws.accept()

    queue: asyncio.Queue[TrainingEvent] = asyncio.Queue(maxsize=512)
    state.registry.attach_listener(job_id, queue)
    try:
        replayed_terminal = False
        for ev in _job_events(job):  # replay buffered or persisted events
            await ws.send_json(ev.to_dict())
            replayed_terminal = ev.type is EventType.done
        terminal_state = job.state in {
            JobState.succeeded,
            JobState.failed,
            JobState.canceled,
            JobState.interrupted,
        }
        while not replayed_terminal and not terminal_state:
            ev = await queue.get()
            await ws.send_json(ev.to_dict())
            if ev.type is EventType.done:
                break
    except WebSocketDisconnect:
        pass
    finally:
        state.registry.detach_listener(job_id, queue)
        with contextlib.suppress(Exception):
            await ws.close()


@app.websocket("/api/backend/bootstrap/stream")
async def stream_bootstrap(ws: WebSocket) -> None:
    sess = _bootstrap_session
    if sess is None:
        await ws.close(code=4404)
        return
    await ws.accept()

    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=512)
    backlog = sess.attach(queue)
    try:
        for event in backlog:
            await ws.send_json(event)
        if not sess.is_running():
            return
        while True:
            event = await queue.get()
            if event.get("step") == "__terminal__":
                break
            await ws.send_json(event)
            if event.get("level") in {"done", "error"}:
                break
    except WebSocketDisconnect:
        pass
    finally:
        sess.detach(queue)
        with contextlib.suppress(Exception):
            await ws.close()


@app.websocket("/api/system/stream")
async def stream_system(ws: WebSocket) -> None:
    """Push a hardware/host snapshot every second until the client disconnects."""
    from lorahub.api.system_stats import collect_snapshot  # noqa: PLC0415

    await ws.accept()
    try:
        while True:
            try:
                await ws.send_json(collect_snapshot().to_dict())
            except WebSocketDisconnect:
                raise
            except Exception:  # noqa: BLE001
                pass
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        pass
    finally:
        with contextlib.suppress(Exception):
            await ws.close()


_WEB_DIST = _resolve_web_dist()
if _WEB_DIST is not None:
    _WEB_ROOT = _WEB_DIST
    _ASSETS_DIR = _WEB_ROOT / "assets"
    if _ASSETS_DIR.is_dir():
        app.mount("/assets", StaticFiles(directory=_ASSETS_DIR), name="assets")

    _INDEX = _WEB_ROOT / "index.html"

    @app.get("/", include_in_schema=False)
    def _index() -> FileResponse:
        return FileResponse(_INDEX)

    @app.get("/{full_path:path}", include_in_schema=False)
    def _spa_fallback(full_path: str) -> Response:
        # `/api`, `/docs`, etc. are owned by FastAPI — never shadow them.
        if full_path.startswith(("api/", "docs", "openapi.json", "redoc")):
            raise HTTPException(status_code=404, detail="not found")
        # Serve concrete static files from dist (favicon, robots.txt, …); else
        # fall back to index.html so React Router can take over.
        candidate = (_WEB_ROOT / full_path).resolve()
        try:
            candidate.relative_to(_WEB_ROOT)
        except ValueError:
            raise HTTPException(status_code=404, detail="not found") from None
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(_INDEX)
else:
    log.info("web/dist not found — serving API only (run `npm run build` in web/)")
