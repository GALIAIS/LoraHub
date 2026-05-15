"""LoraHub HTTP API.

Run with `lorahub serve` (preferred) or `uvicorn lorahub.api.app:app`.
The API surface is intentionally small for v0.2 — list/create/cancel jobs,
read recipe schema, stream events. Auth is out of scope; bind to localhost.

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
from lorahub.api import state
from lorahub.api.bootstrap_session import (
    _BootstrapSession,
    default_build_bootstrap_runner,
)
from lorahub.api.helpers import _resolve_web_dist
from lorahub.api.jobs_helpers import _job_events
from lorahub.api.settings import SettingsStore
from lorahub.api.state import JobState
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
    yield


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
