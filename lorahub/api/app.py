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

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
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
from lorahub.api.image_studio_store import (
    ImageStudioStore,
    default_image_studio_store_path,
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
_image_studio_store: ImageStudioStore | None = None


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
    global _sweep_store, _session_store, _ai_store, _image_studio_store  # noqa: PLW0603
    if _sweep_store is None:
        _sweep_store = SweepStore(default_sweep_store_path())
    if _session_store is None:
        _session_store = SessionStore(default_session_store_path())
    if _ai_store is None:
        _ai_store = AIStore(default_ai_store_path())
    if _image_studio_store is None:
        _image_studio_store = ImageStudioStore(default_image_studio_store_path())
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
            ("quality.score", "VLM 评估图片质量 (0-100 + 优/中/差)"),
            ("trigger.suggest", "根据数据集特征建议 trigger word"),
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
            _migrate_snapshots_to_camel,
            _requeue_pending_jobs,
        )

        # Catch-up migration: ensure every JobRecord.config_snapshot is
        # camelCase so the resume-with-edit form sees the same shape
        # newer jobs use. No-op once a process has been through this
        # pass; safe to run on every boot.
        _migrate_snapshots_to_camel()

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

    # Sweep restart recovery: rebuild MaterialisedSweep instances for
    # any TPE sweep whose study sqlite file still exists. Without this,
    # in-flight TPE sweeps silently degrade after a server restart —
    # dangling RUNNING trials in the RDB never get their score, and
    # the live registry stays empty so the next ask() ignores prior
    # work. Best-effort: a missing optuna install just skips it.
    try:
        from lorahub.api import sweep_runtime  # noqa: PLC0415

        sweep_runtime.rebuild_active_sweeps(state, _sweep_store)
    except Exception:  # noqa: BLE001
        log.exception("sweep rebuild hook failed; continuing startup")

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


# Idle-traffic heartbeat: AutoDL / nginx / similar reverse proxies tend to
# axe a websocket that goes silent for ~60s. We push a tiny `ping` frame
# every WS_PING_INTERVAL seconds while waiting on the queue so the proxy
# always sees activity. Clients ignore type=ping frames.
_WS_PING_INTERVAL = 25.0

# Same idea for SSE — a `: comment` line counts as activity but is
# discarded by EventSource so it never reaches the client's onmessage.
_SSE_PING_INTERVAL = 25.0


def _sse_format(*, data: str | None = None, event: str | None = None,
                event_id: str | None = None, comment: str | None = None,
                retry_ms: int | None = None) -> str:
    """Format one SSE message frame.

    Spec: each line prefixed with `field: value`, blank line terminates
    the message. Multi-line `data` is split into multiple `data:` lines.
    """
    out: list[str] = []
    if comment is not None:
        out.append(f": {comment}")
    if event_id is not None:
        out.append(f"id: {event_id}")
    if event is not None:
        out.append(f"event: {event}")
    if retry_ms is not None:
        out.append(f"retry: {retry_ms}")
    if data is not None:
        for line in data.splitlines() or [""]:
            out.append(f"data: {line}")
    out.append("")  # blank line = end of message
    return "\n".join(out) + "\n"


def _resume_index_from_header(request: Request) -> int:
    """Pick the resume offset from the SSE Last-Event-ID header.

    Returns the index of the *next* event to send; 0 means "send everything
    from the beginning". Bad input falls back to 0 so a corrupt cookie
    can't deadlock the stream.
    """
    raw = request.headers.get("last-event-id")
    if not raw:
        return 0
    try:
        return max(0, int(raw) + 1)
    except (TypeError, ValueError):
        return 0


@app.get("/api/jobs/{job_id}/sse")
async def stream_events_sse(job_id: str, request: Request) -> StreamingResponse:
    """SSE counterpart to /api/jobs/{job_id}/stream.

    Each event is tagged with `id: <index>` where index = position in
    the merged (replayed + live) sequence. On reconnect EventSource
    sends `Last-Event-ID` automatically and we skip past it, so the
    client never sees duplicates and never misses an event.

    The legacy WS endpoint is preserved as a fallback.
    """
    job = state.registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")

    resume_from = _resume_index_from_header(request)

    async def gen() -> Any:  # AsyncIterator[str]
        # Hint the client to back off 2s on a clean reconnect.
        yield _sse_format(retry_ms=2000, comment="lorahub job stream")

        queue: asyncio.Queue[TrainingEvent] = asyncio.Queue(maxsize=512)
        state.registry.attach_listener(job_id, queue)
        sent = 0
        try:
            replayed_terminal = False
            for ev in _job_events(job):
                if sent >= resume_from:
                    yield _sse_format(
                        event_id=str(sent),
                        data=ev.to_json(),
                    )
                replayed_terminal = ev.type is EventType.done
                sent += 1

            terminal_state = job.state in {
                JobState.succeeded,
                JobState.failed,
                JobState.canceled,
                JobState.interrupted,
            }
            while not replayed_terminal and not terminal_state:
                if await request.is_disconnected():
                    return
                try:
                    ev = await asyncio.wait_for(queue.get(), timeout=_SSE_PING_INTERVAL)
                except asyncio.TimeoutError:
                    yield _sse_format(comment="ping")
                    continue
                yield _sse_format(event_id=str(sent), data=ev.to_json())
                sent += 1
                if ev.type is EventType.done:
                    break
        finally:
            state.registry.detach_listener(job_id, queue)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            # Disable response buffering so events flush immediately
            # past nginx and AutoDL's reverse proxy.
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/api/backend/bootstrap/sse")
async def stream_bootstrap_sse(request: Request) -> StreamingResponse:
    sess = _bootstrap_session
    if sess is None:
        raise HTTPException(status_code=404, detail="no bootstrap session")

    resume_from = _resume_index_from_header(request)

    async def gen() -> Any:
        import json as _json  # noqa: PLC0415

        yield _sse_format(retry_ms=2000, comment="lorahub bootstrap stream")

        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=512)
        backlog = sess.attach(queue)
        sent = 0
        try:
            for event in backlog:
                if sent >= resume_from:
                    yield _sse_format(
                        event_id=str(sent),
                        data=_json.dumps(event),
                    )
                sent += 1
            if not sess.is_running():
                return
            while True:
                if await request.is_disconnected():
                    return
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=_SSE_PING_INTERVAL)
                except asyncio.TimeoutError:
                    yield _sse_format(comment="ping")
                    continue
                if event.get("step") == "__terminal__":
                    break
                yield _sse_format(event_id=str(sent), data=_json.dumps(event))
                sent += 1
                if event.get("level") in {"done", "error"}:
                    break
        finally:
            sess.detach(queue)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/api/system/sse")
async def stream_system_sse(request: Request) -> StreamingResponse:
    """SSE telemetry — a host snapshot every second.

    No replay buffer here: snapshots are stateless, the freshest one is
    always good enough, so Last-Event-ID is ignored.
    """
    from lorahub.api.system_stats import collect_snapshot  # noqa: PLC0415
    import json as _json  # noqa: PLC0415

    async def gen() -> Any:
        yield _sse_format(retry_ms=2000, comment="lorahub system stream")
        sent = 0
        while True:
            if await request.is_disconnected():
                return
            try:
                snap = collect_snapshot().to_dict()
                yield _sse_format(event_id=str(sent), data=_json.dumps(snap))
                sent += 1
            except Exception:  # noqa: BLE001
                pass
            await asyncio.sleep(1.0)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


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
            try:
                ev = await asyncio.wait_for(queue.get(), timeout=_WS_PING_INTERVAL)
            except asyncio.TimeoutError:
                # Heartbeat: keep the proxy from cutting the idle channel.
                await ws.send_json({"type": "ping"})
                continue
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
            try:
                event = await asyncio.wait_for(queue.get(), timeout=_WS_PING_INTERVAL)
            except asyncio.TimeoutError:
                await ws.send_json({"type": "ping"})
                continue
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

    # `index.html` references hashed chunk filenames under /assets. After a
    # redeploy the chunk hashes change and the old ones get rotated out;
    # the only way for browsers to discover the new ones is to re-fetch
    # index.html. Setting `Cache-Control: no-store` on the entry point —
    # but NOT on /assets/* (those are content-hashed and immutable) —
    # forces every navigation to revalidate the HTML while preserving CDN
    # / browser caching for the heavy JS / CSS bundles.
    _INDEX_HEADERS = {
        "Cache-Control": "no-store, must-revalidate",
        "Pragma": "no-cache",
    }

    @app.get("/", include_in_schema=False)
    def _index() -> FileResponse:
        return FileResponse(_INDEX, headers=_INDEX_HEADERS)

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
            # Static files other than index.html (favicon, woff2 fonts, …)
            # don't carry hashes; let the browser cache them lightly.
            return FileResponse(candidate)
        # SPA route fallback hits index.html — same no-store treatment so
        # deep-link refreshes pick up the latest chunk pointers.
        return FileResponse(_INDEX, headers=_INDEX_HEADERS)
else:
    log.info("web/dist not found — serving API only (run `npm run build` in web/)")
