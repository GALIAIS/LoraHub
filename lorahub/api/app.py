"""LoraHub HTTP API.

Run with `lorahub serve` (preferred) or `uvicorn lorahub.api.app:app`.
The API surface is intentionally small for v0.2 — list/create/cancel jobs,
read recipe schema, stream events. Auth is out of scope; bind to localhost.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from lorahub import __version__
from lorahub.api import state
from lorahub.api.state import JobState
from lorahub.core.backends.kohya.backend import KohyaBackend
from lorahub.core.config.loader import dump_recipe
from lorahub.core.config.schema import RecipeConfig
from lorahub.core.events import EventType, JsonlEventSink, TrainingEvent

log = logging.getLogger(__name__)


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


class HealthResponse(BaseModel):
    status: str
    version: str


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__)


@app.get("/recipes/schema")
def recipe_schema() -> dict[str, Any]:
    """JSON Schema for the recipe — used by the future UI to render forms."""
    return RecipeConfig.model_json_schema()


@app.get("/jobs")
def list_jobs() -> dict[str, Any]:
    return {"jobs": [j.to_summary() for j in state.registry.list()]}


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    job = state.registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job.to_summary()


@app.get("/jobs/{job_id}/events")
def get_recent_events(job_id: str, limit: int = 100) -> dict[str, Any]:
    job = state.registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    events = list(job.events)[-limit:]
    return {"events": [e.to_dict() for e in events]}


class CreateJobRequest(BaseModel):
    recipe: dict[str, Any]
    workspace: str | None = None


@app.post("/jobs", status_code=202)
def create_job(req: CreateJobRequest) -> dict[str, Any]:
    try:
        cfg = RecipeConfig.model_validate(req.recipe)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=str(e)) from e

    workspace = Path(req.workspace).resolve() if req.workspace else (
        Path.cwd() / "runs" / cfg.output.name
    ).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    snapshot = cfg.model_dump(mode="json")
    job = state.registry.create(workspace=workspace, recipe_snapshot=snapshot)
    dump_recipe(cfg, workspace / "recipe.yaml")

    backend = KohyaBackend()

    sink = JsonlEventSink(workspace / "events.jsonl")
    sink.__enter__()

    def on_event(ev: TrainingEvent) -> None:
        sink(ev)
        state.registry.record_event(job.id, ev)
        if ev.type is EventType.done:
            j = state.registry.get(job.id)
            if j is not None:
                rc = ev.payload.get("returncode")
                j.returncode = rc
                j.state = JobState.succeeded if rc == 0 else JobState.failed
                j.finished_at = datetime.now(UTC)
                state.registry.update(j)
            sink.__exit__(None, None, None)

    handle = backend.launch(cfg, workspace=workspace, on_event=on_event)
    job.handle = handle
    job.pid = handle.pid
    job.state = JobState.running
    job.started_at = datetime.now(UTC)
    state.registry.update(job)

    return job.to_summary()


@app.delete("/jobs/{job_id}")
def cancel_job(job_id: str) -> dict[str, Any]:
    job = state.registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.state in (JobState.succeeded, JobState.failed, JobState.canceled, JobState.interrupted):
        return job.to_summary()
    job.state = JobState.canceling
    state.registry.update(job)
    if job.handle is not None:
        job.handle.stop(graceful=True)
    return job.to_summary()


@app.websocket("/jobs/{job_id}/stream")
async def stream_events(ws: WebSocket, job_id: str) -> None:
    job = state.registry.get(job_id)
    if job is None:
        await ws.close(code=4404)
        return
    await ws.accept()

    queue: asyncio.Queue[TrainingEvent] = asyncio.Queue(maxsize=512)
    state.registry.attach_listener(job_id, queue)
    try:
        for ev in list(job.events):  # replay buffered events
            await ws.send_json(ev.to_dict())
        while True:
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
