"""LoraHub HTTP API.

Run with `lorahub serve` (preferred) or `uvicorn lorahub.api.app:app`.
The API surface is intentionally small for v0.2 — list/create/cancel jobs,
read recipe schema, stream events. Auth is out of scope; bind to localhost.

All API routes live under `/api`. The site root and `/{spa-path}` are reserved
for the React frontend (mounted from `web/dist` when present).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from lorahub import __version__
from lorahub.api import state
from lorahub.api.settings import Settings, SettingsStore, probe_backend
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

api = APIRouter(prefix="/api")


# Module-level so tests can monkeypatch in an isolated SettingsStore via
# `app._settings_store = SettingsStore(tmp_path)` before issuing requests.
_settings_store: SettingsStore = SettingsStore()


def _store() -> SettingsStore:
    return _settings_store


class HealthResponse(BaseModel):
    status: str
    version: str
    backend: dict[str, Any]


@api.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=__version__,
        backend=probe_backend(_store().load()),
    )


class SettingsResponse(BaseModel):
    settings: dict[str, Any]
    backend: dict[str, Any]
    path: str


@api.get("/settings", response_model=SettingsResponse)
def get_settings() -> SettingsResponse:
    store = _store()
    s = store.load()
    return SettingsResponse(
        settings=s.to_dict(),
        backend=probe_backend(s),
        path=str(store.path),
    )


class UpdateSettingsRequest(BaseModel):
    sd_scripts_path: str | None = None
    python_executable: str | None = None
    tagger_device: str | None = None


@api.put("/settings", response_model=SettingsResponse)
def update_settings(req: UpdateSettingsRequest) -> SettingsResponse:
    store = _store()
    current = store.load()

    # Treat empty strings as "clear this field".
    def _norm(v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip()
        return v or None

    new = Settings(
        sd_scripts_path=_norm(req.sd_scripts_path),
        python_executable=_norm(req.python_executable),
        tagger_device=(req.tagger_device or current.tagger_device or "auto").strip() or "auto",
        extra=current.extra,
    )
    if new.tagger_device not in {"auto", "cpu", "cuda"}:
        raise HTTPException(
            status_code=422,
            detail=f"tagger_device must be auto/cpu/cuda, got {new.tagger_device!r}",
        )
    store.save(new)
    return SettingsResponse(
        settings=new.to_dict(),
        backend=probe_backend(new),
        path=str(store.path),
    )


@api.get("/recipes/schema")
def recipe_schema() -> dict[str, Any]:
    """JSON Schema for the recipe — used by the future UI to render forms."""
    return RecipeConfig.model_json_schema()


def _recipes_dir() -> Path:
    """Resolve the recipes/ directory.

    Honors $LORAHUB_RECIPES_DIR (absolute path); otherwise looks at
    `<cwd>/recipes` so users get whatever templates ship with their checkout
    when running `lorahub serve` from the repo root.
    """
    override = os.environ.get("LORAHUB_RECIPES_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return (Path.cwd() / "recipes").resolve()


def _recipe_path(name: str) -> Path:
    """Resolve a recipe by name within the recipes/ dir, blocking traversal."""
    if not name or "/" in name or "\\" in name or name.startswith(".."):
        raise HTTPException(status_code=400, detail="invalid recipe name")
    base = _recipes_dir()
    # Accept "foo" or "foo.yaml"
    candidates = [base / name, base / f"{name}.yaml", base / f"{name}.yml"]
    for c in candidates:
        c_resolved = c.resolve()
        try:
            c_resolved.relative_to(base)
        except ValueError:
            continue
        if c_resolved.is_file():
            return c_resolved
    raise HTTPException(status_code=404, detail="recipe not found")


@api.get("/recipes")
def list_recipes() -> dict[str, Any]:
    """List YAML recipe templates discovered under the recipes/ directory."""
    base = _recipes_dir()
    if not base.is_dir():
        return {"dir": str(base), "recipes": []}

    from lorahub.core.config.loader import load_recipe  # noqa: PLC0415

    items: list[dict[str, Any]] = []
    for p in sorted(base.glob("*.y*ml")):
        if p.suffix.lower() not in {".yaml", ".yml"}:
            continue
        entry: dict[str, Any] = {
            "name": p.stem,
            "filename": p.name,
            "size": p.stat().st_size,
            "valid": False,
            "arch": None,
            "summary": None,
            "error": None,
        }
        try:
            cfg = load_recipe(p)
            entry["valid"] = True
            entry["arch"] = cfg.base_model.arch
            entry["summary"] = (
                f"{cfg.base_model.arch} · "
                f"{cfg.schedule.epochs} epoch(s) × bs {cfg.schedule.batch_size}"
            )
        except Exception as exc:  # noqa: BLE001
            entry["error"] = str(exc).splitlines()[0][:200]
        items.append(entry)
    return {"dir": str(base), "recipes": items}


@api.get("/recipes/{name}")
def get_recipe(name: str) -> dict[str, Any]:
    """Return a recipe's raw YAML and parsed dict (for previewing or launching)."""
    if name in {"schema", "validate"}:  # sibling endpoints share the prefix
        raise HTTPException(status_code=404, detail="recipe not found")
    path = _recipe_path(name)

    from lorahub.core.config.loader import load_recipe  # noqa: PLC0415

    raw = path.read_text(encoding="utf-8")
    parsed: dict[str, Any] | None = None
    error: str | None = None
    try:
        parsed = load_recipe(path).model_dump(mode="json")
    except Exception as exc:  # noqa: BLE001
        error = str(exc)
    return {
        "name": path.stem,
        "filename": path.name,
        "path": str(path),
        "content": raw,
        "parsed": parsed,
        "error": error,
    }


class ValidateRecipeRequest(BaseModel):
    recipe: dict[str, Any]


@api.post("/recipes/validate")
def validate_recipe(req: ValidateRecipeRequest) -> dict[str, Any]:
    """Validate a recipe payload without persisting or training.

    Always returns 200 — the response carries `valid: bool` and a list of
    structured field errors. This lets the form highlight bad fields without
    interpreting HTTP status codes.
    """
    from pydantic import ValidationError as _PydanticValidationError  # noqa: PLC0415

    try:
        cfg = RecipeConfig.model_validate(req.recipe)
    except _PydanticValidationError as exc:
        return {
            "valid": False,
            "errors": [
                {
                    "loc": list(e.get("loc", [])),
                    "msg": e.get("msg", ""),
                    "type": e.get("type", ""),
                }
                for e in exc.errors()
            ],
        }
    except Exception as exc:  # noqa: BLE001
        return {"valid": False, "errors": [{"loc": [], "msg": str(exc), "type": "internal"}]}

    return {"valid": True, "normalized": cfg.model_dump(mode="json")}


_NAME_RE_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$"


class SaveRecipeRequest(BaseModel):
    name: str
    recipe: dict[str, Any]
    overwrite: bool = False


@api.post("/recipes", status_code=201)
def save_recipe(req: SaveRecipeRequest) -> dict[str, Any]:
    """Validate and persist a recipe to recipes/<name>.yaml."""
    import re  # noqa: PLC0415

    name = req.name.strip().removesuffix(".yaml").removesuffix(".yml")
    if not re.match(_NAME_RE_PATTERN, name):
        raise HTTPException(
            status_code=400,
            detail="name must match [A-Za-z0-9][A-Za-z0-9_.-]{0,63}",
        )

    try:
        cfg = RecipeConfig.model_validate(req.recipe)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    base = _recipes_dir()
    base.mkdir(parents=True, exist_ok=True)
    target = (base / f"{name}.yaml").resolve()
    try:
        target.relative_to(base)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid name") from exc

    if target.exists() and not req.overwrite:
        raise HTTPException(
            status_code=409,
            detail=f"recipe {name!r} already exists; pass overwrite=true to replace",
        )

    dump_recipe(cfg, target)
    return {
        "name": name,
        "filename": target.name,
        "path": str(target),
        "overwritten": target.exists(),
    }


@api.get("/jobs")
def list_jobs() -> dict[str, Any]:
    return {"jobs": [j.to_summary() for j in state.registry.list()]}


@api.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    job = state.registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job.to_summary()


@api.get("/jobs/{job_id}/events")
def get_recent_events(job_id: str, limit: int = 100) -> dict[str, Any]:
    job = state.registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    events = list(job.events)[-limit:]
    return {"events": [e.to_dict() for e in events]}


class CreateJobRequest(BaseModel):
    recipe: dict[str, Any]
    workspace: str | None = None


@api.post("/jobs", status_code=202)
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


@api.delete("/jobs/{job_id}")
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


app.include_router(api)


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


def _resolve_web_dist() -> Path | None:
    """Locate the built web frontend (`web/dist`).

    Search order:
      1. $LORAHUB_WEB_DIST (explicit override, e.g. for packaged installs)
      2. <repo_root>/web/dist (development checkout)
    """
    override = os.environ.get("LORAHUB_WEB_DIST")
    if override:
        candidate = Path(override).expanduser().resolve()
        return candidate if (candidate / "index.html").is_file() else None

    repo_dist = Path(__file__).resolve().parents[2] / "web" / "dist"
    if (repo_dist / "index.html").is_file():
        return repo_dist
    return None


_WEB_DIST = _resolve_web_dist()
if _WEB_DIST is not None:
    _ASSETS_DIR = _WEB_DIST / "assets"
    if _ASSETS_DIR.is_dir():
        app.mount("/assets", StaticFiles(directory=_ASSETS_DIR), name="assets")

    _INDEX = _WEB_DIST / "index.html"

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
        candidate = (_WEB_DIST / full_path).resolve()
        try:
            candidate.relative_to(_WEB_DIST)
        except ValueError:
            raise HTTPException(status_code=404, detail="not found") from None
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(_INDEX)
else:
    log.info("web/dist not found — serving API only (run `npm run build` in web/)")
