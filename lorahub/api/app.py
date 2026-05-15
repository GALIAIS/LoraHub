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
import threading
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import asdict
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


_IMAGE_SUFFIXES = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp"}


def _preflight_recipe(cfg: RecipeConfig) -> dict[str, Any]:
    backend = KohyaBackend()
    issues = [
        {
            **asdict(issue),
            "severity": issue.severity.value,
        }
        for issue in backend.validate(cfg)
    ]
    estimate = backend.estimate_vram(cfg)

    image_files: list[Path] = []
    caption_files = 0
    missing_caption_files: list[str] = []
    if cfg.dataset.source.is_dir():
        image_files = sorted(
            p
            for p in cfg.dataset.source.iterdir()
            if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
        )
        for image in image_files:
            if image.with_suffix(".txt").is_file():
                caption_files += 1
            else:
                missing_caption_files.append(image.name)

    return {
        "issues": issues,
        "vram": {
            "model_mib": estimate.model_mib,
            "optimizer_mib": estimate.optimizer_mib,
            "activations_mib": estimate.activations_mib,
            "overhead_mib": estimate.overhead_mib,
            "total_mib": estimate.total_mib,
            "total_gib": round(estimate.total_gib, 2),
        },
        "paths": {
            "checkpoint_exists": cfg.base_model.checkpoint.is_file(),
            "dataset_exists": cfg.dataset.source.is_dir(),
            "image_files": len(image_files),
            "caption_files": caption_files,
            "missing_caption_files": missing_caption_files[:20],
            "missing_caption_files_truncated": len(missing_caption_files) > 20,
        },
    }


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

    return {
        "valid": True,
        "normalized": cfg.model_dump(mode="json"),
        "preflight": _preflight_recipe(cfg),
    }


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


def _scan_dataset_path(path: Path, *, recursive: bool = False, limit: int = 40) -> dict[str, Any]:
    root = path.expanduser().resolve()
    exists = root.is_dir()
    image_files: list[Path] = []
    caption_files = 0
    missing_caption_files: list[str] = []
    samples: list[dict[str, Any]] = []

    if exists:
        iterator = root.rglob("*") if recursive else root.iterdir()
        image_files = sorted(
            p for p in iterator if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
        )
        for image in image_files:
            caption_path = image.with_suffix(".txt")
            caption: str | None = None
            if caption_path.is_file():
                caption_files += 1
                with contextlib.suppress(Exception):
                    caption = caption_path.read_text(encoding="utf-8").strip()
            else:
                missing_caption_files.append(image.relative_to(root).as_posix())
            if len(samples) < max(limit, 0):
                samples.append(
                    {
                        "name": image.name,
                        "path": str(image),
                        "relative_path": image.relative_to(root).as_posix(),
                        "caption_exists": caption_path.is_file(),
                        "caption": caption,
                    }
                )

    return {
        "path": str(root),
        "exists": exists,
        "recursive": recursive,
        "image_files": len(image_files),
        "caption_files": caption_files,
        "missing_caption_files": missing_caption_files[: max(limit, 0)],
        "missing_caption_files_truncated": len(missing_caption_files) > max(limit, 0),
        "samples": samples,
    }


@api.get("/datasets/scan")
def scan_dataset(path: str, recursive: bool = False, limit: int = 40) -> dict[str, Any]:
    return _scan_dataset_path(Path(path), recursive=recursive, limit=limit)


@api.get("/jobs")
def list_jobs() -> dict[str, Any]:
    return {"jobs": [j.to_summary() for j in state.registry.list()]}


@api.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    job = state.registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job.to_summary()


def _job_events(job: state.JobRecord, limit: int | None = None) -> list[TrainingEvent]:
    events = list(job.events)
    if not events:
        event_log = job.workspace / "events.jsonl"
        if event_log.is_file():
            with contextlib.suppress(Exception):
                events = list(JsonlEventSink.replay(event_log))
    if limit is not None:
        events = events[-max(limit, 0) :]
    return events


@api.get("/jobs/{job_id}/events")
def get_recent_events(job_id: str, limit: int = 100) -> dict[str, Any]:
    job = state.registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    events = _job_events(job, limit)
    return {"events": [e.to_dict() for e in events]}


class CreateJobRequest(BaseModel):
    recipe: dict[str, Any]
    workspace: str | None = None


def _launch_job(cfg: RecipeConfig, workspace: Path) -> dict[str, Any]:
    """Materialize a workspace, register a job, and start the kohya backend.

    Shared by both `POST /jobs` (fresh) and `POST /jobs/{id}/rerun`. The caller
    is responsible for resolving `workspace` (the rerun path needs a fresh dir
    so it doesn't collide with the original run's artifacts).
    """
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


@api.post("/jobs", status_code=202)
def create_job(req: CreateJobRequest) -> dict[str, Any]:
    try:
        cfg = RecipeConfig.model_validate(req.recipe)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=str(e)) from e

    workspace = Path(req.workspace).resolve() if req.workspace else (
        Path.cwd() / "runs" / cfg.output.name
    ).resolve()
    return _launch_job(cfg, workspace)


@api.post("/jobs/{job_id}/rerun", status_code=202)
def rerun_job(job_id: str) -> dict[str, Any]:
    """Start a fresh job from an existing job's recipe snapshot.

    The original job is left untouched. The new run gets its own workspace
    sibling to the original (suffixed with a short timestamp so the two never
    fight over the same `events.jsonl`).
    """
    job = state.registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")

    try:
        cfg = RecipeConfig.model_validate(job.recipe_snapshot)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=422, detail=f"recipe snapshot is no longer valid: {exc}"
        ) from exc

    base = job.workspace.resolve()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    workspace = (base.parent / f"{base.name}-rerun-{stamp}").resolve()
    return _launch_job(cfg, workspace)


@api.post("/jobs/{job_id}/reveal")
def reveal_job(job_id: str) -> dict[str, Any]:
    """Open the job's workspace directory in the host file browser.

    Local-first tool: the API process is on the user's machine, so we shell out
    to the platform's native file manager (`explorer`, `open`, `xdg-open`).
    Always uses an argv list — never `shell=True` — to avoid command injection
    via the workspace path.
    """
    import subprocess  # noqa: PLC0415
    import sys  # noqa: PLC0415

    job = state.registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")

    workspace = job.workspace
    if not workspace.exists():
        raise HTTPException(
            status_code=409, detail=f"workspace no longer exists: {workspace}"
        )

    if sys.platform == "win32":
        argv = ["explorer", str(workspace)]
    elif sys.platform == "darwin":
        argv = ["open", str(workspace)]
    else:
        argv = ["xdg-open", str(workspace)]

    try:
        subprocess.Popen(argv, close_fds=True)  # noqa: S603
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500, detail=f"file manager not available: {exc}"
        ) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"opened": str(workspace)}


_TERMINAL_STATES = (
    JobState.succeeded,
    JobState.failed,
    JobState.canceled,
    JobState.interrupted,
)


def _archive_workspace(workspace: Path, job_id: str) -> tuple[Path | None, list[str]]:
    """Move `workspace` under `<parent>/_archive/<name>-<short_id>`.

    Returns (new_path, warnings). `new_path` is None if the workspace did not
    exist or the move failed; in either case the caller still drops the store
    record and the warnings explain what happened.
    """
    warnings: list[str] = []
    if not workspace.exists():
        warnings.append(f"workspace did not exist: {workspace}")
        return None, warnings

    parent = workspace.parent
    archive_dir = parent / "_archive"
    short_id = job_id[-8:] if len(job_id) > 8 else job_id
    target = archive_dir / f"{workspace.name}-{short_id}"

    try:
        archive_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        warnings.append(f"could not create archive dir {archive_dir}: {exc}")
        return None, warnings

    # Avoid clobbering an existing archive entry from a previous archive of the
    # same workspace name — append a counter until we find a free slot.
    final = target
    counter = 1
    while final.exists():
        final = archive_dir / f"{workspace.name}-{short_id}-{counter}"
        counter += 1

    try:
        workspace.rename(final)
    except OSError as exc:
        warnings.append(f"could not move workspace: {exc}")
        return None, warnings

    return final, warnings


@api.delete("/jobs/{job_id}")
def cancel_job(job_id: str, archive: bool = False) -> dict[str, Any]:
    job = state.registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")

    if archive:
        if job.state not in _TERMINAL_STATES:
            raise HTTPException(
                status_code=409,
                detail=f"job is {job.state.value}; cancel before archiving",
            )
        moved, warnings = _archive_workspace(job.workspace, job.id)
        state.registry.delete(job.id)
        return {
            "archived": True,
            "workspace_moved_to": str(moved) if moved is not None else None,
            "warnings": warnings,
        }

    if job.state in _TERMINAL_STATES:
        return job.to_summary()
    job.state = JobState.canceling
    state.registry.update(job)
    if job.handle is not None:
        job.handle.stop(graceful=True)
    return job.to_summary()


# --------------------------------------------------------------------------- #
# Backend bootstrap (one-click kohya install)
# --------------------------------------------------------------------------- #


class _BootstrapSession:
    """Singleton wrapper around `installer.bootstrap` running on a worker thread.

    The session buffers structured events for late-joining HTTP polls and fans
    them out to attached `asyncio.Queue` listeners for the WebSocket stream.
    Each install step turns into one event; a final `done` or `error` event
    marks the terminal state and triggers listener wake-ups so they can close.
    """

    _STATUS_RUNNING = "running"
    _STATUS_SUCCEEDED = "succeeded"
    _STATUS_FAILED = "failed"

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.status: str = self._STATUS_RUNNING
        self.events: list[dict[str, Any]] = []
        self._listeners: list[asyncio.Queue[dict[str, Any]]] = []
        self._lock = threading.RLock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    def attach(self, queue: asyncio.Queue[dict[str, Any]]) -> list[dict[str, Any]]:
        """Register a listener and return the buffered backlog atomically."""
        with self._lock:
            self._listeners.append(queue)
            return list(self.events)

    def detach(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        with self._lock:
            if queue in self._listeners:
                self._listeners.remove(queue)

    def is_running(self) -> bool:
        return self.status == self._STATUS_RUNNING

    def to_status_payload(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": self.status,
                "session_id": self.session_id,
                "events": list(self.events),
            }

    def start(
        self,
        runner: Callable[[Callable[[str], None]], None],
        loop: asyncio.AbstractEventLoop | None,
    ) -> None:
        """Spawn the worker thread that calls `runner(progress_cb)`."""
        self._loop = loop
        self._thread = threading.Thread(
            target=self._run, args=(runner,), name="lorahub-bootstrap", daemon=True
        )
        self._thread.start()

    def _run(self, runner: Callable[[Callable[[str], None]], None]) -> None:
        try:
            runner(lambda step: self._emit("info", step, message=step))
        except Exception as exc:  # noqa: BLE001 — surface any installer failure
            step = getattr(exc, "step", "bootstrap")
            self._emit("error", step, message=str(exc))
            self._finalize(self._STATUS_FAILED)
            return
        self._emit("done", "complete", message="kohya backend installed")
        self._finalize(self._STATUS_SUCCEEDED)

    def _emit(self, level: str, step: str, *, message: str) -> None:
        event = {
            "step": step,
            "level": level,
            "message": message,
            "ts": datetime.now(UTC).timestamp(),
        }
        with self._lock:
            self.events.append(event)
            listeners = list(self._listeners)
        for queue in listeners:
            self._dispatch(queue, event)

    def _finalize(self, status: str) -> None:
        with self._lock:
            self.status = status
            listeners = list(self._listeners)
        # Wake any listener still parked on `queue.get()` so it can close.
        sentinel: dict[str, Any] = {"step": "__terminal__", "level": status}
        for queue in listeners:
            self._dispatch(queue, sentinel)

    def _dispatch(self, queue: asyncio.Queue[dict[str, Any]], event: dict[str, Any]) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(queue.put_nowait, event)
        except RuntimeError:
            # Loop already torn down — listener is gone, drop the event.
            pass


_bootstrap_session: _BootstrapSession | None = None
_bootstrap_lock = threading.Lock()


class BootstrapRequest(BaseModel):
    target: str | None = None
    cuda: str = "cu124"
    torch_version: str = "2.6.0"
    torchvision_version: str = "0.21.0"
    install_xformers: bool = True
    force: bool = False


def _build_bootstrap_runner(req: BootstrapRequest) -> Callable[[Callable[[str], None]], None]:
    """Produce a (progress_cb -> None) closure that runs the kohya installer.

    Factored out so tests can monkeypatch this builder with a stub runner that
    doesn't touch the network or the filesystem.
    """
    from lorahub.core.backends.kohya import installer  # noqa: PLC0415

    target_path = (
        Path(req.target).expanduser().resolve()
        if req.target
        else (Path.cwd() / "sd-scripts").resolve()
    )
    plan = installer.BootstrapPlan(
        target=target_path,
        cuda_version=req.cuda,
        torch_version=req.torch_version,
        torchvision_version=req.torchvision_version,
        install_xformers=req.install_xformers,
    )
    if plan.target.exists() and any(plan.target.iterdir()):
        if not req.force:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"target {plan.target} is not empty; "
                    "pass force=true to wipe it first."
                ),
            )
        installer.cleanup_partial(plan)

    def runner(progress: Callable[[str], None]) -> None:
        installer.bootstrap(plan, progress=progress)

    return runner


@api.get("/backend/bootstrap/status")
def bootstrap_status() -> dict[str, Any]:
    sess = _bootstrap_session
    if sess is None:
        return {"status": "idle", "session_id": None, "events": []}
    return sess.to_status_payload()


@api.post("/backend/bootstrap", status_code=202)
async def start_bootstrap(req: BootstrapRequest) -> dict[str, Any]:
    global _bootstrap_session
    with _bootstrap_lock:
        existing = _bootstrap_session
        if existing is not None and existing.is_running():
            raise HTTPException(
                status_code=409, detail="a bootstrap session is already running"
            )
        # Resolve the runner first — this validates the target dir before we
        # spin a thread. HTTPException raised here surfaces as a 4xx directly.
        runner = _build_bootstrap_runner(req)
        sess = _BootstrapSession(session_id=str(ulid_new()))
        _bootstrap_session = sess

    loop = asyncio.get_running_loop()
    sess.start(runner, loop)
    return {"session_id": sess.session_id, "status": sess.status}


def ulid_new() -> Any:
    """Wrapper so tests can patch ULID generation if needed."""
    import ulid  # noqa: PLC0415

    return ulid.new()


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
