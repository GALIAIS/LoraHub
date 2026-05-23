"""Backend catalog: lists training backends and their install state.

Read-only -- the bootstrap router (`POST /api/backend/bootstrap`) is what
actually installs one. The catalog gives the UI the metadata it needs to
render the "Backends" panel without each frontend route having to hand-roll
its own probe.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from lorahub.api import app as app_module
from lorahub.api.backend_update import apply_update, check_update
from lorahub.api.settings import probe_all_backends
from lorahub.core.backends.anima_lora.models import (
    DownloadEvent,
    download_models as _download_anima_models,
    missing_files as _anima_missing_models,
)
from lorahub.core.backends.anima_lora import msvc as _anima_msvc
from lorahub.core.backends.diffusion_pipe.bootstrap import (
    default_repo_path as _dp_default_repo,
)
from lorahub.core.backends.kohya.bootstrap import (
    default_sd_scripts_path as _kohya_default_path,
)
from lorahub.core.backends.registry import list_backends

router = APIRouter(prefix="/api")


class BackendEntry(BaseModel):
    id: str
    name: str
    description: str
    repo_url: str
    default_path: str
    ready: bool
    status: dict[str, Any]


class BackendsResponse(BaseModel):
    backends: list[BackendEntry]
    default: str


@router.get("/backends", response_model=BackendsResponse)
def list_backend_catalog() -> BackendsResponse:
    settings = app_module._settings_store.load()
    probes = probe_all_backends(settings)
    entries: list[BackendEntry] = []
    for desc in list_backends():
        status = probes.get(desc.id, {})
        entries.append(
            BackendEntry(
                id=desc.id,
                name=desc.name,
                description=desc.description,
                repo_url=desc.repo_url,
                default_path=str(desc.default_path),
                ready=bool(status.get("ready", False)),
                status=status,
            )
        )
    return BackendsResponse(backends=entries, default=settings.default_backend)


# --------------------------------------------------------------------------- #
# Backend git update detection
# --------------------------------------------------------------------------- #


def _resolve_backend_repo(backend_id: str) -> Path | None:
    """Resolve the git repo path for a given backend id."""
    settings = app_module._settings_store.load()
    if backend_id == "kohya":
        explicit = settings.sd_scripts_path
        return Path(explicit) if explicit else _kohya_default_path()
    elif backend_id == "diffusion-pipe":
        explicit = settings.diffusion_pipe_repo_path
        return Path(explicit) if explicit else _dp_default_repo()
    return None


@router.get("/backends/{backend_id}/check-update")
def check_backend_update(backend_id: str) -> dict[str, Any]:
    """Check if a backend repo has upstream updates available."""
    if backend_id not in ("kohya", "diffusion-pipe"):
        raise HTTPException(
            status_code=400,
            detail=f"update check not supported for backend {backend_id!r}",
        )
    repo = _resolve_backend_repo(backend_id)
    if repo is None or not repo.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"backend repo not found at {repo}",
        )
    result = check_update(repo)
    return {"backend_id": backend_id, "repo_path": str(repo), **result.to_dict()}


@router.post("/backends/{backend_id}/update")
def update_backend(backend_id: str) -> dict[str, Any]:
    """Pull the latest commits for a backend repo."""
    if backend_id not in ("kohya", "diffusion-pipe"):
        raise HTTPException(
            status_code=400,
            detail=f"update not supported for backend {backend_id!r}",
        )
    repo = _resolve_backend_repo(backend_id)
    if repo is None or not repo.is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"backend repo not found at {repo}",
        )
    result = apply_update(repo)
    if result.error:
        raise HTTPException(status_code=500, detail=result.error)
    return {"backend_id": backend_id, "repo_path": str(repo), **result.to_dict()}


# --------------------------------------------------------------------------- #
# Anima model download — separate from the generic /api/models/download
# endpoint because we want a single button that grabs exactly the three
# files anima needs and lays them out under the unified <root>/models/
# tree (instead of under models/<repo_id>/split_files/...).
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class _AnimaModelSession:
    session_id: str
    status: Literal["running", "succeeded", "failed"] = "running"
    percent: float = 0
    files_done: int = 0
    files_total: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add_event(self, event: DownloadEvent) -> None:
        with self.lock:
            self.percent = max(self.percent, min(100, float(event.percent)))
            self.files_done = event.files_done
            self.files_total = event.files_total
            self.events.append(asdict(event) | {"ts": time.time()})
            self.events = self.events[-200:]

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "session_id": self.session_id,
                "status": self.status,
                "percent": self.percent,
                "files_done": self.files_done,
                "files_total": self.files_total,
                "events": list(self.events),
                "error": self.error,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
            }


_anima_sessions: dict[str, _AnimaModelSession] = {}
_anima_sessions_lock = threading.Lock()
_anima_active_session: str | None = None


@router.post("/backends/anima_lora/download-models", status_code=202)
def start_anima_model_download() -> dict[str, Any]:
    """Kick off a background download of the three anima checkpoints.

    Refuses if another download is already in flight (returns its
    existing snapshot via 409). The download writes into
    ``<lorahub_root>/models/{diffusion_models,text_encoders,vae}/``.
    """
    global _anima_active_session

    with _anima_sessions_lock:
        if _anima_active_session is not None:
            existing = _anima_sessions.get(_anima_active_session)
            if existing and existing.status == "running":
                raise HTTPException(
                    status_code=409,
                    detail={
                        "message": "another anima model download is already running",
                        "session": existing.snapshot(),
                    },
                )

    settings = app_module._settings_store.load()
    session = _AnimaModelSession(session_id=uuid.uuid4().hex)
    session.add_event(DownloadEvent("queued", 0, 0, 0))

    with _anima_sessions_lock:
        _anima_sessions[session.session_id] = session
        _anima_active_session = session.session_id

    def run() -> None:
        global _anima_active_session
        try:
            _download_anima_models(
                huggingface_endpoint=settings.huggingface_endpoint,
                huggingface_token=settings.huggingface_token,
                proxy=settings.download_proxy,
                threads=3,
                progress=session.add_event,
            )
            with session.lock:
                session.status = "succeeded"
                session.percent = 100
                session.finished_at = time.time()
        except Exception as exc:  # noqa: BLE001
            with session.lock:
                session.status = "failed"
                session.error = str(exc)
                session.finished_at = time.time()
            session.add_event(DownloadEvent(f"failed: {exc}", session.percent, session.files_done, session.files_total))
        finally:
            with _anima_sessions_lock:
                if _anima_active_session == session.session_id:
                    _anima_active_session = None

    thread = threading.Thread(
        target=run,
        name=f"anima-models-{session.session_id[:8]}",
        daemon=True,
    )
    thread.start()
    return session.snapshot()


@router.get("/backends/anima_lora/download-models/status")
def anima_model_download_status() -> dict[str, Any]:
    """Poll the most recent (or in-flight) anima model download session."""
    with _anima_sessions_lock:
        sid = _anima_active_session
        if sid is None:
            # Return the most recent finished session, if any, so the UI
            # can show "succeeded" after the user navigates away and back.
            recent = max(
                _anima_sessions.values(),
                key=lambda s: s.started_at,
                default=None,
            )
            if recent is None:
                return {
                    "status": "idle",
                    "missing_files": _anima_missing_models(),
                }
            return recent.snapshot() | {"missing_files": _anima_missing_models()}
        session = _anima_sessions.get(sid)
    if session is None:
        return {"status": "idle", "missing_files": _anima_missing_models()}
    return session.snapshot() | {"missing_files": _anima_missing_models()}


# --------------------------------------------------------------------------- #
# Visual Studio Build Tools installer — Windows only.
#
# Driven by ``winget install Microsoft.VisualStudio.2022.BuildTools`` with
# the C++ workload + Win 11 SDK selected via ``--override``. We capture
# winget's stdout/stderr line-by-line into a session log so the UI can
# stream a tail; winget itself prints rich progress indicators that
# don't translate well to a percent bar, so we just expose the latest
# few lines and let the user decide if it's progressing.
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class _MsvcInstallSession:
    session_id: str
    status: Literal["running", "succeeded", "failed"] = "running"
    log: list[str] = field(default_factory=list)
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def append_log(self, line: str) -> None:
        with self.lock:
            self.log.append(line)
            # Cap the buffer — winget can emit hundreds of progress
            # lines via its spinner. Tail-style 200-line cap keeps
            # the snapshot small enough to JSON-encode cheaply.
            if len(self.log) > 200:
                self.log = self.log[-200:]

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "session_id": self.session_id,
                "status": self.status,
                "log": list(self.log),
                "error": self.error,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
            }


_msvc_sessions: dict[str, _MsvcInstallSession] = {}
_msvc_sessions_lock = threading.Lock()
_msvc_active_session: str | None = None


@router.post("/backends/anima_lora/install-msvc", status_code=202)
def start_msvc_install() -> dict[str, Any]:
    """Drive ``winget install Microsoft.VisualStudio.2022.BuildTools``.

    Refuses to start when MSVC is already detected (no point) or when
    another install session is already in flight (409). Returns the
    initial session snapshot otherwise; clients poll
    ``GET .../install-msvc/status`` for updates.
    """
    global _msvc_active_session

    detection = _anima_msvc.detect()
    if detection.installed:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "already_installed",
                "message": "MSVC build tools already detected",
                "cl_path": detection.cl_path,
                "msvc_version": detection.msvc_version,
            },
        )

    cmd = _anima_msvc.install_command()
    if cmd is None:
        raise HTTPException(
            status_code=409,
            detail=(
                "winget is not available on this host — install "
                "https://aka.ms/winget-cli or run the Build Tools "
                "installer manually from "
                "https://aka.ms/vs/17/release/vs_BuildTools.exe"
            ),
        )

    with _msvc_sessions_lock:
        if _msvc_active_session is not None:
            existing = _msvc_sessions.get(_msvc_active_session)
            if existing and existing.status == "running":
                raise HTTPException(
                    status_code=409,
                    detail={
                        "message": "another MSVC install is already running",
                        "session": existing.snapshot(),
                    },
                )

    session = _MsvcInstallSession(session_id=uuid.uuid4().hex)
    session.append_log("queued: " + " ".join(cmd))
    with _msvc_sessions_lock:
        _msvc_sessions[session.session_id] = session
        _msvc_active_session = session.session_id

    def run() -> None:
        global _msvc_active_session
        import subprocess as _sp  # noqa: PLC0415
        try:
            proc = _sp.Popen(
                cmd,
                stdout=_sp.PIPE,
                stderr=_sp.STDOUT,
                stdin=_sp.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    session.append_log(line)
            rc = proc.wait()
            with session.lock:
                if rc == 0:
                    session.status = "succeeded"
                else:
                    session.status = "failed"
                    session.error = f"winget exited with code {rc}"
                session.finished_at = time.time()
        except Exception as exc:  # noqa: BLE001
            with session.lock:
                session.status = "failed"
                session.error = str(exc)
                session.finished_at = time.time()
            session.append_log(f"failed: {exc}")
        finally:
            with _msvc_sessions_lock:
                if _msvc_active_session == session.session_id:
                    _msvc_active_session = None

    thread = threading.Thread(
        target=run,
        name=f"msvc-install-{session.session_id[:8]}",
        daemon=True,
    )
    thread.start()
    return session.snapshot()


@router.get("/backends/anima_lora/install-msvc/status")
def msvc_install_status() -> dict[str, Any]:
    """Poll the most recent (or in-flight) MSVC install session.

    Always re-runs detection so a freshly-completed install flips
    ``msvc_ok`` to True without the user having to refresh the
    backends list.
    """
    fresh = _anima_msvc.detect()
    detection = {
        "ok": fresh.installed,
        "cl_path": fresh.cl_path,
        "msvc_version": fresh.msvc_version,
        "reason": fresh.reason,
    }
    with _msvc_sessions_lock:
        sid = _msvc_active_session
        if sid is None:
            recent = max(
                _msvc_sessions.values(),
                key=lambda s: s.started_at,
                default=None,
            )
            if recent is None:
                return {"status": "idle", "msvc": detection}
            return recent.snapshot() | {"msvc": detection}
        session = _msvc_sessions.get(sid)
    if session is None:
        return {"status": "idle", "msvc": detection}
    return session.snapshot() | {"msvc": detection}
