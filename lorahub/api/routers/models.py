"""Model download endpoints with progress-tracked background sessions."""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from lorahub.api import app as app_module
from lorahub.api.task_sessions import (
    TaskEvent,
    TaskSession,
    TaskSessionStore,
    default_task_store_path,
)
from lorahub.core.models.downloader import (
    DEFAULT_ALLOW_PATTERNS,
    DEFAULT_IGNORE_PATTERNS,
    DownloadProgress,
    DownloadRequest,
    DownloadResult,
    download,
    list_remote_files,
)

router = APIRouter(prefix="/api")
_KIND_MODEL_DOWNLOAD = "model_download"


class DownloadModelRequest(BaseModel):
    source: Literal["huggingface", "modelscope"] = "modelscope"
    repo_id: str
    revision: str = "master"
    target_dir: str | None = None
    threads: int = Field(default=4, ge=1, le=16)
    paths: list[str] = Field(default_factory=list, max_length=2048)
    allow_patterns: list[str] | None = None
    ignore_patterns: list[str] | None = None


class ListModelFilesRequest(BaseModel):
    source: Literal["huggingface", "modelscope"] = "modelscope"
    repo_id: str
    revision: str = "master"
    paths: list[str] = Field(default_factory=list, max_length=2048)
    allow_patterns: list[str] | None = None
    ignore_patterns: list[str] | None = None


@dataclass(slots=True)
class _DownloadSession:
    session_id: str
    source: str
    repo_id: str
    revision: str
    target_dir: str | None
    threads: int
    paths: list[str]
    allow_patterns: list[str] = field(default_factory=list)
    ignore_patterns: list[str] = field(default_factory=list)
    status: Literal["running", "succeeded", "failed"] = "running"
    percent: float = 0
    events: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add_progress(self, event: DownloadProgress) -> None:
        payload = asdict(event)
        ts = time.time()
        with self.lock:
            if event.percent is not None:
                self.percent = max(self.percent, min(100, float(event.percent)))
            self.events.append(payload | {"ts": ts})
            self.events = self.events[-200:]
        try:
            _task_store().append_event(
                self.session_id,
                TaskEvent(
                    level="info",
                    message=event.message,
                    percent=event.percent,
                    payload=payload,
                    ts=ts,
                ),
            )
        except Exception:
            pass

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "session_id": self.session_id,
                "source": self.source,
                "repo_id": self.repo_id,
                "revision": self.revision,
                "target_dir": self.target_dir,
                "threads": self.threads,
                "paths": list(self.paths),
                "allow_patterns": list(self.allow_patterns),
                "ignore_patterns": list(self.ignore_patterns),
                "status": self.status,
                "percent": self.percent,
                "events": list(self.events),
                "result": self.result,
                "error": self.error,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
            }


_sessions: dict[str, _DownloadSession] = {}
_sessions_lock = threading.Lock()
_latest_session_id: str | None = None


def _task_store() -> TaskSessionStore:
    store = app_module._task_session_store
    if store is None:
        store = TaskSessionStore(default_task_store_path())
        app_module._task_session_store = store
    return store


def _result_payload(req: DownloadModelRequest, result: DownloadResult) -> dict[str, Any]:
    return {
        "source": req.source,
        "repo_id": req.repo_id,
        "revision": req.revision,
        "target": str(result.target),
        "files": result.files,
        "total_bytes": result.total_bytes,
    }


def _store_session(session: _DownloadSession) -> None:
    global _latest_session_id
    with _sessions_lock:
        _sessions[session.session_id] = session
        _latest_session_id = session.session_id


def _get_session(session_id: str) -> _DownloadSession:
    with _sessions_lock:
        session = _sessions.get(session_id)
    if session is not None:
        return session
    task = _task_store().get(session_id)
    if task is None or task.kind != _KIND_MODEL_DOWNLOAD:
        raise HTTPException(status_code=404, detail="download session not found")
    return _session_from_task(task)


def _latest_session() -> _DownloadSession | None:
    with _sessions_lock:
        if _latest_session_id is not None:
            session = _sessions.get(_latest_session_id)
            if session is not None:
                return session
        session = max(_sessions.values(), key=lambda s: s.started_at, default=None)
    if session is not None:
        return session
    task = _task_store().latest(_KIND_MODEL_DOWNLOAD)
    if task is None:
        return None
    return _session_from_task(task)


def _session_from_task(task: TaskSession) -> _DownloadSession:
    metadata = task.metadata
    status: Literal["running", "succeeded", "failed"]
    if task.status == "succeeded":
        status = "succeeded"
    elif task.status in {"failed", "interrupted", "canceled"}:
        status = "failed"
    else:
        status = "running"

    return _DownloadSession(
        session_id=task.id,
        source=str(metadata.get("source") or "modelscope"),
        repo_id=str(metadata.get("repo_id") or ""),
        revision=str(metadata.get("revision") or "master"),
        target_dir=metadata.get("target_dir"),
        threads=int(metadata.get("threads") or 4),
        paths=list(metadata.get("paths") or []),
        allow_patterns=list(metadata.get("allow_patterns") or []),
        ignore_patterns=list(metadata.get("ignore_patterns") or []),
        status=status,
        percent=task.percent,
        events=[
            (dict(event.payload) if event.payload else {})
            | {"message": event.message, "percent": event.percent, "ts": event.ts}
            for event in task.events
        ],
        result=task.result,
        error=task.error,
        started_at=task.started_at,
        finished_at=task.finished_at,
    )


def _validate_repo_id(repo_id: str) -> None:
    if not repo_id or "/" not in repo_id:
        raise HTTPException(status_code=400, detail="repo_id must be 'owner/name'")


def _download_request_from_api(
    req: DownloadModelRequest | ListModelFilesRequest,
    *,
    target: Path | None = None,
    threads: int = 4,
) -> DownloadRequest:
    settings = app_module._settings_store.load()
    return DownloadRequest(
        source=req.source,
        repo_id=req.repo_id,
        revision=req.revision,
        target_dir=target,
        huggingface_endpoint=settings.huggingface_endpoint,
        huggingface_token=settings.huggingface_token,
        modelscope_token=settings.modelscope_token,
        threads=threads,
        proxy=settings.download_proxy,
        paths=tuple(req.paths),
        allow_patterns=(
            tuple(req.allow_patterns)
            if req.allow_patterns is not None
            else DEFAULT_ALLOW_PATTERNS
        ),
        ignore_patterns=(
            tuple(req.ignore_patterns)
            if req.ignore_patterns is not None
            else DEFAULT_IGNORE_PATTERNS
        ),
    )


@router.post("/models/files")
def list_model_files(req: ListModelFilesRequest) -> dict[str, Any]:
    _validate_repo_id(req.repo_id)
    files = list_remote_files(_download_request_from_api(req))
    selected = [f for f in files if f.selected]
    return {
        "source": req.source,
        "repo_id": req.repo_id,
        "revision": req.revision,
        "files": [
            {
                "path": f.path,
                "size": f.size,
                "selected": f.selected,
                "reason": f.reason,
            }
            for f in files
        ],
        "selected_count": len(selected),
        "selected_bytes": sum(f.size for f in selected),
        "total_count": len(files),
        "total_bytes": sum(f.size for f in files),
    }


@router.post("/models/download", status_code=202)
def download_model(req: DownloadModelRequest) -> dict[str, Any]:
    _validate_repo_id(req.repo_id)

    target = Path(req.target_dir).expanduser().resolve() if req.target_dir else None
    download_req = _download_request_from_api(req, target=target, threads=req.threads)
    task = _task_store().create(
        kind=_KIND_MODEL_DOWNLOAD,
        title=f"{req.source}:{req.repo_id}",
        metadata={
            "source": req.source,
            "repo_id": req.repo_id,
            "revision": req.revision,
            "target_dir": str(target) if target else None,
            "threads": req.threads,
            "paths": list(req.paths),
            "allow_patterns": list(download_req.allow_patterns),
            "ignore_patterns": list(download_req.ignore_patterns),
        },
    )
    session = _DownloadSession(
        session_id=task.id,
        source=req.source,
        repo_id=req.repo_id,
        revision=req.revision,
        target_dir=str(target) if target else None,
        threads=req.threads,
        paths=list(req.paths),
        allow_patterns=list(download_req.allow_patterns),
        ignore_patterns=list(download_req.ignore_patterns),
    )
    session.add_progress(DownloadProgress(message="download queued", percent=0))
    _store_session(session)

    def run() -> None:
        try:
            _task_store().update(
                session.session_id,
                status="running",
                percent=session.percent,
            )
            result = download(download_req, session.add_progress)
            result_payload = _result_payload(req, result)
            session.add_progress(DownloadProgress(message="download complete", percent=100))
            with session.lock:
                session.status = "succeeded"
                session.percent = 100
                session.result = result_payload
                session.finished_at = time.time()
            _task_store().update(
                session.session_id,
                status="succeeded",
                percent=100,
                result=result_payload,
                finished=True,
            )
        except Exception as exc:  # noqa: BLE001
            with session.lock:
                session.status = "failed"
                session.error = str(exc)
                session.finished_at = time.time()
            _task_store().update(
                session.session_id,
                status="failed",
                error=str(exc),
                finished=True,
            )
            session.add_progress(DownloadProgress(message=f"download failed: {exc}"))

    thread = threading.Thread(target=run, name=f"model-download-{session.session_id[:8]}", daemon=True)
    thread.start()
    return session.snapshot()


@router.get("/models/download/latest")
def latest_model_download_status() -> dict[str, Any]:
    session = _latest_session()
    if session is None:
        return {
            "session_id": None,
            "status": "idle",
            "events": [],
            "result": None,
            "error": None,
            "percent": 0,
        }
    return session.snapshot()


@router.get("/models/download/{session_id}")
def download_model_status(session_id: str) -> dict[str, Any]:
    return _get_session(session_id).snapshot()


# ---- model scan -----------------------------------------------------

class ScannedModel(BaseModel):
    """One entry in a /models/scan response."""

    path: str
    relative_path: str
    name: str
    size_bytes: int
    mtime: float


class ScannedModelsResponse(BaseModel):
    root: str
    files: list[ScannedModel]
    elapsed_s: float


_MODEL_EXTS = (".safetensors", ".sft", ".ckpt", ".pt", ".bin", ".gguf")


def _scan_models_dir(root: Path) -> list[ScannedModel]:
    if not root.is_dir():
        return []
    out: list[ScannedModel] = []
    # ``rglob`` walks the entire tree which is what users want (they
    # often nest models under ``models/<vendor>/<arch>/``). Hidden
    # dotfiles are skipped because some platforms drop ``.DS_Store`` /
    # ``._foo`` siblings that aren't real model files.
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        if path.suffix.lower() not in _MODEL_EXTS:
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        try:
            relative = str(path.relative_to(root)).replace("\\", "/")
        except ValueError:
            relative = str(path)
        out.append(
            ScannedModel(
                path=str(path),
                relative_path=relative,
                name=path.name,
                size_bytes=stat.st_size,
                mtime=stat.st_mtime,
            )
        )
    out.sort(key=lambda m: m.relative_path.lower())
    return out


@router.get("/models/scan", response_model=ScannedModelsResponse)
def scan_models(root: str | None = None) -> ScannedModelsResponse:
    """Walk the configured models folder and return discovered weights.

    Triggered manually from the UI's model picker. Response is built
    on each request — caching it would race with users dropping new
    files into the directory between calls. With a typical ``models/``
    of < 1k files the walk completes in < 100ms even on Windows.
    """
    started = time.monotonic()
    base = Path(root).expanduser() if root else (Path.cwd() / "models")
    base = base.resolve()
    files = _scan_models_dir(base)
    return ScannedModelsResponse(
        root=str(base),
        files=files,
        elapsed_s=round(time.monotonic() - started, 3),
    )
