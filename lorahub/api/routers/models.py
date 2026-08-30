"""Model download endpoints with progress-tracked background sessions."""

from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, cast

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from lorahub.api import app as app_module
from lorahub.api.paths import models_dir, resolve_model_path
from lorahub.api.task_sessions import (
    TaskEvent,
    TaskSession,
    TaskSessionStore,
    default_task_store_path,
    persist_stop_request,
    prune_terminal_session_cache,
)
from lorahub.core.models.downloader import (
    DEFAULT_ALLOW_PATTERNS,
    DEFAULT_IGNORE_PATTERNS,
    DownloadCanceledError,
    DownloadProgress,
    DownloadRequest,
    DownloadResult,
    download,
    list_remote_files,
    select_files,
    validate_repo_id,
)
from lorahub.core.redaction import redact_command_text

router = APIRouter(prefix="/api")
_KIND_MODEL_DOWNLOAD = "model_download"
_DownloadStatus = Literal[
    "running",
    "stop_requested",
    "succeeded",
    "failed",
    "canceled",
    "interrupted",
]


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
    status: _DownloadStatus = "running"
    percent: float = 0
    events: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)

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

    def request_stop(self) -> bool:
        with self.lock:
            if self.status != "running":
                return False
            percent = self.percent
            persisted = persist_stop_request(
                _task_store(),
                self.session_id,
                percent=percent,
            )
            if not persisted:
                return False
            self.cancel_event.set()
            self.status = "stop_requested"
        self.add_progress(DownloadProgress(message="cancel requested", percent=self.percent))
        return True


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
    status: _DownloadStatus = (
        cast(_DownloadStatus, task.status)
        if task.status
        in {"stop_requested", "succeeded", "failed", "interrupted", "canceled"}
        else "running"
    )

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
    try:
        validate_repo_id(repo_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _validate_selected_paths(paths: list[str]) -> None:
    if not paths:
        return
    try:
        select_files([], paths=tuple(paths))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _resolve_target_dir(target_dir: str | None) -> Path | None:
    if not target_dir:
        return None
    try:
        return resolve_model_path(target_dir)
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"invalid target_dir: {exc}") from exc


def _paths_overlap(first: Path, second: Path) -> bool:
    try:
        first.relative_to(second)
        return True
    except ValueError:
        pass
    try:
        second.relative_to(first)
        return True
    except ValueError:
        return False


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
    _validate_selected_paths(req.paths)
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
    global _latest_session_id

    _validate_repo_id(req.repo_id)
    _validate_selected_paths(req.paths)

    target = _resolve_target_dir(req.target_dir) or resolve_model_path(
        req.repo_id.replace("/", "__")
    )
    download_req = _download_request_from_api(req, target=target, threads=req.threads)
    target_key = target.resolve()
    with _sessions_lock:
        for active in _sessions.values():
            if (
                active.status in {"running", "stop_requested"}
                and active.target_dir is not None
                and _paths_overlap(Path(active.target_dir).resolve(), target_key)
            ):
                raise HTTPException(
                    status_code=409,
                    detail=f"another download is already writing to {target_key}",
                )
        task = _task_store().create(
            kind=_KIND_MODEL_DOWNLOAD,
            title=f"{req.source}:{req.repo_id}",
            metadata={
                "source": req.source,
                "repo_id": req.repo_id,
                "revision": req.revision,
                "target_dir": str(target),
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
            target_dir=str(target),
            threads=req.threads,
            paths=list(req.paths),
            allow_patterns=list(download_req.allow_patterns),
            ignore_patterns=list(download_req.ignore_patterns),
        )
        _sessions[session.session_id] = session
        prune_terminal_session_cache(_sessions)
        _latest_session_id = session.session_id
    session.add_progress(DownloadProgress(message="download queued", percent=0))
    download_req = replace(download_req, cancel_event=session.cancel_event)

    def run() -> None:
        try:
            with session.lock:
                if session.cancel_event.is_set():
                    raise DownloadCanceledError("download canceled by user")
                _task_store().update(
                    session.session_id,
                    status="running",
                    percent=session.percent,
                )
            result = download(download_req, session.add_progress)
            result_payload = _result_payload(req, result)
            with session.lock:
                if session.cancel_event.is_set():
                    raise DownloadCanceledError("download canceled by user")
                session.status = "succeeded"
                session.percent = 100
                session.result = result_payload
                session.finished_at = time.time()
            session.add_progress(DownloadProgress(message="download complete", percent=100))
            _task_store().update(
                session.session_id,
                status="succeeded",
                percent=100,
                result=result_payload,
                finished=True,
            )
        except DownloadCanceledError as exc:
            with session.lock:
                session.status = "canceled"
                session.error = str(exc)
                session.finished_at = time.time()
            session.add_progress(DownloadProgress(message=str(exc), percent=session.percent))
            _task_store().update(
                session.session_id,
                status="canceled",
                error=str(exc),
                result=session.snapshot(),
                finished=True,
            )
        except Exception as exc:  # noqa: BLE001
            error = redact_command_text(str(exc))
            with session.lock:
                session.status = "failed"
                session.error = error
                session.finished_at = time.time()
            _task_store().update(
                session.session_id,
                status="failed",
                error=error,
                finished=True,
            )
            session.add_progress(DownloadProgress(message=f"download failed: {error}"))

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


@router.post("/models/download/{session_id}/stop")
def stop_model_download(session_id: str) -> dict[str, Any]:
    with _sessions_lock:
        session = _sessions.get(session_id)
    if session is None:
        task = _task_store().get(session_id)
        if task is None or task.kind != _KIND_MODEL_DOWNLOAD:
            raise HTTPException(status_code=404, detail="download session not found")
        raise HTTPException(status_code=409, detail=f"download is {task.status}")
    if not session.request_stop():
        with session.lock:
            status = session.status
        raise HTTPException(status_code=409, detail=f"download is {status}")
    return {"session_id": session_id, "status": "stop_requested"}


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
    try:
        base = resolve_model_path(root, allow_root=True) if root else models_dir().resolve()
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"invalid model root: {exc}") from exc
    files = _scan_models_dir(base)
    return ScannedModelsResponse(
        root=str(base),
        files=files,
        elapsed_s=round(time.monotonic() - started, 3),
    )
