"""Model download endpoints with progress-tracked background sessions."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from lorahub.api import app as app_module
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


class DownloadModelRequest(BaseModel):
    source: Literal["huggingface", "modelscope"]
    repo_id: str
    revision: str = "master"
    target_dir: str | None = None
    threads: int = Field(default=4, ge=1, le=16)
    paths: list[str] = Field(default_factory=list, max_length=2048)
    allow_patterns: list[str] | None = None
    ignore_patterns: list[str] | None = None


class ListModelFilesRequest(BaseModel):
    source: Literal["huggingface", "modelscope"]
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
    status: Literal["running", "succeeded", "failed"] = "running"
    percent: float = 0
    events: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add_progress(self, event: DownloadProgress) -> None:
        with self.lock:
            if event.percent is not None:
                self.percent = max(self.percent, min(100, float(event.percent)))
            self.events.append(asdict(event) | {"ts": time.time()})
            self.events = self.events[-200:]

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
    with _sessions_lock:
        _sessions[session.session_id] = session


def _get_session(session_id: str) -> _DownloadSession:
    with _sessions_lock:
        session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="download session not found")
    return session


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
    session = _DownloadSession(
        session_id=uuid.uuid4().hex,
        source=req.source,
        repo_id=req.repo_id,
        revision=req.revision,
        target_dir=str(target) if target else None,
        threads=req.threads,
        paths=list(req.paths),
    )
    session.add_progress(DownloadProgress(message="download queued", percent=0))
    _store_session(session)

    def run() -> None:
        try:
            result = download(download_req, session.add_progress)
            with session.lock:
                session.status = "succeeded"
                session.percent = 100
                session.result = _result_payload(req, result)
                session.finished_at = time.time()
            session.add_progress(DownloadProgress(message="download complete", percent=100))
        except Exception as exc:  # noqa: BLE001
            with session.lock:
                session.status = "failed"
                session.error = str(exc)
                session.finished_at = time.time()
            session.add_progress(DownloadProgress(message=f"download failed: {exc}"))

    thread = threading.Thread(target=run, name=f"model-download-{session.session_id[:8]}", daemon=True)
    thread.start()
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
