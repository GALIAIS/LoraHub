"""WD14 / JoyTag auto-tagging endpoints with progress-tracked background sessions."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from lorahub.core.dataset.anima import AnimaDatasetTransformer
from lorahub.core.tagging.base import BaseTagger
from lorahub.core.tagging.joytag import DEFAULT_THRESHOLD as JOYTAG_DEFAULT_THRESHOLD
from lorahub.core.tagging.joytag import JoyTagger
from lorahub.core.tagging.wd14 import DEFAULT_MODEL, CudaUnavailableError, WD14Tagger

router = APIRouter(prefix="/api")


class TagDatasetRequest(BaseModel):
    path: str
    tagger: Literal["wd14", "joytag"] = "wd14"
    model_id: str = DEFAULT_MODEL
    general: float = Field(default=0.35, ge=0.0, le=1.0)
    character: float = Field(default=0.85, ge=0.0, le=1.0)
    # JoyTag's single threshold; ignored when tagger='wd14'.
    joytag_threshold: float = Field(default=JOYTAG_DEFAULT_THRESHOLD, ge=0.0, le=1.0)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    overwrite: bool = False
    recursive: bool = False
    include_character: bool = True
    underscores: bool = False


@dataclass(slots=True)
class _TaggingSession:
    session_id: str
    path: str
    tagger: str
    model_id: str
    device: str
    general: float
    character: float
    joytag_threshold: float
    overwrite: bool
    recursive: bool
    include_character: bool
    underscores: bool
    status: Literal["running", "succeeded", "failed"] = "running"
    percent: float = 0.0
    events: list[dict[str, Any]] = field(default_factory=list)
    written: int = 0
    total: int | None = None
    active_provider: str = ""
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def push(self, message: str, *, percent: float | None = None, image: str | None = None) -> None:
        with self.lock:
            if percent is not None:
                self.percent = max(self.percent, min(100.0, float(percent)))
            self.events.append(
                {"ts": time.time(), "message": message, "percent": self.percent, "image": image}
            )
            self.events = self.events[-200:]

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "session_id": self.session_id,
                "path": self.path,
                "tagger": self.tagger,
                "model_id": self.model_id,
                "device": self.device,
                "general": self.general,
                "character": self.character,
                "joytag_threshold": self.joytag_threshold,
                "overwrite": self.overwrite,
                "recursive": self.recursive,
                "include_character": self.include_character,
                "underscores": self.underscores,
                "status": self.status,
                "percent": self.percent,
                "events": list(self.events),
                "written": self.written,
                "total": self.total,
                "active_provider": self.active_provider,
                "error": self.error,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
            }


_sessions: dict[str, _TaggingSession] = {}
_sessions_lock = threading.Lock()


def _store(session: _TaggingSession) -> None:
    with _sessions_lock:
        _sessions[session.session_id] = session


def _get(session_id: str) -> _TaggingSession:
    with _sessions_lock:
        session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="tagging session not found")
    return session


def _get_persisted_tagging(session_id: str) -> dict[str, Any] | None:
    """Look up a tagging session in `sessions.sqlite`.

    Both wd14/joytag tagging runs and the anima caption rewriter persist
    into the same `tagging` table — the snapshot blob carries the kind
    information for the caller to dispatch on.
    """
    try:
        from lorahub.api import app as _app  # noqa: PLC0415

        store = getattr(_app, "_session_store", None)
        if store is None:
            return None
        row = store.get("tagging", session_id)
    except Exception:  # noqa: BLE001
        return None
    if row is None:
        return None
    snap = row.get("snapshot")
    return snap if isinstance(snap, dict) else None


def _persist_tagging_snapshot(session: Any) -> None:
    """Best-effort flush of a session snapshot to the SessionStore."""
    try:
        from lorahub.api import app as _app  # noqa: PLC0415

        store = getattr(_app, "_session_store", None)
        if store is not None:
            store.upsert_tagging(session.snapshot())
    except Exception:  # noqa: BLE001
        pass


# Indirection so tests can monkeypatch the tagger class without touching the
# concrete implementations. Returns a `BaseTagger`-conformant instance so the
# session loop is backend-agnostic.
def _build_tagger(req: TagDatasetRequest) -> BaseTagger:
    if req.tagger == "joytag":
        return JoyTagger(
            predict_threshold=req.joytag_threshold,
            device=req.device,
        )
    return WD14Tagger(
        model_id=req.model_id,
        general_threshold=req.general,
        character_threshold=req.character,
        device=req.device,
    )


@router.post("/tagging/tag", status_code=202)
def tag_dataset(req: TagDatasetRequest) -> dict[str, Any]:
    target = Path(req.path).expanduser()
    if not target.is_dir():
        raise HTTPException(status_code=400, detail=f"not a directory: {target}")

    session = _TaggingSession(
        session_id=uuid.uuid4().hex,
        path=str(target),
        tagger=req.tagger,
        model_id=req.model_id,
        device=req.device,
        general=req.general,
        character=req.character,
        joytag_threshold=req.joytag_threshold,
        overwrite=req.overwrite,
        recursive=req.recursive,
        include_character=req.include_character,
        underscores=req.underscores,
    )
    session.push("tagging queued", percent=0)
    _store(session)

    def run() -> None:
        try:
            tagger = _build_tagger(req)
            session.push(f"loading {req.tagger}")
            tagger.load()
            with session.lock:
                session.active_provider = tagger.active_provider
            session.push(f"running on {tagger.active_provider}", percent=2)

            # Pre-count for percent. Use the WD14 helper since both taggers
            # consume the same image extension set; this is just a glob.
            from lorahub.core.tagging.wd14 import _iter_images  # noqa: PLC0415

            all_images = list(_iter_images(target, recursive=req.recursive))
            with session.lock:
                session.total = len(all_images)
            if not all_images:
                session.push("no images found", percent=100)
                with session.lock:
                    session.status = "succeeded"
                    session.percent = 100
                    session.finished_at = time.time()
                return

            total = len(all_images)

            def on_progress(image: Path, _result: object) -> None:
                with session.lock:
                    session.written += 1
                    pct = min(100.0, 2 + 98 * session.written / max(total, 1))
                session.push(f"tagged {image.name}", percent=pct, image=str(image))

            tagger.tag_directory(
                target,
                recursive=req.recursive,
                write_caption=True,
                skip_existing=not req.overwrite,
                underscores=req.underscores,
                include_character=req.include_character,
                on_progress=on_progress,
            )
            with session.lock:
                session.status = "succeeded"
                session.percent = 100
                session.finished_at = time.time()
            session.push(f"done — wrote {session.written} caption(s)", percent=100)
        except CudaUnavailableError as exc:
            with session.lock:
                session.status = "failed"
                session.error = str(exc)
                session.finished_at = time.time()
            session.push(f"cuda unavailable: {exc}")
        except Exception as exc:  # noqa: BLE001
            with session.lock:
                session.status = "failed"
                session.error = str(exc)
                session.finished_at = time.time()
            session.push(f"tagging failed: {exc}")
        finally:
            _persist_tagging_snapshot(session)

    threading.Thread(
        target=run, name=f"tag-{req.tagger}-{session.session_id[:8]}", daemon=True
    ).start()
    return session.snapshot()


@router.get("/tagging/tag/{session_id}")
def tag_dataset_status(session_id: str) -> dict[str, Any]:
    with _sessions_lock:
        session = _sessions.get(session_id)
    if session is not None:
        return session.snapshot()
    persisted = _get_persisted_tagging(session_id)
    if persisted is not None:
        return persisted
    raise HTTPException(status_code=404, detail="tagging session not found")


@router.get("/tagging/tag")
def list_tagging_sessions(limit: int = 50) -> dict[str, Any]:
    """Recent tagging runs — live in-memory take precedence over persisted."""
    out: dict[str, dict[str, Any]] = {}
    try:
        from lorahub.api import app as _app  # noqa: PLC0415

        store = getattr(_app, "_session_store", None)
        if store is not None:
            for row in store.list_recent("tagging", limit=limit):
                snap = row.get("snapshot")
                if isinstance(snap, dict):
                    out[snap["session_id"]] = snap
    except Exception:  # noqa: BLE001
        pass
    with _sessions_lock:
        for sid, sess in _sessions.items():
            out[sid] = sess.snapshot()
    sessions = sorted(
        out.values(),
        key=lambda s: s.get("started_at") or 0,
        reverse=True,
    )[:limit]
    return {"sessions": sessions}


# --------------------------------------------------------------------------- #
# Anima caption rewriter (text-only — no tagger inference)
# --------------------------------------------------------------------------- #


class AnimaCaptionRequest(BaseModel):
    path: str
    dataset_tag: str | None = None
    quality: list[str] | None = None
    score: list[str] | None = None
    year: list[str] | None = None
    safety: str | None = "safe"
    overwrite: bool = False
    recursive: bool = False


@dataclass(slots=True)
class _AnimaSession:
    session_id: str
    path: str
    dataset_tag: str | None
    quality: list[str] | None
    score: list[str] | None
    year: list[str] | None
    safety: str | None
    overwrite: bool
    recursive: bool
    status: Literal["running", "succeeded", "failed"] = "running"
    percent: float = 0.0
    events: list[dict[str, Any]] = field(default_factory=list)
    written: int = 0
    total: int | None = None
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def push(self, message: str, *, percent: float | None = None, file: str | None = None) -> None:
        with self.lock:
            if percent is not None:
                self.percent = max(self.percent, min(100.0, float(percent)))
            self.events.append(
                {"ts": time.time(), "message": message, "percent": self.percent, "file": file}
            )
            self.events = self.events[-200:]

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "session_id": self.session_id,
                "path": self.path,
                "dataset_tag": self.dataset_tag,
                "quality": self.quality,
                "score": self.score,
                "year": self.year,
                "safety": self.safety,
                "overwrite": self.overwrite,
                "recursive": self.recursive,
                "status": self.status,
                "percent": self.percent,
                "events": list(self.events),
                "written": self.written,
                "total": self.total,
                "error": self.error,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
            }


_anima_sessions: dict[str, _AnimaSession] = {}
_anima_sessions_lock = threading.Lock()


def _store_anima(session: _AnimaSession) -> None:
    with _anima_sessions_lock:
        _anima_sessions[session.session_id] = session


def _get_anima(session_id: str) -> _AnimaSession:
    with _anima_sessions_lock:
        session = _anima_sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="anima caption session not found")
    return session


# Indirection for tests so they can swap the transformer without monkey-patching
# the dataset module directly.
def _build_anima_transformer(req: AnimaCaptionRequest) -> AnimaDatasetTransformer:
    return AnimaDatasetTransformer(
        default_quality=req.quality,
        default_score=req.score,
        default_year=req.year,
        default_safety=req.safety,
        dataset_tag=req.dataset_tag,
    )


@router.post("/anima/caption", status_code=202)
def anima_caption(req: AnimaCaptionRequest) -> dict[str, Any]:
    target = Path(req.path).expanduser()
    if not target.is_dir():
        raise HTTPException(status_code=400, detail=f"not a directory: {target}")

    session = _AnimaSession(
        session_id=uuid.uuid4().hex,
        path=str(target),
        dataset_tag=req.dataset_tag,
        quality=req.quality,
        score=req.score,
        year=req.year,
        safety=req.safety,
        overwrite=req.overwrite,
        recursive=req.recursive,
    )
    session.push("anima caption queued", percent=0)
    _store_anima(session)

    def run() -> None:
        try:
            transformer = _build_anima_transformer(req)

            # Pre-count for percent display.
            pattern = "**/*.txt" if req.recursive else "*.txt"
            captions = sorted(p for p in target.glob(pattern) if p.is_file())
            with session.lock:
                session.total = len(captions)

            if not captions:
                session.push("no caption files found", percent=100)
                with session.lock:
                    session.status = "succeeded"
                    session.percent = 100
                    session.finished_at = time.time()
                return

            total = len(captions)

            def on_progress(path: Path) -> None:
                with session.lock:
                    session.written += 1
                    pct = min(100.0, 2 + 98 * session.written / max(total, 1))
                session.push(f"rewrote {path.name}", percent=pct, file=str(path))

            written = transformer.transform_directory(
                target,
                recursive=req.recursive,
                overwrite=req.overwrite,
                progress=on_progress,
            )
            with session.lock:
                session.written = written
                session.status = "succeeded"
                session.percent = 100
                session.finished_at = time.time()
            session.push(f"done — wrote {written} caption(s)", percent=100)
        except Exception as exc:  # noqa: BLE001
            with session.lock:
                session.status = "failed"
                session.error = str(exc)
                session.finished_at = time.time()
            session.push(f"anima caption failed: {exc}")
        finally:
            _persist_tagging_snapshot(session)

    threading.Thread(
        target=run, name=f"anima-{session.session_id[:8]}", daemon=True
    ).start()
    return session.snapshot()


@router.get("/anima/caption/{session_id}")
def anima_caption_status(session_id: str) -> dict[str, Any]:
    with _anima_sessions_lock:
        session = _anima_sessions.get(session_id)
    if session is not None:
        return session.snapshot()
    persisted = _get_persisted_tagging(session_id)
    if persisted is not None:
        return persisted
    raise HTTPException(status_code=404, detail="anima caption session not found")
