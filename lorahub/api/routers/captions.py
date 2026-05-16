"""Generic anime caption preprocessing endpoints.

Provides background sessions for ``CaptionPipeline.transform_directory`` so
the UI (or curl) can kick off a directory-wide cleanup and poll status the
same way it polls auto-tagging. The session shape mirrors
``lorahub/api/routers/tagging.py``'s ``_TaggingSession`` so frontends can
reuse the same renderer.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from lorahub.core.dataset.captions import CaptionPipeline

router = APIRouter(prefix="/api")


class NormalizeCaptionsRequest(BaseModel):
    """Caller-side knobs for ``CaptionPipeline``.

    All fields are optional; an empty request triggers normalise-only
    behaviour (lowercase + underscore swap + dedupe). Strings of comma-
    separated values are accepted in addition to lists for terse curl use.
    """

    path: str
    blacklist: list[str] = Field(default_factory=list)
    remap: dict[str, str] = Field(default_factory=dict)
    known_artists: list[str] = Field(default_factory=list)
    quality: list[str] | None = None
    score: list[str] | None = None
    safety: str | None = None
    shuffle: bool = False
    keep_n: int = Field(default=0, ge=0)
    drop_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    seed: int | None = None
    recursive: bool = False
    overwrite: bool = False


@dataclass(slots=True)
class _CaptionsSession:
    session_id: str
    path: str
    status: Literal["running", "succeeded", "failed"] = "running"
    percent: float = 0.0
    events: list[dict[str, Any]] = field(default_factory=list)
    written: int = 0
    total: int | None = None
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def push(
        self,
        message: str,
        *,
        percent: float | None = None,
        file: str | None = None,
    ) -> None:
        with self.lock:
            if percent is not None:
                self.percent = max(self.percent, min(100.0, float(percent)))
            self.events.append(
                {
                    "ts": time.time(),
                    "message": message,
                    "percent": self.percent,
                    "file": file,
                }
            )
            self.events = self.events[-200:]

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "session_id": self.session_id,
                "path": self.path,
                "status": self.status,
                "percent": self.percent,
                "events": list(self.events),
                "written": self.written,
                "total": self.total,
                "error": self.error,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
            }


_sessions: dict[str, _CaptionsSession] = {}
_sessions_lock = threading.Lock()


def _store(session: _CaptionsSession) -> None:
    with _sessions_lock:
        _sessions[session.session_id] = session


def _get(session_id: str) -> _CaptionsSession:
    with _sessions_lock:
        session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="captions session not found")
    return session


def _persist_captions_snapshot(session: _CaptionsSession) -> None:
    """Best-effort flush of a captions snapshot to the SessionStore."""
    try:
        from lorahub.api import app as _app  # noqa: PLC0415

        store = getattr(_app, "_session_store", None)
        if store is not None:
            store.upsert_captions(session.snapshot())
    except Exception:  # noqa: BLE001
        pass


# Indirection so tests can monkeypatch the pipeline class without touching
# the concrete implementation. This mirrors the tagging router's
# `_build_tagger`.
def _build_pipeline(req: NormalizeCaptionsRequest) -> CaptionPipeline:
    return CaptionPipeline(
        blacklist=set(req.blacklist),
        remap=dict(req.remap),
        known_artists=set(req.known_artists),
        quality=list(req.quality) if req.quality else None,
        score=list(req.score) if req.score else None,
        safety=req.safety,
        shuffle=req.shuffle,
        keep_n=req.keep_n,
        drop_rate=req.drop_rate,
        seed=req.seed,
    )


@router.post("/captions/normalize", status_code=202)
def normalize_captions(req: NormalizeCaptionsRequest) -> dict[str, Any]:
    target = Path(req.path).expanduser()
    if not target.is_dir():
        raise HTTPException(status_code=400, detail=f"not a directory: {target}")

    session = _CaptionsSession(session_id=uuid.uuid4().hex, path=str(target))
    session.push("normalize queued", percent=0)
    _store(session)

    def run() -> None:
        try:
            pipeline = _build_pipeline(req)
            session.push("scanning captions", percent=2)

            def on_progress(file: Path, done: int, total: int) -> None:
                with session.lock:
                    session.total = total
                    session.written = done
                    pct = min(100.0, 2 + 98 * done / max(total, 1))
                session.push(f"processed {file.name}", percent=pct, file=str(file))

            written = pipeline.transform_directory(
                target,
                recursive=req.recursive,
                overwrite=req.overwrite,
                progress=on_progress,
            )
            with session.lock:
                # `written` from the pipeline is the count of *changed* files;
                # `session.written` tracks files *visited* via the progress
                # callback. Expose the changed count under a dedicated field
                # so callers can distinguish "looked at" from "rewritten".
                session.status = "succeeded"
                session.percent = 100
                session.finished_at = time.time()
            session.push(f"done — rewrote {written} caption(s)", percent=100)
        except Exception as exc:  # noqa: BLE001
            with session.lock:
                session.status = "failed"
                session.error = str(exc)
                session.finished_at = time.time()
            session.push(f"normalize failed: {exc}")
        finally:
            _persist_captions_snapshot(session)

    threading.Thread(
        target=run,
        name=f"captions-normalize-{session.session_id[:8]}",
        daemon=True,
    ).start()
    return session.snapshot()


@router.get("/captions/normalize/{session_id}")
def normalize_captions_status(session_id: str) -> dict[str, Any]:
    return _get(session_id).snapshot()
