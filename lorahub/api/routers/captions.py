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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from lorahub.api import app as app_module
from lorahub.api.dataset_files import resolve_dataset_directory
from lorahub.api.task_sessions import (
    TaskEvent,
    TaskSession,
    TaskSessionStore,
    default_task_store_path,
    persist_stop_request,
    prune_terminal_session_cache,
)
from lorahub.core.dataset.captions import CaptionPipeline

router = APIRouter(prefix="/api")
_KIND_CAPTIONS_NORMALIZE = "captions_normalize"
_CaptionsStatus = Literal[
    "running",
    "stop_requested",
    "succeeded",
    "failed",
    "canceled",
    "interrupted",
]


def _task_store() -> TaskSessionStore:
    store = app_module._task_session_store
    if store is None:
        store = TaskSessionStore(default_task_store_path())
        app_module._task_session_store = store
    return store


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
    task_kind: str | None = None
    status: _CaptionsStatus = "running"
    percent: float = 0.0
    events: list[dict[str, Any]] = field(default_factory=list)
    written: int = 0
    total: int | None = None
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)

    def push(
        self,
        message: str,
        *,
        percent: float | None = None,
        file: str | None = None,
    ) -> None:
        ts = time.time()
        with self.lock:
            if percent is not None:
                self.percent = max(self.percent, min(100.0, float(percent)))
            event = {
                "ts": ts,
                "message": message,
                "percent": self.percent,
                "file": file,
            }
            self.events.append(event)
            self.events = self.events[-200:]
        if self.task_kind:
            try:
                _task_store().append_event(
                    self.session_id,
                    TaskEvent(
                        level="info",
                        message=message,
                        percent=event["percent"],
                        payload=event,
                        ts=ts,
                    ),
                )
            except Exception:
                pass

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

    def request_stop(self) -> bool:
        with self.lock:
            if self.status != "running":
                return False
            if self.task_kind:
                persisted = persist_stop_request(
                    _task_store(),
                    self.session_id,
                    percent=self.percent,
                )
                if not persisted:
                    return False
            self.cancel_event.set()
            self.status = "stop_requested"
        self.push("cancel requested")
        return True

    def should_stop(self) -> bool:
        return self.cancel_event.is_set()


_sessions: dict[str, _CaptionsSession] = {}
_sessions_lock = threading.Lock()


def _store(session: _CaptionsSession) -> None:
    with _sessions_lock:
        _sessions[session.session_id] = session
        prune_terminal_session_cache(_sessions)


def _get(session_id: str) -> _CaptionsSession:
    with _sessions_lock:
        session = _sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="captions session not found")
    return session


def _get_persisted(session_id: str) -> dict[str, Any] | None:
    """Look up a captions session in the on-disk store.

    The in-memory `_sessions` dict only carries runs from this process.
    Older finished runs persisted in `sessions.sqlite` (via the try/finally
    in `run()`) are recovered through this lookup so a server restart
    doesn't make completed sessions vanish from the API.
    """
    try:
        from lorahub.api import app as _app  # noqa: PLC0415

        store = getattr(_app, "_session_store", None)
        if store is None:
            return None
        row = store.get("captions", session_id)
    except Exception:  # noqa: BLE001
        return None
    if row is None:
        return None
    snap = row.get("snapshot")
    return snap if isinstance(snap, dict) else None


def _task_snapshot(session_id: str) -> dict[str, Any] | None:
    try:
        task = _task_store().get(session_id)
    except Exception:
        return None
    if task is None or task.kind != _KIND_CAPTIONS_NORMALIZE:
        return None
    if isinstance(task.result, dict):
        result = dict(task.result)
        result.setdefault("events", [event.to_dict() for event in task.events])
        return result
    return _task_to_status_snapshot(task)


def _task_to_status_snapshot(task: TaskSession) -> dict[str, Any] | None:
    if task.status not in {
        "queued",
        "running",
        "stop_requested",
        "interrupted",
        "failed",
        "canceled",
    }:
        return None
    metadata = task.metadata
    return {
        "session_id": task.id,
        "path": str(metadata.get("path") or ""),
        "status": (
            task.status
            if task.status in {"stop_requested", "interrupted", "failed", "canceled"}
            else "running"
        ),
        "percent": task.percent,
        "events": [
            {
                "ts": event.ts,
                "message": event.message,
                "percent": event.percent if event.percent is not None else task.percent,
                "file": event.payload.get("file"),
            }
            for event in task.events
        ],
        "written": 0,
        "total": None,
        "error": task.error,
        "started_at": task.started_at,
        "finished_at": task.finished_at,
    }


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
    try:
        target = resolve_dataset_directory(req.path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    task = _task_store().create(
        kind=_KIND_CAPTIONS_NORMALIZE,
        title=f"captions normalize:{target.name}",
        metadata={
            "path": str(target),
            "recursive": req.recursive,
            "overwrite": req.overwrite,
        },
    )
    session = _CaptionsSession(
        session_id=task.id,
        path=str(target),
        task_kind=_KIND_CAPTIONS_NORMALIZE,
    )
    session.push("normalize queued", percent=0)
    _store(session)

    def run() -> None:
        try:
            _task_store().update(session.session_id, status="running", percent=0)
            if session.should_stop():
                raise InterruptedError("normalization canceled by user")
            pipeline = _build_pipeline(req)
            session.push("scanning captions", percent=2)

            def on_progress(file: Path, done: int, total: int) -> None:
                if session.should_stop():
                    raise InterruptedError("normalization canceled by user")
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
                should_stop=session.should_stop,
            )
            if session.should_stop():
                raise InterruptedError("normalization canceled by user")
            with session.lock:
                if session.cancel_event.is_set():
                    raise InterruptedError("normalization canceled by user")
                # `written` from the pipeline is the count of *changed* files;
                # `session.written` tracks files *visited* via the progress
                # callback. Expose the changed count under a dedicated field
                # so callers can distinguish "looked at" from "rewritten".
                session.status = "succeeded"
                session.percent = 100
                session.finished_at = time.time()
            session.push(f"done — rewrote {written} caption(s)", percent=100)
            _task_store().update(
                session.session_id,
                status="succeeded",
                percent=100,
                result=session.snapshot() | {"changed": written},
                finished=True,
            )
        except InterruptedError as exc:
            with session.lock:
                session.status = "canceled"
                session.error = str(exc)
                session.finished_at = time.time()
            session.push(str(exc))
            _task_store().update(
                session.session_id,
                status="canceled",
                error=str(exc),
                result=session.snapshot(),
                finished=True,
            )
        except Exception as exc:  # noqa: BLE001
            with session.lock:
                session.status = "failed"
                session.error = str(exc)
                session.finished_at = time.time()
            session.push(f"normalize failed: {exc}")
            _task_store().update(
                session.session_id,
                status="failed",
                error=str(exc),
                result=session.snapshot(),
                finished=True,
            )
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
    with _sessions_lock:
        session = _sessions.get(session_id)
    if session is not None:
        return session.snapshot()
    task = _task_snapshot(session_id)
    if task is not None:
        return task
    persisted = _get_persisted(session_id)
    if persisted is not None:
        return persisted
    raise HTTPException(status_code=404, detail="captions session not found")


@router.post("/captions/normalize/{session_id}/stop")
def stop_captions_normalize(session_id: str) -> dict[str, Any]:
    session = _get(session_id)
    if not session.request_stop():
        with session.lock:
            status = session.status
        raise HTTPException(status_code=409, detail=f"normalization is {status}")
    return {"session_id": session_id, "status": "stop_requested"}


@router.get("/captions/normalize")
def list_captions_sessions(limit: int = 50) -> dict[str, Any]:
    """Return recent captions sessions, merging in-memory + persisted rows.

    Live sessions take precedence over their persisted snapshots so a
    running session's progress doesn't get masked by an older finished
    state with the same id (which can happen briefly between
    `_persist_captions_snapshot` flushes).
    """
    out: dict[str, dict[str, Any]] = {}
    try:
        from lorahub.api import app as _app  # noqa: PLC0415

        store = getattr(_app, "_session_store", None)
        if store is not None:
            for row in store.list_recent("captions", limit=limit):
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
