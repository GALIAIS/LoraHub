"""Image Studio tagging session endpoints (start/status/stop)."""

from __future__ import annotations

import threading
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from lorahub.api import app as app_module
from lorahub.api.task_sessions import (
    TaskEvent,
    TaskSession,
    TaskSessionStore,
    default_task_store_path,
    persist_stop_request,
    prune_terminal_session_cache,
)

from ._shared import _writable_dataset_directory

if TYPE_CHECKING:
    from lorahub.core.tagging.base import BaseTagger

router = APIRouter(prefix="/api/image-studio", tags=["image-studio"])
_KIND_IMAGE_STUDIO_TAGGING = "image_studio_tagging"
_ISTaggingStatus = Literal[
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


def _persisted_task_result(session_id: str) -> dict[str, Any] | None:
    try:
        task = _task_store().get(session_id)
    except Exception:
        return None
    if task is None or task.kind != _KIND_IMAGE_STUDIO_TAGGING:
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
        "tagger": str(metadata.get("tagger") or "wd14"),
        "model_id": str(metadata.get("model_id") or ""),
        "device": str(metadata.get("device") or "auto"),
        "general": 0.35,
        "character": 0.85,
        "joytag_threshold": 0.5,
        "overwrite": bool(metadata.get("overwrite") or False),
        "recursive": bool(metadata.get("recursive") or False),
        "include_character": True,
        "underscores": False,
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
                "image": event.payload.get("image"),
            }
            for event in task.events
        ],
        "written": 0,
        "total": None,
        "active_provider": "",
        "error": task.error,
        "started_at": task.started_at,
        "finished_at": task.finished_at,
    }


class ISTaggingStartInput(BaseModel):
    path: str
    tagger: str = "wd14"  # "wd14" | "joytag"
    model_id: str | None = None
    general: float = Field(default=0.35, ge=0.0, le=1.0)
    character: float = Field(default=0.85, ge=0.0, le=1.0)
    joytag_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    device: str = "auto"
    overwrite: bool = False
    recursive: bool = False
    include_character: bool = True
    underscores: bool = False


@dataclass(slots=True)
class _ISTaggingSession:
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
    task_kind: str | None = None
    status: _ISTaggingStatus = "running"
    percent: float = 0.0
    events: list[dict[str, Any]] = field(default_factory=list)
    written: int = 0
    total: int | None = None
    active_provider: str = ""
    error: str | None = None
    started_at: float = field(default_factory=_time.time)
    finished_at: float | None = None
    _stop_flag: bool = field(default=False, repr=False)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def push(
        self, message: str, *, percent: float | None = None, image: str | None = None
    ) -> None:
        ts = _time.time()
        with self.lock:
            if percent is not None:
                self.percent = max(self.percent, min(100.0, float(percent)))
            event = {"ts": ts, "message": message, "percent": self.percent, "image": image}
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

    @property
    def should_stop(self) -> bool:
        with self.lock:
            return self._stop_flag

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
            self._stop_flag = True
            self.status = "stop_requested"
            percent = self.percent
        if self.task_kind:
            try:
                _task_store().append_event(
                    self.session_id,
                    TaskEvent(
                        level="warn",
                        message="stop requested",
                        percent=percent,
                        ts=_time.time(),
                    ),
                )
            except Exception:
                pass
        return True


# Module-level registry for active tagging sessions. Stays at module scope so
# all three endpoints share the same dict.
_is_tagging_sessions: dict[str, _ISTaggingSession] = {}
_is_tagging_lock = threading.Lock()


def _build_is_tagger(req: ISTaggingStartInput) -> BaseTagger:
    """Build a tagger instance from the request params."""
    from lorahub.core.tagging.joytag import JoyTagger  # noqa: PLC0415
    from lorahub.core.tagging.wd14 import DEFAULT_MODEL, WD14Tagger  # noqa: PLC0415

    if req.tagger == "joytag":
        return JoyTagger(
            predict_threshold=req.joytag_threshold,
            device=req.device,
        )
    return WD14Tagger(
        model_id=req.model_id or DEFAULT_MODEL,
        general_threshold=req.general,
        character_threshold=req.character,
        device=req.device,
    )


@router.post("/tagging/start", status_code=202)
def is_tagging_start(req: ISTaggingStartInput) -> dict[str, Any]:
    """Start a background tagging session from Image Studio."""
    from lorahub.core.tagging.wd14 import DEFAULT_MODEL  # noqa: PLC0415

    target = _writable_dataset_directory(req.path)

    task = _task_store().create(
        kind=_KIND_IMAGE_STUDIO_TAGGING,
        title=f"{req.tagger}:{target.name}",
        metadata={
            "path": str(target),
            "tagger": req.tagger,
            "model_id": req.model_id or DEFAULT_MODEL,
            "device": req.device,
            "overwrite": req.overwrite,
            "recursive": req.recursive,
        },
    )
    session = _ISTaggingSession(
        session_id=task.id,
        path=str(target),
        tagger=req.tagger,
        model_id=req.model_id or DEFAULT_MODEL,
        device=req.device,
        general=req.general,
        character=req.character,
        joytag_threshold=req.joytag_threshold,
        overwrite=req.overwrite,
        recursive=req.recursive,
        include_character=req.include_character,
        underscores=req.underscores,
        task_kind=_KIND_IMAGE_STUDIO_TAGGING,
    )
    session.push("tagging queued", percent=0)
    with _is_tagging_lock:
        _is_tagging_sessions[session.session_id] = session
        prune_terminal_session_cache(_is_tagging_sessions)


    def run() -> None:
        try:
            _task_store().update(session.session_id, status="running", percent=0)
            from lorahub.core.tagging.wd14 import (  # noqa: PLC0415
                CudaUnavailableError,
                _iter_images,
            )

            tagger = _build_is_tagger(req)
            session.push(f"loading {req.tagger}")
            tagger.load(should_stop=lambda: session.should_stop)
            if session.should_stop:
                raise InterruptedError("stopped by user")
            with session.lock:
                session.active_provider = tagger.active_provider
            session.push(f"running on {tagger.active_provider}", percent=2)

            all_images = list(_iter_images(target, recursive=req.recursive))
            with session.lock:
                session.total = len(all_images)
            if session.should_stop:
                raise InterruptedError("stopped by user")
            if not all_images:
                session.push("no images found", percent=100)
                with session.lock:
                    if session._stop_flag:
                        raise InterruptedError("stopped by user")
                    session.status = "succeeded"
                    session.percent = 100
                    session.finished_at = _time.time()
                _task_store().update(
                    session.session_id,
                    status="succeeded",
                    percent=100,
                    result=session.snapshot(),
                    finished=True,
                )
                return

            total = len(all_images)

            def on_progress(image: Path, _result: object) -> None:
                if session.should_stop:
                    raise InterruptedError("stopped by user")
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
                should_stop=lambda: session.should_stop,
            )
            if session.should_stop:
                raise InterruptedError("stopped by user")
            with session.lock:
                if session._stop_flag:
                    raise InterruptedError("stopped by user")
                session.status = "succeeded"
                session.percent = 100
                session.finished_at = _time.time()
            session.push(f"done - wrote {session.written} caption(s)", percent=100)
            _task_store().update(
                session.session_id,
                status="succeeded",
                percent=100,
                result=session.snapshot(),
                finished=True,
            )
        except CudaUnavailableError as exc:
            with session.lock:
                session.status = "failed"
                session.error = str(exc)
                session.finished_at = _time.time()
            session.push(f"cuda unavailable: {exc}")
            _task_store().update(
                session.session_id,
                status="failed",
                error=str(exc),
                result=session.snapshot(),
                finished=True,
            )
        except InterruptedError:
            with session.lock:
                session.status = "canceled"
                session.error = "stopped by user"
                session.finished_at = _time.time()
            session.push("stopped by user")
            _task_store().update(
                session.session_id,
                status="canceled",
                error="stopped by user",
                result=session.snapshot(),
                finished=True,
            )
        except Exception as exc:  # noqa: BLE001
            with session.lock:
                session.status = "failed"
                session.error = str(exc)
                session.finished_at = _time.time()
            session.push(f"tagging failed: {exc}")
            _task_store().update(
                session.session_id,
                status="failed",
                error=str(exc),
                result=session.snapshot(),
                finished=True,
            )

    threading.Thread(
        target=run,
        name=f"is-tag-{req.tagger}-{session.session_id[:8]}",
        daemon=True,
    ).start()
    return {"session_id": session.session_id}


@router.get("/tagging/{session_id}")
def is_tagging_status(session_id: str) -> dict[str, Any]:
    """Return the current snapshot of a tagging session."""
    with _is_tagging_lock:
        session = _is_tagging_sessions.get(session_id)
    if session is None:
        persisted = _persisted_task_result(session_id)
        if persisted is not None:
            return persisted
        raise HTTPException(404, "tagging session not found")
    return session.snapshot()


@router.post("/tagging/stop/{session_id}")
def is_tagging_stop(session_id: str) -> dict[str, Any]:
    """Request a running tagging session to stop."""
    with _is_tagging_lock:
        session = _is_tagging_sessions.get(session_id)
    if session is None:
        raise HTTPException(404, "tagging session not found")
    if not session.request_stop():
        with session.lock:
            status = session.status
        raise HTTPException(409, f"tagging is {status}")
    return {"session_id": session_id, "status": "stop_requested"}
