"""Image Studio AI endpoints — batch caption, quality scoring, and smart caption.

Smart caption combines a local WD14 tagger with a vision LLM to produce
Anima-format captions used for LoRA training.
"""

from __future__ import annotations

import re
import threading
import time as _time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from lorahub.api import app as app_module
from lorahub.api.dataset_files import _resolve_under_roots
from lorahub.api.image_studio_store import ImageAnnotation, ImageStudioStore
from lorahub.api.task_sessions import TaskEvent, TaskSessionStore

from ._shared import _file_sha256, _scan_images, _store
from .ai_tasks import get_task_store, persisted_task_result

if TYPE_CHECKING:
    from lorahub.core.tagging.wd14 import WD14Tagger

router = APIRouter(prefix="/api/image-studio", tags=["image-studio"])
_KIND_CAPTION = "image_studio_caption"
_KIND_SMART_CAPTION = "image_studio_smart_caption"
_KIND_QUALITY = "image_studio_quality"
_KIND_TRIGGER_WORDS = "image_studio_trigger_words"


def _task_store() -> TaskSessionStore:
    return get_task_store()


def _persisted_task_result(session_id: str, kind: str) -> dict[str, Any] | None:
    return persisted_task_result(session_id, kind)


def _ulid_safe() -> str:
    """Stand-in for ulid that's safe to call without the package.

    `ulid-py` is on the dependency list but the smart-caption sessions
    don't really need lexicographic sortability — uuid4 is plenty.
    """
    return uuid.uuid4().hex


@dataclass
class _SmartCaptionSession:
    """Live state for a background smart-caption batch.

    Mirrors the shape of `_ISTaggingSession` so the frontend can reuse
    the same polling pattern. ``stop_requested`` is honoured between
    images, so cancel arrives at most one image-render late.
    """

    session_id: str
    path: str
    total: int
    task_kind: str | None = None
    status: str = "running"  # running / succeeded / failed / canceled
    processed: int = 0
    results: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)
    error: str | None = None
    last_image: str = ""
    started_at: float = field(default_factory=_time.time)
    finished_at: float | None = None
    _stop_flag: bool = field(default=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add_result(self, item: dict[str, Any], image_name: str) -> None:
        with self._lock:
            self.results.append(item)
            self.processed += 1
            self.last_image = image_name
            processed = self.processed
        self._append_task_event(
            f"captioned {image_name}",
            percent=self.percent,
            payload={"image": image_name, "processed": processed, "item": item},
        )

    def add_error(self, path: str, msg: str, image_name: str) -> None:
        with self._lock:
            self.errors.append({"path": path, "error": msg})
            self.processed += 1
            self.last_image = image_name
            processed = self.processed
        self._append_task_event(
            f"failed {image_name}: {msg}",
            level="error",
            percent=self.percent,
            payload={
                "path": path,
                "image": image_name,
                "error": msg,
                "processed": processed,
            },
        )

    def set_error(self, msg: str) -> None:
        with self._lock:
            self.error = msg
        self._append_task_event(msg, level="error", percent=self.percent)

    def finish(self, status: str) -> None:
        with self._lock:
            self.status = status
            self.finished_at = _time.time()
        self._append_task_event(f"finished: {status}", percent=self.percent)
        self._finalize_task(status)

    def request_stop(self) -> None:
        with self._lock:
            self._stop_flag = True
        self._append_task_event("cancel requested", level="warn", percent=self.percent)

    def should_stop(self) -> bool:
        with self._lock:
            return self._stop_flag

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "session_id": self.session_id,
                "path": self.path,
                "status": self.status,
                "processed": self.processed,
                "total": self.total,
                "percent": (
                    100.0 * self.processed / self.total
                    if self.total > 0
                    else 0.0
                ),
                "last_image": self.last_image,
                "results": list(self.results),
                "errors": list(self.errors),
                "error": self.error,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
            }

    @property
    def percent(self) -> float:
        with self._lock:
            return 100.0 * self.processed / self.total if self.total > 0 else 0.0

    def _append_task_event(
        self,
        message: str,
        *,
        level: str = "info",
        percent: float | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        if not self.task_kind:
            return
        try:
            _task_store().append_event(
                self.session_id,
                TaskEvent(
                    level=level,
                    message=message,
                    percent=percent,
                    payload=payload or {},
                    ts=_time.time(),
                ),
            )
        except Exception:
            pass

    def _finalize_task(self, status: str) -> None:
        if not self.task_kind:
            return
        task_status = "succeeded"
        if status == "failed":
            task_status = "failed"
        elif status == "canceled":
            task_status = "canceled"
        try:
            _task_store().update(
                self.session_id,
                status=task_status,  # type: ignore[arg-type]
                percent=100 if status == "succeeded" else self.percent,
                result=self.snapshot(),
                error=self.error,
                finished=True,
            )
        except Exception:
            pass


# Module-level session registry. Same shape as the tagging tab — the only
# state we keep across requests is "what's running right now"; finished
# sessions stick around so the frontend can pull final results once.
# Memory bound: cleared whenever the process restarts, plus best-effort
# eviction of sessions older than 1h after they finish (see snapshot).
_smart_caption_sessions: dict[str, _SmartCaptionSession] = {}
_smart_caption_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# AI batch endpoints
# --------------------------------------------------------------------------- #


class AIBatchCaptionInput(BaseModel):
    path: str
    recursive: bool = False
    task: str = "tagging.assist"
    mergeStrategy: str = "replace"
    # Skip images that already have a non-empty .txt sidecar. Empty /
    # zero-byte sidecars are NOT skipped (they're usually crash-leftover
    # half-writes that should be reprocessed).
    skipAnnotated: bool = True


@dataclass
class _CaptionSession:
    session_id: str
    path: str
    total: int
    skipped: int
    status: str = "running"
    processed: int = 0
    results: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)
    error: str | None = None
    last_image: str = ""
    started_at: float = field(default_factory=_time.time)
    finished_at: float | None = None
    _stop_flag: bool = field(default=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def percent(self) -> float:
        with self._lock:
            return 100.0 * self.processed / self.total if self.total > 0 else 100.0

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "session_id": self.session_id,
                "path": self.path,
                "status": self.status,
                "processed": self.processed,
                "total": self.total,
                "skipped": self.skipped,
                "percent": (
                    100.0 * self.processed / self.total
                    if self.total > 0
                    else 100.0
                ),
                "last_image": self.last_image,
                "results": list(self.results),
                "errors": list(self.errors),
                "error": self.error,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
            }

    def add_result(self, item: dict[str, Any], image_name: str) -> None:
        with self._lock:
            self.results.append(item)
            self.processed += 1
            self.last_image = image_name
            processed = self.processed
            percent = 100.0 * processed / self.total if self.total > 0 else 100.0
        self._append_task_event(
            f"captioned {image_name}",
            percent=percent,
            payload={"image": image_name, "processed": processed, "item": item},
        )

    def add_error(self, path: str, msg: str, image_name: str) -> None:
        with self._lock:
            self.errors.append({"path": path, "error": msg})
            self.processed += 1
            self.last_image = image_name
            processed = self.processed
            percent = 100.0 * processed / self.total if self.total > 0 else 100.0
        self._append_task_event(
            f"failed {image_name}: {msg}",
            level="error",
            percent=percent,
            payload={
                "path": path,
                "image": image_name,
                "error": msg,
                "processed": processed,
            },
        )

    def finish(self, status: str) -> None:
        with self._lock:
            self.status = status
            self.finished_at = _time.time()
        self._append_task_event(f"finished: {status}", percent=self.percent)
        try:
            _task_store().update(
                self.session_id,
                status="succeeded" if status == "succeeded" else "canceled",
                percent=100 if status == "succeeded" else self.percent,
                result=self.snapshot(),
                error=self.error,
                finished=True,
            )
        except Exception:
            pass

    def request_stop(self) -> None:
        with self._lock:
            self._stop_flag = True
        self._append_task_event("cancel requested", level="warn", percent=self.percent)

    def should_stop(self) -> bool:
        with self._lock:
            return self._stop_flag

    def fail(self, msg: str) -> None:
        with self._lock:
            self.status = "failed"
            self.error = msg
            self.finished_at = _time.time()
        self._append_task_event(msg, level="error", percent=self.percent)
        try:
            _task_store().update(
                self.session_id,
                status="failed",
                percent=self.percent,
                result=self.snapshot(),
                error=msg,
                finished=True,
            )
        except Exception:
            pass

    def _append_task_event(
        self,
        message: str,
        *,
        level: str = "info",
        percent: float | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        try:
            _task_store().append_event(
                self.session_id,
                TaskEvent(
                    level=level,
                    message=message,
                    percent=percent,
                    payload=payload or {},
                    ts=_time.time(),
                ),
            )
        except Exception:
            pass


_caption_sessions: dict[str, _CaptionSession] = {}
_caption_lock = threading.Lock()


def _caption_images_for_request(
    body: AIBatchCaptionInput,
    directory: Path,
) -> tuple[list[Path], int]:
    images = _scan_images(directory, body.recursive)
    skipped = 0
    if body.skipAnnotated:
        before = len(images)
        images = [
            p for p in images
            if not (
                p.with_suffix(".txt").is_file()
                and p.with_suffix(".txt").stat().st_size > 0
            )
        ]
        skipped = before - len(images)
    return images, skipped


def _caption_images(
    body: AIBatchCaptionInput,
    directory: Path,
    images: list[Path],
    *,
    on_result: Callable[[dict[str, Any], str], None] | None = None,
    on_error: Callable[[str, str, str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    from datetime import UTC, datetime  # noqa: PLC0415

    from lorahub.api import app as app_mod  # noqa: PLC0415
    from lorahub.core.ai import client as ai_client  # noqa: PLC0415

    ai_store = app_mod._ai_store
    if ai_store is None:
        raise HTTPException(503, "AI store not initialised")

    route = ai_store.get_route(body.task)
    if route is None or not (route.provider_id and route.model_id):
        route = ai_store.get_route("global.default")
    if route is None or not (route.provider_id and route.model_id):
        raise HTTPException(409, f"no AI route for task {body.task!r}")

    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    store = _store()
    for img_path in images:
        if should_stop is not None and should_stop():
            raise InterruptedError("stopped by user")
        try:
            import base64  # noqa: PLC0415
            import mimetypes  # noqa: PLC0415

            mime = mimetypes.guess_type(str(img_path))[0] or "image/png"
            data = img_path.read_bytes()
            b64 = base64.b64encode(data).decode("ascii")
            data_url = f"data:{mime};base64,{b64}"

            messages: list[dict[str, Any]] = []
            if route.system_prompt:
                messages.append({"role": "system", "content": route.system_prompt})
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            })

            result = ai_client.invoke(
                ai_store,
                provider_id=route.provider_id,
                model_id=route.model_id,
                messages=messages,
                route=route,
            )

            caption_path = img_path.with_suffix(".txt")
            existing = caption_path.read_text(encoding="utf-8") if caption_path.is_file() else ""

            if body.mergeStrategy == "append":
                new_caption = (existing.strip() + ", " + result.content).strip(", ")
            elif body.mergeStrategy == "rewrite":
                new_caption = result.content
            else:
                new_caption = result.content

            caption_path.write_text(new_caption, encoding="utf-8")

            ann = store.get_annotation(str(img_path))
            if ann is None:
                ann = ImageAnnotation(
                    image_path=str(img_path),
                    sha256=_file_sha256(img_path),
                )
            ann.ai_caption = result.content
            ann.ai_caption_provider = f"{result.provider_name}/{result.model_id}"
            ann.ai_caption_at = datetime.now(UTC).isoformat()
            store.upsert_annotation(ann)

            item = {"path": str(img_path), "caption": new_caption}
            results.append(item)
            if on_result is not None:
                on_result(item, img_path.name)
        except Exception as exc:  # noqa: BLE001
            errors.append({"path": str(img_path), "error": str(exc)})
            if on_error is not None:
                on_error(str(img_path), str(exc), img_path.name)

    return results, errors


@router.post("/ai/caption")
def ai_batch_caption(body: AIBatchCaptionInput) -> dict[str, Any]:
    """Caption all images in a directory synchronously, preserving legacy API."""
    directory = _resolve_under_roots(body.path)
    if not directory.is_dir():
        raise HTTPException(400, "not a directory")
    images, skipped = _caption_images_for_request(body, directory)
    results, errors = _caption_images(body, directory, images)

    return {
        "processed": len(results),
        "skipped": skipped,
        "results": results,
        "errors": errors,
    }


@router.post("/ai/caption/start", status_code=202)
def ai_batch_caption_start(body: AIBatchCaptionInput) -> dict[str, Any]:
    """Start a persistent background captioning session."""
    directory = _resolve_under_roots(body.path)
    if not directory.is_dir():
        raise HTTPException(400, "not a directory")
    # Validate route before returning 202 so configuration errors are immediate.
    from lorahub.api import app as app_mod  # noqa: PLC0415

    ai_store = app_mod._ai_store
    if ai_store is None:
        raise HTTPException(503, "AI store not initialised")
    route = ai_store.get_route(body.task)
    if route is None or not (route.provider_id and route.model_id):
        route = ai_store.get_route("global.default")
    if route is None or not (route.provider_id and route.model_id):
        raise HTTPException(409, f"no AI route for task {body.task!r}")

    images, skipped = _caption_images_for_request(body, directory)
    task = _task_store().create(
        kind=_KIND_CAPTION,
        title=f"caption:{directory.name}",
        metadata={
            "path": str(directory),
            "recursive": body.recursive,
            "task": body.task,
            "mergeStrategy": body.mergeStrategy,
            "skipAnnotated": body.skipAnnotated,
            "skipped": skipped,
        },
    )
    session = _CaptionSession(
        session_id=task.id,
        path=str(directory),
        total=len(images),
        skipped=skipped,
    )
    session._append_task_event("captioning queued", percent=0)
    with _caption_lock:
        _caption_sessions[session.session_id] = session

    def run() -> None:
        try:
            _task_store().update(session.session_id, status="running", percent=0)
            _caption_images(
                body,
                directory,
                images,
                on_result=session.add_result,
                on_error=session.add_error,
                should_stop=session.should_stop,
            )
            session.finish("canceled" if session.should_stop() else "succeeded")
        except InterruptedError:
            session.finish("canceled")
        except Exception as exc:  # noqa: BLE001
            session.fail(str(exc))

    threading.Thread(
        target=run,
        name=f"caption-{session.session_id[:8]}",
        daemon=True,
    ).start()
    return {
        "session_id": session.session_id,
        "total": len(images),
        "skipped": skipped,
        "status_url": f"/api/image-studio/ai/caption/status/{session.session_id}",
    }


@router.get("/ai/caption/status/{session_id}")
def ai_batch_caption_status(session_id: str) -> dict[str, Any]:
    with _caption_lock:
        session = _caption_sessions.get(session_id)
    if session is not None:
        return session.snapshot()
    persisted = _persisted_task_result(session_id, _KIND_CAPTION)
    if persisted is not None:
        return persisted
    raise HTTPException(404, "caption session not found")


@router.post("/ai/caption/cancel/{session_id}")
def ai_batch_caption_cancel(session_id: str) -> dict[str, Any]:
    with _caption_lock:
        session = _caption_sessions.get(session_id)
    if session is None:
        raise HTTPException(404, "caption session not found")
    session.request_stop()
    return {"session_id": session_id, "status": "stop_requested"}


class AIBatchQualityInput(BaseModel):
    path: str
    recursive: bool = False
    task: str = "quality.score"
    # Skip images that already have an AI quality score in the store.
    # Quality scoring writes to the store (not to .txt), so the
    # "completed" check is different from caption batches.
    skipScored: bool = True


@dataclass
class _QualitySession:
    session_id: str
    path: str
    total: int
    skipped: int
    status: str = "running"
    processed: int = 0
    results: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)
    error: str | None = None
    last_image: str = ""
    started_at: float = field(default_factory=_time.time)
    finished_at: float | None = None
    _stop_flag: bool = field(default=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def percent(self) -> float:
        with self._lock:
            return 100.0 * self.processed / self.total if self.total > 0 else 100.0

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "session_id": self.session_id,
                "path": self.path,
                "status": self.status,
                "processed": self.processed,
                "total": self.total,
                "skipped": self.skipped,
                "percent": (
                    100.0 * self.processed / self.total
                    if self.total > 0
                    else 100.0
                ),
                "last_image": self.last_image,
                "results": list(self.results),
                "errors": list(self.errors),
                "error": self.error,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
            }

    def add_result(self, item: dict[str, Any], image_name: str) -> None:
        with self._lock:
            self.results.append(item)
            self.processed += 1
            self.last_image = image_name
            processed = self.processed
            percent = 100.0 * processed / self.total if self.total > 0 else 100.0
        self._append_task_event(
            f"scored {image_name}",
            percent=percent,
            payload={"image": image_name, "processed": processed, "item": item},
        )

    def add_error(self, path: str, msg: str, image_name: str) -> None:
        with self._lock:
            self.errors.append({"path": path, "error": msg})
            self.processed += 1
            self.last_image = image_name
            processed = self.processed
            percent = 100.0 * processed / self.total if self.total > 0 else 100.0
        self._append_task_event(
            f"failed {image_name}: {msg}",
            level="error",
            percent=percent,
            payload={
                "path": path,
                "image": image_name,
                "error": msg,
                "processed": processed,
            },
        )

    def finish(self, status: str) -> None:
        with self._lock:
            self.status = status
            self.finished_at = _time.time()
        self._append_task_event(f"finished: {status}", percent=self.percent)
        try:
            _task_store().update(
                self.session_id,
                status="succeeded" if status == "succeeded" else "canceled",
                percent=100 if status == "succeeded" else self.percent,
                result=self.snapshot(),
                error=self.error,
                finished=True,
            )
        except Exception:
            pass

    def request_stop(self) -> None:
        with self._lock:
            self._stop_flag = True
        self._append_task_event("cancel requested", level="warn", percent=self.percent)

    def should_stop(self) -> bool:
        with self._lock:
            return self._stop_flag

    def fail(self, msg: str) -> None:
        with self._lock:
            self.status = "failed"
            self.error = msg
            self.finished_at = _time.time()
        self._append_task_event(msg, level="error", percent=self.percent)
        try:
            _task_store().update(
                self.session_id,
                status="failed",
                percent=self.percent,
                result=self.snapshot(),
                error=msg,
                finished=True,
            )
        except Exception:
            pass

    def _append_task_event(
        self,
        message: str,
        *,
        level: str = "info",
        percent: float | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        try:
            _task_store().append_event(
                self.session_id,
                TaskEvent(
                    level=level,
                    message=message,
                    percent=percent,
                    payload=payload or {},
                    ts=_time.time(),
                ),
            )
        except Exception:
            pass


_quality_sessions: dict[str, _QualitySession] = {}
_quality_lock = threading.Lock()


def _score_quality_images(
    body: AIBatchQualityInput,
    directory: Path,
    images: list[Path],
    *,
    on_result: Callable[[dict[str, Any], str], None] | None = None,
    on_error: Callable[[str, str, str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    from datetime import UTC, datetime  # noqa: PLC0415

    from lorahub.api import app as app_mod  # noqa: PLC0415
    from lorahub.core.ai import client as ai_client  # noqa: PLC0415

    ai_store = app_mod._ai_store
    if ai_store is None:
        raise HTTPException(503, "AI store not initialised")

    route = ai_store.get_route(body.task)
    if route is None or not (route.provider_id and route.model_id):
        route = ai_store.get_route("global.default")
    if route is None or not (route.provider_id and route.model_id):
        raise HTTPException(409, f"no AI route for task {body.task!r}")

    store = _store()
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for img_path in images:
        if should_stop is not None and should_stop():
            raise InterruptedError("stopped by user")
        try:
            import base64  # noqa: PLC0415
            import json as json_mod  # noqa: PLC0415
            import mimetypes  # noqa: PLC0415

            mime = mimetypes.guess_type(str(img_path))[0] or "image/png"
            data = img_path.read_bytes()
            b64 = base64.b64encode(data).decode("ascii")
            data_url = f"data:{mime};base64,{b64}"

            system_prompt = route.system_prompt or (
                'Rate this training image on a 0-100 scale. '
                'Return JSON: {"score": 0-100, "label": "good"|"medium"|"bad", "reason": "..."}'
            )

            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]},
            ]

            result = ai_client.invoke(
                ai_store,
                provider_id=route.provider_id,
                model_id=route.model_id,
                messages=messages,
                route=route,
            )

            score: float | None = None
            label: str | None = None
            reason: str | None = None
            try:
                parsed = json_mod.loads(result.content)
                score = float(parsed.get("score", 0)) / 100.0
                label = parsed.get("label")
                reason = parsed.get("reason")
            except (json_mod.JSONDecodeError, ValueError, TypeError):
                reason = result.content

            ann = store.get_annotation(str(img_path))
            if ann is None:
                ann = ImageAnnotation(
                    image_path=str(img_path),
                    sha256=_file_sha256(img_path),
                )
            ann.ai_quality_score = score
            ann.ai_quality_label = label
            ann.ai_quality_reason = reason
            ann.ai_quality_at = datetime.now(UTC).isoformat()
            store.upsert_annotation(ann)

            item = {
                "path": str(img_path),
                "score": score,
                "label": label,
                "reason": reason,
            }
            results.append(item)
            if on_result is not None:
                on_result(item, img_path.name)
        except Exception as exc:  # noqa: BLE001
            error = {"path": str(img_path), "error": str(exc)}
            errors.append(error)
            if on_error is not None:
                on_error(str(img_path), str(exc), img_path.name)
    return results, errors


def _quality_images_for_request(
    body: AIBatchQualityInput,
    directory: Path,
) -> tuple[list[Path], int]:
    images = _scan_images(directory, body.recursive)
    skipped = 0
    if body.skipScored:
        store = _store()
        before = len(images)
        images = [
            p for p in images
            if not (
                (ann := store.get_annotation(str(p))) is not None
                and ann.ai_quality_label is not None
            )
        ]
        skipped = before - len(images)
    return images, skipped


@router.post("/ai/quality")
def ai_batch_quality(body: AIBatchQualityInput) -> dict[str, Any]:
    """Score image quality via VLM for all images in a directory."""
    directory = _resolve_under_roots(body.path)
    if not directory.is_dir():
        raise HTTPException(400, "not a directory")

    images, skipped = _quality_images_for_request(body, directory)
    results, errors = _score_quality_images(body, directory, images)
    return {
        "processed": len(results),
        "skipped": skipped,
        "results": results,
        "errors": errors,
    }


@router.post("/ai/quality/start", status_code=202)
def ai_batch_quality_start(body: AIBatchQualityInput) -> dict[str, Any]:
    """Start a persistent background quality scoring session."""
    directory = _resolve_under_roots(body.path)
    if not directory.is_dir():
        raise HTTPException(400, "not a directory")
    # Validate route before returning 202 so configuration errors are immediate.
    from lorahub.api import app as app_mod  # noqa: PLC0415

    ai_store = app_mod._ai_store
    if ai_store is None:
        raise HTTPException(503, "AI store not initialised")
    route = ai_store.get_route(body.task)
    if route is None or not (route.provider_id and route.model_id):
        route = ai_store.get_route("global.default")
    if route is None or not (route.provider_id and route.model_id):
        raise HTTPException(409, f"no AI route for task {body.task!r}")

    images, skipped = _quality_images_for_request(body, directory)
    task = _task_store().create(
        kind=_KIND_QUALITY,
        title=f"quality:{directory.name}",
        metadata={
            "path": str(directory),
            "recursive": body.recursive,
            "task": body.task,
            "skipScored": body.skipScored,
            "skipped": skipped,
        },
    )
    session = _QualitySession(
        session_id=task.id,
        path=str(directory),
        total=len(images),
        skipped=skipped,
    )
    session._append_task_event("quality scoring queued", percent=0)
    with _quality_lock:
        _quality_sessions[session.session_id] = session

    def run() -> None:
        try:
            _task_store().update(session.session_id, status="running", percent=0)
            _score_quality_images(
                body,
                directory,
                images,
                on_result=session.add_result,
                on_error=session.add_error,
                should_stop=session.should_stop,
            )
            session.finish("canceled" if session.should_stop() else "succeeded")
        except InterruptedError:
            session.finish("canceled")
        except Exception as exc:  # noqa: BLE001
            session.fail(str(exc))

    threading.Thread(
        target=run,
        name=f"quality-score-{session.session_id[:8]}",
        daemon=True,
    ).start()
    return {
        "session_id": session.session_id,
        "total": len(images),
        "skipped": skipped,
        "status_url": f"/api/image-studio/ai/quality/status/{session.session_id}",
    }


@router.get("/ai/quality/status/{session_id}")
def ai_batch_quality_status(session_id: str) -> dict[str, Any]:
    with _quality_lock:
        session = _quality_sessions.get(session_id)
    if session is not None:
        return session.snapshot()
    persisted = _persisted_task_result(session_id, _KIND_QUALITY)
    if persisted is not None:
        return persisted
    raise HTTPException(404, "quality session not found")


@router.post("/ai/quality/cancel/{session_id}")
def ai_batch_quality_cancel(session_id: str) -> dict[str, Any]:
    with _quality_lock:
        session = _quality_sessions.get(session_id)
    if session is None:
        raise HTTPException(404, "quality session not found")
    session.request_stop()
    return {"session_id": session_id, "status": "stop_requested"}


# --------------------------------------------------------------------------- #
# Smart caption (WD14 + Vision LLM)
# --------------------------------------------------------------------------- #

# Tags that are pure quality/medium noise; don't auto-carry them into
# generated training captions.
_QUALITY_NOISE_TAGS = {
    "highres", "absurdres", "best quality", "masterpiece", "high quality",
    "low quality", "worst quality", "normal quality", "lowres",
    "official art", "key visual", "promotional art", "screencap",
    "artist name", "signature", "watermark", "logo", "english text",
    "dated", "twitter username", "patreon username", "artist logo",
    "score_7", "safe",
}
_FORBIDDEN_CAPTION_TAGS = {
    "masterpiece",
    "best quality",
    "score_7",
    "score 7",
    "safe",
}

# Style / medium / rendering descriptors. Stripped when the user asks for
# "auto-strip style tags" — the whole point of training a style LoRA is
# that the trigger word (or @artist tag) carries the style; baking
# explicit medium descriptors into every caption teaches the model that
# the style is "anime + illustration + cel shading", which the bare
# trigger then has to compete with.  Covers the WD14 vocabulary's most
# common medium / aesthetic / palette / line-style buckets.
_STYLE_NOISE_TAGS = {
    # Medium / format
    "anime", "anime coloring", "anime screencap", "anime style",
    "manga", "comic", "western comics (style)", "amerimanga",
    "illustration", "digital illustration", "traditional media",
    "digital media", "concept art", "fan art", "fanart",
    # Realism axis
    "realistic", "photorealistic", "semi-realistic", "hyperrealistic",
    "photo (medium)", "photograph", "rendered",
    # Stylisation
    "chibi", "deformed", "super deformed", "cartoon", "cartoonish",
    "kawaii", "moe (style)", "ligne claire",
    # Rendering / shading
    "cel shading", "cel-shaded", "soft shading", "flat color",
    "flat colors", "flat shading", "no shading",
    "lineart", "line art", "sketch", "rough sketch",
    "outline", "thick outlines", "thin outlines", "no outlines",
    "painterly", "painting (medium)", "oil painting (medium)",
    "watercolor (medium)", "watercolor", "acrylic paint (medium)",
    "ink (medium)", "marker (medium)", "pastel (medium)",
    "colored pencil (medium)", "graphite (medium)",
    "screentone", "halftone", "halftone background",
    # Palette / mood
    "monochrome", "greyscale", "grayscale", "limited palette",
    "pastel colors", "vivid colors", "muted colors", "saturated",
    "high contrast",
    # Era / movement
    "1980s (style)", "1990s (style)", "2000s (style)", "retro artstyle",
    "ukiyo-e", "art nouveau", "minimalism", "surreal", "abstract",
    # Dimensionality
    "3d", "2d", "3dcg", "2.5d",
}


_STYLE_DROP_INSTRUCTION = (
    "STRICT: do NOT use the words anime, manga, illustration, cartoon, chibi, "
    "realistic, photorealistic, lineart, sketch, painterly, watercolor, "
    "monochrome, cel-shaded, flat color, screentone, 2d, 3d (or any close "
    "synonym) anywhere in your output. Those describe the medium and would "
    "compete with the trigger word's job of owning the style. Describe the "
    "specific visible craft (brush economy, edge softness, palette warmth, "
    "shading contrast) instead of naming a genre.\n\n"
)


# Tags describing the character's FIXED PHYSICAL IDENTITY — face / hair /
# eyes / skin / body. For character LoRAs the trigger word is supposed
# to own this signal; baking these into every caption teaches the model
# that "the character" is the conjunction of trigger + "blue eyes + long
# hair + ..." which dilutes the trigger's hold.
#
# We use three filters together:
#   - exact stoplist for solo identity tags (ahoge, sidelocks, freckles,
#     ...).
#   - a "must-keep" allowlist for tags whose name happens to contain
#     identity words but whose meaning is expression / action / state
#     (closed eyes, eyes closed, half-closed eyes, glowing eyes,
#     tearful eyes, wet hair, disheveled hair, ...).
#   - suffix tests for the long-tail "<modifier> hair" / "<modifier>
#     eyes" / "<modifier> skin" / "<modifier> breasts" patterns that
#     WD14 produces in thousands of variants.

_APPEARANCE_KEEP_EXACT: frozenset[str] = frozenset({
    # Eye state / expression — same shape as identity tags but describe
    # what the character is doing, not how they look fixedly.
    "closed eyes", "eyes closed", "one eye closed", "half-closed eyes",
    "half closed eyes", "tearful eyes", "wide-eyed", "wide eyes",
    "rolling eyes", "narrowed eyes", "glowing eyes", "shining eyes",
    "sparkling eyes", "crying with eyes open",
    # Hair state — environment / action, not fixed style.
    "wet hair", "disheveled hair", "windswept hair", "floating hair",
    "hair between eyes", "hair over one eye", "hair flip", "hair pull",
})

_APPEARANCE_DROP_EXACT: frozenset[str] = frozenset({
    # Hairstyle / hair feature
    "ahoge", "sidelocks", "sidelock", "bangs", "blunt bangs",
    "swept bangs", "asymmetrical bangs", "parted bangs",
    "short hair", "very short hair", "medium hair", "long hair",
    "very long hair", "absurdly long hair",
    "ponytail", "low ponytail", "high ponytail", "side ponytail",
    "twintails", "twin tails", "low twintails", "drill hair",
    "drill twintails", "braid", "braids", "single braid", "twin braids",
    "french braid", "crown braid", "hime cut", "bob cut", "bowl cut",
    "pixie cut", "undercut", "mohawk", "afro", "updo", "hair bun",
    "double bun", "single hair bun", "messy hair", "wavy hair",
    "curly hair", "straight hair", "spiky hair", "shoulder-length hair",
    "hair ornament", "hair flower", "hair ribbon", "hairband",
    "hair clip", "hairclip", "x hair ornament", "star hair ornament",
    # Eye features (color/shape/decor)
    "heterochromia", "pupils", "+ +", "x x", "@ @", "no pupils",
    "pointy eyes", "tareme", "tsurime",
    # Body / proportions / size
    "muscular", "muscular female", "muscular male", "abs", "biceps",
    "wide hips", "thigh gap", "slender", "petite", "tall female",
    "short female", "thick thighs", "small breasts", "medium breasts",
    "large breasts", "huge breasts", "gigantic breasts", "flat chest",
    "cleavage", "underboob", "sideboob", "breasts apart",
    # Skin / face
    "freckles", "beauty mark", "mole under eye", "mole on cheek",
    "mole on neck", "scar on face", "scar on cheek", "scar across eye",
    "fang", "fangs", "single fang", "sharp teeth",
    # Species markers / physical features
    "pointy ears", "cat ears", "fox ears", "dog ears", "rabbit ears",
    "animal ears", "horns", "single horn", "halo", "wings",
    "fairy wings", "demon wings", "angel wings", "tail",
    "cat tail", "fox tail", "dog tail",
    # Age markers
    "loli", "shota", "milf", "old man", "old woman", "elderly",
    "child", "teenage", "young adult",
})

# Suffix tests for the long-tail "<modifier> hair / eyes / skin / breasts"
# patterns. WD14 vocabulary has thousands of "<color> hair" combinations
# (red hair, light blue hair, two-tone hair, gradient hair, ...) — we
# can't enumerate every one, so any tag ending in " hair" / " eyes" /
# " skin" / " breasts" / " thighs" is treated as identity.
_APPEARANCE_DROP_SUFFIXES: tuple[str, ...] = (
    " hair", " eyes", " skin", " breasts", " thighs", " ears",
)


def _is_appearance_tag(tag: str) -> bool:
    """True iff the WD14 tag describes a fixed physical-identity feature."""
    low = tag.lower().strip()
    if not low:
        return False
    if low in _APPEARANCE_KEEP_EXACT:
        return False
    if low in _APPEARANCE_DROP_EXACT:
        return True
    return any(low.endswith(suf) for suf in _APPEARANCE_DROP_SUFFIXES)


def _drop_appearance_tags(tags: list[str]) -> list[str]:
    """Strip every WD14 tag that names a fixed physical-identity feature.

    Used in character mode so the trigger word ends up the only signal
    correlated with the character's appearance across the dataset.
    """
    return [t for t in tags if not _is_appearance_tag(t)]

# Process-level cache for loaded WD14 taggers, keyed by full config tuple.
# EVA02-large weights are ~1.2GB so re-loading per request kills throughput.
_TAGGER_CACHE: dict[tuple[str, float, float, str], Any] = {}
_TAGGER_LOCK = threading.Lock()


def _get_tagger(
    model_id: str,
    general_threshold: float,
    character_threshold: float,
    device: str,
) -> Any:
    """Return a WD14Tagger that's loaded once per process per config."""
    from lorahub.core.tagging.wd14 import WD14Tagger  # noqa: PLC0415

    key = (model_id, general_threshold, character_threshold, device)
    with _TAGGER_LOCK:
        cached = _TAGGER_CACHE.get(key)
        if cached is not None:
            return cached
        tagger = WD14Tagger(
            model_id=model_id,
            general_threshold=general_threshold,
            character_threshold=character_threshold,
            device=device,
        )
        tagger.load()
        _TAGGER_CACHE[key] = tagger
        return tagger

_SMART_CAPTION_PROMPT_STYLE = (
    "You are writing the ultimate training caption for a STYLE LoRA in Stable Diffusion.\n\n"
    "Training rule: the trigger word will OWN the visual style. Your sentences must "
    "describe everything that VARIES across images (concrete entities, actions, clothing, "
    "spatial relationships) so the model learns to attribute the only shared signal "
    "(the style/render quality) strictly to the trigger word.\n\n"
    "Be exhaustive and concrete about CONTENTS, but completely BLIND to STYLE. For every "
    "visible element, write what it IS, not what it looks like as art. Cover, in this "
    "order:\n"
    " 1. Subjects: count and kind, apparent age range, gender.\n"
    " 2. Per subject: pose, action verb, facial expression, gaze direction.\n"
    " 3. Per subject: every visible clothing item with color and pattern, every accessory.\n"
    " 4. Held / nearby props: every distinct object.\n"
    " 5. Setting and background: location type, weather, time-of-day, or plain background "
    "color.\n"
    " 6. Composition and framing: shot type, camera angle.\n\n"
    "[STRICT STYLE DROP INSTRUCTION]\n"
    "Do NOT use the words anime, manga, illustration, cartoon, chibi, realistic, "
    "photorealistic, lineart, sketch, painterly, watercolor, monochrome, cel-shaded, "
    "flat color, screentone, 2d, 3d, 3dcg, render, octane, masterpiece.\n"
    "CRITICAL: Do NOT describe the specific visible craft (e.g., brushstrokes, edge "
    "softness, palette warmth, shading contrast, lighting quality, glossy skin). The "
    "lighting and rendering techniques MUST NOT BE DESCRIBED so the trigger word can "
    "absorb them. Describe ONLY physical facts.\n\n"
    "[TAG PRUNING INSTRUCTION]\n"
    "I will provide you with raw WD14 tags: {tags}\n"
    "Use them to ground your entity descriptions. MORE IMPORTANTLY, you must act as a "
    "filter. If the WD14 tags contain contradictions (e.g., both \"skirt\" and \"dress\", "
    "both \"black hair\" and \"blue hair\"), or hallucinated items, you must resolve them "
    "based on the actual image.\n\n"
    "[OUTPUT FORMAT]\n"
    "Output your response strictly in two parts:\n"
    "Part 1: 3-4 concise, highly dense natural language sentences describing the content.\n"
    "Part 2: A comma-separated list of the PRUNED and CORRECTED WD14 tags. Do NOT include "
    "any tags that contradict your sentences.\n\n"
    "Output ONLY the final caption text (Part 1 followed by Part 2, separated by a comma). "
    "Do not use headers or labels."
)

_SMART_CAPTION_PROMPT_CHARACTER = (
    "You are writing the ultimate training caption for a CHARACTER LoRA in Stable Diffusion.\n\n"
    "Training rule: the trigger word will OWN the character's FIXED PHYSICAL IDENTITY "
    "(face, hair, eyes, body, skin tone). Your sentences must describe everything else "
    "that VARIES across images (outfit, accessories, props, setting, pose, expression, "
    "framing) so the model attributes the only shared signal (the character's appearance) "
    "strictly to the trigger word.\n\n"
    "Be exhaustive and concrete about CONTENTS, but completely BLIND to physical "
    "appearance. For every visible element, write what it IS. Cover, in this order:\n"
    " 1. Subject framing: solo / 2girls / group; the character's role in the composition.\n"
    " 2. Pose, action verb, facial expression, gaze direction.\n"
    " 3. Outfit: every visible clothing item with color, pattern, material.\n"
    " 4. Every accessory: glasses, hat, scarf, gloves, necklace, earrings, ribbon, bag.\n"
    " 5. Held / nearby props: every distinct object the character is holding.\n"
    " 6. Setting: location type, weather, time-of-day.\n"
    " 7. Composition / framing: shot type, camera angle.\n\n"
    "[STRICT IDENTITY DROP INSTRUCTION]\n"
    "Do NOT describe any aspect of the character's PHYSICAL APPEARANCE: hair color, hair "
    "length or style (long hair, short hair, ponytail, twintails, braid, bangs, ahoge), "
    "eye color or shape, skin tone or texture (pale skin, tan, freckles), face shape, body "
    "proportions (slim, curvy, muscular, tall, short), height, breast size, age markers "
    "(young, elderly).\n"
    "CRITICAL: Do NOT describe vague praise (beautiful, pretty, handsome, cute, "
    "gorgeous). The trigger word MUST absorb the character's identity. Describe ONLY "
    "what changes from one image to the next.\n\n"
    "[TAG PRUNING INSTRUCTION]\n"
    "I will provide you with raw WD14 tags: {tags}\n"
    "Use them to ground your entity descriptions. MORE IMPORTANTLY, you must act as a "
    "filter. If the WD14 tags contain contradictions (e.g., both \"skirt\" and \"dress\", "
    "both \"standing\" and \"sitting\"), hallucinated items, or any tag describing the "
    "FIXED IDENTITY (hair color, eye color, body proportions...), you must drop them. "
    "Resolve real contradictions based on the actual image.\n\n"
    "[OUTPUT FORMAT]\n"
    "Output your response strictly in two parts:\n"
    "Part 1: 3-4 concise, highly dense natural language sentences describing the content.\n"
    "Part 2: A comma-separated list of the PRUNED and CORRECTED WD14 tags. Identity tags "
    "(hair / eyes / skin / body) MUST NOT appear. Do NOT include any tags that contradict "
    "your sentences.\n\n"
    "Output ONLY the final caption text (Part 1 followed by Part 2, separated by a comma). "
    "Do not use headers or labels."
)

_SMART_CAPTION_PROMPT_GENERAL = (
    "Write a 2-3 sentence natural-language description of the image for LoRA training. "
    "Cover subject, pose, clothing, background, lighting, composition. Plain English, no "
    "headers or labels.\n\nReference WD14 tags: {tags}"
)


# -- Tags-only mode (no VLM, LLM composes from WD14 tags alone) --------------
#
# Used when ``captionSource == "tags"`` — the LLM never sees the image,
# only the WD14 tag list. The prompts are explicit about that constraint
# so the model doesn't hallucinate details that aren't in the tags
# (a generic VLM prompt would casually invent "soft blue lighting" out
# of nothing). Each variant mirrors the VLM-mode counterpart so the
# downstream caption assembly (`_build_anima_caption`) can stay
# agnostic to which path produced ``nl_text``.

_TAGS_ONLY_PROMPT_STYLE = (
    "You are writing the ultimate training caption for a STYLE LoRA in Stable Diffusion. "
    "You DO NOT have access to the image — only the WD14 tagger's output for it. Treat "
    "the tag list as the ground truth and do NOT invent entities, props, or background "
    "elements unsupported by the tags.\n\n"
    "Training rule: the trigger word will OWN the visual style. Your sentences must "
    "describe everything that VARIES across images (concrete entities, actions, clothing, "
    "spatial relationships) so the model attributes the only shared signal (style/render "
    "quality) strictly to the trigger word.\n\n"
    "Be exhaustive about CONTENTS, but completely BLIND to STYLE. Walk the tag list and "
    "convert it into prose covering, in this order:\n"
    " 1. Subjects: count and kind, apparent age range, gender (from 1girl/2boys/etc).\n"
    " 2. Per subject: pose, action, expression, gaze.\n"
    " 3. Per subject: clothing items + colors, accessories.\n"
    " 4. Held / nearby props.\n"
    " 5. Setting and background tags, weather, time-of-day.\n"
    " 6. Composition and framing tags.\n\n"
    "[STRICT STYLE DROP INSTRUCTION]\n"
    "Do NOT use the words anime, manga, illustration, cartoon, chibi, realistic, "
    "photorealistic, lineart, sketch, painterly, watercolor, monochrome, cel-shaded, "
    "flat color, screentone, 2d, 3d, 3dcg, render, octane, masterpiece. Do NOT describe "
    "the visible craft (brushstrokes, edge softness, palette warmth, shading). The "
    "lighting and rendering techniques MUST NOT BE DESCRIBED so the trigger word can "
    "absorb them.\n\n"
    "[TAG PRUNING INSTRUCTION]\n"
    "Raw WD14 tags: {tags}\n"
    "Act as a filter. If tags contain contradictions (e.g., both \"skirt\" and \"dress\"), "
    "or low-confidence hallucinations, drop them. You may also drop quality / medium "
    "tags entirely.\n\n"
    "[OUTPUT FORMAT]\n"
    "Output strictly in two parts joined by a comma:\n"
    "Part 1: 3-4 dense natural-language sentences describing the content.\n"
    "Part 2: comma-separated list of the PRUNED WD14 tags. Drop any tag that contradicts "
    "your sentences or describes style/medium.\n\n"
    "Output ONLY the final caption text. No headers or labels."
)

_TAGS_ONLY_PROMPT_CHARACTER = (
    "You are writing the ultimate training caption for a CHARACTER LoRA in Stable "
    "Diffusion. You DO NOT have access to the image — only the WD14 tagger's output for "
    "it. Treat the tag list as the ground truth and do NOT invent entities unsupported "
    "by the tags.\n\n"
    "Training rule: the trigger word will OWN the character's FIXED PHYSICAL IDENTITY "
    "(face, hair, eyes, body, skin tone). Your sentences must describe everything else "
    "the tags expose (outfit, accessories, props, setting, pose, expression, framing) "
    "so the model attributes the only shared signal (appearance) strictly to the trigger.\n\n"
    "Walk the tag list and convert it into prose covering, in this order:\n"
    " 1. Subject framing: solo / 2girls / group; spatial role.\n"
    " 2. Pose, action, expression, gaze.\n"
    " 3. Outfit clothing tags + adjacent color tags, accessories.\n"
    " 4. Held / nearby object tags.\n"
    " 5. Setting / background tags, weather, time-of-day.\n"
    " 6. Composition / framing tags.\n\n"
    "[STRICT IDENTITY DROP INSTRUCTION]\n"
    "Do NOT describe any aspect of PHYSICAL APPEARANCE, even if the tag is in the list: "
    "hair color, hair length / style (long hair, short hair, ponytail, twintails, braid, "
    "bangs, ahoge), eye color, eye shape, skin tone / texture (pale skin, freckles), "
    "face shape, body proportions, height, breast size, age markers (young, elderly).\n"
    "CRITICAL: Do NOT use vague praise (beautiful, pretty, handsome, cute, gorgeous). "
    "The trigger word MUST absorb the character's identity.\n\n"
    "[TAG PRUNING INSTRUCTION]\n"
    "Raw WD14 tags: {tags}\n"
    "Act as a filter. Drop any contradicting tags, low-confidence hallucinations, AND "
    "every identity tag (hair / eyes / skin / face / body shape).\n\n"
    "[OUTPUT FORMAT]\n"
    "Output strictly in two parts joined by a comma:\n"
    "Part 1: 3-4 dense natural-language sentences describing the content.\n"
    "Part 2: comma-separated list of the PRUNED WD14 tags. Identity tags MUST NOT "
    "appear. Drop any tag contradicted by your sentences.\n\n"
    "Output ONLY the final caption text. No headers or labels."
)

_TAGS_ONLY_PROMPT_GENERAL = (
    "Write a 2-3 sentence natural-language description for an Anima LoRA training caption. "
    "You do NOT have the image — only the WD14 tagger's output for it. Compose the sentences "
    "strictly from what the tags support: subject, pose, framing, background, lighting, "
    "composition. If a given axis isn't supported by any tag, omit it instead of inventing. "
    "Plain English, no headers or labels.\n\nWD14 tags: {tags}"
)

_TORIIGATE_SYSTEM_PROMPT = (
    "You are image captioning expert. Describe user's picture according to requested "
    "format and instructions."
)
_TORIIGATE_SHORT_PROMPT = (
    "The caption for image should be quite short without long purple prose and slop. "
    "Cover main objects and details."
)


def _drop_tags(tags: list[str], drop: set[str]) -> list[str]:
    """Case-insensitive filter — keep order, drop matches."""
    return [t for t in tags if t.lower() not in drop]


def _split_normalize_tags(raw: str) -> list[str]:
    """Split a comma list, lowercase, dedupe in-order."""
    seen: set[str] = set()
    out: list[str] = []
    for piece in raw.split(","):
        t = piece.strip().lower()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _strip_forbidden_caption_tags(text: str) -> str:
    """Remove tags that LoraHub must never auto-write into captions."""
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        kept: list[str] = []
        for piece in line.split(","):
            token = piece.strip()
            low = token.lower()
            if low in _FORBIDDEN_CAPTION_TAGS or low.replace("_", " ") in _FORBIDDEN_CAPTION_TAGS:
                continue
            if token:
                kept.append(token)
        if kept:
            cleaned_lines.append(", ".join(kept))
    return "\n".join(cleaned_lines).strip()


def _toriigate_user_query(s1: _StageOneResult, caption_mode: str) -> str:
    drop = set(_QUALITY_NOISE_TAGS)
    if caption_mode == "style":
        drop |= _STYLE_NOISE_TAGS
    tags = _drop_tags(s1.general_tags, drop)
    if caption_mode == "character":
        tags = _drop_appearance_tags(tags)
    tags_string = " ".join(t.replace(" ", "_") for t in tags)
    query = f"# Captioning format:\n{_TORIIGATE_SHORT_PROMPT}\n"
    if tags_string:
        query += f"\n# Booru tags for the image\n[{tags_string}]\n"
    if s1.character_tags:
        chars = " ".join(t.replace(" ", "_") for t in s1.character_tags)
        query += (
            "\n# Characters on picture:\n"
            f"Here are names/tags for characters from the picture, make sure to use them: [{chars}].\n"
        )
    else:
        query += "\n# Characters on picture:\nAvoid to guess names for characters.\n"
    return query


def _clean_toriigate_caption(raw: str) -> str:
    """Drop prompt echoes and markdown section labels from ToriiGate output."""
    kept: list[str] = []
    for line in raw.splitlines():
        line = line.strip().strip("-").strip()
        if not line:
            continue
        low = line.lower()
        if low.startswith("#"):
            continue
        if low.startswith(("part 1:", "part 2:")):
            tail = line.split(":", 1)[1].strip()
            if re.search(
                r"\b(sentence|tag|caption|describ\w*|description|list|natural language)\b",
                tail,
                re.I,
            ):
                continue
            line = tail
        if re.search(r"\b(captioning format|booru tags|characters on picture)\b", line, re.I):
            continue
        if re.search(r"\b(output only|do not use|strict|precise list|cover main objects)\b", line, re.I):
            continue
        line = re.sub(r"^[-*]\s*[^:]{1,40}:\s*", "", line)
        kept.append(line)
    return " ".join(kept).strip() or raw.strip()


def _build_anima_caption(
    *,
    rating_tag: str | None,
    general_tags: list[str],
    character_tags: list[str],
    nl_text: str,
    caption_mode: str,
    trigger_word: str | None,
    strip_style_tags: bool = True,
) -> str:
    """Assemble a training caption.

    Layout (trigger-first):
      <trigger word>,
      [1girl/1boy/etc],            # subject-count tag from WD14
      [WD14 character predictions],
      <LLM payload>                # sentences + LLM-pruned tag list
                                    # (or empty when skip_llm path took over)

    The LLM output is expected to contain BOTH the natural-language
    description AND a corrected/filtered tag list (Part 1 + Part 2 in
    the prompt format), so the backend no longer pastes the raw WD14
    ``general_tags`` tail. That was the exact path that let style
    contradictions ("anime, realistic, painterly") and identity tags
    ("blue eyes, long hair") leak into character / style captions.

    ``general_tags`` is still used for two specific things:
      * extracting the subject-count tag (1girl, 2girls, ...) — these
        are high-confidence WD14 metadata that the LLM doesn't always
        bother to surface.
      * upstream filtering (in stage_one) of what the LLM sees as
        ``{tags}``, so style/identity words never reach the prompt.

    Anything else from WD14 reaches disk only via the LLM's pruned
    Part 2 list inside ``nl_text``. ``strip_style_tags`` is now a
    pure stage-one knob; the in-caption rebuild ignores it.
    """
    # Pick subject-count tag (1girl, 2girls, 1boy, etc.). Everything else from
    # ``general_tags`` is intentionally dropped: the LLM has already
    # written its own pruned tag list inside ``nl_text``.
    subject_pattern = (
        "1girl", "2girls", "3girls", "4girls", "5girls", "6+girls",
        "1boy", "2boys", "3boys", "multiple_girls", "multiple_boys",
        "solo", "no humans",
    )
    subject_tags = [t for t in general_tags if t in subject_pattern]

    # Trigger / character / artist line.
    trig = (trigger_word or "").strip().lower()
    if trig and caption_mode == "style":
        # Style LoRA → format as @artist-style trigger if not already.
        if not trig.startswith("@"):
            trig = f"@{trig}"

    line2 = ", ".join([*subject_tags, *character_tags])

    parts: list[str] = []
    # Trigger word leads — kohya keep_tokens convention.
    if trig:
        parts.append(trig)
    line2 = _strip_forbidden_caption_tags(line2)
    nl_clean = _strip_forbidden_caption_tags(nl_text.strip())
    if line2:
        parts.append(line2)
    if nl_clean:
        parts.append(nl_clean)
    # Suppress unused-arg linter: kept in signature for back-compat
    # callers + future reintroduction of stage-three filtering.
    _ = (rating_tag, strip_style_tags)
    return ",\n".join(parts)


class SmartCaptionBatchInput(BaseModel):
    path: str
    recursive: bool = False
    taggerModel: str = "SmilingWolf/wd-eva02-large-tagger-v3"
    visionTask: str = "tagging.assist"
    mergeStrategy: str = "replace"
    device: str = "auto"
    generalThreshold: float = 0.35
    characterThreshold: float = 0.85
    captionMode: str = "style"  # general | style | character
    promptTemplate: str | None = None
    # "vlm" — multimodal LLM sees the image directly (default behaviour
    #         since the feature shipped). Best caption quality but
    #         requires a vision-capable model and burns image tokens.
    # "tags" — LLM only sees the WD14 tag list, never the image. Cheap
    #          and works against text-only models; useful when the
    #          configured VLM is rate-limited / quota-exhausted, or the
    #          user wants a faster cheaper pass.
    captionSource: str = "vlm"
    triggerWord: str | None = None
    # Style/medium descriptors (anime, illustration, lineart, monochrome, ...)
    # are stripped from both the WD14 reference list shown to the LLM and
    # the final caption tail when this is True AND captionMode == "style".
    # Off by default for character / general modes since "anime" anchors
    # rendering for those.
    stripStyleTags: bool = True
    # When False, skip the WD14 tagger entirely. The caption then reduces
    # to ``[trigger,] [<LLM nl_text>]`` (or just the trigger word in
    # ``style`` mode where the nl text is also disabled). Useful for
    # users who want a clean trigger-only training set, or who do their
    # own tagging upstream.
    useWd14: bool = True
    # Parallelism + reliability knobs.
    #
    # Pipeline shape:
    #   images -> [WD14 pool, taggerConcurrency workers]
    #          -> intermediate queue (tags + base64 image)
    #          -> [VLM pool, concurrency workers]
    #          -> caption written to disk
    #
    # ``concurrency`` controls the VLM stage (network-bound — we want
    # this fairly high so the API rate is the only floor). The default
    # of 8 covers most providers without 429s; cap is 64 because beyond
    # that the upstream usually starts throttling anyway.
    #
    # ``taggerConcurrency`` controls the WD14 stage. WD14 is a
    # single-GPU ONNX session — running >2-3 inferences in parallel on
    # one GPU saturates the SM scheduler with no real wall-clock gain
    # and risks OOM on smaller cards. Cap at 4.
    #
    # Per-image timeout protects the VLM stage; WD14 is fast enough we
    # don't bother timing it (a hung WD14 means the GPU is wedged and
    # the user needs to restart anyway).
    concurrency: int = 8
    taggerConcurrency: int = 2
    perImageTimeoutSec: float = 90.0
    maxRetries: int = 2
    # Skip images that already have a non-empty .txt sidecar. Useful
    # for re-running a batch that hit upstream rate-limits — the
    # second run only retries the images that failed the first time.
    # Defaults to true so the common case ("don't waste tokens
    # re-captioning already-tagged images") just works.
    skipExisting: bool = True


@dataclass
class _StageOneResult:
    """Output of the WD14 / image-prep stage handed to the VLM stage."""

    img_path: Path
    rating_name: str | None
    general_tags: list[str]
    character_tags: list[str]
    prompt_text: str
    data_url: str
    # "vlm" → stage two sends an image_url + text content list.
    # "tags" → stage two sends a single text message; the LLM never
    # sees the picture and composes the natural-language sentence
    # from ``prompt_text`` (which already has the WD14 tag list
    # baked in via the tags-only prompt template).
    caption_source: str = "vlm"
    # Mirrors the SmartCaptionBatchInput flag so stage two can decide
    # whether to also strip style tags from the final assembled tail
    # (in addition to the already-filtered tags_for_prompt).
    strip_style_tags: bool = True
    # Whether stage two should call the LLM at all. False when WD14 is
    # disabled AND mode==style (the trigger word alone is the whole
    # caption, no nl text needed) — this is the cleanest setup for
    # training a style LoRA where every caption being identical is the
    # desired behaviour.
    skip_llm: bool = False


def _smart_caption_stage_one(
    img_path: Path,
    tagger: WD14Tagger | None,
    caption_mode: str,
    *,
    caption_source: str = "vlm",
    strip_style_tags: bool = True,
    use_wd14: bool = True,
    custom_prompt_template: str | None = None,
    trigger_word: str | None = None,
) -> _StageOneResult:
    """Run WD14 tagging + image prep — everything that doesn't need the VLM.

    Pulled out of ``_smart_caption_single_image`` so the batch worker
    can run this on a small GPU-bound pool while the VLM stage runs on
    a much wider network-bound pool. Side-effect free: returns a plain
    dataclass, doesn't write files or touch the store.

    When ``use_wd14`` is False the WD14 tagger is bypassed entirely and
    every tag-derived field is empty; ``tagger`` may be ``None``.

    The LLM is the source of truth for everything except the trigger
    word — its output covers both natural-
    language description AND the pruned-and-corrected tag list. The
    backend no longer re-pastes raw WD14 output into the caption tail
    because that's exactly the path that lets contradictions and
    style-words leak in. Only ``use_wd14=False`` AND ``mode==style``
    skips the LLM entirely (the trigger word alone is the caption).
    """
    import base64  # noqa: PLC0415
    import mimetypes  # noqa: PLC0415

    if use_wd14 and tagger is not None:
        tag_result = tagger.tag_image(img_path)
        general_tags_underscore = [t.name for t in tag_result.general]
        general_tags = [t.replace("_", " ").lower() for t in general_tags_underscore]
        character_tags = [
            t.name.replace("_", " ").lower() for t in tag_result.character
        ]
        rating_name = tag_result.rating.name if tag_result.rating else None
    else:
        general_tags = []
        character_tags = []
        rating_name = None

    # The LLM is now responsible for both nl_text AND the pruned tag
    # list (Part 1 + Part 2 in the new prompt format). The only path
    # that skips the LLM is "WD14 disabled in style mode" — there's
    # nothing for the LLM to write in that case (no image input, no
    # tag context, just the trigger word).
    skip_llm = (
        caption_mode == "style"
        and not use_wd14
        and caption_source != "toriigate"
    )

    # Build the LLM-facing reference tags. Always strip the quality-noise
    # set; additionally strip the style/medium set when requested AND the
    # caption mode is "style" (other modes leave them in — character LoRAs
    # legitimately benefit from "anime" anchoring the rendering). For
    # character mode also strip every appearance/identity tag so the LLM
    # can't latch onto "blue eyes long hair" and reintroduce them in its
    # natural-language sentence.
    drop_for_prompt = set(_QUALITY_NOISE_TAGS)
    if strip_style_tags and caption_mode == "style":
        drop_for_prompt = drop_for_prompt | _STYLE_NOISE_TAGS
    pre_filtered = _drop_tags(general_tags, drop_for_prompt)
    if caption_mode == "character":
        pre_filtered = _drop_appearance_tags(pre_filtered)
    tags_for_prompt = ", ".join(pre_filtered)

    # When skipping the LLM there is no prompt to assemble and no image
    # to encode — return early with the empty payload.
    if skip_llm:
        return _StageOneResult(
            img_path=img_path,
            rating_name=rating_name,
            general_tags=general_tags,
            character_tags=character_tags,
            prompt_text="",
            data_url="",
            caption_source=caption_source,
            strip_style_tags=strip_style_tags,
            skip_llm=True,
        )

    if caption_source == "tags":
        # Tags-only path — skip the base64 encode entirely; stage two
        # sends a plain text completion request.
        data_url = ""
        if caption_mode == "style":
            prompt_template = _TAGS_ONLY_PROMPT_STYLE
        elif caption_mode == "character":
            prompt_template = _TAGS_ONLY_PROMPT_CHARACTER
        else:
            prompt_template = _TAGS_ONLY_PROMPT_GENERAL
    else:
        mime = mimetypes.guess_type(str(img_path))[0] or "image/png"
        data = img_path.read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        data_url = f"data:{mime};base64,{b64}"
        if caption_mode == "style":
            prompt_template = _SMART_CAPTION_PROMPT_STYLE
        elif caption_mode == "character":
            prompt_template = _SMART_CAPTION_PROMPT_CHARACTER
        else:
            prompt_template = _SMART_CAPTION_PROMPT_GENERAL
    if custom_prompt_template and custom_prompt_template.strip():
        prompt_text = (
            custom_prompt_template
            .replace("{tags}", tags_for_prompt)
            .replace("{wd14_tags}", tags_for_prompt)
            .replace("{trigger}", (trigger_word or "").strip())
        )
    else:
        prompt_text = prompt_template.format(tags=tags_for_prompt)
    # Note: the new prompt templates already embed the
    # [STRICT STYLE DROP INSTRUCTION] / [STRICT IDENTITY DROP INSTRUCTION]
    # blocks inline; we no longer prepend the legacy
    # ``_STYLE_DROP_INSTRUCTION`` override (it's still defined as
    # documentation but unused at runtime).

    s1 = _StageOneResult(
        img_path=img_path,
        rating_name=rating_name,
        general_tags=general_tags,
        character_tags=character_tags,
        prompt_text=prompt_text,
        data_url=data_url,
        caption_source=caption_source,
        strip_style_tags=strip_style_tags,
        skip_llm=False,
    )
    if caption_source == "toriigate" and not (custom_prompt_template or "").strip():
        s1.prompt_text = _toriigate_user_query(s1, caption_mode)
    return s1


def _smart_caption_stage_two(
    s1: _StageOneResult,
    ai_store: Any,
    route: Any,
    merge_strategy: str,
    store: ImageStudioStore,
    caption_mode: str,
    trigger_word: str | None,
) -> dict[str, Any]:
    """Network-bound VLM (or text-only LLM) call + caption assembly + disk + store write."""
    from datetime import UTC, datetime  # noqa: PLC0415

    from lorahub.core.ai import client as ai_client  # noqa: PLC0415

    # Style-mode (or any LLM-skip flow): no provider call, no nl_text.
    # The caption is whatever ``_build_anima_caption`` produces from
    # ``trigger_word + WD14 tag tail`` alone. Provider metadata stays
    # blank so the store row reflects that no LLM saw this image.
    if s1.skip_llm:
        new_caption = _build_anima_caption(
            rating_tag=s1.rating_name,
            general_tags=s1.general_tags,
            character_tags=s1.character_tags,
            nl_text="",
            caption_mode=caption_mode,
            trigger_word=trigger_word,
            strip_style_tags=s1.strip_style_tags,
        )

        caption_path = s1.img_path.with_suffix(".txt")
        existing = caption_path.read_text(encoding="utf-8") if caption_path.is_file() else ""
        if merge_strategy == "append":
            new_caption = (existing.strip() + "\n" + new_caption).strip()
        elif merge_strategy == "prepend":
            new_caption = (new_caption + "\n" + existing.strip()).strip()
        caption_path.write_text(new_caption, encoding="utf-8")

        ann = store.get_annotation(str(s1.img_path))
        if ann is None:
            ann = ImageAnnotation(
                image_path=str(s1.img_path),
                sha256=_file_sha256(s1.img_path),
            )
        ann.ai_caption = ""
        ann.ai_caption_provider = ""
        ann.ai_caption_at = datetime.now(UTC).isoformat()
        store.upsert_annotation(ann)

        return {
            "path": str(s1.img_path),
            "wd14Tags": ", ".join(s1.general_tags),
            "caption": new_caption,
        }

    messages: list[dict[str, Any]] = []
    is_toriigate = s1.caption_source == "toriigate"
    if is_toriigate:
        messages.append({"role": "system", "content": _TORIIGATE_SYSTEM_PROMPT})
    elif route.system_prompt:
        messages.append({"role": "system", "content": route.system_prompt})
    if s1.caption_source == "tags":
        # Text-only path — many cheap / non-vision LLMs reject the
        # multimodal content list with a 400 ("invalid content type:
        # image_url") so we send a plain string. The prompt template
        # already contains the WD14 tag list inline.
        messages.append({"role": "user", "content": s1.prompt_text})
    else:
        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": s1.data_url}},
                {"type": "text", "text": s1.prompt_text},
            ],
        })

    result = ai_client.invoke(
        ai_store,
        provider_id=route.provider_id,
        model_id=route.model_id,
        messages=messages,
        route=route,
    )

    nl_text = result.content.strip()
    if is_toriigate:
        nl_text = _clean_toriigate_caption(nl_text)
    new_caption = _build_anima_caption(
        rating_tag=s1.rating_name,
        general_tags=s1.general_tags,
        character_tags=s1.character_tags,
        nl_text=nl_text,
        caption_mode=caption_mode,
        trigger_word=trigger_word,
        strip_style_tags=s1.strip_style_tags,
    )

    caption_path = s1.img_path.with_suffix(".txt")
    existing = caption_path.read_text(encoding="utf-8") if caption_path.is_file() else ""
    if merge_strategy == "append":
        new_caption = (existing.strip() + "\n" + new_caption).strip()
    elif merge_strategy == "prepend":
        new_caption = (new_caption + "\n" + existing.strip()).strip()
    # else replace — keep as-is.

    caption_path.write_text(new_caption, encoding="utf-8")

    ann = store.get_annotation(str(s1.img_path))
    if ann is None:
        ann = ImageAnnotation(
            image_path=str(s1.img_path),
            sha256=_file_sha256(s1.img_path),
        )
    ann.ai_caption = nl_text
    ann.ai_caption_provider = f"{result.provider_name}/{result.model_id}"
    ann.ai_caption_at = datetime.now(UTC).isoformat()
    store.upsert_annotation(ann)

    return {
        "path": str(s1.img_path),
        "wd14Tags": ", ".join(s1.general_tags),
        "caption": new_caption,
    }


def _smart_caption_single_image(
    img_path: Path,
    tagger: WD14Tagger | None,
    ai_store: Any,
    route: Any,
    merge_strategy: str,
    store: ImageStudioStore,
    caption_mode: str = "general",
    trigger_word: str | None = None,
    strip_style_tags: bool = True,
    *,
    caption_source: str = "vlm",
    use_wd14: bool = True,
    prompt_template: str | None = None,
) -> dict[str, Any]:
    """Single-image pipeline kept for the /single endpoint and tests.

    Composes ``_smart_caption_stage_one`` and ``_smart_caption_stage_two``.
    The batch path uses the two stages directly so it can keep them
    on separate thread pools (WD14 on a small GPU pool, VLM on a wide
    network pool).
    """
    s1 = _smart_caption_stage_one(
        img_path,
        tagger,
        caption_mode,
        caption_source=caption_source,
        strip_style_tags=strip_style_tags,
        use_wd14=use_wd14,
        custom_prompt_template=prompt_template,
        trigger_word=trigger_word,
    )
    return _smart_caption_stage_two(
        s1, ai_store, route, merge_strategy, store, caption_mode, trigger_word,
    )


@router.post("/ai/smart-caption", status_code=202)
def ai_smart_caption_batch(body: SmartCaptionBatchInput) -> dict[str, Any]:
    """Run WD14 tagging + vision LLM captioning for all images in a directory.

    Background-task shape (unblocks uvicorn for big batches): the request
    validates inputs, returns a session_id immediately with HTTP 202, and
    runs the for-loop in a worker thread. Progress is polled via
    ``GET /api/image-studio/ai/smart-caption/status/<id>``; cancel via
    ``POST /api/image-studio/ai/smart-caption/cancel/<id>``.

    Synchronous return shape (the legacy ``processed`` / ``results`` / ``errors``
    fields) is preserved on the status endpoint when the session finishes,
    so existing callers can poll-then-pull without code changes beyond
    going through the session_id.
    """
    from lorahub.api import app as app_mod  # noqa: PLC0415

    directory = _resolve_under_roots(body.path)
    if not directory.is_dir():
        raise HTTPException(400, "not a directory")

    ai_store = app_mod._ai_store
    if ai_store is None:
        raise HTTPException(503, "AI store not initialised")

    route = ai_store.get_route(body.visionTask)
    if route is None or not (route.provider_id and route.model_id):
        route = ai_store.get_route("global.default")
    if route is None or not (route.provider_id and route.model_id):
        raise HTTPException(409, f"no AI route for task {body.visionTask!r}")

    images = _scan_images(directory, body.recursive)
    if body.skipExisting:
        # Drop images that already have a non-empty .txt sidecar.
        # Empty/zero-byte sidecars are NOT counted as completed —
        # they're usually the half-written remnant of a crashed
        # caption attempt and should be reprocessed.
        before = len(images)
        images = [
            p for p in images
            if not (p.with_suffix(".txt").is_file()
                    and p.with_suffix(".txt").stat().st_size > 0)
        ]
        skipped = before - len(images)
    else:
        skipped = 0
    store = _store()
    task = _task_store().create(
        kind=_KIND_SMART_CAPTION,
        title=f"smart caption:{directory.name}",
        metadata={
            "path": str(directory),
            "recursive": body.recursive,
            "visionTask": body.visionTask,
            "captionMode": body.captionMode,
            "captionSource": body.captionSource,
            "skipExisting": body.skipExisting,
            "skipped": skipped,
            "total": len(images),
        },
    )
    session = _SmartCaptionSession(
        session_id=task.id,
        path=str(directory),
        total=len(images),
        task_kind=_KIND_SMART_CAPTION,
    )
    _task_store().append_event(
        session.session_id,
        TaskEvent(
            level="info",
            message="smart caption queued",
            percent=0,
            payload={"total": len(images), "skipped": skipped},
            ts=_time.time(),
        ),
    )
    with _smart_caption_lock:
        _smart_caption_sessions[session.session_id] = session

    def run() -> None:
        # Two-stage pipeline:
        #   stage 1 (WD14 + image prep) on a small GPU-bound pool
        #   intermediate queue (bounded so we don't blow RAM with
        #     base64-encoded payloads when stage 2 falls behind)
        #   stage 2 (VLM call + write) on a wide network-bound pool
        #
        # We deliberately do NOT use one ThreadPoolExecutor for both
        # stages: that bottlenecks the VLM stage to the GPU pool's
        # worker count and was the throughput floor we hit during
        # smoke testing on the qing0ying0 dataset.
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as _Timeout  # noqa: PLC0415
        import queue as _queue  # noqa: PLC0415

        vlm_workers = max(1, min(int(body.concurrency or 1), 64))
        wd14_workers = max(1, min(int(body.taggerConcurrency or 1), 4))
        timeout = float(body.perImageTimeoutSec or 90.0)
        max_retries = max(0, int(body.maxRetries or 0))

        # Bounded intermediate queue. Stage 2 is the slow stage (VLM
        # network call); buffering more than ~2x the VLM pool keeps
        # workers fed during transients without retaining hundreds of
        # base64-encoded images in RAM (each ~1-3 MiB).
        s1_queue: _queue.Queue[_StageOneResult | None] = _queue.Queue(
            maxsize=max(vlm_workers * 2, 8)
        )
        # Stage-one errors get short-circuited to the session error
        # list directly — no need to round-trip them through stage 2.
        # Tracked by a counter so the stage-two consumer knows when
        # producers are done.
        s1_done = threading.Event()

        def stage_one_worker(img_path: Path) -> None:
            if session.should_stop():
                return
            try:
                s1 = _smart_caption_stage_one(
                    img_path,
                    tagger,
                    body.captionMode,
                    caption_source=body.captionSource,
                    strip_style_tags=body.stripStyleTags,
                    use_wd14=body.useWd14,
                    custom_prompt_template=body.promptTemplate,
                    trigger_word=body.triggerWord,
                )
            except Exception as exc:  # noqa: BLE001
                err_msg = f"WD14: {type(exc).__name__}: {exc}"
                session.add_error(str(img_path), err_msg, img_path.name)
                return
            # block-put so we honour back-pressure when stage 2 is
            # behind. Cancel checks are cheap so just retry every
            # second instead of using a queue timeout exception path.
            while not session.should_stop():
                try:
                    s1_queue.put(s1, timeout=1.0)
                    return
                except _queue.Full:
                    continue

        def stage_two_worker() -> None:
            while True:
                try:
                    s1 = s1_queue.get(timeout=1.0)
                except _queue.Empty:
                    if s1_done.is_set() and s1_queue.empty():
                        return
                    continue
                if s1 is None:
                    # Sentinel — push it back and exit so peer
                    # workers also see it. Using None as the sentinel
                    # avoids needing a separate "drained" event.
                    s1_queue.put(None)
                    return
                if session.should_stop():
                    s1_queue.task_done()
                    continue
                last_err: Exception | None = None
                for attempt in range(max_retries + 1):
                    if session.should_stop():
                        break
                    try:
                        item = _smart_caption_stage_two(
                            s1, ai_store, route, body.mergeStrategy, store,
                            body.captionMode, body.triggerWord,
                        )
                        session.add_result(item, s1.img_path.name)
                        last_err = None
                        break
                    except Exception as exc:  # noqa: BLE001
                        last_err = exc
                        if attempt < max_retries:
                            # 429 / quota errors need much longer backoff —
                            # the upstream's window is usually minute-scale,
                            # so 2-4s isn't enough to clear the bucket. We
                            # detect "429" / "rate" / "exhausted" / "quota"
                            # in the message and step up to 30s+30s*attempt
                            # (capped at 120s). Other errors keep the fast
                            # 2-4s exponential backoff.
                            msg_l = str(exc).lower()
                            is_rate_limit = (
                                "429" in msg_l
                                or "rate" in msg_l
                                or "exhausted" in msg_l
                                or "quota" in msg_l
                            )
                            if is_rate_limit:
                                _time.sleep(min(30.0 + 30.0 * attempt, 120.0))
                            else:
                                _time.sleep(min(2.0 ** attempt, 4.0))
                            continue
                if last_err is not None:
                    err_msg = f"VLM: {type(last_err).__name__}: {last_err}"
                    session.add_error(str(s1.img_path), err_msg, s1.img_path.name)
                s1_queue.task_done()

        try:
            _task_store().update(session.session_id, status="running", percent=0)
            if body.useWd14:
                session._append_task_event(
                    "loading wd14",
                    percent=0,
                    payload={
                        "model_id": body.taggerModel,
                        "device": body.device,
                    },
                )
                tagger = _get_tagger(
                    body.taggerModel,
                    body.generalThreshold,
                    body.characterThreshold,
                    body.device,
                )
            else:
                # WD14 disabled — caption is trigger + LLM nl_text only.
                tagger = None
            # Producer pool — small, GPU-bound. We use the executor as
            # a futures collector so we can apply a per-image timeout
            # against stage 1 (a hung WD14 forward shouldn't stall
            # producers indefinitely).
            with ThreadPoolExecutor(
                max_workers=wd14_workers,
                thread_name_prefix=f"sc-wd14-{session.session_id[:8]}",
            ) as wd14_pool, ThreadPoolExecutor(
                max_workers=vlm_workers,
                thread_name_prefix=f"sc-vlm-{session.session_id[:8]}",
            ) as vlm_pool:
                # Spin up consumers first so producers can start
                # back-pressuring immediately.
                vlm_futures = [
                    vlm_pool.submit(stage_two_worker)
                    for _ in range(vlm_workers)
                ]
                wd14_futures = {
                    wd14_pool.submit(stage_one_worker, p): p for p in images
                }

                # Wait for stage-one producers, applying the per-image
                # timeout against each as a stuck-WD14 safety net.
                for fut in list(wd14_futures.keys()):
                    if session.should_stop():
                        break
                    try:
                        fut.result(timeout=timeout)
                    except _Timeout:
                        p = wd14_futures[fut]
                        session.add_error(
                            str(p), f"WD14 timeout after {timeout:.0f}s", p.name,
                        )
                        fut.cancel()
                    except Exception as exc:  # noqa: BLE001
                        # Stage-one worker swallows errors; reaching
                        # here means the executor itself failed.
                        p = wd14_futures[fut]
                        session.add_error(str(p), str(exc), p.name)

                # Producers done — drop a sentinel so each consumer
                # eventually exits. We push exactly one None and rely
                # on the consumer chain (each one re-pushes it before
                # exiting) to fan it out.
                s1_done.set()
                s1_queue.put(None)

                # Wait for VLM consumers to drain. fut.result() with no
                # timeout is fine here: any stuck VLM request has its
                # own per-call timeout via stage_two_worker's retries.
                for fut in vlm_futures:
                    try:
                        fut.result(timeout=timeout * (max_retries + 2))
                    except Exception:  # noqa: BLE001
                        # A worker dying is a bug, not a per-image
                        # error; ignore so we still finish the batch.
                        pass

            session.finish("succeeded" if not session.should_stop() else "canceled")
        except Exception as exc:  # noqa: BLE001
            # Catastrophic failure (e.g. AI route token revoked mid-run).
            # Mark the session failed instead of leaking the traceback into
            # the request thread (which has long since returned 202).
            session.set_error(str(exc))
            session.finish("failed")

    threading.Thread(
        target=run,
        name=f"smart-caption-{session.session_id[:8]}",
        daemon=True,
    ).start()

    return {
        "session_id": session.session_id,
        "total": len(images),
        "skipped": skipped,
        "status_url": (
            f"/api/image-studio/ai/smart-caption/status/{session.session_id}"
        ),
    }


@router.get("/ai/smart-caption/status/{session_id}")
def ai_smart_caption_status(session_id: str) -> dict[str, Any]:
    """Poll a smart-caption batch session's progress and final results."""
    with _smart_caption_lock:
        session = _smart_caption_sessions.get(session_id)
    if session is None:
        persisted = _persisted_task_result(session_id, _KIND_SMART_CAPTION)
        if persisted is not None:
            return persisted
        raise HTTPException(404, "session not found")
    return session.snapshot()


@router.post("/ai/smart-caption/cancel/{session_id}")
def ai_smart_caption_cancel(session_id: str) -> dict[str, Any]:
    """Request a running smart-caption batch session to stop after the current image."""
    with _smart_caption_lock:
        session = _smart_caption_sessions.get(session_id)
    if session is None:
        raise HTTPException(404, "session not found")
    session.request_stop()
    return {"session_id": session_id, "stop_requested": True}


class SmartCaptionSingleInput(BaseModel):
    path: str
    taggerModel: str = "SmilingWolf/wd-eva02-large-tagger-v3"
    visionTask: str = "tagging.assist"
    mergeStrategy: str = "replace"
    device: str = "auto"
    generalThreshold: float = 0.35
    characterThreshold: float = 0.85
    captionMode: str = "style"
    promptTemplate: str | None = None
    captionSource: str = "vlm"
    triggerWord: str | None = None
    stripStyleTags: bool = True
    useWd14: bool = True


@router.post("/ai/smart-caption/single")
def ai_smart_caption_single(body: SmartCaptionSingleInput) -> dict[str, Any]:
    """Run WD14 tagging + vision LLM captioning for a single image."""
    from lorahub.api import app as app_mod  # noqa: PLC0415

    file_path = _resolve_under_roots(body.path)
    if not file_path.is_file():
        raise HTTPException(404, "image not found")

    ai_store = app_mod._ai_store
    if ai_store is None:
        raise HTTPException(503, "AI store not initialised")

    route = ai_store.get_route(body.visionTask)
    if route is None or not (route.provider_id and route.model_id):
        route = ai_store.get_route("global.default")
    if route is None or not (route.provider_id and route.model_id):
        raise HTTPException(409, f"no AI route for task {body.visionTask!r}")

    if body.useWd14:
        tagger = _get_tagger(
            body.taggerModel,
            body.generalThreshold,
            body.characterThreshold,
            body.device,
        )
    else:
        tagger = None

    store = _store()
    try:
        item = _smart_caption_single_image(
            file_path, tagger, ai_store, route, body.mergeStrategy, store,
            caption_mode=body.captionMode,
            trigger_word=body.triggerWord,
            strip_style_tags=body.stripStyleTags,
            caption_source=body.captionSource,
            use_wd14=body.useWd14,
            prompt_template=body.promptTemplate,
        )
        return {"ok": True, **item}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"smart caption failed: {exc}") from exc


# --------------------------------------------------------------------------- #
# Independent stage endpoints
# --------------------------------------------------------------------------- #
#
# /ai/smart-caption is a one-shot two-stage pipeline (WD14 → VLM). For
# users who want to run the steps separately — e.g. WD14 first, eyeball
# the tags, then ask the VLM to write the caption — the two endpoints
# below expose the stages as standalone tools. The composed
# /ai/smart-caption endpoint stays for the common case.


class Wd14PrefilterInput(BaseModel):
    """Input for /ai/wd14-prefilter — single image, WD14 only."""

    path: str
    taggerModel: str = "SmilingWolf/wd-eva02-large-tagger-v3"
    device: str = "auto"
    generalThreshold: float = 0.35
    characterThreshold: float = 0.85
    captionMode: str = "general"
    promptTemplate: str | None = None
    captionSource: str = "vlm"
    triggerWord: str | None = None
    stripStyleTags: bool = True


@router.post("/ai/wd14-prefilter")
def ai_wd14_prefilter(body: Wd14PrefilterInput) -> dict[str, Any]:
    """Run WD14 + prompt assembly for one image, no LLM call, no disk write.

    Returns the same fields the batch worker would hand to stage two,
    so a caller can review the WD14 output and the assembled prompt
    before committing to a (paid) VLM call. The result's ``promptText``
    and ``dataUrl`` are accepted as-is by /ai/vlm-anima-rewrite.
    """
    file_path = _resolve_under_roots(body.path)
    if not file_path.is_file():
        raise HTTPException(404, "image not found")

    tagger = _get_tagger(
        body.taggerModel,
        body.generalThreshold,
        body.characterThreshold,
        body.device,
    )

    s1 = _smart_caption_stage_one(
        file_path,
        tagger,
        body.captionMode,
        caption_source=body.captionSource,
        strip_style_tags=body.stripStyleTags,
        use_wd14=True,
        custom_prompt_template=body.promptTemplate,
        trigger_word=body.triggerWord,
    )
    return {
        "path": str(s1.img_path),
        "ratingName": s1.rating_name,
        "generalTags": s1.general_tags,
        "characterTags": s1.character_tags,
        "promptText": s1.prompt_text,
        # data_url can be multi-MB; only return when the caller will
        # actually need it (vlm path). Tags-only path doesn't.
        "dataUrl": s1.data_url if body.captionSource != "tags" else "",
        "captionSource": s1.caption_source,
        "stripStyleTags": s1.strip_style_tags,
        "skipLlm": s1.skip_llm,
    }


class VlmAnimaRewriteInput(BaseModel):
    """Input for /ai/vlm-anima-rewrite — runs the LLM + writes caption.

    Accepts the per-image fields produced by /ai/wd14-prefilter (or
    hand-built from any source that can fill the same shape).
    ``visionTask`` selects the AI route, ``mergeStrategy`` controls
    how the new caption is merged with the existing .txt sidecar.
    """

    path: str
    visionTask: str = "tagging.assist"
    mergeStrategy: str = "replace"
    captionMode: str = "general"
    captionSource: str = "vlm"
    triggerWord: str | None = None
    stripStyleTags: bool = True
    # Stage-one outputs the caller is forwarding. ``promptText`` is the
    # one field the LLM actually sees. ``dataUrl`` is required for vlm
    # source, empty for tags source.
    ratingName: str | None = None
    generalTags: list[str] = []
    characterTags: list[str] = []
    promptText: str = ""
    dataUrl: str = ""
    skipLlm: bool = False


@router.post("/ai/vlm-anima-rewrite")
def ai_vlm_anima_rewrite(body: VlmAnimaRewriteInput) -> dict[str, Any]:
    """Stage two of smart-caption as a standalone tool.

    Skips the WD14 step entirely; expects the caller (typically
    /ai/wd14-prefilter, but can be hand-built) to have already filled
    in ``promptText`` / ``dataUrl`` / tag fields. Calls the configured
    vision route, assembles the Anima caption, writes the .txt sidecar
    and updates the annotation row — exactly what stage two does inside
    /ai/smart-caption.
    """
    from lorahub.api import app as app_mod  # noqa: PLC0415

    file_path = _resolve_under_roots(body.path)
    if not file_path.is_file():
        raise HTTPException(404, "image not found")

    ai_store = app_mod._ai_store
    if ai_store is None:
        raise HTTPException(503, "AI store not initialised")

    route = ai_store.get_route(body.visionTask)
    if route is None or not (route.provider_id and route.model_id):
        route = ai_store.get_route("global.default")
    if route is None or not (route.provider_id and route.model_id):
        raise HTTPException(409, f"no AI route for task {body.visionTask!r}")

    s1 = _StageOneResult(
        img_path=file_path,
        rating_name=body.ratingName,
        general_tags=list(body.generalTags),
        character_tags=list(body.characterTags),
        prompt_text=body.promptText,
        data_url=body.dataUrl,
        caption_source=body.captionSource,
        strip_style_tags=body.stripStyleTags,
        skip_llm=body.skipLlm,
    )
    store = _store()
    try:
        item = _smart_caption_stage_two(
            s1,
            ai_store,
            route,
            body.mergeStrategy,
            store,
            body.captionMode,
            body.triggerWord,
        )
        return {"ok": True, **item}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"vlm rewrite failed: {exc}") from exc


# --------------------------------------------------------------------------- #
# Trigger word suggestion
# --------------------------------------------------------------------------- #
#
# A trigger word is the rare-token-grade label LoRA training relies on to
# bind a learned concept ("blue-haired magical girl with a star wand")
# without leaking into normal prompt vocabulary. The task here is "per
# image, suggest 1-3 trigger word *candidates* that capture this image's
# distinctive identity content" — what the user would later wrap into
# the dataset's keepTokens prefix.
#
# Why per-image and not dataset-level: the user's existing inspector
# panel already renders ann.aiTriggerWords as chips next to each image,
# and the per-image signal is what makes "is this image off-distribution
# for the chosen trigger?" auditable. A dataset-level top-k can be
# computed cheaply over the per-image results (collections.Counter on
# the union of all suggestions) — this endpoint returns the per-image
# results plus that aggregation as a `dataset_top` field.

_TRIGGER_WORD_PROMPT = (
    "You are helping pick LoRA training trigger words for an image dataset. "
    "Look at this single image and suggest 1-3 short, content-distinctive "
    "phrases that uniquely identify what's in it — the character / concept / "
    "object / scene specifics that this image is *about*. "
    "\n"
    "Strict rules:\n"
    "- Phrases must be 1-3 words each, lowercase, English.\n"
    "- Prefer concrete identity ('crimson robe', 'lop ears', 'glass dome city') "
    "over generic descriptors ('cute', 'high quality', 'detailed').\n"
    "- Skip art-style words ('anime', 'illustration', 'masterpiece') — they're "
    "not trigger material.\n"
    "- Skip rating tags (safe / nsfw / etc).\n"
    "- If the image has a clear named character or franchise, lead with that.\n"
    "\n"
    "Output JSON only, no surrounding prose: "
    '{"triggers": ["phrase one", "phrase two"]}'
)


class TriggerWordsBatchInput(BaseModel):
    path: str
    recursive: bool = False
    task: str = "trigger.words"
    # Skip images that already have a trigger word suggestion stored.
    skipAnalyzed: bool = True


def _parse_trigger_words(raw: str) -> list[str]:
    """Best-effort parse of the VLM response into a clean trigger list.

    Accepts either the JSON-only output the prompt asks for or a fallback
    comma-separated string the model might emit when it ignores the JSON
    instruction. Always returns at most 3 entries, deduped, lowercased.
    """
    import json as _json  # noqa: PLC0415
    import re as _re  # noqa: PLC0415

    text = raw.strip()
    triggers: list[str] = []
    # Most VLMs honour the JSON-only request, sometimes wrapping in ```json
    # code fences. Strip those before parsing.
    fenced = _re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    if fenced:
        text = fenced.group(1).strip()
    try:
        data = _json.loads(text)
        candidate = data.get("triggers") if isinstance(data, dict) else None
        if isinstance(candidate, list):
            triggers = [str(t) for t in candidate]
    except (_json.JSONDecodeError, AttributeError, TypeError):
        # Fallback: comma / newline split.
        parts = [p.strip().strip("\"'") for p in _re.split(r"[,\n]", text)]
        triggers = [p for p in parts if p]

    seen: set[str] = set()
    cleaned: list[str] = []
    for t in triggers:
        norm = t.strip().lower()
        # Drop punctuation-only or empty tokens, cap at 3 words, skip dups.
        if not norm or not _re.search(r"[a-z]", norm):
            continue
        words = norm.split()
        if len(words) > 3:
            norm = " ".join(words[:3])
        if norm in seen:
            continue
        seen.add(norm)
        cleaned.append(norm)
        if len(cleaned) >= 3:
            break
    return cleaned


@dataclass
class _TriggerWordsSession:
    session_id: str
    path: str
    total: int
    skipped: int
    status: str = "running"
    processed: int = 0
    results: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)
    dataset_top: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    last_image: str = ""
    started_at: float = field(default_factory=_time.time)
    finished_at: float | None = None
    _stop_flag: bool = field(default=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def percent(self) -> float:
        with self._lock:
            return 100.0 * self.processed / self.total if self.total > 0 else 100.0

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "session_id": self.session_id,
                "path": self.path,
                "status": self.status,
                "processed": self.processed,
                "total": self.total,
                "skipped": self.skipped,
                "percent": (
                    100.0 * self.processed / self.total
                    if self.total > 0
                    else 100.0
                ),
                "last_image": self.last_image,
                "results": list(self.results),
                "errors": list(self.errors),
                "dataset_top": list(self.dataset_top),
                "error": self.error,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
            }

    def add_result(self, item: dict[str, Any], image_name: str) -> None:
        with self._lock:
            self.results.append(item)
            self.processed += 1
            self.last_image = image_name
            processed = self.processed
            percent = 100.0 * processed / self.total if self.total > 0 else 100.0
        self._append_task_event(
            f"analyzed {image_name}",
            percent=percent,
            payload={"image": image_name, "processed": processed, "item": item},
        )

    def add_error(self, path: str, msg: str, image_name: str) -> None:
        with self._lock:
            self.errors.append({"path": path, "error": msg})
            self.processed += 1
            self.last_image = image_name
            processed = self.processed
            percent = 100.0 * processed / self.total if self.total > 0 else 100.0
        self._append_task_event(
            f"failed {image_name}: {msg}",
            level="error",
            percent=percent,
            payload={
                "path": path,
                "image": image_name,
                "error": msg,
                "processed": processed,
            },
        )

    def finish(self, status: str, dataset_top: list[dict[str, Any]]) -> None:
        with self._lock:
            self.status = status
            self.dataset_top = list(dataset_top)
            self.finished_at = _time.time()
        self._append_task_event(f"finished: {status}", percent=self.percent)
        try:
            _task_store().update(
                self.session_id,
                status="succeeded" if status == "succeeded" else "canceled",
                percent=100 if status == "succeeded" else self.percent,
                result=self.snapshot(),
                error=self.error,
                finished=True,
            )
        except Exception:
            pass

    def request_stop(self) -> None:
        with self._lock:
            self._stop_flag = True
        self._append_task_event("cancel requested", level="warn", percent=self.percent)

    def should_stop(self) -> bool:
        with self._lock:
            return self._stop_flag

    def fail(self, msg: str) -> None:
        with self._lock:
            self.status = "failed"
            self.error = msg
            self.finished_at = _time.time()
        self._append_task_event(msg, level="error", percent=self.percent)
        try:
            _task_store().update(
                self.session_id,
                status="failed",
                percent=self.percent,
                result=self.snapshot(),
                error=msg,
                finished=True,
            )
        except Exception:
            pass

    def _append_task_event(
        self,
        message: str,
        *,
        level: str = "info",
        percent: float | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        try:
            _task_store().append_event(
                self.session_id,
                TaskEvent(
                    level=level,
                    message=message,
                    percent=percent,
                    payload=payload or {},
                    ts=_time.time(),
                ),
            )
        except Exception:
            pass


_trigger_words_sessions: dict[str, _TriggerWordsSession] = {}
_trigger_words_lock = threading.Lock()


def _trigger_words_images_for_request(
    body: TriggerWordsBatchInput,
    directory: Path,
) -> tuple[list[Path], int]:
    images = _scan_images(directory, body.recursive)
    skipped = 0
    if body.skipAnalyzed:
        store = _store()
        before = len(images)
        images = [
            p for p in images
            if not (
                (ann := store.get_annotation(str(p))) is not None
                and ann.ai_trigger_words is not None
                and len(ann.ai_trigger_words) > 0
            )
        ]
        skipped = before - len(images)
    return images, skipped


def _analyze_trigger_words_images(
    body: TriggerWordsBatchInput,
    directory: Path,
    images: list[Path],
    *,
    on_result: Callable[[dict[str, Any], str], None] | None = None,
    on_error: Callable[[str, str, str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, str]], list[dict[str, Any]]]:
    from collections import Counter  # noqa: PLC0415
    from datetime import UTC, datetime  # noqa: PLC0415

    from lorahub.api import app as app_mod  # noqa: PLC0415
    from lorahub.core.ai import client as ai_client  # noqa: PLC0415

    ai_store = app_mod._ai_store
    if ai_store is None:
        raise HTTPException(503, "AI store not initialised")

    route = ai_store.get_route(body.task)
    if route is None or not (route.provider_id and route.model_id):
        route = ai_store.get_route("global.default")
    if route is None or not (route.provider_id and route.model_id):
        raise HTTPException(409, f"no AI route for task {body.task!r}")

    store = _store()
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    counter: Counter[str] = Counter()

    # Pre-seed the counter with already-analysed images so the dataset_top
    # aggregation reflects the whole dataset, not just this batch.
    for p in _scan_images(directory, body.recursive):
        ann_existing = store.get_annotation(str(p))
        if ann_existing and ann_existing.ai_trigger_words:
            counter.update(ann_existing.ai_trigger_words)

    for img_path in images:
        if should_stop is not None and should_stop():
            raise InterruptedError("stopped by user")
        try:
            import base64  # noqa: PLC0415
            import mimetypes  # noqa: PLC0415

            mime = mimetypes.guess_type(str(img_path))[0] or "image/png"
            data = img_path.read_bytes()
            b64 = base64.b64encode(data).decode("ascii")
            data_url = f"data:{mime};base64,{b64}"

            messages: list[dict[str, Any]] = [
                {"role": "system", "content": route.system_prompt or _TRIGGER_WORD_PROMPT},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]},
            ]
            # If the configured route override didn't mention triggers in
            # the system prompt, re-state the JSON contract on the user
            # turn so we still get parseable output.
            if route.system_prompt and "trigger" not in route.system_prompt.lower():
                messages[1]["content"].append({"type": "text", "text": _TRIGGER_WORD_PROMPT})

            result = ai_client.invoke(
                ai_store,
                provider_id=route.provider_id,
                model_id=route.model_id,
                messages=messages,
                route=route,
            )

            triggers = _parse_trigger_words(result.content)
            if not triggers:
                # Don't store an empty list — that would mark the image
                # "analyzed but produced nothing", which the next run's
                # skipAnalyzed would then skip forever. Treat empty as
                # an error so the user can retry.
                msg = "model returned no parseable triggers"
                errors.append({"path": str(img_path), "error": msg})
                if on_error is not None:
                    on_error(str(img_path), msg, img_path.name)
                continue

            counter.update(triggers)

            ann = store.get_annotation(str(img_path))
            if ann is None:
                ann = ImageAnnotation(
                    image_path=str(img_path),
                    sha256=_file_sha256(img_path),
                )
            ann.ai_trigger_words = triggers
            ann.ai_trigger_words_at = datetime.now(UTC).isoformat()
            store.upsert_annotation(ann)

            item = {"path": str(img_path), "triggers": triggers}
            results.append(item)
            if on_result is not None:
                on_result(item, img_path.name)
        except Exception as exc:  # noqa: BLE001
            errors.append({"path": str(img_path), "error": str(exc)})
            if on_error is not None:
                on_error(str(img_path), str(exc), img_path.name)

    # Top-N most common across the dataset. 8 is a sensible upper bound
    # for a "pick your trigger word" picker — beyond that the tail is
    # just noise.
    dataset_top = [
        {"trigger": t, "count": c}
        for t, c in counter.most_common(8)
    ]

    return results, errors, dataset_top


@router.post("/ai/trigger-words")
def ai_batch_trigger_words(body: TriggerWordsBatchInput) -> dict[str, Any]:
    """Suggest 1-3 LoRA-trigger-word candidates per image, then aggregate."""
    directory = _resolve_under_roots(body.path)
    if not directory.is_dir():
        raise HTTPException(400, "not a directory")
    images, skipped = _trigger_words_images_for_request(body, directory)
    results, errors, dataset_top = _analyze_trigger_words_images(
        body,
        directory,
        images,
    )
    return {
        "processed": len(results),
        "skipped": skipped,
        "results": results,
        "errors": errors,
        "dataset_top": dataset_top,
    }


@router.post("/ai/trigger-words/start", status_code=202)
def ai_batch_trigger_words_start(body: TriggerWordsBatchInput) -> dict[str, Any]:
    """Start a persistent background trigger-word analysis session."""
    directory = _resolve_under_roots(body.path)
    if not directory.is_dir():
        raise HTTPException(400, "not a directory")
    # Validate route before returning 202 so configuration errors are immediate.
    from lorahub.api import app as app_mod  # noqa: PLC0415

    ai_store = app_mod._ai_store
    if ai_store is None:
        raise HTTPException(503, "AI store not initialised")
    route = ai_store.get_route(body.task)
    if route is None or not (route.provider_id and route.model_id):
        route = ai_store.get_route("global.default")
    if route is None or not (route.provider_id and route.model_id):
        raise HTTPException(409, f"no AI route for task {body.task!r}")

    images, skipped = _trigger_words_images_for_request(body, directory)
    task = _task_store().create(
        kind=_KIND_TRIGGER_WORDS,
        title=f"trigger-words:{directory.name}",
        metadata={
            "path": str(directory),
            "recursive": body.recursive,
            "task": body.task,
            "skipAnalyzed": body.skipAnalyzed,
            "skipped": skipped,
        },
    )
    session = _TriggerWordsSession(
        session_id=task.id,
        path=str(directory),
        total=len(images),
        skipped=skipped,
    )
    session._append_task_event("trigger-word analysis queued", percent=0)
    with _trigger_words_lock:
        _trigger_words_sessions[session.session_id] = session

    def run() -> None:
        try:
            _task_store().update(session.session_id, status="running", percent=0)
            _results, _errors, dataset_top = _analyze_trigger_words_images(
                body,
                directory,
                images,
                on_result=session.add_result,
                on_error=session.add_error,
                should_stop=session.should_stop,
            )
            session.finish(
                "canceled" if session.should_stop() else "succeeded",
                dataset_top,
            )
        except InterruptedError:
            session.finish("canceled", [])
        except Exception as exc:  # noqa: BLE001
            session.fail(str(exc))

    threading.Thread(
        target=run,
        name=f"trigger-words-{session.session_id[:8]}",
        daemon=True,
    ).start()
    return {
        "session_id": session.session_id,
        "total": len(images),
        "skipped": skipped,
        "status_url": f"/api/image-studio/ai/trigger-words/status/{session.session_id}",
    }


@router.get("/ai/trigger-words/status/{session_id}")
def ai_batch_trigger_words_status(session_id: str) -> dict[str, Any]:
    with _trigger_words_lock:
        session = _trigger_words_sessions.get(session_id)
    if session is not None:
        return session.snapshot()
    persisted = _persisted_task_result(session_id, _KIND_TRIGGER_WORDS)
    if persisted is not None:
        return persisted
    raise HTTPException(404, "trigger-words session not found")


@router.post("/ai/trigger-words/cancel/{session_id}")
def ai_batch_trigger_words_cancel(session_id: str) -> dict[str, Any]:
    with _trigger_words_lock:
        session = _trigger_words_sessions.get(session_id)
    if session is None:
        raise HTTPException(404, "trigger-words session not found")
    session.request_stop()
    return {"session_id": session_id, "status": "stop_requested"}
