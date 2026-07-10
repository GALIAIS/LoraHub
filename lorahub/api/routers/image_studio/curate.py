"""Dataset curation operations — batch image edits driven by the audit
report or explicit path lists.

Five capabilities:

  - POST /curate/auto-rotate     EXIF orientation → bake into pixels
  - POST /curate/quarantine       move file + caption to .workbench/quarantine/
  - POST /curate/restore          move back from quarantine
  - POST /curate/batch-resize     Lanczos resample so short edge ≥ N
  - POST /curate/batch-by-issue   take audit issue kind + action, applies in bulk

Every destructive op writes a backup to ``<dataset>/.workbench/backups/<rel>``
*before* mutating the source file. ``GET /curate/backups`` lists what's
restorable; ``POST /curate/restore-backup`` rolls a single file back.

No GPU required. Pillow + numpy already in the project.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import tempfile
import threading
import time as _time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field

from lorahub.api import app as app_module
from lorahub.api.dataset_files import (
    IMAGE_SUFFIXES,
    is_link_like,
    iter_safe_files,
    resolve_dataset_directory,
    resolve_file_under,
)
from lorahub.api.task_sessions import (
    TaskEvent,
    TaskSession,
    TaskSessionStore,
    default_task_store_path,
    persist_stop_request,
    prune_terminal_session_cache,
)

from ._shared import (
    _atomic_save_image,
    _clear_dataset_view_caches,
    _file_mutation,
)

router = APIRouter(prefix="/api/image-studio", tags=["image-studio"])
_KIND_AUTO_ROTATE = "image_studio_auto_rotate"
_KIND_BATCH_RESIZE = "image_studio_batch_resize"


def _ensure_dataset(raw: str) -> Path:
    try:
        return resolve_dataset_directory(raw)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


def _task_store() -> TaskSessionStore:
    store = app_module._task_session_store
    if store is None:
        store = TaskSessionStore(default_task_store_path())
        app_module._task_session_store = store
    return store


def _persisted_task_result(session_id: str, kind: str) -> dict[str, Any] | None:
    try:
        task = _task_store().get(session_id)
    except Exception:
        return None
    if task is None or task.kind != kind:
        return None
    return _task_to_status_snapshot(task)


def _task_to_status_snapshot(task: TaskSession) -> dict[str, Any] | None:
    if isinstance(task.result, dict):
        result = dict(task.result)
        result.setdefault("events", [event.to_dict() for event in task.events])
        return result
    if task.status in {
        "queued",
        "running",
        "stop_requested",
        "interrupted",
        "failed",
        "canceled",
    }:
        metadata = task.metadata
        return {
            "session_id": task.id,
            "dataset_path": str(metadata.get("dataset_path") or metadata.get("path") or ""),
            "status": task.status,
            "processed": 0,
            "total": int(metadata.get("total") or metadata.get("selected") or 0),
            "percent": task.percent,
            "last_image": "",
            "rotated": [],
            "rotated_count": 0,
            "resampled": [],
            "resampled_count": 0,
            "skipped_count": int(metadata.get("skipped") or 0),
            "failed": [],
            "error": task.error,
            "started_at": task.started_at,
            "finished_at": task.finished_at,
            "events": [event.to_dict() for event in task.events],
        }
    return None


# --------------------------------------------------------------------------- #
# Path conventions
# --------------------------------------------------------------------------- #


def _managed_directory(root: Path, relative: Path, *, create: bool) -> Path:
    """Resolve a managed directory without traversing links or junctions."""
    current = root.resolve()
    for part in relative.parts:
        current = current / part
        if is_link_like(current):
            raise HTTPException(400, f"managed path cannot be a link: {current}")
        if current.exists():
            if not current.is_dir():
                raise HTTPException(400, f"managed path is not a directory: {current}")
        elif create:
            current.mkdir()
        else:
            continue
        try:
            current.resolve().relative_to(root.resolve())
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(400, "managed path escaped the dataset") from exc
    return current.resolve() if current.exists() else current


def _workbench_root(dataset_path: str, *, create: bool = False) -> Path:
    return _managed_directory(
        _ensure_dataset(dataset_path),
        Path(".workbench"),
        create=create,
    )


def _quarantine_root(dataset_path: str, *, create: bool = False) -> Path:
    return _managed_directory(
        _ensure_dataset(dataset_path),
        Path(".workbench") / "quarantine",
        create=create,
    )


def _backups_root(dataset_path: str, *, create: bool = False) -> Path:
    return _managed_directory(
        _ensure_dataset(dataset_path),
        Path(".workbench") / "backups",
        create=create,
    )


def _audit_cache_path(dataset_path: str) -> Path:
    return _workbench_root(dataset_path) / "audit.json"


def _resolve_under(dataset_path: str, candidate: str) -> Path:
    """Return a Path inside ``dataset_path`` only, or raise.

    Rejects paths that escape the dataset via ``..`` / absolute paths to
    elsewhere. The image studio routes everywhere take absolute paths
    that originated in our own listings response, so we know they're
    inside the dataset; this is belt-and-suspenders.
    """
    root = _ensure_dataset(dataset_path).resolve()
    raw = Path(candidate).expanduser()
    if not raw.is_absolute():
        raw = root / raw
    lexical = Path(os.path.abspath(raw))
    try:
        relative = lexical.relative_to(root)
    except ValueError as exc:
        raise HTTPException(400, f"path is outside dataset: {candidate}") from exc
    if not relative.parts or relative.parts[0] == ".workbench":
        raise HTTPException(400, f"path is not a dataset source file: {candidate}")
    current = root
    for part in relative.parts:
        current /= part
        if is_link_like(current):
            raise HTTPException(400, f"path cannot traverse a link: {candidate}")
    try:
        resolved = lexical.resolve()
        resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(400, f"path is outside dataset: {candidate}") from exc
    return resolved


def _relative_under(root: Path, p: Path) -> Path:
    return p.resolve().relative_to(root.resolve())


# --------------------------------------------------------------------------- #
# Backup helper
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _BackupResult:
    backup_path: Path


def _copy2_atomic(src: Path, dst: Path) -> None:
    """Copy a file into place without exposing a partially written backup."""
    with tempfile.NamedTemporaryFile(
        delete=False,
        dir=dst.parent,
        prefix=f".{dst.name}.",
        suffix=".tmp",
    ) as handle:
        temp_path = Path(handle.name)
    try:
        shutil.copy2(src, temp_path)
        temp_path.replace(dst)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _backup_file(dataset_path: str, src: Path) -> _BackupResult | None:
    """Copy ``src`` (and its .txt sidecar) to .workbench/backups/<rel>.

    Returns the backup path, or ``None`` when src no longer exists. The
    sidecar is best-effort — if the .txt is gone the image still gets
    backed up.
    """
    if not src.is_file():
        return None
    root = _ensure_dataset(dataset_path)
    rel = _relative_under(root, src)
    backups = _backups_root(dataset_path, create=True)
    dst_parent = _managed_directory(backups, rel.parent, create=True)
    dst = dst_parent / rel.name
    # Don't overwrite — keep the *first* backup before any chain of
    # edits. ``restore-backup`` rolls back to that pristine version.
    if not dst.exists():
        _copy2_atomic(src, dst)
    cap_src = resolve_file_under(root, src.with_suffix(".txt"))
    cap_dst = dst.with_suffix(".txt")
    if cap_src is not None and not cap_dst.exists():
        _copy2_atomic(cap_src, cap_dst)
    return _BackupResult(backup_path=dst)


# --------------------------------------------------------------------------- #
# Auto-rotate — bake EXIF orientation into pixels
# --------------------------------------------------------------------------- #


class AutoRotateRequest(BaseModel):
    dataset_path: str
    paths: list[str] | None = Field(
        default=None,
        description="Specific files to rotate. None = scan dataset for "
        "any image whose EXIF orientation isn't 1.",
    )
    recursive: bool = True


@dataclass
class _AutoRotateSession:
    session_id: str
    dataset_path: str
    total: int
    status: str = "running"
    processed: int = 0
    rotated: list[str] = field(default_factory=list)
    failed: list[dict[str, str]] = field(default_factory=list)
    skipped_count: int = 0
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
                "dataset_path": self.dataset_path,
                "status": self.status,
                "processed": self.processed,
                "total": self.total,
                "percent": (
                    100.0 * self.processed / self.total
                    if self.total > 0
                    else 100.0
                ),
                "last_image": self.last_image,
                "rotated": list(self.rotated),
                "rotated_count": len(self.rotated),
                "skipped_count": self.skipped_count,
                "failed": list(self.failed),
                "error": self.error,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
            }

    def add_rotated(self, path: str, image_name: str) -> None:
        with self._lock:
            self.rotated.append(path)
            self.processed += 1
            self.last_image = image_name
            processed = self.processed
            percent = 100.0 * processed / self.total if self.total > 0 else 100.0
        self._append_task_event(
            f"rotated {image_name}",
            percent=percent,
            payload={"path": path, "image": image_name, "processed": processed},
        )

    def add_skipped(self, image_name: str) -> None:
        with self._lock:
            self.skipped_count += 1
            self.processed += 1
            self.last_image = image_name
            processed = self.processed
            percent = 100.0 * processed / self.total if self.total > 0 else 100.0
        self._append_task_event(
            f"skipped {image_name}",
            percent=percent,
            payload={"image": image_name, "processed": processed},
        )

    def add_failed(self, path: str, msg: str, image_name: str) -> None:
        with self._lock:
            self.failed.append({"path": path, "error": msg})
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
            if status == "succeeded" and self._stop_flag:
                status = "canceled"
            self.status = status
            self.finished_at = _time.time()
        self._append_task_event(f"finished: {status}", percent=self.percent)
        task_status = (
            "succeeded"
            if status == "succeeded"
            else "canceled"
            if status == "canceled"
            else "failed"
        )
        try:
            _task_store().update(
                self.session_id,
                status=task_status,
                percent=100 if status == "succeeded" else self.percent,
                result=self.snapshot(),
                error=self.error,
                finished=True,
            )
        except Exception:
            pass

    def request_stop(self) -> bool:
        with self._lock:
            if self.status != "running":
                return False
            percent = 100.0 * self.processed / self.total if self.total > 0 else 100.0
            persisted = persist_stop_request(
                _task_store(),
                self.session_id,
                percent=percent,
            )
            if not persisted:
                return False
            self._stop_flag = True
            self.status = "stop_requested"
        self._append_task_event("stop requested", level="warn", percent=percent)
        return True

    def should_stop(self) -> bool:
        with self._lock:
            return self._stop_flag

    def cancel(self, msg: str = "stopped by user") -> None:
        with self._lock:
            self.status = "canceled"
            self.error = msg
            self.finished_at = _time.time()
        self._append_task_event(msg, level="warn", percent=self.percent)
        try:
            _task_store().update(
                self.session_id,
                status="canceled",
                percent=self.percent,
                result=self.snapshot(),
                error=msg,
                finished=True,
            )
        except Exception:
            pass

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


_auto_rotate_sessions: dict[str, _AutoRotateSession] = {}
_auto_rotate_lock = threading.Lock()


def _auto_rotate_targets(req: AutoRotateRequest, root: Path) -> list[Path]:
    if req.paths:
        return [_resolve_under(req.dataset_path, p) for p in req.paths]
    return list(_walk_images(root, req.recursive))


def _auto_rotate_images(
    req: AutoRotateRequest,
    targets: list[Path],
    *,
    on_rotated: Callable[[str, str], None] | None = None,
    on_skipped: Callable[[str], None] | None = None,
    on_failed: Callable[[str, str, str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Apply EXIF orientation, write pixels back, strip the EXIF tag.

    Idempotent: a file whose orientation is already 1 (or missing)
    is left untouched.
    """
    root = _ensure_dataset(req.dataset_path)
    if not root.is_dir():
        raise HTTPException(404, f"dataset not found: {root}")

    rotated: list[str] = []
    skipped: list[str] = []
    failed: list[dict[str, str]] = []

    for src in targets:
        if should_stop is not None and should_stop():
            raise InterruptedError("stopped by user")
        try:
            with _file_mutation(src):
                with Image.open(src) as img:
                    image_format = img.format
                    exif = img.getexif()
                    orientation = exif.get(0x0112) if exif else None
                    if not orientation or orientation == 1:
                        skipped.append(str(src))
                        if on_skipped is not None:
                            on_skipped(src.name)
                        continue
                    # ``exif_transpose`` reads the orientation tag and
                    # returns a pixel-rotated copy with the tag dropped.
                    rotated_img = ImageOps.exif_transpose(img)
                    if rotated_img is None:
                        skipped.append(str(src))
                        if on_skipped is not None:
                            on_skipped(src.name)
                        continue
                    rotated_img.load()
                _backup_file(req.dataset_path, src)
                _atomic_save_image(
                    rotated_img,
                    src,
                    image_format=image_format,
                    exif=b"",
                )
            path = str(src)
            rotated.append(path)
            if on_rotated is not None:
                on_rotated(path, src.name)
        except (UnidentifiedImageError, OSError) as exc:
            failed.append({"path": str(src), "error": str(exc)})
            if on_failed is not None:
                on_failed(str(src), str(exc), src.name)

    return {
        "rotated": rotated,
        "rotated_count": len(rotated),
        "skipped_count": len(skipped),
        "failed": failed,
    }


@router.post("/curate/auto-rotate")
def curate_auto_rotate(req: AutoRotateRequest) -> dict[str, Any]:
    """Apply EXIF orientation synchronously for legacy callers."""
    root = _ensure_dataset(req.dataset_path)
    if not root.is_dir():
        raise HTTPException(404, f"dataset not found: {root}")
    result = _auto_rotate_images(req, _auto_rotate_targets(req, root))
    if result.get("rotated_count"):
        _clear_dataset_view_caches(root)
    return result


@router.post("/curate/auto-rotate/start", status_code=202)
def curate_auto_rotate_start(req: AutoRotateRequest) -> dict[str, Any]:
    """Start a persistent background auto-rotate session."""
    root = _ensure_dataset(req.dataset_path)
    if not root.is_dir():
        raise HTTPException(404, f"dataset not found: {root}")
    targets = _auto_rotate_targets(req, root)
    task = _task_store().create(
        kind=_KIND_AUTO_ROTATE,
        title=f"auto-rotate:{root.name}",
        metadata={
            "dataset_path": str(root),
            "paths_count": len(req.paths or []),
            "recursive": req.recursive,
        },
    )
    session = _AutoRotateSession(
        session_id=task.id,
        dataset_path=str(root),
        total=len(targets),
    )
    session._append_task_event("auto rotate queued", percent=0)
    with _auto_rotate_lock:
        _auto_rotate_sessions[session.session_id] = session
        prune_terminal_session_cache(_auto_rotate_sessions)

    def run() -> None:
        try:
            _task_store().update(session.session_id, status="running", percent=0)
            _auto_rotate_images(
                req,
                targets,
                on_rotated=session.add_rotated,
                on_skipped=session.add_skipped,
                on_failed=session.add_failed,
                should_stop=session.should_stop,
            )
            _clear_dataset_view_caches(root)
            session.finish("canceled" if session.should_stop() else "succeeded")
        except InterruptedError:
            session.cancel()
        except Exception as exc:  # noqa: BLE001
            session.fail(str(exc))

    threading.Thread(
        target=run,
        name=f"is-auto-rotate-{session.session_id[:8]}",
        daemon=True,
    ).start()
    return {
        "session_id": session.session_id,
        "total": len(targets),
        "status_url": (
            f"/api/image-studio/curate/auto-rotate/status/{session.session_id}"
        ),
    }


@router.get("/curate/auto-rotate/status/{session_id}")
def curate_auto_rotate_status(session_id: str) -> dict[str, Any]:
    with _auto_rotate_lock:
        session = _auto_rotate_sessions.get(session_id)
    if session is not None:
        return session.snapshot()
    persisted = _persisted_task_result(session_id, _KIND_AUTO_ROTATE)
    if persisted is not None:
        return persisted
    raise HTTPException(404, "auto-rotate session not found")


@router.post("/curate/auto-rotate/stop/{session_id}")
def curate_auto_rotate_stop(session_id: str) -> dict[str, Any]:
    with _auto_rotate_lock:
        session = _auto_rotate_sessions.get(session_id)
    if session is None:
        raise HTTPException(404, "auto-rotate session not found")
    if not session.request_stop():
        raise HTTPException(409, f"auto-rotate session is {session.status}")
    return {"session_id": session_id, "status": "stop_requested"}


# --------------------------------------------------------------------------- #
# Quarantine — soft delete
# --------------------------------------------------------------------------- #


class QuarantineRequest(BaseModel):
    dataset_path: str
    paths: list[str]
    # Free-form note shown in the UI's quarantine list. Useful when
    # the user moves images for a specific reason ("blurry batch
    # 2026-05-22") and wants to remember why later.
    reason: str | None = None


_quarantine_index_lock = threading.RLock()


def _path_occupied(path: Path) -> bool:
    return path.exists() or is_link_like(path)


def _available_destination(path: Path, *, with_caption: bool) -> Path:
    """Choose a free image name without overwriting its caption pair."""
    index = 1
    while True:
        candidate = (
            path
            if index == 1
            else path.with_name(f"{path.stem}-{index}{path.suffix}")
        )
        if not _path_occupied(candidate) and not (
            with_caption and _path_occupied(candidate.with_suffix(".txt"))
        ):
            return candidate
        index += 1


def _load_quarantine_index(index_path: Path) -> list[dict[str, Any]]:
    if is_link_like(index_path):
        raise HTTPException(400, "quarantine index cannot be a link")
    if not index_path.is_file():
        return []
    entries: list[dict[str, Any]] = []
    with index_path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(value, dict):
                continue
            if not all(
                isinstance(value.get(key), str) and value[key]
                for key in ("original_path", "quarantine_path")
            ):
                continue
            caption_path = value.get("caption_quarantine_path")
            if caption_path is not None and not isinstance(caption_path, str):
                continue
            entries.append(value)
    return entries


def _write_quarantine_index(
    index_path: Path,
    entries: list[dict[str, Any]],
) -> None:
    if is_link_like(index_path):
        raise OSError("quarantine index cannot be a link")
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        dir=index_path.parent,
        prefix=".quarantine-index-",
        suffix=".tmp",
    ) as handle:
        temp_path = Path(handle.name)
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        temp_path.replace(index_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _rollback_moves(moves: list[tuple[Path, Path]]) -> list[str]:
    errors: list[str] = []
    for current, original in reversed(moves):
        try:
            if not _path_occupied(current):
                continue
            if _path_occupied(original):
                raise OSError(f"rollback target already exists: {original}")
            original.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(current), str(original))
        except OSError as exc:
            errors.append(str(exc))
    return errors


def _restore_target(root: Path, raw: str) -> Path:
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    lexical = Path(os.path.abspath(candidate))
    try:
        relative = lexical.relative_to(root.resolve())
    except ValueError as exc:
        raise HTTPException(400, "restore target is outside dataset") from exc
    if not relative.parts or relative.parts[0] == ".workbench":
        raise HTTPException(400, "restore target is not a dataset path")
    parent = _managed_directory(root, relative.parent, create=True)
    target = parent / relative.name
    if is_link_like(target):
        raise HTTPException(400, "restore target cannot be a link")
    return target


@router.post("/curate/quarantine")
def curate_quarantine(req: QuarantineRequest) -> dict[str, Any]:
    """Move files (and their .txt sidecars) to ``.workbench/quarantine/``.

    Uses ``shutil.move`` so cross-device renames degrade to copy + remove
    transparently. Each path is tracked in
    ``.workbench/quarantine/index.jsonl`` so the restore endpoint
    knows the original location.
    """
    root = _ensure_dataset(req.dataset_path)
    if not root.is_dir():
        raise HTTPException(404, f"dataset not found: {root}")

    qroot = _quarantine_root(req.dataset_path, create=True)
    index_path = qroot / "index.jsonl"

    moved: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    timestamp = datetime.now(UTC).replace(microsecond=0).isoformat()
    with _quarantine_index_lock:
        entries = _load_quarantine_index(index_path)
        for raw in req.paths:
            entry: dict[str, Any] | None = None
            completed_moves: list[tuple[Path, Path]] = []
            try:
                resolved = _resolve_under(req.dataset_path, raw)
                with _file_mutation(resolved):
                    src = resolve_file_under(root, resolved)
                    if src is None:
                        failed.append(
                            {"path": str(raw), "error": "not a regular file"}
                        )
                        continue
                    rel = _relative_under(root, src)
                    destination_parent = _managed_directory(
                        qroot,
                        rel.parent,
                        create=True,
                    )
                    cap_src = resolve_file_under(root, src.with_suffix(".txt"))
                    dst = _available_destination(
                        destination_parent / rel.name,
                        with_caption=cap_src is not None,
                    )
                    cap_dst = dst.with_suffix(".txt") if cap_src is not None else None
                    entry = {
                        "moved_at": timestamp,
                        "original_path": str(src),
                        "quarantine_path": str(dst),
                        "caption_quarantine_path": str(cap_dst) if cap_dst else None,
                        "reason": req.reason,
                        "state": "moving",
                    }
                    entries.append(entry)
                    _write_quarantine_index(index_path, entries)

                    shutil.move(str(src), str(dst))
                    completed_moves.append((dst, src))
                    if cap_src is not None and cap_dst is not None:
                        shutil.move(str(cap_src), str(cap_dst))
                        completed_moves.append((cap_dst, cap_src))

                    entry["state"] = "quarantined"
                    _write_quarantine_index(index_path, entries)
                    moved.append(dict(entry))
            except HTTPException:
                raise
            except OSError as exc:
                rollback_errors = _rollback_moves(completed_moves)
                if entry is not None:
                    if rollback_errors:
                        entry["state"] = "recovery_required"
                        entry["error"] = "; ".join(rollback_errors)
                    else:
                        with contextlib.suppress(ValueError):
                            entries.remove(entry)
                    with contextlib.suppress(OSError):
                        _write_quarantine_index(index_path, entries)
                message = str(exc)
                if rollback_errors:
                    message += f"; rollback failed: {'; '.join(rollback_errors)}"
                failed.append({"path": raw, "error": message})

    if moved:
        _clear_dataset_view_caches(root)
    return {
        "moved": moved,
        "moved_count": len(moved),
        "failed": failed,
    }


@router.get("/curate/quarantine")
def curate_quarantine_list(dataset_path: str) -> dict[str, Any]:
    """Return the quarantine index — what was moved, when, why."""
    qroot = _quarantine_root(dataset_path)
    index_path = qroot / "index.jsonl"
    with _quarantine_index_lock:
        entries = _load_quarantine_index(index_path)
    return {"entries": entries}


class RestoreRequest(BaseModel):
    dataset_path: str
    quarantine_paths: list[str]


@router.post("/curate/restore-quarantine")
def curate_restore_quarantine(req: RestoreRequest) -> dict[str, Any]:
    """Move quarantined files back to their original locations.

    The original path is read from index.jsonl. After a successful
    move, the index entry is rewritten with ``"restored_at": ...`` so
    the audit trail survives but the entry stops counting as "still
    quarantined".
    """
    root = _ensure_dataset(req.dataset_path)
    qroot = _quarantine_root(req.dataset_path)
    index_path = qroot / "index.jsonl"
    if not index_path.is_file():
        raise HTTPException(404, "no quarantine index found for this dataset")

    restored: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    timestamp = datetime.now(UTC).replace(microsecond=0).isoformat()
    target_set = set(req.quarantine_paths)

    with _quarantine_index_lock:
        entries = _load_quarantine_index(index_path)
        for entry in entries:
            quarantine_path = str(entry.get("quarantine_path") or "")
            if entry.get("restored_at") or quarantine_path not in target_set:
                continue
            completed_moves: list[tuple[Path, Path]] = []
            try:
                src = resolve_file_under(qroot, Path(quarantine_path))
                if src is None:
                    failed.append(
                        {"path": quarantine_path, "error": "quarantined file missing"},
                    )
                    continue
                original_path = str(entry.get("original_path") or "")
                dst_base = _restore_target(root, original_path)
                with _file_mutation(dst_base):
                    caption_value = entry.get("caption_quarantine_path")
                    cap_q_path = (
                        resolve_file_under(qroot, Path(caption_value))
                        if isinstance(caption_value, str) and caption_value
                        else None
                    )
                    if caption_value and cap_q_path is None:
                        failed.append(
                            {
                                "path": str(caption_value),
                                "error": "caption is outside quarantine or missing",
                            },
                        )
                        continue
                    dst = _available_destination(
                        dst_base,
                        with_caption=cap_q_path is not None,
                    )
                    cap_dst = (
                        dst.with_suffix(".txt") if cap_q_path is not None else None
                    )
                    entry["state"] = "restoring"
                    entry["restore_target"] = str(dst)
                    _write_quarantine_index(index_path, entries)

                    shutil.move(str(src), str(dst))
                    completed_moves.append((dst, src))
                    if cap_q_path is not None and cap_dst is not None:
                        shutil.move(str(cap_q_path), str(cap_dst))
                        completed_moves.append((cap_dst, cap_q_path))

                    entry["state"] = "restored"
                    entry["restored_at"] = timestamp
                    entry["restored_path"] = str(dst)
                    entry.pop("restore_target", None)
                    _write_quarantine_index(index_path, entries)
                    restored.append(dict(entry))
            except HTTPException as exc:
                failed.append({"path": quarantine_path, "error": str(exc.detail)})
            except OSError as exc:
                rollback_errors = _rollback_moves(completed_moves)
                entry["state"] = (
                    "recovery_required" if rollback_errors else "quarantined"
                )
                entry.pop("restore_target", None)
                if rollback_errors:
                    entry["error"] = "; ".join(rollback_errors)
                with contextlib.suppress(OSError):
                    _write_quarantine_index(index_path, entries)
                message = str(exc)
                if rollback_errors:
                    message += f"; rollback failed: {'; '.join(rollback_errors)}"
                failed.append({"path": quarantine_path, "error": message})

    if restored:
        _clear_dataset_view_caches(root)
    return {
        "restored": restored,
        "restored_count": len(restored),
        "failed": failed,
    }


# --------------------------------------------------------------------------- #
# Batch resize
# --------------------------------------------------------------------------- #


class BatchResizeRequest(BaseModel):
    dataset_path: str
    paths: list[str] | None = Field(
        default=None,
        description="Specific files. None = every image whose short edge < target.",
    )
    target_short_edge: int = Field(default=768, ge=128, le=4096)
    # Lanczos is the highest-quality classical filter for downscale +
    # decent for upscale; our default. Bilinear / bicubic available
    # for users who want speed over quality.
    filter: Literal["lanczos", "bicubic", "bilinear"] = "lanczos"
    # If True, scale up under-sized images. Default False — most LoRA
    # workflows prefer dropping tiny images over making them up.
    upscale: bool = False
    recursive: bool = True


_PIL_RESAMPLE = {
    "lanczos": Image.Resampling.LANCZOS,
    "bicubic": Image.Resampling.BICUBIC,
    "bilinear": Image.Resampling.BILINEAR,
}


@dataclass
class _BatchResizeSession:
    session_id: str
    dataset_path: str
    total: int
    status: str = "running"
    processed: int = 0
    resampled: list[dict[str, Any]] = field(default_factory=list)
    failed: list[dict[str, str]] = field(default_factory=list)
    skipped_count: int = 0
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
                "dataset_path": self.dataset_path,
                "status": self.status,
                "processed": self.processed,
                "total": self.total,
                "percent": (
                    100.0 * self.processed / self.total
                    if self.total > 0
                    else 100.0
                ),
                "last_image": self.last_image,
                "resampled": list(self.resampled),
                "resampled_count": len(self.resampled),
                "skipped_count": self.skipped_count,
                "failed": list(self.failed),
                "error": self.error,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
            }

    def add_resampled(self, item: dict[str, Any], image_name: str) -> None:
        with self._lock:
            self.resampled.append(item)
            self.processed += 1
            self.last_image = image_name
            processed = self.processed
            percent = 100.0 * processed / self.total if self.total > 0 else 100.0
        self._append_task_event(
            f"resized {image_name}",
            percent=percent,
            payload={"image": image_name, "processed": processed, "item": item},
        )

    def add_skipped(self, image_name: str) -> None:
        with self._lock:
            self.skipped_count += 1
            self.processed += 1
            self.last_image = image_name
            processed = self.processed
            percent = 100.0 * processed / self.total if self.total > 0 else 100.0
        self._append_task_event(
            f"skipped {image_name}",
            percent=percent,
            payload={"image": image_name, "processed": processed},
        )

    def add_failed(self, path: str, msg: str, image_name: str) -> None:
        with self._lock:
            self.failed.append({"path": path, "error": msg})
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
            if status == "succeeded" and self._stop_flag:
                status = "canceled"
            self.status = status
            self.finished_at = _time.time()
        self._append_task_event(f"finished: {status}", percent=self.percent)
        task_status = (
            "succeeded"
            if status == "succeeded"
            else "canceled"
            if status == "canceled"
            else "failed"
        )
        try:
            _task_store().update(
                self.session_id,
                status=task_status,
                percent=100 if status == "succeeded" else self.percent,
                result=self.snapshot(),
                error=self.error,
                finished=True,
            )
        except Exception:
            pass

    def request_stop(self) -> bool:
        with self._lock:
            if self.status != "running":
                return False
            percent = 100.0 * self.processed / self.total if self.total > 0 else 100.0
            persisted = persist_stop_request(
                _task_store(),
                self.session_id,
                percent=percent,
            )
            if not persisted:
                return False
            self._stop_flag = True
            self.status = "stop_requested"
        self._append_task_event("stop requested", level="warn", percent=percent)
        return True

    def should_stop(self) -> bool:
        with self._lock:
            return self._stop_flag

    def cancel(self, msg: str = "stopped by user") -> None:
        with self._lock:
            self.status = "canceled"
            self.error = msg
            self.finished_at = _time.time()
        self._append_task_event(msg, level="warn", percent=self.percent)
        try:
            _task_store().update(
                self.session_id,
                status="canceled",
                percent=self.percent,
                result=self.snapshot(),
                error=msg,
                finished=True,
            )
        except Exception:
            pass

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


_batch_resize_sessions: dict[str, _BatchResizeSession] = {}
_batch_resize_lock = threading.Lock()


def _batch_resize_targets(req: BatchResizeRequest, root: Path) -> list[Path]:
    if req.paths:
        return [_resolve_under(req.dataset_path, p) for p in req.paths]
    return list(_walk_images(root, req.recursive))


def _resize_images(
    req: BatchResizeRequest,
    targets: list[Path],
    *,
    on_resampled: Callable[[dict[str, Any], str], None] | None = None,
    on_skipped: Callable[[str], None] | None = None,
    on_failed: Callable[[str, str, str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    root = _ensure_dataset(req.dataset_path)
    if not root.is_dir():
        raise HTTPException(404, f"dataset not found: {root}")

    resampled: list[dict[str, Any]] = []
    skipped: list[str] = []
    failed: list[dict[str, str]] = []
    resample = _PIL_RESAMPLE[req.filter]

    for src in targets:
        if should_stop is not None and should_stop():
            raise InterruptedError("stopped by user")
        try:
            with _file_mutation(src):
                with Image.open(src) as img:
                    image_format = img.format
                    w, h = img.size
                    short = min(w, h)
                    if short == req.target_short_edge:
                        skipped.append(str(src))
                        if on_skipped is not None:
                            on_skipped(src.name)
                        continue
                    if short > req.target_short_edge:
                        # Downscale — always allowed.
                        pass
                    else:
                        # Upscale — gated by the flag.
                        if not req.upscale:
                            skipped.append(str(src))
                            if on_skipped is not None:
                                on_skipped(src.name)
                            continue
                    scale = req.target_short_edge / short
                    new_w = max(1, round(w * scale))
                    new_h = max(1, round(h * scale))
                    new_img = img.resize((new_w, new_h), resample=resample)
                    new_img.load()
                _backup_file(req.dataset_path, src)
                _atomic_save_image(new_img, src, image_format=image_format)
            item = {
                "path": str(src),
                "from": [w, h],
                "to": [new_w, new_h],
            }
            resampled.append(item)
            if on_resampled is not None:
                on_resampled(item, src.name)
        except (UnidentifiedImageError, OSError) as exc:
            failed.append({"path": str(src), "error": str(exc)})
            if on_failed is not None:
                on_failed(str(src), str(exc), src.name)

    return {
        "resampled": resampled,
        "resampled_count": len(resampled),
        "skipped_count": len(skipped),
        "failed": failed,
    }


@router.post("/curate/batch-resize")
def curate_batch_resize(req: BatchResizeRequest) -> dict[str, Any]:
    """Resample images so their short edge equals ``target_short_edge``.

    Aspect ratio is preserved; long edge scales by the same factor.
    Backups land in .workbench/backups/.
    """
    root = _ensure_dataset(req.dataset_path)
    if not root.is_dir():
        raise HTTPException(404, f"dataset not found: {root}")
    result = _resize_images(req, _batch_resize_targets(req, root))
    if result.get("resampled_count"):
        _clear_dataset_view_caches(root)
    return result


@router.post("/curate/batch-resize/start", status_code=202)
def curate_batch_resize_start(req: BatchResizeRequest) -> dict[str, Any]:
    """Start a persistent background batch-resize session."""
    root = _ensure_dataset(req.dataset_path)
    if not root.is_dir():
        raise HTTPException(404, f"dataset not found: {root}")
    targets = _batch_resize_targets(req, root)
    task = _task_store().create(
        kind=_KIND_BATCH_RESIZE,
        title=f"batch-resize:{root.name}",
        metadata={
            "dataset_path": str(root),
            "paths_count": len(req.paths or []),
            "target_short_edge": req.target_short_edge,
            "filter": req.filter,
            "upscale": req.upscale,
            "recursive": req.recursive,
        },
    )
    session = _BatchResizeSession(
        session_id=task.id,
        dataset_path=str(root),
        total=len(targets),
    )
    session._append_task_event("batch resize queued", percent=0)
    with _batch_resize_lock:
        _batch_resize_sessions[session.session_id] = session
        prune_terminal_session_cache(_batch_resize_sessions)

    def run() -> None:
        try:
            _task_store().update(session.session_id, status="running", percent=0)
            _resize_images(
                req,
                targets,
                on_resampled=session.add_resampled,
                on_skipped=session.add_skipped,
                on_failed=session.add_failed,
                should_stop=session.should_stop,
            )
            _clear_dataset_view_caches(root)
            session.finish("canceled" if session.should_stop() else "succeeded")
        except InterruptedError:
            session.cancel()
        except Exception as exc:  # noqa: BLE001
            session.fail(str(exc))

    threading.Thread(
        target=run,
        name=f"is-batch-resize-{session.session_id[:8]}",
        daemon=True,
    ).start()
    return {
        "session_id": session.session_id,
        "total": len(targets),
        "status_url": (
            f"/api/image-studio/curate/batch-resize/status/{session.session_id}"
        ),
    }


@router.get("/curate/batch-resize/status/{session_id}")
def curate_batch_resize_status(session_id: str) -> dict[str, Any]:
    with _batch_resize_lock:
        session = _batch_resize_sessions.get(session_id)
    if session is not None:
        return session.snapshot()
    persisted = _persisted_task_result(session_id, _KIND_BATCH_RESIZE)
    if persisted is not None:
        return persisted
    raise HTTPException(404, "batch-resize session not found")


@router.post("/curate/batch-resize/stop/{session_id}")
def curate_batch_resize_stop(session_id: str) -> dict[str, Any]:
    with _batch_resize_lock:
        session = _batch_resize_sessions.get(session_id)
    if session is None:
        raise HTTPException(404, "batch-resize session not found")
    if not session.request_stop():
        raise HTTPException(409, f"batch-resize session is {session.status}")
    return {"session_id": session_id, "status": "stop_requested"}


# --------------------------------------------------------------------------- #
# Batch by audit issue
# --------------------------------------------------------------------------- #


class BatchByIssueRequest(BaseModel):
    dataset_path: str
    issue_kinds: list[str] = Field(
        ...,
        description="Audit issue kinds to act on (corrupt, tiny, blurry, ...)",
    )
    action: Literal["quarantine", "delete"] = "quarantine"
    reason: str | None = None


@router.post("/curate/batch-by-issue")
def curate_batch_by_issue(req: BatchByIssueRequest) -> dict[str, Any]:
    """Apply ``action`` to every file flagged with the named audit issues.

    Reads the cached audit report; refuses if the report is missing or
    out of date (call POST /audit/scan first). ``delete`` performs the
    same move-to-trash as the existing image-studio delete op via
    quarantine + a marker entry — actual filesystem ``rm`` requires a
    separate ``apply ops`` call so deletion remains undoable.
    """
    cache = _audit_cache_path(req.dataset_path)
    if not cache.is_file():
        raise HTTPException(
            404,
            "no audit report cached; run POST /audit/scan before batch-by-issue",
        )
    try:
        report = json.loads(cache.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(500, f"corrupt audit cache: {exc}") from None

    kinds = set(req.issue_kinds)
    paths: list[str] = []
    seen: set[str] = set()
    for iss in report.get("issues", []):
        if iss.get("kind") in kinds:
            p = iss.get("path")
            if p and p not in seen:
                seen.add(p)
                paths.append(p)

    if not paths:
        return {"action": req.action, "matched_count": 0, "result": None}

    if req.action == "quarantine":
        result = curate_quarantine(
            QuarantineRequest(
                dataset_path=req.dataset_path,
                paths=paths,
                reason=req.reason or f"audit issues: {','.join(kinds)}",
            ),
        )
    else:
        # ``delete`` semantics here = quarantine with a flag. We don't
        # call rm so the operation stays reversible until the user
        # explicitly empties the quarantine. Future endpoint
        # /curate/empty-quarantine handles the irreversible step.
        result = curate_quarantine(
            QuarantineRequest(
                dataset_path=req.dataset_path,
                paths=paths,
                reason=req.reason or f"deleted via audit issues: {','.join(kinds)}",
            ),
        )

    return {
        "action": req.action,
        "matched_count": len(paths),
        "result": result,
    }


# --------------------------------------------------------------------------- #
# Backup roll-back
# --------------------------------------------------------------------------- #


@router.get("/curate/backups")
def curate_backups_list(dataset_path: str) -> dict[str, Any]:
    """List restorable backups (path, mtime, size)."""
    backups = _backups_root(dataset_path)
    if not backups.is_dir():
        return {"entries": []}
    entries: list[dict[str, Any]] = []
    for f in _walk_images(backups, True):
        st = f.stat()
        try:
            rel = str(_relative_under(backups, f))
        except ValueError:
            continue
        entries.append(
            {
                "backup_path": str(f),
                "relative_path": rel,
                "size": st.st_size,
                "mtime": st.st_mtime,
            },
        )
    entries.sort(key=lambda e: e["mtime"], reverse=True)
    return {"entries": entries}


class RestoreBackupRequest(BaseModel):
    dataset_path: str
    backup_paths: list[str]


@router.post("/curate/restore-backup")
def curate_restore_backup(req: RestoreBackupRequest) -> dict[str, Any]:
    """Copy a backup back to its original location, overwriting the
    current file. The backup itself is preserved — repeated restores
    are idempotent."""
    root = _ensure_dataset(req.dataset_path)
    backups = _backups_root(req.dataset_path)
    if not backups.is_dir():
        raise HTTPException(404, "no backups directory")

    restored: list[str] = []
    failed: list[dict[str, str]] = []
    for raw in req.backup_paths:
        try:
            candidate = Path(raw).expanduser()
            if not candidate.is_absolute():
                candidate = backups / candidate
            src = resolve_file_under(backups, candidate)
            if src is None:
                raise ValueError("backup is outside the backup root or is a link")
            rel = _relative_under(backups, src)
            dst = _restore_target(root, str(root / rel))
            with _file_mutation(dst):
                _copy2_atomic(src, dst)
                cap_src = resolve_file_under(backups, src.with_suffix(".txt"))
                if cap_src is not None:
                    cap_dst = _restore_target(root, str(dst.with_suffix(".txt")))
                    _copy2_atomic(cap_src, cap_dst)
            restored.append(str(dst))
        except (HTTPException, OSError, ValueError) as exc:
            message = str(exc.detail) if isinstance(exc, HTTPException) else str(exc)
            failed.append({"path": raw, "error": message})
    if restored:
        _clear_dataset_view_caches(root)
    return {
        "restored": restored,
        "restored_count": len(restored),
        "failed": failed,
    }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _walk_images(root: Path, recursive: bool):
    """Yield image files inside root, skipping the .workbench tree."""
    for path in iter_safe_files(
        root,
        recursive=recursive,
        skip_dirs=frozenset({".workbench"}),
    ):
        if path.suffix.lower() in IMAGE_SUFFIXES:
            yield path
