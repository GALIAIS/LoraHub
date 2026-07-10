"""Shared helpers for the Image Studio router package.

These helpers are reused by multiple sub-routers (listings, annotations,
ops, ai, datasets, dedupe, similarity, tagging). Keep this module free of
its own ``APIRouter`` so submodules can ``from ._shared import ...``
without provoking circular imports through the package ``__init__``.
"""

from __future__ import annotations

import contextlib
import hashlib
import json as json_stdlib
import os
import shutil
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from lorahub.api import app as app_module
from lorahub.api import paths as api_paths
from lorahub.api.dataset_files import (
    IMAGE_SUFFIXES,
    is_link_like,
    iter_safe_files,
    resolve_file_under,
    resolve_dataset_directory,
    resolve_dataset_file,
)
from lorahub.api.image_studio_store import (
    ImageAnnotation,
    ImageStudioStore,
)

_SCAN_CACHE_TTL_S = 2.0
_SCAN_CACHE_MAX = 64
_SCAN_CACHE: dict[tuple[str, bool], tuple[float, list[Path]]] = {}
_SCAN_CACHE_LOCK = threading.Lock()
_FILE_MUTATION_REGISTRY_LOCK = threading.Lock()
_FILE_MUTATION_LOCKS: dict[str, tuple[threading.RLock, int]] = {}


def _mutation_key(path: Path) -> str:
    """Use one lock for an image and its same-stem caption sidecar."""
    try:
        normalised = path.expanduser().resolve(strict=False)
    except (OSError, RuntimeError):
        normalised = Path(os.path.abspath(path.expanduser()))
    return os.path.normcase(str(normalised.with_suffix("")))


@contextlib.contextmanager
def _file_mutation(path: Path) -> Iterator[None]:
    """Serialise in-process edits to an image/caption pair."""
    key = _mutation_key(path)
    with _FILE_MUTATION_REGISTRY_LOCK:
        entry = _FILE_MUTATION_LOCKS.get(key)
        if entry is None:
            lock = threading.RLock()
            users = 0
        else:
            lock, users = entry
        _FILE_MUTATION_LOCKS[key] = (lock, users + 1)
    try:
        with lock:
            yield
    finally:
        with _FILE_MUTATION_REGISTRY_LOCK:
            current = _FILE_MUTATION_LOCKS.get(key)
            if current is not None and current[0] is lock:
                if current[1] <= 1:
                    _FILE_MUTATION_LOCKS.pop(key, None)
                else:
                    _FILE_MUTATION_LOCKS[key] = (lock, current[1] - 1)


def _atomic_write_text(path: Path, value: str, *, encoding: str = "utf-8") -> None:
    """Write a sidecar without exposing a truncated intermediate file."""
    with _file_mutation(path):
        if is_link_like(path) or is_link_like(path.parent):
            raise OSError(f"text destination cannot be a link: {path}")
        if not path.parent.is_dir():
            raise OSError(f"text destination directory does not exist: {path.parent}")
        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding=encoding,
                delete=False,
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
            ) as handle:
                temp_path = Path(handle.name)
                handle.write(value)
                handle.flush()
                os.fsync(handle.fileno())
            temp_path.replace(path)
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)


def _atomic_save_image(
    image: Any,
    path: Path,
    *,
    image_format: str | None,
    **save_kwargs: Any,
) -> None:
    """Encode beside an image and publish it with one atomic replace."""
    with _file_mutation(path):
        if is_link_like(path) or is_link_like(path.parent):
            raise OSError(f"image destination cannot be a link: {path}")
        fd, raw_temp = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=path.suffix,
        )
        os.close(fd)
        temp_path = Path(raw_temp)
        try:
            image.save(temp_path, format=image_format, **save_kwargs)
            # Windows rejects fsync on a read-only descriptor.
            with temp_path.open("rb+") as handle:
                os.fsync(handle.fileno())
            temp_path.replace(path)
        finally:
            temp_path.unlink(missing_ok=True)


def _clear_scan_cache(path: Path | None = None) -> None:
    with _SCAN_CACHE_LOCK:
        if path is None:
            _SCAN_CACHE.clear()
            return
        root = str(path.resolve())
        for key in list(_SCAN_CACHE):
            if (
                key[0] == root
                or key[0].startswith(root + os.sep)
                or root.startswith(key[0] + os.sep)
            ):
                _SCAN_CACHE.pop(key, None)


def _clear_dataset_view_caches(path: Path | None = None) -> None:
    _clear_scan_cache(path)
    from lorahub.api.helpers import _clear_dataset_scan_cache  # noqa: PLC0415

    _clear_dataset_scan_cache(path)


def _store() -> ImageStudioStore:
    """Return the process-wide ImageStudioStore (or 503 if not initialised)."""
    s = app_module._image_studio_store
    if s is None:
        raise HTTPException(503, "image studio store not initialised")
    return s


def _writable_dataset_directory(raw: str) -> Path:
    try:
        return resolve_dataset_directory(raw)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


def _writable_dataset_file(raw: str) -> Path:
    try:
        return resolve_dataset_file(raw)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


def _stored_path_is_within(raw: str, directory: Path) -> bool:
    """Check stored paths without vulnerable string-prefix matching."""
    try:
        Path(raw).expanduser().resolve().relative_to(directory.resolve())
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def _scan_images(directory: Path, recursive: bool) -> list[Path]:
    """Collect image files from directory, respecting IMAGE_SUFFIXES."""
    now = time.monotonic()
    key = (str(directory.resolve()), recursive)
    with _SCAN_CACHE_LOCK:
        cached = _SCAN_CACHE.get(key)
    if cached is not None:
        built_at, paths = cached
        if now - built_at <= _SCAN_CACHE_TTL_S:
            return [path for path in paths if path.is_file() and not is_link_like(path)]

    results = [
        path
        for path in iter_safe_files(directory, recursive=recursive)
        if path.suffix.lower() in IMAGE_SUFFIXES
    ]
    with _SCAN_CACHE_LOCK:
        _SCAN_CACHE[key] = (now, results)
        if len(_SCAN_CACHE) > _SCAN_CACHE_MAX:
            for old_key in list(_SCAN_CACHE)[: len(_SCAN_CACHE) - _SCAN_CACHE_MAX]:
                _SCAN_CACHE.pop(old_key, None)
    return results


def _ann_to_dict(ann: ImageAnnotation) -> dict[str, Any]:
    return {
        "aiCaption": ann.ai_caption,
        "aiQualityScore": ann.ai_quality_score,
        "aiQualityLabel": ann.ai_quality_label,
        "aiQualityReason": ann.ai_quality_reason,
        "aiComposition": ann.ai_composition,
        "aiTriggerWords": ann.ai_trigger_words,
        "userQualityLabel": ann.user_quality_label,
        "userNotes": ann.user_notes,
        "softDeleted": ann.soft_deleted,
        "favorite": ann.favorite,
    }


def _image_item(p: Path, directory: Path, store: ImageStudioStore) -> dict[str, Any]:
    """Build a single item dict for the listing response."""
    stat = p.stat()
    rel = str(p.relative_to(directory)).replace("\\", "/")
    ann = store.get_annotation(str(p))
    caption_path = resolve_file_under(directory, p.with_suffix(".txt"))
    caption_exists = caption_path is not None
    caption = (
        caption_path.read_text(encoding="utf-8", errors="replace").strip()
        if caption_path
        else None
    )
    return {
        "path": str(p),
        "relativePath": rel,
        "name": p.name,
        "width": ann.width if ann else None,
        "height": ann.height if ann else None,
        "bytes": stat.st_size,
        "mtime": stat.st_mtime,
        "caption": caption,
        "captionExists": caption_exists,
        "annotation": _ann_to_dict(ann) if ann else None,
        "thumbUrl": f"/api/datasets/thumb?path={str(p)}&size=256",
    }


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _unique_path(path: Path) -> Path:
    if not path.exists() and not is_link_like(path):
        return path
    idx = 2
    while True:
        candidate = path.with_name(f"{path.stem}-{idx}{path.suffix}")
        if not candidate.exists() and not is_link_like(candidate):
            return candidate
        idx += 1


def _unique_image_pair_path(path: Path, *, with_caption: bool) -> Path:
    index = 1
    while True:
        candidate = (
            path
            if index == 1
            else path.with_name(f"{path.stem}-{index}{path.suffix}")
        )
        caption = candidate.with_suffix(".txt")
        if (
            not candidate.exists()
            and not is_link_like(candidate)
            and (
                not with_caption
                or (not caption.exists() and not is_link_like(caption))
            )
        ):
            return candidate
        index += 1


def _soft_delete(file_path: Path) -> None:
    """Move file + sidecar to trash directory."""
    from datetime import UTC, datetime  # noqa: PLC0415

    with _file_mutation(file_path):
        if is_link_like(file_path) or not file_path.is_file():
            raise OSError(f"image is missing or is not a regular file: {file_path}")
        trash_dir = _safe_runs_subdir(
            "_image_studio_trash",
            datetime.now(UTC).strftime("%Y-%m-%d"),
        )
        caption = resolve_file_under(file_path.parent, file_path.with_suffix(".txt"))
        dest = _unique_image_pair_path(
            trash_dir / file_path.name,
            with_caption=caption is not None,
        )
        moved_image = False
        try:
            shutil.move(str(file_path), str(dest))
            moved_image = True
            if caption is not None:
                shutil.move(str(caption), str(dest.with_suffix(".txt")))
        except OSError as exc:
            if moved_image and dest.exists() and not file_path.exists():
                try:
                    shutil.move(str(dest), str(file_path))
                except OSError as rollback_exc:
                    raise OSError(
                        f"soft delete failed: {exc}; rollback failed: {rollback_exc}"
                    ) from exc
            raise


def _safe_runs_subdir(*parts: str) -> Path:
    """Create a managed runs subdirectory without traversing links."""
    root_path = api_paths.runs_dir()
    if is_link_like(root_path):
        raise HTTPException(409, f"managed runs path cannot be a link: {root_path}")
    root_path.mkdir(parents=True, exist_ok=True)
    root = root_path.resolve()
    current = root
    for part in parts:
        candidate = current / part
        if is_link_like(candidate):
            raise HTTPException(409, f"managed runs path cannot be a link: {candidate}")
        candidate.mkdir(exist_ok=True)
        current = candidate.resolve()
        try:
            current.relative_to(root)
        except ValueError as exc:
            raise HTTPException(409, "managed runs path escapes the runs directory") from exc
    return current


def _sse_event(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json_stdlib.dumps(data, ensure_ascii=False)}\n\n"
