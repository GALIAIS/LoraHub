"""Shared helpers for the Image Studio router package.

These helpers are reused by multiple sub-routers (listings, annotations,
ops, ai, datasets, dedupe, similarity, tagging). Keep this module free of
its own ``APIRouter`` so submodules can ``from ._shared import ...``
without provoking circular imports through the package ``__init__``.
"""

from __future__ import annotations

import hashlib
import json as json_stdlib
import os
import shutil
import time
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from lorahub.api import app as app_module
from lorahub.api.dataset_files import IMAGE_SUFFIXES
from lorahub.api.image_studio_store import (
    ImageAnnotation,
    ImageStudioStore,
)

_SCAN_CACHE_TTL_S = 2.0
_SCAN_CACHE_MAX = 64
_SCAN_CACHE: dict[tuple[str, bool], tuple[float, list[Path]]] = {}


def _clear_scan_cache(path: Path | None = None) -> None:
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


def _scan_images(directory: Path, recursive: bool) -> list[Path]:
    """Collect image files from directory, respecting IMAGE_SUFFIXES."""
    now = time.monotonic()
    key = (str(directory.resolve()), recursive)
    cached = _SCAN_CACHE.get(key)
    if cached is not None:
        built_at, paths = cached
        if now - built_at <= _SCAN_CACHE_TTL_S:
            return list(paths)

    results: list[Path] = []
    if recursive:
        for root, _dirs, files in os.walk(directory):
            for f in files:
                p = Path(root) / f
                if p.suffix.lower() in IMAGE_SUFFIXES:
                    results.append(p)
    else:
        for p in directory.iterdir():
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
                results.append(p)
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
    caption_path = p.with_suffix(".txt")
    caption_exists = caption_path.is_file()
    caption = caption_path.read_text(encoding="utf-8").strip() if caption_exists else None
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
    if not path.exists():
        return path
    idx = 2
    while True:
        candidate = path.with_name(f"{path.stem}-{idx}{path.suffix}")
        if not candidate.exists():
            return candidate
        idx += 1


def _soft_delete(file_path: Path) -> None:
    """Move file + sidecar to trash directory."""
    from datetime import UTC, datetime  # noqa: PLC0415

    trash_dir = Path("runs") / "_image_studio_trash" / datetime.now(UTC).strftime("%Y-%m-%d")
    trash_dir.mkdir(parents=True, exist_ok=True)
    dest = _unique_path(trash_dir / file_path.name)
    caption = file_path.with_suffix(".txt")
    shutil.move(str(file_path), str(dest))
    if caption.is_file():
        shutil.move(str(caption), str(dest.with_suffix(".txt")))


def _sse_event(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json_stdlib.dumps(data, ensure_ascii=False)}\n\n"
