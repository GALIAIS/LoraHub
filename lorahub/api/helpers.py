"""Shared helpers for the LoraHub HTTP API.

Pure-ish functions and constants that are reused by more than one router
module. Keep these free of FastAPI-router state so they can be imported in
either direction without cycles.
"""

from __future__ import annotations

import contextlib
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from lorahub.api.dataset_files import is_link_like, iter_safe_files, resolve_file_under
from lorahub.core.backends.registry import get_backend
from lorahub.core.config.schema import TrainingConfig

_IMAGE_SUFFIXES = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp"}

# Matches the leading-char + 1-63 trailing chars name rule used by save_config.
_NAME_RE_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$"
_DATASET_SCAN_TTL_S = 2.0
_DATASET_SCAN_CACHE_MAX = 64
_DATASET_SCAN_CACHE: dict[
    tuple[str, bool],
    tuple[float, list[Path], list[bool], list[str], int],
] = {}


def _clear_dataset_scan_cache(path: Path | None = None) -> None:
    if path is None:
        _DATASET_SCAN_CACHE.clear()
        return
    root = str(path.expanduser().resolve())
    for key in list(_DATASET_SCAN_CACHE):
        if (
            key[0] == root
            or key[0].startswith(root + os.sep)
            or root.startswith(key[0] + os.sep)
        ):
            _DATASET_SCAN_CACHE.pop(key, None)


def _configs_dir() -> Path:
    """Resolve the configs/ directory.

    Honors $LORAHUB_configs_dir (absolute path); otherwise looks at
    `<cwd>/configs` so users get whatever templates ship with their checkout
    when running `lorahub serve` from the repo root.
    """
    override = os.environ.get("LORAHUB_configs_dir")
    if override:
        return Path(override).expanduser().resolve()
    return (Path.cwd() / "configs").resolve()


def _config_path(name: str) -> Path:
    """Resolve a config by name within the configs/ dir, blocking traversal."""
    if not name or "/" in name or "\\" in name or name.startswith(".."):
        raise HTTPException(status_code=400, detail="invalid config name")
    base = _configs_dir()
    # Accept "foo" or "foo.yaml"
    candidates = [base / name, base / f"{name}.yaml", base / f"{name}.yml"]
    for c in candidates:
        if is_link_like(c):
            raise HTTPException(status_code=400, detail="config path cannot be a link")
        c_absolute = Path(os.path.abspath(c))
        try:
            c_absolute.relative_to(base)
        except ValueError:
            continue
        if c_absolute.is_file():
            return c_absolute
    raise HTTPException(status_code=404, detail="config not found")


def _training_dataset_roots(cfg: TrainingConfig) -> list[Path | None]:
    """Return the directories the selected backend will actually train from."""
    if cfg.backend.type != "anima_lora" and cfg.dataset.subsets:
        return [subset.path for subset in cfg.dataset.subsets]
    return [cfg.dataset.source] if cfg.dataset.source is not None else []


def _preflight_config(cfg: TrainingConfig) -> dict[str, Any]:
    backend = get_backend(cfg.backend.type).backend_class()
    issues = [
        {
            **asdict(issue),
            "severity": issue.severity.value,
        }
        for issue in backend.validate(cfg)
    ]
    estimate = backend.estimate_vram(cfg)

    image_files: list[Path] = []
    caption_files = 0
    missing_caption_files: list[str] = []
    dataset_roots = _training_dataset_roots(cfg)
    for source in dataset_roots:
        if source is None or not source.is_dir():
            continue
        source_images = sorted(
            p
            for p in source.iterdir()
            if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
        )
        image_files.extend(source_images)
        for image in source_images:
            if image.with_suffix(".txt").is_file():
                caption_files += 1
            else:
                missing_caption_files.append(image.name)

    return {
        "issues": issues,
        "vram": {
            "model_mib": estimate.model_mib,
            "optimizer_mib": estimate.optimizer_mib,
            "activations_mib": estimate.activations_mib,
            "overhead_mib": estimate.overhead_mib,
            "total_mib": estimate.total_mib,
            "total_gib": round(estimate.total_gib, 2),
        },
        "paths": {
            "checkpoint_exists": cfg.base_model.checkpoint.is_file(),
            "dataset_exists": bool(dataset_roots) and all(
                source is not None and source.is_dir() for source in dataset_roots
            ),
            "image_files": len(image_files),
            "caption_files": caption_files,
            "missing_caption_files": missing_caption_files[:20],
            "missing_caption_files_truncated": len(missing_caption_files) > 20,
        },
    }


def _scan_dataset_path(
    path: Path, *, recursive: bool = False, limit: int = 40, offset: int = 0
) -> dict[str, Any]:
    root = path.expanduser().resolve()
    exists = root.is_dir()
    image_files: list[Path] = []
    caption_files = 0
    missing_caption_files: list[str] = []
    samples: list[dict[str, Any]] = []
    capped_limit = min(max(int(limit), 0), 500)
    capped_offset = max(int(offset), 0)

    if exists:
        image_files, caption_exists, missing_caption_files, caption_files = (
            _dataset_scan_index(root, recursive=recursive)
        )
        # Use the cached scan index for counts and the requested page
        # slice. Caption text itself is read live for visible rows only.
        for index, image in enumerate(image_files):
            has_caption = caption_exists[index]
            caption: str | None = None
            in_page = capped_offset <= index < capped_offset + capped_limit
            if in_page:
                caption_path = resolve_file_under(root, image.with_suffix(".txt"))
                if has_caption and caption_path is not None:
                    with contextlib.suppress(Exception):
                        caption = caption_path.read_text(
                            encoding="utf-8"
                        ).strip()
                samples.append(
                    {
                        "name": image.name,
                        "path": str(image),
                        "relative_path": image.relative_to(root).as_posix(),
                        "caption_exists": has_caption,
                        "caption": caption,
                    }
                )

    return {
        "path": str(root),
        "exists": exists,
        "recursive": recursive,
        "image_files": len(image_files),
        "caption_files": caption_files,
        "missing_caption_files": missing_caption_files[:capped_limit],
        "missing_caption_files_truncated": len(missing_caption_files) > capped_limit,
        "samples": samples,
        "limit": capped_limit,
        "offset": capped_offset,
    }


def _dataset_scan_index(
    root: Path, *, recursive: bool
) -> tuple[list[Path], list[bool], list[str], int]:
    now = time.monotonic()
    key = (str(root), recursive)
    cached = _DATASET_SCAN_CACHE.get(key)
    if cached is not None:
        built_at, images, caption_exists, missing, caption_count = cached
        if now - built_at <= _DATASET_SCAN_TTL_S:
            return images, caption_exists, missing, caption_count

    images = sorted(
        p
        for p in iter_safe_files(root, recursive=recursive)
        if p.suffix.lower() in _IMAGE_SUFFIXES
    )
    caption_exists = []
    missing = []
    caption_count = 0
    for image in images:
        has_caption = resolve_file_under(root, image.with_suffix(".txt")) is not None
        caption_exists.append(has_caption)
        if has_caption:
            caption_count += 1
        else:
            missing.append(image.relative_to(root).as_posix())

    _DATASET_SCAN_CACHE[key] = (now, images, caption_exists, missing, caption_count)
    if len(_DATASET_SCAN_CACHE) > _DATASET_SCAN_CACHE_MAX:
        for old_key in list(_DATASET_SCAN_CACHE)[
            : len(_DATASET_SCAN_CACHE) - _DATASET_SCAN_CACHE_MAX
        ]:
            _DATASET_SCAN_CACHE.pop(old_key, None)
    return images, caption_exists, missing, caption_count


def ulid_new() -> Any:
    """Wrapper so tests can patch ULID generation if needed."""
    import ulid  # noqa: PLC0415

    return ulid.new()


def _resolve_web_dist() -> Path | None:
    """Locate the built web frontend (`web/dist`).

    Search order:
      1. $LORAHUB_WEB_DIST (explicit override, e.g. for packaged installs)
      2. <repo_root>/web/dist (development checkout)
    """
    override = os.environ.get("LORAHUB_WEB_DIST")
    if override:
        candidate = Path(override).expanduser().resolve()
        return candidate if (candidate / "index.html").is_file() else None

    repo_dist = Path(__file__).resolve().parents[2] / "web" / "dist"
    if (repo_dist / "index.html").is_file():
        return repo_dist
    return None
