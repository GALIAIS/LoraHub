"""Shared helpers for the LoraHub HTTP API.

Pure-ish functions and constants that are reused by more than one router
module. Keep these free of FastAPI-router state so they can be imported in
either direction without cycles.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from lorahub.core.backends.kohya.backend import KohyaBackend
from lorahub.core.config.schema import TrainingConfig

_IMAGE_SUFFIXES = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp"}

# Matches the leading-char + 1-63 trailing chars name rule used by save_config.
_NAME_RE_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$"


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
        c_resolved = c.resolve()
        try:
            c_resolved.relative_to(base)
        except ValueError:
            continue
        if c_resolved.is_file():
            return c_resolved
    raise HTTPException(status_code=404, detail="config not found")


def _preflight_config(cfg: TrainingConfig) -> dict[str, Any]:
    backend = KohyaBackend()
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
    if cfg.dataset.source.is_dir():
        image_files = sorted(
            p
            for p in cfg.dataset.source.iterdir()
            if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
        )
        for image in image_files:
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
            "dataset_exists": cfg.dataset.source.is_dir(),
            "image_files": len(image_files),
            "caption_files": caption_files,
            "missing_caption_files": missing_caption_files[:20],
            "missing_caption_files_truncated": len(missing_caption_files) > 20,
        },
    }


def _scan_dataset_path(
    path: Path, *, recursive: bool = False, limit: int = 40
) -> dict[str, Any]:
    root = path.expanduser().resolve()
    exists = root.is_dir()
    image_files: list[Path] = []
    caption_files = 0
    missing_caption_files: list[str] = []
    samples: list[dict[str, Any]] = []

    if exists:
        iterator = root.rglob("*") if recursive else root.iterdir()
        image_files = sorted(
            p for p in iterator if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
        )
        for image in image_files:
            caption_path = image.with_suffix(".txt")
            caption: str | None = None
            if caption_path.is_file():
                caption_files += 1
                with contextlib.suppress(Exception):
                    caption = caption_path.read_text(encoding="utf-8").strip()
            else:
                missing_caption_files.append(image.relative_to(root).as_posix())
            if len(samples) < max(limit, 0):
                samples.append(
                    {
                        "name": image.name,
                        "path": str(image),
                        "relative_path": image.relative_to(root).as_posix(),
                        "caption_exists": caption_path.is_file(),
                        "caption": caption,
                    }
                )

    return {
        "path": str(root),
        "exists": exists,
        "recursive": recursive,
        "image_files": len(image_files),
        "caption_files": caption_files,
        "missing_caption_files": missing_caption_files[: max(limit, 0)],
        "missing_caption_files_truncated": len(missing_caption_files) > max(limit, 0),
        "samples": samples,
    }


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
