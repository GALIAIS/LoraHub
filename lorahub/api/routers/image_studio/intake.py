"""Intake — bring images into a dataset.

Three on-ramps:

  - POST /intake/preflight        Hash incoming files + the existing
                                  dataset, classify each candidate as
                                  new / duplicate-of-existing /
                                  duplicate-within-batch.
  - POST /intake/local-path       Copy from a server-side directory
                                  (recursive optional, phash dedupe
                                  optional). Distinct from the dataset
                                  upload endpoint which only takes a
                                  multipart blob from the browser.
  - POST /intake/from-dataset     Copy a subset of one dataset into
                                  another (glob filter against
                                  relative paths). Source untouched.

All three preserve the .txt sidecar when present. None mutates
existing files in the destination — every imported file gets a
disambiguated name (``stem-2.png``) if it would collide.

EXIF orientation is *not* baked in here — the user can run the curate
stage's auto-rotate after the fact. Keeps intake fast and side-effect-
narrow.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from lorahub.api.dataset_files import IMAGE_SUFFIXES

router = APIRouter(prefix="/api/image-studio", tags=["image-studio"])


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _ensure_dataset(dataset_path: str) -> Path:
    p = Path(dataset_path).resolve()
    if not p.is_dir():
        raise HTTPException(404, f"dataset not found: {p}")
    return p


def _resolve_external(p: str) -> Path:
    """Resolve a server-side path. Doesn't restrict to the dataset
    tree — this *is* the import-from-elsewhere path."""
    out = Path(p).expanduser().resolve()
    if not out.exists():
        raise HTTPException(404, f"path not found: {out}")
    return out


def _walk_images(root: Path, recursive: bool):
    if recursive:
        for cur, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d != ".workbench"]
            for f in files:
                p = Path(cur) / f
                if p.suffix.lower() in IMAGE_SUFFIXES:
                    yield p
    else:
        for p in root.iterdir():
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
                yield p


def _disambiguate(p: Path) -> Path:
    """Return ``p`` if free, else ``p`` with ``-2 / -3 / ...``."""
    if not p.exists():
        return p
    i = 2
    while True:
        cand = p.with_name(f"{p.stem}-{i}{p.suffix}")
        if not cand.exists():
            return cand
        i += 1


def _phash(path: Path) -> str | None:
    """Try ``lorahub.core.phash.phash64``; degrade to None on failure.

    Returns the hex-encoded 64-bit hash string used everywhere else
    in the codebase. ``None`` means we couldn't read the image
    (corrupt / unsupported); that file gets imported anyway but isn't
    dedup'd.
    """
    try:
        from lorahub.core.phash import phash64  # noqa: PLC0415

        return phash64(path)
    except Exception:  # noqa: BLE001
        return None


def _existing_phash_set(dataset: Path) -> set[str]:
    """Compute phash for every image already in the dataset.

    Walks .workbench-free; returns a set of hash strings for fast
    lookup. Worst case ``O(n)`` reads; for a 1k-image dataset this is
    ~3-5 seconds on cold disk.
    """
    out: set[str] = set()
    for img in _walk_images(dataset, recursive=True):
        h = _phash(img)
        if h is not None:
            out.add(h)
    return out


def _hamming(a: str, b: str) -> int:
    """Hex-string hamming distance.

    The phash strings are 16 hex chars (= 64 bits). XOR via int +
    popcount.
    """
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def _is_near_duplicate(h: str, others: set[str], threshold: int) -> bool:
    """phash64 hamming ≤ threshold counts as duplicate.

    Default threshold 4 catches near-identical images (re-encoding
    artefacts / minor crops); raise to 6-8 for "loosely similar"
    matches; lower to 0 for exact only.
    """
    if h in others:
        return True
    if threshold == 0:
        return False
    for o in others:
        if _hamming(h, o) <= threshold:
            return True
    return False


def _copy_with_sidecar(src: Path, dst: Path) -> None:
    """Copy ``src`` to ``dst`` preserving mtime; bring its .txt
    caption sidecar if one exists."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    side = src.with_suffix(".txt")
    if side.is_file():
        shutil.copy2(side, dst.with_suffix(".txt"))


# --------------------------------------------------------------------------- #
# Preflight — classify candidates without writing
# --------------------------------------------------------------------------- #


class PreflightRequest(BaseModel):
    dataset_path: str
    source_path: str = Field(..., description="A directory or single file.")
    recursive: bool = True
    phash_threshold: int = Field(
        default=4,
        ge=0,
        le=16,
        description="Hamming distance ≤ threshold counts as duplicate. 0 = exact bits, 4 ≈ near-identical.",
    )


@router.post("/intake/preflight")
def intake_preflight(req: PreflightRequest) -> dict[str, Any]:
    """Classify every image under ``source_path`` as new / duplicate.

    Returns three lists keyed by source path:
      - ``new``: not present in dataset (worth importing).
      - ``duplicate_existing``: phash matches an image already in the
        dataset (would be a redundant import).
      - ``duplicate_within_batch``: phash matches another file from
        the same source — typical when a folder has both a source and
        a re-encoded copy of the same shot.

    No filesystem writes. The UI calls preflight, shows a confirmation
    summary, then re-calls the actual import endpoint.
    """
    dst = _ensure_dataset(req.dataset_path)
    src = _resolve_external(req.source_path)

    candidates: list[Path] = []
    if src.is_file():
        if src.suffix.lower() in IMAGE_SUFFIXES:
            candidates.append(src)
    else:
        candidates = list(_walk_images(src, req.recursive))

    existing = _existing_phash_set(dst)

    new_files: list[dict[str, Any]] = []
    dup_existing: list[dict[str, Any]] = []
    dup_within: list[dict[str, Any]] = []
    seen_in_batch: set[str] = set()

    for f in candidates:
        h = _phash(f)
        info: dict[str, Any] = {
            "source_path": str(f),
            "phash": h,
        }
        if h is None:
            new_files.append(info)
            continue
        if _is_near_duplicate(h, existing, req.phash_threshold):
            dup_existing.append(info)
            continue
        if _is_near_duplicate(h, seen_in_batch, req.phash_threshold):
            dup_within.append(info)
            continue
        seen_in_batch.add(h)
        new_files.append(info)

    return {
        "candidate_count": len(candidates),
        "new_count": len(new_files),
        "duplicate_existing_count": len(dup_existing),
        "duplicate_within_batch_count": len(dup_within),
        "new": new_files[:1000],
        "duplicate_existing": dup_existing[:1000],
        "duplicate_within_batch": dup_within[:1000],
        "truncated": (
            len(new_files) > 1000
            or len(dup_existing) > 1000
            or len(dup_within) > 1000
        ),
    }


# --------------------------------------------------------------------------- #
# Local path import
# --------------------------------------------------------------------------- #


class LocalPathRequest(BaseModel):
    dataset_path: str
    source_path: str
    recursive: bool = True
    skip_duplicates: bool = True
    phash_threshold: int = Field(default=4, ge=0, le=16)
    # When True, the source files are moved instead of copied.
    # Useful for one-shot ingestions where the source dir is staging
    # and shouldn't be kept around.
    move: bool = False


@router.post("/intake/local-path")
def intake_local_path(req: LocalPathRequest) -> dict[str, Any]:
    """Import every image under ``source_path`` into the dataset.

    Default behaviour preserves the source: copies + caption sidecars
    come along, ``move=True`` moves them instead. ``skip_duplicates``
    runs preflight inline and silently skips matches.
    """
    dst = _ensure_dataset(req.dataset_path)
    src = _resolve_external(req.source_path)

    candidates: list[Path] = []
    if src.is_file():
        if src.suffix.lower() in IMAGE_SUFFIXES:
            candidates.append(src)
    else:
        candidates = list(_walk_images(src, req.recursive))

    existing = (
        _existing_phash_set(dst) if req.skip_duplicates else set()
    )

    imported: list[dict[str, str]] = []
    skipped: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    seen_in_batch: set[str] = set()

    for f in candidates:
        try:
            if req.skip_duplicates:
                h = _phash(f)
                if h is not None and _is_near_duplicate(
                    h, existing, req.phash_threshold,
                ):
                    skipped.append({"source_path": str(f), "reason": "exists"})
                    continue
                if h is not None and _is_near_duplicate(
                    h, seen_in_batch, req.phash_threshold,
                ):
                    skipped.append(
                        {"source_path": str(f), "reason": "in-batch-duplicate"},
                    )
                    continue
                if h is not None:
                    seen_in_batch.add(h)
                    existing.add(h)

            target = _disambiguate(dst / f.name)
            if req.move:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(f), str(target))
                # Sidecar moves too if it lives next to the source.
                side_src = f.with_suffix(".txt")
                if side_src.is_file():
                    shutil.move(str(side_src), str(target.with_suffix(".txt")))
            else:
                _copy_with_sidecar(f, target)
            imported.append({"source_path": str(f), "imported_path": str(target)})
        except (OSError, shutil.Error) as exc:
            failed.append({"source_path": str(f), "error": str(exc)})

    return {
        "imported_count": len(imported),
        "skipped_count": len(skipped),
        "failed_count": len(failed),
        "imported": imported,
        "skipped": skipped[:500],
        "failed": failed,
    }


# --------------------------------------------------------------------------- #
# From-dataset import
# --------------------------------------------------------------------------- #


class FromDatasetRequest(BaseModel):
    dataset_path: str
    source_dataset_path: str
    pattern: str = Field(default="*", description="fnmatch glob against relative path inside source dataset")
    skip_duplicates: bool = True
    phash_threshold: int = Field(default=4, ge=0, le=16)


@router.post("/intake/from-dataset")
def intake_from_dataset(req: FromDatasetRequest) -> dict[str, Any]:
    """Copy a subset of another dataset into this one.

    ``pattern`` is fnmatch-style (``*portrait*`` / ``char_a/*``);
    matched against each image's path *relative to the source root*.
    Caption sidecars come along automatically.
    """
    import fnmatch  # noqa: PLC0415

    dst = _ensure_dataset(req.dataset_path)
    src = _ensure_dataset(req.source_dataset_path)
    if src.resolve() == dst.resolve():
        raise HTTPException(400, "source and destination are the same dataset")

    candidates: list[tuple[Path, str]] = []
    for img in _walk_images(src, recursive=True):
        rel = str(img.relative_to(src)).replace("\\", "/")
        if fnmatch.fnmatch(rel, req.pattern):
            candidates.append((img, rel))

    existing = (
        _existing_phash_set(dst) if req.skip_duplicates else set()
    )

    imported: list[dict[str, str]] = []
    skipped: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    seen_in_batch: set[str] = set()

    for img, rel in candidates:
        try:
            if req.skip_duplicates:
                h = _phash(img)
                if h is not None and _is_near_duplicate(
                    h, existing, req.phash_threshold,
                ):
                    skipped.append({"source_path": str(img), "reason": "exists"})
                    continue
                if h is not None and _is_near_duplicate(
                    h, seen_in_batch, req.phash_threshold,
                ):
                    skipped.append(
                        {"source_path": str(img), "reason": "in-batch-duplicate"},
                    )
                    continue
                if h is not None:
                    seen_in_batch.add(h)
                    existing.add(h)

            target = _disambiguate(dst / Path(rel).name)
            _copy_with_sidecar(img, target)
            imported.append(
                {
                    "source_path": str(img),
                    "imported_path": str(target),
                    "source_relative": rel,
                },
            )
        except OSError as exc:
            failed.append({"source_path": str(img), "error": str(exc)})

    return {
        "candidate_count": len(candidates),
        "imported_count": len(imported),
        "skipped_count": len(skipped),
        "failed_count": len(failed),
        "imported": imported,
        "skipped": skipped[:500],
        "failed": failed,
    }
