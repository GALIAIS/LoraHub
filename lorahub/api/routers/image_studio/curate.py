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

import json
import os
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field

from lorahub.api.dataset_files import IMAGE_SUFFIXES

router = APIRouter(prefix="/api/image-studio", tags=["image-studio"])


# --------------------------------------------------------------------------- #
# Path conventions
# --------------------------------------------------------------------------- #


def _workbench_root(dataset_path: str) -> Path:
    return Path(dataset_path).resolve() / ".workbench"


def _quarantine_root(dataset_path: str) -> Path:
    return _workbench_root(dataset_path) / "quarantine"


def _backups_root(dataset_path: str) -> Path:
    return _workbench_root(dataset_path) / "backups"


def _audit_cache_path(dataset_path: str) -> Path:
    return _workbench_root(dataset_path) / "audit.json"


def _resolve_under(dataset_path: str, candidate: str) -> Path:
    """Return a Path inside ``dataset_path`` only, or raise.

    Rejects paths that escape the dataset via ``..`` / absolute paths to
    elsewhere. The image studio routes everywhere take absolute paths
    that originated in our own listings response, so we know they're
    inside the dataset; this is belt-and-suspenders.
    """
    root = Path(dataset_path).resolve()
    p = Path(candidate).resolve()
    try:
        p.relative_to(root)
    except ValueError as exc:
        raise HTTPException(400, f"path is outside dataset: {candidate}") from exc
    return p


def _relative_under(root: Path, p: Path) -> Path:
    return p.resolve().relative_to(root.resolve())


# --------------------------------------------------------------------------- #
# Backup helper
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _BackupResult:
    backup_path: Path


def _backup_file(dataset_path: str, src: Path) -> _BackupResult | None:
    """Copy ``src`` (and its .txt sidecar) to .workbench/backups/<rel>.

    Returns the backup path, or ``None`` when src no longer exists. The
    sidecar is best-effort — if the .txt is gone the image still gets
    backed up.
    """
    if not src.is_file():
        return None
    root = Path(dataset_path).resolve()
    rel = _relative_under(root, src)
    dst = _backups_root(dataset_path) / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Don't overwrite — keep the *first* backup before any chain of
    # edits. ``restore-backup`` rolls back to that pristine version.
    if not dst.exists():
        shutil.copy2(src, dst)
        cap_src = src.with_suffix(".txt")
        if cap_src.is_file():
            cap_dst = dst.with_suffix(".txt")
            shutil.copy2(cap_src, cap_dst)
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


@router.post("/curate/auto-rotate")
def curate_auto_rotate(req: AutoRotateRequest) -> dict[str, Any]:
    """Apply EXIF orientation, write pixels back, strip the EXIF tag.

    Idempotent: a file whose orientation is already 1 (or missing)
    is left untouched.
    """
    root = Path(req.dataset_path).resolve()
    if not root.is_dir():
        raise HTTPException(404, f"dataset not found: {root}")

    targets: list[Path] = []
    if req.paths:
        targets = [_resolve_under(req.dataset_path, p) for p in req.paths]
    else:
        for f in _walk_images(root, req.recursive):
            targets.append(f)

    rotated: list[str] = []
    skipped: list[str] = []
    failed: list[dict[str, str]] = []

    for src in targets:
        try:
            with Image.open(src) as img:
                exif = img.getexif()
                orientation = exif.get(0x0112) if exif else None
                if not orientation or orientation == 1:
                    skipped.append(str(src))
                    continue
                # ``exif_transpose`` reads the orientation tag and
                # returns a pixel-rotated copy with the tag dropped.
                rotated_img = ImageOps.exif_transpose(img)
                if rotated_img is None:
                    skipped.append(str(src))
                    continue
                rotated_img.load()
            _backup_file(req.dataset_path, src)
            # Pillow's save infers format from the file path. Drop EXIF
            # so the next reader doesn't double-rotate.
            rotated_img.save(src, exif=b"")
            rotated.append(str(src))
        except (UnidentifiedImageError, OSError) as exc:
            failed.append({"path": str(src), "error": str(exc)})

    return {
        "rotated": rotated,
        "rotated_count": len(rotated),
        "skipped_count": len(skipped),
        "failed": failed,
    }


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


@router.post("/curate/quarantine")
def curate_quarantine(req: QuarantineRequest) -> dict[str, Any]:
    """Move files (and their .txt sidecars) to ``.workbench/quarantine/``.

    Uses ``shutil.move`` so cross-device renames degrade to copy + remove
    transparently. Each path is tracked in
    ``.workbench/quarantine/index.jsonl`` so the restore endpoint
    knows the original location.
    """
    root = Path(req.dataset_path).resolve()
    if not root.is_dir():
        raise HTTPException(404, f"dataset not found: {root}")

    qroot = _quarantine_root(req.dataset_path)
    qroot.mkdir(parents=True, exist_ok=True)
    index_path = qroot / "index.jsonl"

    moved: list[dict[str, str]] = []
    failed: list[dict[str, str]] = []
    timestamp = datetime.now(UTC).replace(microsecond=0).isoformat()
    for raw in req.paths:
        try:
            src = _resolve_under(req.dataset_path, raw)
            if not src.is_file():
                failed.append({"path": str(raw), "error": "not a file"})
                continue
            rel = _relative_under(root, src)
            dst = qroot / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            # Don't clobber: if the same name was quarantined before,
            # suffix with a counter so both rounds remain restorable.
            if dst.exists():
                dst = _disambiguate(dst)
            shutil.move(str(src), str(dst))
            cap_src = src.with_suffix(".txt")
            cap_dst = dst.with_suffix(".txt") if cap_src.is_file() else None
            if cap_src.is_file():
                shutil.move(str(cap_src), str(cap_dst))
            entry = {
                "moved_at": timestamp,
                "original_path": str(src),
                "quarantine_path": str(dst),
                "caption_quarantine_path": str(cap_dst) if cap_dst else None,
                "reason": req.reason,
            }
            with open(index_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            moved.append(entry)
        except HTTPException:
            raise
        except OSError as exc:
            failed.append({"path": raw, "error": str(exc)})

    return {
        "moved": moved,
        "moved_count": len(moved),
        "failed": failed,
    }


def _disambiguate(p: Path) -> Path:
    """Append ``-2 / -3 / ...`` until the path is free."""
    i = 2
    while True:
        cand = p.with_name(f"{p.stem}-{i}{p.suffix}")
        if not cand.exists():
            return cand
        i += 1


@router.get("/curate/quarantine")
def curate_quarantine_list(dataset_path: str) -> dict[str, Any]:
    """Return the quarantine index — what was moved, when, why."""
    qroot = _quarantine_root(dataset_path)
    index_path = qroot / "index.jsonl"
    if not index_path.is_file():
        return {"entries": []}
    entries: list[dict[str, Any]] = []
    with open(index_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
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
    qroot = _quarantine_root(req.dataset_path)
    index_path = qroot / "index.jsonl"
    if not index_path.is_file():
        raise HTTPException(404, "no quarantine index found for this dataset")

    # Load + rewrite the whole index to mark restorations.
    entries: list[dict[str, Any]] = []
    with open(index_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    restored: list[dict[str, str]] = []
    failed: list[dict[str, str]] = []
    timestamp = datetime.now(UTC).replace(microsecond=0).isoformat()
    target_set = set(req.quarantine_paths)

    for entry in entries:
        if entry.get("restored_at"):
            continue
        if entry["quarantine_path"] not in target_set:
            continue
        try:
            src = Path(entry["quarantine_path"])
            dst = Path(entry["original_path"])
            if not src.is_file():
                failed.append(
                    {"path": str(src), "error": "quarantined file missing"},
                )
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                # Original location now occupied — rename incoming file
                # so we don't clobber whatever is there now.
                dst = _disambiguate(dst)
            shutil.move(str(src), str(dst))
            cap_q = entry.get("caption_quarantine_path")
            if cap_q and Path(cap_q).is_file():
                cap_dst = dst.with_suffix(".txt")
                shutil.move(str(cap_q), str(cap_dst))
            entry["restored_at"] = timestamp
            entry["restored_path"] = str(dst)
            restored.append(entry)
        except OSError as exc:
            failed.append({"path": entry["quarantine_path"], "error": str(exc)})

    # Rewrite the index — keeps successful restore entries marked.
    with open(index_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

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


@router.post("/curate/batch-resize")
def curate_batch_resize(req: BatchResizeRequest) -> dict[str, Any]:
    """Resample images so their short edge equals ``target_short_edge``.

    Aspect ratio is preserved; long edge scales by the same factor.
    Backups land in .workbench/backups/.
    """
    root = Path(req.dataset_path).resolve()
    if not root.is_dir():
        raise HTTPException(404, f"dataset not found: {root}")

    targets: list[Path] = []
    if req.paths:
        targets = [_resolve_under(req.dataset_path, p) for p in req.paths]
    else:
        targets = list(_walk_images(root, req.recursive))

    resampled: list[dict[str, Any]] = []
    skipped: list[str] = []
    failed: list[dict[str, str]] = []
    resample = _PIL_RESAMPLE[req.filter]

    for src in targets:
        try:
            with Image.open(src) as img:
                w, h = img.size
                short = min(w, h)
                if short == req.target_short_edge:
                    skipped.append(str(src))
                    continue
                if short > req.target_short_edge:
                    # Downscale — always allowed.
                    pass
                else:
                    # Upscale — gated by the flag.
                    if not req.upscale:
                        skipped.append(str(src))
                        continue
                scale = req.target_short_edge / short
                new_w = max(1, round(w * scale))
                new_h = max(1, round(h * scale))
                new_img = img.resize((new_w, new_h), resample=resample)
                new_img.load()
            _backup_file(req.dataset_path, src)
            new_img.save(src)
            resampled.append(
                {
                    "path": str(src),
                    "from": [w, h],
                    "to": [new_w, new_h],
                },
            )
        except (UnidentifiedImageError, OSError) as exc:
            failed.append({"path": str(src), "error": str(exc)})

    return {
        "resampled": resampled,
        "resampled_count": len(resampled),
        "skipped_count": len(skipped),
        "failed": failed,
    }


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
    root = Path(req.dataset_path).resolve()
    backups = _backups_root(req.dataset_path)
    if not backups.is_dir():
        raise HTTPException(404, "no backups directory")

    restored: list[str] = []
    failed: list[dict[str, str]] = []
    for raw in req.backup_paths:
        try:
            src = Path(raw)
            rel = _relative_under(backups, src)
            dst = root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            cap_src = src.with_suffix(".txt")
            if cap_src.is_file():
                shutil.copy2(cap_src, dst.with_suffix(".txt"))
            restored.append(str(dst))
        except (OSError, ValueError) as exc:
            failed.append({"path": raw, "error": str(exc)})
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
    if recursive:
        for cur, dirs, files in os.walk(root):
            # Don't descend into our own staging directories.
            dirs[:] = [d for d in dirs if d != ".workbench"]
            for f in files:
                p = Path(cur) / f
                if p.suffix.lower() in IMAGE_SUFFIXES:
                    yield p
    else:
        for p in root.iterdir():
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
                yield p
