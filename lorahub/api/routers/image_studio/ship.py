"""Ship stage — training-readiness lint, export, save-as.

Three endpoints:

  - GET  /ship/lint           Roll up audit issues + caption coverage
                              into a green/yellow/red verdict the UI
                              renders as a 'ready to train' card.

  - POST /ship/export         Stream a zip of the dataset (images +
                              .txt sidecars). Excludes .workbench/
                              (quarantine, backups, audit cache) by
                              default; opt-in flags to include
                              backups / quarantined entries / dataset
                              meta.

  - POST /ship/save-as        Copy the current dataset into a new
                              dataset directory under datasets/<name>,
                              optionally filtering by path list (so
                              the user can save the current selection
                              as a fresh dataset).

The lint endpoint is the *gate* for the train-launch button — when
any blocker fires we return ``ready=false`` and the UI disables it.
Warnings are non-blocking; users override at their own risk.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
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


def _audit_cache_path(dataset_path: str) -> Path:
    return Path(dataset_path).resolve() / ".workbench" / "audit.json"


def _datasets_root() -> Path:
    """Match datasets.py — let LORAHUB_DATASETS_ROOT override."""
    extra = os.environ.get("LORAHUB_DATASETS_ROOT")
    if extra:
        root = Path(extra.split(os.pathsep)[0].strip())
    else:
        root = Path.cwd() / "datasets"
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _walk_dataset_files(
    root: Path,
    *,
    include_backups: bool = False,
    include_quarantine: bool = False,
):
    """Yield (Path, arcname) pairs covering every file we should ship.

    ``arcname`` is the relative path inside the zip / new dataset.
    By default skips ``.workbench/`` (curate runtime state) entirely.
    Toggle the flags to include backups (the first-known-good copies)
    and the quarantine tree.
    """
    for cur, dirs, files in os.walk(root):
        rel_cur = Path(cur).relative_to(root)
        if rel_cur.parts and rel_cur.parts[0] == ".workbench":
            # Decide whether to descend based on which sub-tree we're in.
            sub = rel_cur.parts[1] if len(rel_cur.parts) > 1 else None
            if sub == "backups" and include_backups:
                pass
            elif sub == "quarantine" and include_quarantine:
                pass
            elif sub is None:
                # At ``.workbench/`` itself — keep the children we want.
                dirs[:] = [
                    d
                    for d in dirs
                    if (d == "backups" and include_backups)
                    or (d == "quarantine" and include_quarantine)
                ]
                continue
            else:
                # In a .workbench subdir we don't want — prune.
                dirs[:] = []
                continue
        for f in files:
            full = Path(cur) / f
            arc = full.relative_to(root)
            yield full, arc


# --------------------------------------------------------------------------- #
# Lint
# --------------------------------------------------------------------------- #


@dataclass
class _LintIssue:
    severity: Literal["block", "warn"]
    code: str
    message: str
    count: int = 0


_LINT_BLOCKERS = {
    # Issue kind from audit → blocker reason.
    "corrupt": "无法读取的图像会让训练前缓存崩溃",
}

_LINT_WARNINGS = {
    "tiny": "短边 < 512 的图,训练时会被 bucket 丢弃或裁剪不理想",
    "missing_trigger": "缺触发词,可能影响身份 / 风格识别",
    "blurry": "Laplacian 方差低,可能模糊",
    "exif_rotation": "EXIF orientation 未应用 — 训练读取像素时会方向错乱",
    "no_caption": "缺 caption,这些图无法参与基于 caption 的训练",
}


@router.get("/ship/lint")
def ship_lint(dataset_path: str) -> dict[str, Any]:
    """Read the audit cache and roll it up into a training-readiness verdict.

    Returns:
      {
        "ready": bool,                   # green light
        "stale": bool,                   # cache absent / outdated
        "scanned_at": str | null,
        "image_count": int,
        "captioned_count": int,
        "trigger_word": str | null,
        "trigger_word_hits": int,
        "issues": [{severity, code, message, count}, ...],
        "blockers": int,
        "warnings": int,
      }

    ``stale=true`` when the audit cache doesn't exist; the UI should
    guide the user back to the audit stage to scan first. Once
    scanned, ``ready=true`` iff zero blockers.
    """
    root = _ensure_dataset(dataset_path)
    cache = _audit_cache_path(dataset_path)
    if not cache.is_file():
        return {
            "ready": False,
            "stale": True,
            "scanned_at": None,
            "image_count": 0,
            "captioned_count": 0,
            "trigger_word": None,
            "trigger_word_hits": 0,
            "issues": [],
            "blockers": 0,
            "warnings": 1,
            "stale_reason": "尚未审计 — 请先在 审计 阶段扫描。",
        }
    try:
        report = json.loads(cache.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(500, f"audit cache corrupt: {exc}") from None

    by_kind: dict[str, int] = {}
    for iss in report.get("issues", []):
        kind = iss.get("kind") or "unknown"
        by_kind[kind] = by_kind.get(kind, 0) + 1

    issues: list[_LintIssue] = []
    for kind, msg in _LINT_BLOCKERS.items():
        n = by_kind.get(kind, 0)
        if n > 0:
            issues.append(_LintIssue("block", kind, msg, n))
    for kind, msg in _LINT_WARNINGS.items():
        n = by_kind.get(kind, 0)
        if n > 0:
            issues.append(_LintIssue("warn", kind, msg, n))

    image_count = int(report.get("image_count", 0))
    captioned = int(report.get("captioned_count", 0))
    if image_count == 0:
        issues.append(
            _LintIssue("block", "empty", "数据集为空 — 至少需要 1 张图", 0),
        )

    blockers = sum(1 for i in issues if i.severity == "block")
    warnings = sum(1 for i in issues if i.severity == "warn")

    # Detect "stale": cache exists but the on-disk image count
    # disagrees with the cached image_count. Catches the case where
    # the user added or removed images after the last audit (those
    # changes need a re-scan to be reflected in the lint verdict).
    on_disk_imgs = sum(
        1
        for f in root.rglob("*")
        if f.is_file()
        and f.suffix.lower() in IMAGE_SUFFIXES
        and ".workbench" not in f.parts
    )
    stale = on_disk_imgs != image_count

    return {
        "ready": blockers == 0 and not stale and image_count > 0,
        "stale": stale,
        "stale_reason": (
            f"审计时 {image_count} 张,当前 {on_disk_imgs} 张 — 请重新扫描"
            if stale
            else None
        ),
        "scanned_at": report.get("scanned_at"),
        "image_count": image_count,
        "captioned_count": captioned,
        "trigger_word": report.get("trigger_word"),
        "trigger_word_hits": int(report.get("trigger_word_hits", 0)),
        "issues": [
            {
                "severity": i.severity,
                "code": i.code,
                "message": i.message,
                "count": i.count,
            }
            for i in issues
        ],
        "blockers": blockers,
        "warnings": warnings,
    }


# --------------------------------------------------------------------------- #
# Export — streaming zip
# --------------------------------------------------------------------------- #


class ExportRequest(BaseModel):
    dataset_path: str
    include_backups: bool = False
    include_quarantine: bool = False
    include_meta: bool = True
    # Optional path filter — when set, only these images (+ their
    # .txt sidecars) get included. Used by the UI's "export selection"
    # workflow.
    paths: list[str] | None = None


@router.post("/ship/export")
def ship_export(req: ExportRequest) -> StreamingResponse:
    """Stream a zip of the dataset.

    Streaming so users with multi-GB datasets don't have to hold the
    whole archive in memory. ``StreamingResponse`` flushes each
    written chunk to the client as soon as it's available.
    """
    root = _ensure_dataset(req.dataset_path)
    archive_name = root.name + ".zip"

    only: set[str] | None = None
    if req.paths:
        only = {str(Path(p).resolve()) for p in req.paths}

    def gen() -> Any:
        # ZipFile + BytesIO ring: write to an in-memory buffer, yield
        # whatever's accumulated, truncate, repeat. This lets us flush
        # frequently so the HTTP client gets bytes early.
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            files_seen = 0
            for full, arc in _walk_dataset_files(
                root,
                include_backups=req.include_backups,
                include_quarantine=req.include_quarantine,
            ):
                # Filter for path-list mode: keep image only if its
                # path is in the set OR it's the matching .txt
                # sidecar of one that is.
                if only is not None:
                    abs_full = str(full.resolve())
                    keep = abs_full in only
                    if not keep and full.suffix.lower() == ".txt":
                        for ext in (".png", ".jpg", ".jpeg", ".webp"):
                            if str(full.with_suffix(ext).resolve()) in only:
                                keep = True
                                break
                    if not keep:
                        continue
                # Skip dataset_meta unless explicitly included.
                if not req.include_meta and full.name == ".dataset_meta.json":
                    continue
                try:
                    zf.write(full, arcname=str(arc))
                    files_seen += 1
                except OSError:
                    continue

                # Flush every 50 files or 4MB of buffered data.
                if files_seen % 50 == 0 or buf.tell() > 4 * 1024 * 1024:
                    pos = buf.tell()
                    if pos > 0:
                        buf.seek(0)
                        chunk = buf.read(pos)
                        buf.seek(0)
                        buf.truncate()
                        yield chunk

        # Finalise — yield the closing central directory + EOCD.
        pos = buf.tell()
        if pos > 0:
            buf.seek(0)
            yield buf.read(pos)

    return StreamingResponse(
        gen(),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{archive_name}"',
        },
    )


# --------------------------------------------------------------------------- #
# Save-as — copy current dataset to a new one
# --------------------------------------------------------------------------- #


class SaveAsRequest(BaseModel):
    source_path: str
    new_name: str = Field(..., min_length=1)
    include_backups: bool = False
    include_quarantine: bool = False
    paths: list[str] | None = None


@router.post("/ship/save-as")
def ship_save_as(req: SaveAsRequest) -> dict[str, Any]:
    """Copy a dataset (or a subset) into datasets/<new_name>/.

    Same path-filter semantics as /ship/export. The new dataset gets
    a fresh ``.dataset_meta.json`` derived from the source's meta
    (description suffixed with "(copied from <source>)").
    """
    src = _ensure_dataset(req.source_path)
    name = req.new_name.strip()
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(400, "invalid dataset name")
    dst = _datasets_root() / name
    if dst.exists():
        raise HTTPException(409, f"dataset '{name}' already exists")
    dst.mkdir(parents=True)

    only: set[str] | None = None
    if req.paths:
        only = {str(Path(p).resolve()) for p in req.paths}

    files_copied = 0
    images_copied = 0
    for full, arc in _walk_dataset_files(
        src,
        include_backups=req.include_backups,
        include_quarantine=req.include_quarantine,
    ):
        if only is not None:
            abs_full = str(full.resolve())
            keep = abs_full in only
            if not keep and full.suffix.lower() == ".txt":
                # Match the txt to its image counterpart.
                for ext in (".png", ".jpg", ".jpeg", ".webp"):
                    if str(full.with_suffix(ext).resolve()) in only:
                        keep = True
                        break
            if not keep:
                continue
        out = dst / arc
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(full, out)
        files_copied += 1
        if full.suffix.lower() in IMAGE_SUFFIXES:
            images_copied += 1

    # Write fresh metadata.
    src_meta_file = src / ".dataset_meta.json"
    new_meta: dict[str, Any] = {"name": name}
    if src_meta_file.is_file():
        try:
            src_meta = json.loads(src_meta_file.read_text(encoding="utf-8"))
            new_meta.update(
                {
                    "description": (
                        (src_meta.get("description") or "")
                        + f" (copied from {src.name})"
                    ).strip(),
                    "targetResolution": src_meta.get("targetResolution"),
                    "triggerWord": src_meta.get("triggerWord"),
                },
            )
        except (OSError, json.JSONDecodeError):
            pass
    new_meta["name"] = name
    (dst / ".dataset_meta.json").write_text(
        json.dumps(new_meta, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    return {
        "ok": True,
        "path": str(dst),
        "files_copied": files_copied,
        "images_copied": images_copied,
        "meta": new_meta,
    }
