"""Image Studio dataset management endpoints (CRUD + upload extraction)."""

from __future__ import annotations

import json as json_stdlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from lorahub.api.dataset_files import IMAGE_SUFFIXES

from ._shared import _sse_event

router = APIRouter(prefix="/api/image-studio", tags=["image-studio"])


_DATASET_META_FILE = "dataset.json"
_ARCHIVE_SUFFIXES = {".zip", ".tar", ".gz", ".tgz", ".7z", ".rar"}
_CAPTION_SUFFIXES = {".txt", ".caption"}


def _datasets_root() -> Path:
    """Return the configured datasets root directory."""
    extra = os.environ.get("LORAHUB_DATASETS_ROOT")
    if extra:
        root = Path(extra.split(os.pathsep)[0].strip())
    else:
        root = Path.cwd() / "datasets"
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _validate_dataset_name(name: str) -> str:
    """Return the canonical dataset name, rejecting traversal-like values."""
    canonical = name.strip()
    if (
        not canonical
        or canonical in {".", ".."}
        or "/" in canonical
        or "\\" in canonical
        or ".." in canonical
    ):
        raise HTTPException(400, "invalid dataset name")
    return canonical


def _dataset_path_by_name(name: str) -> Path:
    root = _datasets_root()
    canonical = _validate_dataset_name(name)
    ds_path = (root / canonical).resolve()
    try:
        ds_path.relative_to(root)
    except ValueError as exc:
        raise HTTPException(400, "invalid dataset name") from exc
    if ds_path == root:
        raise HTTPException(400, "invalid dataset name")
    return ds_path


def _read_dataset_meta(ds_path: Path) -> dict[str, Any]:
    meta_file = ds_path / _DATASET_META_FILE
    if meta_file.is_file():
        return json_stdlib.loads(meta_file.read_text(encoding="utf-8"))
    return {}


def _write_dataset_meta(ds_path: Path, meta: dict[str, Any]) -> None:
    meta_file = ds_path / _DATASET_META_FILE
    meta_file.write_text(
        json_stdlib.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _count_images(ds_path: Path) -> int:
    count = 0
    for p in ds_path.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            count += 1
    return count


def _dataset_cover(ds_path: Path) -> str | None:
    """Return the first image path as cover thumbnail."""
    for p in sorted(ds_path.iterdir(), key=lambda x: x.name.lower()):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            return str(p)
    return None


@router.get("/datasets")
def list_datasets() -> dict[str, Any]:
    """List all datasets under the datasets root."""
    root = _datasets_root()
    datasets: list[dict[str, Any]] = []
    for entry in sorted(root.iterdir(), key=lambda x: x.name.lower()):
        if not entry.is_dir():
            continue
        if entry.name.startswith("."):
            continue
        meta = _read_dataset_meta(entry)
        image_count = _count_images(entry)
        cover = _dataset_cover(entry)
        datasets.append({
            "name": entry.name,
            "path": str(entry),
            "imageCount": image_count,
            "coverPath": cover,
            "coverUrl": (
                f"/api/datasets/thumb?path={cover}&size=256" if cover else None
            ),
            "meta": meta,
        })
    return {"root": str(root), "datasets": datasets}

class CreateDatasetInput(BaseModel):
    name: str
    description: str = ""
    targetResolution: str = ""
    triggerWord: str = ""


@router.post("/datasets")
def create_dataset(body: CreateDatasetInput) -> dict[str, Any]:
    """Create a new dataset directory with metadata."""
    name = _validate_dataset_name(body.name)
    ds_path = _dataset_path_by_name(name)
    if ds_path.exists():
        raise HTTPException(409, f"dataset '{name}' already exists")
    ds_path.mkdir(parents=True)
    meta = {
        "name": name,
        "description": body.description,
        "targetResolution": body.targetResolution,
        "triggerWord": body.triggerWord,
    }
    _write_dataset_meta(ds_path, meta)
    return {"ok": True, "path": str(ds_path), "meta": meta}


@router.get("/datasets/{name}/meta")
def get_dataset_meta(name: str) -> dict[str, Any]:
    ds_path = _dataset_path_by_name(name)
    if not ds_path.is_dir():
        raise HTTPException(404, "dataset not found")
    return _read_dataset_meta(ds_path)


class UpdateDatasetMetaInput(BaseModel):
    description: str | None = None
    targetResolution: str | None = None
    triggerWord: str | None = None


@router.put("/datasets/{name}/meta")
def update_dataset_meta(name: str, body: UpdateDatasetMetaInput) -> dict[str, Any]:
    ds_path = _dataset_path_by_name(name)
    if not ds_path.is_dir():
        raise HTTPException(404, "dataset not found")
    meta = _read_dataset_meta(ds_path)
    if body.description is not None:
        meta["description"] = body.description
    if body.targetResolution is not None:
        meta["targetResolution"] = body.targetResolution
    if body.triggerWord is not None:
        meta["triggerWord"] = body.triggerWord
    _write_dataset_meta(ds_path, meta)
    return {"ok": True, "meta": meta}


@router.delete("/datasets/{name}")
def delete_dataset(name: str) -> dict[str, Any]:
    """Move dataset to trash (not permanent delete)."""
    canonical = _validate_dataset_name(name)
    ds_path = _dataset_path_by_name(canonical)
    if not ds_path.is_dir():
        raise HTTPException(404, "dataset not found")
    from datetime import UTC, datetime  # noqa: PLC0415
    trash = Path("runs") / "_dataset_trash" / datetime.now(UTC).strftime("%Y-%m-%d")
    trash.mkdir(parents=True, exist_ok=True)
    shutil.move(str(ds_path), str(trash / canonical))
    return {"ok": True}


# --------------------------------------------------------------------------- #
# Upload + extract
# --------------------------------------------------------------------------- #


def _is_archive(filename: str) -> bool:
    lower = filename.lower()
    return any(lower.endswith(s) for s in _ARCHIVE_SUFFIXES)


def _is_image_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in IMAGE_SUFFIXES


def _is_caption_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in _CAPTION_SUFFIXES


def _upload_basename(filename: str) -> str:
    return Path(filename.replace("\\", "/")).name


def _safe_upload_target(dest: Path, filename: str) -> Path | None:
    basename = _upload_basename(filename)
    if not basename:
        return None
    target = (dest / basename).resolve()
    try:
        target.relative_to(dest.resolve())
    except ValueError:
        return None
    return target


def _extract_archive(
    archive_path: Path,
    dest: Path,
    keep_captions: bool,
    on_conflict: str,
) -> tuple[int, list[str]]:
    """Extract images from archive. Returns (count, errors)."""
    import tarfile  # noqa: PLC0415
    import zipfile  # noqa: PLC0415

    extracted = 0
    errors: list[str] = []
    suffix = archive_path.suffix.lower()
    name_lower = archive_path.name.lower()

    if suffix == ".zip":
        try:
            with zipfile.ZipFile(archive_path, "r") as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    target = _safe_upload_target(dest, info.filename)
                    if target is None:
                        continue
                    if _is_image_file(target.name) or (
                        keep_captions and _is_caption_file(target.name)
                    ):
                        target = _resolve_conflict(target, on_conflict)
                        if target is None:
                            continue
                        with zf.open(info) as src, open(target, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        extracted += 1
        except zipfile.BadZipFile as e:
            errors.append(f"bad zip: {e}")
    elif suffix in (".tar", ".gz", ".tgz") or name_lower.endswith(".tar.gz"):
        try:
            mode = "r:gz" if suffix in (".gz", ".tgz") or name_lower.endswith(".tar.gz") else "r"
            with tarfile.open(archive_path, mode) as tf:
                for member in tf.getmembers():
                    if not member.isfile():
                        continue
                    target = _safe_upload_target(dest, member.name)
                    if target is None:
                        continue
                    if _is_image_file(target.name) or (
                        keep_captions and _is_caption_file(target.name)
                    ):
                        target = _resolve_conflict(target, on_conflict)
                        if target is None:
                            continue
                        f = tf.extractfile(member)
                        if f is None:
                            continue
                        with open(target, "wb") as dst:
                            shutil.copyfileobj(f, dst)
                        extracted += 1
        except (tarfile.TarError, OSError) as e:
            errors.append(f"tar error: {e}")
    elif suffix == ".7z":
        try:
            import py7zr  # noqa: PLC0415
            with py7zr.SevenZipFile(archive_path, "r") as sz:
                for fname, bio in sz.read().items():
                    target = _safe_upload_target(dest, fname)
                    if target is None:
                        continue
                    if _is_image_file(target.name) or (
                        keep_captions and _is_caption_file(target.name)
                    ):
                        target = _resolve_conflict(target, on_conflict)
                        if target is None:
                            continue
                        with open(target, "wb") as dst:
                            dst.write(bio.read())
                        extracted += 1
        except Exception as e:  # noqa: BLE001
            errors.append(f"7z error: {e}")
    else:
        errors.append(f"unsupported archive format: {suffix}")

    return extracted, errors


def _resolve_conflict(target: Path, strategy: str) -> Path | None:
    """Handle file name conflicts. Returns final path or None to skip."""
    if not target.exists():
        return target
    if strategy == "skip":
        return None
    if strategy == "overwrite":
        return target
    # rename: append _1, _2, etc.
    stem = target.stem
    suffix = target.suffix
    parent = target.parent
    i = 1
    while True:
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


@router.post("/datasets/{name}/upload")
async def upload_to_dataset(
    name: str,
    files: list[UploadFile] = File(...),
    keepCaptions: bool = Form(True),
    onConflict: str = Form("rename"),
) -> StreamingResponse:
    """Upload files to a dataset. Supports images and archives.

    Returns SSE stream with progress events.
    """
    ds_path = _dataset_path_by_name(name)
    if not ds_path.is_dir():
        raise HTTPException(404, "dataset not found")

    import asyncio  # noqa: PLC0415

    async def _generate():
        total_files = len(files)
        total_extracted = 0
        all_errors: list[str] = []

        for idx, upload in enumerate(files):
            filename = upload.filename or "unknown"
            yield _sse_event("progress", {
                "file": filename,
                "index": idx,
                "total": total_files,
                "status": "processing",
            })

            if _is_archive(filename):
                # Save to temp, then extract
                tmp = tempfile.NamedTemporaryFile(
                    delete=False, suffix=Path(filename).suffix
                )
                try:
                    content = await upload.read()
                    tmp.write(content)
                    tmp.close()
                    count, errs = _extract_archive(
                        Path(tmp.name), ds_path, keepCaptions, onConflict
                    )
                    total_extracted += count
                    all_errors.extend(errs)
                    yield _sse_event("extracted", {
                        "file": filename,
                        "count": count,
                        "errors": errs,
                    })
                finally:
                    os.unlink(tmp.name)
            elif _is_image_file(filename) or (keepCaptions and _is_caption_file(filename)):
                target = _safe_upload_target(ds_path, filename)
                if target is None:
                    all_errors.append(f"skipped unsafe filename: {filename}")
                    continue
                target = _resolve_conflict(target, onConflict)
                if target is not None:
                    content = await upload.read()
                    target.write_bytes(content)
                    total_extracted += 1
            else:
                all_errors.append(f"skipped non-image: {filename}")

            yield _sse_event("progress", {
                "file": filename,
                "index": idx + 1,
                "total": total_files,
                "status": "done",
            })
            await asyncio.sleep(0)

        yield _sse_event("complete", {
            "totalExtracted": total_extracted,
            "errors": all_errors,
        })

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
