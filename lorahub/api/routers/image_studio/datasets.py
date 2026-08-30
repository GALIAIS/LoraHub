"""Image Studio dataset management endpoints (CRUD + upload extraction)."""

from __future__ import annotations

import asyncio
import contextlib
import json as json_stdlib
import logging
import os
import shutil
import tempfile
import threading
from collections.abc import AsyncIterator
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Literal

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from lorahub.api import paths as api_paths
from lorahub.api.dataset_files import (
    DATASET_META_FILENAME,
    IMAGE_SUFFIXES,
    is_link_like,
    resolve_file_under,
)

from ._shared import (
    _clear_dataset_view_caches,
    _safe_runs_subdir,
    _sse_event,
    _unique_path,
)

router = APIRouter(prefix="/api/image-studio", tags=["image-studio"])


_ARCHIVE_SUFFIXES = {".zip", ".tar", ".gz", ".tgz", ".7z"}
_CAPTION_SUFFIXES = {".txt", ".caption"}
_UPLOAD_CHUNK_BYTES = 1024 * 1024
_MAX_UPLOAD_BYTES = 50 * 1024**3
_MAX_UPLOAD_REQUEST_BYTES = 100 * 1024**3
_MAX_EXTRACT_BYTES = 200 * 1024**3
_MAX_ARCHIVE_ENTRIES = 100_000
_MIN_FREE_BYTES = 512 * 1024**2
_WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}
_WINDOWS_INVALID_CHARS = frozenset('<>:"|?*')

log = logging.getLogger(__name__)
_ACTIVE_UPLOADS: set[str] = set()
_ACTIVE_UPLOADS_LOCK = threading.Lock()


def _env_limit(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


async def _write_upload(upload: UploadFile, target: Path, *, limit: int | None = None) -> int:
    file_limit = _env_limit("LORAHUB_MAX_UPLOAD_BYTES", _MAX_UPLOAD_BYTES)
    effective_limit = min(file_limit, limit) if limit is not None else file_limit
    written = 0
    try:
        with target.open("wb") as dst:
            while chunk := await upload.read(_UPLOAD_CHUNK_BYTES):
                written += len(chunk)
                if written > effective_limit:
                    raise HTTPException(413, f"upload exceeds {effective_limit} bytes")
                dst.write(chunk)
            dst.flush()
            os.fsync(dst.fileno())
    except Exception:
        target.unlink(missing_ok=True)
        raise
    return written


def _validate_archive_budget(
    entries: list[tuple[str, int]],
    dest: Path,
    *,
    disk_copies: int = 1,
) -> None:
    max_entries = _env_limit("LORAHUB_MAX_ARCHIVE_ENTRIES", _MAX_ARCHIVE_ENTRIES)
    max_bytes = _env_limit("LORAHUB_MAX_EXTRACT_BYTES", _MAX_EXTRACT_BYTES)
    if len(entries) > max_entries:
        raise ValueError(f"archive has {len(entries)} entries; limit is {max_entries}")
    declared = sum(max(0, size) for _name, size in entries)
    if declared > max_bytes:
        raise ValueError(f"archive expands to {declared} bytes; limit is {max_bytes}")
    required = declared * max(1, disk_copies)
    free = shutil.disk_usage(dest).free
    if required + _MIN_FREE_BYTES > free:
        raise ValueError(
            f"archive needs {required} bytes of working space but only {free} bytes are free"
        )


def _datasets_root() -> Path:
    """Return the configured datasets root directory."""
    extra = os.environ.get("LORAHUB_DATASETS_ROOT")
    if extra:
        root = Path(extra.split(os.pathsep)[0].strip())
    else:
        root = api_paths.project_root() / "datasets"
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
        or not _safe_file_name(canonical)
    ):
        raise HTTPException(400, "invalid dataset name")
    return canonical


def _dataset_path_by_name(name: str) -> Path:
    root = _datasets_root()
    canonical = _validate_dataset_name(name)
    lexical = root / canonical
    if is_link_like(lexical):
        raise HTTPException(400, "dataset path cannot be a link")
    ds_path = lexical.resolve()
    try:
        ds_path.relative_to(root)
    except ValueError as exc:
        raise HTTPException(400, "invalid dataset name") from exc
    if ds_path == root:
        raise HTTPException(400, "invalid dataset name")
    return ds_path


def _normalised_dataset_key(path: Path) -> str:
    return os.path.normcase(str(path.resolve(strict=False)))


def _resolve_runtime_path(raw: object) -> Path | None:
    if not isinstance(raw, (str, os.PathLike)) or not str(raw).strip():
        return None
    try:
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = api_paths.project_root() / candidate
        return candidate.resolve(strict=False)
    except (OSError, RuntimeError, TypeError, ValueError):
        return None


def _paths_overlap(left: Path, right: Path) -> bool:
    try:
        left.relative_to(right)
        return True
    except ValueError:
        pass
    try:
        right.relative_to(left)
        return True
    except ValueError:
        return False


def _training_dataset_paths(snapshot: dict[str, Any]) -> list[Path]:
    dataset = snapshot.get("dataset")
    if not isinstance(dataset, dict):
        return []
    path_keys = {
        "source",
        "path",
        "mask_path",
        "maskPath",
        "conditioning_data_dir",
        "conditioningDataDir",
        "conditioning_dir",
        "conditioningDir",
        "reg_source",
        "regSource",
    }
    paths: list[Path] = []

    def visit(value: object, key: str | None = None) -> None:
        if isinstance(value, dict):
            for child_key, child in value.items():
                visit(child, str(child_key))
            return
        if isinstance(value, list):
            for child in value:
                visit(child, key)
            return
        if key not in path_keys:
            return
        resolved = _resolve_runtime_path(value)
        if resolved is not None:
            paths.append(resolved)

    visit(dataset)
    return paths


def _active_dataset_users(ds_path: Path) -> list[str]:
    """Return active work that would lose its source if the dataset moved."""
    users: list[str] = []
    key = _normalised_dataset_key(ds_path)
    if key in _ACTIVE_UPLOADS:
        users.append("upload")

    from lorahub.api import app as app_module  # noqa: PLC0415
    from lorahub.api import state  # noqa: PLC0415

    task_store = getattr(app_module, "_task_session_store", None)
    if task_store is not None:
        try:
            tasks = task_store.list_active()
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "cannot inspect active image-studio tasks before dataset delete: %s",
                type(exc).__name__,
            )
            users.append("task-registry-unavailable")
        else:
            for task in tasks:
                for field in ("path", "dataset_path"):
                    target = _resolve_runtime_path(task.metadata.get(field))
                    if target is not None and _paths_overlap(target, ds_path):
                        users.append(f"task:{task.id}")
                        break

    live_states = {"queued", "preparing", "running", "canceling"}
    for job in state.registry.list():
        if job.state.value not in live_states:
            continue
        if any(
            _paths_overlap(path, ds_path)
            for path in _training_dataset_paths(job.config_snapshot)
        ):
            users.append(f"training:{job.id}")
    return users


def _read_dataset_meta(ds_path: Path) -> dict[str, Any]:
    meta_file = ds_path / DATASET_META_FILENAME
    if meta_file.is_file():
        try:
            value = json_stdlib.loads(meta_file.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, json_stdlib.JSONDecodeError) as exc:
            log.warning("cannot read dataset metadata %s: %s", meta_file, exc)
    return {}


def _write_dataset_meta(ds_path: Path, meta: dict[str, Any]) -> None:
    meta_file = ds_path / DATASET_META_FILENAME
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        dir=ds_path,
        prefix=".dataset-meta-",
        suffix=".tmp",
    ) as handle:
        temp_path = Path(handle.name)
        handle.write(json_stdlib.dumps(meta, ensure_ascii=False, indent=2))
    try:
        temp_path.replace(meta_file)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _count_images(ds_path: Path) -> int:
    return sum(
        1
        for path in ds_path.iterdir()
        if path.suffix.lower() in IMAGE_SUFFIXES
        and resolve_file_under(ds_path, path) is not None
    )


def _dataset_cover(ds_path: Path) -> str | None:
    """Return the first image path as cover thumbnail."""
    for p in sorted(ds_path.iterdir(), key=lambda x: x.name.lower()):
        if p.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        resolved = resolve_file_under(ds_path, p)
        if resolved is not None:
            return str(resolved)
    return None


@router.get("/datasets")
def list_datasets() -> dict[str, Any]:
    """List all datasets under the datasets root."""
    root = _datasets_root()
    datasets: list[dict[str, Any]] = []
    for entry in sorted(root.iterdir(), key=lambda x: x.name.lower()):
        if is_link_like(entry) or not entry.is_dir():
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
    ds_path.mkdir()
    meta = {
        "name": name,
        "description": body.description,
        "targetResolution": body.targetResolution,
        "triggerWord": body.triggerWord,
    }
    try:
        _write_dataset_meta(ds_path, meta)
    except Exception:
        with contextlib.suppress(OSError):
            ds_path.rmdir()
        raise
    _clear_dataset_view_caches(ds_path)
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
    _clear_dataset_view_caches(ds_path)
    return {"ok": True, "meta": meta}


@router.delete("/datasets/{name}")
def delete_dataset(name: str) -> dict[str, Any]:
    """Move dataset to trash (not permanent delete)."""
    canonical = _validate_dataset_name(name)
    ds_path = _dataset_path_by_name(canonical)
    if not ds_path.is_dir():
        raise HTTPException(404, "dataset not found")
    from datetime import UTC, datetime  # noqa: PLC0415

    # Share the upload registry lock through the final move so an upload
    # cannot claim the dataset between the idle check and the rename.
    with _ACTIVE_UPLOADS_LOCK:
        active = _active_dataset_users(ds_path)
        if active:
            raise HTTPException(
                409,
                "dataset is in use: " + ", ".join(active[:8]),
            )
        trash = _safe_runs_subdir(
            "_dataset_trash",
            datetime.now(UTC).strftime("%Y-%m-%d"),
        )
        shutil.move(str(ds_path), str(_unique_path(trash / canonical)))
    _clear_dataset_view_caches(ds_path)
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


def _safe_file_name(value: str) -> bool:
    if (
        not value
        or value in {".", ".."}
        or value[-1:] in {" ", "."}
        or any(ord(char) < 32 or char in _WINDOWS_INVALID_CHARS for char in value)
        or len(value.encode("utf-8", errors="surrogatepass")) > 240
    ):
        return False
    stem = value.split(".", 1)[0].upper()
    return stem not in _WINDOWS_RESERVED_NAMES


def _safe_upload_target(dest: Path, filename: str) -> Path | None:
    if is_link_like(dest):
        return None
    resolved_dest = dest.resolve()
    basename = _upload_basename(filename)
    if not _safe_file_name(basename):
        return None
    lexical = dest / basename
    if is_link_like(lexical):
        return None
    target = lexical.resolve()
    try:
        target.relative_to(resolved_dest)
    except ValueError:
        return None
    return target


def _archive_member_parts(filename: str) -> tuple[str, ...] | None:
    normalised = filename.replace("\\", "/")
    member = PurePosixPath(normalised)
    if member.is_absolute() or PureWindowsPath(filename).drive:
        return None
    if not member.parts or any(
        part in {"", ".", ".."} or not _safe_file_name(part)
        for part in member.parts
    ):
        return None
    return member.parts


def _safe_staged_member(stage: Path, filename: str) -> Path | None:
    """Resolve an archive member in a staging tree without flattening it."""
    parts = _archive_member_parts(filename)
    if parts is None:
        return None
    target = stage.joinpath(*parts).resolve()
    try:
        target.relative_to(stage.resolve())
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
    actual_bytes = 0
    max_bytes = _env_limit("LORAHUB_MAX_EXTRACT_BYTES", _MAX_EXTRACT_BYTES)
    suffix = archive_path.suffix.lower()
    name_lower = archive_path.name.lower()

    def copy_member(src: Any, target: Path) -> None:
        nonlocal actual_bytes
        with tempfile.NamedTemporaryFile(
            delete=False,
            dir=target.parent,
            prefix=".lorahub-extract-",
            suffix=".part",
        ) as handle:
            temp_path = Path(handle.name)
        try:
            with temp_path.open("wb") as dst:
                while chunk := src.read(_UPLOAD_CHUNK_BYTES):
                    actual_bytes += len(chunk)
                    if actual_bytes > max_bytes:
                        raise ValueError(
                            f"archive exceeds extraction limit of {max_bytes} bytes"
                        )
                    dst.write(chunk)
                dst.flush()
                os.fsync(dst.fileno())
            temp_path.replace(target)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

    if suffix == ".zip":
        try:
            with zipfile.ZipFile(archive_path, "r") as zf:
                infos = zf.infolist()
                _validate_archive_budget(
                    [(info.filename, info.file_size) for info in infos], dest
                )
                for info in infos:
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
                        with zf.open(info) as src:
                            copy_member(src, target)
                        extracted += 1
        except (ValueError, zipfile.BadZipFile, OSError) as e:
            errors.append(f"bad zip: {e}")
    elif suffix in (".tar", ".gz", ".tgz") or name_lower.endswith(".tar.gz"):
        try:
            mode: Literal["r:gz", "r"] = (
                "r:gz"
                if suffix in (".gz", ".tgz")
                or name_lower.endswith(".tar.gz")
                else "r"
            )
            with tarfile.open(archive_path, mode) as tf:
                members = tf.getmembers()
                _validate_archive_budget(
                    [(member.name, member.size) for member in members], dest
                )
                for member in members:
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
                        with f:
                            copy_member(f, target)
                        extracted += 1
        except (ValueError, tarfile.TarError, OSError) as e:
            errors.append(f"tar error: {e}")
    elif suffix == ".7z":
        try:
            import py7zr  # noqa: PLC0415
            with py7zr.SevenZipFile(archive_path, "r") as sz:
                infos = sz.list()
                _validate_archive_budget(
                    [
                        (str(info.filename), int(getattr(info, "uncompressed", 0) or 0))
                        for info in infos
                    ],
                    dest,
                    disk_copies=2,
                )
                names: list[str] = []
                for info in infos:
                    fname = str(info.filename)
                    if bool(getattr(info, "is_directory", False)):
                        continue
                    if bool(getattr(info, "is_symlink", False)):
                        raise ValueError(f"7z links are not supported: {fname}")
                    output_target = _safe_upload_target(dest, fname)
                    if output_target is None:
                        continue
                    if not _is_image_file(output_target.name) and not (
                        keep_captions and _is_caption_file(output_target.name)
                    ):
                        continue
                    if _archive_member_parts(fname) is None:
                        raise ValueError(f"unsafe 7z member path: {fname}")
                    names.append(fname)
                with tempfile.TemporaryDirectory(
                    prefix=".lorahub-7z-",
                    dir=dest.parent,
                ) as stage_raw:
                    stage = Path(stage_raw).resolve()
                    sz.extract(path=stage, targets=names)
                    for fname in names:
                        staged = _safe_staged_member(stage, fname)
                        if staged is None or is_link_like(staged) or not staged.is_file():
                            continue
                        target = _safe_upload_target(dest, fname)
                        if target is None:
                            continue
                        target = _resolve_conflict(target, on_conflict)
                        if target is None:
                            continue
                        with staged.open("rb") as src:
                            copy_member(src, target)
                        extracted += 1
        except Exception as e:  # noqa: BLE001
            errors.append(f"7z error: {e}")
    else:
        errors.append(f"unsupported archive format: {suffix}")

    return extracted, errors


def _resolve_conflict(target: Path, strategy: str) -> Path | None:
    """Handle file name conflicts. Returns final path or None to skip."""
    if is_link_like(target):
        return None
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
        if not candidate.exists() and not is_link_like(candidate):
            return candidate
        i += 1


def _claim_dataset_upload(ds_path: Path) -> bool:
    key = _normalised_dataset_key(ds_path)
    with _ACTIVE_UPLOADS_LOCK:
        if key in _ACTIVE_UPLOADS:
            return False
        _ACTIVE_UPLOADS.add(key)
        return True


def _release_dataset_upload(ds_path: Path) -> None:
    key = _normalised_dataset_key(ds_path)
    with _ACTIVE_UPLOADS_LOCK:
        _ACTIVE_UPLOADS.discard(key)


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
    if onConflict not in {"skip", "overwrite", "rename"}:
        raise HTTPException(400, "invalid conflict strategy")
    if len(files) > _MAX_ARCHIVE_ENTRIES:
        raise HTTPException(413, "too many uploaded files")
    upload_limit = _env_limit("LORAHUB_MAX_UPLOAD_BYTES", _MAX_UPLOAD_BYTES)
    request_limit = _env_limit(
        "LORAHUB_MAX_UPLOAD_REQUEST_BYTES",
        _MAX_UPLOAD_REQUEST_BYTES,
    )
    if any(upload.size is not None and upload.size > upload_limit for upload in files):
        raise HTTPException(413, f"upload exceeds {upload_limit} bytes")
    known_total = sum(upload.size or 0 for upload in files)
    if known_total > request_limit:
        raise HTTPException(413, f"upload request exceeds {request_limit} bytes")
    if not _claim_dataset_upload(ds_path):
        raise HTTPException(409, "another upload is already writing this dataset")

    extract_task: asyncio.Task[tuple[int, list[str]]] | None = None

    async def _extract_uploaded_archive(archive_path: Path) -> tuple[int, list[str]]:
        """Keep the staged upload alive until its worker has stopped reading it."""
        try:
            return await asyncio.to_thread(
                _extract_archive,
                archive_path,
                ds_path,
                keepCaptions,
                onConflict,
            )
        finally:
            archive_path.unlink(missing_ok=True)

    def _release_after_extract(
        task: asyncio.Task[tuple[int, list[str]]],
    ) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception:  # noqa: BLE001
            log.exception("dataset archive extraction failed after client disconnect")
        finally:
            with contextlib.suppress(Exception):
                _clear_dataset_view_caches(ds_path)
            _release_dataset_upload(ds_path)

    async def _generate_inner() -> AsyncIterator[str]:
        nonlocal extract_task
        total_files = len(files)
        total_extracted = 0
        total_uploaded = 0
        all_errors: list[str] = []

        for idx, upload in enumerate(files):
            filename = upload.filename or "unknown"
            yield _sse_event("progress", {
                "file": filename,
                "index": idx,
                "total": total_files,
                "status": "processing",
            })

            try:
                remaining = request_limit - total_uploaded
                if remaining <= 0:
                    raise HTTPException(413, f"upload request exceeds {request_limit} bytes")

                if _is_archive(filename):
                    # Keep the upload and 7z staging on the dataset volume so
                    # free-space validation describes the disk actually used.
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        dir=ds_path.parent,
                        prefix=".lorahub-upload-",
                        suffix=Path(filename).suffix,
                    ) as tmp:
                        temp_path = Path(tmp.name)
                    task_owns_temp = False
                    try:
                        total_uploaded += await _write_upload(
                            upload,
                            temp_path,
                            limit=remaining,
                        )
                        extract_task = asyncio.create_task(
                            _extract_uploaded_archive(temp_path)
                        )
                        task_owns_temp = True
                        count, errs = await asyncio.shield(extract_task)
                        extract_task = None
                        total_extracted += count
                        all_errors.extend(errs)
                        yield _sse_event("extracted", {
                            "file": filename,
                            "count": count,
                            "errors": errs,
                        })
                    finally:
                        if not task_owns_temp:
                            temp_path.unlink(missing_ok=True)
                elif _is_image_file(filename) or (
                    keepCaptions and _is_caption_file(filename)
                ):
                    target = _safe_upload_target(ds_path, filename)
                    if target is None:
                        all_errors.append(f"skipped unsafe filename: {filename}")
                    else:
                        target = _resolve_conflict(target, onConflict)
                        if target is not None:
                            with tempfile.NamedTemporaryFile(
                                delete=False,
                                dir=target.parent,
                                prefix=".lorahub-upload-",
                                suffix=".part",
                            ) as tmp:
                                temp_path = Path(tmp.name)
                            try:
                                total_uploaded += await _write_upload(
                                    upload,
                                    temp_path,
                                    limit=remaining,
                                )
                                temp_path.replace(target)
                            finally:
                                temp_path.unlink(missing_ok=True)
                            total_extracted += 1
                else:
                    all_errors.append(f"skipped non-image: {filename}")
            except HTTPException as exc:
                message = str(exc.detail)
                all_errors.append(message)
                yield _sse_event("error", {
                    "file": filename,
                    "message": message,
                    "fatal": True,
                })
                yield _sse_event("complete", {
                    "totalExtracted": total_extracted,
                    "errors": all_errors,
                    "fatal": True,
                    "message": message,
                })
                return
            except Exception:  # noqa: BLE001
                log.exception("dataset upload failed for %s", filename)
                message = f"upload failed while processing {filename}"
                all_errors.append(message)
                yield _sse_event("error", {
                    "file": filename,
                    "message": message,
                    "fatal": True,
                })
                yield _sse_event("complete", {
                    "totalExtracted": total_extracted,
                    "errors": all_errors,
                    "fatal": True,
                    "message": message,
                })
                return
            finally:
                await upload.close()

            yield _sse_event("progress", {
                "file": filename,
                "index": idx + 1,
                "total": total_files,
                "status": "done",
            })
            await asyncio.sleep(0)

        _clear_dataset_view_caches(ds_path)
        yield _sse_event("complete", {
            "totalExtracted": total_extracted,
            "errors": all_errors,
        })

    async def _generate() -> AsyncIterator[str]:
        nonlocal extract_task
        release_deferred = False
        try:
            async for event in _generate_inner():
                yield event
        finally:
            for upload in files:
                with contextlib.suppress(Exception):
                    await upload.close()
            if extract_task is not None:
                # ``asyncio.to_thread`` keeps running after the response task
                # is cancelled. Keep the dataset claimed until that worker
                # exits so delete or another upload cannot race its writes.
                extract_task.add_done_callback(_release_after_extract)
                release_deferred = True
            if not release_deferred:
                with contextlib.suppress(Exception):
                    _clear_dataset_view_caches(ds_path)
                _release_dataset_upload(ds_path)

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
