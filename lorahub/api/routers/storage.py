"""Storage maintenance: disk usage, archive listing/pruning, HuggingFace cache cleanup.

These endpoints exist so users can keep the box healthy without SSHing in
to run ``df -h`` / ``rm -rf``. All destructive operations require an
explicit POST/DELETE — never deleted via GET — and the archive deletion
guards against absolute paths to keep the blast radius local to
``runs/_archive`` siblings.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class _DirSize:
    bytes: int
    files: int


def _du(path: Path) -> _DirSize:
    """Recursive du-like total. Skips broken symlinks and unreadable nodes."""
    total = 0
    files = 0
    if not path.exists():
        return _DirSize(0, 0)
    for child in path.rglob("*"):
        try:
            if child.is_file() and not child.is_symlink():
                total += child.stat().st_size
                files += 1
        except OSError:
            continue
    return _DirSize(total, files)


def _runs_root() -> Path:
    from lorahub.api.paths import runs_dir  # noqa: PLC0415

    return runs_dir().resolve()


def _archive_root() -> Path:
    return _runs_root() / "_archive"


def _resolve_archive_entry(name: str) -> Path:
    """Resolve a user-supplied archive entry name against `runs/_archive`.

    Rejects empty names, names containing path separators, names that
    resolve outside the archive root (symlink escape, `..` traversal), or
    names that do not exist on disk.
    """
    if not name or any(sep in name for sep in ("/", "\\", "..")):
        raise HTTPException(status_code=400, detail="invalid archive entry name")
    root = _archive_root().resolve()
    target = (root / name).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail="archive entry escapes archive root"
        ) from exc
    if not target.exists():
        raise HTTPException(status_code=404, detail="archive entry not found")
    return target


def _hf_cache_root() -> Path | None:
    """Best-effort HuggingFace cache directory lookup.

    Honors HF_HOME / HUGGINGFACE_HUB_CACHE if set; falls back to the
    huggingface_hub library's own constant if importable; finally to the
    historical default `~/.cache/huggingface/hub`. Returns None when none
    of the candidates exist on disk.
    """
    import os  # noqa: PLC0415

    candidates: list[Path] = []
    if hub_cache := os.environ.get("HUGGINGFACE_HUB_CACHE"):
        candidates.append(Path(hub_cache))
    if hf_home := os.environ.get("HF_HOME"):
        candidates.append(Path(hf_home) / "hub")
    try:
        from huggingface_hub.constants import HF_HUB_CACHE  # noqa: PLC0415

        candidates.append(Path(HF_HUB_CACHE))
    except Exception:  # noqa: BLE001
        pass
    candidates.append(Path.home() / ".cache" / "huggingface" / "hub")
    for c in candidates:
        try:
            if c.is_dir():
                return c.resolve()
        except OSError:
            continue
    return None


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #


@router.get("/storage/usage")
def storage_usage() -> dict[str, Any]:
    """High-level disk usage snapshot for the directories lorahub owns.

    Returns the host filesystem free/total for cwd, plus per-folder
    recursive sizes for `runs/`, `runs/_archive/`, `models/`, and the
    HuggingFace hub cache (if any). Numbers are bytes.
    """
    cwd = Path.cwd().resolve()
    fs = shutil.disk_usage(str(cwd))

    runs = _runs_root()
    archive = _archive_root()
    models = (cwd / "models").resolve()
    hf = _hf_cache_root()

    return {
        "filesystem": {
            "path": str(cwd),
            "total_bytes": fs.total,
            "used_bytes": fs.used,
            "free_bytes": fs.free,
        },
        "directories": {
            "runs": _dir_payload(runs),
            "runs_archive": _dir_payload(archive),
            "models": _dir_payload(models),
            "huggingface_cache": _dir_payload(hf) if hf else None,
        },
    }


def _dir_payload(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"path": str(path) if path else None, "exists": False, "bytes": 0, "files": 0}
    size = _du(path)
    return {
        "path": str(path),
        "exists": True,
        "bytes": size.bytes,
        "files": size.files,
    }


@router.get("/storage/archive")
def storage_list_archive() -> dict[str, Any]:
    """List every directory under `runs/_archive/` with its on-disk size."""
    archive = _archive_root()
    if not archive.is_dir():
        return {"archive_root": str(archive), "entries": []}
    entries: list[dict[str, Any]] = []
    for child in sorted(archive.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        size = _du(child)
        try:
            mtime = child.stat().st_mtime
        except OSError:
            mtime = 0.0
        entries.append(
            {
                "name": child.name,
                "path": str(child),
                "bytes": size.bytes,
                "files": size.files,
                "mtime": mtime,
            }
        )
    return {"archive_root": str(archive), "entries": entries}


@router.delete("/storage/archive/{name}")
def storage_delete_archive_entry(name: str) -> dict[str, Any]:
    """Permanently remove a single archived workspace from `runs/_archive/`."""
    target = _resolve_archive_entry(name)
    size = _du(target)
    try:
        shutil.rmtree(target)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"deleted": str(target), "bytes_freed": size.bytes, "files_removed": size.files}


@router.delete("/storage/archive")
def storage_clear_archive() -> dict[str, Any]:
    """Wipe every archived workspace at once. Returns the freed-byte total."""
    archive = _archive_root()
    if not archive.is_dir():
        return {"deleted": [], "bytes_freed": 0, "files_removed": 0}
    deleted: list[str] = []
    total_bytes = 0
    total_files = 0
    failures: list[dict[str, str]] = []
    for child in list(archive.iterdir()):
        if not child.is_dir():
            continue
        size = _du(child)
        try:
            shutil.rmtree(child)
            deleted.append(child.name)
            total_bytes += size.bytes
            total_files += size.files
        except OSError as exc:
            failures.append({"name": child.name, "error": str(exc)})
    return {
        "deleted": deleted,
        "bytes_freed": total_bytes,
        "files_removed": total_files,
        "failures": failures,
    }


@router.delete("/storage/hf-cache")
def storage_clear_hf_cache() -> dict[str, Any]:
    """Wipe the HuggingFace hub cache directory.

    Resolves HF_HUB_CACHE / HF_HOME / the default `~/.cache/huggingface/hub`.
    Returns 404 when no cache directory could be located. Refuses to delete
    if the resolved cache path is `/` or the user's home directory itself
    (sanity guard).
    """
    cache = _hf_cache_root()
    if cache is None:
        raise HTTPException(status_code=404, detail="huggingface cache not found")
    if cache == Path("/") or cache == Path.home():
        raise HTTPException(status_code=400, detail="refusing to wipe root or home")
    size = _du(cache)
    try:
        shutil.rmtree(cache)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "deleted": str(cache),
        "bytes_freed": size.bytes,
        "files_removed": size.files,
    }
