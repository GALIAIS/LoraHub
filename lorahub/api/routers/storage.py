"""Storage maintenance: disk usage, archive listing/pruning, HuggingFace cache cleanup.

These endpoints exist so users can keep the box healthy without SSHing in
to run ``df -h`` / ``rm -rf``. All destructive operations require an
explicit POST/DELETE — never deleted via GET — and the archive deletion
guards against absolute paths to keep the blast radius local to
``runs/_archive`` siblings.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _is_link_path(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        is_junction = getattr(path, "is_junction", None)
        return bool(is_junction and is_junction())
    except OSError:
        return True


def _remove_link_path(path: Path) -> None:
    if path.is_symlink():
        path.unlink()
    else:
        os.rmdir(path)


def _rmtree(target: Path) -> None:
    """``shutil.rmtree`` with a Windows-friendly only-readable retry hook.

    Git pack files, virtualenv ``python.exe`` shims, and HuggingFace cache
    blobs are written read-only on Windows. Plain ``shutil.rmtree`` raises
    ``PermissionError`` on them and bails out half-deleted. This wrapper
    flips the read-only bit on the offender and retries — same idiom the
    backend installer's ``_remove_target`` uses.
    """

    if _is_link_path(target):
        raise OSError(f"refusing to recursively delete linked directory: {target}")

    def _onerror(func: Any, path: str, _exc_info: Any) -> None:  # noqa: ANN401
        import stat as _stat  # noqa: PLC0415

        try:
            Path(path).chmod(_stat.S_IWRITE | _stat.S_IREAD)
            func(path)
        except OSError:
            pass

    if sys.version_info >= (3, 12):
        shutil.rmtree(target, onexc=_onerror)
    else:
        shutil.rmtree(target, onerror=_onerror)


@dataclass(slots=True)
class _DirSize:
    bytes: int
    files: int


def _du(path: Path) -> _DirSize:
    """Recursive du-like total. Skips broken symlinks and unreadable nodes."""
    total = 0
    files = 0
    if not path.exists() or _is_link_path(path):
        return _DirSize(0, 0)
    for current, dirs, names in os.walk(path, followlinks=False):
        current_path = Path(current)
        dirs[:] = [name for name in dirs if not _is_link_path(current_path / name)]
        for name in names:
            child = current_path / name
            try:
                if child.is_file() and not _is_link_path(child):
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
    target = root / name
    if not target.exists() and not _is_link_path(target):
        raise HTTPException(status_code=404, detail="archive entry not found")
    if _is_link_path(target):
        return target
    resolved = target.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail="archive entry escapes archive root"
        ) from exc
    return resolved


def _hf_cache_root() -> Path | None:
    """Best-effort HuggingFace cache directory lookup.

    Prefer LoraHub's project-local cache. Environment variables are kept as
    fallbacks only for older installs and manual user caches.
    """
    import os  # noqa: PLC0415

    from lorahub.core.paths import project_root  # noqa: PLC0415

    candidates: list[Path] = []
    candidates.append(project_root() / "models" / "huggingface" / "hub")
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
                # Preserve the lexical candidate until the destructive
                # validator can compare it with its resolved target. Resolving
                # here would hide a cache-path symlink to protected data.
                return c.expanduser().absolute()
        except OSError:
            continue
    return None


def _validate_hf_cache_delete_target(raw: Path) -> Path:
    """Reject ambiguous or protected directories before recursive cache removal."""
    from lorahub.api.paths import project_root  # noqa: PLC0415

    requested = raw.expanduser().absolute()
    if _is_link_path(requested):
        raise ValueError("refusing to recursively delete a linked cache directory")
    target = requested.resolve()
    project = project_root().resolve()
    known_project_cache = (project / "models" / "huggingface" / "hub").absolute()
    is_known_project_cache = requested == known_project_cache

    if target.parent == target or target.parent.parent == target.parent:
        raise ValueError("cache path is too close to the filesystem root")

    broad_roots = (Path.home().resolve(), project)
    for item in broad_roots:
        try:
            contains_root = item.relative_to(target) is not None
        except ValueError:
            contains_root = False
        if target == item or contains_root:
            raise ValueError(f"cache path overlaps protected directory: {item}")

    protected = (
        Path.home().resolve() / ".gnupg",
        Path.home().resolve() / ".ssh",
        project / ".git",
        project / ".venv",
        project / ".lorahub",
        project / "configs",
        project / "datasets",
        project / "external",
        project / "lorahub",
        project / "models",
        project / "output",
        project / "runs",
        project / "scripts",
        project / "web",
    )
    for item in protected:
        item = item.resolve()
        try:
            target.relative_to(item)
            target_inside = True
        except ValueError:
            target_inside = False
        try:
            item.relative_to(target)
            contains_protected = True
        except ValueError:
            contains_protected = False
        # The canonical project cache is intentionally below models/. It may
        # also live on a separate disk when models/ is a managed symlink. Skip
        # only that expected parent overlap; every other protected target,
        # including a hub symlink to configs/.ssh/project root, remains fatal.
        if (
            is_known_project_cache
            and item == (project / "models").resolve()
            and target_inside
            and target != item
        ):
            continue
        if target == item or target_inside or contains_protected:
            raise ValueError(f"cache path overlaps protected directory: {item}")

    cache_names = {"hub", "hf-cache", "hub-cache", "huggingface-cache"}
    has_marker = False
    try:
        has_marker = any(
            child.name == ".locks"
            or child.name.startswith(("models--", "datasets--", "spaces--"))
            for child in target.iterdir()
        )
    except OSError:
        pass
    if target.name.lower() not in cache_names and not has_marker:
        raise ValueError("directory does not look like a Hugging Face hub cache")
    return target


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
        linked = _is_link_path(child)
        if not linked and not child.is_dir():
            continue
        size = _du(child)
        try:
            mtime = child.lstat().st_mtime
        except OSError:
            mtime = 0.0
        entries.append(
            {
                "name": child.name,
                "path": str(child),
                "bytes": size.bytes,
                "files": size.files,
                "mtime": mtime,
                "linked": linked,
            }
        )
    return {"archive_root": str(archive), "entries": entries}


@router.delete("/storage/archive/{name}")
def storage_delete_archive_entry(name: str) -> dict[str, Any]:
    """Permanently remove a single archived workspace from `runs/_archive/`."""
    target = _resolve_archive_entry(name)
    size = _du(target)
    try:
        if _is_link_path(target):
            _remove_link_path(target)
        else:
            _rmtree(target)
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
        linked = _is_link_path(child)
        if not linked and not child.is_dir():
            continue
        size = _du(child)
        try:
            if linked:
                _remove_link_path(child)
            else:
                _rmtree(child)
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
    try:
        cache = _validate_hf_cache_delete_target(cache)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    size = _du(cache)
    try:
        _rmtree(cache)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "deleted": str(cache),
        "bytes_freed": size.bytes,
        "files_removed": size.files,
    }
