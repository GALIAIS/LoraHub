"""Filesystem browser endpoints for the dataset/IDE pane.

These power the left-side file tree in the dataset page (think marimo /
Jupyter file panel). Two safety modes:

  * Default: paths must resolve under `dataset_files._allowed_roots()` --
    the same allow-list the dataset thumbnail/caption endpoints already
    use (cwd, $LORAHUB_DATASETS_ROOT, job workspaces).

  * Escape hatch: when `Settings.allow_filesystem_browse` is true, paths
    are accepted as long as they resolve to something the server process
    can stat. The flag is opt-in and lives in the Settings JSON, so it
    must be flipped explicitly. Localhost-only API + per-user opt-in is
    the same posture VS Code's "open folder" command takes.

The text endpoints refuse to read/write anything that smells binary
(based on suffix + a NUL-byte sniff). Image previews go through the
existing `/api/datasets/thumb` endpoint, which already handles allow-list
+ caching.
"""

from __future__ import annotations

import contextlib
import os
import platform
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from lorahub.api import app as app_module
from lorahub.api.dataset_files import _allowed_roots

router = APIRouter(prefix="/api")

# Suffixes the editor will treat as text. Anything else is shown read-only
# with a "binary file" placeholder so the UI doesn't ship megabytes of
# binary garbage to the textarea.
_TEXT_SUFFIXES: frozenset[str] = frozenset(
    {
        ".txt", ".md", ".rst", ".csv", ".tsv", ".log",
        ".json", ".jsonl", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".env",
        ".py", ".pyi", ".sh", ".bash", ".zsh", ".ps1", ".bat", ".cmd",
        ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
        ".html", ".htm", ".css", ".scss", ".sass", ".less",
        ".xml", ".svg",
        ".c", ".h", ".cpp", ".hpp", ".cc", ".cs", ".java", ".kt", ".rs", ".go",
        ".rb", ".php", ".swift", ".lua", ".sql", ".r", ".dart",
        ".gitignore", ".dockerignore", ".editorconfig",
    }
)
_IMAGE_SUFFIXES: frozenset[str] = frozenset(
    {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp"}
)
# Hard cap on text reads to avoid sending a multi-MB file into a textarea.
_MAX_TEXT_BYTES = 2 * 1024 * 1024  # 2 MiB
# Cap directory listings so a /tmp full of a million files doesn't OOM us.
_MAX_LIST_ENTRIES = 5000


def _allow_anywhere() -> bool:
    """Read the per-user opt-in for unrestricted browsing."""
    with contextlib.suppress(Exception):
        return bool(app_module._settings_store.load().allow_filesystem_browse)
    return False


def _resolve(raw: str) -> Path:
    """Resolve a path with allow-list (or pass-through when opt-in is on)."""
    if not raw or not raw.strip():
        raise HTTPException(status_code=400, detail="path is required")
    try:
        target = Path(raw).expanduser().resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail="invalid path") from exc

    if _allow_anywhere():
        return target

    for root in _allowed_roots():
        with contextlib.suppress(ValueError):
            target.relative_to(root)
            return target
    raise HTTPException(
        status_code=403,
        detail=(
            "path is outside allowed roots. Enable "
            "`allow_filesystem_browse` in settings to browse anywhere."
        ),
    )


def _entry(p: Path, parent: Path) -> dict[str, Any]:
    """Build a single directory-entry payload, suppressing stat errors."""
    is_dir = False
    size = 0
    mtime: float | None = None
    with contextlib.suppress(OSError):
        st = p.stat()
        is_dir = p.is_dir()
        size = 0 if is_dir else st.st_size
        mtime = st.st_mtime
    suffix = "" if is_dir else p.suffix.lower()
    kind = "dir"
    if not is_dir:
        if suffix in _IMAGE_SUFFIXES:
            kind = "image"
        elif suffix in _TEXT_SUFFIXES:
            kind = "text"
        else:
            kind = "binary"
    return {
        "name": p.name,
        "path": str(p),
        "relative_path": p.relative_to(parent).as_posix() if parent != p else p.name,
        "is_dir": is_dir,
        "kind": kind,
        "suffix": suffix,
        "size": size,
        "mtime": mtime,
    }


@router.get("/fs/roots")
def list_roots() -> dict[str, Any]:
    """Return the roots the browser should expose by default.

    When `allow_filesystem_browse` is on, also include drive letters
    (Windows) or `/` (POSIX) so users can navigate anywhere.
    """
    roots: list[dict[str, Any]] = []
    seen: set[str] = set()
    for r in _allowed_roots():
        key = str(r)
        if key in seen:
            continue
        seen.add(key)
        roots.append(
            {
                "name": r.name or key,
                "path": key,
                "kind": "dataset_root",
            }
        )
    if _allow_anywhere():
        if platform.system() == "Windows":
            for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                drive = Path(f"{letter}:\\")
                if drive.exists():
                    key = str(drive)
                    if key in seen:
                        continue
                    seen.add(key)
                    roots.append({"name": f"{letter}:", "path": key, "kind": "drive"})
        else:
            roots.append({"name": "/", "path": "/", "kind": "drive"})
    return {"roots": roots, "unrestricted": _allow_anywhere()}


@router.get("/fs/list")
def list_directory(path: str, show_hidden: bool = False) -> dict[str, Any]:
    """List entries directly inside `path` (one level, no recursion)."""
    target = _resolve(path)
    if not target.exists():
        raise HTTPException(status_code=404, detail="directory not found")
    if not target.is_dir():
        raise HTTPException(status_code=400, detail="not a directory")

    entries: list[dict[str, Any]] = []
    try:
        with os.scandir(target) as it:
            for de in it:
                if not show_hidden and de.name.startswith("."):
                    continue
                entries.append(_entry(Path(de.path), target))
                if len(entries) >= _MAX_LIST_ENTRIES:
                    break
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    entries.sort(key=lambda e: (not e["is_dir"], e["name"].lower()))
    parent: str | None = None
    if target.parent != target:
        # Only expose the parent if we'd be allowed to navigate to it.
        parent_path = target.parent
        if _allow_anywhere():
            parent = str(parent_path)
        else:
            for root in _allowed_roots():
                with contextlib.suppress(ValueError):
                    parent_path.relative_to(root)
                    parent = str(parent_path)
                    break
    return {
        "path": str(target),
        "parent": parent,
        "entries": entries,
        "truncated": len(entries) >= _MAX_LIST_ENTRIES,
    }


@router.get("/fs/subdirs")
def list_subdirs(path: str) -> dict[str, Any]:
    """Return only subdirectories (cheap dropdown source)."""
    target = _resolve(path)
    if not target.is_dir():
        raise HTTPException(status_code=400, detail="not a directory")
    subdirs: list[dict[str, Any]] = []
    with contextlib.suppress(OSError):
        with os.scandir(target) as it:
            for de in it:
                if de.name.startswith("."):
                    continue
                if de.is_dir():
                    subdirs.append({"name": de.name, "path": de.path})
    subdirs.sort(key=lambda e: e["name"].lower())
    return {"path": str(target), "subdirs": subdirs}


def _looks_binary(data: bytes) -> bool:
    if b"\x00" in data[:8192]:
        return True
    return False


@router.get("/fs/read")
def read_file(path: str) -> dict[str, Any]:
    """Read a text file. Returns kind + content (or kind=binary placeholder)."""
    target = _resolve(path)
    if not target.exists():
        raise HTTPException(status_code=404, detail="file not found")
    if not target.is_file():
        raise HTTPException(status_code=400, detail="not a file")

    suffix = target.suffix.lower()
    size = 0
    with contextlib.suppress(OSError):
        size = target.stat().st_size

    if suffix in _IMAGE_SUFFIXES:
        return {
            "path": str(target),
            "kind": "image",
            "size": size,
            "content": None,
        }

    if size > _MAX_TEXT_BYTES:
        return {
            "path": str(target),
            "kind": "binary",
            "size": size,
            "content": None,
            "reason": f"file too large ({size} bytes); max {_MAX_TEXT_BYTES}",
        }

    try:
        raw = target.read_bytes()
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if _looks_binary(raw):
        return {
            "path": str(target),
            "kind": "binary",
            "size": size,
            "content": None,
            "reason": "file contains NUL bytes",
        }

    try:
        text = raw.decode("utf-8")
        encoding = "utf-8"
    except UnicodeDecodeError:
        try:
            text = raw.decode("utf-8-sig")
            encoding = "utf-8-sig"
        except UnicodeDecodeError:
            return {
                "path": str(target),
                "kind": "binary",
                "size": size,
                "content": None,
                "reason": "file is not UTF-8",
            }
    return {
        "path": str(target),
        "kind": "text",
        "suffix": suffix,
        "size": size,
        "encoding": encoding,
        "content": text,
    }


class FileWriteRequest(BaseModel):
    path: str = Field(..., description="Absolute or relative path of the file")
    content: str = Field(..., description="UTF-8 text to write")
    create: bool = Field(default=False, description="Create the file if missing")


@router.put("/fs/write")
def write_file(body: FileWriteRequest) -> dict[str, Any]:
    """Persist UTF-8 text to *path*. Refuses obviously-binary suffixes."""
    target = _resolve(body.path)
    if target.is_dir():
        raise HTTPException(status_code=400, detail="path is a directory")
    suffix = target.suffix.lower()
    if suffix in _IMAGE_SUFFIXES or (
        suffix and suffix not in _TEXT_SUFFIXES and target.exists()
    ):
        # Refuse to clobber unknown binary file types via the text editor.
        raise HTTPException(
            status_code=400,
            detail=f"refusing to write text into {suffix!r} file",
        )
    if not target.exists() and not body.create:
        raise HTTPException(status_code=404, detail="file not found")

    text = body.content.replace("\r\n", "\n").replace("\r", "\n")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8", newline="\n")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "path": str(target),
        "bytes": len(text.encode("utf-8")),
    }


__all__ = ["router"]
