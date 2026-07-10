"""Project root + canonical LoraHub data directories.

Why this module exists: the API layer historically resolved every
local path via ``Path.cwd() / "runs" / ...``. That works exactly
once — at first launch from the project root — and silently breaks
the second a user starts the server from a different directory.
Common ways the cwd diverges:

* ``lorahub serve`` from anywhere (the entry point doesn't chdir).
* Double-click on a desktop / Start-menu shortcut (cwd lands on
  the user profile, sometimes ``system32``).
* uvicorn restarted by a service manager / nssm wrapper.
* The Tauri/Electron-style launcher we're shipping.

Effect: every restart from a different cwd creates a brand-new
empty ``runs/`` somewhere, the SQLite stores get rebuilt empty,
and the user sees their training history disappear. The runs/
directory on disk is fine — the API just stops looking at it.

This module pins the project root once, stores it as a module-
level singleton, and offers ``project_root()`` / ``runs_dir()``
helpers. The lifespan hook calls ``ensure_initialised()`` early so
later code that still resolves via ``Path.cwd()`` lands on the
same tree (lifespan also chdir's there as a belt-and-suspenders
safeguard).

Resolution order:
  1. ``LORAHUB_HOME`` env var, when set and a directory.
  2. The directory walked up from this file, looking for a
     ``pyproject.toml`` whose project name is ``lorahub``. Works
     for editable installs (``pip install -e .``) and the source
     checkout. Won't fire from a wheel install where the package
     lives under ``site-packages`` — for that case we fall back
     to:
  3. ``Path.cwd()``. The historic behaviour, kept so a user who
     deliberately runs ``cd /some/other/place && lorahub serve``
     still gets a self-contained data directory there.

The first call resolves and caches; subsequent calls return the
cached value. ``ensure_initialised()`` is the explicit entry
point used by the lifespan hook.
"""

from __future__ import annotations

import os
import sys
import threading
from functools import lru_cache
from pathlib import Path

_LOCK = threading.Lock()
_resolved: Path | None = None


def _detect_from_source_tree() -> Path | None:
    """Walk parents of this file looking for a LoraHub source checkout.

    Returns the project root (containing ``pyproject.toml`` with
    ``name = "lorahub"``) when found. Skips wheels installed under
    ``site-packages`` / ``dist-packages``: those don't ship the
    pyproject so the walk hits the venv root and returns None.
    """
    here = Path(__file__).resolve()
    for ancestor in here.parents:
        # Bail when we hit a packaged install — there's no project
        # checkout above ``site-packages``.
        if ancestor.name in {"site-packages", "dist-packages"}:
            return None
        py = ancestor / "pyproject.toml"
        if not py.is_file():
            continue
        try:
            text = py.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        # Cheap fingerprint — we don't pull tomllib for a single
        # ``name = "lorahub"`` check. Match either single or double
        # quotes; whitespace is whatever the user wrote.
        if 'name = "lorahub"' in text or "name = 'lorahub'" in text:
            return ancestor
    return None


def _resolve() -> Path:
    """Decide where the project root lives. See module docstring."""
    env = os.environ.get("LORAHUB_HOME", "").strip()
    if env:
        candidate = Path(env).expanduser().resolve()
        if candidate.is_dir():
            return candidate
        # Honour the env var even if the directory doesn't exist yet —
        # creating a fresh data tree at the user's chosen location is
        # less surprising than silently falling back to cwd.
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate

    detected = _detect_from_source_tree()
    if detected is not None:
        return detected.resolve()

    return Path.cwd().resolve()


def ensure_initialised() -> Path:
    """Resolve + cache the project root. Idempotent.

    Side effect: chdir's the process to the resolved root the first
    time it's called. We do this so any downstream code that still
    uses ``Path.cwd()`` (jobs router, sweep router, storage router,
    image_studio_store …) lands on the correct tree without each
    site needing an audit. Subsequent calls are no-ops.
    """
    global _resolved
    with _LOCK:
        if _resolved is not None:
            return _resolved
        root = _resolve()
        try:
            os.chdir(root)
        except OSError:
            # Read-only filesystem / permission issue — keep going,
            # the explicit helpers below still return the right path.
            pass
        _resolved = root
    return _resolved


def project_root() -> Path:
    """Return the resolved project root, initialising on first call."""
    if _resolved is not None:
        return _resolved
    return ensure_initialised()


def runs_dir() -> Path:
    """``<project_root>/runs`` — top-level data directory."""
    d = project_root() / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def models_dir() -> Path:
    """Return the default writable model root under the project."""
    directory = project_root() / "models"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def model_roots() -> tuple[Path, ...]:
    """Return model roots writable through the API.

    ``LORAHUB_MODELS_ROOT`` is an explicit opt-in for installations that keep
    large weights on a separate disk. The project model directory remains
    valid so backend links and existing relative configs continue to work.
    """
    roots = [models_dir().resolve()]
    configured = os.environ.get("LORAHUB_MODELS_ROOT", "").strip()
    if configured:
        custom = Path(configured).expanduser().resolve()
        custom.mkdir(parents=True, exist_ok=True)
        if custom not in roots:
            roots.append(custom)
    return tuple(roots)


def resolve_model_path(raw: str | Path, *, allow_root: bool = False) -> Path:
    """Resolve a model path while preventing writes or scans outside model roots."""
    value = Path(raw).expanduser()
    if value.is_absolute():
        target = value.resolve()
    else:
        parts = value.parts
        if parts and parts[0].lower() == "models":
            target = (project_root() / value).resolve()
        else:
            target = (models_dir() / value).resolve()

    for root in model_roots():
        try:
            target.relative_to(root)
        except ValueError:
            continue
        if target == root and not allow_root:
            raise ValueError("model path must be a child of a configured model root")
        return target
    raise ValueError("model path is outside configured model roots")


def resolve_run_path(raw: str | Path, *, allow_root: bool = False) -> Path:
    """Resolve a user-supplied run path without permitting tree escape."""
    target = Path(raw).expanduser().resolve()
    root = runs_dir().resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"workspace must be under {root}") from exc
    if target == root and not allow_root:
        raise ValueError("workspace must be a child of the runs directory")
    return target


def resolve_sweep_variant_path(workspace_root: str | Path, variant_name: str) -> Path:
    """Resolve one sweep workspace as a direct child of its sweep root.

    Variant names are generated from user-controlled templates and axis values.
    Keeping them to one path component prevents a sweep from colliding with an
    unrelated run through ``../`` or an absolute path while still allowing the
    punctuation used by existing generated names.
    """
    root = resolve_run_path(workspace_root, allow_root=True)
    name = variant_name.strip()
    if (
        not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or Path(name).is_absolute()
    ):
        raise ValueError("sweep variant name must be a single path component")

    target = (root / name).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError("sweep variant workspace escapes the sweep root") from exc
    if target == root:
        raise ValueError("sweep variant workspace must be below the sweep root")
    return resolve_run_path(target)


@lru_cache(maxsize=1)
def is_windows() -> bool:
    return sys.platform == "win32"


__all__ = [
    "ensure_initialised",
    "is_windows",
    "model_roots",
    "models_dir",
    "project_root",
    "resolve_model_path",
    "resolve_run_path",
    "resolve_sweep_variant_path",
    "runs_dir",
]
