"""Shared bootstrap helpers for resolving a backend checkout + interpreter.

Every backend wants the same priority cascade: explicit recipe field,
environment variable, default location. And every backend wants to detect a
``venv/`` next to the checkout before falling back to the running
interpreter. The kohya and diffusion-pipe modules used to ship near-identical
private helpers; they now compose the functions in this module instead.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from platformdirs import user_data_path

from lorahub.core.backends.errors import BootstrapError


def path_from_env(name: str) -> Path | None:
    """Read ``$name`` from the environment and return it as a Path, or None."""
    raw = os.environ.get(name)
    return Path(raw) if raw else None


def venv_python(repo: Path) -> Path | None:
    """Find a Python interpreter inside a backend's local venv.

    Both kohya-ss/sd-scripts and tdrussell/diffusion-pipe README templates
    ship a ``venv/`` (or ``.venv/``) directory next to the checkout, so we
    search the four standard layouts in order: Windows + POSIX, ``venv`` +
    ``.venv``. Returns ``None`` if none are present so callers can fall back
    to the host interpreter.
    """
    candidates = (
        repo / "venv" / "Scripts" / "python.exe",
        repo / "venv" / "bin" / "python",
        repo / ".venv" / "Scripts" / "python.exe",
        repo / ".venv" / "bin" / "python",
    )
    for c in candidates:
        if c.is_file():
            return c
    return None


def default_repo_path(dir_name: str) -> Path:
    """Where lorahub looks for a backend checkout when nothing is configured.

    Priority order:
      1. ``<cwd>/<dir_name>`` -- the project-local convention bootstrap
         creates and `lorahub init` recommends.
      2. ``<platformdirs user_data>/lorahub/lorahub/backends/<dir_name>`` --
         OS-standard per-user data location.

    The first existing directory wins; if neither exists, the cwd-relative
    path is returned so error messages point users at the conventional spot.
    """
    cwd_local = Path.cwd() / dir_name
    if cwd_local.is_dir():
        return cwd_local
    user_local = user_data_path("lorahub", "lorahub") / "backends" / dir_name
    if user_local.is_dir():
        return user_local
    return cwd_local


def check_python(python: Path) -> None:
    """Raise BootstrapError if ``python`` does not point at a real file."""
    if not python.exists():
        msg = f"Python executable not found: {python}"
        raise BootstrapError(msg)
    if not python.is_file():
        msg = f"Python executable is not a file: {python}"
        raise BootstrapError(msg)


def check_repo(
    path: Path,
    *,
    label: str,
    required_files: tuple[str, ...],
    env_var: str,
    default_path: Path,
    config_field: str,
) -> None:
    """Validate that ``path`` is a non-empty backend checkout.

    Emits remediation messages telling the user how to point lorahub at a
    valid checkout. Both kohya and diffusion-pipe drove identical logic
    inline; centralising it keeps the wording consistent.
    """
    if not path.exists():
        msg = (
            f"{label} not found at {path}.\n"
            f"Either:\n"
            f"  1. Set backend.{config_field} in your recipe, or\n"
            f"  2. Set the {env_var} environment variable, or\n"
            f"  3. Clone {label} into {default_path}"
        )
        raise BootstrapError(msg)
    if not path.is_dir():
        msg = f"{label} path is not a directory: {path}"
        raise BootstrapError(msg)
    missing = [f for f in required_files if not (path / f).is_file()]
    if missing:
        msg = (
            f"{label} checkout at {path} is missing required files: "
            f"{', '.join(missing)}. Is this really {label}?"
        )
        raise BootstrapError(msg)


def resolve_python(
    repo: Path,
    *,
    config_python: Path | None,
    env_var: str,
) -> Path:
    """Apply the recipe -> env -> venv -> sys.executable cascade for python."""
    return (
        config_python
        or path_from_env(env_var)
        or venv_python(repo)
        or Path(sys.executable)
    )


__all__ = [
    "check_python",
    "check_repo",
    "default_repo_path",
    "path_from_env",
    "resolve_python",
    "venv_python",
]
