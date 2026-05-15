"""Resolve the tdrussell/diffusion-pipe checkout and the Python that runs it.

Mirrors `lorahub.core.backends.kohya.bootstrap` exactly so the bootstrap
session and probes can treat both backends symmetrically.

Priority order for both the repo path and the python interpreter:
  1. Explicit recipe field (passed in by `resolve()` callers)
  2. `LORAHUB_DIFFUSION_PIPE_REPO` / `LORAHUB_DIFFUSION_PIPE_PYTHON`
  3. The default location next to the project (or in user-data)
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from platformdirs import user_data_path

from lorahub.core.backends.errors import BootstrapError

_ENV_REPO = "LORAHUB_DIFFUSION_PIPE_REPO"
_ENV_PYTHON = "LORAHUB_DIFFUSION_PIPE_PYTHON"
# diffusion-pipe's main entrypoint is `train.py` in the repo root.
_REQUIRED_FILES = ("train.py",)


@dataclass(frozen=True, slots=True)
class DiffusionPipeEnv:
    """A resolved diffusion-pipe runtime."""

    repo_path: Path
    python_executable: Path

    def script(self, name: str) -> Path:
        return self.repo_path / name


def default_repo_path() -> Path:
    """Where lorahub looks for diffusion-pipe when nothing else is configured."""
    cwd_local = Path.cwd() / "diffusion-pipe"
    if cwd_local.is_dir():
        return cwd_local
    user_local = user_data_path("lorahub", "lorahub") / "backends" / "diffusion-pipe"
    if user_local.is_dir():
        return user_local
    return cwd_local


def resolve(
    recipe_path: Path | None = None,
    recipe_python: Path | None = None,
) -> DiffusionPipeEnv:
    """Resolve the diffusion-pipe environment using recipe -> env var -> default."""
    repo = (
        recipe_path
        or _path_from_env(_ENV_REPO)
        or default_repo_path()
    )
    python = (
        recipe_python
        or _path_from_env(_ENV_PYTHON)
        or _venv_python(repo)
        or Path(sys.executable)
    )

    _check_repo(repo)
    _check_python(python)

    return DiffusionPipeEnv(
        repo_path=repo.resolve(),
        python_executable=python.resolve(),
    )


def _venv_python(repo: Path) -> Path | None:
    """Look for the venv python a diffusion-pipe checkout typically ships with."""
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


def _path_from_env(name: str) -> Path | None:
    raw = os.environ.get(name)
    return Path(raw) if raw else None


def _check_repo(path: Path) -> None:
    if not path.exists():
        msg = (
            f"diffusion-pipe checkout not found at {path}.\n"
            f"Either:\n"
            f"  1. Set backend.repo_path in your recipe, or\n"
            f"  2. Set the {_ENV_REPO} environment variable, or\n"
            f"  3. Clone tdrussell/diffusion-pipe into {default_repo_path()}"
        )
        raise BootstrapError(msg)
    if not path.is_dir():
        msg = f"diffusion-pipe path is not a directory: {path}"
        raise BootstrapError(msg)
    missing = [f for f in _REQUIRED_FILES if not (path / f).is_file()]
    if missing:
        msg = (
            f"diffusion-pipe checkout at {path} is missing required files: "
            f"{', '.join(missing)}. Is this really tdrussell/diffusion-pipe?"
        )
        raise BootstrapError(msg)


def _check_python(python: Path) -> None:
    if not python.exists():
        msg = f"Python executable not found: {python}"
        raise BootstrapError(msg)
    if not python.is_file():
        msg = f"Python executable is not a file: {python}"
        raise BootstrapError(msg)


__all__ = [
    "BootstrapError",
    "DiffusionPipeEnv",
    "default_repo_path",
    "resolve",
]
