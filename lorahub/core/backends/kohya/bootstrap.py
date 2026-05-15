"""Resolve the kohya-ss/sd-scripts checkout and the Python that runs it.

v0.1 scope: locate an existing sd-scripts checkout via (in priority order)
explicit recipe field, environment variable, or default user-data location.
Fail fast with a clear remediation message if it isn't there. Automated
clone+pip-install is deferred to v0.2.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from platformdirs import user_data_path

_ENV_SD_SCRIPTS = "LORAHUB_KOHYA_SD_SCRIPTS"
_ENV_PYTHON = "LORAHUB_KOHYA_PYTHON"
_REQUIRED_SCRIPTS = ("train_network.py", "sdxl_train_network.py")


class BootstrapError(RuntimeError):
    """Raised when the kohya backend cannot be located or is incomplete."""


@dataclass(frozen=True, slots=True)
class KohyaEnv:
    """A resolved kohya runtime: where the scripts are and which Python runs them."""

    sd_scripts_path: Path
    python_executable: Path

    def script(self, name: str) -> Path:
        """Return the absolute path to a kohya script (e.g. `sdxl_train_network.py`)."""
        return self.sd_scripts_path / name


def default_sd_scripts_path() -> Path:
    return user_data_path("lorahub", "lorahub") / "backends" / "sd-scripts"


def resolve(
    recipe_path: Path | None = None,
    recipe_python: Path | None = None,
) -> KohyaEnv:
    """Resolve the kohya environment using recipe → env var → default.

    For the Python interpreter, after exhausting the recipe field and env var
    we look for a `venv/` next to sd-scripts (the layout kohya's README sets
    up). Only if none is present do we fall back to the current interpreter.
    """
    sd_scripts = (
        recipe_path
        or _path_from_env(_ENV_SD_SCRIPTS)
        or default_sd_scripts_path()
    )
    python = (
        recipe_python
        or _path_from_env(_ENV_PYTHON)
        or _venv_python(sd_scripts)
        or Path(sys.executable)
    )

    _check_sd_scripts(sd_scripts)
    _check_python(python)

    return KohyaEnv(sd_scripts_path=sd_scripts.resolve(), python_executable=python.resolve())


def _venv_python(sd_scripts: Path) -> Path | None:
    """Look for the venv python kohya's README sets up next to its checkout."""
    candidates = (
        sd_scripts / "venv" / "Scripts" / "python.exe",
        sd_scripts / "venv" / "bin" / "python",
        sd_scripts / ".venv" / "Scripts" / "python.exe",
        sd_scripts / ".venv" / "bin" / "python",
    )
    for c in candidates:
        if c.is_file():
            return c
    return None


def _path_from_env(name: str) -> Path | None:
    raw = os.environ.get(name)
    return Path(raw) if raw else None


def _check_sd_scripts(path: Path) -> None:
    if not path.exists():
        msg = (
            f"kohya sd-scripts not found at {path}.\n"
            f"Either:\n"
            f"  1. Set backend.sd_scripts_path in your recipe, or\n"
            f"  2. Set the {_ENV_SD_SCRIPTS} environment variable, or\n"
            f"  3. Clone kohya-ss/sd-scripts into {default_sd_scripts_path()}"
        )
        raise BootstrapError(msg)
    if not path.is_dir():
        msg = f"sd-scripts path is not a directory: {path}"
        raise BootstrapError(msg)
    missing = [s for s in _REQUIRED_SCRIPTS if not (path / s).is_file()]
    if missing:
        msg = (
            f"sd-scripts checkout at {path} is missing required files: "
            f"{', '.join(missing)}. Is this really kohya-ss/sd-scripts?"
        )
        raise BootstrapError(msg)


def _check_python(python: Path) -> None:
    if not python.exists():
        msg = f"Python executable not found: {python}"
        raise BootstrapError(msg)
    if not python.is_file():
        msg = f"Python executable is not a file: {python}"
        raise BootstrapError(msg)
