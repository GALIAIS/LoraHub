"""Resolve the kohya-ss/sd-scripts checkout and the Python that runs it.

Locates an existing sd-scripts checkout via (in priority order) explicit
recipe field, environment variable, or default user-data location. Fails
fast with a clear remediation message if it isn't there. Shared helpers
live in ``lorahub.core.backends._common.bootstrap`` so the diffusion-pipe
backend can apply the same cascade against its own paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lorahub.core.backends._common import bootstrap as _common
from lorahub.core.backends.errors import BootstrapError

_ENV_SD_SCRIPTS = "LORAHUB_KOHYA_SD_SCRIPTS"
_ENV_PYTHON = "LORAHUB_KOHYA_PYTHON"
# Every entry script lorahub may shell out to. Mirrors compiler._KOHYA_SCRIPT_MAP
# but kept duplicated here to avoid a backend->compiler import cycle through
# bootstrap (probe_kohya_backend imports this module from api.settings).
_REQUIRED_SCRIPTS = (
    "train_network.py",
    "sdxl_train_network.py",
    "sd3_train_network.py",
    "flux_train_network.py",
    "lumina_train_network.py",
    "hunyuan_image_train_network.py",
    "anima_train_network.py",
)
_LABEL = "kohya-ss/sd-scripts"


# Re-export the shared venv-python lookup so the api.settings probe can
# keep using the historical private name without reaching into _common.
_venv_python = _common.venv_python


__all__ = [
    "BootstrapError",
    "KohyaEnv",
    "default_sd_scripts_path",
    "resolve",
]


@dataclass(frozen=True, slots=True)
class KohyaEnv:
    """A resolved kohya runtime: where the scripts are and which Python runs them."""

    sd_scripts_path: Path
    python_executable: Path

    def script(self, name: str) -> Path:
        """Return the absolute path to a kohya script (e.g. `sdxl_train_network.py`)."""
        return self.sd_scripts_path / name


def default_sd_scripts_path() -> Path:
    """Where lorahub looks for sd-scripts when nothing else is configured."""
    return _common.default_repo_path("sd-scripts")


def resolve(
    recipe_path: Path | None = None,
    recipe_python: Path | None = None,
) -> KohyaEnv:
    """Resolve the kohya environment using recipe -> env var -> default.

    For the Python interpreter, after exhausting the recipe field and env var
    we look for a `venv/` next to sd-scripts (the layout kohya's README sets
    up). Only if none is present do we fall back to the current interpreter.
    """
    sd_scripts = (
        recipe_path
        or _common.path_from_env(_ENV_SD_SCRIPTS)
        or default_sd_scripts_path()
    )
    python = _common.resolve_python(
        sd_scripts, recipe_python=recipe_python, env_var=_ENV_PYTHON
    )

    _common.check_repo(
        sd_scripts,
        label=_LABEL,
        required_files=_REQUIRED_SCRIPTS,
        env_var=_ENV_SD_SCRIPTS,
        default_path=default_sd_scripts_path(),
        recipe_field="sd_scripts_path",
    )
    _common.check_python(python)

    return KohyaEnv(
        sd_scripts_path=sd_scripts.resolve(),
        python_executable=python.resolve(),
    )
