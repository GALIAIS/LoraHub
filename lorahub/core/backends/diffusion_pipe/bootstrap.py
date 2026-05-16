"""Resolve the tdrussell/diffusion-pipe checkout and the Python that runs it.

Mirrors `lorahub.core.backends.kohya.bootstrap` -- both backends compose the
same helpers from ``lorahub.core.backends._common.bootstrap`` so probes and
the bootstrap session can treat them symmetrically.

Priority order for both the repo path and the python interpreter:
  1. Explicit recipe field (passed in by `resolve()` callers)
  2. `LORAHUB_DIFFUSION_PIPE_REPO` / `LORAHUB_DIFFUSION_PIPE_PYTHON`
  3. The default location next to the project (or in user-data)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lorahub.core.backends._common import bootstrap as _common
from lorahub.core.backends.errors import BootstrapError

_ENV_REPO = "LORAHUB_DIFFUSION_PIPE_REPO"
_ENV_PYTHON = "LORAHUB_DIFFUSION_PIPE_PYTHON"
# diffusion-pipe's main entrypoint is `train.py` in the repo root.
_REQUIRED_FILES = ("train.py",)
_LABEL = "tdrussell/diffusion-pipe"


# Re-export the shared venv-python lookup so the api.settings probe can
# keep using the historical private name.
_venv_python = _common.venv_python


@dataclass(frozen=True, slots=True)
class DiffusionPipeEnv:
    """A resolved diffusion-pipe runtime."""

    repo_path: Path
    python_executable: Path

    def script(self, name: str) -> Path:
        return self.repo_path / name


def default_repo_path() -> Path:
    """Where lorahub looks for diffusion-pipe when nothing else is configured."""
    return _common.default_repo_path("diffusion-pipe")


def resolve(
    config_path: Path | None = None,
    config_python: Path | None = None,
) -> DiffusionPipeEnv:
    """Resolve the diffusion-pipe environment using recipe -> env var -> default."""
    repo = (
        config_path
        or _common.path_from_env(_ENV_REPO)
        or default_repo_path()
    )
    python = _common.resolve_python(
        repo, config_python=config_python, env_var=_ENV_PYTHON
    )

    _common.check_repo(
        repo,
        label=_LABEL,
        required_files=_REQUIRED_FILES,
        env_var=_ENV_REPO,
        default_path=default_repo_path(),
        config_field="repo_path",
    )
    _common.check_python(python)

    return DiffusionPipeEnv(
        repo_path=repo.resolve(),
        # NB: `absolute()`, not `resolve()`. A venv's `bin/python` is
        # typically a symlink to the system interpreter; resolving it would
        # bypass the venv and load the system site-packages instead, so
        # `import wandb` inside the venv would fail.
        python_executable=python.absolute(),
    )


__all__ = [
    "BootstrapError",
    "DiffusionPipeEnv",
    "default_repo_path",
    "resolve",
]
