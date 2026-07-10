"""Resolve the vendored Ostris AI Toolkit runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lorahub.core.backends._common import bootstrap as _common
from lorahub.core.backends.errors import BootstrapError
from lorahub.core.paths import project_root

_ENV_REPO = "LORAHUB_AI_TOOLKIT_REPO"
_ENV_PYTHON = "LORAHUB_AI_TOOLKIT_PYTHON"
_REQUIRED_FILES = ("run.py", "toolkit/job.py", "extensions_built_in/sd_trainer/__init__.py")
_LABEL = "ostris/ai-toolkit (vendored)"

_venv_python = _common.venv_python


@dataclass(frozen=True, slots=True)
class AIToolkitEnv:
    repo_path: Path
    python_executable: Path

    def script(self, name: str) -> Path:
        return self.repo_path / name


def default_repo_path() -> Path:
    import os

    env = os.environ.get(_ENV_REPO)
    if env:
        return Path(env).expanduser()

    candidate = project_root() / "external" / "ai_toolkit"
    if (candidate / "run.py").is_file():
        return candidate

    from platformdirs import user_data_path  # noqa: PLC0415

    return user_data_path("lorahub", "lorahub") / "backends" / "ai_toolkit"


def resolve(
    config_path: Path | None = None,
    config_python: Path | None = None,
) -> AIToolkitEnv:
    repo = config_path or _common.path_from_env(_ENV_REPO) or default_repo_path()
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

    return AIToolkitEnv(repo_path=repo.resolve(), python_executable=python.absolute())


__all__ = [
    "AIToolkitEnv",
    "BootstrapError",
    "_ENV_PYTHON",
    "_ENV_REPO",
    "_REQUIRED_FILES",
    "_venv_python",
    "default_repo_path",
    "resolve",
]
