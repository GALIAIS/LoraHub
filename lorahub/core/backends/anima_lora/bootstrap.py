"""Resolve the vendored anima_lora copy + the Python that runs it.

Distinguishing twist vs kohya / diffusion-pipe: the repo is **vendored**
under ``external/anima_lora/`` in the LoraHub source tree. We do not
expect the user to clone anything — the source ships in the box. The
env var ``LORAHUB_ANIMA_LORA_REPO`` exists only so a developer can
point at a different checkout for ad-hoc debugging.

Python interpreter resolution still cascades recipe → env →
``<repo>/venv`` → ``<repo>/.venv`` → host. anima_lora needs torch 2.11
nightly + CUDA 13.x which the LoraHub main venv typically does not
satisfy, so the user is expected to maintain a dedicated venv and
point ``LORAHUB_ANIMA_LORA_PYTHON`` at it. As a convenience, when no
override is set we look for ``external/anima_lora/.venv`` first.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lorahub.core.backends._common import bootstrap as _common
from lorahub.core.backends.errors import BootstrapError

_ENV_REPO = "LORAHUB_ANIMA_LORA_REPO"
_ENV_PYTHON = "LORAHUB_ANIMA_LORA_PYTHON"
# anima_lora ships its trainer + inference at the repo root, plus the
# library/anima/ subpackage that train.py imports. If any of these
# vanish the vendored copy is corrupted and there's no point trying.
_REQUIRED_FILES = ("train.py", "inference.py", "library/anima/__init__.py")
_LABEL = "sorryhyun/anima_lora (vendored)"


# Re-export the shared venv-python lookup so the api.settings probe can
# keep using the historical private name.
_venv_python = _common.venv_python


@dataclass(frozen=True, slots=True)
class AnimaLoraEnv:
    """A resolved anima_lora runtime."""

    repo_path: Path
    python_executable: Path

    def script(self, name: str) -> Path:
        return self.repo_path / name


def default_repo_path() -> Path:
    """Where lorahub looks for the vendored anima_lora copy.

    Resolution order (matches kohya / dp shape):
      1. ``LORAHUB_ANIMA_LORA_REPO`` env var, when set.
      2. Walk up from this file to the LoraHub project root and descend
         into ``external/anima_lora`` — the source-checkout path.
         Returned when the directory actually contains anima's
         ``pyproject.toml`` so wheel installs don't masquerade as a
         valid source tree.
      3. ``platformdirs.user_data_path("lorahub", "lorahub") /
         backends / anima_lora`` — fallback for wheel installs where
         the source tree isn't reachable from this module's path.

    The third leg lets a wheel-installed lorahub still locate (or
    create) a writable anima copy without relying on the source
    layout. Tests that chdir into tmp_path don't trip the env-var or
    user-data legs because ``Path(__file__)`` is fixed.
    """
    import os

    env = os.environ.get("LORAHUB_ANIMA_LORA_REPO")
    if env:
        return Path(env).expanduser()

    here = Path(__file__).resolve()
    # lorahub/core/backends/anima_lora/bootstrap.py → up 5 = project root
    try:
        project_root = here.parents[4]
    except IndexError:
        project_root = None

    candidate = (
        project_root / "external" / "anima_lora" if project_root else None
    )
    if candidate is not None and (candidate / "pyproject.toml").is_file():
        return candidate

    # Wheel install fallback. Same convention kohya / dp use for their
    # ``default_repo_path`` source legs (see _common.bootstrap.default_repo_path).
    from platformdirs import user_data_path  # noqa: PLC0415

    return user_data_path("lorahub", "lorahub") / "backends" / "anima_lora"


def resolve(
    config_path: Path | None = None,
    config_python: Path | None = None,
) -> AnimaLoraEnv:
    """Resolve the anima_lora environment using recipe -> env var -> .venv.

    The Python cascade prefers the dedicated ``<repo>/.venv`` (CPython
    3.13 + torch 2.11/2.12 nightly) installed by ``uv sync``. When that
    venv hasn't been built yet, we fall back to the host
    ``sys.executable`` so :func:`AnimaLoraBackend.validate` can still
    materialise an env (the probe layer surfaces ``ready=False`` so the
    UI prompts the user to run the install). Actual ``launch`` will
    fail loudly if the wrong interpreter is used — anima_lora's
    ``import library.anima`` requires Python 3.13 and torch nightly.
    """
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

    return AnimaLoraEnv(
        repo_path=repo.resolve(),
        # NB: `absolute()` not `resolve()` — see diffusion_pipe.bootstrap
        # for the rationale (resolving a venv symlink bypasses site-packages).
        python_executable=python.absolute(),
    )


__all__ = [
    "BootstrapError",
    "AnimaLoraEnv",
    "default_repo_path",
    "resolve",
]
