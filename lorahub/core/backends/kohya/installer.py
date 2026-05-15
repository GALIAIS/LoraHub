"""Automate the kohya-ss/sd-scripts install — clone + venv + PyTorch + requirements + xformers.

Mirrors the steps from kohya's official Windows README so users don't have to
shell out themselves. Each step is a stand-alone function that runs a single
subprocess; on failure the exception bubbles up with the failing step name so
callers can show a clear error.

All package operations go through ``lorahub.core.toolchain.uv`` (uv venv + uv
pip install). uv is hard-link-aware and shares its global wheel cache across
every venv we ever build, so installing a 6 GB torch into both kohya and
diffusion-pipe costs roughly the size of one install on disk.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lorahub.core.backends.errors import BootstrapError
from lorahub.core.toolchain import uv as _uv

KOHYA_REPO_URL = "https://github.com/kohya-ss/sd-scripts.git"
DEFAULT_TORCH = "2.6.0"
DEFAULT_TORCHVISION = "0.21.0"
DEFAULT_CUDA = "cu124"
DEFAULT_DEPTH = 1


@dataclass(frozen=True, slots=True)
class BootstrapPlan:
    target: Path
    cuda_version: str = DEFAULT_CUDA
    torch_version: str = DEFAULT_TORCH
    torchvision_version: str = DEFAULT_TORCHVISION
    install_xformers: bool = True
    git_depth: int = DEFAULT_DEPTH
    # Optional HTTPS prefix that rewrites `https://github.com/...` URLs at
    # clone time (e.g. "https://gh-proxy.org"). Empty means direct.
    github_proxy: str | None = None
    # Optional path to a CPython executable used as the venv base. When
    # left None, ``create_venv`` defers to uv's default (which falls back
    # to the interpreter currently running the API).
    base_python: Path | None = None

    @property
    def venv_python(self) -> Path:
        return _uv.venv_python(self.target)

    @property
    def torch_index(self) -> str:
        return f"https://download.pytorch.org/whl/{self.cuda_version}"


ProgressCallback = Callable[[str], None]


def _run(cmd: list[str], step: str, progress: ProgressCallback | None) -> None:
    """Run a non-package command (git clone, etc.) with stderr capture."""
    if progress is not None:
        progress(step)
    result = subprocess.run(cmd, check=False, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        if progress is not None and result.stderr:
            tail = "\n".join(result.stderr.strip().splitlines()[-12:])
            progress(f"{step} failed (exit {result.returncode}):\n{tail}")
        raise BootstrapError(step, result.returncode)


def clone(plan: BootstrapPlan, *, progress: ProgressCallback | None = None) -> None:
    if plan.target.exists() and any(plan.target.iterdir()):
        msg = f"target directory is not empty: {plan.target}"
        raise BootstrapError("clone", 1) from FileExistsError(msg)
    plan.target.parent.mkdir(parents=True, exist_ok=True)
    from lorahub.api.settings import apply_github_proxy  # noqa: PLC0415

    repo_url = apply_github_proxy(KOHYA_REPO_URL, plan.github_proxy)
    cmd = [
        "git",
        "clone",
        "--depth",
        str(plan.git_depth),
        repo_url,
        str(plan.target),
    ]
    _run(cmd, f"clone kohya-ss/sd-scripts -> {plan.target}", progress)


def create_venv(plan: BootstrapPlan, *, progress: ProgressCallback | None = None) -> None:
    try:
        _uv.create_venv(plan.target, python=plan.base_python, progress=progress)
    except RuntimeError as exc:
        raise BootstrapError("create venv", 1) from exc


def upgrade_pip(plan: BootstrapPlan, *, progress: ProgressCallback | None = None) -> None:
    """No-op under uv — uv ships its own resolver and doesn't need pip+wheel.

    Kept on the bootstrap plan so the per-step progress UI keeps lining up;
    we just emit a status line and move on.
    """
    if progress is not None:
        progress("upgrade pip + wheel + setuptools (skipped under uv)")


def install_torch(plan: BootstrapPlan, *, progress: ProgressCallback | None = None) -> None:
    args = [
        f"torch=={plan.torch_version}",
        f"torchvision=={plan.torchvision_version}",
        "--index-url",
        plan.torch_index,
    ]
    try:
        _uv.pip_install(
            plan.venv_python,
            args,
            step=f"install torch=={plan.torch_version} ({plan.cuda_version})",
            progress=progress,
        )
    except RuntimeError as exc:
        raise BootstrapError(f"install torch=={plan.torch_version}", 1) from exc


def install_requirements(plan: BootstrapPlan, *, progress: ProgressCallback | None = None) -> None:
    requirements = plan.target / "requirements.txt"
    if not requirements.is_file():
        msg = f"missing {requirements} - clone may have failed"
        raise BootstrapError("install requirements", 1) from FileNotFoundError(msg)
    try:
        _uv.pip_install(
            plan.venv_python,
            ["-r", str(requirements)],
            step="install kohya requirements.txt",
            progress=progress,
        )
    except RuntimeError as exc:
        raise BootstrapError("install kohya requirements.txt", 1) from exc


def install_xformers(plan: BootstrapPlan, *, progress: ProgressCallback | None = None) -> None:
    if not plan.install_xformers:
        return
    try:
        _uv.pip_install(
            plan.venv_python,
            ["xformers", "--index-url", plan.torch_index],
            step=f"install xformers ({plan.cuda_version})",
            progress=progress,
        )
    except RuntimeError as exc:
        raise BootstrapError(f"install xformers ({plan.cuda_version})", 1) from exc


def bootstrap(plan: BootstrapPlan, *, progress: ProgressCallback | None = None) -> None:
    """Execute every install step in order."""
    _uv.ensure_uv(progress)
    clone(plan, progress=progress)
    create_venv(plan, progress=progress)
    upgrade_pip(plan, progress=progress)
    install_torch(plan, progress=progress)
    install_requirements(plan, progress=progress)
    install_xformers(plan, progress=progress)


def cleanup_partial(plan: BootstrapPlan) -> None:
    """Remove a half-installed target so the user can retry.

    Git pack files inside `.git/objects/pack/*.idx` are written read-only on
    Windows, so the default ``shutil.rmtree`` raises PermissionError on them.
    Hook ``onexc`` (Python 3.12+) / ``onerror`` to flip the read-only bit and
    retry, otherwise the user gets stuck in a 409 loop on every reinstall.
    """
    if not plan.target.exists():
        return

    def _force_writable(func: Any, path: str, _exc_info: Any) -> None:  # noqa: ANN401
        import stat as _stat  # noqa: PLC0415

        try:
            Path(path).chmod(_stat.S_IWRITE | _stat.S_IREAD)
            func(path)
        except OSError:
            pass

    # `onexc` is the Python 3.12 replacement for the deprecated `onerror`.
    if sys.version_info >= (3, 12):
        shutil.rmtree(plan.target, onexc=_force_writable)
    else:
        shutil.rmtree(plan.target, onerror=_force_writable)
