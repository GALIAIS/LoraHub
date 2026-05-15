"""Automate the kohya-ss/sd-scripts install — clone + venv + PyTorch + requirements + xformers.

Mirrors the steps from kohya's official Windows README so users don't have to
shell out themselves. Each step is a stand-alone function that runs a single
subprocess; on failure the exception bubbles up with the failing step name so
callers can show a clear error.
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

    @property
    def venv_python(self) -> Path:
        if sys.platform == "win32":
            return self.target / "venv" / "Scripts" / "python.exe"
        return self.target / "venv" / "bin" / "python"

    @property
    def torch_index(self) -> str:
        return f"https://download.pytorch.org/whl/{self.cuda_version}"


ProgressCallback = Callable[[str], None]


def _run(cmd: list[str], step: str, progress: ProgressCallback | None) -> None:
    if progress is not None:
        progress(step)
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
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
    cmd = [sys.executable, "-m", "venv", str(plan.target / "venv")]
    _run(cmd, "create venv", progress)


def upgrade_pip(plan: BootstrapPlan, *, progress: ProgressCallback | None = None) -> None:
    cmd = [str(plan.venv_python), "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"]
    _run(cmd, "upgrade pip + wheel + setuptools", progress)


def install_torch(plan: BootstrapPlan, *, progress: ProgressCallback | None = None) -> None:
    cmd = [
        str(plan.venv_python),
        "-m",
        "pip",
        "install",
        f"torch=={plan.torch_version}",
        f"torchvision=={plan.torchvision_version}",
        "--index-url",
        plan.torch_index,
    ]
    _run(cmd, f"install torch=={plan.torch_version} ({plan.cuda_version})", progress)


def install_requirements(plan: BootstrapPlan, *, progress: ProgressCallback | None = None) -> None:
    requirements = plan.target / "requirements.txt"
    if not requirements.is_file():
        msg = f"missing {requirements} - clone may have failed"
        raise BootstrapError("install requirements", 1) from FileNotFoundError(msg)
    cmd = [
        str(plan.venv_python),
        "-m",
        "pip",
        "install",
        "--upgrade",
        "-r",
        str(requirements),
    ]
    _run(cmd, "install kohya requirements.txt", progress)


def install_xformers(plan: BootstrapPlan, *, progress: ProgressCallback | None = None) -> None:
    if not plan.install_xformers:
        return
    cmd = [
        str(plan.venv_python),
        "-m",
        "pip",
        "install",
        "xformers",
        "--index-url",
        plan.torch_index,
    ]
    _run(cmd, f"install xformers ({plan.cuda_version})", progress)


def bootstrap(plan: BootstrapPlan, *, progress: ProgressCallback | None = None) -> None:
    """Execute every install step in order."""
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
        import os as _os  # noqa: PLC0415
        import stat as _stat  # noqa: PLC0415

        try:
            _os.chmod(path, _stat.S_IWRITE | _stat.S_IREAD)
            func(path)
        except OSError:
            pass

    # `onexc` is the Python 3.12 replacement for the deprecated `onerror`.
    if sys.version_info >= (3, 12):
        shutil.rmtree(plan.target, onexc=_force_writable)
    else:
        shutil.rmtree(plan.target, onerror=_force_writable)
