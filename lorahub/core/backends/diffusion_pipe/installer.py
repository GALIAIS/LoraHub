"""Automate the tdrussell/diffusion-pipe install: clone + venv + requirements + deepspeed.

Mirrors the public surface of `lorahub.core.backends.kohya.installer` so the
bootstrap session can drive either backend with the same plumbing. Each step
runs a single subprocess; failure raises `BootstrapError` annotated with the
step name and exit code.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from lorahub.core.backends.errors import BootstrapError

DIFFUSION_PIPE_REPO_URL = "https://github.com/tdrussell/diffusion-pipe.git"
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
    install_deepspeed: bool = True
    git_depth: int = DEFAULT_DEPTH

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
    cmd = [
        "git",
        "clone",
        "--depth",
        str(plan.git_depth),
        DIFFUSION_PIPE_REPO_URL,
        str(plan.target),
    ]
    _run(cmd, f"clone tdrussell/diffusion-pipe -> {plan.target}", progress)


def create_venv(plan: BootstrapPlan, *, progress: ProgressCallback | None = None) -> None:
    cmd = [sys.executable, "-m", "venv", str(plan.target / "venv")]
    _run(cmd, "create venv", progress)


def upgrade_pip(plan: BootstrapPlan, *, progress: ProgressCallback | None = None) -> None:
    cmd = [
        str(plan.venv_python),
        "-m",
        "pip",
        "install",
        "--upgrade",
        "pip",
        "wheel",
        "setuptools",
    ]
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
    _run(cmd, "install diffusion-pipe requirements.txt", progress)


def install_deepspeed(plan: BootstrapPlan, *, progress: ProgressCallback | None = None) -> None:
    if not plan.install_deepspeed:
        return
    cmd = [str(plan.venv_python), "-m", "pip", "install", "deepspeed"]
    _run(cmd, "install deepspeed", progress)


def bootstrap(plan: BootstrapPlan, *, progress: ProgressCallback | None = None) -> None:
    """Execute every install step in order."""
    clone(plan, progress=progress)
    create_venv(plan, progress=progress)
    upgrade_pip(plan, progress=progress)
    install_torch(plan, progress=progress)
    install_requirements(plan, progress=progress)
    install_deepspeed(plan, progress=progress)


def cleanup_partial(plan: BootstrapPlan) -> None:
    """Remove a half-installed target so the user can retry."""
    if plan.target.exists():
        shutil.rmtree(plan.target, ignore_errors=True)


__all__ = [
    "BootstrapError",
    "BootstrapPlan",
    "DIFFUSION_PIPE_REPO_URL",
    "bootstrap",
    "cleanup_partial",
    "clone",
    "create_venv",
    "install_deepspeed",
    "install_requirements",
    "install_torch",
    "upgrade_pip",
]
