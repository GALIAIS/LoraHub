"""Install the vendored Ostris AI Toolkit venv."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lorahub.core.backends._common import installer as _common
from lorahub.core.backends._common.installer import (
    DEFAULT_CUDA,
    DEFAULT_TORCH,
    DEFAULT_TORCHVISION,
    ProgressCallback,
)
from lorahub.core.backends.errors import BootstrapError
from lorahub.core.toolchain import uv as _uv

AI_TOOLKIT_REPO_URL = "https://github.com/ostris/ai-toolkit"


@dataclass(frozen=True, slots=True)
class BootstrapPlan:
    target: Path
    cuda_version: str = DEFAULT_CUDA
    torch_version: str = DEFAULT_TORCH
    torchvision_version: str = DEFAULT_TORCHVISION
    base_python: Path | None = None
    pypi_index: str | None = None
    torch_index_base: str | None = None

    @property
    def venv_python(self) -> Path:
        return _uv.venv_python(self.target)

    @property
    def torch_index(self) -> str:
        return _common.torch_index_from_base(self.torch_index_base, self.cuda_version)


def create_venv(plan: BootstrapPlan, *, progress: ProgressCallback | None = None) -> None:
    _common.create_venv(plan, progress=progress)


def install_torch(plan: BootstrapPlan, *, progress: ProgressCallback | None = None) -> None:
    _common.install_torch(plan, progress=progress)


def install_requirements(
    plan: BootstrapPlan, *, progress: ProgressCallback | None = None
) -> None:
    requirements = plan.target / "requirements.txt"
    if not requirements.is_file():
        msg = f"missing {requirements}"
        raise BootstrapError("install ai-toolkit requirements", 1) from FileNotFoundError(msg)
    try:
        _uv.pip_install(
            plan.venv_python,
            ["-r", str(requirements)],
            step="install ai-toolkit requirements.txt",
            progress=progress,
            pypi_index=plan.pypi_index,
        )
    except RuntimeError as exc:
        raise BootstrapError("install ai-toolkit requirements.txt", 1) from exc


def bootstrap(plan: BootstrapPlan, *, progress: ProgressCallback | None = None) -> None:
    if not (plan.target / "run.py").is_file():
        msg = f"vendored ai-toolkit copy is missing run.py: {plan.target}"
        raise BootstrapError("bootstrap ai-toolkit", 1) from FileNotFoundError(msg)
    _uv.ensure_uv(progress)
    create_venv(plan, progress=progress)
    install_torch(plan, progress=progress)
    install_requirements(plan, progress=progress)


def cleanup_partial(plan: BootstrapPlan) -> None:
    import shutil

    for venv in (plan.target / "venv", plan.target / ".venv"):
        if venv.is_dir():
            shutil.rmtree(venv, ignore_errors=True)


__all__ = [
    "AI_TOOLKIT_REPO_URL",
    "BootstrapError",
    "BootstrapPlan",
    "ProgressCallback",
    "bootstrap",
    "cleanup_partial",
    "create_venv",
    "install_requirements",
    "install_torch",
]
