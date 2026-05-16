"""Automate the kohya-ss/sd-scripts install: clone + venv + torch + requirements + xformers.

Mirrors the steps from kohya's official Windows README so users don't have to
shell out themselves. The plumbing for clone/venv/torch is shared with the
diffusion-pipe backend through ``lorahub.core.backends._common.installer``;
this module only carries kohya-specific knobs (the repo URL, xformers, the
location of requirements.txt).

All package operations go through ``lorahub.core.toolchain.uv`` (uv venv +
uv pip install). uv is hard-link-aware and shares its global wheel cache
across every venv we ever build, so installing a 6 GB torch into both kohya
and diffusion-pipe costs roughly the size of one install on disk.
"""

from __future__ import annotations

import subprocess  # noqa: F401  -- re-exported for tests that monkeypatch it
from dataclasses import dataclass
from pathlib import Path

from lorahub.core.backends._common import installer as _common
from lorahub.core.backends._common.installer import (
    DEFAULT_CUDA,
    DEFAULT_DEPTH,
    DEFAULT_TORCH,
    DEFAULT_TORCHVISION,
    ProgressCallback,
)
from lorahub.core.backends.errors import BootstrapError
from lorahub.core.toolchain import uv as _uv

KOHYA_REPO_URL = "https://github.com/kohya-ss/sd-scripts.git"


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
    # Optional PyPI index URL for `uv pip install` (e.g. TUNA mirror).
    # Only applied to plain dependency installs; torch / xformers keep
    # their pinned --index-url because those wheels live on a separate
    # CDN regardless of which PyPI mirror the user picked.
    pypi_index: str | None = None

    @property
    def venv_python(self) -> Path:
        return _uv.venv_python(self.target)

    @property
    def torch_index(self) -> str:
        return f"https://download.pytorch.org/whl/{self.cuda_version}"


def clone(plan: BootstrapPlan, *, progress: ProgressCallback | None = None) -> None:
    _common.clone_repo(
        plan,
        repo_url=KOHYA_REPO_URL,
        label="kohya-ss/sd-scripts",
        progress=progress,
    )


def create_venv(plan: BootstrapPlan, *, progress: ProgressCallback | None = None) -> None:
    _common.create_venv(plan, progress=progress)


def upgrade_pip(plan: BootstrapPlan, *, progress: ProgressCallback | None = None) -> None:
    _common.upgrade_pip(plan, progress=progress)


def install_torch(plan: BootstrapPlan, *, progress: ProgressCallback | None = None) -> None:
    _common.install_torch(plan, progress=progress)


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
            pypi_index=plan.pypi_index,
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
    """Remove a half-installed target so the user can retry."""
    _common.cleanup_partial(plan.target)


__all__ = [
    "DEFAULT_CUDA",
    "DEFAULT_DEPTH",
    "DEFAULT_TORCH",
    "DEFAULT_TORCHVISION",
    "KOHYA_REPO_URL",
    "BootstrapError",
    "BootstrapPlan",
    "ProgressCallback",
    "bootstrap",
    "cleanup_partial",
    "clone",
    "create_venv",
    "install_requirements",
    "install_torch",
    "install_xformers",
    "upgrade_pip",
]
