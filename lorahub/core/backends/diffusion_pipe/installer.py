"""Automate the tdrussell/diffusion-pipe install: clone + venv + requirements + deepspeed.

Mirrors the public surface of `lorahub.core.backends.kohya.installer` so the
bootstrap session can drive either backend with the same plumbing. Each step
runs a single subprocess; failure raises `BootstrapError` annotated with the
step name and exit code. Shared steps (clone, venv, torch) compose helpers
from ``lorahub.core.backends._common.installer``.

All package operations go through ``lorahub.core.toolchain.uv``, so torch
and its sibling wheels are hard-linked from the shared uv cache instead of
re-downloaded into every backend's venv.
"""

from __future__ import annotations

import subprocess  # noqa: F401  -- re-exported so tests can monkeypatch it
import sys
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

DIFFUSION_PIPE_REPO_URL = "https://github.com/tdrussell/diffusion-pipe.git"


@dataclass(frozen=True, slots=True)
class BootstrapPlan:
    target: Path
    cuda_version: str = DEFAULT_CUDA
    torch_version: str = DEFAULT_TORCH
    torchvision_version: str = DEFAULT_TORCHVISION
    install_deepspeed: bool = True
    git_depth: int = DEFAULT_DEPTH
    # Optional HTTPS prefix that rewrites `https://github.com/...` URLs at
    # clone time (e.g. "https://gh-proxy.org"). Empty means direct.
    github_proxy: str | None = None
    # Optional path to a CPython executable used as the venv base.
    base_python: Path | None = None
    # Optional PyPI index URL for `uv pip install`.
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
        repo_url=DIFFUSION_PIPE_REPO_URL,
        label="tdrussell/diffusion-pipe",
        # ComfyUI / HunyuanVideo / Cosmos / etc. live as submodules and
        # are import-time dependencies of `utils/dataset.py`.
        recurse_submodules=True,
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

    # diffusion-pipe pins `deepspeed` in requirements.txt, but DeepSpeed
    # ships no Windows wheel -- pip then tries to build from source which
    # needs CUDA toolkit + MSVC and almost always fails. Strip every line
    # that mentions deepspeed and run the rest; the dedicated
    # install_deepspeed step takes care of it (and skips on Windows).
    filtered = plan.target / "requirements.lorahub.txt"
    raw = requirements.read_text(encoding="utf-8")
    kept: list[str] = []
    skipped: list[str] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "deepspeed" in stripped.lower():
            skipped.append(stripped)
            continue
        kept.append(line)
    filtered.write_text("\n".join(kept) + "\n", encoding="utf-8")
    if progress is not None and skipped:
        progress(f"skipping from requirements: {', '.join(skipped)} (handled separately)")

    try:
        _uv.pip_install(
            plan.venv_python,
            ["-r", str(filtered)],
            step="install diffusion-pipe requirements.txt",
            progress=progress,
            pypi_index=plan.pypi_index,
        )
    except RuntimeError as exc:
        raise BootstrapError("install diffusion-pipe requirements.txt", 1) from exc


def install_deepspeed(plan: BootstrapPlan, *, progress: ProgressCallback | None = None) -> None:
    if not plan.install_deepspeed:
        return
    if sys.platform == "win32":
        if progress is not None:
            progress(
                "skip deepspeed: no Windows wheel available. "
                "DeepSpeed needs CUDA toolkit + MSVC to build from source. "
                "Install manually once your build environment is ready, "
                "or run training under WSL2/Linux."
            )
        return
    try:
        _uv.pip_install(
            plan.venv_python,
            ["deepspeed"],
            step="install deepspeed",
            progress=progress,
            pypi_index=plan.pypi_index,
        )
    except RuntimeError as exc:
        raise BootstrapError("install deepspeed", 1) from exc


def bootstrap(plan: BootstrapPlan, *, progress: ProgressCallback | None = None) -> None:
    """Execute every install step in order."""
    _uv.ensure_uv(progress)
    clone(plan, progress=progress)
    create_venv(plan, progress=progress)
    upgrade_pip(plan, progress=progress)
    install_torch(plan, progress=progress)
    install_requirements(plan, progress=progress)
    install_deepspeed(plan, progress=progress)


def cleanup_partial(plan: BootstrapPlan) -> None:
    """Remove a half-installed target so the user can retry."""
    _common.cleanup_partial(plan.target)


__all__ = [
    "DEFAULT_CUDA",
    "DEFAULT_DEPTH",
    "DEFAULT_TORCH",
    "DEFAULT_TORCHVISION",
    "DIFFUSION_PIPE_REPO_URL",
    "BootstrapError",
    "BootstrapPlan",
    "ProgressCallback",
    "bootstrap",
    "cleanup_partial",
    "clone",
    "create_venv",
    "install_deepspeed",
    "install_requirements",
    "install_torch",
    "upgrade_pip",
]
