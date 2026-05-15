"""Automate the tdrussell/diffusion-pipe install: clone + venv + requirements + deepspeed.

Mirrors the public surface of `lorahub.core.backends.kohya.installer` so the
bootstrap session can drive either backend with the same plumbing. Each step
runs a single subprocess; failure raises `BootstrapError` annotated with the
step name and exit code.

All package operations go through ``lorahub.core.toolchain.uv``, so torch and
its sibling wheels are hard-linked from the shared uv cache instead of
re-downloaded into every backend's venv.
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
    # Optional HTTPS prefix that rewrites `https://github.com/...` URLs at
    # clone time (e.g. "https://gh-proxy.org"). Empty means direct.
    github_proxy: str | None = None

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

    repo_url = apply_github_proxy(DIFFUSION_PIPE_REPO_URL, plan.github_proxy)
    cmd = [
        "git",
        "clone",
        "--depth",
        str(plan.git_depth),
        repo_url,
        str(plan.target),
    ]
    _run(cmd, f"clone tdrussell/diffusion-pipe -> {plan.target}", progress)


def create_venv(plan: BootstrapPlan, *, progress: ProgressCallback | None = None) -> None:
    try:
        _uv.create_venv(plan.target, progress=progress)
    except RuntimeError as exc:
        raise BootstrapError("create venv", 1) from exc


def upgrade_pip(plan: BootstrapPlan, *, progress: ProgressCallback | None = None) -> None:
    """No-op under uv."""
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

    # diffusion-pipe pins `deepspeed` in requirements.txt, but DeepSpeed
    # ships no Windows wheel — pip then tries to build from source which
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

    if sys.version_info >= (3, 12):
        shutil.rmtree(plan.target, onexc=_force_writable)
    else:
        shutil.rmtree(plan.target, onerror=_force_writable)


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
