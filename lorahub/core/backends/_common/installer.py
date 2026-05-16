"""Shared install-step helpers for backend bootstrappers.

Both `kohya.installer` and `diffusion_pipe.installer` clone a Git repo,
build a uv venv, install pinned torch wheels, then install the rest of the
backend's `requirements.txt` -- always with the same progress callback shape
and the same error wrapping. The functions here factor that out and take a
plan-shaped object via the `BootstrapPlanLike` protocol so each backend can
keep its own `BootstrapPlan` dataclass with its own extras (xformers vs
deepspeed, etc.).
"""

from __future__ import annotations

import collections
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from lorahub.core.backends.errors import BootstrapError
from lorahub.core.toolchain import uv as _uv

ProgressCallback = Callable[[str], None]

DEFAULT_TORCH = "2.6.0"
DEFAULT_TORCHVISION = "0.21.0"
DEFAULT_CUDA = "cu124"
DEFAULT_DEPTH = 1


class BootstrapPlanLike(Protocol):
    """The minimum shape every backend's BootstrapPlan must expose.

    Both backends' frozen dataclasses already satisfy this implicitly; the
    protocol just documents the surface so the helpers below can stay
    backend-agnostic.
    """

    target: Path
    cuda_version: str
    torch_version: str
    torchvision_version: str
    git_depth: int
    github_proxy: str | None
    base_python: Path | None
    pypi_index: str | None

    @property
    def venv_python(self) -> Path: ...

    @property
    def torch_index(self) -> str: ...


def run_step(
    cmd: list[str],
    step: str,
    progress: ProgressCallback | None,
) -> None:
    """Run a non-package subprocess (typically `git clone`) with stderr capture.

    Streams the subprocess's stderr **line by line** to ``progress`` so the
    dashboard can surface git's own progress output (e.g. ``Receiving objects:
    23% (...)``) while the clone is still running, instead of waiting for the
    process to exit. Reports the step name through ``progress`` before
    launching, and on a non-zero exit code attaches the last 12 stderr lines
    to the progress stream so the UI can surface a useful error message.
    """
    if progress is not None:
        progress(step)
    proc = subprocess.Popen(  # noqa: S603 -- caller controls argv
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # line-buffered so we get progress lines as they're written
    )
    tail: collections.deque[str] = collections.deque(maxlen=12)
    assert proc.stderr is not None  # noqa: S101 -- PIPE above guarantees this
    for raw_line in proc.stderr:
        line = raw_line.rstrip()
        if not line:
            continue
        tail.append(line)
        if progress is not None:
            # Forward each line so the dashboard sees git's own progress
            # output. Indent with two spaces so multiple concurrent steps
            # stay readable in a combined log stream.
            progress(f"  {line}")
    rc = proc.wait()
    if rc != 0:
        if progress is not None and tail:
            progress(f"{step} failed (exit {rc}):\n" + "\n".join(tail))
        raise BootstrapError(step, rc)


def clone_repo(
    plan: BootstrapPlanLike,
    *,
    repo_url: str,
    label: str,
    progress: ProgressCallback | None = None,
) -> None:
    """Run ``git clone --depth ... <repo_url> <plan.target>``.

    Refuses to clone into a non-empty directory and applies the optional
    GitHub proxy from settings to the URL before invoking git.
    """
    if plan.target.exists() and any(plan.target.iterdir()):
        msg = f"target directory is not empty: {plan.target}"
        raise BootstrapError("clone", 1) from FileExistsError(msg)
    plan.target.parent.mkdir(parents=True, exist_ok=True)
    from lorahub.api.settings import apply_github_proxy  # noqa: PLC0415

    proxied = apply_github_proxy(repo_url, plan.github_proxy)
    cmd = [
        "git",
        "clone",
        "--progress",
        "--depth",
        str(plan.git_depth),
        proxied,
        str(plan.target),
    ]
    run_step(cmd, f"clone {label} -> {plan.target}", progress)


def create_venv(
    plan: BootstrapPlanLike,
    *,
    progress: ProgressCallback | None = None,
) -> None:
    try:
        _uv.create_venv(plan.target, python=plan.base_python, progress=progress)
    except RuntimeError as exc:
        raise BootstrapError("create venv", 1) from exc


def upgrade_pip(plan: BootstrapPlanLike, *, progress: ProgressCallback | None = None) -> None:
    """No-op under uv -- uv ships its own resolver and skips pip+wheel.

    Kept on the bootstrap plan so the per-step progress UI keeps lining up;
    we just emit a status line and move on.
    """
    if progress is not None:
        progress("upgrade pip + wheel + setuptools (skipped under uv)")
    _ = plan  # keep signature symmetric with sibling helpers


def install_torch(
    plan: BootstrapPlanLike,
    *,
    progress: ProgressCallback | None = None,
) -> None:
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


def cleanup_partial(target: Path) -> None:
    """Remove a half-installed checkout so the user can retry.

    Git pack files inside ``.git/objects/pack/*.idx`` are written read-only
    on Windows, so the default ``shutil.rmtree`` raises PermissionError on
    them. Hook ``onexc`` (Python 3.12+) / ``onerror`` to flip the read-only
    bit and retry, otherwise the user gets stuck in a 409 loop on every
    reinstall.
    """
    if not target.exists():
        return

    def _force_writable(func: Any, path: str, _exc_info: Any) -> None:  # noqa: ANN401
        import stat as _stat  # noqa: PLC0415

        try:
            Path(path).chmod(_stat.S_IWRITE | _stat.S_IREAD)
            func(path)
        except OSError:
            pass

    if sys.version_info >= (3, 12):
        shutil.rmtree(target, onexc=_force_writable)
    else:
        shutil.rmtree(target, onerror=_force_writable)


__all__ = [
    "DEFAULT_CUDA",
    "DEFAULT_DEPTH",
    "DEFAULT_TORCH",
    "DEFAULT_TORCHVISION",
    "BootstrapPlanLike",
    "ProgressCallback",
    "cleanup_partial",
    "clone_repo",
    "create_venv",
    "install_torch",
    "run_step",
    "upgrade_pip",
]
