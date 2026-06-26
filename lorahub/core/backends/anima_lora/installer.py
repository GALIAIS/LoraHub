"""Automate anima_lora venv installation via ``uv sync``.

Different shape from kohya / dp's installer: anima_lora ships
**vendored** under ``external/anima_lora/`` so there is no clone step.
What we manage is the **venv** — anima_lora's ``pyproject.toml``
declares ``requires-python = "==3.13.*"`` plus torch 2.11 nightly /
2.12 nightly that the LoraHub main venv (3.11/3.12) cannot share.
``uv sync`` reads the vendored ``uv.lock`` and produces a fresh
``.venv/`` next to the source tree, fetching CPython 3.13 if the
host doesn't already have it.

The single bootstrap step (``sync``) is enough — torch + accelerate +
diffusers all land via the lock file. We expose it through the same
``BootstrapPlan`` / ``bootstrap`` shape kohya / dp use so the install
session in the API can drive any backend uniformly.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from lorahub.core.backends._common import installer as _common
from lorahub.core.backends._common.installer import ProgressCallback
from lorahub.core.backends.errors import BootstrapError
from lorahub.core.paths import project_root
from lorahub.core.toolchain import uv as _uv

# anima_lora is vendored — no remote URL to clone from. We keep this
# constant for API parity with the other two backends (the registry
# descriptor expects a ``repo_url`` field).
ANIMA_LORA_REPO_URL = "https://github.com/sorryhyun/anima_lora"


@dataclass(frozen=True, slots=True)
class BootstrapPlan:
    """Install plan for the anima_lora venv.

    ``target`` is the vendored copy directory (``external/anima_lora``).
    ``base_python`` is optional — when None, ``uv sync`` reads the
    ``requires-python`` constraint from ``pyproject.toml`` and fetches
    CPython 3.13 itself.

    Note: anima's ``pyproject.toml`` declares a named ``pytorch-cu124``
    index and pins torch / torchvision to it via ``[tool.uv.sources]``.
    Without that, ``uv sync`` would silently resolve torch against
    PyPI and land a ``+cpu`` build on Windows — preprocess / train
    would then hang on CPU tensor ops. Don't drop the source map
    upstream without rechecking ``cache_latents.py`` throughput.
    """

    target: Path
    # Optional: pin the base interpreter `uv sync` builds the venv on.
    # Leave None to let uv auto-fetch CPython matching `requires-python`.
    base_python: Path | None = None
    # Optional PyPI index URL forwarded to `uv sync` via ``--default-index``.
    # Useful in regions where the default PyPI is slow.
    pypi_index: str | None = None
    # Optional DeepSpeed add-on. Needed only for
    # backend.distributed.strategy=deepspeed_zero; skipped on Windows
    # because there is no reliable wheel and source builds require a
    # CUDA toolkit + MSVC setup outside LoraHub's control.
    install_deepspeed: bool = True
    # Optional post-sync torch wheel override. Used for hosts whose driver
    # cannot load anima_lora's upstream CUDA pin.
    torch_override: bool = sys.platform == "win32"
    cuda_version: str = "cu128"
    torch_version: str = "2.11.0"
    torchvision_version: str = "0.26.0"
    torch_index_base: str | None = None

    @property
    def venv_dir(self) -> Path:
        return self.target / ".venv"

    @property
    def venv_python(self) -> Path:
        if sys.platform == "win32":
            return self.venv_dir / "Scripts" / "python.exe"
        return self.venv_dir / "bin" / "python"

    @property
    def uv_cache_dir(self) -> Path:
        return project_root() / ".cache" / "uv"

    @property
    def temp_dir(self) -> Path:
        return project_root() / ".cache" / "tmp"

    @property
    def torch_index(self) -> str:
        return _common.torch_index_from_base(self.torch_index_base, self.cuda_version)


def sync(plan: BootstrapPlan, *, progress: ProgressCallback | None = None) -> None:
    """Run ``uv sync`` inside the vendored anima_lora directory.

    Resolves the dependency graph from ``uv.lock`` and creates / updates
    ``.venv/`` to match. Idempotent: re-running on an up-to-date venv
    is a fast no-op.
    """
    if not (plan.target / "pyproject.toml").is_file():
        msg = (
            f"missing {plan.target / 'pyproject.toml'} - is the vendored "
            "external/anima_lora copy intact?"
        )
        raise BootstrapError("uv sync anima_lora", 1) from FileNotFoundError(msg)

    if plan.base_python is not None:
        python_args = ["--python", str(plan.base_python)]
    else:
        python_args = []
    label = f"uv sync -> {plan.venv_dir}"
    env = _uv_sync_env(plan)
    errors: list[str] = []
    indexes = _common.torch_index_candidates(plan.torch_index_base, plan.cuda_version)
    for idx, torch_index in enumerate(indexes):
        args = [
            "sync",
            "--no-dev",
            "--index",
            torch_index,
            "--directory",
            str(plan.target),
            *python_args,
        ]
        step = label if idx == 0 else f"{label} (torch source {idx + 1}/{len(indexes)})"
        try:
            # ``run_uv`` injects ``--default-index <plan.pypi_index>`` for
            # non-torch packages. The explicit ``--index <torch_index>``
            # supplies CUDA wheels for torch / torchvision and is tried
            # with the same mirror fallback list used by the other backends.
            _uv.run_uv(
                args,
                step=step,
                progress=progress,
                pypi_index=plan.pypi_index,
                env=env,
            )
            return
        except RuntimeError as exc:
            errors.append(f"{torch_index}: {exc}")
            if idx + 1 < len(indexes) and progress is not None:
                progress(f"{step} failed; trying {indexes[idx + 1]}")
    detail = "\n".join(errors[-4:])
    raise BootstrapError("uv sync anima_lora", 1) from RuntimeError(detail)


def _uv_sync_env(plan: BootstrapPlan) -> dict[str, str]:
    """Keep uv's large torch extraction cache on the project volume by default."""
    import os

    env: dict[str, str] = {}
    if not os.environ.get("UV_CACHE_DIR"):
        plan.uv_cache_dir.mkdir(parents=True, exist_ok=True)
        env["UV_CACHE_DIR"] = str(plan.uv_cache_dir)

    if sys.platform == "win32":
        if not os.environ.get("TEMP"):
            plan.temp_dir.mkdir(parents=True, exist_ok=True)
            env["TEMP"] = str(plan.temp_dir)
        if not os.environ.get("TMP"):
            plan.temp_dir.mkdir(parents=True, exist_ok=True)
            env["TMP"] = str(plan.temp_dir)
    elif not os.environ.get("TMPDIR"):
        plan.temp_dir.mkdir(parents=True, exist_ok=True)
        env["TMPDIR"] = str(plan.temp_dir)

    return env


def install_deepspeed(
    plan: BootstrapPlan, *, progress: ProgressCallback | None = None
) -> None:
    if not plan.install_deepspeed:
        return
    if sys.platform == "win32":
        if progress is not None:
            progress(
                "skip deepspeed: no Windows wheel available. "
                "DeepSpeed ZeRO requires WSL2/Linux or a manual CUDA/MSVC build."
            )
        return
    try:
        _uv.pip_install(
            plan.venv_python,
            ["deepspeed"],
            step="install anima_lora deepspeed",
            progress=progress,
            pypi_index=plan.pypi_index,
        )
    except RuntimeError as exc:
        raise BootstrapError("install anima_lora deepspeed", 1) from exc


def install_bitsandbytes(
    plan: BootstrapPlan, *, progress: ProgressCallback | None = None
) -> None:
    if sys.platform == "win32":
        return
    try:
        _uv.pip_install(
            plan.venv_python,
            ["bitsandbytes"],
            step="install anima_lora bitsandbytes",
            progress=progress,
            pypi_index=plan.pypi_index,
        )
    except RuntimeError as exc:
        raise BootstrapError("install anima_lora bitsandbytes", 1) from exc


def install_torch_override(
    plan: BootstrapPlan, *, progress: ProgressCallback | None = None
) -> None:
    if not plan.torch_override:
        return
    _ensure_no_torch_override_running(plan)
    torch_version = _with_cuda_local_version(plan.torch_version, plan.cuda_version)
    torchvision_version = _with_cuda_local_version(
        plan.torchvision_version,
        plan.cuda_version,
    )
    args = [
        f"torch=={torch_version}",
        f"torchvision=={torchvision_version}",
        "--index-url",
        plan.torch_index,
    ]
    try:
        _common.pip_install_with_torch_index_fallback(
            plan,
            args,
            step=(
                "override anima_lora torch=="
                f"{torch_version} ({plan.cuda_version})"
            ),
            progress=progress,
        )
    except RuntimeError as exc:
        raise BootstrapError(
            f"override anima_lora torch=={torch_version}",
            1,
        ) from exc


def _with_cuda_local_version(version: str, cuda: str) -> str:
    """Pin CUDA wheel local version so uv cannot satisfy it with torch+cpu."""
    if "+" in version:
        return version
    return f"{version}+{cuda}"


def _ensure_no_torch_override_running(plan: BootstrapPlan) -> None:
    if sys.platform == "win32":
        return
    try:
        result = subprocess.run(
            ["pgrep", "-af", "uv.*pip install.*torch=="],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return
    if result.returncode not in (0, 1):
        return
    needle = str(plan.venv_python)
    current_pid = str(__import__("os").getpid())
    lines = [
        line
        for line in result.stdout.splitlines()
        if needle in line and not line.startswith(current_pid + " ")
    ]
    if lines:
        msg = "another anima_lora torch install is still running:\n" + "\n".join(lines[:3])
        raise BootstrapError("override anima_lora torch", 1) from RuntimeError(msg)


def bootstrap(plan: BootstrapPlan, *, progress: ProgressCallback | None = None) -> None:
    """Execute every install step in order.

    ``uv sync`` reads the vendored uv.lock and materialises the venv.
    DeepSpeed is installed as an optional post-sync add-on because it is
    only needed for ZeRO and is not part of upstream anima_lora's lock.
    """
    sync(plan, progress=progress)
    install_torch_override(plan, progress=progress)
    install_bitsandbytes(plan, progress=progress)
    install_deepspeed(plan, progress=progress)


def cleanup_partial(plan: BootstrapPlan) -> None:
    """Remove a half-installed .venv plus the models junction.

    Symmetrical to dp / kohya cleanup_partial. anima_lora source itself
    is vendored and must never be removed — only ``.venv`` and the
    ``models`` symlink/junction we create at install time. The models
    junction needs explicit removal because Windows ``mklink /J`` and
    POSIX symlinks both survive ``rmtree`` of their target dir; if the
    user re-runs ``force=true`` install the link could otherwise still
    point at a stale tree from a previous install run.
    """
    import shutil

    if plan.venv_dir.is_dir():
        shutil.rmtree(plan.venv_dir, ignore_errors=True)

    # ``<repo>/models`` is the link that ``_link_anima_models_dir``
    # creates pointing at the unified ``<lorahub_root>/models/``. We
    # don't unlink it when it's a real directory the user populated
    # manually (no junction marker on dirs they own); only delete
    # symlinks / junctions.
    models_link = plan.target / "models"
    try:
        if models_link.is_symlink():
            models_link.unlink()
        elif models_link.exists():
            # Windows junctions report ``is_dir() == True`` and
            # ``is_symlink() == False``. Detect via the reparse
            # bit so we only remove links, not real directories.
            import os as _os  # noqa: PLC0415

            try:
                stat = _os.lstat(models_link)
                # FILE_ATTRIBUTE_REPARSE_POINT (0x400) marks both
                # symlinks and junctions on Windows; pure dirs have
                # ``st_file_attributes`` without this bit.
                attrs = getattr(stat, "st_file_attributes", 0)
                if attrs & 0x400:
                    _os.rmdir(models_link)
            except OSError:
                pass
    except OSError:
        # cleanup_partial must never block the retry path; swallow.
        pass


__all__ = [
    "ANIMA_LORA_REPO_URL",
    "BootstrapError",
    "BootstrapPlan",
    "ProgressCallback",
    "bootstrap",
    "cleanup_partial",
    "install_bitsandbytes",
    "install_deepspeed",
    "install_torch_override",
    "sync",
]
