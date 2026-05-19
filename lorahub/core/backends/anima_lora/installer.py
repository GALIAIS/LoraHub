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

import sys
from dataclasses import dataclass
from pathlib import Path

from lorahub.core.backends._common.installer import ProgressCallback
from lorahub.core.backends.errors import BootstrapError
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

    @property
    def venv_dir(self) -> Path:
        return self.target / ".venv"

    @property
    def venv_python(self) -> Path:
        if sys.platform == "win32":
            return self.venv_dir / "Scripts" / "python.exe"
        return self.venv_dir / "bin" / "python"


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

    args = ["sync", "--directory", str(plan.target)]
    if plan.base_python is not None:
        args += ["--python", str(plan.base_python)]
    if plan.pypi_index:
        # Forward only when the user opted in via Settings. We use
        # `--default-index` (uv ≥ 0.4) instead of the deprecated
        # `--index-url` so future uv versions don't drop support.
        # The named ``pytorch-cu124`` index in anima's pyproject still
        # wins for torch + torchvision via ``[tool.uv.sources]`` — only
        # the non-torch packages (accelerate, diffusers, transformers,
        # ~30 deps) get routed through the user's mirror, which is
        # still a useful speedup in regions where pypi.org is slow.
        args += ["--default-index", plan.pypi_index]
    label = f"uv sync -> {plan.venv_dir}"
    try:
        _uv.run_uv(args, step=label, progress=progress)
    except RuntimeError as exc:
        raise BootstrapError("uv sync anima_lora", 1) from exc


def bootstrap(plan: BootstrapPlan, *, progress: ProgressCallback | None = None) -> None:
    """Execute every install step in order.

    Single step today — ``uv sync`` reads the vendored uv.lock and
    materialises the venv. Kept as a wrapping function so the install
    session can swap to a multi-step pipeline if anima_lora ever grows
    out-of-band setup (e.g. ``make download-models``).
    """
    sync(plan, progress=progress)


def cleanup_partial(plan: BootstrapPlan) -> None:
    """Remove a half-installed .venv so the user can retry.

    Symmetrical to dp / kohya cleanup_partial. anima_lora source itself
    is vendored and must never be removed — only ``.venv``.
    """
    import shutil

    if plan.venv_dir.is_dir():
        shutil.rmtree(plan.venv_dir, ignore_errors=True)


__all__ = [
    "ANIMA_LORA_REPO_URL",
    "BootstrapError",
    "BootstrapPlan",
    "ProgressCallback",
    "bootstrap",
    "cleanup_partial",
    "sync",
]
