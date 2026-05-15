"""uv toolchain helpers.

`uv` (https://docs.astral.sh/uv) is the fast Rust-based replacement we use for
every package operation: creating virtual environments, installing packages,
upgrading them. Compared to the stock `python -m venv` + `pip` combo it's
roughly 5-20× faster *and* hard-links its global wheel cache into per-project
site-packages, which means installing the same large dependency (`torch`,
`xformers`, `deepspeed`) across multiple backend venvs costs effectively
zero extra disk after the first install.

The module exposes three responsibilities:

1. **Discovery / bootstrap** -- `ensure_uv()` returns a path to a usable
   `uv` executable, installing it on demand into a stable user-data
   location when the system PATH doesn't already have one.
2. **venv** -- `create_venv()` shells out to `uv venv` to materialise an
   environment under `<target>/venv`.
3. **Install** -- `pip_install()` runs `uv pip install` against a specific
   venv's interpreter (using ``--python`` so the install lands in the right
   tree even when the parent shell has another interpreter active).

Every helper takes an optional ``progress`` callback so installers can
stream their step descriptions through the API event bus.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from platformdirs import user_data_path

ProgressCallback = Callable[[str], None]


# Where we drop the bootstrapped `uv` binary when the system doesn't have
# one. We reuse the LoraHub user-data directory so it persists across
# venvs and across LoraHub versions.
def _bin_dir() -> Path:
    return user_data_path("lorahub", "lorahub") / "bin"


def _local_uv_path() -> Path:
    name = "uv.exe" if sys.platform == "win32" else "uv"
    return _bin_dir() / name


_UV_CACHED: str | None = None


def find_uv() -> str | None:
    """Return the path to a discoverable `uv` binary or None if not present."""
    onpath = shutil.which("uv")
    if onpath:
        return onpath
    local = _local_uv_path()
    if local.is_file():
        return str(local)
    return None


def _bootstrap_uv(progress: ProgressCallback | None) -> str:
    """Install `uv` into the LoraHub user-data dir using `pip install --target`.

    We deliberately avoid touching the system or user site-packages; uv ships
    as a single static binary inside the wheel, and installing it into a
    sandbox keeps the user's interpreter clean. After the wheel lands we
    locate the entry-point script and copy it into ``_bin_dir()``.
    """
    if progress is not None:
        progress("uv: bootstrapping toolchain (one-time install)")

    bin_dir = _bin_dir()
    bin_dir.mkdir(parents=True, exist_ok=True)
    target = bin_dir / "_uv_pkg"
    target.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--target",
        str(target),
        "uv",
    ]
    result = subprocess.run(cmd, check=False, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        tail = "\n".join((result.stderr or "").strip().splitlines()[-10:])
        msg = f"failed to install uv via pip (exit {result.returncode}):\n{tail}"
        raise RuntimeError(msg)

    # `uv` ships its console script under {site-packages}/bin or
    # {site-packages}/Scripts depending on the platform; locate it and copy.
    bin_subdir = "Scripts" if sys.platform == "win32" else "bin"
    script_name = "uv.exe" if sys.platform == "win32" else "uv"
    src = target / bin_subdir / script_name
    if not src.is_file():
        # Fallback: scan target/bin for the binary.
        for p in target.rglob(script_name):
            if p.is_file():
                src = p
                break
    if not src.is_file():
        msg = f"installed uv but couldn't locate its binary under {target}"
        raise RuntimeError(msg)

    dst = _local_uv_path()
    shutil.copy2(src, dst)
    if sys.platform != "win32":
        dst.chmod(0o755)
    if progress is not None:
        progress(f"uv: installed at {dst}")
    return str(dst)


def ensure_uv(progress: ProgressCallback | None = None) -> str:
    """Return a usable `uv` path, bootstrapping into user-data on demand."""
    global _UV_CACHED
    if _UV_CACHED:
        return _UV_CACHED
    found = find_uv()
    if found:
        _UV_CACHED = found
        return found
    _UV_CACHED = _bootstrap_uv(progress)
    return _UV_CACHED


# --------------------------------------------------------------------------- #
# Subprocess helpers                                                          #
# --------------------------------------------------------------------------- #


def _capture(cmd: list[str], step: str, progress: ProgressCallback | None) -> None:
    if progress is not None:
        progress(step)
    result = subprocess.run(cmd, check=False, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        if progress is not None and result.stderr:
            tail = "\n".join(result.stderr.strip().splitlines()[-12:])
            progress(f"{step} failed (exit {result.returncode}):\n{tail}")
        msg = f"{step} failed (exit {result.returncode})"
        raise RuntimeError(msg)


def create_venv(
    target: Path, *, progress: ProgressCallback | None = None
) -> Path:
    """Create ``<target>/venv`` using `uv venv` and return the python path."""
    uv = ensure_uv(progress)
    venv_dir = target / "venv"
    cmd = [uv, "venv", str(venv_dir)]
    _capture(cmd, f"uv venv -> {venv_dir}", progress)
    return venv_python(target)


def venv_python(target: Path) -> Path:
    if sys.platform == "win32":
        return target / "venv" / "Scripts" / "python.exe"
    return target / "venv" / "bin" / "python"


def pip_install(
    venv_py: Path,
    args: list[str],
    *,
    step: str,
    progress: ProgressCallback | None = None,
) -> None:
    """Run `uv pip install <args>` against the given venv interpreter."""
    uv = ensure_uv(progress)
    cmd = [uv, "pip", "install", "--python", str(venv_py), *args]
    _capture(cmd, step, progress)


__all__ = [
    "ProgressCallback",
    "create_venv",
    "ensure_uv",
    "find_uv",
    "pip_install",
    "venv_python",
]
