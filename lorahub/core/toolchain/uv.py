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

import os
import shutil
import subprocess
import sys
from collections import deque
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlsplit

from lorahub.core.paths import lorahub_dir
from lorahub.core.redaction import redact_command_text

ProgressCallback = Callable[[str], None]


# Where we drop the bootstrapped ``uv`` binary when the system doesn't
# have one. Lives under the project's ``.lorahub/`` directory so a
# ``scripts/install.{sh,bat}`` invocation and the API end up reading
# from / writing to the same place.
def _bin_dir() -> Path:
    return lorahub_dir() / "uv"


def _local_uv_path() -> Path:
    name = "uv.exe" if sys.platform == "win32" else "uv"
    return _bin_dir() / name


_UV_CACHED: str | None = None


def _subprocess_no_window() -> int:
    if sys.platform == "win32":
        return getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return 0


def find_uv() -> str | None:
    """Return the path to a discoverable `uv` binary or None if not present.

    Priority: project-local (``.lorahub/uv``) > system PATH. The local
    copy wins so a user who runs ``scripts/install.{sh,bat}`` always
    keeps using *that* uv, even if a different version exists on PATH.
    Mirroring this priority on the API side prevents the "two uv's
    disagree about installed Pythons" class of bugs.
    """
    local = _local_uv_path()
    if local.is_file():
        return str(local)
    onpath = shutil.which("uv")
    if onpath:
        return onpath
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
    # Use configured PyPI mirror if available (helps in China where pypi.org is slow)
    try:
        from lorahub.api import app as _app  # noqa: PLC0415

        pypi_index = (_app._settings_store.load().pypi_index_url or "").strip()
        if pypi_index:
            index_args = ["--index-url", pypi_index]
            hostname = urlsplit(pypi_index).hostname
            if hostname:
                index_args += ["--trusted-host", hostname]
            cmd[5:5] = index_args
    except Exception:  # noqa: BLE001
        pass
    result = subprocess.run(
        cmd,
        check=False,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=_subprocess_no_window(),
    )
    if result.returncode != 0:
        tail_text = redact_command_text((result.stderr or "").strip())
        tail = "\n".join(tail_text.splitlines()[-10:])
        # Linux distros following PEP 668 (Debian 12+, Ubuntu 23.04+, recent
        # Fedora) refuse to `pip install` against the system interpreter even
        # when --target is used. Translate the kernel of that error into
        # something actionable instead of a generic exit-1 dump.
        if "externally-managed-environment" in tail_text:
            msg = (
                "pip refused to install uv because this Python is marked as "
                "externally-managed (PEP 668). Install uv on your system "
                "first — the official one-liners are:\n"
                "  Linux/macOS: curl -LsSf https://astral.sh/uv/install.sh | sh\n"
                "  Windows:     irm https://astral.sh/uv/install.ps1 | iex\n"
                "Then restart lorahub. Original pip error:\n" + tail
            )
            raise RuntimeError(msg)
        msg = f"failed to install uv via pip (exit {result.returncode}):\n{tail}"
        raise RuntimeError(msg)

    # The uv wheel ships the binary in a few different locations depending on
    # platform / wheel format / pip version. Probe the common ones first,
    # then fall back to a recursive scan so we don't lock ourselves out of a
    # working install just because the layout shifted between releases.
    script_name = "uv.exe" if sys.platform == "win32" else "uv"
    candidates: list[Path] = [
        # Layout when the script is installed via setuptools/console_scripts
        target / ("Scripts" if sys.platform == "win32" else "bin") / script_name,
        # Layout when the wheel ships the binary in {pkg}/data/{Scripts|bin}
        target / "uv" / script_name,
        target / "uv" / ("Scripts" if sys.platform == "win32" else "bin") / script_name,
        # Some wheel formats drop binaries straight into the target root
        target / script_name,
    ]
    src = next((p for p in candidates if p.is_file()), None)
    if src is None:
        # Recursive fallback — should always find it but is the slowest path.
        for p in target.rglob(script_name):
            if p.is_file():
                src = p
                break
    if src is None:
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


def _capture(
    cmd: list[str],
    step: str,
    progress: ProgressCallback | None,
    *,
    env: dict[str, str] | None = None,
) -> None:
    if progress is not None:
        progress(step)
    full_env = None if env is None else {**os.environ, **env}
    proc = subprocess.Popen(  # noqa: S603
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=full_env,
        creationflags=_subprocess_no_window(),
    )
    tail: deque[str] = deque(maxlen=12)
    assert proc.stdout is not None  # noqa: S101
    for raw in proc.stdout:
        line = redact_command_text(raw.rstrip())
        if not line:
            continue
        tail.append(line)
        if progress is not None:
            progress(line)
    rc = proc.wait()
    if rc != 0:
        if progress is not None and tail:
            progress(f"{step} failed (exit {rc}):\n" + "\n".join(tail))
        msg = f"{step} failed (exit {rc})"
        raise RuntimeError(msg)


def create_venv(
    target: Path,
    *,
    python: Path | str | None = None,
    progress: ProgressCallback | None = None,
) -> Path:
    """Create ``<target>/venv`` using `uv venv` and return the python path.

    When ``python`` is provided it's passed to ``uv venv --python``, so the
    venv is built on top of a portable runtime managed by
    ``lorahub.core.toolchain.python_runtime`` instead of whatever
    interpreter happens to be running the API.
    """
    uv = ensure_uv(progress)
    venv_dir = target / "venv"
    cmd: list[str] = [uv, "venv"]
    if python is not None:
        cmd += ["--python", str(python)]
    cmd.append(str(venv_dir))
    label = f"uv venv -> {venv_dir}"
    if python is not None:
        label += f" (python={python})"
    _capture(cmd, label, progress)
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
    pypi_index: str | None = None,
) -> None:
    """Run `uv pip install <args>` against the given venv interpreter.

    `pypi_index` overrides the default PyPI index — typically a Chinese
    mirror like https://pypi.tuna.tsinghua.edu.cn/simple. It only takes
    effect when the caller's `args` don't already specify `--index-url`
    or `-i`, because pinned wheel-store URLs (e.g. download.pytorch.org)
    must not be silently rewritten.
    """
    uv = ensure_uv(progress)
    cmd = [uv, "pip", "install", "--python", str(venv_py)]
    if pypi_index and not _has_index_override(args):
        cmd += ["--index-url", pypi_index]
    cmd += args
    _capture(cmd, step, progress)


def _has_index_override(args: list[str]) -> bool:
    flag_indices = {"--index-url", "-i"}
    return any(a in flag_indices or a.startswith("--index-url=") for a in args)


def run_uv(
    args: list[str],
    *,
    step: str,
    progress: ProgressCallback | None = None,
    pypi_index: str | None = None,
    env: dict[str, str] | None = None,
) -> None:
    """Run an arbitrary ``uv <args>`` invocation through the bootstrap binary.

    Public adapter around the private ``_capture`` helper for callers
    that need ``uv sync`` / ``uv lock`` etc. and don't fit the
    pip_install / create_venv shapes. Surfaces failures as RuntimeError
    so callers can wrap into their own backend-specific error type.

    ``pypi_index`` injects ``--default-index <url>`` into the argv unless
    the caller already passed an index override. This lets ``uv sync``
    pick up the user's PyPI mirror (e.g. tsinghua) the same way
    ``pip_install`` does. Pinned source maps inside the target's
    ``[tool.uv.sources]`` (such as anima_lora's pytorch-cu124 index)
    still take precedence — uv only consults ``--default-index`` for
    packages without an explicit source.
    """
    uv = ensure_uv(progress)
    final_args = list(args)
    if pypi_index and not _has_index_override(final_args):
        # uv sync wants ``--default-index`` to come *after* the
        # subcommand but before any --extra / package args. Inserting
        # right after the subcommand keeps it simple.
        if final_args and final_args[0] in {"sync", "lock", "add", "pip"}:
            final_args = [final_args[0], "--default-index", pypi_index, *final_args[1:]]
        else:
            final_args = [*final_args, "--default-index", pypi_index]
    _capture([uv, *final_args], step, progress, env=env)


__all__ = [
    "ProgressCallback",
    "create_venv",
    "ensure_uv",
    "find_uv",
    "pip_install",
    "run_uv",
    "venv_python",
]
