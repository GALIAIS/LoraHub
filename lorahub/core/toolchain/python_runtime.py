"""Portable CPython management.

We use uv's `python install/list` machinery (which downloads
python-build-standalone from astral-sh's official mirror) to keep a portable
CPython under `<project_root>/.lorahub/python/`. Every backend's bootstrap
plan can opt to use this runtime as the venv base, so the user never needs
a system Python installed at all to train models.

uv supports the three platforms we ship to (Windows / Linux / macOS) and
auto-selects the right build for the current arch + libc.

Public surface:

* ``installed_runtimes()`` — list runtimes uv has cached locally.
* ``install_runtime(version, progress)`` — fetch one if missing.
* ``runtime_python(version)`` — return the python executable path uv picked.
* ``default_version()`` — recommended version (3.11 today; matches the
  range our bootstrap plans test against).
* ``status()`` — combined payload for the API/UI.
"""

from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path
from typing import Any

from lorahub.core.paths import lorahub_dir
from lorahub.core.toolchain import uv as _uv

# Anchor for runtimes uv installs into. Lives under the project's
# ``.lorahub/`` directory so installs done via ``scripts/install.{sh,bat}``
# (which writes to the same location) are visible to the API and vice
# versa. The legacy split between ``platformdirs.user_data_path`` and
# ``<repo>/.tools`` was the single biggest source of "I just installed it
# but the UI says it's missing" reports.
PYTHON_ROOT = lorahub_dir() / "python"

# We pin 3.11 by default because that's what `requires-python` in pyproject
# (and most model-training tooling) demands. The user can ask for 3.12 or
# 3.13 in the UI; everything older we hide. 3.13 is required by anima_lora
# (its own pyproject pins ``==3.13.*``); the install button on the
# Dependencies tab will pre-fetch it so anima_lora's ``uv sync`` doesn't
# stall on a separate runtime download mid-install.
DEFAULT_VERSION = "3.11"
RECOMMENDED_VERSIONS: tuple[str, ...] = ("3.11", "3.12", "3.13")


_PB_PROGRESS = _uv.ProgressCallback


def default_version() -> str:
    return DEFAULT_VERSION


def detected_platform() -> dict[str, str]:
    """Best-effort label for the UI (uv's own probe wins where available)."""
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "release": platform.release(),
    }


def _uv_python_command() -> list[str]:
    uv_bin = _uv.ensure_uv()
    return [uv_bin, "python"]


def _uv_python_env() -> dict[str, str]:
    """Env vars that point uv at our project-local install dir.

    Setting both ``UV_PYTHON_INSTALL_DIR`` and ``UV_PYTHON_BIN_DIR`` makes
    every ``uv python ...`` subprocess we spawn read/write under
    ``PYTHON_ROOT`` regardless of whether the caller passes
    ``--install-dir`` explicitly. ``--install-dir`` is forwarded too as
    a belt-and-suspenders for older uv builds that ignore the env var.
    """
    import os as _os  # noqa: PLC0415

    env = dict(_os.environ)
    env["UV_PYTHON_INSTALL_DIR"] = str(PYTHON_ROOT)
    return env


def installed_runtimes() -> list[dict[str, Any]]:
    """Return uv's view of locally installed CPython builds.

    Entries look like ``{"version": "3.11.10", "path": "...", "implementation":
    "cpython", "arch": "x86_64", ...}``. uv emits a JSON line per runtime
    when ``--output-format json`` is set; older uv versions returned plain
    text so we tolerate both. Always scopes the listing to ``PYTHON_ROOT``
    so we never surface a runtime the user installed via a different uv
    invocation outside the project.
    """
    PYTHON_ROOT.mkdir(parents=True, exist_ok=True)
    cmd = [
        *_uv_python_command(),
        "list",
        "--only-installed",
        "--output-format", "json",
        "--install-dir", str(PYTHON_ROOT),
    ]
    result = subprocess.run(
        cmd, check=False, capture_output=True, text=True, env=_uv_python_env(),
    )
    if result.returncode != 0:
        # Older uv builds reject ``--install-dir`` on list. Retry without.
        cmd_legacy = [
            *_uv_python_command(),
            "list", "--only-installed", "--output-format", "json",
        ]
        result = subprocess.run(
            cmd_legacy, check=False, capture_output=True, text=True,
            env=_uv_python_env(),
        )
        if result.returncode != 0:
            return _scan_install_dir()
    parsed = _parse_uv_python_listing(result.stdout)
    if parsed:
        return parsed
    # uv saw nothing — fall back to a manual scan of PYTHON_ROOT so a
    # runtime installed by ``scripts/install.{sh,bat}`` still shows up
    # even if the uv on PATH disagrees about the layout.
    return _scan_install_dir()


def available_runtimes() -> list[dict[str, Any]]:
    """Return runtimes uv knows it could install on this host."""
    cmd = [*_uv_python_command(), "list", "--output-format", "json"]
    result = subprocess.run(
        cmd, check=False, capture_output=True, text=True, env=_uv_python_env(),
    )
    if result.returncode != 0:
        return []
    return _parse_uv_python_listing(result.stdout)


def _scan_install_dir() -> list[dict[str, Any]]:
    """Filesystem fallback for ``installed_runtimes`` when uv is silent.

    Walks ``PYTHON_ROOT`` looking for the standard
    ``cpython-<version>-<os>-<arch>-...`` layout that uv (and our
    ``scripts/install.{sh,bat}`` helpers) produce. Returns an empty
    list if the directory doesn't exist yet.
    """
    if not PYTHON_ROOT.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for entry in sorted(PYTHON_ROOT.iterdir()):
        if not entry.is_dir():
            continue
        name = entry.name
        if not name.startswith("cpython-"):
            continue
        # Resolve symlinks so the patched real directory wins over uv's
        # ``cpython-3.12`` minor-version alias (we'd otherwise list both).
        real = entry.resolve()
        py = real / "bin" / "python"
        if not py.is_file():
            py = real / ("python.exe" if py.parent.parent.name.endswith("none") else "python")
        if not py.is_file():
            # Windows: the python.exe lives at the runtime root.
            py = real / "python.exe"
        if not py.is_file():
            continue
        # Pull the version out of the directory name. e.g.
        # ``cpython-3.12.7-linux-x86_64-gnu`` -> ``3.12.7``.
        rest = name.removeprefix("cpython-")
        version = rest.split("-", 1)[0]
        out.append({
            "version": version,
            "implementation": "cpython",
            "arch": "",
            "os": "",
            "path": str(py),
            "key": name,
            "installed": True,
        })
    return out


def _parse_uv_python_listing(stdout: str) -> list[dict[str, Any]]:
    text = stdout.strip()
    if not text:
        return []
    # uv >= 0.4 emits a JSON array.
    if text.startswith("["):
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return [_normalise_entry(d) for d in data if isinstance(d, dict)]
        except json.JSONDecodeError:
            return []
    # Older uv versions: one JSON object per line.
    out: list[dict[str, Any]] = []
    for line in text.splitlines():
        try:
            d = json.loads(line)
            if isinstance(d, dict):
                out.append(_normalise_entry(d))
        except json.JSONDecodeError:
            continue
    return out


def _normalise_entry(raw: dict[str, Any]) -> dict[str, Any]:
    # uv's schema has shifted across releases; pick the fields that matter
    # to us and gracefully fall back when a key isn't present.
    return {
        "version": str(raw.get("version") or raw.get("python_version") or ""),
        "implementation": str(raw.get("implementation") or "cpython"),
        "arch": str(raw.get("arch") or raw.get("architecture") or ""),
        "os": str(raw.get("os") or raw.get("platform") or ""),
        "path": str(raw.get("path") or raw.get("executable") or ""),
        "key": str(raw.get("key") or raw.get("identifier") or ""),
        "installed": bool(raw.get("path") or raw.get("installed", False)),
    }


def install_runtime(
    version: str = DEFAULT_VERSION, *, progress: _PB_PROGRESS | None = None
) -> dict[str, Any]:
    """Ask uv to download a portable CPython for ``version``.

    uv stores its python builds under its own data dir; we point that at
    PYTHON_ROOT via ``--install-dir`` so everything LoraHub-specific lives in
    one place and `lorahub uninstall` can clean it without touching the
    user's other uv runtimes.
    """
    PYTHON_ROOT.mkdir(parents=True, exist_ok=True)
    cmd = [
        *_uv_python_command(),
        "install",
        "--install-dir",
        str(PYTHON_ROOT),
        version,
    ]
    if progress is not None:
        progress(f"uv python install {version} -> {PYTHON_ROOT}")
    result = subprocess.run(
        cmd, check=False, stderr=subprocess.PIPE, text=True, env=_uv_python_env(),
    )
    if result.returncode != 0:
        tail = "\n".join((result.stderr or "").strip().splitlines()[-12:])
        msg = f"uv python install failed (exit {result.returncode}):\n{tail}"
        raise RuntimeError(msg)
    if progress is not None:
        progress(f"installed cpython-{version} into {PYTHON_ROOT}")
    return runtime_info(version) or {"version": version, "path": ""}


def runtime_info(version: str = DEFAULT_VERSION) -> dict[str, Any] | None:
    """Return the installed-runtime entry whose version starts with ``version``."""
    for entry in installed_runtimes():
        if entry.get("version", "").startswith(version):
            return entry
    return None


def runtime_python(version: str = DEFAULT_VERSION) -> Path | None:
    """Return the Python executable for ``version`` if installed."""
    entry = runtime_info(version)
    if not entry:
        return None
    raw = entry.get("path") or ""
    if not raw:
        return None
    p = Path(raw)
    return p if p.is_file() else None


def status() -> dict[str, Any]:
    """One-shot payload for the Settings UI."""
    installed = installed_runtimes()
    return {
        "default_version": DEFAULT_VERSION,
        "recommended_versions": list(RECOMMENDED_VERSIONS),
        "install_dir": str(PYTHON_ROOT),
        "platform": detected_platform(),
        "installed": installed,
        "active": runtime_info(DEFAULT_VERSION),
    }


__all__ = [
    "DEFAULT_VERSION",
    "PYTHON_ROOT",
    "RECOMMENDED_VERSIONS",
    "available_runtimes",
    "default_version",
    "detected_platform",
    "install_runtime",
    "installed_runtimes",
    "runtime_info",
    "runtime_python",
    "status",
]
