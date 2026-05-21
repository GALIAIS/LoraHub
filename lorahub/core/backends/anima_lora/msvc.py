"""Detect Visual Studio Build Tools (MSVC) on Windows.

Why this exists: anima_lora's training path passes ``--torch_compile``
to upstream's train.py, which forces PyTorch Inductor to JIT-compile
kernels through triton-windows. triton invokes ``cl.exe`` from a
Visual Studio Build Tools install; without it the run dies with a
``TypeError: unsupported operand type(s) for /: 'WindowsPath' and
'NoneType'`` from triton's MSVC discovery on the very first
``torch.compile`` call.

LoraHub installs the anima_lora venv automatically via the install
panel, but VS Build Tools cannot be packaged inside the project tree
(Microsoft installer is system-wide, registry-tracked, no portable
mode). The best we can do is detect whether MSVC is present and offer
to drive ``winget install Microsoft.VisualStudio.2022.BuildTools``
when it isn't.

This module is import-safe on every platform — it just returns
``DetectionResult(installed=False, reason="not Windows")`` on non-win.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DetectionResult:
    """What we know about the local MSVC install."""

    # True only when an MSVC ``cl.exe`` is reachable through one of the
    # paths triton-windows would actually use.
    installed: bool
    # Path to the discovered ``cl.exe`` when ``installed``; informational.
    cl_path: str | None = None
    # Resolved MSVC tools version (e.g. ``"14.40.33807"``); informational.
    msvc_version: str | None = None
    # When ``not installed``, a one-line user-facing reason.
    reason: str | None = None


def _find_vswhere() -> Path | None:
    """Locate vswhere.exe — the canonical way to enumerate VS installs.

    Microsoft ships it in a fixed location that doesn't move between VS
    versions, but a Build Tools-only install may not even include it
    (older 2017 era). Also accepts a vswhere on PATH.
    """
    fixed = Path(
        os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    ) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if fixed.is_file():
        return fixed
    on_path = shutil.which("vswhere.exe") or shutil.which("vswhere")
    return Path(on_path) if on_path else None


def _msvc_root_via_vswhere() -> Path | None:
    """Ask vswhere for the latest VS install with C++ build tools."""
    vswhere = _find_vswhere()
    if vswhere is None:
        return None
    try:
        out = subprocess.check_output(
            [
                str(vswhere),
                "-prerelease",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-latest",
                "-property",
                "installationPath",
            ],
            text=True,
            timeout=15,
        ).strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return None
    if not out:
        return None
    msvc_root = Path(out) / "VC" / "Tools" / "MSVC"
    return msvc_root if msvc_root.is_dir() else None


def _msvc_root_via_path() -> Path | None:
    """Match a ``...\\VC\\Tools\\MSVC\\<ver>\\bin\\Hostx64\\x64`` entry on PATH."""
    for piece in os.environ.get("PATH", "").split(os.pathsep):
        if not piece:
            continue
        norm = piece.replace("/", "\\").lower()
        if r"\vc\tools\msvc\\".replace("\\\\", "\\") in norm:
            # Walk up to the ``MSVC`` root. The PATH entry we look for
            # is ``<root>\<ver>\bin\Hostx64\x64``; root is parents[3].
            parts = Path(piece).parts
            for i in range(len(parts) - 1, 0, -1):
                if parts[i].lower() == "msvc":
                    return Path(*parts[: i + 1])
    return None


def _check_cl(msvc_root: Path) -> tuple[Path, str] | None:
    """Pick the newest ``<ver>`` subdir whose ``cl.exe`` actually exists.

    Returns ``(cl_path, version)`` or ``None``.
    """
    candidates: list[tuple[Path, str]] = []
    try:
        subdirs = list(msvc_root.iterdir())
    except OSError:
        return None
    for sub in subdirs:
        if not sub.is_dir():
            continue
        cl = sub / "bin" / "Hostx64" / "x64" / "cl.exe"
        if cl.is_file():
            candidates.append((cl, sub.name))
    if not candidates:
        return None
    # Pick the largest version string. Tuple sort on the version's
    # dotted ints is good enough — "14.40.33807" > "14.39.33523".
    candidates.sort(
        key=lambda c: tuple(int(p) for p in c[1].split(".") if p.isdigit()),
        reverse=True,
    )
    return candidates[0]


def detect() -> DetectionResult:
    """Return whether a usable MSVC is reachable on the current host."""
    if sys.platform != "win32":
        return DetectionResult(
            installed=False,
            reason="MSVC build tools are only relevant on Windows",
        )

    msvc_root = _msvc_root_via_vswhere() or _msvc_root_via_path()
    if msvc_root is None:
        return DetectionResult(
            installed=False,
            reason=(
                "Visual Studio Build Tools 2022 not detected. "
                "anima_lora's torch.compile path needs MSVC ``cl.exe`` "
                "via triton-windows; without it the trainer crashes "
                "during the first compile pass with a TypeError from "
                "triton's MSVC discovery."
            ),
        )

    cl = _check_cl(msvc_root)
    if cl is None:
        return DetectionResult(
            installed=False,
            reason=(
                f"Found a partial MSVC install at {msvc_root} but no "
                f"cl.exe under any version. Re-run the Build Tools "
                "installer with the 'Desktop development with C++' "
                "workload selected."
            ),
        )

    cl_path, version = cl
    return DetectionResult(
        installed=True,
        cl_path=str(cl_path),
        msvc_version=version,
    )


# ---- Install driver ------------------------------------------------------


# winget package id for the Build Tools 2022 standalone installer.
_BUILDTOOLS_PACKAGE_ID = "Microsoft.VisualStudio.2022.BuildTools"

# ``--add`` selectors fed straight to the VS installer via
# ``--override``. The first is the C++ workload (cl.exe + linker +
# vcruntime); the second is the Windows 11 SDK that triton's MSVC
# discovery checks for as a sanity gate.
_BUILDTOOLS_OVERRIDE_ARGS = (
    "--quiet",
    "--wait",
    "--norestart",
    "--add",
    "Microsoft.VisualStudio.Workload.VCTools",
    "--add",
    "Microsoft.VisualStudio.Component.Windows11SDK.22621",
    "--includeRecommended",
)


def winget_available() -> bool:
    return shutil.which("winget") is not None


def install_command() -> list[str] | None:
    """Build the ``winget install`` argv that drives the Build Tools setup.

    Returns None on non-Windows or when winget is missing.
    """
    if sys.platform != "win32" or not winget_available():
        return None
    override = " ".join(_BUILDTOOLS_OVERRIDE_ARGS)
    return [
        "winget",
        "install",
        "--id",
        _BUILDTOOLS_PACKAGE_ID,
        "--accept-package-agreements",
        "--accept-source-agreements",
        "--silent",
        "--override",
        override,
    ]


__all__ = [
    "DetectionResult",
    "detect",
    "install_command",
    "winget_available",
]
