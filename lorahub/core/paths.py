"""Locate the LoraHub project root from anywhere in the source tree.

Mirrors ``lorahub.api.paths.project_root`` but lives under ``core`` so
toolchain helpers (uv, python_runtime) can resolve the repo root
without taking a dependency on the API layer. Both modules end up at
the same path — the API one wins when both are reachable because the
lifespan hook also chdir's the process there.

Resolution order:
  1. ``LORAHUB_HOME`` env var, when set and a directory.
  2. Walk up from this file looking for ``pyproject.toml`` whose
     project name is ``lorahub``. Works for editable installs and
     source checkouts; returns None for wheel installs under
     site-packages.
  3. ``Path.cwd()`` as a last resort.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


def _detect_from_source_tree() -> Path | None:
    here = Path(__file__).resolve()
    for ancestor in here.parents:
        if ancestor.name in {"site-packages", "dist-packages"}:
            return None
        py = ancestor / "pyproject.toml"
        if not py.is_file():
            continue
        try:
            text = py.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if 'name = "lorahub"' in text or "name = 'lorahub'" in text:
            return ancestor
    return None


@lru_cache(maxsize=1)
def project_root() -> Path:
    env = os.environ.get("LORAHUB_HOME", "").strip()
    if env:
        candidate = Path(env).expanduser().resolve()
        if candidate.is_dir():
            return candidate
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate
    detected = _detect_from_source_tree()
    if detected is not None:
        return detected.resolve()
    return Path.cwd().resolve()


def lorahub_dir() -> Path:
    """``<project_root>/.lorahub`` — single home for managed toolchains.

    Replaces the legacy split between ``platformdirs.user_data_path``
    (where API installed its uv + python) and ``<repo>/.tools``
    (where install.sh installed its uv + python). Centralising under
    one repo-relative folder means install.sh and the API see the
    same binaries no matter which path the user took to set up.
    """
    d = project_root() / ".lorahub"
    d.mkdir(parents=True, exist_ok=True)
    return d


__all__ = ["lorahub_dir", "project_root"]
