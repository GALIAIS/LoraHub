"""User-level Settings store.

Persists workbench-wide defaults (per-backend checkout + python, default
backend choice, tagger device) to a single JSON file under the user's data
directory. Env vars (LORAHUB_*) still take precedence at recipe-launch
time -- Settings are the *fallback* defaults the UI lets users override
without touching their shell config.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from platformdirs import user_data_path

from lorahub.core.backends.registry import known_ids


@dataclass
class Settings:
    """User-configurable defaults applied when a recipe doesn't specify them."""

    # kohya backend
    sd_scripts_path: str | None = None
    python_executable: str | None = None

    # diffusion-pipe backend
    diffusion_pipe_repo_path: str | None = None
    diffusion_pipe_python: str | None = None

    # Which backend the UI defaults to when starting a fresh recipe.
    default_backend: str = "kohya"

    tagger_device: str = "auto"  # "auto" | "cpu" | "cuda"

    # --- Network acceleration ---
    # Optional GitHub mirror prefix (e.g. "https://gh-proxy.org") rewriting
    # `https://github.com/...` URLs at clone time. Leave empty for direct.
    github_proxy: str | None = None
    # HuggingFace endpoint mirror (set as HF_ENDPOINT env var on subprocess
    # launches). Leave empty for the official site.
    huggingface_endpoint: str | None = None
    # When true, downloads default to ModelScope where applicable.
    modelscope_enabled: bool = False
    # Optional access token for private ModelScope models.
    modelscope_token: str | None = None

    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Settings:
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in known}
        kwargs["extra"] = {k: v for k, v in data.items() if k not in known}
        return cls(**kwargs)


def default_settings_path() -> Path:
    """`<platformdirs user_data>/lorahub/lorahub/settings.json`."""
    return user_data_path("lorahub", "lorahub") / "settings.json"


class SettingsStore:
    """Tiny JSON-backed store. Atomic writes via a sibling temp file."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = (path or default_settings_path()).resolve()

    def load(self) -> Settings:
        if not self.path.is_file():
            return Settings()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return Settings()
        if not isinstance(data, dict):
            return Settings()
        return Settings.from_dict(data)

    def save(self, settings: Settings) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(settings.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(self.path)


# --------------------------------------------------------------------------- #
# Per-backend probes (pure read-only; never raise).
# --------------------------------------------------------------------------- #


def probe_kohya_backend(settings: Settings) -> dict[str, Any]:
    """Inspect whether the configured kohya checkout looks usable."""
    from lorahub.core.backends.kohya.bootstrap import (  # noqa: PLC0415
        _ENV_PYTHON,
        _ENV_SD_SCRIPTS,
        _REQUIRED_SCRIPTS,
        _venv_python,
        default_sd_scripts_path,
    )

    sd_raw = (
        os.environ.get(_ENV_SD_SCRIPTS)
        or settings.sd_scripts_path
        or str(default_sd_scripts_path())
    )
    sd_path = Path(sd_raw).expanduser()

    sd_ok = sd_path.is_dir()
    missing = (
        [s for s in _REQUIRED_SCRIPTS if not (sd_path / s).is_file()]
        if sd_ok
        else list(_REQUIRED_SCRIPTS)
    )

    py_raw = (
        os.environ.get(_ENV_PYTHON)
        or settings.python_executable
        or (str(_venv_python(sd_path)) if _venv_python(sd_path) else None)
    )
    py_path = Path(py_raw).expanduser() if py_raw else None
    py_ok = bool(py_path and py_path.is_file())

    if os.environ.get(_ENV_SD_SCRIPTS):
        source = "env"
    elif settings.sd_scripts_path:
        source = "settings"
    else:
        source = "default"

    return {
        "id": "kohya",
        "sd_scripts_path": str(sd_path),
        "sd_scripts_ok": sd_ok and not missing,
        "missing_scripts": missing,
        "python": str(py_path) if py_path else None,
        "python_ok": py_ok,
        "venv_detected": _venv_python(sd_path) is not None if sd_ok else False,
        "ready": (sd_ok and not missing) and py_ok,
        "source": source,
    }


def probe_diffusion_pipe_backend(settings: Settings) -> dict[str, Any]:
    """Inspect whether the configured diffusion-pipe checkout looks usable."""
    from lorahub.core.backends.diffusion_pipe.bootstrap import (  # noqa: PLC0415
        _ENV_PYTHON,
        _ENV_REPO,
        _REQUIRED_FILES,
        _venv_python,
        default_repo_path,
    )

    repo_raw = (
        os.environ.get(_ENV_REPO)
        or settings.diffusion_pipe_repo_path
        or str(default_repo_path())
    )
    repo_path = Path(repo_raw).expanduser()

    repo_ok = repo_path.is_dir()
    missing = (
        [f for f in _REQUIRED_FILES if not (repo_path / f).is_file()]
        if repo_ok
        else list(_REQUIRED_FILES)
    )

    py_raw = (
        os.environ.get(_ENV_PYTHON)
        or settings.diffusion_pipe_python
        or (str(_venv_python(repo_path)) if _venv_python(repo_path) else None)
    )
    py_path = Path(py_raw).expanduser() if py_raw else None
    py_ok = bool(py_path and py_path.is_file())

    if os.environ.get(_ENV_REPO):
        source = "env"
    elif settings.diffusion_pipe_repo_path:
        source = "settings"
    else:
        source = "default"

    return {
        "id": "diffusion-pipe",
        "repo_path": str(repo_path),
        "repo_ok": repo_ok and not missing,
        "missing_files": missing,
        "python": str(py_path) if py_path else None,
        "python_ok": py_ok,
        "venv_detected": _venv_python(repo_path) is not None if repo_ok else False,
        "ready": (repo_ok and not missing) and py_ok,
        "source": source,
    }


def probe_all_backends(settings: Settings) -> dict[str, dict[str, Any]]:
    """Return a probe payload for every backend in the registry."""
    return {
        "kohya": probe_kohya_backend(settings),
        "diffusion-pipe": probe_diffusion_pipe_backend(settings),
    }


def probe_backend(settings: Settings) -> dict[str, Any]:
    """Backwards-compatible probe -- returns the kohya status payload.

    Kept so legacy callers (notably `/api/health`) keep working without
    every site having to opt in to the multi-backend payload at once.
    """
    return probe_kohya_backend(settings)


VALID_BACKEND_IDS: frozenset[str] = frozenset(known_ids())


# --------------------------------------------------------------------------- #
# Network acceleration helpers (consumed by the installer + subprocess launch)
# --------------------------------------------------------------------------- #


def apply_github_proxy(url: str, proxy: str | None) -> str:
    """Rewrite a github.com URL through `proxy` (e.g. "https://gh-proxy.org").

    No-op when `proxy` is empty or when the URL doesn't point at github.com.
    Strips the trailing slash from `proxy` so callers can paste either form.
    """
    if not proxy:
        return url
    p = proxy.strip().rstrip("/")
    if not p:
        return url
    if not url.startswith(("https://github.com/", "http://github.com/")):
        return url
    return f"{p}/{url}"


def env_overrides(settings: Settings) -> dict[str, str]:
    """Environment variables to inject into subprocesses started by lorahub.

    Currently routes HuggingFace traffic through a mirror when configured.
    The shell environment still wins — these are defaults applied only when
    the user hasn't set the variable themselves.
    """
    overrides: dict[str, str] = {}
    hf = (settings.huggingface_endpoint or "").strip().rstrip("/")
    if hf and "HF_ENDPOINT" not in os.environ:
        overrides["HF_ENDPOINT"] = hf
        # huggingface_hub also reads HUGGINGFACE_HUB_ENDPOINT historically.
        overrides["HUGGINGFACE_HUB_ENDPOINT"] = hf
    if settings.modelscope_token and "MODELSCOPE_API_TOKEN" not in os.environ:
        overrides["MODELSCOPE_API_TOKEN"] = settings.modelscope_token
    return overrides


__all__ = [
    "Settings",
    "SettingsStore",
    "VALID_BACKEND_IDS",
    "apply_github_proxy",
    "default_settings_path",
    "env_overrides",
    "probe_all_backends",
    "probe_backend",
    "probe_diffusion_pipe_backend",
    "probe_kohya_backend",
]
