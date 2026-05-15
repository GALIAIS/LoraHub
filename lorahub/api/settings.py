"""User-level Settings store.

Persists workbench-wide defaults (kohya checkout, python, tagger device) to a
single JSON file under the user's data directory. Env vars (LORAHUB_KOHYA_*)
still take precedence at recipe-launch time — Settings are the *fallback*
defaults the UI lets users override without touching their shell config.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from platformdirs import user_data_path


@dataclass
class Settings:
    """User-configurable defaults applied when a recipe doesn't specify them."""

    sd_scripts_path: str | None = None
    python_executable: str | None = None
    tagger_device: str = "auto"  # "auto" | "cpu" | "cuda"
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


def probe_backend(settings: Settings) -> dict[str, Any]:
    """Inspect whether the configured kohya checkout looks usable.

    Pure read-only — never raises. Returns a dict the UI can render directly:
        { sd_scripts_path, sd_scripts_ok, python, python_ok, venv_detected,
          missing_scripts, source }
    """
    from lorahub.core.backends.kohya.bootstrap import (
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
    missing = [
        s for s in _REQUIRED_SCRIPTS if not (sd_path / s).is_file()
    ] if sd_ok else list(_REQUIRED_SCRIPTS)

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
        "sd_scripts_path": str(sd_path),
        "sd_scripts_ok": sd_ok and not missing,
        "missing_scripts": missing,
        "python": str(py_path) if py_path else None,
        "python_ok": py_ok,
        "venv_detected": _venv_python(sd_path) is not None if sd_ok else False,
        "source": source,
    }


__all__ = ["Settings", "SettingsStore", "default_settings_path", "probe_backend"]
