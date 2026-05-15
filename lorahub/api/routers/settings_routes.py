"""Settings GET/PUT.

The route file is named `settings_routes` to avoid colliding with the
sibling `lorahub.api.settings` module that holds the `Settings` dataclass and
`SettingsStore`.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from lorahub.api import app as app_module
from lorahub.api.settings import Settings, probe_backend

router = APIRouter(prefix="/api")


class SettingsResponse(BaseModel):
    settings: dict[str, Any]
    backend: dict[str, Any]
    path: str


class UpdateSettingsRequest(BaseModel):
    sd_scripts_path: str | None = None
    python_executable: str | None = None
    tagger_device: str | None = None


@router.get("/settings", response_model=SettingsResponse)
def get_settings() -> SettingsResponse:
    store = app_module._settings_store
    s = store.load()
    return SettingsResponse(
        settings=s.to_dict(),
        backend=probe_backend(s),
        path=str(store.path),
    )


@router.put("/settings", response_model=SettingsResponse)
def update_settings(req: UpdateSettingsRequest) -> SettingsResponse:
    store = app_module._settings_store
    current = store.load()

    # Treat empty strings as "clear this field".
    def _norm(v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip()
        return v or None

    new = Settings(
        sd_scripts_path=_norm(req.sd_scripts_path),
        python_executable=_norm(req.python_executable),
        tagger_device=(req.tagger_device or current.tagger_device or "auto").strip() or "auto",
        extra=current.extra,
    )
    if new.tagger_device not in {"auto", "cpu", "cuda"}:
        raise HTTPException(
            status_code=422,
            detail=f"tagger_device must be auto/cpu/cuda, got {new.tagger_device!r}",
        )
    store.save(new)
    return SettingsResponse(
        settings=new.to_dict(),
        backend=probe_backend(new),
        path=str(store.path),
    )
