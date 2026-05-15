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
from lorahub.api.settings import (
    VALID_BACKEND_IDS,
    Settings,
    probe_all_backends,
    probe_kohya_backend,
)

router = APIRouter(prefix="/api")


class SettingsResponse(BaseModel):
    settings: dict[str, Any]
    backend: dict[str, Any]
    backends: dict[str, dict[str, Any]]
    path: str


class UpdateSettingsRequest(BaseModel):
    sd_scripts_path: str | None = None
    python_executable: str | None = None
    diffusion_pipe_repo_path: str | None = None
    diffusion_pipe_python: str | None = None
    default_backend: str | None = None
    tagger_device: str | None = None
    github_proxy: str | None = None
    huggingface_endpoint: str | None = None
    modelscope_enabled: bool | None = None
    modelscope_token: str | None = None


def _norm(v: str | None) -> str | None:
    """Treat empty / whitespace-only strings as 'clear this field'."""
    if v is None:
        return None
    v = v.strip()
    return v or None


def _to_response(s: Settings, path: str) -> SettingsResponse:
    return SettingsResponse(
        settings=s.to_dict(),
        backend=probe_kohya_backend(s),
        backends=probe_all_backends(s),
        path=path,
    )


@router.get("/settings", response_model=SettingsResponse)
def get_settings() -> SettingsResponse:
    store = app_module._settings_store
    return _to_response(store.load(), str(store.path))


@router.put("/settings", response_model=SettingsResponse)
def update_settings(req: UpdateSettingsRequest) -> SettingsResponse:
    store = app_module._settings_store
    current = store.load()

    default_backend = (
        req.default_backend.strip()
        if req.default_backend is not None
        else current.default_backend
    )
    if default_backend not in VALID_BACKEND_IDS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"default_backend must be one of "
                f"{sorted(VALID_BACKEND_IDS)}, got {default_backend!r}"
            ),
        )

    tagger_device = (req.tagger_device or current.tagger_device or "auto").strip() or "auto"
    if tagger_device not in {"auto", "cpu", "cuda"}:
        raise HTTPException(
            status_code=422,
            detail=f"tagger_device must be auto/cpu/cuda, got {tagger_device!r}",
        )

    new = Settings(
        sd_scripts_path=_norm(req.sd_scripts_path),
        python_executable=_norm(req.python_executable),
        diffusion_pipe_repo_path=_norm(req.diffusion_pipe_repo_path),
        diffusion_pipe_python=_norm(req.diffusion_pipe_python),
        default_backend=default_backend,
        tagger_device=tagger_device,
        github_proxy=_norm(req.github_proxy),
        huggingface_endpoint=_norm(req.huggingface_endpoint),
        modelscope_enabled=(
            req.modelscope_enabled
            if req.modelscope_enabled is not None
            else current.modelscope_enabled
        ),
        modelscope_token=_norm(req.modelscope_token),
        extra=current.extra,
    )
    store.save(new)
    return _to_response(new, str(store.path))
