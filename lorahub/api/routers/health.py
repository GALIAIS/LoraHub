"""Health probe."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from lorahub.api import app as app_module
from lorahub.api.settings import probe_all_backends, probe_kohya_backend
from lorahub.api.system_update import _current_version

router = APIRouter(prefix="/api")


class HealthResponse(BaseModel):
    status: str
    version: str
    backend: dict[str, Any]
    backends: dict[str, dict[str, Any]]


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = app_module._settings_store.load()
    return HealthResponse(
        status="ok",
        # Resolve through the same chain the Settings → 维护 tab uses,
        # which prefers a live ``git describe`` over the static
        # ``_version.py`` snapshot. ``pip install -e .`` only writes
        # _version.py once; subsequent commits leave it stale, so
        # reading ``lorahub.__version__`` directly here would have the
        # /api/health version lag the frontend bundle's git-describe
        # version on every editable-install dev tree.
        version=_current_version(),
        backend=probe_kohya_backend(settings),
        backends=probe_all_backends(settings),
    )
