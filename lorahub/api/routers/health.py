"""Health probe."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from lorahub import __version__
from lorahub.api import app as app_module
from lorahub.api.settings import probe_all_backends, probe_kohya_backend

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
        version=__version__,
        backend=probe_kohya_backend(settings),
        backends=probe_all_backends(settings),
    )
