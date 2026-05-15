"""Health probe."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from lorahub import __version__
from lorahub.api import app as app_module
from lorahub.api.settings import probe_backend

router = APIRouter(prefix="/api")


class HealthResponse(BaseModel):
    status: str
    version: str
    backend: dict[str, Any]


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=__version__,
        backend=probe_backend(app_module._settings_store.load()),
    )
