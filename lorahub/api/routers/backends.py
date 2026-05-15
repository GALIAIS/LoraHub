"""Backend catalog: lists training backends and their install state.

Read-only -- the bootstrap router (`POST /api/backend/bootstrap`) is what
actually installs one. The catalog gives the UI the metadata it needs to
render the "Backends" panel without each frontend route having to hand-roll
its own probe.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from lorahub.api import app as app_module
from lorahub.api.settings import probe_all_backends
from lorahub.core.backends.registry import list_backends

router = APIRouter(prefix="/api")


class BackendEntry(BaseModel):
    id: str
    name: str
    description: str
    repo_url: str
    default_path: str
    ready: bool
    status: dict[str, Any]


class BackendsResponse(BaseModel):
    backends: list[BackendEntry]
    default: str


@router.get("/backends", response_model=BackendsResponse)
def list_backend_catalog() -> BackendsResponse:
    settings = app_module._settings_store.load()
    probes = probe_all_backends(settings)
    entries: list[BackendEntry] = []
    for desc in list_backends():
        status = probes.get(desc.id, {})
        entries.append(
            BackendEntry(
                id=desc.id,
                name=desc.name,
                description=desc.description,
                repo_url=desc.repo_url,
                default_path=str(desc.default_path),
                ready=bool(status.get("ready", False)),
                status=status,
            )
        )
    return BackendsResponse(backends=entries, default=settings.default_backend)
