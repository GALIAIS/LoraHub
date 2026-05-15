"""Portable Python runtime endpoints.

GET  /api/runtime/python                — current portable runtime status
POST /api/runtime/python/install        — fetch a runtime via uv
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from lorahub.core.toolchain import python_runtime

router = APIRouter(prefix="/api")


class InstallRuntimeRequest(BaseModel):
    version: str | None = None  # defaults to python_runtime.DEFAULT_VERSION


@router.get("/runtime/python")
def get_runtime_status() -> dict[str, Any]:
    return python_runtime.status()


@router.post("/runtime/python/install")
def install_runtime(req: InstallRuntimeRequest) -> dict[str, Any]:
    version = (req.version or python_runtime.DEFAULT_VERSION).strip()
    try:
        info = python_runtime.install_runtime(version)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"installed": info, "status": python_runtime.status()}
