"""Domain-segmented routers for the LoraHub HTTP API.

Each router defines its own `APIRouter(prefix="/api")`. `app.py` imports
`all_routers` and includes them on the FastAPI instance. WebSocket endpoints
remain on the FastAPI app proper (FastAPI's APIRouter has historical caveats
with WS routes), so they live in `app.py` and reuse helpers from this package.
"""

from __future__ import annotations

from fastapi import APIRouter

from .bootstrap import router as bootstrap_router
from .datasets import router as datasets_router
from .health import router as health_router
from .jobs import router as jobs_router
from .recipes import router as recipes_router
from .settings_routes import router as settings_router

all_routers: list[APIRouter] = [
    health_router,
    settings_router,
    recipes_router,
    datasets_router,
    jobs_router,
    bootstrap_router,
]

__all__ = ["all_routers"]
