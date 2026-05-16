"""Domain-segmented routers for the LoraHub HTTP API.

Each router defines its own `APIRouter(prefix="/api")`. `app.py` imports
`all_routers` and includes them on the FastAPI instance. WebSocket endpoints
remain on the FastAPI app proper (FastAPI's APIRouter has historical caveats
with WS routes), so they live in `app.py` and reuse helpers from this package.
"""

from __future__ import annotations

from fastapi import APIRouter

from .backends import router as backends_router
from .bootstrap import router as bootstrap_router
from .captions import router as captions_router
from .datasets import router as datasets_router
from .health import router as health_router
from .jobs import router as jobs_router
from .models import router as models_router
from .network import router as network_router
from .recipes import router as recipes_router
from .runtime import router as runtime_router
from .samples import router as samples_router
from .settings_routes import router as settings_router
from .system import router as system_router
from .tagging import router as tagging_router

all_routers: list[APIRouter] = [
    health_router,
    settings_router,
    recipes_router,
    datasets_router,
    jobs_router,
    backends_router,
    bootstrap_router,
    captions_router,
    models_router,
    network_router,
    runtime_router,
    samples_router,
    system_router,
    tagging_router,
]

__all__ = ["all_routers"]
