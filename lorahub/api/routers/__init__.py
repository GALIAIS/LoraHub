"""Domain-segmented routers for the LoraHub HTTP API.

Each router defines its own `APIRouter(prefix="/api")`. `app.py` imports
`all_routers` and includes them on the FastAPI instance. WebSocket endpoints
remain on the FastAPI app proper (FastAPI's APIRouter has historical caveats
with WS routes), so they live in `app.py` and reuse helpers from this package.
"""

from __future__ import annotations

from fastapi import APIRouter

from .ai import router as ai_router
from .artifacts import router as artifacts_router
from .backends import router as backends_router
from .bootstrap import router as bootstrap_router
from .captions import router as captions_router
from .datasets import router as datasets_router
from .error_reports import router as error_reports_router
from .health import router as health_router
from .image_studio import router as image_studio_router
from .jobs import router as jobs_router
from .models import router as models_router
from .network import router as network_router
from .configs import router as configs_router
from .runtime import router as runtime_router
from .samples import router as samples_router
from .settings_routes import router as settings_router
from .storage import router as storage_router
from .sweeps import router as sweeps_router
from .system import router as system_router
from .tagging import router as tagging_router
from .terminal import router as terminal_router
from .wandb_routes import router as wandb_router

all_routers: list[APIRouter] = [
    health_router,
    settings_router,
    configs_router,
    datasets_router,
    image_studio_router,
    jobs_router,
    artifacts_router,
    backends_router,
    bootstrap_router,
    ai_router,
    captions_router,
    error_reports_router,
    models_router,
    network_router,
    runtime_router,
    samples_router,
    storage_router,
    sweeps_router,
    system_router,
    tagging_router,
    terminal_router,
    wandb_router,
]

__all__ = ["all_routers"]
