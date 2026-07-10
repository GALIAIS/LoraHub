"""Health probe."""

from __future__ import annotations

import hmac
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel

from lorahub.api import app as app_module
from lorahub.api.settings import Settings
from lorahub.api.system_update import _current_version
from lorahub.core.backends.ai_toolkit.bootstrap import default_repo_path as _ait_repo
from lorahub.core.backends.anima_lora.bootstrap import default_repo_path as _anima_repo
from lorahub.core.backends.diffusion_pipe.bootstrap import default_repo_path as _dp_repo
from lorahub.core.backends.kohya.bootstrap import default_sd_scripts_path as _kohya_repo

router = APIRouter(prefix="/api")


class HealthResponse(BaseModel):
    status: str
    version: str
    service_token_match: bool = False
    backend: dict[str, Any]
    backends: dict[str, dict[str, Any]]


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    settings = app_module._settings_store.load()
    backends = _light_backends(settings)
    expected = os.environ.get("LORAHUB_SERVICE_TOKEN", "")
    provided = request.headers.get("x-lorahub-service-token", "")
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
        service_token_match=bool(
            expected and provided and hmac.compare_digest(expected, provided)
        ),
        backend=backends["kohya"],
        backends=backends,
    )


def _path_status(path: Path, required: tuple[str, ...]) -> dict[str, Any]:
    exists = path.is_dir()
    missing = [name for name in required if not (path / name).is_file()] if exists else list(required)
    return {
        "path": str(path),
        "exists": exists,
        "missing": missing,
        "ready": exists and not missing,
    }


def _light_backends(settings: Settings) -> dict[str, dict[str, Any]]:
    """Cheap backend status for /api/health.

    ``service start`` waits on this endpoint, so it must not spawn backend
    venv interpreters, scan model folders, or run package checks. The full
    install state stays on ``/api/backends``.
    """
    kohya_path = Path(settings.sd_scripts_path).expanduser() if settings.sd_scripts_path else _kohya_repo()
    dp_path = Path(settings.diffusion_pipe_repo_path).expanduser() if settings.diffusion_pipe_repo_path else _dp_repo()
    anima_path = Path(settings.anima_lora_repo_path).expanduser() if settings.anima_lora_repo_path else _anima_repo()
    ai_toolkit_path = (
        Path(settings.ai_toolkit_repo_path).expanduser()
        if settings.ai_toolkit_repo_path
        else _ait_repo()
    )
    kohya = _path_status(kohya_path, ("train_network.py",))
    return {
        "kohya": {
            "id": "kohya",
            "sd_scripts_path": kohya["path"],
            "sd_scripts_ok": kohya["ready"],
            "missing_scripts": kohya["missing"],
            "ready": kohya["ready"],
        },
        "diffusion-pipe": {
            "id": "diffusion-pipe",
            **_path_status(dp_path, ("train.py", "requirements.txt")),
        },
        "anima_lora": {
            "id": "anima_lora",
            **_path_status(anima_path, ("train.py", "inference.py")),
        },
        "ai_toolkit": {
            "id": "ai_toolkit",
            **_path_status(ai_toolkit_path, ("run.py", "toolkit/job.py")),
        },
    }
