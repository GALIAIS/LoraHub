"""Backend bootstrap (one-click kohya install).

The runner factory and active session are kept on `lorahub.api.app` so tests
can monkeypatch them: `app._build_bootstrap_runner` (callable) and
`app._bootstrap_session` (the live singleton). Both are dereferenced at
request time rather than imported at module import.
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from lorahub.api import app as app_module
from lorahub.api.bootstrap_session import (
    BootstrapRequest,
    _bootstrap_lock,
    _BootstrapSession,
)
from lorahub.api.helpers import ulid_new

router = APIRouter(prefix="/api")


@router.get("/backend/bootstrap/status")
def bootstrap_status() -> dict[str, Any]:
    sess = app_module._bootstrap_session
    if sess is not None:
        return sess.to_status_payload()
    # Fallback: surface the most recent persisted bootstrap session so a
    # restart doesn't make a finished install vanish from the UI. This
    # only fires when no live session is in progress.
    persisted = _latest_persisted_bootstrap()
    if persisted is not None:
        return persisted
    return {"status": "idle", "session_id": None, "events": []}


@router.get("/backend/bootstrap/sessions")
def list_bootstrap_sessions(limit: int = 20) -> dict[str, Any]:
    """Recent bootstrap sessions persisted in `sessions.sqlite`."""
    try:
        store = getattr(app_module, "_session_store", None)
        if store is None:
            return {"sessions": []}
        rows = store.list_recent("bootstrap", limit=limit)
    except Exception:  # noqa: BLE001
        return {"sessions": []}
    sessions = [r["snapshot"] for r in rows if isinstance(r.get("snapshot"), dict)]
    return {"sessions": sessions}


def _latest_persisted_bootstrap() -> dict[str, Any] | None:
    try:
        store = getattr(app_module, "_session_store", None)
        if store is None:
            return None
        rows = store.list_recent("bootstrap", limit=1)
    except Exception:  # noqa: BLE001
        return None
    if not rows:
        return None
    snap = rows[0].get("snapshot")
    return snap if isinstance(snap, dict) else None


@router.post("/backend/bootstrap", status_code=202)
async def start_bootstrap(req: BootstrapRequest) -> dict[str, Any]:
    with _bootstrap_lock:
        existing = app_module._bootstrap_session
        if existing is not None and existing.is_running():
            raise HTTPException(
                status_code=409, detail="a bootstrap session is already running"
            )
        # Resolve the runner first — this validates the target dir before we
        # spin a thread. HTTPException raised here surfaces as a 4xx directly.
        runner = app_module._build_bootstrap_runner(req)
        sess = _BootstrapSession(session_id=str(ulid_new()), backend=req.backend)
        app_module._bootstrap_session = sess

    loop = asyncio.get_running_loop()
    sess.start(runner, loop)
    return {
        "session_id": sess.session_id,
        "status": sess.status,
        "backend": sess.backend,
    }


class InstallDepsRequest(BaseModel):
    backend: Literal["kohya", "diffusion-pipe"] = "diffusion-pipe"


@router.post("/backend/install-deps", status_code=202)
async def install_deps(req: InstallDepsRequest) -> dict[str, Any]:
    """Run only the requirements install step for an existing backend venv."""
    with _bootstrap_lock:
        existing = app_module._bootstrap_session
        if existing is not None and existing.is_running():
            raise HTTPException(
                status_code=409, detail="a bootstrap session is already running"
            )
        runner = _build_deps_runner(req)
        sess = _BootstrapSession(session_id=str(ulid_new()), backend=req.backend)
        app_module._bootstrap_session = sess

    loop = asyncio.get_running_loop()
    sess.start(runner, loop)
    return {
        "session_id": sess.session_id,
        "status": sess.status,
        "backend": sess.backend,
    }


def _build_deps_runner(
    req: InstallDepsRequest,
) -> Any:
    """Build a runner that only installs requirements.txt into the existing venv."""
    from collections.abc import Callable  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    from lorahub.api import app as _app  # noqa: PLC0415

    settings = _app._settings_store.load()

    if req.backend == "diffusion-pipe":
        from lorahub.core.backends.diffusion_pipe import installer  # noqa: PLC0415
        from lorahub.core.backends.diffusion_pipe.bootstrap import (  # noqa: PLC0415
            _venv_python,
            default_repo_path,
        )
        import os  # noqa: PLC0415

        repo_raw = (
            os.environ.get("LORAHUB_DIFFUSION_PIPE_REPO")
            or settings.diffusion_pipe_repo_path
            or str(default_repo_path())
        )
        repo_path = Path(repo_raw).expanduser()
        venv_py = _venv_python(repo_path)
        if not venv_py or not venv_py.is_file():
            raise HTTPException(
                status_code=400,
                detail=f"No venv found at {repo_path}. Run a full install first.",
            )
        plan = installer.BootstrapPlan(
            target=repo_path,
            cuda_version="cu124",
            torch_version="2.6.0",
            torchvision_version="0.21.0",
            install_deepspeed=False,
            github_proxy=settings.github_proxy,
            base_python=venv_py,
            pypi_index=settings.pypi_index_url,
        )

        def runner(progress: Callable[[str], None]) -> None:
            installer.install_requirements(plan, progress=progress)

    else:
        from lorahub.core.backends.kohya import installer  # noqa: PLC0415
        from lorahub.core.backends.kohya.bootstrap import (  # noqa: PLC0415
            _venv_python,
            default_sd_scripts_path,
        )
        import os  # noqa: PLC0415

        sd_raw = (
            os.environ.get("LORAHUB_SD_SCRIPTS")
            or settings.sd_scripts_path
            or str(default_sd_scripts_path())
        )
        sd_path = Path(sd_raw).expanduser()
        venv_py = _venv_python(sd_path)
        if not venv_py or not venv_py.is_file():
            raise HTTPException(
                status_code=400,
                detail=f"No venv found at {sd_path}. Run a full install first.",
            )
        plan = installer.BootstrapPlan(
            target=sd_path,
            cuda_version="cu124",
            torch_version="2.6.0",
            torchvision_version="0.21.0",
            install_xformers=False,
            github_proxy=settings.github_proxy,
            base_python=venv_py,
            pypi_index=settings.pypi_index_url,
        )

        def runner(progress: Callable[[str], None]) -> None:
            installer.install_requirements(plan, progress=progress)

    return runner
