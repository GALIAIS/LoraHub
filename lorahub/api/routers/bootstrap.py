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
from lorahub.api.task_sessions import TaskSessionStore, default_task_store_path
from lorahub.api.torch_options import get_torch_options

router = APIRouter(prefix="/api")
_KIND_BACKEND_BOOTSTRAP = "backend_bootstrap"


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


@router.get("/backend/torch-options")
def torch_options() -> dict[str, Any]:
    """Selectable PyTorch/CUDA wheel combinations for this host."""

    return get_torch_options()


def _latest_persisted_bootstrap() -> dict[str, Any] | None:
    try:
        store = getattr(app_module, "_session_store", None)
        rows = store.list_recent("bootstrap", limit=1) if store is not None else []
    except Exception:  # noqa: BLE001
        rows = []
    if rows:
        snap = rows[0].get("snapshot")
        if isinstance(snap, dict):
            return snap
    try:
        task = _task_store().latest(_KIND_BACKEND_BOOTSTRAP)
    except Exception:  # noqa: BLE001
        return None
    if task is None:
        return None
    if isinstance(task.result, dict):
        return task.result
    return {
        "status": task.status,
        "session_id": task.id,
        "backend": str(task.metadata.get("backend") or ""),
        "events": [
            {
                "level": event.level,
                "message": event.message,
                "ts": event.ts,
                **event.payload,
            }
            for event in task.events
        ],
        "error": task.error,
    }


def _task_store() -> TaskSessionStore:
    store = app_module._task_session_store
    if store is None:
        store = TaskSessionStore(default_task_store_path())
        app_module._task_session_store = store
    return store


def _new_bootstrap_session(
    *,
    backend: str,
    title: str,
    metadata: dict[str, Any],
) -> _BootstrapSession:
    task = _task_store().create(
        kind=_KIND_BACKEND_BOOTSTRAP,
        title=title,
        metadata={"backend": backend, **metadata},
    )
    return _BootstrapSession(
        session_id=task.id,
        backend=backend,
        task_kind=_KIND_BACKEND_BOOTSTRAP,
    )


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
        sess = _new_bootstrap_session(
            backend=req.backend,
            title=f"{req.backend} backend bootstrap",
            metadata={
                "operation": "bootstrap",
                "target": req.target,
                "cuda": req.cuda,
                "torch_version": req.torch_version,
                "torchvision_version": req.torchvision_version,
                "install_xformers": req.install_xformers,
                "install_deepspeed": req.install_deepspeed,
                "torch_override": req.torch_override,
                "force": req.force,
            },
        )
        app_module._bootstrap_session = sess

    loop = asyncio.get_running_loop()
    sess.start(runner, loop)
    return {
        "session_id": sess.session_id,
        "status": sess.status,
        "backend": sess.backend,
    }


class InstallDepsRequest(BaseModel):
    backend: Literal["kohya", "diffusion-pipe", "anima_lora"] = "diffusion-pipe"
    install_deepspeed: bool = True


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
        sess = _new_bootstrap_session(
            backend=req.backend,
            title=f"{req.backend} dependency install",
            metadata={"operation": "install_deps"},
        )
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
            torch_index_base=settings.torch_index_url,
        )

        def runner(progress: Callable[[str], None]) -> None:
            installer.install_requirements(plan, progress=progress)

    elif req.backend == "anima_lora":
        # anima_lora doesn't have a "requirements only" install step —
        # the whole flow is ``uv sync``. Re-running it is the moral
        # equivalent of "reinstall deps", and the fast-path in uv
        # makes it a near no-op when nothing changed.
        from lorahub.core.backends.anima_lora import (  # noqa: PLC0415
            bootstrap as al_bootstrap,
            installer as al_installer,
        )
        from lorahub.api.torch_options import recommended_torch_option  # noqa: PLC0415

        target_path = al_bootstrap.default_repo_path()
        if not (target_path / "pyproject.toml").is_file():
            raise HTTPException(
                status_code=400,
                detail=(
                    f"vendored anima_lora copy at {target_path} missing — "
                    "the repo source tree may be corrupted."
                ),
            )
        torch_option = recommended_torch_option()
        plan = al_installer.BootstrapPlan(
            target=target_path,
            base_python=None,
            pypi_index=settings.pypi_index_url,
            install_deepspeed=req.install_deepspeed,
            torch_override=True,
            cuda_version=torch_option.cuda,
            torch_version=torch_option.torch_version,
            torchvision_version=torch_option.torchvision_version,
            torch_index_base=settings.torch_index_url,
        )

        def runner(progress: Callable[[str], None]) -> None:
            al_installer.sync(plan, progress=progress)
            al_installer.install_torch_override(plan, progress=progress)
            al_installer.install_bitsandbytes(plan, progress=progress)
            al_installer.install_deepspeed(plan, progress=progress)

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
            torch_index_base=settings.torch_index_url,
        )

        def runner(progress: Callable[[str], None]) -> None:
            installer.install_requirements(plan, progress=progress)

    return runner


# --------------------------------------------------------------------------- #
# FlashAttention install
# --------------------------------------------------------------------------- #
#
# FA2: stable on PyPI (`flash-attn`). The Dao-AILab wheel index resolves
# the right artefact for the active torch/cu/python/glibc combo, so we
# just shell out to `uv pip install flash-attn --no-build-isolation` and
# let it pick.
#
# FA3 (Hopper) and FA4 (Hopper/Blackwell beta) ship as in-repo source
# builds with no stable PyPI presence; the kernel team has shipped them
# under multiple distribution names (`flash-attn`, `flash-attn-3`,
# `flash-attn-4`) at various points and the right install command
# depends on CUDA toolkit version + glibc + GPU silicon. Rather than
# guess and risk a typo-squat install, we surface 501 + the upstream
# README link so the user installs the wheel manually for those.

_FLASH_ATTN_DOC_URL = "https://github.com/Dao-AILab/flash-attention#installation-and-features"


class InstallFlashAttnRequest(BaseModel):
    backend: Literal["kohya", "diffusion-pipe", "anima_lora"] = "diffusion-pipe"
    # FA2 is the only version we install automatically. FA3/FA4 are still
    # accepted so the frontend can surface a single endpoint, but they
    # return 501 with a link to the upstream install docs.
    version: Literal["2", "3", "4"] = "2"


@router.post("/backend/install-flash-attn", status_code=202)
async def install_flash_attn(req: InstallFlashAttnRequest) -> dict[str, Any]:
    """Install FlashAttention into the chosen backend's venv.

    Version 2 runs ``uv pip install flash-attn --no-build-isolation`` and
    returns 202 with a bootstrap session id (poll the same
    ``/backend/bootstrap/status`` channel to follow progress). Versions
    3 and 4 are not auto-installable — see the module-level comment for
    why — and return 501 with ``install_doc_url`` pointing at the
    upstream README.
    """
    if req.version != "2":
        raise HTTPException(
            status_code=501,
            detail={
                "message": (
                    f"Automatic FlashAttention {req.version} install is not "
                    f"implemented yet — install the wheel into the {req.backend} "
                    "venv manually."
                ),
                "backend": req.backend,
                "version": req.version,
                "install_doc_url": _FLASH_ATTN_DOC_URL,
            },
        )

    with _bootstrap_lock:
        existing = app_module._bootstrap_session
        if existing is not None and existing.is_running():
            raise HTTPException(
                status_code=409, detail="a bootstrap session is already running"
            )
        runner = _build_flash_attn2_runner(req)
        sess = _new_bootstrap_session(
            backend=req.backend,
            title=f"{req.backend} FlashAttention {req.version} install",
            metadata={"operation": "install_flash_attn", "version": req.version},
        )
        app_module._bootstrap_session = sess

    loop = asyncio.get_running_loop()
    sess.start(runner, loop)
    return {
        "session_id": sess.session_id,
        "status": sess.status,
        "backend": sess.backend,
        "version": req.version,
    }


def _build_flash_attn2_runner(req: InstallFlashAttnRequest) -> Any:
    """Build a runner that pip-installs flash-attn (FA2) into the backend venv.

    Uses the shared uv pip plumbing so the install honours the configured
    pypi_index mirror. ``--no-build-isolation`` is required because
    flash-attn's setup.py needs the venv's torch to be visible — the
    upstream README explicitly says so.
    """
    from collections.abc import Callable  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    from lorahub.api import app as _app  # noqa: PLC0415
    from lorahub.core.toolchain import uv as _uv  # noqa: PLC0415

    settings = _app._settings_store.load()

    if req.backend == "diffusion-pipe":
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
    elif req.backend == "anima_lora":
        from lorahub.core.backends.anima_lora import bootstrap as al_bootstrap  # noqa: PLC0415

        repo_path = al_bootstrap.default_repo_path()
        venv_py = al_bootstrap._venv_python(repo_path)
    else:
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
        repo_path = Path(sd_raw).expanduser()
        venv_py = _venv_python(repo_path)

    if not venv_py or not venv_py.is_file():
        raise HTTPException(
            status_code=400,
            detail=f"No venv found at {repo_path}. Run a full install first.",
        )

    pypi_index = settings.pypi_index_url

    def runner(progress: Callable[[str], None]) -> None:
        progress("flash-attn (FA2): resolving wheel via uv pip")
        _uv.pip_install(
            venv_py,
            ["flash-attn", "--no-build-isolation"],
            step="flash-attn install",
            progress=progress,
            pypi_index=pypi_index,
        )
        progress("flash-attn (FA2): install complete")

    return runner
