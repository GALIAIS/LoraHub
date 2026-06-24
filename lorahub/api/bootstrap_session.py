"""Backend bootstrap session: a singleton install thread + event fanout.

The session class lives in its own module so the bootstrap router can import
it without dragging in unrelated app state. The default
`_build_bootstrap_runner` factory is exposed here too so `app.py` can bind it
under the test-patchable name.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from fastapi import HTTPException
from pydantic import BaseModel

from lorahub.api.task_sessions import TaskEvent


class BootstrapRequest(BaseModel):
    backend: Literal["kohya", "diffusion-pipe", "anima_lora"] = "kohya"
    target: str | None = None
    cuda: str = "cu124"
    torch_version: str = "2.6.0"
    torchvision_version: str = "0.21.0"
    install_xformers: bool = True
    install_deepspeed: bool = True
    force: bool = False


class _BootstrapSession:
    """Singleton wrapper around `installer.bootstrap` running on a worker thread.

    The session buffers structured events for late-joining HTTP polls and fans
    them out to attached `asyncio.Queue` listeners for the WebSocket stream.
    Each install step turns into one event; a final `done` or `error` event
    marks the terminal state and triggers listener wake-ups so they can close.
    """

    _STATUS_RUNNING = "running"
    _STATUS_SUCCEEDED = "succeeded"
    _STATUS_FAILED = "failed"

    def __init__(
        self,
        session_id: str,
        backend: str = "kohya",
        *,
        task_kind: str | None = None,
    ) -> None:
        self.session_id = session_id
        self.backend = backend
        self.task_kind = task_kind
        self.status: str = self._STATUS_RUNNING
        self.events: list[dict[str, Any]] = []
        self._listeners: list[asyncio.Queue[dict[str, Any]]] = []
        self._lock = threading.RLock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    def attach(self, queue: asyncio.Queue[dict[str, Any]]) -> list[dict[str, Any]]:
        """Register a listener and return the buffered backlog atomically."""
        with self._lock:
            self._listeners.append(queue)
            return list(self.events)

    def detach(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        with self._lock:
            if queue in self._listeners:
                self._listeners.remove(queue)

    def is_running(self) -> bool:
        return self.status == self._STATUS_RUNNING

    def to_status_payload(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": self.status,
                "session_id": self.session_id,
                "backend": self.backend,
                "events": list(self.events),
            }

    def start(
        self,
        runner: Callable[[Callable[[str], None]], None],
        loop: asyncio.AbstractEventLoop | None,
    ) -> None:
        """Spawn the worker thread that calls `runner(progress_cb)`."""
        self._loop = loop
        self._thread = threading.Thread(
            target=self._run, args=(runner,), name="lorahub-bootstrap", daemon=True
        )
        self._thread.start()

    def _run(self, runner: Callable[[Callable[[str], None]], None]) -> None:
        self._mark_task_running()
        try:
            runner(lambda step: self._emit("info", step, message=step))
        except Exception as exc:  # noqa: BLE001 — surface any installer failure
            step = getattr(exc, "step", "bootstrap")
            self._emit("error", step, message=str(exc))
            self._finalize(self._STATUS_FAILED)
            return
        self._emit("done", "complete", message=f"{self.backend} backend installed")
        self._finalize(self._STATUS_SUCCEEDED)

    def _emit(self, level: str, step: str, *, message: str) -> None:
        event = {
            "step": step,
            "level": level,
            "message": message,
            "ts": datetime.now(UTC).timestamp(),
        }
        with self._lock:
            self.events.append(event)
            listeners = list(self._listeners)
        for queue in listeners:
            self._dispatch(queue, event)
        self._append_task_event(event)

    def _finalize(self, status: str) -> None:
        with self._lock:
            self.status = status
            listeners = list(self._listeners)
        # Wake any listener still parked on `queue.get()` so it can close.
        sentinel: dict[str, Any] = {"step": "__terminal__", "level": status}
        for queue in listeners:
            self._dispatch(queue, sentinel)
        self._finalize_task(status)
        # Persist the terminal snapshot so a server restart can still
        # show "what happened" to past installs. Best-effort — a corrupt
        # session DB must never sink the live install.
        try:
            from lorahub.api import app as _app  # noqa: PLC0415

            store = getattr(_app, "_session_store", None)
            if store is not None:
                store.upsert_bootstrap(self.to_status_payload())
        except Exception:  # noqa: BLE001
            pass

    def _dispatch(self, queue: asyncio.Queue[dict[str, Any]], event: dict[str, Any]) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        # Loop already torn down — listener is gone, drop the event.
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(queue.put_nowait, event)

    def _mark_task_running(self) -> None:
        if not self.task_kind:
            return
        try:
            from lorahub.api import app as _app  # noqa: PLC0415

            store = getattr(_app, "_task_session_store", None)
            if store is not None:
                store.update(self.session_id, status="running", percent=0)
        except Exception:  # noqa: BLE001
            pass

    def _append_task_event(self, event: dict[str, Any]) -> None:
        if not self.task_kind:
            return
        try:
            from lorahub.api import app as _app  # noqa: PLC0415

            store = getattr(_app, "_task_session_store", None)
            if store is not None:
                store.append_event(
                    self.session_id,
                    TaskEvent(
                        level=str(event.get("level") or "info"),
                        message=str(event.get("message") or event.get("step") or ""),
                        payload=dict(event),
                        ts=float(event.get("ts") or datetime.now(UTC).timestamp()),
                    ),
                )
        except Exception:  # noqa: BLE001
            pass

    def _finalize_task(self, status: str) -> None:
        if not self.task_kind:
            return
        try:
            from lorahub.api import app as _app  # noqa: PLC0415

            store = getattr(_app, "_task_session_store", None)
            if store is not None:
                if status == self._STATUS_SUCCEEDED:
                    store.update(
                        self.session_id,
                        status="succeeded",
                        percent=100,
                        result=self.to_status_payload(),
                        finished=True,
                    )
                else:
                    payload = self.to_status_payload()
                    last_error = next(
                        (
                            event.get("message")
                            for event in reversed(payload.get("events", []))
                            if event.get("level") == "error"
                        ),
                        None,
                    )
                    store.update(
                        self.session_id,
                        status="failed",
                        error=str(last_error or "bootstrap failed"),
                        result=payload,
                        finished=True,
                    )
        except Exception:  # noqa: BLE001
            pass


def default_build_bootstrap_runner(
    req: BootstrapRequest,
) -> Callable[[Callable[[str], None]], None]:
    """Produce a (progress_cb -> None) closure that runs the chosen installer.

    Factored out so tests can monkeypatch this builder with a stub runner that
    doesn't touch the network or the filesystem.
    """
    if req.backend == "diffusion-pipe":
        return _build_diffusion_pipe_runner(req)
    if req.backend == "anima_lora":
        return _build_anima_lora_runner(req)
    return _build_kohya_runner(req)


def _resolve_base_python(version: str | None = None) -> Path | None:
    """Return the portable Python uv has cached for us, if any.

    None means "let uv pick whatever interpreter created it" — i.e. the
    interpreter running the API. The Settings UI's Dependencies tab makes
    sure a portable runtime is always installed before the user gets here,
    so this fallback only fires for scripted / power-user flows.

    ``version`` lets a caller request a non-default runtime (e.g.
    ``"3.13"`` for anima_lora). Falls through to the default version
    when omitted.
    """
    from lorahub.core.toolchain import python_runtime  # noqa: PLC0415

    return python_runtime.runtime_python(version or python_runtime.DEFAULT_VERSION)


def _build_kohya_runner(
    req: BootstrapRequest,
) -> Callable[[Callable[[str], None]], None]:
    from lorahub.api import app as app_module  # noqa: PLC0415
    from lorahub.core.backends.kohya import installer  # noqa: PLC0415

    target_path = (
        Path(req.target).expanduser().resolve()
        if req.target
        else (Path.cwd() / "sd-scripts").resolve()
    )
    settings = app_module._settings_store.load()
    plan = installer.BootstrapPlan(
        target=target_path,
        cuda_version=req.cuda,
        torch_version=req.torch_version,
        torchvision_version=req.torchvision_version,
        install_xformers=req.install_xformers,
        github_proxy=settings.github_proxy,
        base_python=_resolve_base_python(),
        pypi_index=settings.pypi_index_url,
        torch_index_base=settings.torch_index_url,
    )
    if plan.target.exists() and any(plan.target.iterdir()):
        if not req.force:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"target {plan.target} is not empty; "
                    "pass force=true to wipe it first."
                ),
            )
        installer.cleanup_partial(plan)
        # cleanup_partial swallows individual file errors (e.g. Windows file
        # locks); double-check the directory is actually gone before we tell
        # the runner to clone into it.
        if plan.target.exists() and any(plan.target.iterdir()):
            raise HTTPException(
                status_code=409,
                detail=(
                    f"failed to clear {plan.target}; some files may be locked. "
                    "Close any tools using the directory and retry, or delete "
                    "it manually."
                ),
            )

    def runner(progress: Callable[[str], None]) -> None:
        installer.bootstrap(plan, progress=progress)

    return runner


def _build_diffusion_pipe_runner(
    req: BootstrapRequest,
) -> Callable[[Callable[[str], None]], None]:
    from lorahub.api import app as app_module  # noqa: PLC0415
    from lorahub.core.backends.diffusion_pipe import installer  # noqa: PLC0415

    target_path = (
        Path(req.target).expanduser().resolve()
        if req.target
        else (Path.cwd() / "diffusion-pipe").resolve()
    )
    settings = app_module._settings_store.load()
    plan = installer.BootstrapPlan(
        target=target_path,
        cuda_version=req.cuda,
        torch_version=req.torch_version,
        torchvision_version=req.torchvision_version,
        install_deepspeed=req.install_deepspeed,
        github_proxy=settings.github_proxy,
        base_python=_resolve_base_python(),
        pypi_index=settings.pypi_index_url,
        torch_index_base=settings.torch_index_url,
    )
    if plan.target.exists() and any(plan.target.iterdir()):
        if not req.force:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"target {plan.target} is not empty; "
                    "pass force=true to wipe it first."
                ),
            )
        installer.cleanup_partial(plan)
        # cleanup_partial swallows individual file errors (e.g. Windows file
        # locks); double-check the directory is actually gone before we tell
        # the runner to clone into it.
        if plan.target.exists() and any(plan.target.iterdir()):
            raise HTTPException(
                status_code=409,
                detail=(
                    f"failed to clear {plan.target}; some files may be locked. "
                    "Close any tools using the directory and retry, or delete "
                    "it manually."
                ),
            )

    def runner(progress: Callable[[str], None]) -> None:
        installer.bootstrap(plan, progress=progress)

    return runner


def _build_anima_lora_runner(
    req: BootstrapRequest,
) -> Callable[[Callable[[str], None]], None]:
    """anima_lora install: ``uv sync`` against the vendored copy.

    The target directory is **always** ``external/anima_lora`` (vendored).
    ``req.target`` is honoured only as a dev override — pointing at an
    alternate checkout for upstream development. ``req.cuda`` /
    ``req.torch_version`` are ignored because anima_lora's torch pin
    lives in its own ``pyproject.toml`` + ``uv.lock``.
    """
    from lorahub.api import app as app_module  # noqa: PLC0415
    from lorahub.core.backends.anima_lora import bootstrap as al_bootstrap  # noqa: PLC0415
    from lorahub.core.backends.anima_lora import installer  # noqa: PLC0415

    target_path = (
        Path(req.target).expanduser().resolve()
        if req.target
        else al_bootstrap.default_repo_path()
    )
    settings = app_module._settings_store.load()
    plan = installer.BootstrapPlan(
        target=target_path,
        # anima_lora's pyproject pins ``requires-python = "==3.13.*"``.
        # If the user already pre-fetched 3.13 from the Dependencies tab
        # (or via ``scripts/install.{sh,bat}``) we hand uv that path so
        # ``uv sync`` doesn't burn another 30s pulling its own copy of
        # python-build-standalone. When 3.13 is missing we leave
        # base_python None and let uv self-fetch — same behaviour as
        # before, just opportunistically faster.
        base_python=_resolve_base_python("3.13"),
        pypi_index=settings.pypi_index_url,
        install_deepspeed=req.install_deepspeed,
    )
    if not (plan.target / "pyproject.toml").is_file():
        raise HTTPException(
            status_code=409,
            detail=(
                f"vendored anima_lora copy at {plan.target} is missing "
                "pyproject.toml — the source tree may be corrupted."
            ),
        )
    if plan.venv_dir.exists() and any(plan.venv_dir.iterdir()):
        if not req.force:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"venv {plan.venv_dir} already exists; "
                    "pass force=true to wipe and rebuild it."
                ),
            )
        installer.cleanup_partial(plan)
        if plan.venv_dir.exists() and any(plan.venv_dir.iterdir()):
            raise HTTPException(
                status_code=409,
                detail=(
                    f"failed to clear {plan.venv_dir}; some files may be "
                    "locked. Close any tools using the venv and retry."
                ),
            )

    def runner(progress: Callable[[str], None]) -> None:
        installer.bootstrap(plan, progress=progress)

    return runner


# Lock guarding singleton creation/transition. The session itself lives on
# `lorahub.api.app._bootstrap_session` so tests can reset it via monkeypatch.
_bootstrap_lock = threading.Lock()
