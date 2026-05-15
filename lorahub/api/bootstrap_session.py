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


class BootstrapRequest(BaseModel):
    backend: Literal["kohya", "diffusion-pipe"] = "kohya"
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

    def __init__(self, session_id: str, backend: str = "kohya") -> None:
        self.session_id = session_id
        self.backend = backend
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

    def _finalize(self, status: str) -> None:
        with self._lock:
            self.status = status
            listeners = list(self._listeners)
        # Wake any listener still parked on `queue.get()` so it can close.
        sentinel: dict[str, Any] = {"step": "__terminal__", "level": status}
        for queue in listeners:
            self._dispatch(queue, sentinel)

    def _dispatch(self, queue: asyncio.Queue[dict[str, Any]], event: dict[str, Any]) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        # Loop already torn down — listener is gone, drop the event.
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(queue.put_nowait, event)


def default_build_bootstrap_runner(
    req: BootstrapRequest,
) -> Callable[[Callable[[str], None]], None]:
    """Produce a (progress_cb -> None) closure that runs the chosen installer.

    Factored out so tests can monkeypatch this builder with a stub runner that
    doesn't touch the network or the filesystem.
    """
    if req.backend == "diffusion-pipe":
        return _build_diffusion_pipe_runner(req)
    return _build_kohya_runner(req)


def _build_kohya_runner(
    req: BootstrapRequest,
) -> Callable[[Callable[[str], None]], None]:
    from lorahub.core.backends.kohya import installer  # noqa: PLC0415

    target_path = (
        Path(req.target).expanduser().resolve()
        if req.target
        else (Path.cwd() / "sd-scripts").resolve()
    )
    plan = installer.BootstrapPlan(
        target=target_path,
        cuda_version=req.cuda,
        torch_version=req.torch_version,
        torchvision_version=req.torchvision_version,
        install_xformers=req.install_xformers,
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

    def runner(progress: Callable[[str], None]) -> None:
        installer.bootstrap(plan, progress=progress)

    return runner


def _build_diffusion_pipe_runner(
    req: BootstrapRequest,
) -> Callable[[Callable[[str], None]], None]:
    from lorahub.core.backends.diffusion_pipe import installer  # noqa: PLC0415

    target_path = (
        Path(req.target).expanduser().resolve()
        if req.target
        else (Path.cwd() / "diffusion-pipe").resolve()
    )
    plan = installer.BootstrapPlan(
        target=target_path,
        cuda_version=req.cuda,
        torch_version=req.torch_version,
        torchvision_version=req.torchvision_version,
        install_deepspeed=req.install_deepspeed,
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

    def runner(progress: Callable[[str], None]) -> None:
        installer.bootstrap(plan, progress=progress)

    return runner


# Lock guarding singleton creation/transition. The session itself lives on
# `lorahub.api.app._bootstrap_session` so tests can reset it via monkeypatch.
_bootstrap_lock = threading.Lock()
