"""System & hardware telemetry endpoint.

Pure read-only — `/api/system/stats` returns a snapshot every call, and the
WebSocket at `/api/system/stream` (registered in app.py) streams a snapshot
every second.

Self-update endpoints (`/api/system/version`, `/api/system/update`) live
here too; they're the workbench-control surface, conceptually adjacent to
``stats`` (both answer "tell me about the system itself"). The streaming
update endpoint emits SSE events for git/deps/build phases so the UI can
mirror the bootstrap-session log shape users already know.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import AsyncIterator
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from lorahub.api.runtime_bind import read_runtime_bind, restart_args
from lorahub.api import system_update
from lorahub.api.system_stats import (
    ALL_ATTENTION_BACKENDS,
    attention_backends_for_gpu,
    collect_snapshot,
)
from lorahub.api import app as app_module
from lorahub.api.task_sessions import (
    TaskEvent,
    TaskSessionStore,
    default_task_store_path,
)

router = APIRouter(prefix="/api")


@router.get("/system/stats")
def system_stats() -> dict[str, Any]:
    return collect_snapshot().to_dict()


@router.get("/system/attention-backends")
def system_attention_backends() -> dict[str, Any]:
    """Report which attention training backends are usable on this host.

    The frontend uses the `supported` list to grey out options the local
    GPU can't run (e.g. flash3 on Ada). When there's no NVIDIA GPU we
    fall back to the safe PyTorch-native set.
    """
    snapshot = collect_snapshot()
    cap: str | None = None
    for gpu in snapshot.gpus:
        if gpu.vendor == "nvidia" and gpu.compute_capability:
            cap = gpu.compute_capability
            break
    supported = attention_backends_for_gpu(cap)
    return {
        "compute_capability": cap,
        "supported": supported,
        "all": list(ALL_ATTENTION_BACKENDS),
    }


@router.get("/system/cluster")
def system_cluster() -> dict[str, Any]:
    """Multi-node DeepSpeed launcher readiness probe (B8).

    The frontend uses this to surface "single-node only" vs "cluster
    ready" so users picking ``backend.diffusionPipe.multiNode`` see
    upfront whether their environment is wired up. We don't attempt
    SSH / DeepSpeed health-check here — that's the launcher's job at
    job start. Instead we just report what we can see locally:

      * is ``deepspeed`` resolvable in the dp venv?
      * does the user's currently-configured hostfile exist?
      * do the two key prerequisites (passwordless SSH + matching
        repo paths) have any signal we can detect?

    Returns ``ready`` true only when DeepSpeed is callable; everything
    else surfaces as advisory hints. Mostly read-only and cheap.
    """
    from pathlib import Path  # noqa: PLC0415
    from shutil import which  # noqa: PLC0415

    from lorahub.api import app as app_module  # noqa: PLC0415
    from lorahub.core.backends.diffusion_pipe import bootstrap as dp_bootstrap  # noqa: PLC0415

    try:
        env = dp_bootstrap.resolve()
    except dp_bootstrap.BootstrapError as exc:
        return {
            "ready": False,
            "reason": f"diffusion-pipe not configured: {exc}",
            "deepspeed_path": None,
            "hostfile": None,
            "hostfile_exists": False,
        }

    deepspeed_bin = env.python_executable.parent / "deepspeed"
    if not deepspeed_bin.is_file():
        deepspeed_bin = env.python_executable.parent / "deepspeed.exe"
    deepspeed_ok = deepspeed_bin.is_file()

    settings = (
        app_module._settings_store.load() if app_module._settings_store else None
    )
    extra: dict[str, Any] = {}
    if settings is not None:
        # Surface the user's last-saved hostfile if any — pulled from
        # the global Settings extras bag so users don't have to bake it
        # into every config. Settings.extra is the existing escape
        # hatch for fields not yet in the schema.
        hostfile_str = settings.extra.get("multi_node_hostfile")
        if isinstance(hostfile_str, str) and hostfile_str.strip():
            hostfile = Path(hostfile_str).expanduser()
            extra["hostfile"] = str(hostfile)
            extra["hostfile_exists"] = hostfile.is_file()
        else:
            extra["hostfile"] = None
            extra["hostfile_exists"] = False
    else:
        extra["hostfile"] = None
        extra["hostfile_exists"] = False

    return {
        "ready": deepspeed_ok,
        "deepspeed_path": str(deepspeed_bin) if deepspeed_ok else None,
        "ssh_available": which("ssh") is not None,
        **extra,
    }


# --------------------------------------------------------------------------- #
# Self-update                                                                 #
# --------------------------------------------------------------------------- #


# ``main`` is accepted alongside the canonical ``dev`` so old API
# clients (and any cached UpdateInfo blob persisted before the
# v1.0.4 rename) keep working — system_update.check rewrites it
# through ``_LEGACY_CHANNEL_ALIASES``.
_VALID_CHANNELS = ("dev", "tag", "main")
_UPDATE_LOCK = threading.Lock()
_UPDATE_SSE_PING_INTERVAL = 15.0
_KIND_SYSTEM_UPDATE = "system_update"


def _task_store() -> TaskSessionStore:
    store = app_module._task_session_store
    if store is None:
        store = TaskSessionStore(default_task_store_path())
        app_module._task_session_store = store
    return store


@router.get("/system/version")
def system_version(
    channel: Literal["dev", "tag", "main"] = "tag",
    force: bool = False,
) -> dict[str, Any]:
    """Resolve current vs remote for the given channel.

    Cached for 5 minutes per channel; pass ``force=true`` to bypass.
    Network errors degrade to the cached payload + an ``error`` field
    so the UI can render "offline, last seen v1.0.5" rather than
    nothing.
    """
    info = system_update.check(channel=channel, force=force)
    return info.to_dict()


class _UpdateRequest(BaseModel):
    channel: Literal["dev", "tag", "main"] = "tag"
    build: bool = True
    restart: bool = True
    # Destructive: when True, ``git reset --hard`` + ``git clean -fd``
    # blow away local changes before checkout. The UI guards this
    # behind an explicit confirm dialog; the API itself stays cheap
    # and stateless and trusts the caller's intent.
    force: bool = False


@router.post("/system/update")
async def system_update_apply(req: _UpdateRequest) -> StreamingResponse:
    """Run the upgrade. Streams progress via SSE.

    Each event is a JSON object on a single line:

        {"phase": "git", "level": "info", "message": "git fetch ..."}

    Terminal events:
        {"phase": "done", "level": "info", "message": "update applied"}
        {"phase": "error", "level": "error", "message": "..."}
        {"phase": "restart", "level": "info", "message": "service restarting"}
    """
    if req.channel not in _VALID_CHANNELS:
        raise HTTPException(422, f"channel must be one of {_VALID_CHANNELS}")
    if not _UPDATE_LOCK.acquire(blocking=False):
        raise HTTPException(409, "system update is already running")

    task = _task_store().create(
        kind=_KIND_SYSTEM_UPDATE,
        title=f"system update:{req.channel}",
        metadata={
            "channel": req.channel,
            "build": req.build,
            "restart": req.restart,
            "force": req.force,
        },
    )
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def emit(phase: str, level: str, message: str) -> None:
        event = {"phase": phase, "level": level, "message": message}
        try:
            _task_store().append_event(
                task.id,
                TaskEvent(
                    level=level,
                    message=message,
                    payload=event,
                    ts=time.time(),
                ),
            )
        except Exception:  # noqa: BLE001
            pass
        # Cross-thread put_nowait via the running event loop.
        loop.call_soon_threadsafe(queue.put_nowait, event)

    def runner() -> None:
        try:
            _task_store().update(task.id, status="running", percent=0)
            system_update.apply(
                channel=req.channel,
                build=req.build,
                progress=emit,
                force=req.force,
            )
        except Exception as exc:  # noqa: BLE001
            emit("error", "error", f"{type(exc).__name__}: {exc}")
            _task_store().update(
                task.id,
                status="failed",
                error=str(exc),
                result={"channel": req.channel, "build": req.build},
                finished=True,
            )
            loop.call_soon_threadsafe(queue.put_nowait, None)
            _UPDATE_LOCK.release()
            return
        try:
            _task_store().update(
                task.id,
                status="succeeded",
                percent=100,
                result={"channel": req.channel, "build": req.build},
                finished=True,
            )
            if req.restart:
                bind = read_runtime_bind()
                suffix = (
                    f" on {bind.host}:{bind.port}"
                    if bind is not None
                    else " with the current command"
                )
                emit("restart", "info", f"scheduling service restart{suffix} in 1.5s")
                # Defer the actual restart so the SSE stream has time to
                # flush the final event to the browser before uvicorn
                # shuts down. We use a daemon thread instead of asyncio
                # because the executor that ran apply() has already gone.
                threading.Timer(1.5, _trigger_restart).start()
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)
            _UPDATE_LOCK.release()

    threading.Thread(
        target=runner, name="system-update", daemon=True
    ).start()

    async def stream() -> AsyncIterator[bytes]:
        # SSE framing: each event is "data: <json>\n\n".
        yield b": lorahub system update stream\n\n"
        while True:
            try:
                event = await asyncio.wait_for(
                    queue.get(),
                    timeout=_UPDATE_SSE_PING_INTERVAL,
                )
            except asyncio.TimeoutError:
                yield b": ping\n\n"
                continue
            if event is None:
                break
            yield (f"data: {json.dumps(event, ensure_ascii=False)}\n\n").encode("utf-8")

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _trigger_restart() -> None:
    """Re-exec the current process so the just-installed code wins.

    POSIX: ``os.execv`` replaces the running uvicorn process in place.
    Windows: spawn a fresh process detached + exit cleanly. Either way
    the systemd / launchd / Task Scheduler unit (if any) sees a clean
    exit and won't fight us — we set ``Restart=on-failure`` so a
    voluntary exit followed by a re-spawn from the unit is fine.
    """
    import os
    import sys

    args = restart_args(sys.executable, sys.argv)
    if sys.platform == "win32":
        # Spawn detached child running same argv, then bail.
        import subprocess

        subprocess.Popen(  # noqa: S603
            args,
            close_fds=True,
            creationflags=0x00000008 | 0x08000000,  # DETACHED_PROCESS|CREATE_NO_WINDOW
        )
        os._exit(0)  # noqa: SLF001 — needed: skip atexit / running tasks
        return
    os.execv(sys.executable, args)
