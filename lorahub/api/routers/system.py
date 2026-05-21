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
from collections.abc import AsyncIterator
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from lorahub.api import system_update
from lorahub.api.system_stats import (
    ALL_ATTENTION_BACKENDS,
    attention_backends_for_gpu,
    collect_snapshot,
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
        # into every recipe. Settings.extra is the existing escape
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


_VALID_CHANNELS = ("main", "tag")


@router.get("/system/version")
def system_version(
    channel: Literal["main", "tag"] = "tag",
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
    channel: Literal["main", "tag"] = "tag"
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

    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def emit(phase: str, level: str, message: str) -> None:
        # Cross-thread put_nowait via the running event loop.
        loop.call_soon_threadsafe(
            queue.put_nowait,
            {"phase": phase, "level": level, "message": message},
        )

    def runner() -> None:
        try:
            system_update.apply(
                channel=req.channel,
                build=req.build,
                progress=emit,
                force=req.force,
            )
        except Exception as exc:  # noqa: BLE001
            emit("error", "error", f"{type(exc).__name__}: {exc}")
            loop.call_soon_threadsafe(queue.put_nowait, None)
            return
        if req.restart:
            emit("restart", "info", "scheduling service restart in 1.5s")
            # Defer the actual restart so the SSE stream has time to
            # flush the final event to the browser before uvicorn
            # shuts down. We use a daemon thread instead of asyncio
            # because the executor that ran apply() has already gone.
            threading.Timer(1.5, _trigger_restart).start()
        loop.call_soon_threadsafe(queue.put_nowait, None)

    threading.Thread(
        target=runner, name="system-update", daemon=True
    ).start()

    async def stream() -> AsyncIterator[bytes]:
        # SSE framing: each event is "data: <json>\n\n".
        while True:
            event = await queue.get()
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

    args = [sys.executable, *sys.argv]
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
