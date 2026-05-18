"""System & hardware telemetry endpoint.

Pure read-only — `/api/system/stats` returns a snapshot every call, and the
WebSocket at `/api/system/stream` (registered in app.py) streams a snapshot
every second.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

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
