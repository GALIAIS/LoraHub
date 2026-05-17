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
