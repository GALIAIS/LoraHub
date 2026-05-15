"""System & hardware telemetry endpoint.

Pure read-only — `/api/system/stats` returns a snapshot every call, and the
WebSocket at `/api/system/stream` (registered in app.py) streams a snapshot
every second.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from lorahub.api.system_stats import collect_snapshot

router = APIRouter(prefix="/api")


@router.get("/system/stats")
def system_stats() -> dict[str, Any]:
    return collect_snapshot().to_dict()
