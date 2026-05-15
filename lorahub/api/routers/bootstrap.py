"""Backend bootstrap (one-click kohya install).

The runner factory and active session are kept on `lorahub.api.app` so tests
can monkeypatch them: `app._build_bootstrap_runner` (callable) and
`app._bootstrap_session` (the live singleton). Both are dereferenced at
request time rather than imported at module import.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException

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
    if sess is None:
        return {"status": "idle", "session_id": None, "events": []}
    return sess.to_status_payload()


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
