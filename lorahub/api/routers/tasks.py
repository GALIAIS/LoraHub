"""Generic background task session read endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query

from lorahub.api import app as app_module
from lorahub.api.task_sessions import TaskSessionStore, default_task_store_path

router = APIRouter(prefix="/api")


def _store() -> TaskSessionStore:
    store = app_module._task_session_store
    if store is None:
        store = TaskSessionStore(default_task_store_path())
        app_module._task_session_store = store
    return store


@router.get("/tasks")
def list_tasks(
    kind: str | None = None,
    limit: int = Query(default=20, ge=1, le=100),
) -> dict[str, Any]:
    return {
        "tasks": [
            session.to_dict()
            for session in _store().list_recent(kind=kind, limit=limit)
        ],
    }


@router.get("/tasks/latest")
def latest_task(kind: str) -> dict[str, Any]:
    session = _store().latest(kind)
    if session is None:
        raise HTTPException(status_code=404, detail="task session not found")
    return session.to_dict()


@router.get("/tasks/{session_id}")
def get_task(session_id: str) -> dict[str, Any]:
    session = _store().get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="task session not found")
    return session.to_dict()
