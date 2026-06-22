"""Shared task-session helpers for Image Studio AI endpoints."""

from __future__ import annotations

from typing import Any

from lorahub.api import app as app_module
from lorahub.api.task_sessions import (
    TaskSession,
    TaskSessionStore,
    default_task_store_path,
)


def get_task_store() -> TaskSessionStore:
    store = app_module._task_session_store
    if store is None:
        store = TaskSessionStore(default_task_store_path())
        app_module._task_session_store = store
    return store


def persisted_task_result(session_id: str, kind: str) -> dict[str, Any] | None:
    try:
        task = get_task_store().get(session_id)
    except Exception:
        return None
    if task is None or task.kind != kind:
        return None
    return task_to_status_snapshot(task)


def task_to_status_snapshot(task: TaskSession) -> dict[str, Any] | None:
    if isinstance(task.result, dict):
        result = dict(task.result)
        result.setdefault("events", [event.to_dict() for event in task.events])
        return result
    if task.status in {"queued", "running", "interrupted", "failed", "canceled"}:
        metadata = task.metadata
        total = int(metadata.get("total") or metadata.get("selected") or 0)
        path = str(metadata.get("path") or metadata.get("dataset_path") or "")
        return {
            "session_id": task.id,
            "path": path,
            "status": task.status,
            "processed": 0,
            "total": total,
            "skipped": int(metadata.get("skipped") or 0),
            "percent": task.percent,
            "last_image": "",
            "results": [],
            "errors": [],
            "dataset_top": [],
            "error": task.error,
            "started_at": task.started_at,
            "finished_at": task.finished_at,
            "events": [event.to_dict() for event in task.events],
        }
    return None
