"""Persistent background task session store.

This module intentionally owns only durable task state. Routers still own
execution and concurrency; they append state transitions here so status reads
can survive browser refreshes and API restarts.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


TaskStatus = Literal[
    "queued",
    "running",
    "succeeded",
    "failed",
    "canceled",
    "interrupted",
]


@dataclass(frozen=True, slots=True)
class TaskEvent:
    level: str
    message: str
    percent: float | None = None
    payload: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "message": self.message,
            "percent": self.percent,
            "payload": dict(self.payload),
            "ts": self.ts,
        }


@dataclass(frozen=True, slots=True)
class TaskSession:
    id: str
    kind: str
    title: str
    status: TaskStatus
    percent: float
    metadata: dict[str, Any]
    result: dict[str, Any] | None
    error: str | None
    started_at: float
    updated_at: float
    finished_at: float | None
    events: list[TaskEvent]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "title": self.title,
            "status": self.status,
            "percent": self.percent,
            "metadata": dict(self.metadata),
            "result": self.result,
            "error": self.error,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "finished_at": self.finished_at,
            "events": [event.to_dict() for event in self.events],
        }


class TaskSessionStore:
    def __init__(self, path: Path, *, max_events: int = 200) -> None:
        self.path = path.expanduser().resolve()
        self.max_events = max(1, int(max_events))
        self._lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS task_sessions (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    title TEXT NOT NULL,
                    status TEXT NOT NULL,
                    percent REAL NOT NULL,
                    metadata_json TEXT NOT NULL,
                    result_json TEXT,
                    error TEXT,
                    started_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    finished_at REAL
                )
                """,
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS task_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    percent REAL,
                    payload_json TEXT NOT NULL,
                    ts REAL NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES task_sessions(id) ON DELETE CASCADE
                )
                """,
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_task_sessions_kind_updated
                ON task_sessions(kind, updated_at DESC)
                """,
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_task_events_session_id_id
                ON task_events(session_id, id)
                """,
            )

    def create(self, *, kind: str, title: str, metadata: dict[str, Any]) -> TaskSession:
        now = time.time()
        session_id = uuid.uuid4().hex
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO task_sessions (
                    id, kind, title, status, percent, metadata_json,
                    result_json, error, started_at, updated_at, finished_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    kind,
                    title,
                    "queued",
                    0.0,
                    json.dumps(metadata),
                    None,
                    None,
                    now,
                    now,
                    None,
                ),
            )
        loaded = self.get(session_id)
        assert loaded is not None
        return loaded

    def update(
        self,
        session_id: str,
        *,
        status: TaskStatus | None = None,
        percent: float | None = None,
        result: dict[str, Any] | None = None,
        error: str | None = None,
        finished: bool = False,
    ) -> None:
        loaded = self.get(session_id)
        if loaded is None:
            return
        next_status = status or loaded.status
        next_percent = (
            loaded.percent if percent is None else max(0.0, min(100.0, float(percent)))
        )
        now = time.time()
        finished_at = now if finished else loaded.finished_at
        result_json = (
            json.dumps(result)
            if result is not None
            else json.dumps(loaded.result)
            if loaded.result is not None
            else None
        )
        next_error = error if error is not None else loaded.error
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                UPDATE task_sessions
                SET status=?, percent=?, result_json=?, error=?, updated_at=?, finished_at=?
                WHERE id=?
                """,
                (
                    next_status,
                    next_percent,
                    result_json,
                    next_error,
                    now,
                    finished_at,
                    session_id,
                ),
            )

    def append_event(self, session_id: str, event: TaskEvent) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO task_events (session_id, level, message, percent, payload_json, ts)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    event.level,
                    event.message,
                    event.percent,
                    json.dumps(event.payload),
                    event.ts,
                ),
            )
            conn.execute(
                """
                UPDATE task_sessions
                SET updated_at=?, percent=COALESCE(?, percent)
                WHERE id=?
                """,
                (event.ts, event.percent, session_id),
            )
            rows = conn.execute(
                """
                SELECT id
                FROM task_events
                WHERE session_id=?
                ORDER BY id DESC
                LIMIT -1 OFFSET ?
                """,
                (session_id, self.max_events),
            ).fetchall()
            if rows:
                ids = [int(row["id"]) for row in rows]
                placeholders = ",".join("?" for _ in ids)
                conn.execute(
                    f"DELETE FROM task_events WHERE id IN ({placeholders})",
                    ids,
                )

    def get(self, session_id: str) -> TaskSession | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM task_sessions WHERE id=?",
                (session_id,),
            ).fetchone()
            if row is None:
                return None
            return self._row_to_session(conn, row)

    def latest(self, kind: str) -> TaskSession | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT *
                FROM task_sessions
                WHERE kind=?
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (kind,),
            ).fetchone()
            if row is None:
                return None
            return self._row_to_session(conn, row)

    def list_recent(self, *, kind: str | None = None, limit: int = 20) -> list[TaskSession]:
        limit = max(1, min(100, int(limit)))
        with self._lock, self._connect() as conn:
            if kind:
                rows = conn.execute(
                    """
                    SELECT *
                    FROM task_sessions
                    WHERE kind=?
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """,
                    (kind, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT *
                    FROM task_sessions
                    ORDER BY updated_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            return [self._row_to_session(conn, row) for row in rows]

    def mark_stale_interrupted(self) -> int:
        count = 0
        for session in self.list_recent(limit=100):
            if session.status not in ("queued", "running"):
                continue
            self.update(
                session.id,
                status="interrupted",
                error="task interrupted by server restart",
                finished=True,
            )
            self.append_event(
                session.id,
                TaskEvent(
                    level="warn",
                    message="task interrupted by server restart",
                    percent=session.percent,
                ),
            )
            count += 1
        return count

    def _row_to_session(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
    ) -> TaskSession:
        event_rows = conn.execute(
            """
            SELECT *
            FROM task_events
            WHERE session_id=?
            ORDER BY id ASC
            """,
            (row["id"],),
        ).fetchall()
        events = [
            TaskEvent(
                level=event_row["level"],
                message=event_row["message"],
                percent=event_row["percent"],
                payload=json.loads(event_row["payload_json"] or "{}"),
                ts=event_row["ts"],
            )
            for event_row in event_rows
        ]
        return TaskSession(
            id=row["id"],
            kind=row["kind"],
            title=row["title"],
            status=row["status"],
            percent=float(row["percent"]),
            metadata=json.loads(row["metadata_json"] or "{}"),
            result=json.loads(row["result_json"]) if row["result_json"] else None,
            error=row["error"],
            started_at=float(row["started_at"]),
            updated_at=float(row["updated_at"]),
            finished_at=(
                float(row["finished_at"]) if row["finished_at"] is not None else None
            ),
            events=events,
        )
