"""SQLite-backed persistence for one-off sessions (tagging / captions / bootstrap).

These three subsystems share the same shape: a long-running background task
that emits structured events and lands in a terminal state (succeeded /
failed). Today they live entirely in memory — restarting the API server
loses every record, even of *completed* runs.

This store keeps a row per session so users can still see "what happened"
after a restart. It deliberately doesn't keep the live event stream — that's
the responsibility of the in-memory session class while the task is alive.
Only terminal snapshots are flushed here.

Schema is one table per session kind so different domains can grow their
own columns without invalidating each other's rows.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tagging_sessions (
    session_id    TEXT PRIMARY KEY,
    path          TEXT NOT NULL,
    status        TEXT NOT NULL,
    written       INTEGER,
    total         INTEGER,
    device        TEXT,
    error         TEXT,
    started_at    REAL NOT NULL,
    finished_at   REAL,
    snapshot      TEXT
);

CREATE TABLE IF NOT EXISTS captions_sessions (
    session_id    TEXT PRIMARY KEY,
    path          TEXT NOT NULL,
    status        TEXT NOT NULL,
    written       INTEGER,
    total         INTEGER,
    error         TEXT,
    started_at    REAL NOT NULL,
    finished_at   REAL,
    snapshot      TEXT
);

CREATE TABLE IF NOT EXISTS bootstrap_sessions (
    session_id    TEXT PRIMARY KEY,
    backend       TEXT NOT NULL,
    status        TEXT NOT NULL,
    error         TEXT,
    started_at    REAL NOT NULL,
    finished_at   REAL,
    snapshot      TEXT
);

CREATE INDEX IF NOT EXISTS idx_tagging_started   ON tagging_sessions(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_captions_started  ON captions_sessions(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_bootstrap_started ON bootstrap_sessions(started_at DESC);
"""

SessionKind = Literal["tagging", "captions", "bootstrap"]
_TABLES: dict[SessionKind, str] = {
    "tagging": "tagging_sessions",
    "captions": "captions_sessions",
    "bootstrap": "bootstrap_sessions",
}


class SessionStore:
    """CRUD wrapper around the three session tables in one SQLite file."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.RLock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            conn.commit()

    @property
    def path(self) -> Path:
        return self._path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path), isolation_level=None, timeout=10.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        return conn

    # ---------------- Tagging ---------------- #

    def upsert_tagging(self, snapshot: dict[str, Any]) -> None:
        """Persist a tagging session snapshot. Safe to call repeatedly."""
        row = {
            "session_id": snapshot["session_id"],
            "path": snapshot.get("path", ""),
            "status": snapshot.get("status", "running"),
            "written": snapshot.get("written"),
            "total": snapshot.get("total"),
            "device": snapshot.get("device") or snapshot.get("active_provider"),
            "error": snapshot.get("error"),
            "started_at": _coerce_ts(snapshot.get("started_at")),
            "finished_at": _coerce_ts(snapshot.get("finished_at")),
            "snapshot": json.dumps(snapshot, ensure_ascii=False, default=str),
        }
        with self._lock, self._connect() as conn:
            conn.execute(_UPSERT_TAGGING_SQL, row)

    # ---------------- Captions ---------------- #

    def upsert_captions(self, snapshot: dict[str, Any]) -> None:
        row = {
            "session_id": snapshot["session_id"],
            "path": snapshot.get("path", ""),
            "status": snapshot.get("status", "running"),
            "written": snapshot.get("written"),
            "total": snapshot.get("total"),
            "error": snapshot.get("error"),
            "started_at": _coerce_ts(snapshot.get("started_at")),
            "finished_at": _coerce_ts(snapshot.get("finished_at")),
            "snapshot": json.dumps(snapshot, ensure_ascii=False, default=str),
        }
        with self._lock, self._connect() as conn:
            conn.execute(_UPSERT_CAPTIONS_SQL, row)

    # ---------------- Bootstrap ---------------- #

    def upsert_bootstrap(self, snapshot: dict[str, Any]) -> None:
        row = {
            "session_id": snapshot["session_id"],
            "backend": snapshot.get("backend", ""),
            "status": snapshot.get("status", "running"),
            "error": snapshot.get("error"),
            "started_at": _coerce_ts(
                snapshot.get("started_at") or snapshot.get("created_at")
            ),
            "finished_at": _coerce_ts(snapshot.get("finished_at")),
            "snapshot": json.dumps(snapshot, ensure_ascii=False, default=str),
        }
        with self._lock, self._connect() as conn:
            conn.execute(_UPSERT_BOOTSTRAP_SQL, row)

    # ---------------- Query helpers ---------------- #

    def list_recent(self, kind: SessionKind, limit: int = 50) -> list[dict[str, Any]]:
        table = _TABLES[kind]
        limit = max(1, min(100, int(limit)))
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                f"SELECT * FROM {table} ORDER BY started_at DESC LIMIT ?",  # noqa: S608
                (limit,),
            ).fetchall()
        return [_row_to_dict(r) for r in rows]

    def get(self, kind: SessionKind, session_id: str) -> dict[str, Any] | None:
        table = _TABLES[kind]
        with self._lock, self._connect() as conn:
            row = conn.execute(
                f"SELECT * FROM {table} WHERE session_id = ?",  # noqa: S608
                (session_id,),
            ).fetchone()
        return _row_to_dict(row) if row else None


_UPSERT_TAGGING_SQL = """
INSERT INTO tagging_sessions (session_id, path, status, written, total, device,
                              error, started_at, finished_at, snapshot)
VALUES (:session_id, :path, :status, :written, :total, :device,
        :error, :started_at, :finished_at, :snapshot)
ON CONFLICT(session_id) DO UPDATE SET
    status      = excluded.status,
    written     = excluded.written,
    total       = excluded.total,
    device      = excluded.device,
    error       = excluded.error,
    finished_at = excluded.finished_at,
    snapshot    = excluded.snapshot
"""

_UPSERT_CAPTIONS_SQL = """
INSERT INTO captions_sessions (session_id, path, status, written, total,
                               error, started_at, finished_at, snapshot)
VALUES (:session_id, :path, :status, :written, :total,
        :error, :started_at, :finished_at, :snapshot)
ON CONFLICT(session_id) DO UPDATE SET
    status      = excluded.status,
    written     = excluded.written,
    total       = excluded.total,
    error       = excluded.error,
    finished_at = excluded.finished_at,
    snapshot    = excluded.snapshot
"""

_UPSERT_BOOTSTRAP_SQL = """
INSERT INTO bootstrap_sessions (session_id, backend, status, error,
                                started_at, finished_at, snapshot)
VALUES (:session_id, :backend, :status, :error,
        :started_at, :finished_at, :snapshot)
ON CONFLICT(session_id) DO UPDATE SET
    status      = excluded.status,
    error       = excluded.error,
    finished_at = excluded.finished_at,
    snapshot    = excluded.snapshot
"""


def _coerce_ts(value: Any) -> float | None:
    """Accept either ISO strings, datetimes, or numeric timestamps."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, datetime):
        return value.timestamp()
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value).timestamp()
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return None
    return None


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    out = {k: row[k] for k in row.keys()}
    if out.get("snapshot"):
        try:
            out["snapshot"] = json.loads(out["snapshot"])
        except (TypeError, json.JSONDecodeError):
            pass
    return out


def default_session_store_path() -> Path:
    """``<project_root>/runs/sessions.sqlite``."""
    from lorahub.api.paths import runs_dir  # noqa: PLC0415

    return runs_dir() / "sessions.sqlite"


__all__ = ["SessionStore", "SessionKind", "default_session_store_path"]
