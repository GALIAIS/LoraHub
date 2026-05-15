"""SQLite-backed persistence for the job registry.

Holds job metadata so the API can survive a restart with history intact.
The live `TrainingHandle` and the event ring buffer stay in memory only —
events are already streamed to `<workspace>/events.jsonl` so they replay
from disk if needed.

Schema migrations are not yet supported: the table is created lazily and
only one schema_version is supported. Bump `_SCHEMA_VERSION` and add an
ALTER path before changing columns in a published release.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from lorahub.api.state import JobRecord, JobState

_SCHEMA_VERSION = 1
_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id              TEXT PRIMARY KEY,
    state           TEXT NOT NULL,
    workspace       TEXT NOT NULL,
    recipe_snapshot TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    started_at      TEXT,
    finished_at     TEXT,
    returncode      INTEGER,
    error           TEXT,
    pid             INTEGER
);

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY
);
"""


class JobStore:
    """Tiny CRUD wrapper around a single SQLite file."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.RLock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            conn.execute(
                "INSERT OR IGNORE INTO schema_version (version) VALUES (?)",
                (_SCHEMA_VERSION,),
            )
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

    def upsert(self, record: JobRecord) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO jobs (id, state, workspace, recipe_snapshot, created_at,
                                  started_at, finished_at, returncode, error, pid)
                VALUES (:id, :state, :workspace, :recipe_snapshot, :created_at,
                        :started_at, :finished_at, :returncode, :error, :pid)
                ON CONFLICT(id) DO UPDATE SET
                    state           = excluded.state,
                    workspace       = excluded.workspace,
                    recipe_snapshot = excluded.recipe_snapshot,
                    started_at      = excluded.started_at,
                    finished_at     = excluded.finished_at,
                    returncode      = excluded.returncode,
                    error           = excluded.error,
                    pid             = excluded.pid
                """,
                _record_to_row(record),
            )

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return _row_to_record(row) if row is not None else None

    def list(self) -> list[JobRecord]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM jobs ORDER BY created_at"
            ).fetchall()
        return [_row_to_record(r) for r in rows]

    def mark_orphans_interrupted(self) -> int:
        """Convert any non-terminal jobs to `interrupted`. Returns rows affected.

        Run this once at server startup: any job that was running, queued, or
        canceling when the previous process died is no longer reachable.
        """
        live = ", ".join(f"'{s.value}'" for s in _LIVE_STATES)
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                f"UPDATE jobs SET state = ? WHERE state IN ({live})",  # noqa: S608
                (JobState.interrupted.value,),
            )
            return cur.rowcount


_LIVE_STATES: tuple[JobState, ...] = (
    JobState.queued,
    JobState.running,
    JobState.canceling,
)


def _record_to_row(r: JobRecord) -> dict[str, Any]:
    return {
        "id": r.id,
        "state": r.state.value,
        "workspace": str(r.workspace),
        "recipe_snapshot": json.dumps(r.recipe_snapshot, ensure_ascii=False),
        "created_at": r.created_at.isoformat(),
        "started_at": r.started_at.isoformat() if r.started_at else None,
        "finished_at": r.finished_at.isoformat() if r.finished_at else None,
        "returncode": r.returncode,
        "error": r.error,
        "pid": r.pid,
    }


def _row_to_record(row: sqlite3.Row) -> JobRecord:
    return JobRecord(
        id=row["id"],
        state=JobState(row["state"]),
        workspace=Path(row["workspace"]),
        recipe_snapshot=json.loads(row["recipe_snapshot"]),
        created_at=_parse_dt(row["created_at"]),
        started_at=_parse_dt_optional(row["started_at"]),
        finished_at=_parse_dt_optional(row["finished_at"]),
        returncode=row["returncode"],
        error=row["error"],
        pid=row["pid"],
    )


def _parse_dt(s: str) -> datetime:
    return datetime.fromisoformat(s)


def _parse_dt_optional(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s) if s else None


def default_store_path() -> Path:
    return Path.cwd() / "runs" / ".lorahub.sqlite"


__all__ = ["JobStore", "default_store_path"]
