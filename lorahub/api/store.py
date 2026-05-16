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
    config_snapshot TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    started_at      TEXT,
    finished_at     TEXT,
    returncode      INTEGER,
    error           TEXT,
    pid             INTEGER,
    metadata        TEXT
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
            # Idempotent migration: legacy databases created before the
            # `metadata` column existed need an in-place ADD COLUMN. SQLite's
            # ALTER doesn't support IF NOT EXISTS for columns, so we sniff
            # `PRAGMA table_info` first and only ALTER when missing. Safe to
            # run on fresh DBs (the column is already there from _SCHEMA).
            cur = conn.execute("PRAGMA table_info(jobs)")
            cols = {row[1] for row in cur.fetchall()}
            if "metadata" not in cols:
                conn.execute("ALTER TABLE jobs ADD COLUMN metadata TEXT")
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
                INSERT INTO jobs (id, state, workspace, config_snapshot, created_at,
                                  started_at, finished_at, returncode, error, pid,
                                  metadata)
                VALUES (:id, :state, :workspace, :config_snapshot, :created_at,
                        :started_at, :finished_at, :returncode, :error, :pid,
                        :metadata)
                ON CONFLICT(id) DO UPDATE SET
                    state           = excluded.state,
                    workspace       = excluded.workspace,
                    config_snapshot = excluded.config_snapshot,
                    started_at      = excluded.started_at,
                    finished_at     = excluded.finished_at,
                    returncode      = excluded.returncode,
                    error           = excluded.error,
                    pid             = excluded.pid,
                    metadata        = excluded.metadata
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

    def delete(self, job_id: str) -> bool:
        """Remove a job row. Returns True if a row was deleted."""
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            return cur.rowcount > 0

    def mark_orphans_interrupted(self) -> int:
        """Convert any non-terminal jobs to `interrupted`. Returns rows affected.

        Run this once at server startup: any job that was running or
        canceling when the previous process died is no longer reachable —
        unless its PID is still alive on this host. Training subprocesses
        spawned via deepspeed launch can outlive the API process (especially
        when uvicorn is killed without graceful shutdown), so we probe each
        live job's PID with `os.kill(pid, 0)` first and skip the row if the
        kernel reports it's still running.

        Queued jobs are intentionally left alone — they never started, so
        they have no PID and no checkpoint to resume from. The lifespan
        hook re-enqueues them into the scheduler so they pick up where
        they were waiting (see ``requeue_pending`` below).
        """
        import os  # noqa: PLC0415

        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT id, pid FROM jobs WHERE state IN ({})".format(  # noqa: S608, UP032
                    ", ".join(f"'{s.value}'" for s in _LIVE_STATES)
                ),
            ).fetchall()
            stale_ids: list[str] = []
            survivors: list[str] = []
            for row in rows:
                pid = row["pid"]
                if pid is None or not _pid_alive(pid):
                    stale_ids.append(row["id"])
                else:
                    survivors.append(row["id"])
            if not stale_ids:
                return 0
            placeholders = ", ".join("?" * len(stale_ids))
            conn.execute(
                f"UPDATE jobs SET state = ? WHERE id IN ({placeholders})",  # noqa: S608
                (JobState.interrupted.value, *stale_ids),
            )
            if survivors:
                # Best-effort log so operators can see why some interrupted
                # jobs didn't get re-marked. Print rather than `log.info` to
                # avoid coupling the store to a logger config.
                print(
                    f"[store] kept {len(survivors)} job(s) marked running because "
                    f"their PID was still alive: {survivors}",
                    flush=True,
                )
            return len(stale_ids)


def _pid_alive(pid: int) -> bool:
    """Return True if the OS still has a process with this PID."""
    import os  # noqa: PLC0415

    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # The process exists but is owned by another user; treat as alive
        # rather than risk reaping a real run.
        return True
    except OSError:
        return False
    return True


_LIVE_STATES: tuple[JobState, ...] = (
    JobState.running,
    JobState.canceling,
)


def _record_to_row(r: JobRecord) -> dict[str, Any]:
    return {
        "id": r.id,
        "state": r.state.value,
        "workspace": str(r.workspace),
        "config_snapshot": json.dumps(r.config_snapshot, ensure_ascii=False),
        "created_at": r.created_at.isoformat(),
        "started_at": r.started_at.isoformat() if r.started_at else None,
        "finished_at": r.finished_at.isoformat() if r.finished_at else None,
        "returncode": r.returncode,
        "error": r.error,
        "pid": r.pid,
        "metadata": (
            json.dumps(r.metadata, ensure_ascii=False) if r.metadata is not None else None
        ),
    }


def _row_to_record(row: sqlite3.Row) -> JobRecord:
    # Legacy rows (pre-metadata migration) won't have the column even after
    # the ALTER; sqlite3.Row.keys() reflects the SELECT *, so guard the lookup.
    metadata: dict[str, Any] | None = None
    if "metadata" in row.keys() and row["metadata"] is not None:
        metadata = json.loads(row["metadata"])
    return JobRecord(
        id=row["id"],
        state=JobState(row["state"]),
        workspace=Path(row["workspace"]),
        config_snapshot=json.loads(row["config_snapshot"]),
        created_at=_parse_dt(row["created_at"]),
        started_at=_parse_dt_optional(row["started_at"]),
        finished_at=_parse_dt_optional(row["finished_at"]),
        returncode=row["returncode"],
        error=row["error"],
        pid=row["pid"],
        metadata=metadata,
    )


def _parse_dt(s: str) -> datetime:
    return datetime.fromisoformat(s)


def _parse_dt_optional(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s) if s else None


def default_store_path() -> Path:
    """Where lorahub keeps the jobs DB.

    Returns ``runs/jobs.sqlite`` going forward. If a legacy
    ``runs/.lorahub.sqlite`` exists from an older release we keep using it
    so users don't lose history on upgrade.
    """
    runs = Path.cwd() / "runs"
    legacy = runs / ".lorahub.sqlite"
    if legacy.is_file():
        return legacy
    return runs / "jobs.sqlite"


__all__ = ["JobStore", "default_store_path"]
