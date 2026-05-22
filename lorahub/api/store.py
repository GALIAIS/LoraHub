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
import logging
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from lorahub.api.state import JobRecord, JobState

log = logging.getLogger(__name__)

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
            # Idempotent migration for old jobs.sqlite files predating the
            # `metadata` column. SQLite ALTER doesn't support IF NOT EXISTS
            # for columns, so we sniff PRAGMA table_info first.
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
        live job's PID with `os.kill(pid, 0)` first.

        Reachability matters for resource hygiene too: a training process
        that survived a uvicorn crash holds GPU memory until it's killed.
        Once it's been reparented to PID 1 we're never going to talk to it
        again — we'd just be tracking a zombie that the user can't cancel.
        Such processes are reaped (SIGKILL the whole pgroup) so their VRAM
        gets freed before the new lifecycle begins.

        Queued jobs are intentionally left alone — they never started, so
        they have no PID and no checkpoint to resume from. The lifespan
        hook re-enqueues them into the scheduler so they pick up where
        they were waiting (see ``requeue_pending`` below).
        """
        import os  # noqa: PLC0415

        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT id, pid, metadata FROM jobs WHERE state IN ({})".format(  # noqa: S608, UP032
                    ", ".join(f"'{s.value}'" for s in _LIVE_STATES)
                ),
            ).fetchall()
            stale_ids: list[str] = []
            survivors: list[str] = []
            reaped_pids: list[int] = []
            for row in rows:
                pid = row["pid"]
                expected_create_time: float | None = None
                if row["metadata"] is not None:
                    try:
                        meta = json.loads(row["metadata"])
                    except (TypeError, ValueError):
                        meta = None
                    if isinstance(meta, dict):
                        candidate = meta.get("_pid_create_time")
                        if isinstance(candidate, (int, float)):
                            expected_create_time = float(candidate)
                if pid is None or not _pid_is_ours(pid, expected_create_time):
                    stale_ids.append(row["id"])
                    continue
                # PID alive and matches our creation timestamp — but a
                # fresh uvicorn can never re-attach to a subprocess from
                # the previous run (Popen handle is gone, the child was
                # reparented to init). Reap the orphan so we don't leak
                # GPU memory across restarts.
                if _reap_orphan(pid, expected_create_time):
                    reaped_pids.append(pid)
                    stale_ids.append(row["id"])
                else:
                    survivors.append(row["id"])
            if reaped_pids:
                log.info(
                    "reaped %d orphan training process(es) from a previous run: pids=%s",
                    len(reaped_pids),
                    reaped_pids,
                )
            if not stale_ids:
                return 0
            placeholders = ", ".join("?" * len(stale_ids))
            conn.execute(
                f"UPDATE jobs SET state = ? WHERE id IN ({placeholders})",  # noqa: S608
                (JobState.interrupted.value, *stale_ids),
            )
            if survivors:
                # Best-effort log so operators can see why some interrupted
                # jobs didn't get re-marked.
                log.info(
                    "kept %d job(s) marked running because their PID was still alive: %s",
                    len(survivors),
                    survivors,
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


def _pid_create_time(pid: int) -> float | None:
    """Read the kernel's process-start timestamp for ``pid``.

    Used to defend against PID reuse: matching the PID alone after a
    server restart isn't enough on long-running hosts (Linux's PID
    space wraps at 32k by default, AutoDL boxes routinely reuse
    in-flight pids over a long uptime). We pin each spawned training
    process to its create-time at launch and re-validate on every
    cross-process check.

    Returns None when psutil is missing (older deployments) or the
    process can't be read — callers must treat that as "can't verify"
    rather than "reuse confirmed".
    """
    try:
        import psutil  # noqa: PLC0415
    except ImportError:
        return None
    try:
        return float(psutil.Process(pid).create_time())
    except Exception:  # noqa: BLE001 — psutil-specific exceptions vary
        return None


def _pid_is_ours(pid: int, expected_create_time: float | None) -> bool:
    """Verify ``pid`` still refers to the process we originally spawned.

    Returns True only when:
      * ``_pid_alive(pid)`` agrees the kernel knows about it, AND
      * either we have no recorded create-time (legacy rows / psutil
        unavailable — fall back to the historic alive-only check), OR
        the recorded create-time matches what /proc reports within a
        small tolerance (sub-second clock drift between kernel ticks).
    """
    if not _pid_alive(pid):
        return False
    if expected_create_time is None:
        return True
    actual = _pid_create_time(pid)
    if actual is None:
        # Can't read /proc but PID is alive. Better safe than sorry —
        # treat as "ours" so we don't accidentally kill an unrelated
        # process. The orphan reaper will surface this as a survivor
        # rather than a reaped pid.
        return True
    # Allow up to 1s of drift. boot-time timestamps from /proc/<pid>/stat
    # are jiffy-quantised and can disagree with psutil's Linux
    # implementation by less than a tick.
    return abs(actual - expected_create_time) < 1.0


def _reap_orphan(pid: int, expected_create_time: float | None = None) -> bool:
    """SIGKILL ``pid`` and its process group. Returns True on success.

    Used by :meth:`JobStore.mark_orphans_interrupted` at uvicorn startup
    to release GPU memory held by training subprocesses that outlived the
    previous API process. We send SIGTERM first for a brief grace window
    (~3s) so PyTorch / CUDA contexts can flush, then SIGKILL the entire
    process group.

    ``expected_create_time`` lets us refuse to kill a PID that's
    almost-certainly been reused — without that guard, a long-uptime
    box where the kernel cycled through PIDs would risk the orphan
    reaper murdering an unrelated user process. Passing None disables
    the check (used by tests / callers that don't have the stamp).
    """
    if expected_create_time is not None and not _pid_is_ours(
        pid, expected_create_time
    ):
        # PID belongs to a different process now; do not kill.
        return False
    import os  # noqa: PLC0415
    import signal  # noqa: PLC0415
    import sys  # noqa: PLC0415
    import time  # noqa: PLC0415

    if sys.platform == "win32":
        # Windows: use taskkill /T to walk the process tree.
        import subprocess  # noqa: PLC0415

        try:
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                capture_output=True,
                check=False,
                timeout=10,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
        return not _pid_alive(pid)

    # POSIX: prefer the process group so accelerate / deepspeed children
    # come along for the ride. Falls back to per-PID kills when getpgid
    # races a final exit.
    try:
        pgid = os.getpgid(pid)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False

    for sig, wait_s in ((signal.SIGTERM, 3.0), (signal.SIGKILL, 1.0)):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return True
        except PermissionError:
            # Different uid — shouldn't happen for our own children, but
            # bail out rather than spamming a kill loop.
            return False
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            if not _pid_alive(pid):
                return True
            time.sleep(0.1)

    return not _pid_alive(pid)


_LIVE_STATES: tuple[JobState, ...] = (
    JobState.preparing,
    JobState.running,
    JobState.canceling,
)


def _record_to_row(r: JobRecord) -> dict[str, Any]:
    # `pid_create_time` rides along inside the metadata blob so we don't
    # have to add (and migrate) a new column. Round-trips back via
    # ``_row_to_record``. Stored under a reserved key with a leading
    # underscore to mark it as system-owned (callers get to use the rest
    # of metadata freely; sweep_id, axis_values, etc.).
    metadata = dict(r.metadata) if r.metadata is not None else None
    if r.pid_create_time is not None:
        metadata = metadata or {}
        metadata["_pid_create_time"] = r.pid_create_time
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
            json.dumps(metadata, ensure_ascii=False) if metadata is not None else None
        ),
    }


def _row_to_record(row: sqlite3.Row) -> JobRecord:
    # Legacy rows (pre-metadata migration) won't have the column even after
    # the ALTER; sqlite3.Row.keys() reflects the SELECT *, so guard the lookup.
    metadata: dict[str, Any] | None = None
    pid_create_time: float | None = None
    if "metadata" in row.keys() and row["metadata"] is not None:
        raw_meta = json.loads(row["metadata"])
        if isinstance(raw_meta, dict):
            # Strip the system key out of user-visible metadata.
            pid_create_time = raw_meta.pop("_pid_create_time", None)
            metadata = raw_meta or None
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
        pid_create_time=(
            float(pid_create_time) if isinstance(pid_create_time, (int, float)) else None
        ),
        metadata=metadata,
    )


def _parse_dt(s: str) -> datetime:
    return datetime.fromisoformat(s)


def _parse_dt_optional(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s) if s else None


def default_store_path() -> Path:
    """Where lorahub keeps the jobs DB.

    Always ``<project_root>/runs/jobs.sqlite``, anchored on the resolved
    project root rather than ``Path.cwd()``. The cwd-anchored layout was
    the historical source of "training history disappears after a
    restart" reports — every uvicorn restart from a different cwd would
    land on a fresh empty SQLite file. See ``lorahub.api.paths`` for the
    resolution rules.
    """
    from lorahub.api.paths import runs_dir  # noqa: PLC0415

    return runs_dir() / "jobs.sqlite"


__all__ = ["JobStore", "default_store_path"]
