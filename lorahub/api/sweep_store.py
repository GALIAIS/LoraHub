"""SQLite-backed persistence for sweep metadata.

Sweeps are currently a *view* over jobs (``state.registry.list()`` filtered
by ``metadata.sweep_id``). That works at runtime but loses the sweep's
*plan* (the axes, the name template, the create timestamp) on restart —
the API can still group jobs by their stamped ``sweep_id``, but the
"sweep recipe" itself is gone.

This store keeps one row per sweep with the immutable plan as JSON.
Jobs continue to live in ``jobs.sqlite``; this DB only holds sweep
descriptors. Splitting them avoids cross-table migrations when either
schema evolves.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sweeps (
    id            TEXT PRIMARY KEY,
    name          TEXT NOT NULL,
    name_prefix   TEXT,
    plan          TEXT NOT NULL,
    base_config   TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    job_ids       TEXT
);

CREATE INDEX IF NOT EXISTS idx_sweeps_created ON sweeps(created_at DESC);
"""


@dataclass
class SweepRecord:
    """Persisted sweep descriptor.

    `plan` and `base_config` are kept as JSON-serialisable dicts so the
    sweep can be replayed (e.g. "add 5 more variants") without losing
    its origin. `job_ids` is the immutable child set spawned at create
    time — joining against the jobs table is what gives live status.
    """

    id: str
    name: str
    plan: dict[str, Any]
    base_config: dict[str, Any]
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    name_prefix: str | None = None
    job_ids: list[str] = field(default_factory=list)


class SweepStore:
    """CRUD wrapper around the sweeps table."""

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

    def upsert(self, record: SweepRecord) -> None:
        row = {
            "id": record.id,
            "name": record.name,
            "name_prefix": record.name_prefix,
            "plan": json.dumps(record.plan, ensure_ascii=False),
            "base_config": json.dumps(record.base_config, ensure_ascii=False),
            "created_at": record.created_at.isoformat(),
            "job_ids": json.dumps(record.job_ids, ensure_ascii=False),
        }
        with self._lock, self._connect() as conn:
            conn.execute(_UPSERT_SQL, row)

    def get(self, sweep_id: str) -> SweepRecord | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM sweeps WHERE id = ?", (sweep_id,)
            ).fetchone()
        return _row_to_record(row) if row else None

    def list(self) -> list[SweepRecord]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM sweeps ORDER BY created_at DESC"
            ).fetchall()
        return [_row_to_record(r) for r in rows]

    def delete(self, sweep_id: str) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute("DELETE FROM sweeps WHERE id = ?", (sweep_id,))
            return cur.rowcount > 0


_UPSERT_SQL = """
INSERT INTO sweeps (id, name, name_prefix, plan, base_config, created_at, job_ids)
VALUES (:id, :name, :name_prefix, :plan, :base_config, :created_at, :job_ids)
ON CONFLICT(id) DO UPDATE SET
    name        = excluded.name,
    name_prefix = excluded.name_prefix,
    plan        = excluded.plan,
    base_config = excluded.base_config,
    job_ids     = excluded.job_ids
"""


def _row_to_record(row: sqlite3.Row) -> SweepRecord:
    return SweepRecord(
        id=row["id"],
        name=row["name"],
        name_prefix=row["name_prefix"],
        plan=json.loads(row["plan"]),
        base_config=json.loads(row["base_config"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        job_ids=json.loads(row["job_ids"] or "[]"),
    )


def default_sweep_store_path() -> Path:
    """`<cwd>/runs/sweeps.sqlite`."""
    return Path.cwd() / "runs" / "sweeps.sqlite"


__all__ = ["SweepRecord", "SweepStore", "default_sweep_store_path"]
