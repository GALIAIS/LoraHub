"""In-memory job registry — the v0.2 single-process state.

Each `JobRecord` carries the immutable job descriptor plus its live
progress, the training handle (so we can stop/wait), and a ring buffer of
recent `TrainingEvent`s so HTTP polling clients (and reconnecting WebSocket
clients) can catch up without a full re-stream.

Persistence and multi-process orchestration belong in v0.5 — until then this
module is the single source of truth and lives in process memory.
"""

from __future__ import annotations

import contextlib
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import ulid

from lorahub.core.backends.base import TrainingHandle
from lorahub.core.events import TrainingEvent

_DEFAULT_RING_SIZE = 1024


class JobState(StrEnum):
    queued = "queued"
    running = "running"
    canceling = "canceling"
    succeeded = "succeeded"
    failed = "failed"
    canceled = "canceled"
    interrupted = "interrupted"  # process was lost (e.g. server restart)


@dataclass(slots=True)
class JobRecord:
    id: str
    state: JobState
    workspace: Path
    recipe_snapshot: dict[str, Any]
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    returncode: int | None = None
    error: str | None = None
    pid: int | None = None
    handle: TrainingHandle | None = field(default=None, repr=False)
    events: deque[TrainingEvent] = field(default_factory=lambda: deque(maxlen=_DEFAULT_RING_SIZE))
    # Free-form metadata bag set by callers that orchestrate jobs (e.g. the
    # sweep router stamps `{"sweep_id": ..., "axis_values": {...}}` here so
    # later GETs can group jobs by their parent sweep). In-memory only —
    # the SQLite store does not currently persist this; if a server restart
    # rehydrates jobs they'll come back with metadata=None until re-tagged.
    metadata: dict[str, Any] | None = None

    def to_summary(self) -> dict[str, Any]:
        """Serializable view that the API returns. Strips the live handle."""
        return {
            "id": self.id,
            "state": self.state.value,
            "workspace": str(self.workspace),
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "returncode": self.returncode,
            "error": self.error,
            "pid": self.pid,
            "metadata": self.metadata,
        }


class JobRegistry:
    """Thread-safe in-memory job store with WS subscription fan-out.

    Optionally backed by a `JobStore` for SQLite persistence: every state
    mutation is mirrored to the store, and `load_persisted()` rehydrates the
    in-memory dict from disk on startup.
    """

    def __init__(self, store: Any | None = None) -> None:
        self._jobs: dict[str, JobRecord] = {}
        self._listeners: dict[str, list[Any]] = {}  # job_id -> list of asyncio.Queue
        self._lock = threading.RLock()
        self._store = store

    @property
    def store(self) -> Any | None:
        return self._store

    def load_persisted(self) -> int:
        """Hydrate the in-memory dict from the store. Returns rows loaded."""
        if self._store is None:
            return 0
        with self._lock:
            for record in self._store.list():
                self._jobs[record.id] = record
                self._listeners.setdefault(record.id, [])
            return len(self._jobs)

    def _persist(self, record: JobRecord) -> None:
        if self._store is None:
            return
        with contextlib.suppress(Exception):
            self._store.upsert(record)

    def create(self, workspace: Path, recipe_snapshot: dict[str, Any]) -> JobRecord:
        with self._lock:
            job = JobRecord(
                id=str(ulid.new()),
                state=JobState.queued,
                workspace=workspace,
                recipe_snapshot=recipe_snapshot,
                created_at=datetime.now(UTC),
            )
            self._jobs[job.id] = job
            self._listeners[job.id] = []
        self._persist(job)
        return job

    def update(self, job: JobRecord) -> None:
        """Persist mutations made on the live `JobRecord` (state, returncode, ...)."""
        self._persist(job)

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def delete(self, job_id: str) -> bool:
        """Drop a job from memory and the backing store. Returns True if removed."""
        with self._lock:
            removed = self._jobs.pop(job_id, None) is not None
            self._listeners.pop(job_id, None)
        if self._store is not None:
            with contextlib.suppress(Exception):
                self._store.delete(job_id)
        return removed

    def list(self) -> list[JobRecord]:
        with self._lock:
            return list(self._jobs.values())

    def record_event(self, job_id: str, event: TrainingEvent) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.events.append(event)
            listeners = list(self._listeners.get(job_id, ()))
        for q in listeners:
            with contextlib.suppress(Exception):
                q.put_nowait(event)

    def attach_listener(self, job_id: str, queue: Any) -> None:
        with self._lock:
            self._listeners.setdefault(job_id, []).append(queue)

    def detach_listener(self, job_id: str, queue: Any) -> None:
        with self._lock:
            lst = self._listeners.get(job_id, [])
            if queue in lst:
                lst.remove(queue)


registry = JobRegistry()
