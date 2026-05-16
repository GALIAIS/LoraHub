"""Tests for the SQLite-backed job store + registry persistence integration."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from lorahub.api import state
from lorahub.api.store import JobStore, default_store_path


def _job(record_id: str = "j1", state_: state.JobState = state.JobState.queued) -> state.JobRecord:
    return state.JobRecord(
        id=record_id,
        state=state_,
        workspace=Path("/tmp/ws") / record_id,
        config_snapshot={"base_model": {"checkpoint": "/m.safetensors"}},
        created_at=datetime(2026, 5, 15, 12, 0, tzinfo=UTC),
    )


def test_store_creates_db_and_round_trips_record(tmp_path: Path) -> None:
    s = JobStore(tmp_path / "lorahub.sqlite")
    assert s.path.exists()

    rec = _job("a")
    rec.pid = 1234
    rec.started_at = datetime(2026, 5, 15, 12, 0, 5, tzinfo=UTC)
    s.upsert(rec)

    out = s.get("a")
    assert out is not None
    assert out.id == "a"
    assert out.state is state.JobState.queued
    assert out.pid == 1234
    assert out.started_at == rec.started_at
    assert out.config_snapshot == rec.config_snapshot


def test_store_upsert_updates_state(tmp_path: Path) -> None:
    s = JobStore(tmp_path / "x.sqlite")
    rec = _job("a")
    s.upsert(rec)

    rec.state = state.JobState.running
    rec.pid = 5555
    s.upsert(rec)

    out = s.get("a")
    assert out is not None
    assert out.state is state.JobState.running
    assert out.pid == 5555


def test_store_list_orders_by_created_at(tmp_path: Path) -> None:
    s = JobStore(tmp_path / "x.sqlite")
    earlier = _job("a")
    later = _job("b")
    later.created_at = datetime(2026, 5, 16, 12, 0, tzinfo=UTC)
    s.upsert(later)
    s.upsert(earlier)
    listed = s.list()
    assert [r.id for r in listed] == ["a", "b"]


def test_mark_orphans_interrupted_only_touches_live_states(tmp_path: Path) -> None:
    s = JobStore(tmp_path / "x.sqlite")
    s.upsert(_job("queued", state.JobState.queued))
    s.upsert(_job("running", state.JobState.running))
    s.upsert(_job("canceling", state.JobState.canceling))
    s.upsert(_job("succeeded", state.JobState.succeeded))
    s.upsert(_job("failed", state.JobState.failed))

    affected = s.mark_orphans_interrupted()
    assert affected == 3

    states = {r.id: r.state for r in s.list()}
    assert states["queued"] is state.JobState.interrupted
    assert states["running"] is state.JobState.interrupted
    assert states["canceling"] is state.JobState.interrupted
    assert states["succeeded"] is state.JobState.succeeded
    assert states["failed"] is state.JobState.failed


def test_registry_persists_creates(tmp_path: Path) -> None:
    s = JobStore(tmp_path / "x.sqlite")
    reg = state.JobRegistry(store=s)

    job = reg.create(workspace=tmp_path / "ws", config_snapshot={"x": 1})
    assert s.get(job.id) is not None


def test_registry_persists_updates(tmp_path: Path) -> None:
    s = JobStore(tmp_path / "x.sqlite")
    reg = state.JobRegistry(store=s)
    job = reg.create(workspace=tmp_path / "ws", config_snapshot={"x": 1})

    job.state = state.JobState.running
    job.pid = 7890
    reg.update(job)

    persisted = s.get(job.id)
    assert persisted is not None
    assert persisted.state is state.JobState.running
    assert persisted.pid == 7890


def test_registry_load_persisted_rehydrates(tmp_path: Path) -> None:
    s = JobStore(tmp_path / "x.sqlite")
    s.upsert(_job("a", state.JobState.succeeded))
    s.upsert(_job("b", state.JobState.failed))

    reg = state.JobRegistry(store=s)
    loaded = reg.load_persisted()
    assert loaded == 2
    assert {j.id for j in reg.list()} == {"a", "b"}


def test_registry_works_without_store(tmp_path: Path) -> None:
    reg = state.JobRegistry()
    job = reg.create(workspace=tmp_path / "ws", config_snapshot={})
    assert reg.get(job.id) is job
    assert reg.load_persisted() == 0


def test_default_store_path_under_runs() -> None:
    p = default_store_path()
    assert p.name == ".lorahub.sqlite"
    assert p.parent.name == "runs"


def test_save_and_load_job_with_metadata(tmp_path: Path) -> None:
    """Sweep metadata round-trips through SQLite serialization."""
    db = tmp_path / "sweep.sqlite"
    rec = _job("sweep-job")
    rec.metadata = {
        "sweep_id": "abc",
        "axis_values": {"network.rank": 16},
    }

    s = JobStore(db)
    s.upsert(rec)

    # Reopen to prove the value comes off disk, not in-memory.
    s2 = JobStore(db)
    out = s2.get("sweep-job")
    assert out is not None
    assert out.metadata == {
        "sweep_id": "abc",
        "axis_values": {"network.rank": 16},
    }


def test_save_job_without_metadata_is_none(tmp_path: Path) -> None:
    """No metadata set on the record stores SQL NULL and loads back as None."""
    db = tmp_path / "nometa.sqlite"
    s = JobStore(db)
    s.upsert(_job("plain"))

    out = JobStore(db).get("plain")
    assert out is not None
    assert out.metadata is None


def test_legacy_db_without_metadata_column_migrates(tmp_path: Path) -> None:
    """An older DB created before metadata existed must migrate transparently."""
    db = tmp_path / "legacy.sqlite"
    # Hand-create a pre-metadata schema and seed one row directly. This mirrors
    # what users upgrading from a v0.4 install will hit on next launch.
    raw = sqlite3.connect(str(db))
    raw.executescript(
        """
        CREATE TABLE jobs (
            id              TEXT PRIMARY KEY,
            state           TEXT NOT NULL,
            workspace       TEXT NOT NULL,
            config_snapshot TEXT NOT NULL,
            created_at      TEXT NOT NULL,
            started_at      TEXT,
            finished_at     TEXT,
            returncode      INTEGER,
            error           TEXT,
            pid             INTEGER
        );
        """
    )
    raw.execute(
        """
        INSERT INTO jobs (id, state, workspace, config_snapshot, created_at,
                          started_at, finished_at, returncode, error, pid)
        VALUES ('legacy-1', 'succeeded', '/tmp/ws/legacy-1',
                '{"base_model": {"checkpoint": "/m.safetensors"}}',
                '2026-05-15T12:00:00+00:00',
                NULL, NULL, NULL, NULL, NULL)
        """
    )
    raw.commit()
    raw.close()

    # Opening JobStore must trigger the ADD COLUMN migration without erroring,
    # and the legacy row must continue to load with metadata=None.
    s = JobStore(db)
    out = s.get("legacy-1")
    assert out is not None
    assert out.metadata is None
    assert out.state is state.JobState.succeeded

    # Fresh writes that include metadata work post-migration.
    new_rec = _job("fresh")
    new_rec.metadata = {"sweep_id": "post-migrate"}
    s.upsert(new_rec)
    assert s.get("fresh").metadata == {"sweep_id": "post-migrate"}

    # Re-opening the migrated DB is a no-op (idempotent ALTER guard).
    JobStore(db)
    cols = sqlite3.connect(str(db)).execute("PRAGMA table_info(jobs)").fetchall()
    metadata_cols = [c for c in cols if c[1] == "metadata"]
    assert len(metadata_cols) == 1
