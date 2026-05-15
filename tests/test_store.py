"""Tests for the SQLite-backed job store + registry persistence integration."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from lorahub.api import state
from lorahub.api.store import JobStore, default_store_path


def _job(record_id: str = "j1", state_: state.JobState = state.JobState.queued) -> state.JobRecord:
    return state.JobRecord(
        id=record_id,
        state=state_,
        workspace=Path("/tmp/ws") / record_id,
        recipe_snapshot={"base_model": {"checkpoint": "/m.safetensors"}},
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
    assert out.recipe_snapshot == rec.recipe_snapshot


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

    job = reg.create(workspace=tmp_path / "ws", recipe_snapshot={"x": 1})
    assert s.get(job.id) is not None


def test_registry_persists_updates(tmp_path: Path) -> None:
    s = JobStore(tmp_path / "x.sqlite")
    reg = state.JobRegistry(store=s)
    job = reg.create(workspace=tmp_path / "ws", recipe_snapshot={"x": 1})

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
    job = reg.create(workspace=tmp_path / "ws", recipe_snapshot={})
    assert reg.get(job.id) is job
    assert reg.load_persisted() == 0


def test_default_store_path_under_runs() -> None:
    p = default_store_path()
    assert p.name == ".lorahub.sqlite"
    assert p.parent.name == "runs"
