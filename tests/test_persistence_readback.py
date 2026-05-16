"""Cross-restart persistence smoke tests.

These exercise the read-back paths the SQLite persistence audit flagged
as gaps: SweepStore listing, SessionStore fallbacks (tagging / captions
/ bootstrap), and the queued-job requeue hook.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from lorahub.api import app as app_module
from lorahub.api import scheduler as sched_module
from lorahub.api import state as state_module
from lorahub.api.session_store import SessionStore
from lorahub.api.sweep_store import SweepRecord, SweepStore


@pytest.fixture
def isolated_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Spin up a fresh registry + scheduler + per-test stores."""
    registry = state_module.JobRegistry()
    monkeypatch.setattr(state_module, "registry", registry)

    fresh_sched = sched_module.JobScheduler(concurrency=1)
    monkeypatch.setattr(sched_module, "scheduler", fresh_sched)

    sweep_store = SweepStore(tmp_path / "sweeps.sqlite")
    session_store = SessionStore(tmp_path / "sessions.sqlite")
    monkeypatch.setattr(app_module, "_sweep_store", sweep_store)
    monkeypatch.setattr(app_module, "_session_store", session_store)

    # Module-level session dicts are global; pytest reuses them across
    # tests, so clear them per-fixture to avoid bleed-through.
    from lorahub.api.routers import captions as _cap  # noqa: PLC0415
    from lorahub.api.routers import tagging as _tag  # noqa: PLC0415

    monkeypatch.setattr(_cap, "_sessions", {})
    monkeypatch.setattr(_tag, "_sessions", {})
    monkeypatch.setattr(_tag, "_anima_sessions", {})

    return {
        "registry": registry,
        "sweep_store": sweep_store,
        "session_store": session_store,
    }


def test_list_sweeps_includes_store_only_archived_entries(
    isolated_app: dict[str, Any],
) -> None:
    """A sweep whose every child has been archived still surfaces."""
    isolated_app["sweep_store"].upsert(
        SweepRecord(
            id="01ARCH",
            name="archived-sweep",
            name_prefix="rank",
            plan={"axes": [{"path": "network.rank", "values": [8]}]},
            base_config={"output": {"name": "rank-000"}},
            job_ids=["jobX"],
        )
    )

    with TestClient(app_module.app) as client:
        r = client.get("/api/sweeps")
    assert r.status_code == 200
    sweeps = r.json()["sweeps"]
    matching = [s for s in sweeps if s["sweep_id"] == "01ARCH"]
    assert len(matching) == 1
    assert matching[0].get("archived") is True


def test_get_sweep_falls_back_to_store_when_no_jobs_remain(
    isolated_app: dict[str, Any],
) -> None:
    isolated_app["sweep_store"].upsert(
        SweepRecord(
            id="01STORE",
            name="store-only",
            plan={"axes": []},
            base_config={"foo": "bar"},
            job_ids=[],
        )
    )
    with TestClient(app_module.app) as client:
        r = client.get("/api/sweeps/01STORE")
    assert r.status_code == 200
    body = r.json()
    assert body["sweep_id"] == "01STORE"
    assert body["plan"] == {"axes": []}
    assert body["name"] == "store-only"


def test_tagging_status_falls_back_to_session_store(
    isolated_app: dict[str, Any],
) -> None:
    isolated_app["session_store"].upsert_tagging(
        {
            "session_id": "tag1",
            "path": "/d",
            "status": "succeeded",
            "started_at": 1.0,
            "finished_at": 2.0,
            "written": 5,
            "device": "cuda",
        }
    )
    with TestClient(app_module.app) as client:
        r = client.get("/api/tagging/tag/tag1")
    assert r.status_code == 200
    assert r.json()["session_id"] == "tag1"
    assert r.json()["status"] == "succeeded"


def test_tagging_list_returns_persisted_sessions(
    isolated_app: dict[str, Any],
) -> None:
    for sid, ts in [("a", 1.0), ("b", 3.0), ("c", 2.0)]:
        isolated_app["session_store"].upsert_tagging(
            {
                "session_id": sid,
                "path": "/d",
                "status": "succeeded",
                "started_at": ts,
            }
        )
    with TestClient(app_module.app) as client:
        r = client.get("/api/tagging/tag")
    assert r.status_code == 200
    sids = [s["session_id"] for s in r.json()["sessions"]]
    assert sids == ["b", "c", "a"]


def test_captions_status_falls_back_to_session_store(
    isolated_app: dict[str, Any],
) -> None:
    isolated_app["session_store"].upsert_captions(
        {
            "session_id": "cap1",
            "path": "/d",
            "status": "succeeded",
            "started_at": 4.0,
            "finished_at": 5.0,
            "written": 7,
        }
    )
    with TestClient(app_module.app) as client:
        r = client.get("/api/captions/normalize/cap1")
    assert r.status_code == 200
    assert r.json()["session_id"] == "cap1"


def test_bootstrap_status_surfaces_persisted_session_when_idle(
    isolated_app: dict[str, Any],
) -> None:
    isolated_app["session_store"].upsert_bootstrap(
        {
            "session_id": "boot1",
            "backend": "kohya",
            "status": "succeeded",
            "started_at": 100.0,
            "finished_at": 200.0,
            "events": [],
        }
    )
    with TestClient(app_module.app) as client:
        r = client.get("/api/backend/bootstrap/status")
    assert r.status_code == 200
    body = r.json()
    assert body["session_id"] == "boot1"
    assert body["status"] == "succeeded"


def test_requeue_pending_resubmits_queued_jobs(
    isolated_app: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A queued JobRecord persisted from a prior run gets re-submitted."""
    from lorahub.api import jobs_helpers
    from lorahub.api.state import JobState

    captured: list[str] = []

    def stub_enqueue(job, cfg, *, extra_argv=None):  # type: ignore[no-untyped-def]
        captured.append(job.id)

    monkeypatch.setattr(jobs_helpers, "_enqueue_launch", stub_enqueue)

    ckpt = tmp_path / "model.safetensors"
    ckpt.write_bytes(b"")
    data = tmp_path / "data"
    data.mkdir()
    snapshot: dict[str, Any] = {
        "base_model": {"checkpoint": str(ckpt)},
        "dataset": {"source": str(data)},
        "schedule": {"epochs": 1, "batch_size": 1},
        "sampling": {"enabled": False},
        "backend": {"type": "kohya"},
    }
    rec = isolated_app["registry"].create(
        workspace=tmp_path / "ws", config_snapshot=snapshot
    )
    assert rec.state is JobState.queued

    requeued = jobs_helpers._requeue_pending_jobs()
    assert requeued == 1
    assert captured == [rec.id]


def test_requeue_pending_marks_stale_snapshots_failed(
    isolated_app: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A queued job whose snapshot no longer validates flips to failed."""
    from lorahub.api import jobs_helpers
    from lorahub.api.state import JobState

    monkeypatch.setattr(
        jobs_helpers,
        "_enqueue_launch",
        lambda job, cfg, *, extra_argv=None: None,
    )

    rec = isolated_app["registry"].create(
        workspace=tmp_path / "ws",
        config_snapshot={"this": "is", "stale": True},
    )

    requeued = jobs_helpers._requeue_pending_jobs()
    assert requeued == 0
    refreshed = isolated_app["registry"].get(rec.id)
    assert refreshed is not None
    assert refreshed.state is JobState.failed
    assert refreshed.error and "stale" in refreshed.error
