"""Smoke tests for the split persistence stores."""

from __future__ import annotations

from pathlib import Path

from lorahub.api.session_store import SessionStore
from lorahub.api.sweep_store import SweepRecord, SweepStore


def test_sweep_store_upsert_and_list_round_trip(tmp_path: Path) -> None:
    store = SweepStore(tmp_path / "sweeps.sqlite")
    record = SweepRecord(
        id="01ABC",
        name="my-sweep",
        plan={"axes": [{"path": "network.rank", "values": [8, 16, 32]}]},
        base_config={"output": {"name": "x"}},
        job_ids=["job1", "job2", "job3"],
    )
    store.upsert(record)

    listed = store.list()
    assert len(listed) == 1
    assert listed[0].id == "01ABC"
    assert listed[0].plan["axes"][0]["values"] == [8, 16, 32]
    assert listed[0].job_ids == ["job1", "job2", "job3"]

    fetched = store.get("01ABC")
    assert fetched is not None
    assert fetched.name == "my-sweep"


def test_sweep_store_upsert_overwrites_in_place(tmp_path: Path) -> None:
    store = SweepStore(tmp_path / "sweeps.sqlite")
    rec = SweepRecord(id="X", name="v1", plan={}, base_config={}, job_ids=[])
    store.upsert(rec)
    rec2 = SweepRecord(
        id="X", name="v2", plan={"axes": []}, base_config={"output": {"name": "y"}},
        job_ids=["a"],
    )
    store.upsert(rec2)
    fetched = store.get("X")
    assert fetched is not None
    assert fetched.name == "v2"
    assert fetched.job_ids == ["a"]


def test_session_store_isolates_kinds(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.sqlite")
    store.upsert_tagging(
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
    store.upsert_bootstrap(
        {
            "session_id": "boot1",
            "backend": "kohya",
            "status": "succeeded",
            "started_at": 1.0,
            "finished_at": 9.0,
            "events": [],
        }
    )

    tagging = store.list_recent("tagging")
    bootstraps = store.list_recent("bootstrap")
    captions = store.list_recent("captions")
    assert {t["session_id"] for t in tagging} == {"tag1"}
    assert {b["session_id"] for b in bootstraps} == {"boot1"}
    assert captions == []  # nothing persisted into that table


def test_session_store_list_recent_orders_by_started_at_desc(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.sqlite")
    for i, ts in enumerate([3.0, 1.0, 2.0]):
        store.upsert_tagging(
            {
                "session_id": f"s{i}",
                "path": "/d",
                "status": "succeeded",
                "started_at": ts,
            }
        )
    listed = store.list_recent("tagging")
    started_ats = [r["started_at"] for r in listed]
    assert started_ats == sorted(started_ats, reverse=True)


def test_session_store_list_recent_bounds_limit(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.sqlite")
    for i in range(105):
        store.upsert_tagging(
            {
                "session_id": f"s{i}",
                "path": "/d",
                "status": "succeeded",
                "started_at": float(i),
            }
        )

    assert len(store.list_recent("tagging", limit=-1)) == 1
    assert len(store.list_recent("tagging", limit=1000)) == 100
