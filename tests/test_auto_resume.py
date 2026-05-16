"""Auto-resume hook (`_attempt_auto_resume`) tests.

The hook is invoked from `app._lifespan` after `mark_orphans_interrupted`
flips dead-PID jobs to `interrupted`. We test it in isolation by calling
it directly with a hand-rolled registry — no scheduler workers, no real
backend launch.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from lorahub.api import jobs_helpers
from lorahub.api import state
from lorahub.api.jobs_helpers import _attempt_auto_resume


def _kohya_snapshot(tmp_path: Path) -> dict[str, Any]:
    ckpt = tmp_path / "model.safetensors"
    ckpt.write_bytes(b"")
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    return {
        "base_model": {"checkpoint": str(ckpt)},
        "dataset": {"source": str(data)},
        "schedule": {"epochs": 1, "batch_size": 1},
        "sampling": {"enabled": False},
        "backend": {"type": "kohya"},
    }


def _seed_kohya_artifacts(workspace: Path) -> None:
    """Drop the bare minimum kohya checkpoint shape the resume helper looks for."""
    out = workspace / "out"
    out.mkdir(parents=True)
    (out / "lora-state").mkdir()
    (out / "lora.safetensors").write_bytes(b"")


@pytest.fixture(autouse=True)
def fresh_registry(monkeypatch: pytest.MonkeyPatch) -> Iterator[state.JobRegistry]:
    fresh = state.JobRegistry()
    monkeypatch.setattr(state, "registry", fresh)
    yield fresh


@pytest.fixture
def stub_enqueue(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace `_enqueue_launch` with a recorder so no real scheduler runs."""
    captured: list[dict[str, Any]] = []

    def stub(job, cfg, *, extra_argv=None):  # type: ignore[no-untyped-def]
        captured.append(
            {
                "job_id": job.id,
                "extra_argv": list(extra_argv or []),
                "metadata": dict(job.metadata or {}),
                "workspace": str(job.workspace),
            }
        )

    monkeypatch.setattr(jobs_helpers, "_enqueue_launch", stub)
    return captured


def test_auto_resume_disabled_by_default_skips_everything(
    tmp_path: Path, stub_enqueue: list[dict[str, Any]]
) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    _seed_kohya_artifacts(ws)
    job = state.registry.create(workspace=ws, config_snapshot=_kohya_snapshot(tmp_path))
    job.state = state.JobState.interrupted
    state.registry.update(job)

    resumed = _attempt_auto_resume(max_attempts=3, global_default=False)
    assert resumed == 0
    assert stub_enqueue == []


def test_auto_resume_relaunches_in_place_with_metadata(
    tmp_path: Path, stub_enqueue: list[dict[str, Any]]
) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    _seed_kohya_artifacts(ws)
    original = state.registry.create(
        workspace=ws, config_snapshot=_kohya_snapshot(tmp_path)
    )
    original.state = state.JobState.interrupted
    state.registry.update(original)

    resumed = _attempt_auto_resume(max_attempts=3, global_default=True)
    assert resumed == 1
    assert len(stub_enqueue) == 1
    rec = stub_enqueue[0]
    # Same id — in-place relaunch.
    assert rec["job_id"] == original.id
    assert rec["metadata"]["auto_resume"] is True
    assert rec["metadata"]["auto_resume_attempts"] == 1
    assert "last_resumed_at" in rec["metadata"]
    assert any(a.startswith("--resume=") for a in rec["extra_argv"])
    assert any(a.startswith("--network_weights=") for a in rec["extra_argv"])
    # And the registry now sees the original record back in queued state.
    refreshed = state.registry.get(original.id)
    assert refreshed is not None
    assert refreshed.state is state.JobState.queued


def test_auto_resume_skips_jobs_without_artifacts(
    tmp_path: Path, stub_enqueue: list[dict[str, Any]]
) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    # NO artifacts seeded — resume must skip silently.
    job = state.registry.create(workspace=ws, config_snapshot=_kohya_snapshot(tmp_path))
    job.state = state.JobState.interrupted
    state.registry.update(job)

    resumed = _attempt_auto_resume(max_attempts=3, global_default=True)
    assert resumed == 0
    assert stub_enqueue == []


def test_auto_resume_skips_sweep_children_even_when_global_on(
    tmp_path: Path, stub_enqueue: list[dict[str, Any]]
) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    _seed_kohya_artifacts(ws)
    job = state.registry.create(workspace=ws, config_snapshot=_kohya_snapshot(tmp_path))
    job.state = state.JobState.interrupted
    job.metadata = {"sweep_id": "01XYZ", "axis_values": {"network.rank": 16}}
    state.registry.update(job)

    resumed = _attempt_auto_resume(max_attempts=3, global_default=True)
    assert resumed == 0
    assert stub_enqueue == []


def test_auto_resume_respects_attempt_cap(
    tmp_path: Path, stub_enqueue: list[dict[str, Any]]
) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    _seed_kohya_artifacts(ws)
    job = state.registry.create(workspace=ws, config_snapshot=_kohya_snapshot(tmp_path))
    job.state = state.JobState.interrupted
    job.metadata = {"auto_resume_attempts": 3}
    state.registry.update(job)

    resumed = _attempt_auto_resume(max_attempts=3, global_default=True)
    assert resumed == 0
    assert stub_enqueue == []


def test_auto_resume_per_job_opt_in_overrides_global_off(
    tmp_path: Path, stub_enqueue: list[dict[str, Any]]
) -> None:
    """metadata.auto_resume = True forces resume even when settings flag is off."""
    ws = tmp_path / "ws"
    ws.mkdir()
    _seed_kohya_artifacts(ws)
    job = state.registry.create(workspace=ws, config_snapshot=_kohya_snapshot(tmp_path))
    job.state = state.JobState.interrupted
    job.metadata = {"auto_resume": True}
    state.registry.update(job)

    resumed = _attempt_auto_resume(max_attempts=3, global_default=False)
    assert resumed == 1
    assert len(stub_enqueue) == 1


def test_auto_resume_per_job_opt_out_overrides_global_on(
    tmp_path: Path, stub_enqueue: list[dict[str, Any]]
) -> None:
    """metadata.auto_resume = False blocks resume even when global flag is on."""
    ws = tmp_path / "ws"
    ws.mkdir()
    _seed_kohya_artifacts(ws)
    job = state.registry.create(workspace=ws, config_snapshot=_kohya_snapshot(tmp_path))
    job.state = state.JobState.interrupted
    job.metadata = {"auto_resume": False}
    state.registry.update(job)

    resumed = _attempt_auto_resume(max_attempts=3, global_default=True)
    assert resumed == 0
    assert stub_enqueue == []


def test_auto_resume_handles_stale_config_snapshot(
    tmp_path: Path, stub_enqueue: list[dict[str, Any]]
) -> None:
    """A snapshot that no longer validates must be skipped, not crash."""
    ws = tmp_path / "ws"
    ws.mkdir()
    _seed_kohya_artifacts(ws)
    job = state.registry.create(
        workspace=ws,
        config_snapshot={"this": "is", "not": "a recipe"},
    )
    job.state = state.JobState.interrupted
    state.registry.update(job)

    resumed = _attempt_auto_resume(max_attempts=3, global_default=True)
    assert resumed == 0
    assert stub_enqueue == []
