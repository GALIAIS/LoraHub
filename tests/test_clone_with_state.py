"""Tests for `_validate_resume_target` and the clone-with-state route.

The validator is the single check between an arbitrary user-supplied
``cfg.resume.resume_from`` and a launch — if it lets a bogus path
through, the trainer subprocess crashes with an opaque message.
Locking down each backend's accepted layout shape here means that
"bad path" surfaces as a 400 on the API instead.

Coverage:

  * dp-shaped run dir is accepted; missing ``latest`` / missing
    ``global_step*`` / wrong ``output_dir`` parent are each rejected.
  * accelerate-shaped state dir is accepted for both kohya and
    anima_lora; a plain dir without ``-state`` in the name is rejected.
  * ``resume_from = None`` short-circuits — every backend's
    "no resume" path stays untouched.
  * ``GET /artifacts/{id}/states`` returns the expected shape for both
    backend types.
  * ``POST /jobs/{id}/clone-with-state`` spawns a NEW JobRecord whose
    snapshot has ``resume.resume_from`` pinned, and (for dp) auto-pins
    ``output.output_dir`` to ``state_path.parent``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lorahub.api import app, jobs_helpers, state
from lorahub.api.jobs_helpers import (
    ResumeTargetInvalid,
    _validate_resume_target,
)
from lorahub.core.config.schema import TrainingConfig


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def fresh_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[state.JobRegistry]:
    from lorahub.api import paths as api_paths

    fresh = state.JobRegistry()
    monkeypatch.setattr(state, "registry", fresh)
    monkeypatch.setattr(api_paths, "runs_dir", lambda: tmp_path)
    yield fresh


@pytest.fixture
def stub_enqueue(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    captured: list[dict] = []

    def stub(job, cfg, *, extra_argv=None):  # type: ignore[no-untyped-def]
        captured.append(
            {
                "job_id": job.id,
                "extra_argv": list(extra_argv or []),
                "metadata": dict(job.metadata or {}),
            }
        )

    monkeypatch.setattr(jobs_helpers, "_enqueue_launch", stub)
    return captured


@pytest.fixture
def client() -> TestClient:
    return TestClient(app.app)


# --------------------------------------------------------------------------- #
# Config builders
# --------------------------------------------------------------------------- #


def _seed_dataset(data: Path) -> None:
    """Preflight blocks on empty dataset dirs; drop a 1-byte trainable file."""
    (data / "stub.png").write_bytes(b"")


def _dp_cfg(workspace: Path, *, output_dir: Path | None = None) -> TrainingConfig:
    ckpt = workspace / "model.safetensors"
    ckpt.write_bytes(b"")
    data = workspace / "data"
    data.mkdir(exist_ok=True)
    _seed_dataset(data)
    payload = {
        "base_model": {"arch": "anima", "checkpoint": str(ckpt)},
        "dataset": {"source": str(data)},
        "schedule": {"epochs": 1, "batch_size": 1, "grad_accum": 1},
        "sampling": {"enabled": False},
        "output": {
            "name": "lora_output",
            "output_dir": str(output_dir) if output_dir else None,
        },
        "backend": {"type": "diffusion-pipe", "diffusion_pipe": {}},
    }
    return TrainingConfig.model_validate(payload)


def _kohya_cfg(workspace: Path) -> TrainingConfig:
    ckpt = workspace / "sdxl.safetensors"
    ckpt.write_bytes(b"")
    data = workspace / "data"
    data.mkdir(exist_ok=True)
    _seed_dataset(data)
    payload = {
        "base_model": {"arch": "sdxl", "checkpoint": str(ckpt)},
        "dataset": {"source": str(data)},
        "schedule": {"epochs": 1, "batch_size": 1, "grad_accum": 1},
        "sampling": {"enabled": False},
        "output": {"name": "lora_output"},
        "backend": {"type": "kohya"},
    }
    return TrainingConfig.model_validate(payload)


def _anima_cfg(workspace: Path) -> TrainingConfig:
    ckpt = workspace / "model.safetensors"
    ckpt.write_bytes(b"")
    data = workspace / "data"
    data.mkdir(exist_ok=True)
    _seed_dataset(data)
    payload = {
        "base_model": {"arch": "anima", "checkpoint": str(ckpt)},
        "dataset": {"source": str(data)},
        "schedule": {"epochs": 1, "batch_size": 1, "grad_accum": 1},
        "sampling": {"enabled": False},
        "output": {"name": "lora_output"},
        "backend": {"type": "anima_lora", "anima_lora": {}},
    }
    return TrainingConfig.model_validate(payload)


def _drop_state_dir(parent: Path, name: str, *, current_step: int = 100) -> Path:
    """Create the on-disk shape of an accelerate state save."""
    d = parent / name
    d.mkdir(parents=True)
    (d / "optimizer.bin").write_bytes(b"")
    (d / "scheduler.bin").write_bytes(b"")
    (d / "model.safetensors").write_bytes(b"")
    (d / "random_states_0.pkl").write_bytes(b"")
    (d / "train_state.json").write_text(
        json.dumps({"current_step": current_step, "current_epoch": 1}),
        encoding="utf-8",
    )
    return d


def _drop_dp_run(parent: Path, name: str, *, global_step: int = 4400) -> Path:
    run = parent / name
    run.mkdir(parents=True)
    gs = run / f"global_step{global_step}"
    gs.mkdir()
    (gs / "mp_rank_00_model_states.pt").write_bytes(b"")
    (run / "latest").write_text(f"global_step{global_step}", encoding="utf-8")
    return run


# --------------------------------------------------------------------------- #
# _validate_resume_target — accelerate (kohya / anima_lora)
# --------------------------------------------------------------------------- #


def test_validate_noop_when_resume_from_unset(tmp_path: Path) -> None:
    """The legacy "no resume" path must not require any disk shape."""
    cfg = _kohya_cfg(tmp_path)
    assert cfg.resume.resume_from is None
    # Should simply return without raising.
    _validate_resume_target(cfg)


def test_validate_accepts_kohya_state_dir(tmp_path: Path) -> None:
    cfg = _kohya_cfg(tmp_path)
    state_dir = _drop_state_dir(tmp_path, "lora_output-state-step100")
    cfg.resume.resume_from = state_dir
    _validate_resume_target(cfg)


def test_validate_accepts_anima_state_dir(tmp_path: Path) -> None:
    cfg = _anima_cfg(tmp_path)
    state_dir = _drop_state_dir(tmp_path, "lora_output-state")
    cfg.resume.resume_from = state_dir
    _validate_resume_target(cfg)


def test_validate_rejects_missing_path(tmp_path: Path) -> None:
    cfg = _kohya_cfg(tmp_path)
    cfg.resume.resume_from = tmp_path / "nope-state"
    with pytest.raises(ResumeTargetInvalid, match="does not exist"):
        _validate_resume_target(cfg)


def test_validate_rejects_dir_without_state_in_name(tmp_path: Path) -> None:
    cfg = _kohya_cfg(tmp_path)
    plain = tmp_path / "checkpoint"
    plain.mkdir()
    (plain / "optimizer.bin").write_bytes(b"")
    cfg.resume.resume_from = plain
    with pytest.raises(ResumeTargetInvalid, match="-state"):
        _validate_resume_target(cfg)


def test_validate_rejects_empty_state_dir(tmp_path: Path) -> None:
    cfg = _kohya_cfg(tmp_path)
    empty = tmp_path / "lora_output-state"
    empty.mkdir()
    cfg.resume.resume_from = empty
    with pytest.raises(ResumeTargetInvalid, match="empty"):
        _validate_resume_target(cfg)


def test_validate_rejects_state_with_only_linked_marker(tmp_path: Path) -> None:
    cfg = _kohya_cfg(tmp_path)
    state_dir = tmp_path / "lora_output-state"
    state_dir.mkdir()
    outside = tmp_path / "outside-optimizer.bin"
    outside.write_bytes(b"state")
    try:
        (state_dir / "optimizer.bin").symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    cfg.resume.resume_from = state_dir

    with pytest.raises(ResumeTargetInvalid, match="empty"):
        _validate_resume_target(cfg)


# --------------------------------------------------------------------------- #
# _validate_resume_target — diffusion-pipe
# --------------------------------------------------------------------------- #


def test_validate_accepts_dp_run_dir(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    run = _drop_dp_run(out_dir, "20260518_05-37-00")
    cfg = _dp_cfg(tmp_path, output_dir=out_dir)
    cfg.resume.resume_from = run
    _validate_resume_target(cfg)


def test_validate_rejects_dp_run_without_latest(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    run = out_dir / "20260518_05-37-00"
    run.mkdir()
    (run / "global_step100").mkdir()
    cfg = _dp_cfg(tmp_path, output_dir=out_dir)
    cfg.resume.resume_from = run
    with pytest.raises(ResumeTargetInvalid, match="latest"):
        _validate_resume_target(cfg)


def test_validate_rejects_dp_run_without_global_step(tmp_path: Path) -> None:
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    run = out_dir / "20260518_05-37-00"
    run.mkdir()
    (run / "latest").write_text("global_step100", encoding="utf-8")
    cfg = _dp_cfg(tmp_path, output_dir=out_dir)
    cfg.resume.resume_from = run
    with pytest.raises(ResumeTargetInvalid, match="global_step"):
        _validate_resume_target(cfg)


def test_validate_rejects_dp_when_output_dir_mismatches_run_parent(
    tmp_path: Path,
) -> None:
    """dp resolves --resume_from_checkpoint=<basename> against output_dir;
    if the parent of the run dir doesn't match, dp will look in the
    wrong place. The validator catches this before launch."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    run = _drop_dp_run(out_dir, "20260518_05-37-00")
    other = tmp_path / "elsewhere"
    other.mkdir()

    cfg = _dp_cfg(tmp_path, output_dir=other)  # wrong output_dir
    cfg.resume.resume_from = run
    with pytest.raises(ResumeTargetInvalid, match="output_dir"):
        _validate_resume_target(cfg)


def test_validate_wraps_iterdir_oserror_as_resume_target_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Permission errors on dp run dirs must surface as 400, not 500."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    run = _drop_dp_run(out_dir, "20260518_05-37-00")
    cfg = _dp_cfg(tmp_path, output_dir=out_dir)
    cfg.resume.resume_from = run

    real_iterdir = Path.iterdir

    def boom(self: Path) -> Iterator[Path]:
        if self == run:
            raise PermissionError("simulated EACCES")
        return real_iterdir(self)

    monkeypatch.setattr(Path, "iterdir", boom)
    with pytest.raises(ResumeTargetInvalid, match="cannot enumerate"):
        _validate_resume_target(cfg)


# --------------------------------------------------------------------------- #
# GET /artifacts/{id}/states
# --------------------------------------------------------------------------- #


def test_states_endpoint_returns_kohya_state_dirs(
    tmp_path: Path, client: TestClient
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _drop_state_dir(workspace, "lora_output-state-step100", current_step=100)
    _drop_state_dir(workspace, "lora_output-state-step200", current_step=200)

    cfg = _kohya_cfg(workspace)
    snap = cfg.model_dump(mode="json", by_alias=True)
    job = state.registry.create(workspace=workspace, config_snapshot=snap)

    resp = client.get(f"/api/artifacts/{job.id}/states")
    assert resp.status_code == 200
    body = resp.json()
    assert body["backend_type"] == "kohya"
    states_list = body["states"]
    assert len(states_list) == 2
    # current_step is parsed from train_state.json
    steps = sorted(s.get("current_step") for s in states_list)
    assert steps == [100, 200]


def test_states_endpoint_returns_dp_run_dirs(
    tmp_path: Path, client: TestClient
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    out_dir = workspace / "out"
    out_dir.mkdir()
    _drop_dp_run(out_dir, "20260101_00-00-00", global_step=400)
    _drop_dp_run(out_dir, "20260518_05-37-00", global_step=4400)

    cfg = _dp_cfg(workspace, output_dir=out_dir)
    snap = cfg.model_dump(mode="json", by_alias=True)
    job = state.registry.create(workspace=workspace, config_snapshot=snap)

    resp = client.get(f"/api/artifacts/{job.id}/states")
    assert resp.status_code == 200
    body = resp.json()
    assert body["backend_type"] == "diffusion-pipe"
    states_list = body["states"]
    assert len(states_list) == 2
    # Newest by basename is first.
    assert states_list[0]["basename"] == "20260518_05-37-00"
    assert states_list[0]["latest_step"] == 4400


def test_states_endpoint_404_for_unknown_job(client: TestClient) -> None:
    resp = client.get("/api/artifacts/does-not-exist/states")
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# POST /jobs/{id}/clone-with-state
# --------------------------------------------------------------------------- #


def test_clone_with_state_kohya_spawns_new_job_with_resume_pinned(
    tmp_path: Path, client: TestClient, stub_enqueue: list[dict]
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    state_dir = _drop_state_dir(workspace, "lora_output-state-step100")

    cfg = _kohya_cfg(workspace)
    snap = cfg.model_dump(mode="json", by_alias=True)
    source = state.registry.create(workspace=workspace, config_snapshot=snap)

    new_workspace = tmp_path / "new_ws"
    resp = client.post(
        f"/api/jobs/{source.id}/clone-with-state",
        json={"statePath": str(state_dir), "workspace": str(new_workspace)},
    )
    assert resp.status_code == 202, resp.text
    new_summary = resp.json()
    assert new_summary["id"] != source.id

    # New JobRecord exists, with cloned_from metadata + pinned resume_from.
    new_job = state.registry.get(new_summary["id"])
    assert new_job is not None
    assert new_job.metadata["cloned_from_job_id"] == source.id
    assert new_job.metadata["cloned_from_state"] == str(state_dir.resolve())

    snap2 = new_job.config_snapshot
    assert isinstance(snap2, dict)
    assert snap2["resume"]["resumeFrom"] == str(state_dir.resolve())


def test_clone_with_state_dp_auto_pins_output_dir_to_run_parent(
    tmp_path: Path, client: TestClient, stub_enqueue: list[dict]
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    out_dir = workspace / "out"
    out_dir.mkdir()
    run = _drop_dp_run(out_dir, "20260518_05-37-00")

    cfg = _dp_cfg(workspace, output_dir=out_dir)
    snap = cfg.model_dump(mode="json", by_alias=True)
    source = state.registry.create(workspace=workspace, config_snapshot=snap)

    new_workspace = tmp_path / "new_ws"
    resp = client.post(
        f"/api/jobs/{source.id}/clone-with-state",
        json={"statePath": str(run), "workspace": str(new_workspace)},
    )
    assert resp.status_code == 202, resp.text
    new_job = state.registry.get(resp.json()["id"])
    assert new_job is not None

    snap2 = new_job.config_snapshot
    assert isinstance(snap2, dict)
    # Output dir auto-pinned to the run dir's parent so dp can resolve
    # --resume_from_checkpoint=<basename>.
    assert snap2["output"]["outputDir"] == str(run.resolve().parent)
    assert snap2["resume"]["resumeFrom"] == str(run.resolve())


def test_clone_with_state_rejects_invalid_state_path(
    tmp_path: Path, client: TestClient
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    cfg = _kohya_cfg(workspace)
    snap = cfg.model_dump(mode="json", by_alias=True)
    source = state.registry.create(workspace=workspace, config_snapshot=snap)

    resp = client.post(
        f"/api/jobs/{source.id}/clone-with-state",
        json={"statePath": str(tmp_path / "definitely-not-here-state")},
    )
    assert resp.status_code == 400
    assert "does not exist" in resp.json()["detail"]


def test_clone_with_state_rejects_state_owned_by_another_workspace(
    tmp_path: Path, client: TestClient
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    state_dir = _drop_state_dir(unrelated, "lora_output-state-step100")
    cfg = _kohya_cfg(workspace)
    source = state.registry.create(
        workspace=workspace,
        config_snapshot=cfg.model_dump(mode="json", by_alias=True),
    )

    response = client.post(
        f"/api/jobs/{source.id}/clone-with-state",
        json={"statePath": str(state_dir)},
    )

    assert response.status_code == 400
    assert "not owned by the source job" in response.json()["detail"]


def test_clone_with_state_rejects_locked_field_change(
    tmp_path: Path, client: TestClient
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    state_dir = _drop_state_dir(workspace, "lora_output-state-step100")

    cfg = _kohya_cfg(workspace)
    snap = cfg.model_dump(mode="json", by_alias=True)
    source = state.registry.create(workspace=workspace, config_snapshot=snap)

    # Try to change network.rank — locked, must 409.
    edited = dict(snap)
    edited["network"] = {**snap.get("network", {}), "rank": 64}
    resp = client.post(
        f"/api/jobs/{source.id}/clone-with-state",
        json={"statePath": str(state_dir), "config": edited},
    )
    assert resp.status_code == 409
    assert "network.rank" in resp.json()["detail"]
