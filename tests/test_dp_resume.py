"""Resume round-trip tests for the diffusion-pipe backend.

The unit-level tests on `_dp_resume_spec` lock down the contract that
the pull-up after `git tag v0.3.0` accidentally drifted on:

  1. We need both a `latest` text file AND at least one `global_step*/`
     dir for resume to be considered "ready". Either one missing is a
     409 from the `/jobs/{id}/resume` route.
  2. When multiple timestamped run_dirs exist under output_dir,
     `_find_latest_dp_run_dir` must pick the alphabetically-last one,
     mirroring dp's own `train.get_most_recent_run_dir` selection.
  3. `_dp_resume_spec` emits `--resume_from_checkpoint=<run_dir.name>`
     and pins `output.output_dir` back to the original absolute path
     so a re-launch lands in the same run_dir.

These deliberately don't shell out to dp / DeepSpeed — they're a guard
against schema or path-resolution drift, run in <1s.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from lorahub.api import jobs_helpers, state
from lorahub.api.jobs_helpers import (
    ResumeNotReady,
    _dispatch_resume_spec,
    _dp_output_dir,
    _dp_resume_spec,
    _find_latest_dp_run_dir,
)
from lorahub.core.config.schema import TrainingConfig


def _dp_cfg(
    workspace: Path, *, explicit_output_dir: Path | None = None
) -> TrainingConfig:
    """Build a minimum-viable TrainingConfig pinned to the dp backend.

    `dp_resume_spec` only reads `cfg.output.output_dir` and
    `cfg.backend.type`, so we keep the rest of the cfg as bare as
    pydantic will accept.
    """
    ckpt = workspace / "model.safetensors"
    ckpt.write_bytes(b"")
    data = workspace / "data"
    data.mkdir(exist_ok=True)
    payload = {
        "base_model": {
            "arch": "anima",
            "checkpoint": str(ckpt),
        },
        "dataset": {"source": str(data)},
        "schedule": {"epochs": 1, "batch_size": 1, "grad_accum": 1},
        "sampling": {"enabled": False},
        "output": {
            "name": "lora_output",
            "output_dir": str(explicit_output_dir) if explicit_output_dir else None,
        },
        "backend": {
            "type": "diffusion-pipe",
            "diffusion_pipe": {},
        },
    }
    return TrainingConfig.model_validate(payload)


def _drop_dp_run(
    output_dir: Path,
    run_name: str,
    *,
    global_step: int | None,
    write_latest: bool = True,
) -> Path:
    """Create the on-disk shape dp's saver leaves behind after a checkpoint.

    `global_step=None` means "no global_step* folder yet" — useful for
    asserting the resume helper rejects that case.
    """
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True)
    if global_step is not None:
        gs_dir = run_dir / f"global_step{global_step}"
        gs_dir.mkdir()
        # DeepSpeed writes optimizer state shards here; we only need the
        # directory to exist. Drop one zero-byte file so it isn't empty
        # in case a future check counts contents.
        (gs_dir / "mp_rank_00_model_states.pt").write_bytes(b"")
        # Mirror dp's `step{N}` LoRA-only folder right next to it.
        adapter_dir = run_dir / f"step{global_step}"
        adapter_dir.mkdir()
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"")
    if write_latest:
        (run_dir / "latest").write_text(
            f"global_step{global_step}" if global_step is not None else "",
            encoding="utf-8",
        )
    return run_dir


@pytest.fixture(autouse=True)
def fresh_registry(monkeypatch: pytest.MonkeyPatch) -> Iterator[state.JobRegistry]:
    fresh = state.JobRegistry()
    monkeypatch.setattr(state, "registry", fresh)
    yield fresh


# --------------------------------------------------------------------------- #
# _dp_output_dir
# --------------------------------------------------------------------------- #


def test_dp_output_dir_defaults_to_workspace_output(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    cfg = _dp_cfg(workspace)
    assert _dp_output_dir(workspace, cfg) == (workspace / "output").resolve()


def test_dp_output_dir_honours_explicit_override(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    explicit = tmp_path / "elsewhere"
    explicit.mkdir()
    cfg = _dp_cfg(workspace, explicit_output_dir=explicit)
    assert _dp_output_dir(workspace, cfg) == explicit.resolve()


# --------------------------------------------------------------------------- #
# _find_latest_dp_run_dir
# --------------------------------------------------------------------------- #


def test_find_latest_picks_alphabetically_last_complete_run(tmp_path: Path) -> None:
    """dp uses `sorted([...])[-1]`; with timestamp-named run dirs that
    happens to be the chronologically newest. We must replicate exactly
    so a resume re-enters the same run_dir as the dp launch would."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    cfg = _dp_cfg(workspace)
    out_dir = _dp_output_dir(workspace, cfg)
    out_dir.mkdir(parents=True)

    older = _drop_dp_run(out_dir, "20260101_00-00-00", global_step=400)
    newer = _drop_dp_run(out_dir, "20260518_05-37-00", global_step=4400)

    found = _find_latest_dp_run_dir(workspace, cfg)
    assert found == newer
    assert found != older


def test_find_latest_skips_runs_without_latest_marker(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    cfg = _dp_cfg(workspace)
    out_dir = _dp_output_dir(workspace, cfg)
    out_dir.mkdir(parents=True)

    # Newest by name but missing `latest` — dp couldn't resume from it,
    # so neither can we. Fall back to the older complete one.
    _drop_dp_run(
        out_dir, "20260518_06-00-00", global_step=200, write_latest=False
    )
    older_complete = _drop_dp_run(
        out_dir, "20260101_00-00-00", global_step=400
    )

    assert _find_latest_dp_run_dir(workspace, cfg) == older_complete


def test_find_latest_skips_runs_without_global_step_dir(tmp_path: Path) -> None:
    """A run with `latest` but no `global_step*/` is the shape we
    inherited before `checkpointEveryNMinutes` shipped — only LoRA
    weights got saved, no DeepSpeed state. We refuse to resume those."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    cfg = _dp_cfg(workspace)
    out_dir = _dp_output_dir(workspace, cfg)
    out_dir.mkdir(parents=True)

    # `global_step=None` skips creating `global_stepN/` entirely.
    _drop_dp_run(out_dir, "20260518_05-37-00", global_step=None)

    assert _find_latest_dp_run_dir(workspace, cfg) is None


# --------------------------------------------------------------------------- #
# _dp_resume_spec
# --------------------------------------------------------------------------- #


def test_dp_resume_spec_packs_run_basename_and_pins_output_dir(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    cfg = _dp_cfg(workspace)
    out_dir = _dp_output_dir(workspace, cfg)
    out_dir.mkdir(parents=True)
    run_dir = _drop_dp_run(out_dir, "20260518_05-37-00", global_step=4400)

    spec = _dp_resume_spec(cfg, workspace)

    # extra_argv: `--resume_from_checkpoint=<basename>`. dp resolves it
    # against the configured output_dir, so we pass only the leaf name.
    assert spec.extra_argv == [f"--resume_from_checkpoint={run_dir.name}"]
    # cfg_overrides: the resumed run must land in the same output_dir
    # the original used; serialised as a string for pydantic re-validation.
    assert spec.cfg_overrides == {"output.output_dir": str(out_dir)}
    # And critically, the override is an absolute path — dp's cwd is
    # the diffusion-pipe checkout, so a relative path would resolve
    # somewhere unexpected.
    assert Path(spec.cfg_overrides["output.output_dir"]).is_absolute()


def test_dp_resume_raises_when_output_dir_never_existed(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    cfg = _dp_cfg(workspace)
    # Don't create workspace/output at all.

    with pytest.raises(ResumeNotReady) as exc:
        _dp_resume_spec(cfg, workspace)
    assert "output_dir" in str(exc.value)


def test_dp_resume_raises_when_no_complete_run_dir(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    cfg = _dp_cfg(workspace)
    out_dir = _dp_output_dir(workspace, cfg)
    out_dir.mkdir(parents=True)
    # Same shape as a `save_every_n_steps`-only run: LoRA weights but no
    # DeepSpeed state and no `latest` pointer.
    _drop_dp_run(out_dir, "20260518_05-37-00", global_step=None, write_latest=False)

    with pytest.raises(ResumeNotReady) as exc:
        _dp_resume_spec(cfg, workspace)
    msg = str(exc.value)
    assert "global_step" in msg or "latest" in msg


# --------------------------------------------------------------------------- #
# _dispatch_resume_spec backend selection
# --------------------------------------------------------------------------- #


def test_dispatch_routes_to_dp_helper_on_diffusion_pipe_backend(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    cfg = _dp_cfg(workspace)
    out_dir = _dp_output_dir(workspace, cfg)
    out_dir.mkdir(parents=True)
    _drop_dp_run(out_dir, "20260518_05-37-00", global_step=4400)

    spec = _dispatch_resume_spec(cfg, workspace)
    assert any(a.startswith("--resume_from_checkpoint=") for a in spec.extra_argv)


def test_dispatch_propagates_resume_not_ready_unchanged(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    cfg = _dp_cfg(workspace)
    # No output dir on disk at all.
    with pytest.raises(ResumeNotReady):
        _dispatch_resume_spec(cfg, workspace)


# --------------------------------------------------------------------------- #
# Auto-resume path (mirrors test_auto_resume but for dp)
# --------------------------------------------------------------------------- #


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


def test_dp_auto_resume_relaunches_with_resume_argv(
    tmp_path: Path, stub_enqueue: list[dict]
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    cfg = _dp_cfg(workspace)
    out_dir = _dp_output_dir(workspace, cfg)
    out_dir.mkdir(parents=True)
    run = _drop_dp_run(out_dir, "20260518_05-37-00", global_step=4400)

    snapshot = cfg.model_dump(mode="json")
    job = state.registry.create(workspace=workspace, config_snapshot=snapshot)
    job.state = state.JobState.interrupted
    state.registry.update(job)

    resumed = jobs_helpers._attempt_auto_resume(max_attempts=3, global_default=True)
    assert resumed == 1
    assert len(stub_enqueue) == 1
    rec = stub_enqueue[0]
    assert rec["job_id"] == job.id
    # `--resume_from_checkpoint=<run_basename>` must reach the launcher.
    assert any(
        a == f"--resume_from_checkpoint={run.name}" for a in rec["extra_argv"]
    )
    # And the in-place job is queued back up for the scheduler.
    refreshed = state.registry.get(job.id)
    assert refreshed is not None
    assert refreshed.state is state.JobState.queued
