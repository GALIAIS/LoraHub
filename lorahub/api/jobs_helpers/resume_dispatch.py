"""Resume artifact discovery + auto-resume / requeue / migration hooks.

Three interlocking responsibilities live here:

1. **Per-backend resume specs** — given a workspace, find the most
   recent state-dir / safetensors / dp run-dir and pack the matching
   resume argv (kohya, diffusion-pipe, anima_lora). ``ResumeNotReady``
   surfaces "no checkpoint yet" through a single exception type so
   the router and the auto-resume hook can treat it uniformly.

2. **Auto-resume** — at startup, replay every interrupted job that
   still has resumable artifacts. Run before the scheduler starts so
   resumed work lands at the head of the queue.

3. **Restart housekeeping** — re-enqueue persisted ``queued`` rows
   that survived a restart, and migrate older snake_case config
   snapshots to camelCase so the resume-with-edit form sees the same
   shape newer jobs use.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lorahub.api.state import JobState
from lorahub.core.config.schema import TrainingConfig

# NB: ``_enqueue_launch`` / ``_relaunch_job_in_place`` are imported via the
# package façade (``lorahub.api.jobs_helpers``) at call time, *not* via
# ``from .lifecycle import ...``. Tests replace these symbols on the
# package module to stub the scheduler hookup; a local rebinding here
# would freeze the original references at import time and bypass the
# patch. See tests/test_persistence_readback.py and tests/test_auto_resume.py.

log = logging.getLogger(__name__)

_RESUME_SCAN_EXCLUDE_DIRS = frozenset(
    {
        # anima_lora preprocess scratch — `_anima_te.safetensors` text
        # encoder caches live here and have very fresh mtimes, so without
        # excluding them they'd outrank the real LoRA weights when the
        # resume finder picks max(mtime).
        "post_image_dataset",
        # LoRaHub-managed dataset mirror (caption_filter sanitisation).
        "captions_sanitized",
        # archive / VCS / pycache noise.
        "_archive",
        "__pycache__",
        ".git",
        ".ipynb_checkpoints",
    },
)


def _find_latest_state_dir(workspace: Path) -> Path | None:
    """Most recently modified ``*-state*`` directory under the job workspace.

    sd-scripts writes state directories like ``<output_name>-state`` at
    the end of a run and ``<output_name>-state-step<N>`` at each
    interval. Walks the entire workspace (kohya can put output_dir
    anywhere; anima_lora pins it under ``ckpt/``) but skips
    ``post_image_dataset`` / ``captions_sanitized`` so cache artifacts
    can't poison the most-recent-mtime pick.
    """
    if not workspace.is_dir():
        return None
    candidates: list[Path] = []
    for p in workspace.rglob("*"):
        if any(part in _RESUME_SCAN_EXCLUDE_DIRS for part in p.parts):
            continue
        if p.is_dir() and "-state" in p.name:
            candidates.append(p)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _find_latest_safetensors(workspace: Path) -> Path | None:
    """Most recently modified trainer-output ``*.safetensors`` under workspace.

    Skips ``post_image_dataset`` / ``captions_sanitized`` for the same
    reason ``_find_latest_state_dir`` does — every preprocessed image
    has a ``<stem>_anima_te.safetensors`` text-encoder cache written
    after preprocess, so the freshest-mtime pick used to be one of
    those cache files instead of the actual LoRA weights. Resume then
    fed ``--network_weights=<te_cache>`` into the trainer and
    effectively started training from random init.
    """
    if not workspace.is_dir():
        return None
    candidates: list[Path] = []
    for p in workspace.rglob("*.safetensors"):
        if any(part in _RESUME_SCAN_EXCLUDE_DIRS for part in p.parts):
            continue
        if p.is_file():
            candidates.append(p)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


# --------------------------------------------------------------------------- #
# Resume helpers (backend-aware artifact discovery + argv assembly)
# --------------------------------------------------------------------------- #


class ResumeNotReady(Exception):
    """Raised when a job has no resumable artifacts on disk yet.

    Surfaced by `/jobs/{id}/resume` as 409 and by the auto-resume hook as
    a skip reason. The message is operator-facing — keep it specific.
    """


@dataclass(slots=True)
class ResumeSpec:
    """Backend-agnostic recipe for what to inject into a resume launch.

    `extra_argv` is appended after the compiler's argv (same channel kohya
    /resume already uses). `cfg_overrides` is a flat dot-path mapping the
    caller applies to the validated `TrainingConfig` before launching;
    used by dp resume to redirect `output.output_dir` at the original
    run_dir so `--resume_from_checkpoint=<basename>` resolves.
    """

    extra_argv: list[str]
    cfg_overrides: dict[str, Any] = field(default_factory=dict)


def _kohya_resume_spec(workspace: Path) -> ResumeSpec:
    """Locate kohya `--save_state` artifacts and pack them into a ResumeSpec."""
    state_dir = _find_latest_state_dir(workspace)
    if state_dir is None:
        raise ResumeNotReady(
            f"no kohya state directory found under {workspace}; "
            "resume requires --save_state to have produced at least one snapshot"
        )
    weights = _find_latest_safetensors(workspace)
    if weights is None:
        raise ResumeNotReady(
            f"no .safetensors weights found under {workspace}; "
            "cannot seed --network_weights"
        )
    return ResumeSpec(
        extra_argv=[
            f"--resume={state_dir}",
            f"--network_weights={weights}",
        ],
    )


def _dp_output_dir(workspace: Path, cfg: TrainingConfig) -> Path:
    """Mirror compiler.py's resolution: explicit output_dir wins, else workspace/output."""
    explicit = cfg.output.output_dir
    if explicit is not None:
        return Path(str(explicit)).expanduser().resolve()
    return (workspace / "output").resolve()


def _find_latest_dp_run_dir(workspace: Path, cfg: TrainingConfig) -> Path | None:
    """Most recent dp run directory under the configured output_dir."""
    out_dir = _dp_output_dir(workspace, cfg)
    if not out_dir.is_dir():
        return None
    candidates: list[Path] = []
    for child in out_dir.iterdir():
        if not child.is_dir():
            continue
        if not (child / "latest").is_file():
            continue
        if not any(p.is_dir() and p.name.startswith("global_step") for p in child.iterdir()):
            continue
        candidates.append(child)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.name)


def _dp_resume_spec(cfg: TrainingConfig, workspace: Path) -> ResumeSpec:
    """Locate the dp run_dir and pack `--resume_from_checkpoint` argv."""
    out_dir = _dp_output_dir(workspace, cfg)
    if not out_dir.is_dir():
        raise ResumeNotReady(
            f"no diffusion-pipe output_dir found at {out_dir}; "
            "the run never produced a checkpoint folder"
        )
    run_dir = _find_latest_dp_run_dir(workspace, cfg)
    if run_dir is None:
        raise ResumeNotReady(
            f"no resumable diffusion-pipe run directory under {out_dir}; "
            "expected a timestamped subdir containing `latest` + `global_step*/`"
        )
    return ResumeSpec(
        extra_argv=[f"--resume_from_checkpoint={run_dir.name}"],
        cfg_overrides={"output.outputDir": str(out_dir)},
    )


def _dispatch_resume_spec(cfg: TrainingConfig, workspace: Path) -> ResumeSpec:
    """Dispatch to the per-backend resume helper based on `cfg.backend.type`."""
    backend_type = cfg.backend.type
    if backend_type == "kohya":
        return _kohya_resume_spec(workspace)
    if backend_type == "diffusion-pipe":
        return _dp_resume_spec(cfg, workspace)
    if backend_type == "anima_lora":
        return _anima_lora_resume_spec(workspace)
    raise ResumeNotReady(
        f"resume not implemented for backend.type={backend_type!r}"
    )


def _anima_lora_resume_spec(workspace: Path) -> ResumeSpec:
    """Locate anima_lora ``--save_state`` artifacts and pack them into a ResumeSpec.

    anima_lora is a sd-scripts fork, so its state-dir layout matches kohya.
    Emit ``--resume=<state_dir>`` and let accelerate's ``load_state``
    restore the LoRA state_dict from ``state_dir/model.safetensors``
    along with optimizer/scheduler. We deliberately do *not* pass
    ``--network_weights``.

    Also read ``train_state.json`` from the picked state directory and
    inject ``--initial_step=<N>`` + ``--skip_until_initial_step``.
    """
    state_dir = _find_latest_state_dir(workspace)
    if state_dir is None:
        raise ResumeNotReady(
            f"no anima_lora state directory found under {workspace}; "
            "resume requires ``cfg.resume.saveState=true`` (default) so "
            "the trainer wrote optimizer state at least once"
        )
    extra_argv = [
        f"--resume={state_dir}",
    ]
    train_state = state_dir / "train_state.json"
    try:
        data = json.loads(train_state.read_text(encoding="utf-8"))
        current_step = int(data.get("current_step", 0))
    except (OSError, ValueError, TypeError):
        current_step = 0
    if current_step > 0:
        extra_argv.extend([
            f"--initial_step={current_step}",
            "--skip_until_initial_step",
        ])
    return ResumeSpec(extra_argv=extra_argv)


def _apply_cfg_overrides(cfg: TrainingConfig, overrides: dict[str, Any]) -> TrainingConfig:
    """Apply a flat dot-path override mapping onto a validated TrainingConfig.

    Re-dumps the config to a dict, walks the dot path to set each value,
    then re-validates. Returns a fresh `TrainingConfig` so the caller's
    snapshot stays untouched. Empty overrides short-circuit.
    """
    if not overrides:
        return cfg
    data = cfg.model_dump(mode="json", by_alias=True)
    for dotted, value in overrides.items():
        cur: Any = data
        parts = dotted.split(".")
        for key in parts[:-1]:
            cur = cur.setdefault(key, {})
        cur[parts[-1]] = value
    return TrainingConfig.model_validate(data)


def _should_auto_resume(meta: dict[str, Any] | None, *, global_default: bool) -> bool:
    """Decide whether a single interrupted job qualifies for auto-resume.

    Per-job `metadata.auto_resume` overrides the global flag in either
    direction (True forces yes, False forces no). Sweep children are
    always declined — the sweep router already classifies interrupted
    children as failed and double-spawning would race the operator.
    """
    if meta is None:
        return global_default
    if meta.get("sweep_id") is not None:
        return False
    explicit = meta.get("auto_resume")
    if explicit is True:
        return True
    if explicit is False:
        return False
    return global_default


def _attempt_auto_resume(*, max_attempts: int, global_default: bool) -> int:
    """Re-launch every interrupted job that still has resumable artifacts.

    Returns the number of jobs successfully enqueued. Skips silently when:
      - The job is part of a sweep (sweep router owns those)
      - Per-job opt-out via `metadata.auto_resume = False`
      - Already hit `max_attempts` in this lineage
      - Config snapshot fails schema validation (logged at WARNING)
      - No checkpoint produced yet (logged at INFO; the run never reached
        a save_state / global_step* boundary)

    Hooked from `app._lifespan` after `mark_orphans_interrupted` flips
    survivors to interrupted, before the scheduler starts. Pre-queueing
    means resumed jobs are first-in-line when workers come online.
    """
    from lorahub.api import state as _state  # noqa: PLC0415

    resumed = 0
    for job in list(_state.registry.list()):
        if job.state is not JobState.interrupted:
            continue
        if not _should_auto_resume(job.metadata, global_default=global_default):
            continue
        attempts = (job.metadata or {}).get("auto_resume_attempts", 0)
        if attempts >= max_attempts:
            log.info(
                "auto-resume: skipping job %s — hit max attempts (%d)",
                job.id,
                max_attempts,
            )
            continue
        try:
            cfg = TrainingConfig.model_validate(job.config_snapshot)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "auto-resume: config snapshot for job %s failed validation: %s",
                job.id,
                exc,
            )
            continue
        try:
            spec = _dispatch_resume_spec(cfg, job.workspace)
        except ResumeNotReady as exc:
            log.info("auto-resume: skipping job %s — %s", job.id, exc)
            continue
        cfg = _apply_cfg_overrides(cfg, spec.cfg_overrides)
        try:
            from lorahub.api import jobs_helpers as _jh  # noqa: PLC0415

            _jh._relaunch_job_in_place(
                job,
                cfg,
                extra_argv=spec.extra_argv,
                metadata_patch={
                    "auto_resume": True,
                    "auto_resume_attempts": attempts + 1,
                    "last_resumed_at": datetime.now(UTC).isoformat(),
                },
            )
        except Exception:  # noqa: BLE001
            log.exception("auto-resume: failed to relaunch job %s", job.id)
            continue
        resumed += 1
        log.info(
            "auto-resume: re-enqueued job %s in place (attempt %d)",
            job.id,
            attempts + 1,
        )
    return resumed


def _requeue_pending_jobs() -> int:
    """Re-submit any persisted ``queued`` jobs into the scheduler.

    A queued JobRecord that survives a restart still has its config
    snapshot on disk but no scheduler task waiting for it. This helper
    re-validates the snapshot and pushes a fresh worker closure into
    ``sched.scheduler`` so the row eventually transitions out of
    ``queued`` instead of sitting there forever.

    Snapshot validation failures (stale schema) flip the row to
    ``failed`` rather than silently abandon it — operators see a real
    diagnostic on /jobs.
    """
    from lorahub.api import state as _state  # noqa: PLC0415

    requeued = 0
    for job in list(_state.registry.list()):
        if job.state is not JobState.queued:
            continue
        try:
            cfg = TrainingConfig.model_validate(job.config_snapshot)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "requeue: queued job %s has stale snapshot — marking failed: %s",
                job.id,
                exc,
            )
            job.state = JobState.failed
            job.error = f"stale config snapshot on restart: {exc}"
            job.finished_at = datetime.now(UTC)
            _state.registry.update(job)
            continue
        try:
            from lorahub.api import jobs_helpers as _jh  # noqa: PLC0415

            _jh._enqueue_launch(job, cfg)
        except Exception:  # noqa: BLE001
            log.exception("requeue: failed to re-enqueue queued job %s", job.id)
            continue
        requeued += 1
        log.info("requeue: re-submitted queued job %s to scheduler", job.id)
    return requeued


def _migrate_snapshots_to_camel() -> int:
    """One-shot migration: re-dump every JobRecord's config_snapshot with
    ``by_alias=True`` so the on-disk shape matches what the front-end form
    expects (camelCase). Idempotent: snapshots that are already camelCase
    round-trip unchanged through pydantic.

    Older builds dumped the snapshot with field names (snake_case) while
    the schema's alias_generator emits camelCase. The form widgets read
    camelCase keys, so loading an old job into ResumeWithEditDialog
    silently fell back to the default backend section because
    ``value.backend.anima_lora`` doesn't match ``value.backend.animaLora``.

    Schema-broken snapshots are left alone; the requeue / resume paths
    will surface them as "stale config snapshot on restart" the next
    time the operator interacts with that job.
    """
    from lorahub.api import state as _state  # noqa: PLC0415

    migrated = 0
    skipped = 0
    failed = 0
    for job in list(_state.registry.list()):
        snap = job.config_snapshot
        if not isinstance(snap, dict):
            skipped += 1
            continue
        try:
            cfg = TrainingConfig.model_validate(snap)
        except Exception:  # noqa: BLE001
            failed += 1
            continue
        new_snap = cfg.model_dump(mode="json", by_alias=True)
        if new_snap == snap:
            skipped += 1
            continue
        job.config_snapshot = new_snap
        _state.registry.update(job)
        migrated += 1
    if migrated or failed:
        log.info(
            "snapshot migration: %d converted to camelCase, %d unchanged, "
            "%d schema-broken (left as-is)",
            migrated,
            skipped,
            failed,
        )
    return migrated


__all__ = [
    "ResumeNotReady",
    "ResumeSpec",
    "_anima_lora_resume_spec",
    "_apply_cfg_overrides",
    "_attempt_auto_resume",
    "_dispatch_resume_spec",
    "_dp_output_dir",
    "_dp_resume_spec",
    "_find_latest_dp_run_dir",
    "_find_latest_safetensors",
    "_find_latest_state_dir",
    "_kohya_resume_spec",
    "_migrate_snapshots_to_camel",
    "_requeue_pending_jobs",
    "_should_auto_resume",
]
