"""Job lifecycle orchestration.

Owns the wiring between a validated ``TrainingConfig``, the
``JobRecord`` registry, and the process-wide scheduler. Every call
that creates / re-launches a training job goes through here.

The big function is ``_enqueue_launch`` — it builds the worker
closure that runs on the scheduler thread, opens an events sink,
spawns the GPU sampler + preview worker, and waits on the subprocess
until it exits. The closure is intentionally fat for now (every step
shares state with the next); see the audit report for the long-term
``JobLifecycle`` class refactor.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lorahub.api import scheduler as sched
from lorahub.api import state
from lorahub.api.state import JobRecord, JobState
from lorahub.core.backends.anima_lora.backend import AnimaLoraBackend
from lorahub.core.backends.diffusion_pipe.backend import DiffusionPipeBackend
from lorahub.core.backends.kohya.backend import KohyaBackend
from lorahub.core.config.loader import dump_config
from lorahub.core.config.schema import TrainingConfig
from lorahub.core.events import EventType, JsonlEventSink, TrainingEvent

from .paths_norm import _normalize_recipe_paths
from .preview import _gpu_sampler_loop, _maybe_start_preview_worker

log = logging.getLogger(__name__)

_TERMINAL_STATES = (
    JobState.succeeded,
    JobState.failed,
    JobState.canceled,
    JobState.interrupted,
)


def _select_backend(cfg: TrainingConfig):  # type: ignore[no-untyped-def]
    """Pick the training backend implementation that the config asks for."""
    backend_type = cfg.backend.type
    if backend_type == "kohya":
        return KohyaBackend()
    if backend_type == "diffusion-pipe":
        return DiffusionPipeBackend()
    if backend_type == "anima_lora":
        return AnimaLoraBackend()
    msg = f"unsupported backend type: {backend_type!r}"
    raise ValueError(msg)


def _launch_job(
    cfg: TrainingConfig,
    workspace: Path,
    *,
    extra_argv: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Materialize a workspace, register a queued job, and hand it to the scheduler.

    Returns immediately after enqueueing. The actual `backend.launch()` call
    runs on the scheduler worker thread, which holds its slot until the
    subprocess exits — that is what makes `concurrency=1` actually serialise
    runs instead of fan-out racing.

    `extra_argv` is appended after the compiler's argv (used by `/jobs/{id}/resume`
    to inject `--resume=<state>` and `--network_weights=<safetensors>`).

    `metadata` is stamped onto the JobRecord so callers like the sweep
    router can later filter / aggregate jobs by their orchestrator id.
    """
    workspace.mkdir(parents=True, exist_ok=True)

    _normalize_recipe_paths(cfg)

    snapshot = cfg.model_dump(mode="json", by_alias=True)
    job = state.registry.create(workspace=workspace, config_snapshot=snapshot)
    if metadata is not None:
        job.metadata = metadata
        state.registry.update(job)
    dump_config(cfg, workspace / "config.yaml")

    # Resolve the enqueue helper through the package façade so tests
    # that ``monkeypatch.setattr(jobs_helpers, "_enqueue_launch", ...)``
    # see their stub even though the call originates from this module.
    from lorahub.api import jobs_helpers as _jh  # noqa: PLC0415

    _jh._enqueue_launch(job, cfg, extra_argv=extra_argv)
    return job.to_summary()


def _relaunch_job_in_place(
    job: JobRecord,
    cfg: TrainingConfig,
    *,
    extra_argv: list[str] | None = None,
    metadata_patch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Reset an existing JobRecord and re-enqueue it on its original workspace.

    Used by both ``/rerun`` (no extra_argv) and ``/resume`` (resume argv +
    cfg overrides applied upstream). The id, workspace, and event log are
    preserved so the user keeps a single timeline per logical job; only
    the runtime fields (state, started_at, finished_at, returncode, error,
    pid, handle) are wiped.

    ``metadata_patch`` is shallow-merged onto the existing metadata bag
    so callers can stamp things like ``last_resumed_at`` / ``rerun_count``
    without clobbering sweep tags or auto-resume counters.

    The on-disk ``events.jsonl`` and the in-memory event ring are both
    cleared before the new run starts, so logs reflect the current run
    only. Resume runs (``extra_argv`` non-empty) keep the prior log so
    the timeline survives across the resume boundary.
    """
    workspace = job.workspace
    workspace.mkdir(parents=True, exist_ok=True)

    _normalize_recipe_paths(cfg)

    if not extra_argv:
        job.events.clear()
        event_log = workspace / "events.jsonl"
        if event_log.is_file():
            event_log.unlink()

    snapshot = cfg.model_dump(mode="json", by_alias=True)
    job.config_snapshot = snapshot
    job.state = JobState.queued
    job.started_at = None
    job.finished_at = None
    job.returncode = None
    job.error = None
    job.pid = None
    job.handle = None
    if metadata_patch:
        merged = dict(job.metadata or {})
        merged.update(metadata_patch)
        job.metadata = merged
    state.registry.update(job)
    dump_config(cfg, workspace / "config.yaml")

    from lorahub.api import jobs_helpers as _jh  # noqa: PLC0415

    _jh._enqueue_launch(job, cfg, extra_argv=extra_argv)
    return job.to_summary()


def _enqueue_launch(
    job: JobRecord,
    cfg: TrainingConfig,
    *,
    extra_argv: list[str] | None = None,
) -> None:
    """Wire a JobRecord up to the process-wide scheduler.

    The closure runs on a worker thread: it opens an events sink, calls
    `backend.launch`, then blocks until the subprocess exits so the slot
    is held for the entire run. If the job was canceled while still queued,
    the closure is a no-op so cancellation is observably instant for
    queued jobs.
    """
    workspace = job.workspace
    # ``_select_backend`` is resolved through the package façade so tests
    # that ``monkeypatch.setattr(jobs_helpers, "_select_backend", ...)``
    # to install a FakeBackend take effect even though the call lives in
    # this submodule.
    from lorahub.api import jobs_helpers as _jh  # noqa: PLC0415

    backend = _jh._select_backend(cfg)
    sink = JsonlEventSink(workspace / "events.jsonl")
    # Set lazily once the preview worker is up — the on_event handler
    # forwards `checkpoint_saved` events to it so the worker reacts in
    # < 1s instead of waiting for its polling tick.
    preview_worker_ref: dict[str, Any] = {}
    # Guard against late events arriving after the sink is closed by the
    # `done` handler. GPU sampler / diagnostic threads can still fire
    # events after the subprocess exits; without this flag those calls
    # would hit a closed sink and raise RuntimeError.
    _sink_closed = False

    def on_event(ev: TrainingEvent) -> None:
        nonlocal _sink_closed
        if _sink_closed and ev.type is not EventType.done:
            return
        sink(ev)
        state.registry.record_event(job.id, ev)
        if ev.type is EventType.checkpoint_saved:
            worker = preview_worker_ref.get("worker")
            if worker is not None:
                ckpt_name = _extract_ckpt_name(ev)
                if ckpt_name:
                    try:
                        worker.notify_checkpoint(ckpt_name)
                    except Exception:  # noqa: BLE001
                        log.exception("preview worker notify failed")
        if ev.type is EventType.done:
            j = state.registry.get(job.id)
            if j is not None:
                rc = ev.payload.get("returncode")
                j.returncode = rc
                if j.state is JobState.canceling:
                    j.state = JobState.canceled
                else:
                    j.state = JobState.succeeded if rc == 0 else JobState.failed
                j.finished_at = datetime.now(UTC)
                state.registry.update(j)
                # Persist a structured failure report so the user can
                # find this run in Settings → 错误上报 even after the
                # job list is cleared. ``unknown`` covers the common
                # "exit !=0 with no matched diagnosis pattern" path —
                # the streaming watcher's diagnostic_warning events
                # (matched on the regex catalogue) are still the
                # primary signal during the run.
                if j.state is JobState.failed:
                    _report_job_failure(j, returncode=rc)
                # Sweep feedback: pull the final loss/val_loss out of the
                # workspace and push it into the parent sweep's sampler.
                # No-op for non-sweep jobs and for sweeps that aren't
                # registered as active (restart, already exhausted).
                from lorahub.api.sweep_runtime import (  # noqa: PLC0415
                    report_terminal_job,
                )
                with contextlib.suppress(Exception):
                    report_terminal_job(j)
            _sink_closed = True
            sink.__exit__(None, None, None)

    def task(slot: int) -> None:
        current = state.registry.get(job.id)
        if current is None or current.state is not JobState.queued:
            return
        # Pin the worker subprocess to the assigned GPU. With concurrency=1
        # the slot is always 0; with N>1 each worker gets a distinct GPU id
        # so kohya / diffusion-pipe see exactly one device.
        slot_env = {"CUDA_VISIBLE_DEVICES": str(slot)}
        from lorahub.api.settings import env_overrides  # noqa: PLC0415
        from lorahub.api import app as _app  # noqa: PLC0415

        slot_env.update(env_overrides(_app._settings_store.load()))
        # Switch out of queued before backend.launch — anima_lora's
        # auto-preprocess can run for a couple of minutes (resize +
        # cache_latents + cache_text_embeddings) before the trainer
        # subprocess is spawned. Without this the UI keeps reporting
        # "排队中" even though the worker is already busy.
        current.state = JobState.preparing
        state.registry.update(current)
        sink.__enter__()
        try:
            handle = backend.launch(
                cfg,
                workspace=workspace,
                on_event=on_event,
                extra_argv=extra_argv,
                env=slot_env,
            )
        except Exception as exc:  # noqa: BLE001
            j = state.registry.get(job.id)
            if j is not None:
                j.state = JobState.failed
                j.error = repr(exc)
                j.finished_at = datetime.now(UTC)
                state.registry.update(j)
                from lorahub.api.error_reporter import (  # noqa: PLC0415
                    capture_exception,
                )

                with contextlib.suppress(Exception):
                    capture_exception(
                        exc,
                        source="backend.job",
                        category="launch_failed",
                        title=f"backend.launch raised before training started: {job.id[-12:]}",
                        job_id=job.id,
                        context={
                            "workspace": str(workspace),
                            "stage": "launch",
                        },
                    )
                from lorahub.api.sweep_runtime import (  # noqa: PLC0415
                    report_terminal_job,
                )
                with contextlib.suppress(Exception):
                    report_terminal_job(j)
            with contextlib.suppress(Exception):
                _sink_closed = True
                sink.__exit__(None, None, None)
            return
        j = state.registry.get(job.id)
        if j is not None:
            j.handle = handle
            j.pid = handle.pid
            from lorahub.api.store import _pid_create_time  # noqa: PLC0415

            j.pid_create_time = _pid_create_time(handle.pid) if handle.pid else None
            j.state = JobState.running
            j.started_at = datetime.now(UTC)
            state.registry.update(j)

        # Spawn a low-frequency GPU sampler so the analysis tab gets a
        # post-hoc resource trend (util / VRAM / temp). Lives only as
        # long as the job is running; the `done` event triggers shutdown.
        sampler_stop = threading.Event()
        sampler = threading.Thread(
            target=_gpu_sampler_loop,
            args=(job.id, slot, on_event, sampler_stop),
            daemon=True,
            name=f"gpu-sampler-{job.id[-6:]}",
        )
        sampler.start()

        # Optional: live-preview worker for diffusion-pipe runs. dp itself
        # never renders sample images; this is the lorahub-side hook that
        # turns each new dp checkpoint into a PNG using the user's
        # `sampling.promptsFile`. Off by default (cfg.sampling.enable_live_inference).
        preview_stop = threading.Event()
        preview_handle = _maybe_start_preview_worker(
            cfg, workspace, job.id, on_event, preview_stop,
        )
        if preview_handle is not None:
            preview_thread, preview_worker = preview_handle
            preview_worker_ref["worker"] = preview_worker
        else:
            preview_thread = None

        try:
            handle.wait(timeout=None)
        except Exception:  # noqa: BLE001
            log.exception("worker wait() failed for job %s", job.id)
        finally:
            sampler_stop.set()
            preview_stop.set()
            if preview_thread is not None:
                preview_thread.join(timeout=5.0)

    sched.scheduler.submit(job.id, task)


def _extract_ckpt_name(ev: TrainingEvent) -> str | None:
    """Pick the ckpt directory name out of a `checkpoint_saved` event.

    dp's parser emits the path as `payload.path` (the full
    "Saving model to directory <path>" target). For dp this is
    `<output_dir>/<run_ts>/{step|epoch}{N}` so the ckpt name is the
    last path component.
    """
    payload = ev.payload or {}
    raw = payload.get("path") or payload.get("checkpoint")
    if not isinstance(raw, str) or not raw:
        return None
    name = Path(raw.strip()).name
    return name or None


def _archive_workspace(workspace: Path, job_id: str) -> tuple[Path | None, list[str]]:
    """Move `workspace` under `<parent>/_archive/<name>-<short_id>`.

    Returns (new_path, warnings). `new_path` is None if the workspace did not
    exist or the move failed; in either case the caller still drops the store
    record and the warnings explain what happened.
    """
    warnings: list[str] = []
    if not workspace.exists():
        warnings.append(f"workspace did not exist: {workspace}")
        return None, warnings

    parent = workspace.parent
    archive_dir = parent / "_archive"
    short_id = job_id[-8:] if len(job_id) > 8 else job_id
    target = archive_dir / f"{workspace.name}-{short_id}"

    try:
        archive_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        warnings.append(f"could not create archive dir {archive_dir}: {exc}")
        return None, warnings

    final = target
    counter = 1
    while final.exists():
        final = archive_dir / f"{workspace.name}-{short_id}-{counter}"
        counter += 1

    try:
        workspace.rename(final)
    except OSError as exc:
        warnings.append(f"could not move workspace: {exc}")
        return None, warnings

    return final, warnings


def _report_job_failure(job: JobRecord, *, returncode: int | None) -> None:
    """Capture a structured failure report when a training job exits non-zero.

    Pulls the same context the user would otherwise have to assemble by
    hand: a small tail of ``events.jsonl`` (so the matching
    ``diagnostic_warning`` events the streaming watcher emitted are
    visible), the tail of ``training.log`` (so the trainer's own
    stderr / traceback survives even if the diagnoser couldn't classify
    it), and the diagnose_failure verdict (highest-severity matched
    pattern + remediation).

    Best-effort: any failure inside this helper is swallowed so the job
    cleanup path keeps running.
    """
    try:
        from lorahub.api.error_reporter import capture  # noqa: PLC0415
        from lorahub.api.training_assistant import diagnose_failure  # noqa: PLC0415

        workspace = Path(job.workspace) if job.workspace else None
        diag: dict[str, Any] = {}
        if workspace and workspace.is_dir():
            with contextlib.suppress(Exception):
                diag = diagnose_failure(workspace, returncode=returncode)
        head = None
        category = "exit_non_zero"
        if isinstance(diag, dict):
            findings = diag.get("findings") or []
            if findings:
                # findings is already severity-sorted in diagnose_failure;
                # mirror its top pick as the category for the registry list
                # so users can filter by ``oom`` / ``nan_loss`` / etc.
                head = findings[0]
                if isinstance(head, dict):
                    category = str(head.get("category") or category)
        title = (
            f"job {job.id[-12:]} exited with {returncode}"
            if returncode is not None
            else f"job {job.id[-12:]} failed"
        )
        msg_parts: list[str] = []
        if isinstance(diag, dict) and diag.get("summary"):
            msg_parts.append(str(diag["summary"]))
        if isinstance(head, dict) and head.get("remediation"):
            msg_parts.append(f"remediation: {head['remediation']}")
        message = "\n".join(msg_parts) or (
            f"trainer exit code {returncode}; no log excerpt was reachable."
        )
        ctx: dict[str, Any] = {
            "returncode": returncode,
            "workspace": str(workspace) if workspace else None,
            "log_path": diag.get("log_path") if isinstance(diag, dict) else None,
            "log_excerpt": (
                diag.get("log_excerpt") if isinstance(diag, dict) else ""
            )[:8000],
            "findings": diag.get("findings") if isinstance(diag, dict) else [],
        }
        capture(
            severity="error",
            source="backend.job",
            category=category,
            title=title,
            message=message,
            context=ctx,
            job_id=job.id,
        )
    except Exception:  # noqa: BLE001
        log.exception("failed to record job-failure error report for %s", job.id)


__all__ = [
    "_TERMINAL_STATES",
    "_archive_workspace",
    "_enqueue_launch",
    "_extract_ckpt_name",
    "_launch_job",
    "_relaunch_job_in_place",
    "_select_backend",
]
