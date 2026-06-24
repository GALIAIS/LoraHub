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
import inspect
import logging
import re
import secrets
import threading
from collections import Counter
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

from .paths_norm import _normalize_config_paths
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

    _apply_settings_gpu_dispatch_default(cfg)
    _normalize_config_paths(cfg)
    _resolve_runtime_seeds(cfg)
    _resolve_trigger_word(cfg)
    _materialise_prompts_file(cfg, workspace)

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

    _apply_settings_gpu_dispatch_default(cfg)
    _normalize_config_paths(cfg)
    _resolve_runtime_seeds(cfg)
    _resolve_trigger_word(cfg)
    _materialise_prompts_file(cfg, workspace)

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
        if ev.type is EventType.log:
            _capture_wandb_run_url(job.id, ev)
        if ev.type is EventType.checkpoint_saved:
            worker = preview_worker_ref.get("worker")
            if worker is not None:
                ckpt_name = _extract_ckpt_name(ev)
                if ckpt_name:
                    try:
                        worker.notify_checkpoint(ckpt_name)
                    except Exception:  # noqa: BLE001
                        log.exception("preview worker notify failed")
            # Side-band LoRA spectrum analysis. Runs on a daemon thread so
            # the live training loop is never blocked; results land via
            # ``on_event`` like every other training event.
            if _jh._lora_spectrum_enabled(cfg):
                ckpt_path = _resolve_checkpoint_path(ev, workspace)
                if ckpt_path is not None:
                    from lorahub.api.lora_analysis import (  # noqa: PLC0415
                        is_lora_checkpoint,
                        schedule_lora_spectrum,
                    )

                    if is_lora_checkpoint(ckpt_path):
                        schedule_lora_spectrum(
                            ckpt_path,
                            step=_extract_step(ev),
                            on_event=on_event,
                            job_id=job.id,
                        )
        if ev.type is EventType.sample_ready:
            # Catastrophic-forgetting probe: only flagged prompts (those
            # whose filename matches forget/neutral/preserve markers in
            # the user's sample_prompts file) are compared against their
            # earliest-seen sample for this job.
            sample_path = _resolve_sample_path(ev, workspace)
            if sample_path is not None:
                from lorahub.api.forgetting_probe import (  # noqa: PLC0415
                    derive_prompt_key,
                    is_neutral_prompt,
                    schedule_forgetting_probe,
                )

                if is_neutral_prompt(sample_path):
                    schedule_forgetting_probe(
                        sample_path,
                        prompt_key=derive_prompt_key(sample_path),
                        step=_extract_step(ev),
                        on_event=on_event,
                        job_id=job.id,
                    )
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

    def task(slot: int | list[int]) -> None:
        current = state.registry.get(job.id)
        if current is None or current.state is not JobState.queued:
            return
        # Pin the subprocess to its assigned GPU set. Default mode gets
        # one id; distributed mode gets a comma-separated group.
        assigned_slots = [slot] if isinstance(slot, int) else list(slot)
        slot_env = {"CUDA_VISIBLE_DEVICES": ",".join(str(s) for s in assigned_slots)}
        from lorahub.api.settings import env_overrides  # noqa: PLC0415
        from lorahub.api import app as _app  # noqa: PLC0415

        slot_env.update(env_overrides(_app._settings_store.load()))
        from lorahub.api.wandb_env import wandb_env  # noqa: PLC0415

        slot_env.update(wandb_env(cfg))
        # Switch out of queued before backend.launch — anima_lora's
        # auto-preprocess can run for a couple of minutes (resize +
        # cache_latents + cache_text_embeddings) before the trainer
        # subprocess is spawned. Without this the UI keeps reporting
        # "排队中" even though the worker is already busy.
        current.state = JobState.preparing
        meta = dict(current.metadata or {})
        meta["gpu_slots"] = assigned_slots
        meta["gpu_dispatch_mode"] = _gpu_dispatch_mode(cfg)
        current.metadata = meta
        state.registry.update(current)
        sink.__enter__()
        try:
            kwargs = {
                "extra_argv": extra_argv,
                "env": slot_env,
            }
            if _backend_accepts_gpu_count(backend):
                kwargs["gpu_count"] = len(assigned_slots)
            handle = backend.launch(
                cfg,
                workspace=workspace,
                on_event=on_event,
                **kwargs,
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
            args=(job.id, assigned_slots[0], on_event, sampler_stop),
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

    sched.scheduler.submit(
        job.id,
        task,
        vram_required=_vram_required(backend, cfg),
        slots_required=_slots_required(cfg),
    )


def _gpu_dispatch_mode(cfg: TrainingConfig) -> str:
    return cfg.backend.gpu_dispatch.mode


def _apply_settings_gpu_dispatch_default(cfg: TrainingConfig) -> None:
    if "gpu_dispatch" in cfg.backend.model_fields_set:
        return
    try:
        from lorahub.api import app as _app  # noqa: PLC0415

        settings = _app._settings_store.load()
    except Exception:  # noqa: BLE001
        return
    cfg.backend.gpu_dispatch.mode = settings.gpu_dispatch_mode
    cfg.backend.gpu_dispatch.num_gpus = settings.gpu_dispatch_num_gpus


def _slots_required(cfg: TrainingConfig) -> int:
    dispatch = cfg.backend.gpu_dispatch
    if dispatch.mode != "distributed":
        return 1
    return dispatch.num_gpus or sched.scheduler.concurrency


def _backend_accepts_gpu_count(backend: Any) -> bool:
    try:
        return "gpu_count" in inspect.signature(backend.launch).parameters
    except (TypeError, ValueError):
        return False


def _vram_required(backend: Any, cfg: TrainingConfig) -> float | None:
    estimate = getattr(backend, "estimate_vram", None)
    if not callable(estimate):
        return None
    try:
        return float(estimate(cfg).total_gib)
    except Exception:  # noqa: BLE001
        return None


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


def _resolve_checkpoint_path(
    ev: TrainingEvent, workspace: Path
) -> Path | None:
    """Best-effort absolute path to the saved checkpoint file.

    The training-side parsers emit the path verbatim from the trainer's
    log. Most are absolute, some are workspace-relative. Anything that
    points to a directory rather than a file is resolved to the most
    recently modified ``.safetensors`` inside (dp's
    ``output_dir/<run>/step<N>/`` layout).
    """
    payload = ev.payload or {}
    raw = payload.get("path") or payload.get("checkpoint")
    if not isinstance(raw, str) or not raw:
        return None
    candidate = Path(raw.strip())
    if not candidate.is_absolute():
        candidate = (workspace / candidate).resolve()
    if candidate.is_file():
        return candidate
    if candidate.is_dir():
        # Pick the newest .safetensors / .sft inside.
        files = sorted(
            (
                p
                for p in candidate.rglob("*")
                if p.is_file() and p.suffix.lower() in {".safetensors", ".sft"}
            ),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return files[0] if files else None
    return None


def _extract_step(ev: TrainingEvent) -> int | None:
    payload = ev.payload or {}
    raw = payload.get("step")
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str):
        try:
            return int(raw)
        except ValueError:
            return None
    return None


def _resolve_sample_path(
    ev: TrainingEvent, workspace: Path
) -> Path | None:
    """Absolute path to a sample-ready event's image file, if it exists."""
    payload = ev.payload or {}
    raw = payload.get("path") or payload.get("sample") or payload.get("file")
    if not isinstance(raw, str) or not raw:
        return None
    candidate = Path(raw.strip())
    if not candidate.is_absolute():
        candidate = (workspace / candidate).resolve()
    return candidate if candidate.is_file() else None


def _lora_spectrum_enabled(cfg: Any) -> bool:
    """Whether to run side-band SVD on every checkpoint.

    Default-on; config sets ``sampling.spectrum_analysis = false`` to
    opt out (e.g. air-gapped users with > 100-layer adapters where the
    SVD wall time isn't trivial).
    """
    from lorahub.api.lora_analysis import is_enabled  # noqa: PLC0415

    try:
        return is_enabled(cfg)
    except Exception:  # noqa: BLE001
        return False


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


# Maximum value for randomly-drawn seeds. ComfyUI uses 2**50; mirroring
# that keeps copy-pasting workflow seeds across the two tools sensible.
_RANDOM_SEED_MAX = 1_125_899_906_842_624


def _draw_random_seed() -> int:
    """Cryptographically-strong random seed in [0, _RANDOM_SEED_MAX)."""
    return secrets.randbelow(_RANDOM_SEED_MAX)


def _resolve_runtime_seeds(cfg: TrainingConfig) -> None:
    """Resolve ``-1`` sentinels in seed fields.

    Two different "-1" semantics live here:

    * ``sampling.seed`` (top-level, ComfyUI-style "draw at run time") is
      the *training* seed — controls dataset shuffle, dropout, etc.
      We still draw a fresh integer at job-start so the run is
      reproducible from the snapshot. Same legacy behaviour.

    * ``sampling.prompts[*].seed = -1`` is the **per-preview** sentinel.
      We previously drew a single random integer at job-start and froze
      it into the prompt row, which meant every epoch's preview render
      hit the same ``torch.manual_seed`` and produced visually identical
      images even as the LoRA evolved. Now we collapse ``-1`` to
      ``None`` instead — the prompt-file materialiser then omits
      ``--d`` for that row, and the trainer's ``_sample_image_inference``
      treats a missing seed as "use ambient RNG", giving a fresh sample
      each epoch. Concrete integer seeds (e.g. 42) are still honoured
      verbatim for users who want a reproducible preview.
    """
    sampling = cfg.sampling
    if sampling.seed == -1:
        sampling.seed = _draw_random_seed()
        log.info("sampling.seed: drew runtime random %d", sampling.seed)
    for ps in sampling.prompts:
        if ps.seed == -1:
            ps.seed = None
            log.info(
                "sampling.prompts seed: -1 → None (preview will randomise per epoch)"
            )


# Match ``${TRIGGER}`` / ``${trigger}`` plus an optional trailing ``,``
# and one or more spaces so we can wipe both the placeholder and the
# comma-glue around it cleanly when no trigger word is available.
_TRIGGER_PLACEHOLDER_WITH_GLUE_RE = re.compile(
    r"\$\{trigger\}\s*,?\s*", flags=re.IGNORECASE
)
_TRIGGER_PLACEHOLDER_BARE_RE = re.compile(
    r"\$\{trigger\}", flags=re.IGNORECASE
)
# Sample a small head of the dataset rather than every caption — the
# point is the *most common* first token, and the cheap N=64 head is
# robust to a few outliers.
_TRIGGER_SCAN_LIMIT = 64
# Tags rejected as "obviously not a trigger" when scanning captions —
# these are the universal anime-style quality / character-count tokens
# that virtually every caption opens with.
_NON_TRIGGER_TOKENS = frozenset({
    "1girl", "1boy", "2girls", "2boys", "3girls", "3boys",
    "multiple girls", "multiple boys", "multiple_girls", "multiple_boys",
    "no humans", "no_humans", "solo", "duo", "trio",
    "masterpiece", "best quality", "best_quality", "score_7",
    "score_8", "score_9", "score_8_up", "score_9_up",
    "highres", "absurdres", "ultra-detailed", "ultra_detailed",
    "general", "sensitive", "questionable", "explicit",
})


def _scan_dataset_for_trigger(dataset_dir: Path) -> str | None:
    """Best-effort recovery of a trigger word from caption .txt files.

    Reads up to ``_TRIGGER_SCAN_LIMIT`` ``.txt`` siblings in the
    dataset directory, takes each file's first comma-separated token,
    and returns the most common value that isn't in
    ``_NON_TRIGGER_TOKENS``. Returns ``None`` when:

      * the dataset path doesn't exist
      * no ``.txt`` captions are found
      * every first-token is a generic anime-style quality tag

    Cheap enough to run unconditionally on every job-launch (≤ 64
    file reads, < 50ms even on a cold disk).
    """
    if not dataset_dir.is_dir():
        return None
    counter: Counter[str] = Counter()
    seen = 0
    for txt in dataset_dir.rglob("*.txt"):
        if seen >= _TRIGGER_SCAN_LIMIT:
            break
        try:
            raw = txt.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        head = raw.split(",", 1)[0].strip().lower()
        if not head or head in _NON_TRIGGER_TOKENS:
            continue
        # WD14 underscores → human-readable spaces, matching the
        # dataset prep we already do elsewhere.
        head = head.replace("_", " ")
        counter[head] += 1
        seen += 1
    if not counter:
        return None
    pick, _ = counter.most_common(1)[0]
    return pick


def _resolve_trigger_word(cfg: TrainingConfig) -> None:
    """Resolve ``sampling.trigger_word`` and substitute the ${TRIGGER}
    placeholder in every prompt body.

    Resolution order:
      1. ``sampling.trigger_word`` set explicitly in the config
      2. Most common first comma-separated token across the dataset's
         caption ``.txt`` files (excluding generic quality tags)
      3. None — placeholders are then stripped along with their
         comma-glue so the prompt stays well-formed

    The resolved value is written back into the config so the snapshot
    + prompts.txt materialiser see the same concrete string.
    """
    sampling = cfg.sampling
    if not sampling.prompts:
        return
    trigger = (sampling.trigger_word or "").strip()
    if not trigger:
        # Try dataset recovery. ``cfg.dataset.source`` can be missing
        # / a non-existent path during preflight; ``_scan_dataset_for_trigger``
        # tolerates both.
        try:
            ds_path = Path(str(cfg.dataset.source)).expanduser()
        except (TypeError, AttributeError):
            ds_path = None
        if ds_path is not None:
            trigger = _scan_dataset_for_trigger(ds_path) or ""
        if trigger:
            log.info(
                "sampling.trigger_word: recovered %r from dataset %s",
                trigger, ds_path,
            )
            sampling.trigger_word = trigger
    if not trigger:
        log.info("sampling.trigger_word: unset and unrecovered — stripping placeholders")
    for ps in sampling.prompts:
        if "${" not in ps.prompt and "${" not in (ps.negative or ""):
            continue
        if trigger:
            ps.prompt = _TRIGGER_PLACEHOLDER_BARE_RE.sub(trigger, ps.prompt)
            if ps.negative:
                ps.negative = _TRIGGER_PLACEHOLDER_BARE_RE.sub(trigger, ps.negative)
        else:
            ps.prompt = _TRIGGER_PLACEHOLDER_WITH_GLUE_RE.sub("", ps.prompt).strip(", ")
            if ps.negative:
                ps.negative = _TRIGGER_PLACEHOLDER_WITH_GLUE_RE.sub(
                    "", ps.negative
                ).strip(", ")


def _materialise_prompts_file(cfg: TrainingConfig, workspace: Path) -> None:
    """Render ``cfg.sampling.prompts`` into ``workspace/prompts.txt``.

    The kohya/anima trainers and the in-process preview worker both
    consume a kohya-style prompts.txt (one prompt per line, optional
    ``--w / --h / --d / --s / --l / --n`` flags). When the user has
    populated the structured ``prompts`` list in yaml we materialise
    that here so no upstream tooling has to learn about the new
    schema. ``promptsFile`` (legacy) takes precedence when set, so
    older configs keep working unchanged.
    """
    sampling = cfg.sampling
    if not sampling.prompts:
        return
    if sampling.prompts_file is not None:
        return
    target = workspace / "prompts.txt"
    lines: list[str] = []
    for spec in sampling.prompts:
        body = spec.prompt.strip()
        if not body:
            continue
        flags: list[str] = []
        if spec.width is not None:
            flags.append(f"--w {spec.width}")
        if spec.height is not None:
            flags.append(f"--h {spec.height}")
        if spec.seed is not None:
            flags.append(f"--d {spec.seed}")
        if spec.steps is not None:
            flags.append(f"--s {spec.steps}")
        if spec.cfg is not None:
            flags.append(f"--l {spec.cfg}")
        if spec.sampler is not None:
            flags.append(f"--ss {spec.sampler}")
        if spec.flow_shift is not None:
            flags.append(f"--fs {spec.flow_shift}")
        if spec.negative:
            flags.append(f"--n {spec.negative}")
        line = body
        if flags:
            line = f"{body} {' '.join(flags)}"
        lines.append(line)
    if not lines:
        return
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sampling.prompts_file = target
    log.info("prompts.txt materialised at %s (%d prompts)", target, len(lines))


# wandb prints a single banner line at run startup ("View run at <url>" or
# "wandb: View run at <url>"); the pattern is stable across the wandb 0.16+
# family, which is what every backend in this repo bundles.
_WANDB_RUN_URL_RE = re.compile(
    r"View\s+run\s+(?:.+?\s+)?at\s+(?P<url>https?://\S+)",
    re.IGNORECASE,
)


def _capture_wandb_run_url(job_id: str, ev: TrainingEvent) -> None:
    """Stash the wandb run URL on ``JobRecord.metadata`` when first seen.

    The training subprocess prints the run URL to stdout exactly once,
    right after ``wandb.init``. We persist it the first time it shows
    up so the jobs UI can render a "Open in W&B" link without parsing
    log files retroactively.
    """
    payload = ev.payload or {}
    msg = str(payload.get("message", ""))
    match = _WANDB_RUN_URL_RE.search(msg)
    if match is None:
        return
    rec = state.registry.get(job_id)
    if rec is None:
        return
    metadata = dict(rec.metadata or {})
    if metadata.get("wandb_run_url"):
        return
    metadata["wandb_run_url"] = match.group("url").rstrip(",.;")
    rec.metadata = metadata
    state.registry.update(rec)


__all__ = [
    "_TERMINAL_STATES",
    "_archive_workspace",
    "_enqueue_launch",
    "_extract_ckpt_name",
    "_launch_job",
    "_relaunch_job_in_place",
    "_select_backend",
]
