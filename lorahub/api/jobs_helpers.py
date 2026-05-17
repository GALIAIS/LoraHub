"""Job-management helpers shared between the jobs router and websocket layer."""

from __future__ import annotations

import contextlib
import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lorahub.api import scheduler as sched
from lorahub.api import state
from lorahub.api.state import JobRecord, JobState
from lorahub.core.backends.diffusion_pipe.backend import DiffusionPipeBackend
from lorahub.core.backends.kohya.backend import KohyaBackend
from lorahub.core.config.loader import dump_config
from lorahub.core.config.schema import TrainingConfig
from lorahub.core.events import EventType, JsonlEventSink, TrainingEvent

log = logging.getLogger(__name__)

_TERMINAL_STATES = (
    JobState.succeeded,
    JobState.failed,
    JobState.canceled,
    JobState.interrupted,
)

_CHECKPOINT_SUFFIXES = {".safetensors", ".ckpt"}
_SAMPLE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
_LOG_FILENAMES = {"events.jsonl"}
_SKIP_DIR_NAMES = {"_archive", "__pycache__", ".git", ".ipynb_checkpoints"}
_SKIP_SUFFIXES = {".tmp"}


def _absolutise(p: Path | str | None, base: Path) -> Path | None:
    """Resolve a recipe-relative path against the project root.

    Training subprocesses (kohya / diffusion-pipe) run with cwd set to the
    backend's own repo, so a recipe path like ``./models/foo.safetensors``
    would otherwise be looked up under ``diffusion-pipe/models/`` instead
    of the lorahub project root. We resolve paths against ``base`` (the
    API server's cwd, which is the lorahub project root) before handing
    the cfg to the compiler.
    """
    if p is None:
        return None
    path = Path(str(p)).expanduser()
    if path.is_absolute():
        return path
    return (base / path).resolve()


def _normalize_recipe_paths(cfg: TrainingConfig, base: Path | None = None) -> TrainingConfig:
    """Make every path field in `cfg` absolute, anchored at `base`.

    Mutates a *copy* of the cfg (Pydantic models are effectively
    mutable; we still touch fields in place but only after the cfg
    snapshot has been captured for persistence by callers that care).
    """
    base_dir = (base or Path.cwd()).resolve()

    cfg.base_model.checkpoint = _absolutise(cfg.base_model.checkpoint, base_dir)  # type: ignore[assignment]
    if cfg.base_model.vae is not None:
        cfg.base_model.vae = _absolutise(cfg.base_model.vae, base_dir)
    paths = cfg.base_model.arch_paths
    for fname in (
        "clip_l", "clip_g", "t5xxl", "ae", "transformer", "text_encoder",
        "llm", "byt5", "qwen3", "t5_tokenizer", "llm_adapter",
    ):
        cur = getattr(paths, fname, None)
        if cur is not None:
            setattr(paths, fname, _absolutise(cur, base_dir))

    cfg.dataset.source = _absolutise(cfg.dataset.source, base_dir)  # type: ignore[assignment]
    if cfg.dataset.conditioning_dir is not None:
        cfg.dataset.conditioning_dir = _absolutise(cfg.dataset.conditioning_dir, base_dir)
    if cfg.dataset.reg_source is not None:
        cfg.dataset.reg_source = _absolutise(cfg.dataset.reg_source, base_dir)
    for sub in cfg.dataset.subsets:
        sub.path = _absolutise(sub.path, base_dir)  # type: ignore[assignment]
        if sub.mask_path is not None:
            sub.mask_path = _absolutise(sub.mask_path, base_dir)

    if cfg.output.output_dir is not None:
        cfg.output.output_dir = _absolutise(cfg.output.output_dir, base_dir)

    # Free-form dp model_paths bag — every value is a path string.
    if cfg.backend.diffusion_pipe is not None:
        mp = cfg.backend.diffusion_pipe.model_paths
        if mp:
            cfg.backend.diffusion_pipe.model_paths = {
                k: str(_absolutise(v, base_dir)) for k, v in mp.items()
            }

    if cfg.network.init_from is not None:
        cfg.network.init_from = _absolutise(cfg.network.init_from, base_dir)
    if cfg.network.dim_from_weights is not None:
        cfg.network.dim_from_weights = _absolutise(cfg.network.dim_from_weights, base_dir)
    cfg.network.base_weights = [
        _absolutise(p, base_dir) for p in cfg.network.base_weights  # type: ignore[misc]
    ]

    if cfg.resume.resume_from is not None:
        cfg.resume.resume_from = _absolutise(cfg.resume.resume_from, base_dir)

    if cfg.sampling.prompts_file is not None:
        cfg.sampling.prompts_file = _absolutise(cfg.sampling.prompts_file, base_dir)

    return cfg

# Cap on the number of loss points returned by /metrics. Anything beyond this
# gets uniformly downsampled so the response stays bounded for very long runs.
_METRICS_MAX_POINTS = 5000
# Threshold above which we trigger downsampling. Below this we just return
# every point — keeps the common case lossless.
_METRICS_DOWNSAMPLE_THRESHOLD = 50_000


def _classify_artifact(rel: Path) -> str:
    name = rel.name
    suffix = rel.suffix.lower()
    if name in _LOG_FILENAMES:
        return "logs"
    if suffix in _CHECKPOINT_SUFFIXES:
        return "checkpoints"
    if suffix in _SAMPLE_SUFFIXES:
        return "samples"
    return "other"


def _list_workspace_files(workspace: Path) -> dict[str, list[dict[str, Any]]]:
    """Walk `workspace` and group files into checkpoints/samples/logs/other.

    Skips archive subtrees, pycache, and `.tmp` scratch files. Paths in the
    returned dicts are POSIX-style and relative to `workspace` so the frontend
    can build clean URLs without worrying about platform separators.
    """
    buckets: dict[str, list[dict[str, Any]]] = {
        "checkpoints": [],
        "samples": [],
        "logs": [],
        "other": [],
    }
    if not workspace.is_dir():
        return buckets

    for path in workspace.rglob("*"):
        if not path.is_file():
            continue
        # Skip files anywhere under a skipped directory or with a scratch suffix.
        rel = path.relative_to(workspace)
        if any(part in _SKIP_DIR_NAMES for part in rel.parts[:-1]):
            continue
        if path.suffix.lower() in _SKIP_SUFFIXES:
            continue

        try:
            stat = path.stat()
        except OSError:
            continue

        bucket = _classify_artifact(rel)
        buckets[bucket].append(
            {
                "path": rel.as_posix(),
                "size_bytes": stat.st_size,
                "modified_at": stat.st_mtime,
            }
        )

    for entries in buckets.values():
        entries.sort(key=lambda e: e["path"])
    return buckets


def _resolve_workspace_file(workspace: Path, rel: str) -> Path:
    """Resolve `rel` against `workspace`, blocking traversal.

    Raises `ValueError` if the resolved path would escape `workspace` or if
    `rel` is empty / absolute. The caller maps that to a 400.
    """
    if not rel:
        raise ValueError("path is required")
    candidate = Path(rel)
    if candidate.is_absolute():
        raise ValueError("path must be workspace-relative")

    workspace_resolved = workspace.resolve()
    target = (workspace_resolved / candidate).resolve()
    try:
        target.relative_to(workspace_resolved)
    except ValueError as exc:
        raise ValueError("path escapes workspace") from exc
    return target


def _media_type_for(path: Path) -> tuple[str, str]:
    """Return (media_type, content_disposition) for a workspace artifact.

    Images render inline so the frontend can preview samples directly; every
    other artifact downloads as an attachment.
    """
    suffix = path.suffix.lower()
    image_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }
    if suffix in image_types:
        return image_types[suffix], "inline"
    return "application/octet-stream", "attachment"


def _read_metrics(workspace: Path) -> dict[str, Any]:
    """Parse `events.jsonl` into chartable time-series data.

    Lines that fail to parse (truncated writes, partial flushes, manual edits)
    are skipped silently — one bad line should never sink the whole endpoint.
    Step series longer than `_METRICS_DOWNSAMPLE_THRESHOLD` are uniformly
    downsampled to ~`_METRICS_MAX_POINTS` points (plus the first and last)
    so the response stays bounded for marathon runs.
    """
    log = workspace / "events.jsonl"
    empty: dict[str, Any] = {
        "loss": [],
        "val_loss": [],
        "epochs": [],
        "checkpoints": [],
        "samples": [],
        "gpu_samples": [],
        "first_step_ts": None,
        "last_step_ts": None,
        "duration_s": None,
        "overfit_signal": _empty_overfit_signal(),
    }
    if not log.is_file():
        return empty

    loss: list[dict[str, Any]] = []
    val_loss: list[dict[str, Any]] = []
    epochs: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    gpu_samples: list[dict[str, Any]] = []
    epoch_counter = 0

    with log.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                etype = row.get("type")
                payload = row.get("payload") or {}
                ts = row.get("timestamp")
            except (json.JSONDecodeError, AttributeError, TypeError):
                continue

            if etype == EventType.step.value:
                if "loss" in payload:
                    loss.append(
                        {
                            "step": payload.get("step"),
                            "epoch": epoch_counter,
                            "loss": payload.get("loss"),
                            "ts": ts,
                        }
                    )
            elif etype == EventType.epoch_end.value:
                epoch_counter += 1
                epochs.append({"epoch": payload.get("epoch"), "ts": ts})
            elif etype == EventType.validation.value:
                if "val_loss" in payload:
                    entry: dict[str, Any] = {
                        # Fall back to the running epoch counter when sd-scripts
                        # forgot to print one on the same line.
                        "epoch": payload.get("epoch", epoch_counter),
                        "val_loss": payload.get("val_loss"),
                        "ts": ts,
                    }
                    if "step" in payload:
                        entry["step"] = payload.get("step")
                    val_loss.append(entry)
            elif etype == EventType.checkpoint_saved.value:
                checkpoints.append(
                    {
                        "path": payload.get("path"),
                        "step": payload.get("step"),
                        "ts": ts,
                    }
                )
            elif etype == EventType.sample_ready.value:
                samples.append({"path": payload.get("path"), "ts": ts})
            elif etype == EventType.gpu_sample.value:
                gpu_samples.append(
                    {
                        "gpu_index": payload.get("gpu_index"),
                        "util_percent": payload.get("util_percent"),
                        "vram_used_mib": payload.get("vram_used_mib"),
                        "vram_total_mib": payload.get("vram_total_mib"),
                        "temperature_c": payload.get("temperature_c"),
                        "ts": ts,
                    }
                )

    first_ts = loss[0]["ts"] if loss else None
    last_ts = loss[-1]["ts"] if loss else None
    duration = (
        last_ts - first_ts
        if first_ts is not None and last_ts is not None
        else None
    )

    if len(loss) > _METRICS_DOWNSAMPLE_THRESHOLD:
        loss = _downsample(loss, _METRICS_MAX_POINTS)
    if len(gpu_samples) > _METRICS_DOWNSAMPLE_THRESHOLD:
        gpu_samples = _downsample(gpu_samples, _METRICS_MAX_POINTS)

    overfit_signal = _compute_overfit_signal(loss, val_loss)

    return {
        "loss": loss,
        "val_loss": val_loss,
        "epochs": epochs,
        "checkpoints": checkpoints,
        "samples": samples,
        "gpu_samples": gpu_samples,
        "first_step_ts": first_ts,
        "last_step_ts": last_ts,
        "duration_s": duration,
        "overfit_signal": overfit_signal,
    }


def _empty_overfit_signal() -> dict[str, Any]:
    return {
        "latest_train": None,
        "latest_val": None,
        "gap": None,
        "trend": None,
    }


# Slope thresholds (loss units per index) used to bucket the heuristic.
# A series whose absolute slope falls below this is considered "flat".
_OVERFIT_FLAT_EPS = 1e-4


def _compute_overfit_signal(
    loss: list[dict[str, Any]],
    val_loss: list[dict[str, Any]],
) -> dict[str, Any]:
    """Crude overfit detector that compares train vs val loss slopes.

    Heuristic (only fires when we have >= 2 val points and >= 2 train points):
        * `val_slope > +eps` and `train_slope < -eps`  ->  "overfitting"
        * `val_slope < -eps` and `train_slope < -eps`  ->  "improving"
        * everything else (mixed, both flat, etc.)     ->  "flat"

    `eps` is `_OVERFIT_FLAT_EPS = 1e-4` loss units per sample index. Slopes
    are computed over the *last 3 points* of each series (or fewer if the
    history is shorter) using a simple linear fit through the means — good
    enough for an early-warning UI badge without dragging in numpy.
    """
    out = _empty_overfit_signal()

    train_points = [p["loss"] for p in loss if isinstance(p.get("loss"), (int, float))]
    val_points = [
        p["val_loss"] for p in val_loss if isinstance(p.get("val_loss"), (int, float))
    ]

    if train_points:
        out["latest_train"] = float(train_points[-1])
    if val_points:
        out["latest_val"] = float(val_points[-1])
    if train_points and val_points:
        out["gap"] = float(val_points[-1] - train_points[-1])

    if len(val_points) < 2 or len(train_points) < 2:
        return out

    train_slope = _tail_slope(train_points, window=3)
    val_slope = _tail_slope(val_points, window=3)

    if val_slope > _OVERFIT_FLAT_EPS and train_slope < -_OVERFIT_FLAT_EPS:
        out["trend"] = "overfitting"
    elif val_slope < -_OVERFIT_FLAT_EPS and train_slope < -_OVERFIT_FLAT_EPS:
        out["trend"] = "improving"
    else:
        out["trend"] = "flat"
    return out


def _tail_slope(values: list[float], *, window: int) -> float:
    """Return a simple least-squares slope over the last `window` samples.

    Indices are treated as the x-axis (0..n-1) so the result is in "loss
    units per sample". The classic closed-form is fine here — N is at most 3.
    """
    tail = [float(v) for v in values[-window:]]
    n = len(tail)
    if n < 2:
        return 0.0
    mean_x = (n - 1) / 2.0
    mean_y = sum(tail) / n
    num = sum((i - mean_x) * (tail[i] - mean_y) for i in range(n))
    den = sum((i - mean_x) ** 2 for i in range(n))
    if den == 0:
        return 0.0
    return num / den


def _downsample(points: list[dict[str, Any]], target: int) -> list[dict[str, Any]]:
    """Uniformly sample `points` to ~`target` entries, keeping the endpoints.

    For a chart, we want to preserve the visual shape of the curve. Picking
    every Nth point with the first and last forced in is good enough — the
    target only needs to be approximate.
    """
    n = len(points)
    if n <= target:
        return points
    step = n / target
    indices = sorted({0, n - 1, *(int(i * step) for i in range(target))})
    return [points[i] for i in indices if 0 <= i < n]


def _job_events(job: state.JobRecord, limit: int | None = None) -> list[TrainingEvent]:
    events = list(job.events)
    if not events:
        event_log = job.workspace / "events.jsonl"
        if event_log.is_file():
            with contextlib.suppress(Exception):
                events = list(JsonlEventSink.replay(event_log))
    if limit is not None:
        events = events[-max(limit, 0) :]
    return events


def _select_backend(cfg: TrainingConfig):  # type: ignore[no-untyped-def]
    """Pick the training backend implementation that the config asks for."""
    backend_type = cfg.backend.type
    if backend_type == "kohya":
        return KohyaBackend()
    if backend_type == "diffusion-pipe":
        return DiffusionPipeBackend()
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

    # Normalise every recipe-relative path (./models/..., ./datasets/...)
    # to absolute so the training subprocess can find them regardless of
    # the cwd the backend launches it from (dp uses its own repo dir).
    _normalize_recipe_paths(cfg)

    snapshot = cfg.model_dump(mode="json")
    job = state.registry.create(workspace=workspace, config_snapshot=snapshot)
    if metadata is not None:
        job.metadata = metadata
        state.registry.update(job)
    dump_config(cfg, workspace / "config.yaml")

    _enqueue_launch(job, cfg, extra_argv=extra_argv)
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

    # Plain rerun: wipe stale events so the user sees only the fresh
    # attempt. /resume passes extra_argv, where preserving history is
    # the whole point — leave its log alone.
    if not extra_argv:
        job.events.clear()
        event_log = workspace / "events.jsonl"
        if event_log.is_file():
            event_log.unlink()

    snapshot = cfg.model_dump(mode="json")
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

    _enqueue_launch(job, cfg, extra_argv=extra_argv)
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
    backend = _select_backend(cfg)
    sink = JsonlEventSink(workspace / "events.jsonl")

    def on_event(ev: TrainingEvent) -> None:
        sink(ev)
        state.registry.record_event(job.id, ev)
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
            with contextlib.suppress(Exception):
                sink.__exit__(None, None, None)
            return
        j = state.registry.get(job.id)
        if j is not None:
            j.handle = handle
            j.pid = handle.pid
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
        try:
            handle.wait(timeout=None)
        except Exception:  # noqa: BLE001
            log.exception("worker wait() failed for job %s", job.id)
        finally:
            sampler_stop.set()

    sched.scheduler.submit(job.id, task)


def _gpu_sampler_loop(
    job_id: str,
    slot: int,
    on_event: Any,
    stop_evt: threading.Event,
) -> None:
    """Emit a `gpu_sample` event every ~5s while the job is running.

    Snapshot comes from the same nvidia-smi path the dashboard uses, so
    no extra dependency. Failures are swallowed (the trend chart is a
    nice-to-have; we never want it to crash a training run).
    """
    from lorahub.api.system_stats import _collect_nvidia_gpus  # noqa: PLC0415
    from lorahub.core.events import EventType, TrainingEvent  # noqa: PLC0415

    interval = 5.0
    while not stop_evt.wait(interval):
        try:
            gpus = _collect_nvidia_gpus()
        except Exception:  # noqa: BLE001
            continue
        if not gpus or slot >= len(gpus):
            continue
        g = gpus[slot]
        try:
            on_event(
                TrainingEvent(
                    type=EventType.gpu_sample,
                    job_id=job_id,
                    payload={
                        "gpu_index": slot,
                        "util_percent": g.utilization_percent,
                        "vram_used_mib": (
                            int(g.memory_used_bytes // (1024 * 1024))
                            if getattr(g, "memory_used_bytes", None) is not None
                            else None
                        ),
                        "vram_total_mib": (
                            int(g.memory_total_bytes // (1024 * 1024))
                            if getattr(g, "memory_total_bytes", None) is not None
                            else None
                        ),
                        "temperature_c": g.temperature_c,
                    },
                )
            )
        except Exception:  # noqa: BLE001
            continue


def _find_latest_state_dir(workspace: Path) -> Path | None:
    """Most recently modified `*-state*` directory under the job workspace.

    sd-scripts writes state directories like `<output_name>-state` at the
    end of a run and `<output_name>-state-step<N>` at each interval. We
    look under the workspace tree (kohya defaults `--output_dir` to the
    workspace) and pick the freshest match — that is what `/resume`
    feeds into `--resume=<dir>`.
    """
    if not workspace.is_dir():
        return None
    candidates = [p for p in workspace.rglob("*-state*") if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _find_latest_safetensors(workspace: Path) -> Path | None:
    """Most recently modified `*.safetensors` under the workspace tree."""
    if not workspace.is_dir():
        return None
    files = list(workspace.rglob("*.safetensors"))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


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
    """Most recent dp run directory under the configured output_dir.

    diffusion-pipe writes one timestamped subdirectory per run under
    `output_dir/`. Each contains a `latest` text file pointing at the most
    recent `global_stepN/` checkpoint. We pick the lex-max basename among
    candidates that look complete (have both `latest` and at least one
    `global_step*` folder), matching dp's own selection in
    `train.get_most_recent_run_dir`.
    """
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
    # Match dp's lex-sort: timestamp dir names sort correctly that way.
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
        # Pin the resumed run to the same output_dir so dp picks the
        # same run_dir back up. Stored as an absolute string so
        # subsequent re-validation through pydantic's Path field is happy.
        cfg_overrides={"output.output_dir": str(out_dir)},
    )


def _dispatch_resume_spec(cfg: TrainingConfig, workspace: Path) -> ResumeSpec:
    """Dispatch to the per-backend resume helper based on `cfg.backend.type`."""
    backend_type = cfg.backend.type
    if backend_type == "kohya":
        return _kohya_resume_spec(workspace)
    if backend_type == "diffusion-pipe":
        return _dp_resume_spec(cfg, workspace)
    raise ResumeNotReady(
        f"resume not implemented for backend.type={backend_type!r}"
    )


def _apply_cfg_overrides(cfg: TrainingConfig, overrides: dict[str, Any]) -> TrainingConfig:
    """Apply a flat dot-path override mapping onto a validated TrainingConfig.

    Re-dumps the config to a dict, walks the dot path to set each value,
    then re-validates. Returns a fresh `TrainingConfig` so the caller's
    snapshot stays untouched. Empty overrides short-circuit.
    """
    if not overrides:
        return cfg
    data = cfg.model_dump(mode="json")
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
            _relaunch_job_in_place(
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
            _enqueue_launch(job, cfg)
        except Exception:  # noqa: BLE001
            log.exception("requeue: failed to re-enqueue queued job %s", job.id)
            continue
        requeued += 1
        log.info("requeue: re-submitted queued job %s to scheduler", job.id)
    return requeued


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

    # Avoid clobbering an existing archive entry from a previous archive of the
    # same workspace name — append a counter until we find a free slot.
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
