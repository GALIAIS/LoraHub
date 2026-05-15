"""Job-management helpers shared between the jobs router and websocket layer."""

from __future__ import annotations

import contextlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lorahub.api import state
from lorahub.api.state import JobState
from lorahub.core.backends.kohya.backend import KohyaBackend
from lorahub.core.config.loader import dump_recipe
from lorahub.core.config.schema import RecipeConfig
from lorahub.core.events import EventType, JsonlEventSink, TrainingEvent

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
        "epochs": [],
        "checkpoints": [],
        "samples": [],
        "first_step_ts": None,
        "last_step_ts": None,
        "duration_s": None,
    }
    if not log.is_file():
        return empty

    loss: list[dict[str, Any]] = []
    epochs: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
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

    first_ts = loss[0]["ts"] if loss else None
    last_ts = loss[-1]["ts"] if loss else None
    duration = (
        last_ts - first_ts
        if first_ts is not None and last_ts is not None
        else None
    )

    if len(loss) > _METRICS_DOWNSAMPLE_THRESHOLD:
        loss = _downsample(loss, _METRICS_MAX_POINTS)

    return {
        "loss": loss,
        "epochs": epochs,
        "checkpoints": checkpoints,
        "samples": samples,
        "first_step_ts": first_ts,
        "last_step_ts": last_ts,
        "duration_s": duration,
    }


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


def _launch_job(cfg: RecipeConfig, workspace: Path) -> dict[str, Any]:
    """Materialize a workspace, register a job, and start the kohya backend.

    Shared by both `POST /jobs` (fresh) and `POST /jobs/{id}/rerun`. The caller
    is responsible for resolving `workspace` (the rerun path needs a fresh dir
    so it doesn't collide with the original run's artifacts).
    """
    workspace.mkdir(parents=True, exist_ok=True)

    snapshot = cfg.model_dump(mode="json")
    job = state.registry.create(workspace=workspace, recipe_snapshot=snapshot)
    dump_recipe(cfg, workspace / "recipe.yaml")

    backend = KohyaBackend()

    sink = JsonlEventSink(workspace / "events.jsonl")
    sink.__enter__()

    def on_event(ev: TrainingEvent) -> None:
        sink(ev)
        state.registry.record_event(job.id, ev)
        if ev.type is EventType.done:
            j = state.registry.get(job.id)
            if j is not None:
                rc = ev.payload.get("returncode")
                j.returncode = rc
                j.state = JobState.succeeded if rc == 0 else JobState.failed
                j.finished_at = datetime.now(UTC)
                state.registry.update(j)
            sink.__exit__(None, None, None)

    handle = backend.launch(cfg, workspace=workspace, on_event=on_event)
    job.handle = handle
    job.pid = handle.pid
    job.state = JobState.running
    job.started_at = datetime.now(UTC)
    state.registry.update(job)

    return job.to_summary()


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
