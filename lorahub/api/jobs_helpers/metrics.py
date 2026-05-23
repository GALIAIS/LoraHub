"""Workspace artifact listing + ``events.jsonl`` parsing.

These helpers are read-only: they walk a job workspace directory and
turn what they find into shapes the frontend / API routers consume.
``_read_metrics`` parses ``events.jsonl`` into the bounded chartable
payload the dashboard renders; the rest classify / resolve files for
the file picker and per-artifact download endpoint.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any

from lorahub.api import state
from lorahub.core.events import EventType, JsonlEventSink, TrainingEvent

_CHECKPOINT_SUFFIXES = {".safetensors", ".ckpt"}
_SAMPLE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
_LOG_FILENAMES = {"events.jsonl"}
# Directories whose contents must not be surfaced in the file picker
# / samples gallery. ``post_image_dataset`` is anima_lora's preprocess
# scratch — it holds VAE-resized copies of the user's dataset (PNGs)
# plus latent / TE caches; without this skip the analysis tab's
# samples gallery showed the resized dataset images as if they were
# trainer-generated previews. Latent / TE caches don't have image
# extensions so they fall through to "other" anyway.
_SKIP_DIR_NAMES = {
    "_archive",
    "__pycache__",
    ".git",
    ".ipynb_checkpoints",
    "post_image_dataset",
}
_SKIP_SUFFIXES = {".tmp"}

# Cap on the number of loss points returned by /metrics. Anything beyond this
# gets uniformly downsampled so the response stays bounded for very long runs.
_METRICS_MAX_POINTS = 5000
# Threshold above which we trigger downsampling. Below this we just return
# every point — keeps the common case lossless.
_METRICS_DOWNSAMPLE_THRESHOLD = 50_000

# Slope thresholds (loss units per index) used to bucket the heuristic.
# A series whose absolute slope falls below this is considered "flat".
_OVERFIT_FLAT_EPS = 1e-4


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
    log_path = workspace / "events.jsonl"
    empty: dict[str, Any] = {
        "loss": [],
        "val_loss": [],
        "epochs": [],
        "checkpoints": [],
        "samples": [],
        "gpu_samples": [],
        "lora_spectrum": [],
        "forgetting_probe": [],
        "first_step_ts": None,
        "last_step_ts": None,
        "duration_s": None,
        "total_steps": None,
        "overfit_signal": _empty_overfit_signal(),
    }
    if not log_path.is_file():
        return empty

    loss: list[dict[str, Any]] = []
    val_loss: list[dict[str, Any]] = []
    epochs: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    gpu_samples: list[dict[str, Any]] = []
    lora_spectrum: list[dict[str, Any]] = []
    forgetting_probe: list[dict[str, Any]] = []
    epoch_counter = 0
    # Trainer-reported total step count. We track the latest non-zero
    # value seen on a `step` event so the front-end has a single
    # authoritative denominator to use across the overview / summary /
    # analysis tabs (which used to disagree because each rebuilt this
    # number from a different source — config maxSteps vs step.payload
    # vs config-derived fallback).
    total_steps: int | None = None

    with log_path.open("r", encoding="utf-8") as fh:
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
                    raw_epoch = payload.get("epoch")
                    epoch_value = (
                        int(raw_epoch)
                        if isinstance(raw_epoch, (int, float))
                        else max(epoch_counter, 1)
                    )
                    point: dict[str, Any] = {
                        "step": payload.get("step"),
                        "epoch": epoch_value,
                        "loss": payload.get("loss"),
                        "ts": ts,
                    }
                    # dp emits lr/iter_time/samples_per_sec alongside loss;
                    # forward them so the front-end can render LR + throughput
                    # charts without a second request.
                    for k_src, k_out in (
                        ("lr", "lr"),
                        ("iter_time_s", "iter_time_s"),
                        ("samples_per_sec", "samples_per_sec"),
                    ):
                        if k_src in payload and isinstance(
                            payload[k_src], (int, float)
                        ):
                            point[k_out] = payload[k_src]
                    loss.append(point)
                    raw_total = payload.get("total_steps")
                    if isinstance(raw_total, (int, float)) and raw_total > 0:
                        total_steps = int(raw_total)
            elif etype == EventType.epoch_end.value:
                epoch_counter += 1
                epochs.append({"epoch": payload.get("epoch"), "ts": ts})
            elif etype == EventType.validation.value:
                if "val_loss" in payload:
                    entry: dict[str, Any] = {
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
            elif etype == EventType.lora_spectrum.value:
                lora_spectrum.append(
                    {
                        "step": payload.get("step"),
                        "checkpoint": payload.get("checkpoint"),
                        "layers": payload.get("layers"),
                        "effective_rank": payload.get("effective_rank"),
                        "top1_energy": payload.get("top1_energy"),
                        "fro_norm": payload.get("fro_norm"),
                        "ts": ts,
                    }
                )
            elif etype == EventType.forgetting_probe.value:
                forgetting_probe.append(
                    {
                        "step": payload.get("step"),
                        "checkpoint": payload.get("checkpoint"),
                        "preserved": payload.get("preserved"),
                        "samples": payload.get("samples"),
                        "image_path": payload.get("image_path"),
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
        "lora_spectrum": lora_spectrum,
        "forgetting_probe": forgetting_probe,
        "first_step_ts": first_ts,
        "last_step_ts": last_ts,
        "duration_s": duration,
        "total_steps": total_steps,
        "overfit_signal": overfit_signal,
    }


def _empty_overfit_signal() -> dict[str, Any]:
    return {
        "latest_train": None,
        "latest_val": None,
        "gap": None,
        "trend": None,
    }


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
    """Return a simple least-squares slope over the last `window` samples."""
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
    """Uniformly sample `points` to ~`target` entries, keeping the endpoints."""
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


__all__ = [
    "_LOG_FILENAMES",
    "_SKIP_DIR_NAMES",
    "_SKIP_SUFFIXES",
    "_classify_artifact",
    "_compute_overfit_signal",
    "_downsample",
    "_empty_overfit_signal",
    "_job_events",
    "_list_workspace_files",
    "_media_type_for",
    "_read_metrics",
    "_resolve_workspace_file",
    "_tail_slope",
]
