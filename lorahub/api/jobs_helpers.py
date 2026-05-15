"""Job-management helpers shared between the jobs router and websocket layer."""

from __future__ import annotations

import contextlib
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
