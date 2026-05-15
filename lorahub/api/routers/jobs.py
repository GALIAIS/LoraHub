"""Job CRUD plus rerun / reveal / archive."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from lorahub.api import state
from lorahub.api.jobs_helpers import (
    _TERMINAL_STATES,
    _archive_workspace,
    _job_events,
    _launch_job,
    _list_workspace_files,
    _media_type_for,
    _read_metrics,
    _resolve_workspace_file,
)
from lorahub.api.state import JobState
from lorahub.core.config.schema import RecipeConfig

router = APIRouter(prefix="/api")


class CreateJobRequest(BaseModel):
    recipe: dict[str, Any]
    workspace: str | None = None


@router.get("/jobs")
def list_jobs() -> dict[str, Any]:
    return {"jobs": [j.to_summary() for j in state.registry.list()]}


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    job = state.registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job.to_summary()


@router.get("/jobs/{job_id}/events")
def get_recent_events(job_id: str, limit: int = 100) -> dict[str, Any]:
    job = state.registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    events = _job_events(job, limit)
    return {"events": [e.to_dict() for e in events]}


@router.post("/jobs", status_code=202)
def create_job(req: CreateJobRequest) -> dict[str, Any]:
    try:
        cfg = RecipeConfig.model_validate(req.recipe)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=str(e)) from e

    workspace = Path(req.workspace).resolve() if req.workspace else (
        Path.cwd() / "runs" / cfg.output.name
    ).resolve()
    return _launch_job(cfg, workspace)


@router.post("/jobs/{job_id}/rerun", status_code=202)
def rerun_job(job_id: str) -> dict[str, Any]:
    """Start a fresh job from an existing job's recipe snapshot.

    The original job is left untouched. The new run gets its own workspace
    sibling to the original (suffixed with a short timestamp so the two never
    fight over the same `events.jsonl`).
    """
    job = state.registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")

    try:
        cfg = RecipeConfig.model_validate(job.recipe_snapshot)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=422, detail=f"recipe snapshot is no longer valid: {exc}"
        ) from exc

    base = job.workspace.resolve()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    workspace = (base.parent / f"{base.name}-rerun-{stamp}").resolve()
    return _launch_job(cfg, workspace)


@router.post("/jobs/{job_id}/reveal")
def reveal_job(job_id: str) -> dict[str, Any]:
    """Open the job's workspace directory in the host file browser.

    Local-first tool: the API process is on the user's machine, so we shell out
    to the platform's native file manager (`explorer`, `open`, `xdg-open`).
    Always uses an argv list — never `shell=True` — to avoid command injection
    via the workspace path.
    """
    import subprocess  # noqa: PLC0415
    import sys  # noqa: PLC0415

    job = state.registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")

    workspace = job.workspace
    if not workspace.exists():
        raise HTTPException(
            status_code=409, detail=f"workspace no longer exists: {workspace}"
        )

    if sys.platform == "win32":
        argv = ["explorer", str(workspace)]
    elif sys.platform == "darwin":
        argv = ["open", str(workspace)]
    else:
        argv = ["xdg-open", str(workspace)]

    try:
        subprocess.Popen(argv, close_fds=True)  # noqa: S603
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500, detail=f"file manager not available: {exc}"
        ) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"opened": str(workspace)}


@router.get("/jobs/{job_id}/files")
def list_job_files(job_id: str) -> dict[str, Any]:
    """List training artifacts in the job's workspace, classified by kind.

    Pure read-only inspection — the helper walks the workspace, drops archive
    and scratch entries, and groups the rest into checkpoints / samples /
    logs / other so the dashboard can render distinct sections.
    """
    job = state.registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    buckets = _list_workspace_files(job.workspace)
    return {"workspace": str(job.workspace), **buckets}


@router.get("/jobs/{job_id}/files/raw")
def get_job_file(job_id: str, path: str) -> FileResponse:
    """Stream a single workspace artifact for download or inline preview.

    The caller-supplied `path` is resolved against the job workspace and the
    result is required to stay inside it — anything that resolves outside
    (`..` traversal, absolute paths, symlink escapes) is rejected with 400.
    Images are served `inline` so the frontend can drop them straight into an
    `<img>` tag; everything else is sent as an attachment download.
    """
    job = state.registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")

    try:
        target = _resolve_workspace_file(job.workspace, path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not target.is_file():
        raise HTTPException(status_code=404, detail="file not found")

    media_type, disposition = _media_type_for(target)
    return FileResponse(
        target,
        media_type=media_type,
        filename=target.name,
        content_disposition_type=disposition,
    )


@router.get("/jobs/{job_id}/metrics")
def get_job_metrics(job_id: str) -> dict[str, Any]:
    """Return chartable time-series extracted from `events.jsonl`.

    Empty arrays + null duration when the log doesn't exist yet (the job has
    not started or has not produced events). Long runs are downsampled to
    keep the response bounded.
    """
    job = state.registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return _read_metrics(job.workspace)


@router.delete("/jobs/{job_id}")
def cancel_job(job_id: str, archive: bool = False) -> dict[str, Any]:
    job = state.registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")

    if archive:
        if job.state not in _TERMINAL_STATES:
            raise HTTPException(
                status_code=409,
                detail=f"job is {job.state.value}; cancel before archiving",
            )
        moved, warnings = _archive_workspace(job.workspace, job.id)
        state.registry.delete(job.id)
        return {
            "archived": True,
            "workspace_moved_to": str(moved) if moved is not None else None,
            "warnings": warnings,
        }

    if job.state in _TERMINAL_STATES:
        return job.to_summary()
    job.state = JobState.canceling
    state.registry.update(job)
    if job.handle is not None:
        job.handle.stop(graceful=True)
    return job.to_summary()
