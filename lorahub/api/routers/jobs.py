"""Job CRUD plus rerun / reveal / archive."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from lorahub.api import state
from lorahub.api.jobs_helpers import (
    _TERMINAL_STATES,
    _archive_workspace,
    _job_events,
    _launch_job,
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
