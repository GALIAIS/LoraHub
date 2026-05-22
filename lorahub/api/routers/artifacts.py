"""Training-artifact management.

Spotlights the trainer outputs (LoRA `.safetensors`, kohya state
directories, sample images) across every job, so the user has a
single page to download / clean up / re-locate weights without
hopping between job detail tabs.

Routes:

* ``GET  /api/artifacts``                          — flat list of every
  job's artifacts grouped by job, with size / mtime / role.
* ``GET  /api/artifacts/{job_id}/zip``             — stream the job's
  workspace as a single ``.zip``. ``include`` query (csv) picks
  buckets: ``checkpoints,samples,logs,other``. Default is
  ``checkpoints`` only — the LoRA weights are what users actually
  ship, the rest blow up the archive size needlessly.
* ``DELETE /api/artifacts/{job_id}/file?path=...`` — drop a single
  workspace-relative file. Same path-traversal guards as
  ``/jobs/{id}/files/raw``.
* ``DELETE /api/artifacts/{job_id}/workspace``     — physically remove
  the entire workspace tree. Differs from
  ``DELETE /api/jobs/{id}?archive=true`` (which moves to
  ``_archive/``) — this one is unrecoverable. Refuses non-terminal
  jobs.
"""

from __future__ import annotations

import io
import logging
import shutil
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from lorahub.api import state
from lorahub.api.jobs_helpers import (
    _list_workspace_files,
    _resolve_workspace_file,
)
from lorahub.api.state import JobState

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

_TERMINAL_STATES = (
    JobState.succeeded,
    JobState.failed,
    JobState.canceled,
    JobState.interrupted,
)

# Buckets the zip endpoint understands. ``checkpoints`` is the
# default-and-typical case; ``samples`` is useful when the user
# wants the preview grid alongside the LoRA; ``logs`` and ``other``
# are the long-tail.
_VALID_BUCKETS: frozenset[str] = frozenset({"checkpoints", "samples", "logs", "other"})


def _job_artifact_summary(job_id: str, workspace: Path) -> dict[str, Any]:
    """Reduce a workspace listing to the artifacts page's row shape."""
    buckets = _list_workspace_files(workspace)
    checkpoints = buckets.get("checkpoints", [])
    samples = buckets.get("samples", [])
    total_bytes = (
        sum(int(e.get("size_bytes") or 0) for e in checkpoints)
        + sum(int(e.get("size_bytes") or 0) for e in samples)
        + sum(int(e.get("size_bytes") or 0) for e in buckets.get("logs", []))
        + sum(int(e.get("size_bytes") or 0) for e in buckets.get("other", []))
    )
    return {
        "job_id": job_id,
        "workspace": str(workspace),
        "exists": workspace.is_dir(),
        "checkpoints": checkpoints,
        "samples": samples,
        "total_bytes": total_bytes,
        "checkpoint_count": len(checkpoints),
        "sample_count": len(samples),
    }


@router.get("/artifacts")
def list_artifacts() -> dict[str, Any]:
    """Aggregate every job's artifact summary in one response."""
    rows: list[dict[str, Any]] = []
    for job in state.registry.list():
        summary = _job_artifact_summary(job.id, job.workspace)
        # Stamp a few job-level fields the UI needs to render the row
        # without making a second /jobs/{id} call per artifact.
        summary["state"] = job.state.value
        summary["created_at"] = (
            job.created_at.isoformat() if job.created_at else None
        )
        summary["finished_at"] = (
            job.finished_at.isoformat() if job.finished_at else None
        )
        cfg = job.config_snapshot or {}
        output = cfg.get("output") if isinstance(cfg, dict) else None
        if isinstance(output, dict):
            summary["output_name"] = output.get("name")
        rows.append(summary)
    rows.sort(
        key=lambda r: r.get("finished_at") or r.get("created_at") or "",
        reverse=True,
    )
    return {"jobs": rows}


def _iter_zip_chunks(
    workspace: Path,
    relpaths: Iterable[Path],
) -> Iterable[bytes]:
    """Stream a ZIP archive of the listed files chunk-by-chunk.

    Goes through an in-memory ``BytesIO`` buffer that we drain on
    every yield — ``zipfile`` doesn't have a streaming API, but with
    ``ZIP_STORED`` (no compression — the .safetensors are already
    binary, gzip wouldn't shrink them anyway) we can append + flush
    one file at a time without needing the full archive resident.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as zf:
        for rel in relpaths:
            full = (workspace / rel).resolve()
            try:
                full.relative_to(workspace.resolve())
            except ValueError:
                continue
            if not full.is_file():
                continue
            arcname = rel.as_posix()
            try:
                with full.open("rb") as src:
                    with zf.open(zipfile.ZipInfo(arcname), mode="w") as dst:
                        while True:
                            chunk = src.read(1024 * 1024)
                            if not chunk:
                                break
                            dst.write(chunk)
            except OSError as exc:
                log.warning("artifacts zip: skipping %s — %s", full, exc)
                continue
            # Drain the buffer between files so the caller doesn't
            # accumulate the whole archive in memory.
            data = buffer.getvalue()
            if data:
                yield data
                buffer.seek(0)
                buffer.truncate(0)
    tail = buffer.getvalue()
    if tail:
        yield tail


@router.get("/artifacts/{job_id}/zip")
def download_zip(
    job_id: str,
    include: str = Query(
        default="checkpoints",
        description="Comma-separated buckets to include: checkpoints,samples,logs,other.",
    ),
) -> StreamingResponse:
    """Stream the job workspace as a ZIP, filtered to the requested buckets.

    Default is ``checkpoints`` only — the LoRA weights are what people
    actually ship; samples and logs balloon the archive without
    matching the typical user intent of "give me the model file".
    """
    job = state.registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if not job.workspace.is_dir():
        raise HTTPException(status_code=404, detail="workspace missing on disk")

    requested = {b.strip() for b in include.split(",") if b.strip()}
    invalid = requested - _VALID_BUCKETS
    if invalid:
        raise HTTPException(
            status_code=422,
            detail=f"unknown buckets: {sorted(invalid)}; valid={sorted(_VALID_BUCKETS)}",
        )
    if not requested:
        requested = {"checkpoints"}

    buckets = _list_workspace_files(job.workspace)
    rels: list[Path] = []
    for bucket in requested:
        for entry in buckets.get(bucket, []):
            p = entry.get("path")
            if isinstance(p, str) and p:
                rels.append(Path(p))

    if not rels:
        raise HTTPException(
            status_code=404,
            detail=f"no files matched the requested buckets ({sorted(requested)})",
        )

    output_name = "lorahub-artifacts"
    cfg = job.config_snapshot or {}
    if isinstance(cfg, dict):
        out = cfg.get("output")
        if isinstance(out, dict) and isinstance(out.get("name"), str):
            output_name = out["name"]
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    filename = f"{output_name}_{job_id[-8:]}_{timestamp}.zip"

    return StreamingResponse(
        _iter_zip_chunks(job.workspace, rels),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Accel-Buffering": "no",
        },
    )


@router.delete("/artifacts/{job_id}/file")
def delete_file(job_id: str, path: str) -> dict[str, Any]:
    """Delete one workspace-relative file. Path-traversal protected.

    Returns ``{"deleted": <abs path>, "size_bytes": <freed>}``. 404
    if the file isn't there (idempotent in spirit but the explicit
    error helps the UI show "did you double-click?" feedback).
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

    try:
        size = target.stat().st_size
        target.unlink()
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"unlink failed: {exc}") from exc
    return {"deleted": str(target), "size_bytes": size}


@router.delete("/artifacts/{job_id}/workspace")
def delete_workspace(job_id: str) -> dict[str, Any]:
    """Physically delete the entire workspace tree. Unrecoverable.

    Differs from ``DELETE /api/jobs/{id}?archive=true`` (which moves
    the tree to ``_archive/`` so the user can still recover). Refuses
    non-terminal jobs because ripping the workspace out from under a
    running trainer would crash the subprocess and leak GPU memory.
    """
    job = state.registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.state not in _TERMINAL_STATES:
        raise HTTPException(
            status_code=409,
            detail=(
                f"job is {job.state.value}; cancel it before deleting the workspace"
            ),
        )
    if not job.workspace.exists():
        # Drop the registry row anyway so the artifacts page doesn't
        # keep showing a phantom entry.
        state.registry.delete(job.id)
        return {"deleted": False, "reason": "workspace missing on disk"}

    try:
        shutil.rmtree(job.workspace)
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"rmtree failed: {exc}",
        ) from exc
    state.registry.delete(job.id)
    return {"deleted": True, "workspace": str(job.workspace)}


__all__ = ["router"]
