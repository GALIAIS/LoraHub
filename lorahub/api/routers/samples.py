"""Cross-job sample image gallery.

Aggregates the ``samples`` bucket from every registered job's workspace into
a single time-ordered feed so the UI can show "all the pretty pictures
training has produced lately" without the user having to open each job tab
individually.

Each item carries the job id + workspace context plus a ``raw_url`` that
points back at the existing per-job inline image endpoint. We don't generate
thumbnails here — the per-job ``/files/raw`` route already serves images
``inline`` so a plain ``<img src=raw_url>`` works in the browser, and any
real path safety check happens there.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Query

from lorahub.api import state
from lorahub.api.jobs_helpers import _list_workspace_files

router = APIRouter(prefix="/api")

# Hard ceiling on per-request page size. The aggregation walks every job's
# workspace, so a reckless ``limit=10_000_000`` would blow up the response;
# 500 is more than the UI ever needs in one viewport.
_MAX_LIMIT = 500


def _split_csv(raw: str | None) -> list[str]:
    """Split a comma-separated job_ids query param into a clean list.

    Empty / whitespace entries are dropped so callers can pass things like
    ``,job1,,job2`` without surprises.
    """
    if not raw:
        return []
    return [piece.strip() for piece in raw.split(",") if piece.strip()]


@router.get("/samples")
def list_samples(
    limit: int = Query(default=200, ge=1, le=_MAX_LIMIT),
    offset: int = Query(default=0, ge=0),
    job_ids: str | None = Query(default=None),
) -> dict[str, Any]:
    """Return sample images aggregated across every registered job.

    Filtering:
      ``job_ids`` is a comma-separated list of job ids — when present only
      those jobs are scanned and unknown ids return 404 so the UI can flag
      stale filter chips immediately.

    Ordering:
      Items are sorted by ``modified_at`` descending so the freshest
      samples float to the top of the gallery.
    """
    selected_ids = _split_csv(job_ids)
    if selected_ids:
        jobs = []
        for jid in selected_ids:
            job = state.registry.get(jid)
            if job is None:
                raise HTTPException(
                    status_code=404, detail=f"job not found: {jid}"
                )
            jobs.append(job)
    else:
        jobs = state.registry.list()

    items: list[dict[str, Any]] = []
    for job in jobs:
        snapshot = job.recipe_snapshot or {}
        recipe_name: str | None = None
        output = snapshot.get("output")
        if isinstance(output, dict):
            raw_name = output.get("name")
            if isinstance(raw_name, str) and raw_name.strip():
                recipe_name = raw_name.strip()

        buckets = _list_workspace_files(job.workspace)
        for entry in buckets.get("samples", []):
            rel = entry["path"]
            items.append(
                {
                    "job_id": job.id,
                    "job_name": job.workspace.name,
                    "recipe_name": recipe_name,
                    "path": rel,
                    "size_bytes": entry["size_bytes"],
                    "modified_at": entry["modified_at"],
                    "raw_url": (
                        f"/api/jobs/{job.id}/files/raw?path={quote(rel, safe='')}"
                    ),
                }
            )

    items.sort(key=lambda it: it["modified_at"], reverse=True)
    total = len(items)
    page = items[offset : offset + limit]
    return {
        "items": page,
        "total": total,
        "limit": limit,
        "offset": offset,
    }
