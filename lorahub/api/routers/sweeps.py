"""Sweep — batch-enqueue a cartesian-product grid over a base recipe.

The sweep endpoint takes one validated base recipe plus N axes, materialises
every cartesian-product variant via :class:`SweepPlan`, and pushes each one
through :func:`_launch_job` so the scheduler runs them serially under the
existing single-slot concurrency model.

Each spawned :class:`JobRecord` is stamped with
``metadata = {"sweep_id": ..., "axis_values": {...}}`` so a later
``GET /api/sweeps/{sweep_id}`` can filter the in-memory registry without
needing a join table. The metadata blob is persisted alongside the rest of
the job row in SQLite, so sweep_id grouping survives a server restart.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import ulid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from lorahub.api import state
from lorahub.api.jobs_helpers import _launch_job
from lorahub.api.state import JobState
from lorahub.core.config.schema import RecipeConfig
from lorahub.core.sweep import (
    SweepAxis,
    SweepError,
    SweepPlan,
    SweepTooLargeError,
)

router = APIRouter(prefix="/api")


class SweepAxisRequest(BaseModel):
    path: str
    values: list[Any] = Field(min_length=1)


class CreateSweepRequest(BaseModel):
    base_recipe: dict[str, Any]
    axes: list[SweepAxisRequest] = Field(min_length=1)
    name_template: str = "{base}-{i:03d}"
    workspace_root: str | None = None


@router.post("/sweeps", status_code=202)
def create_sweep(req: CreateSweepRequest) -> dict[str, Any]:
    """Expand a sweep into N variants, enqueue each one, return the manifest.

    Errors:
      422 — base recipe fails schema validation
      400 — axis path doesn't resolve in base, or grid is too large, or a
            materialised variant fails schema validation (likely caused by
            an axis value that violates a pydantic constraint)
    """
    try:
        RecipeConfig.model_validate(req.base_recipe)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=422, detail=f"base_recipe is invalid: {exc}"
        ) from exc

    plan = SweepPlan(
        base_recipe=req.base_recipe,
        axes=[SweepAxis(path=a.path, values=a.values) for a in req.axes],
        name_template=req.name_template,
    )
    try:
        variants = plan.expand()
    except SweepTooLargeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except SweepError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    sweep_id = str(ulid.new())
    workspace_root = (
        Path(req.workspace_root).resolve()
        if req.workspace_root
        else (Path.cwd() / "runs").resolve()
    )

    summary_variants: list[dict[str, Any]] = []
    job_ids: list[str] = []
    for i, (variant_name, variant_recipe) in enumerate(variants, start=1):
        try:
            cfg_v = RecipeConfig.model_validate(variant_recipe)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=400,
                detail=(
                    f"variant {variant_name!r} fails schema validation "
                    f"(an axis value likely violates a pydantic constraint): {exc}"
                ),
            ) from exc

        axis_values = plan.axis_values_for(i)
        workspace_v = (workspace_root / variant_name).resolve()
        metadata = {
            "sweep_id": sweep_id,
            "variant_name": variant_name,
            "axis_values": axis_values,
        }
        result = _launch_job(cfg_v, workspace_v, metadata=metadata)
        job_ids.append(result["id"])
        summary_variants.append(
            {
                "name": variant_name,
                "job_id": result["id"],
                # Compact diff: just the axis paths and their assigned values.
                # The full materialised recipe is too bulky to ship back per
                # variant — callers can re-derive it from base + axis_values.
                "recipe_diff": axis_values,
            }
        )

    return {
        "sweep_id": sweep_id,
        "job_ids": job_ids,
        "variants": summary_variants,
    }


@router.get("/sweeps/{sweep_id}")
def get_sweep(sweep_id: str) -> dict[str, Any]:
    """Aggregate every job tagged with ``metadata.sweep_id == sweep_id``.

    Returns a per-state count plus a list of job summaries so the future
    UI can render a grouped progress bar without paginating ``/api/jobs``.
    A 404 is returned if no job carries the requested ``sweep_id``.
    """
    matched = [
        j
        for j in state.registry.list()
        if j.metadata is not None and j.metadata.get("sweep_id") == sweep_id
    ]
    if not matched:
        raise HTTPException(status_code=404, detail="sweep not found")

    counts = {s.value: 0 for s in JobState}
    for j in matched:
        counts[j.state.value] += 1

    matched.sort(key=lambda j: j.created_at)
    return {
        "sweep_id": sweep_id,
        "total": len(matched),
        "queued": counts[JobState.queued.value],
        "running": counts[JobState.running.value],
        "succeeded": counts[JobState.succeeded.value],
        "failed": counts[JobState.failed.value],
        "canceled": counts[JobState.canceled.value],
        "interrupted": counts[JobState.interrupted.value],
        "canceling": counts[JobState.canceling.value],
        "jobs": [j.to_summary() for j in matched],
    }
