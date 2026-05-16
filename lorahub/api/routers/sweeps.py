"""Sweep — batch-enqueue a cartesian-product grid over a base config.

The sweep endpoint takes one validated base config plus N axes, materialises
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

from collections import defaultdict
from pathlib import Path
from typing import Any

import ulid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from lorahub.api import state
from lorahub.api.jobs_helpers import _launch_job
from lorahub.api.state import JobRecord, JobState
from lorahub.core.config.schema import TrainingConfig
from lorahub.core.sweep import (
    SweepAxis,
    SweepError,
    SweepPlan,
    SweepTooLargeError,
)

router = APIRouter(prefix="/api")


def _common_prefix(names: list[str]) -> str:
    """Longest shared leading substring across `names`, trimmed to a sane stem.

    The sweep router stamps each variant's ``output.name`` with the template
    ``{base}-{i:03d}``, so the shared prefix is normally ``{base}-``. Naïve
    char-by-char prefix matching would also keep a few digits when the index
    happens to be zero-padded (``alpha-001``, ``alpha-002`` → ``alpha-00``),
    so we strip trailing digits and separators (``-_./``) until we land on a
    clean stem. Falls back to the first non-empty name when only one variant
    has been registered.
    """
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    shortest = min(names, key=len)
    cutoff = len(shortest)
    for i, ch in enumerate(shortest):
        if any(name[i] != ch for name in names):
            cutoff = i
            break
    prefix = shortest[:cutoff]
    # Drop the structural suffix the sweep template injects (``-001`` etc.).
    while prefix and (prefix[-1].isdigit() or prefix[-1] in "-_./"):
        prefix = prefix[:-1]
    return prefix or shortest


def _job_name(job: JobRecord) -> str | None:
    """Pull the human-readable variant name from the job's config snapshot."""
    snap = job.config_snapshot or {}
    output = snap.get("output") if isinstance(snap, dict) else None
    if isinstance(output, dict):
        name = output.get("name")
        if isinstance(name, str) and name:
            return name
    return None


class SweepAxisRequest(BaseModel):
    path: str
    values: list[Any] = Field(min_length=1)


class CreateSweepRequest(BaseModel):
    base_config: dict[str, Any]
    axes: list[SweepAxisRequest] = Field(min_length=1)
    name_template: str = "{base}-{i:03d}"
    workspace_root: str | None = None


@router.post("/sweeps", status_code=202)
def create_sweep(req: CreateSweepRequest) -> dict[str, Any]:
    """Expand a sweep into N variants, enqueue each one, return the manifest.

    Errors:
      422 — base config fails schema validation
      400 — axis path doesn't resolve in base, or grid is too large, or a
            materialised variant fails schema validation (likely caused by
            an axis value that violates a pydantic constraint)
    """
    try:
        TrainingConfig.model_validate(req.base_config)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=422, detail=f"base_config is invalid: {exc}"
        ) from exc

    plan = SweepPlan(
        base_config=req.base_config,
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
    for i, (variant_name, variant_config) in enumerate(variants, start=1):
        try:
            cfg_v = TrainingConfig.model_validate(variant_config)
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
                # The full materialised config is too bulky to ship back per
                # variant — callers can re-derive it from base + axis_values.
                "config_diff": axis_values,
            }
        )

    # Persist the sweep descriptor so a server restart can still recover
    # the original axes / base_config / job_ids — jobs.metadata only
    # carries the per-variant axis_values.
    from lorahub.api import app as app_module  # noqa: PLC0415
    from lorahub.api.sweep_store import SweepRecord  # noqa: PLC0415

    sweep_store = getattr(app_module, "_sweep_store", None)
    if sweep_store is not None:
        sweep_store.upsert(
            SweepRecord(
                id=sweep_id,
                name=req.name_template,
                name_prefix=_common_prefix([v["name"] for v in summary_variants]),
                plan={
                    "axes": [{"path": a.path, "values": a.values} for a in req.axes],
                    "name_template": req.name_template,
                    "workspace_root": str(workspace_root),
                },
                base_config=req.base_config,
                job_ids=job_ids,
            )
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
    Falls back to the SweepStore so a sweep whose every child job has
    been deleted/archived still surfaces with its plan and known
    ``job_ids`` — useful for re-spawning lost variants. A 404 is returned
    only when neither the registry nor the store has heard of it.
    """
    matched = [
        j
        for j in state.registry.list()
        if j.metadata is not None and j.metadata.get("sweep_id") == sweep_id
    ]

    record = _load_sweep_record(sweep_id)

    if not matched and record is None:
        raise HTTPException(status_code=404, detail="sweep not found")

    counts = {s.value: 0 for s in JobState}
    for j in matched:
        counts[j.state.value] += 1

    matched.sort(key=lambda j: j.created_at)
    payload: dict[str, Any] = {
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
    if record is not None:
        # Surface the immutable plan (axes, name template, base config)
        # alongside the live aggregate so the UI can show the recipe even
        # when every child job is gone.
        payload["plan"] = record.plan
        payload["name"] = record.name
        payload["name_prefix"] = record.name_prefix
        payload["created_at"] = record.created_at.isoformat()
        payload["known_job_ids"] = list(record.job_ids)
    return payload


@router.get("/sweeps")
def list_sweeps() -> dict[str, Any]:
    """List every sweep, merging the in-registry view with persisted records.

    Sweeps whose child jobs are still in the registry surface their live
    counts; sweeps whose jobs have all been archived still surface from
    the SweepStore so the user can rerun the plan or grep history. The
    list is sorted newest-first by ``latest_modified_at`` (live activity)
    then by ``created_at`` (sweep birth) for store-only entries.
    """
    groups: dict[str, list[JobRecord]] = defaultdict(list)
    for job in state.registry.list():
        meta = job.metadata
        if not isinstance(meta, dict):
            continue
        sweep_id = meta.get("sweep_id")
        if not isinstance(sweep_id, str) or not sweep_id:
            continue
        groups[sweep_id].append(job)

    summaries: list[dict[str, Any]] = []
    for sweep_id, jobs in groups.items():
        counts = {s.value: 0 for s in JobState}
        for j in jobs:
            counts[j.state.value] += 1
        names = [n for n in (_job_name(j) for j in jobs) if n]
        prefix = _common_prefix(names) if names else sweep_id[-8:]
        # ``earliest_created_at`` anchors the sweep on a timeline; the latest
        # mod time is used to surface still-active sweeps at the top.
        earliest = min(j.created_at for j in jobs)
        latest = max(
            (j.finished_at or j.started_at or j.created_at) for j in jobs
        )
        summaries.append(
            {
                "sweep_id": sweep_id,
                "name_prefix": prefix,
                "total": len(jobs),
                "queued": counts[JobState.queued.value],
                "running": counts[JobState.running.value],
                "succeeded": counts[JobState.succeeded.value],
                "failed": counts[JobState.failed.value],
                "canceled": counts[JobState.canceled.value],
                "interrupted": counts[JobState.interrupted.value],
                "canceling": counts[JobState.canceling.value],
                "earliest_created_at": earliest.isoformat(),
                "latest_modified_at": latest.isoformat(),
            }
        )

    # Merge in store-only sweeps (every child job archived/deleted).
    seen = {s["sweep_id"] for s in summaries}
    for record in _list_sweep_records():
        if record.id in seen:
            continue
        summaries.append(
            {
                "sweep_id": record.id,
                "name_prefix": record.name_prefix or record.name,
                "total": len(record.job_ids),
                "queued": 0,
                "running": 0,
                "succeeded": 0,
                "failed": 0,
                "canceled": 0,
                "interrupted": 0,
                "canceling": 0,
                "earliest_created_at": record.created_at.isoformat(),
                "latest_modified_at": record.created_at.isoformat(),
                "archived": True,
            }
        )

    summaries.sort(key=lambda s: s["latest_modified_at"], reverse=True)
    return {"sweeps": summaries}


def _load_sweep_record(sweep_id: str):  # type: ignore[no-untyped-def]
    """Best-effort SweepStore lookup; tolerates store=None for tests."""
    from lorahub.api import app as app_module  # noqa: PLC0415

    store = getattr(app_module, "_sweep_store", None)
    if store is None:
        return None
    try:
        return store.get(sweep_id)
    except Exception:  # noqa: BLE001
        return None


def _list_sweep_records():  # type: ignore[no-untyped-def]
    from lorahub.api import app as app_module  # noqa: PLC0415

    store = getattr(app_module, "_sweep_store", None)
    if store is None:
        return []
    try:
        return store.list()
    except Exception:  # noqa: BLE001
        return []
