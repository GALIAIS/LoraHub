"""Sweep — batch-enqueue variants from grid / random / TPE search.

The sweep endpoint takes one validated base config plus N axes, drives a
:class:`MaterialisedSweep` to produce variants one at a time, and pushes
each one through :func:`_launch_job` so the scheduler runs them serially
under the existing single-slot concurrency model.

Three modes are accepted:

* ``"grid"`` — the legacy default. Cartesian product over each axis's
  enumerated values; ``n_trials`` is ignored.
* ``"random"`` — independent draws from each axis distribution; emits
  ``n_trials`` variants.
* ``"tpe"`` — Optuna Tree-structured Parzen Estimator; needs the
  optional ``[sweep]`` extra. A missing optuna install surfaces as
  HTTP 503.

Adaptive modes (random / tpe) currently materialise their full
``n_trials`` batch up front. TPE without per-trial feedback still
runs — it falls back to its prior — but to actually benefit from
TPE's adaptive behaviour we need the metric-stream → ``report_trial``
loop, which is cut3.

Each spawned :class:`JobRecord` is stamped with
``metadata = {"sweep_id": ..., "axis_values": {...}}`` so a later
``GET /api/sweeps/{sweep_id}`` can filter the in-memory registry without
needing a join table. The metadata blob is persisted alongside the rest of
the job row in SQLite, so sweep_id grouping survives a server restart.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import ulid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator

from lorahub.api import state
from lorahub.api.jobs_helpers import _launch_job
from lorahub.api.state import JobRecord, JobState
from lorahub.core.config.schema import TrainingConfig
from lorahub.core.sweep import (
    SamplerUnavailableError,
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
    """One axis of the search space.

    `kind` selects the distribution:

    * ``categorical`` (default) — uses ``values`` as-is.
    * ``int_uniform`` / ``uniform`` / ``loguniform`` — uses
      ``low`` / ``high`` (and optional ``step``); ``values`` is
      ignored. Loguniform requires ``low > 0``.

    Categorical with ``values=[]`` and a numeric kind without
    ``low``/``high`` are both rejected at the core layer; we mirror
    only the obvious shape check here so the user gets a 400 instead
    of a 500.
    """

    path: str
    kind: Literal["categorical", "uniform", "loguniform", "int_uniform"] = "categorical"
    values: list[Any] = Field(default_factory=list)
    low: float | None = None
    high: float | None = None
    step: float | None = None

    @model_validator(mode="after")
    def _check_kind_shape(self) -> SweepAxisRequest:
        if self.kind == "categorical":
            if not self.values:
                msg = f"axis {self.path!r} (categorical) needs at least one value"
                raise ValueError(msg)
        else:
            if self.low is None or self.high is None:
                msg = f"axis {self.path!r} ({self.kind}) needs both `low` and `high`"
                raise ValueError(msg)
        return self


class CreateSweepRequest(BaseModel):
    """Sweep submission payload.

    `mode` defaults to ``"grid"`` so callers that pre-date cut2 keep
    working without changes. Random / TPE modes require ``n_trials``
    (validated downstream by SweepPlan); ``seed`` lets the user
    reproduce a random/TPE sweep across reruns.
    """

    base_config: dict[str, Any]
    axes: list[SweepAxisRequest] = Field(min_length=1)
    name_template: str = "{base}-{i:03d}"
    workspace_root: str | None = None
    mode: Literal["grid", "random", "tpe"] = "grid"
    n_trials: int | None = None
    seed: int | None = None


def _to_core_axis(a: SweepAxisRequest) -> SweepAxis:
    return SweepAxis(
        path=a.path,
        kind=a.kind,
        values=list(a.values),
        low=a.low,
        high=a.high,
        step=a.step,
    )


@router.post("/sweeps", status_code=202)
def create_sweep(req: CreateSweepRequest) -> dict[str, Any]:
    """Expand a sweep into N variants, enqueue each one, return the manifest.

    Errors:
      422 — base config fails schema validation
      400 — axis path doesn't resolve in base, axis shape is invalid,
            grid is too large, ``n_trials`` is missing for random/tpe,
            or a materialised variant fails schema validation (likely
            caused by an axis value that violates a pydantic constraint)
      503 — ``mode="tpe"`` requested but optuna is not installed
    """
    try:
        TrainingConfig.model_validate(req.base_config)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=422, detail=f"base_config is invalid: {exc}"
        ) from exc

    try:
        plan = SweepPlan(
            base_config=req.base_config,
            axes=[_to_core_axis(a) for a in req.axes],
            name_template=req.name_template,
            mode=req.mode,
            n_trials=req.n_trials,
            seed=req.seed,
        )
    except SweepError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    # Drive the sweep through MaterialisedSweep so all three modes go
    # through the same code path. For grid this is equivalent to the
    # legacy `expand()`; for random/tpe it lazily yields one variant
    # at a time. We currently still drain the iterator up front so the
    # API response shape (variants[]) stays the same — making this
    # truly streaming is part of cut3 (metric feedback loop).
    try:
        materialised = plan.materialize()
    except SamplerUnavailableError as exc:
        # Optuna isn't installed; tell the caller exactly how to fix it.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
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
    while True:
        try:
            nxt = materialised.next_variant()
        except SweepError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if nxt is None:
            break
        variant_name, variant_config, axis_values = nxt
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
                    "axes": [_axis_dump(a) for a in req.axes],
                    "name_template": req.name_template,
                    "workspace_root": str(workspace_root),
                    "mode": req.mode,
                    "n_trials": req.n_trials,
                    "seed": req.seed,
                },
                base_config=req.base_config,
                job_ids=job_ids,
            )
        )

    return {
        "sweep_id": sweep_id,
        "job_ids": job_ids,
        "variants": summary_variants,
        "mode": req.mode,
    }


def _axis_dump(a: SweepAxisRequest) -> dict[str, Any]:
    """Round-trip-friendly axis JSON: drop fields irrelevant to `kind`."""
    out: dict[str, Any] = {"path": a.path, "kind": a.kind}
    if a.kind == "categorical":
        out["values"] = list(a.values)
    else:
        out["low"] = a.low
        out["high"] = a.high
        if a.step is not None:
            out["step"] = a.step
    return out


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

    # Index records by id once so each live sweep can pick up its
    # persisted plan (mode / n_trials / seed) without N round-trips.
    records_by_id = {r.id: r for r in _list_sweep_records()}

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
        record = records_by_id.get(sweep_id)
        plan = record.plan if record else {}
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
                # Surface the search strategy so the UI can group/badge
                # sweeps by mode without round-tripping to GET /sweeps/{id}.
                # Older records predate cut2 and don't carry these keys —
                # default to "grid" so the UI stays consistent.
                "mode": plan.get("mode", "grid") if isinstance(plan, dict) else "grid",
                "n_trials": (
                    plan.get("n_trials") if isinstance(plan, dict) else None
                ),
                "seed": plan.get("seed") if isinstance(plan, dict) else None,
            }
        )

    # Merge in store-only sweeps (every child job archived/deleted).
    seen = {s["sweep_id"] for s in summaries}
    for record in records_by_id.values():
        if record.id in seen:
            continue
        plan = record.plan if isinstance(record.plan, dict) else {}
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
                "mode": plan.get("mode", "grid"),
                "n_trials": plan.get("n_trials"),
                "seed": plan.get("seed"),
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
