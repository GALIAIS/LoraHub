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

Streaming vs batch (cut4.B): grid + random drain every variant up
front and call ``_launch_job`` N times before the response returns —
no adaptive feedback exists for them, so there is nothing to lose by
asking the sampler N independent times in a row. TPE is different —
its ``ask()`` returns a uniform draw until the *previous* trial's
``tell()`` lands, so draining the batch up front would erase the
sampler's adaptive advantage. For TPE we therefore launch only the
**first** trial inside the request handler and let
:func:`lorahub.api.sweep_runtime.report_terminal_job` ask for trial 2,
trial 3, ... after each prior trial reports its score. The router's
202 response carries only the trials launched synchronously
(``launched_count`` / ``pending_count`` reflect the streaming split).

TPE feedback loop (cut3 + cut4.B): the materialised sweep is registered
in :mod:`lorahub.api.sweep_runtime` together with a
:class:`SweepLaunchContext` closing over ``workspace_root`` /
``base_config`` / ``name_template`` so the runtime can drive each
subsequent trial without re-entering the HTTP layer. Whenever a child
job hits a terminal state, ``report_terminal_job`` pushes the final
``loss`` / ``val_loss`` back into the sampler **and then** asks for
the next variant. Failed or metric-less trials report ``float('inf')``
so the study still steers away from the offending region instead of
silently skipping it; the next-trial advance still fires.

The active-sweep registry is in-memory only — server restart loses the
live Optuna study and (intentionally) the launch context. Already-
enqueued trials still run, dangling RUNNING trials get their score on
completion, but the streaming auto-advance is suspended until cut4.C
persists enough launch metadata for the rebuild path to reconstruct
the context. The ``/sweeps/{id}/pareto`` endpoint reads JobRecord
history directly, so historical Pareto / best-trial reporting always
survives a restart.

Each spawned :class:`JobRecord` is stamped with
``metadata = {"sweep_id": ..., "axis_values": {...}}`` so a later
``GET /api/sweeps/{sweep_id}`` can filter the in-memory registry without
needing a join table. The metadata blob is persisted alongside the rest of
the job row in SQLite, so sweep_id grouping survives a server restart.
"""

from __future__ import annotations

import dataclasses
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

import ulid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, model_validator

from lorahub.api import state
from lorahub.api.jobs_helpers import _TERMINAL_STATES, _launch_job
from lorahub.api.paths import resolve_sweep_variant_path
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

log = logging.getLogger(__name__)


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

    sweep_id = str(ulid.new())
    from lorahub.api.paths import resolve_run_path, runs_dir  # noqa: PLC0415
    try:
        workspace_root = (
            resolve_run_path(req.workspace_root, allow_root=True)
            if req.workspace_root
            else runs_dir().resolve()
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # For TPE, point the study at a per-sweep sqlite file under the
    # workspace root. Reopening the same path on a future restart
    # restores every prior trial — completed ones feed the TPE prior,
    # RUNNING ones (left dangling by the restart) get matched by their
    # axis_values when their job's terminal callback fires. Grid /
    # random ignore storage_path so this is a no-op for them.
    study_path: Path | None = None
    if req.mode == "tpe":
        study_path = workspace_root / "_sweeps" / sweep_id / "study.db"
        plan = dataclasses.replace(
            plan, storage_path=study_path, study_name=sweep_id
        )

    # Drive the sweep through MaterialisedSweep so all three modes go
    # through the same code path. For grid this is equivalent to the
    # legacy `expand()`; for random/tpe it lazily yields one variant
    # at a time. Grid + random still drain the iterator up front so the
    # API response shape stays the same — they are non-adaptive, so
    # there is no advantage to deferring the launches. TPE is the
    # streaming case: only the first trial is launched synchronously,
    # and :mod:`lorahub.api.sweep_runtime` advances the sweep one trial
    # at a time as each prior trial reports its score back. That makes
    # the sampler's ``ask()`` actually adaptive instead of degrading
    # to random when the study is empty at create time.
    try:
        materialised = plan.materialize()
    except SamplerUnavailableError as exc:
        # Optuna isn't installed; tell the caller exactly how to fix it.
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except SweepTooLargeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except SweepError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    summary_variants: list[dict[str, Any]] = []
    job_ids: list[str] = []

    def _enqueue(
        variant_name: str,
        variant_config: dict[str, Any],
        axis_values: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Validate one variant, launch it, and append to the response lists.

        Returns the launcher's summary dict on success. ``None`` on
        validation failure — the streaming path uses that to bail out
        quietly (the prior trial's terminal hook can't raise an
        HTTPException to the original caller anyway). The synchronous
        bootstrap path raises HTTPException so the user sees a 400.
        """
        try:
            workspace_v = resolve_sweep_variant_path(workspace_root, variant_name)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        try:
            cfg_v = TrainingConfig.model_validate(variant_config)
        except Exception as exc:  # noqa: BLE001
            # Re-raise on the synchronous path so callers see a 400;
            # the runtime advance path catches this and just logs.
            raise HTTPException(
                status_code=400,
                detail=(
                    f"variant {variant_name!r} fails schema validation "
                    f"(an axis value likely violates a pydantic constraint): {exc}"
                ),
            ) from exc
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
        return result

    def _streaming_launch(
        variant_name: str,
        variant_config: dict[str, Any],
        axis_values: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Launch closure handed to the runtime for post-bootstrap trials.

        Mirrors ``_enqueue`` but swallows validation failures into a
        warning log + ``None`` return — by the time we hit this path the
        original /api/sweeps caller is long gone and HTTPException would
        propagate up into the terminal-job hook of an unrelated job.
        """
        try:
            return _enqueue(variant_name, variant_config, axis_values)
        except HTTPException as exc:
            log.warning(
                "sweep %s: streaming launch refused variant %s: %s",
                sweep_id,
                variant_name,
                exc.detail,
            )
            return None

    # Register the sweep BEFORE launching any child job — the launch
    # closure runs in a worker thread and may transition the job to a
    # terminal state before we'd otherwise reach the registration call.
    # TPE additionally registers a SweepLaunchContext so the runtime
    # can advance trial-by-trial as each prior trial reports its score.
    from lorahub.api.sweep_runtime import (  # noqa: PLC0415
        SweepLaunchContext,
        register_sweep,
    )
    context: SweepLaunchContext | None = None
    if req.mode == "tpe":
        context = SweepLaunchContext(launch=_streaming_launch)
    register_sweep(sweep_id, materialised, context=context)

    # Bootstrap: grid + random fully drain here (their samplers are
    # stateless so there is no benefit to streaming); TPE launches only
    # the first trial and returns — subsequent trials are driven by
    # report_terminal_job advancing the sweep one step per prior
    # finish.
    streaming = req.mode == "tpe"
    while True:
        try:
            nxt = materialised.next_variant()
        except SweepError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if nxt is None:
            break
        variant_name, variant_config, axis_values = nxt
        _enqueue(variant_name, variant_config, axis_values)
        if streaming:
            break

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
                    "study_path": str(study_path) if study_path else None,
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
        # Streaming-aware counters: ``launched_count`` reflects what the
        # router actually enqueued before returning (1 for TPE, N for
        # grid/random). ``pending_count`` is what the runtime owes us
        # via streaming advance — non-zero only for TPE that has more
        # trials in its budget. ``n_trials_total`` is the planned total
        # so clients can render progress without re-deriving it from
        # the request shape.
        "launched_count": len(job_ids),
        "pending_count": materialised.remaining(),
        "n_trials_total": materialised.n_trials,
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
        # alongside the live aggregate so the UI can show the config even
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


@router.get("/sweeps/{sweep_id}/pareto")
def get_sweep_pareto(sweep_id: str) -> dict[str, Any]:
    """Return finished trials, the best score, and a count of pending jobs.

    Single-objective minimisation: ``best`` is the trial with the lowest
    finite score among completed jobs. Trials with no usable metric report
    as ``float('inf')`` and never beat a finite score, but they are still
    listed under ``completed_trials`` so the operator can see them.

    Read-side endpoint — depends only on JobRecord history (workspace
    ``events.jsonl`` + ``metadata.axis_values``), never on the in-memory
    ``_active_sweeps`` dict. That means a server restart loses the live
    Optuna study but pareto data still surfaces correctly for any
    JobRecord that survives the restart in SQLite.

    Returns 404 only when neither the registry nor the SweepStore knows
    the sweep_id. A sweep that's still spawning trials returns its
    partial completed set with ``pending`` reflecting the in-flight
    count.
    """
    matched = [
        j
        for j in state.registry.list()
        if j.metadata is not None and j.metadata.get("sweep_id") == sweep_id
    ]
    record = _load_sweep_record(sweep_id)

    if not matched and record is None:
        raise HTTPException(status_code=404, detail="sweep not found")

    completed_trials: list[dict[str, Any]] = []
    pending = 0
    for job in matched:
        meta = job.metadata or {}
        axis_values = meta.get("axis_values")
        if not isinstance(axis_values, dict):
            # Legacy / non-axis jobs can't surface in the pareto view.
            continue
        if job.state in _TERMINAL_STATES:
            score = _read_final_score(job.workspace)
            completed_trials.append(
                {
                    "axis_values": axis_values,
                    "score": score,
                    "job_id": job.id,
                    "state": job.state.value,
                }
            )
        else:
            pending += 1

    # Single-objective: pick the lowest *finite* score; tie-break by
    # earliest completion. Pure infinities (failed / metric-less) leave
    # ``best=None`` so the UI can render "no successful trial yet".
    best: dict[str, Any] | None = None
    for trial in completed_trials:
        score = trial["score"]
        if not isinstance(score, (int, float)) or score == float("inf"):
            continue
        if best is None or score < best["score"]:
            best = {
                "axis_values": trial["axis_values"],
                "score": score,
                "job_id": trial["job_id"],
            }

    return {
        "sweep_id": sweep_id,
        "completed_trials": completed_trials,
        "best": best,
        "pending": pending,
    }


def _read_final_score(workspace: Path) -> float:
    """Module-level shim into the runtime's score reader.

    Kept here as a thin re-export so the pareto endpoint and the
    feedback hook share a single definition of "final score". The
    feedback path (jobs_helpers / runtime) is the canonical owner —
    importing it lazily keeps the sweeps router free of a hard
    dependency on the runtime module.
    """
    from lorahub.api.sweep_runtime import _read_final_score as _impl  # noqa: PLC0415

    return _impl(workspace)


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
