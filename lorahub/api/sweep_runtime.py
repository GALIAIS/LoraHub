"""Runtime glue between the sweep router and finished training jobs.

The TPE sweep needs a metric-fed feedback loop — every time a child job
hits a terminal state, the sampler must see its score so the next
``ask()`` can avoid the bad regions. This module owns that wiring.

Design:

* ``_active_sweeps`` holds one :class:`MaterialisedSweep` per running
  sweep_id, in process memory only. Restarting the API loses the live
  sampler study (already-enqueued trials still finish, but no further
  adaptive guidance is possible until cut4 wires SQLite-backed study
  recovery). The pareto endpoint never depends on this dict — it
  reads finished JobRecords + their stored ``axis_values`` directly,
  so historical reports survive a restart.

* :func:`report_terminal_job` is the single sink callers invoke when a
  job they care about transitions into a terminal state. It pulls the
  loss / val_loss out of the workspace's ``events.jsonl`` and pushes
  it through ``MaterialisedSweep.report_trial``. Failures + missing
  metrics report ``float('inf')`` — TPE treats infinity as a maximally
  bad outcome and keeps proposing trials elsewhere instead of stalling.

* The hook is wired from :func:`lorahub.api.jobs_helpers._launch_job`
  via the ``done``-event handler, plus the launch-exception path and
  the cancel/kill paths in :mod:`lorahub.api.routers.jobs`.

Pure event-driven — no polling. The runtime never owns a thread; it
runs inline on whatever caller invoked the state transition.
"""

from __future__ import annotations

import json
import logging
import math
import threading
from pathlib import Path
from typing import Any

from lorahub.api.state import JobRecord
from lorahub.core.sweep import MaterialisedSweep

log = logging.getLogger(__name__)

# Module-level registry of in-flight TPE / random / grid sweeps.
# Key is sweep_id, value is the live MaterialisedSweep that the router
# obtained from ``SweepPlan.materialize()``. Not durable — restart
# wipes this dict. The pareto endpoint queries JobRecord history, so
# user-visible analytics still work after a restart; only the
# adaptive sampler's internal Bayes model is lost.
_active_sweeps: dict[str, MaterialisedSweep] = {}
_lock = threading.RLock()


def register_sweep(sweep_id: str, sweep: MaterialisedSweep) -> None:
    """Stash an active sweep so terminal-job callbacks can find it."""
    with _lock:
        _active_sweeps[sweep_id] = sweep


def unregister_sweep(sweep_id: str) -> MaterialisedSweep | None:
    """Drop a sweep from the live registry (e.g. once exhausted)."""
    with _lock:
        return _active_sweeps.pop(sweep_id, None)


def get_active(sweep_id: str) -> MaterialisedSweep | None:
    """Return the live MaterialisedSweep for ``sweep_id`` if any."""
    with _lock:
        return _active_sweeps.get(sweep_id)


def reset_for_tests() -> None:
    """Clear every active sweep — only safe to call from test fixtures."""
    with _lock:
        _active_sweeps.clear()


def _read_final_score(workspace: Path) -> float:
    """Pull a single end-of-run score from a job's ``events.jsonl``.

    Preference order:

      1. Last ``validation`` event's ``val_loss`` — best generalisation
         signal we have.
      2. Last ``step`` event's ``loss`` — fallback when validation
         wasn't enabled.
      3. ``float('inf')`` — no usable metric. Treated by TPE as a
         maximally bad sample so the search avoids the area without
         crashing the study.

    Lines that fail to parse are skipped silently (truncated writes
    happen in the wild and we never want to pollute the sampler with
    a stack trace). Same parsing shape as :func:`_read_metrics` in
    :mod:`lorahub.api.jobs_helpers`.
    """
    events_path = workspace / "events.jsonl"
    if not events_path.is_file():
        return math.inf

    last_loss: float | None = None
    last_val: float | None = None
    try:
        with events_path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except (json.JSONDecodeError, TypeError):
                    continue
                payload = row.get("payload") or {}
                etype = row.get("type")
                if etype == "validation":
                    val = payload.get("val_loss")
                    if isinstance(val, (int, float)):
                        last_val = float(val)
                elif etype == "step":
                    cur = payload.get("loss")
                    if isinstance(cur, (int, float)):
                        last_loss = float(cur)
    except OSError:
        return math.inf

    if last_val is not None and math.isfinite(last_val):
        return last_val
    if last_loss is not None and math.isfinite(last_loss):
        return last_loss
    return math.inf


def report_terminal_job(job: JobRecord) -> None:
    """Push ``job``'s outcome into its parent sweep's sampler if any.

    Safe to call from any context — this is the single hook every
    state-transition site fans into. No-ops when:

      * The job has no ``metadata.sweep_id``
      * The sweep is not in the active registry (restart, never
        registered, or already exhausted)
      * The job is missing ``metadata.axis_values``

    Failures inside the read path swallow into ``float('inf')`` so a
    crashed job still feeds back a "this region is bad" signal
    instead of stalling the study.
    """
    meta = job.metadata if isinstance(job.metadata, dict) else None
    if not meta:
        return
    sweep_id = meta.get("sweep_id")
    axis_values = meta.get("axis_values")
    if not isinstance(sweep_id, str) or not isinstance(axis_values, dict):
        return

    sweep = get_active(sweep_id)
    if sweep is None:
        return

    score = _read_final_score(job.workspace)
    try:
        sweep.report_trial(axis_values, score)
    except Exception:  # noqa: BLE001
        # Sampler errors must never propagate up into the job state
        # machine — the run is already done, this is bookkeeping.
        log.exception(
            "sweep %s: report_trial failed for job %s", sweep_id, job.id
        )
        return

    # If every planned trial has reported back, evict the sweep so we
    # don't leak entries forever in long-lived deployments.
    expected = sweep.n_trials
    if len(sweep.reported_scores) >= expected and sweep.remaining() == 0:
        unregister_sweep(sweep_id)


def rebuild_active_sweeps(state_module: Any, sweep_store: Any) -> int:
    """Re-register live sweeps from persisted state on app startup.

    A server restart wipes ``_active_sweeps``. Without this hook, any
    in-flight TPE sweep silently degrades — the dangling RUNNING trials
    in the optuna RDB never receive their score, and the next call to
    :meth:`MaterialisedSweep.next_variant` (if the user re-asks for a
    fresh batch) starts cold instead of consulting prior trials.

    For every sweep in ``sweep_store`` whose ``plan`` carries a TPE
    ``study_path``, we rebuild the SweepPlan from the stored axes /
    base_config and call ``materialize()`` against the same sqlite
    file. The resulting MaterialisedSweep gets registered under its
    sweep_id so the next terminal-job callback for one of its
    children lands on the right sampler.

    Sweeps with no ``study_path`` (grid / random) are skipped — they
    have nothing in-memory worth restoring; their pareto data still
    surfaces from JobRecord history.

    Returns the number of sweeps re-registered. Best-effort: a missing
    optuna install or an unreadable sqlite file just skips that sweep
    instead of failing startup.
    """
    if sweep_store is None:
        return 0

    # Lazy imports — sweep_runtime is imported during normal request
    # paths and shouldn't drag SweepPlan / SweepAxis until restart.
    from lorahub.core.sweep import (  # noqa: PLC0415
        SamplerUnavailableError,
        SweepError,
        SweepPlan,
    )

    rebuilt = 0
    try:
        records = sweep_store.list()
    except Exception:  # noqa: BLE001
        log.exception("sweep rebuild: list() failed")
        return 0

    # Index live JobRecords by sweep_id so we can decide whether a
    # sweep still has in-flight or unreported trials worth restoring.
    by_sweep: dict[str, list[Any]] = {}
    for j in state_module.registry.list():
        meta = j.metadata if isinstance(j.metadata, dict) else None
        if not meta:
            continue
        sid = meta.get("sweep_id")
        if isinstance(sid, str):
            by_sweep.setdefault(sid, []).append(j)

    for record in records:
        plan_dict = record.plan if isinstance(record.plan, dict) else {}
        study_path_str = plan_dict.get("study_path")
        if not study_path_str:
            continue  # grid / random — no live state worth restoring
        study_path = Path(study_path_str)
        if not study_path.is_file():
            # The sqlite file is gone (manual cleanup, missing volume).
            # Skip rather than spam the log every restart.
            continue

        # Skip sweeps where every child is already terminal — any
        # dangling trials in the RDB will never get a score now, and
        # no future job-completion callback will arrive.
        children = by_sweep.get(record.id, [])
        if children and all(_is_terminal(j) for j in children):
            continue

        try:
            axes = [_axis_from_dump(a) for a in plan_dict.get("axes", [])]
            plan = SweepPlan(
                base_config=record.base_config,
                axes=axes,
                name_template=plan_dict.get("name_template", "{base}-{i:03d}"),
                mode=plan_dict.get("mode", "tpe"),
                n_trials=plan_dict.get("n_trials"),
                seed=plan_dict.get("seed"),
                storage_path=study_path,
                study_name=record.id,
            )
            materialised = plan.materialize()
        except SamplerUnavailableError:
            log.warning(
                "sweep %s: optuna not installed; skipping rebuild", record.id
            )
            continue
        except SweepError:
            log.exception("sweep %s: rebuild failed (plan invalid)", record.id)
            continue
        except Exception:  # noqa: BLE001
            log.exception("sweep %s: rebuild failed", record.id)
            continue

        register_sweep(record.id, materialised)
        rebuilt += 1

    if rebuilt:
        log.info("rebuilt %d active sweep(s) on startup", rebuilt)
    return rebuilt


def _is_terminal(job: Any) -> bool:
    """Lazy state.JobState terminal check — avoids a top-level cycle."""
    from lorahub.api.jobs_helpers import _TERMINAL_STATES  # noqa: PLC0415

    return job.state in _TERMINAL_STATES


def _axis_from_dump(dumped: dict[str, Any]) -> Any:
    """Inverse of routers.sweeps._axis_dump — used only by the rebuild path."""
    from lorahub.core.sweep import SweepAxis  # noqa: PLC0415

    return SweepAxis(
        path=dumped["path"],
        kind=dumped.get("kind", "categorical"),
        values=list(dumped.get("values") or []),
        low=dumped.get("low"),
        high=dumped.get("high"),
        step=dumped.get("step"),
    )


__all__ = [
    "_active_sweeps",
    "get_active",
    "rebuild_active_sweeps",
    "register_sweep",
    "report_terminal_job",
    "reset_for_tests",
    "unregister_sweep",
]
