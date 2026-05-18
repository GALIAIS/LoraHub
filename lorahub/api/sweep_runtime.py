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

* After every successful ``report_trial``, :func:`report_terminal_job`
  invokes the sweep's :class:`SweepLaunchContext` (if any) to ask the
  sampler for the *next* trial and enqueue it through the launch hook
  the router stamped at ``register_sweep`` time. This is what makes the
  TPE feedback loop streaming — the sampler observes the prior trial's
  score before its next ``ask()`` runs, so the proposal is genuinely
  adaptive instead of being a batch of independent draws made before
  any feedback was available. Sweeps without a context (grid / random
  drained up-front, or a TPE sweep rebuilt from disk on startup) just
  no-op the advance — see ``rebuild_active_sweeps`` for the cut4.A
  caveat.

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
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lorahub.api.state import JobRecord
from lorahub.core.sweep import MaterialisedSweep, SweepError

log = logging.getLogger(__name__)


# Callable signature the router hands us at register time. Takes the
# tuple shape ``MaterialisedSweep.next_variant()`` returns and is
# expected to enqueue exactly one job. Returning ``None`` signals the
# router declined to launch (e.g. a stale variant that lost a schema
# race) — the runtime logs and stops advancing rather than spinning.
SweepLauncher = Callable[[str, dict[str, Any], dict[str, Any]], dict[str, Any] | None]


@dataclass(slots=True)
class SweepLaunchContext:
    """Per-sweep launch closure registered alongside the MaterialisedSweep.

    Populated by the router when it creates a streaming TPE sweep so the
    terminal-job hook can ask for the next variant without a round-trip
    through the HTTP layer. The closure must perform schema validation,
    workspace setup, metadata stamping, and the actual ``_launch_job``
    call — all of the per-trial side effects the router runs on the
    happy path. ``None`` is a valid context value: rebuilt sweeps and
    grid/random sweeps register without one and the runtime treats them
    as "report-only" (no auto-advance).
    """

    launch: SweepLauncher


# Module-level registry of in-flight TPE / random / grid sweeps.
# Key is sweep_id, value is the live MaterialisedSweep that the router
# obtained from ``SweepPlan.materialize()``. Not durable — restart
# wipes this dict. The pareto endpoint queries JobRecord history, so
# user-visible analytics still work after a restart; only the
# adaptive sampler's internal Bayes model is lost.
_active_sweeps: dict[str, MaterialisedSweep] = {}
# Optional companion to ``_active_sweeps``: the launch context the
# router stamped at create time. Only TPE registers one (grid/random
# drain up-front so there is nothing left to advance). After a restart
# rebuild this dict stays empty for the rebuilt sweep — see
# :func:`rebuild_active_sweeps` for why we don't try to reconstruct it.
_launch_contexts: dict[str, SweepLaunchContext] = {}
_lock = threading.RLock()


def register_sweep(
    sweep_id: str,
    sweep: MaterialisedSweep,
    context: SweepLaunchContext | None = None,
) -> None:
    """Stash an active sweep + optional launch context.

    ``context`` is only meaningful for streaming TPE — without it the
    terminal-job hook reports the score but does not auto-advance.
    Re-registering an existing ``sweep_id`` replaces both entries; the
    last writer wins.
    """
    with _lock:
        _active_sweeps[sweep_id] = sweep
        if context is not None:
            _launch_contexts[sweep_id] = context
        else:
            # Defensive: drop a stale context if a caller re-registers
            # the same id without one. Otherwise advance() would call
            # back into a closure whose closure-state may be invalid.
            _launch_contexts.pop(sweep_id, None)


def unregister_sweep(sweep_id: str) -> MaterialisedSweep | None:
    """Drop a sweep + its launch context from the live registry."""
    with _lock:
        _launch_contexts.pop(sweep_id, None)
        return _active_sweeps.pop(sweep_id, None)


def get_active(sweep_id: str) -> MaterialisedSweep | None:
    """Return the live MaterialisedSweep for ``sweep_id`` if any."""
    with _lock:
        return _active_sweeps.get(sweep_id)


def get_launch_context(sweep_id: str) -> SweepLaunchContext | None:
    """Return the registered launch context for ``sweep_id`` if any."""
    with _lock:
        return _launch_contexts.get(sweep_id)


def reset_for_tests() -> None:
    """Clear every active sweep + context — only safe from test fixtures."""
    with _lock:
        _active_sweeps.clear()
        _launch_contexts.clear()


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

    After the score lands, if the sweep still has trials remaining and
    a :class:`SweepLaunchContext` was registered, the next variant is
    pulled from the sampler and enqueued via the registered launcher.
    The advance happens *after* the score is told back so a TPE
    ``ask()`` sees the just-finished trial in its prior. Sweeps with
    no context (rebuilt-after-restart, grid/random) skip the advance —
    they are score-only sinks.
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

    # Streaming advance: ask the sampler for the next trial *now* that
    # it has seen this trial's score, then hand it to the router-supplied
    # launcher. Only fires when the router registered a context — grid
    # and random drained all variants up-front so there is nothing left
    # to enqueue, and rebuilt sweeps deliberately don't get a context
    # (the launch metadata is not persisted; cut4.C territory).
    context = get_launch_context(sweep_id)
    if context is not None and sweep.remaining() > 0:
        try:
            nxt = sweep.next_variant()
        except SweepError:
            log.exception(
                "sweep %s: next_variant failed during streaming advance",
                sweep_id,
            )
            nxt = None
        if nxt is not None:
            variant_name, variant_config, next_axis_values = nxt
            try:
                context.launch(variant_name, variant_config, next_axis_values)
            except Exception:  # noqa: BLE001
                # Launch failures here can't propagate — the *previous*
                # trial's terminal hook is already mid-flight. Logging
                # is the best we can do; the operator can re-launch
                # the missing variant manually if desired.
                log.exception(
                    "sweep %s: streaming launch failed for variant %s",
                    sweep_id,
                    variant_name,
                )

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

    Streaming-advance gap (cut4.B → cut4.C): the rebuild path
    deliberately registers each sweep **without** a
    :class:`SweepLaunchContext`. The launch closure depends on the
    router's process-local ``_launch_job`` plus the original
    ``workspace_root`` and request payload — none of which is
    reconstructible from the persisted plan alone. Concretely, after a
    restart the sweep enters "score-only" mode: terminal-job callbacks
    still feed scores into the sampler so the sqlite study stays
    coherent, but the next trial is **not** auto-enqueued. Already-
    enqueued or in-flight children continue to drain normally.
    Persisting enough launch metadata to reconstruct the closure (or
    re-spawning trials via a synthetic /sweeps re-POST) is cut4.C.

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
    "SweepLaunchContext",
    "SweepLauncher",
    "get_active",
    "get_launch_context",
    "rebuild_active_sweeps",
    "register_sweep",
    "report_terminal_job",
    "reset_for_tests",
    "unregister_sweep",
]
