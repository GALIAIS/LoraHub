"""Hyperparameter sweep — grid + random + Optuna TPE.

A :class:`SweepPlan` takes a validated base config dict plus one or more
:class:`SweepAxis` declarations and materialises N variants. Each variant
carries a unique ``output.name`` suffix so the resulting workspaces,
checkpoints, and events.jsonl streams never collide when the API enqueues
the batch.

Three sampling modes:

  * ``"grid"`` — the historical default. Cartesian product over every
    axis's enumerated ``values``. Capped at ``SWEEP_MAX_VARIANTS``
    so an accidental 8-axis detonation can't run for weeks.
  * ``"random"`` — pure random search. Each trial draws independently
    from each axis's distribution. Same cap applies.
  * ``"tpe"`` — Optuna's Tree-structured Parzen Estimator. Calls into
    ``optuna.create_study`` lazily; a missing optuna install surfaces
    as :class:`SamplerUnavailableError`. The caller must report each
    trial's score back via ``Sampler.report_result`` for TPE to make
    informed suggestions on the next call.

Numeric axes (``uniform`` / ``loguniform`` / ``int_uniform``) are
supported alongside the historical categorical axes. Categorical axes
keep their ``values`` list; numeric axes carry ``low`` / ``high`` /
optional ``step``.
"""

from __future__ import annotations

import copy
import itertools
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

# Hard ceiling on the number of variants produced by a single SweepPlan.
# A single training job can take hours; the cap is a guard against an
# accidental "axis with 50 values times another with 20" detonation.
# Bumped only with care — the scheduler has no batch-cancel semantics yet.
SWEEP_MAX_VARIANTS = 256


SweepMode = Literal["grid", "random", "tpe"]
"""Search strategy. Grid is the legacy default; the others are new."""

AxisKind = Literal[
    "categorical",
    "uniform",
    "loguniform",
    "int_uniform",
]
"""Distribution family for a single axis."""


class SweepError(ValueError):
    """Base class for sweep validation errors."""


class SweepTooLargeError(SweepError):
    """Raised when expand() would produce more than SWEEP_MAX_VARIANTS variants."""


class SamplerUnavailableError(SweepError):
    """Raised when a sampler's optional backing dependency is missing.

    Currently only fires for ``mode="tpe"`` when ``optuna`` is not
    importable — the caller (router) maps it to a 503 so the user
    sees "install lorahub[sweep]" instead of an opaque 500.
    """


@dataclass(frozen=True)
class SweepAxis:
    """One axis of the search space.

    `path` is a dotted config path (e.g. ``optimizer.lr.unet`` or
    ``network.rank``). The remaining fields depend on `kind`:

    - ``categorical`` (default, the legacy shape): use `values`. Each
      element is enumerated as-is during grid search; random/tpe pick
      uniformly.
    - ``uniform``: continuous in [low, high]. Optional `step` snaps
      the sampled value to a grid (e.g. step=0.05 → multiples of
      0.05). `values` is ignored.
    - ``loguniform``: continuous in [low, high] sampled in log space.
      Useful for learning rates spanning orders of magnitude. `step`
      is ignored.
    - ``int_uniform``: integers in [low, high], inclusive both ends.
      `step` defaults to 1.
    """

    path: str
    kind: AxisKind = "categorical"
    # Categorical axes use `values`; numeric axes leave it empty.
    values: list[Any] = field(default_factory=list)
    # Numeric axes use these. Categorical axes leave them None.
    low: float | None = None
    high: float | None = None
    step: float | None = None

    def __post_init__(self) -> None:
        if self.kind == "categorical":
            if not self.values:
                msg = f"axis {self.path!r} (categorical) needs at least one value"
                raise SweepError(msg)
            return
        # Numeric axes need a finite [low, high] range.
        if self.low is None or self.high is None:
            msg = f"axis {self.path!r} ({self.kind}) needs both `low` and `high`"
            raise SweepError(msg)
        if self.high <= self.low:
            msg = f"axis {self.path!r} ({self.kind}): low must be < high"
            raise SweepError(msg)
        if self.kind == "loguniform" and self.low <= 0:
            msg = f"axis {self.path!r} (loguniform): low must be > 0"
            raise SweepError(msg)

    def enumerate_grid(self) -> list[Any]:
        """Return a deterministic list of values for grid search.

        Categorical axes use `values` as-is. Numeric axes generate a
        finite list using `step` (or 1 for int_uniform). Loguniform
        without step gets two endpoints — grid mode + loguniform is
        a misuse and we'd rather emit something predictable than fail.
        """
        if self.kind == "categorical":
            return list(self.values)
        assert self.low is not None and self.high is not None
        if self.kind == "int_uniform":
            step = int(self.step) if self.step else 1
            return list(range(int(self.low), int(self.high) + 1, max(step, 1)))
        if self.kind == "uniform":
            if self.step is None or self.step <= 0:
                # Without a step, fall back to 5-point linspace so the
                # grid mode still does *something* with a continuous axis.
                return [
                    self.low + (self.high - self.low) * (i / 4)
                    for i in range(5)
                ]
            out: list[float] = []
            v = self.low
            while v <= self.high + 1e-12:
                out.append(round(v, 12))
                v += self.step
            return out
        # loguniform: log-spaced 5 points.
        log_low = math.log10(self.low)
        log_high = math.log10(self.high)
        return [10 ** (log_low + (log_high - log_low) * (i / 4)) for i in range(5)]

    def sample(self, rng: random.Random) -> Any:
        """Single random draw — used by `RandomSampler` and as a TPE fallback."""
        if self.kind == "categorical":
            return rng.choice(self.values)
        assert self.low is not None and self.high is not None
        if self.kind == "int_uniform":
            return rng.randint(int(self.low), int(self.high))
        if self.kind == "loguniform":
            log_low = math.log(self.low)
            log_high = math.log(self.high)
            return math.exp(rng.uniform(log_low, log_high))
        # uniform
        v = rng.uniform(self.low, self.high)
        if self.step:
            v = round(v / self.step) * self.step
        return v


# --------------------------------------------------------------------------- #
# Sampler protocol + implementations
# --------------------------------------------------------------------------- #


class Sampler(Protocol):
    """Protocol every sampler implementation honours.

    `suggest` returns the next variant's axis-value mapping; `report`
    feeds the trial's outcome back so adaptive samplers (TPE) can
    update their model. Non-adaptive samplers (grid / random) ignore
    `report`.
    """

    def suggest(self) -> dict[str, Any]: ...

    def report(self, axis_values: dict[str, Any], score: float) -> None: ...


class GridSampler:
    """Cartesian product over axis values.

    Stops emitting after the cartesian product is exhausted. The
    public API caps the size at ``SWEEP_MAX_VARIANTS``; we don't
    re-check here — the caller has already validated the count.
    """

    def __init__(self, axes: list[SweepAxis]) -> None:
        self._axes = axes
        self._iter = itertools.product(*(a.enumerate_grid() for a in axes))

    def suggest(self) -> dict[str, Any]:
        try:
            combo = next(self._iter)
        except StopIteration:  # pragma: no cover — caller bounds n_trials
            msg = "grid exhausted"
            raise SweepError(msg) from None
        return {axis.path: value for axis, value in zip(self._axes, combo, strict=True)}

    def report(self, axis_values: dict[str, Any], score: float) -> None:
        # Grid is non-adaptive: feedback ignored.
        del axis_values, score


class RandomSampler:
    """Independent draws from each axis's distribution."""

    def __init__(self, axes: list[SweepAxis], seed: int | None = None) -> None:
        self._axes = axes
        self._rng = random.Random(seed)

    def suggest(self) -> dict[str, Any]:
        return {axis.path: axis.sample(self._rng) for axis in self._axes}

    def report(self, axis_values: dict[str, Any], score: float) -> None:
        del axis_values, score


class OptunaTPESampler:
    """Optuna TPE — adaptive search guided by reported scores.

    The study minimises by default (lower loss is better); flip
    ``direction`` to ``"maximize"`` when wiring up a "higher score
    wins" metric.

    Persistence: when ``storage_path`` is provided, the study is backed
    by a SQLite file via Optuna's RDBStorage. Reopening the same path
    loads every prior trial — completed trials feed the TPE prior,
    RUNNING trials (left dangling by a server restart) are matched in
    :meth:`report` by their ``params`` so the metric still lands on
    the right trial. Without ``storage_path`` the study is in-memory
    only (the original cut1 behaviour, kept for tests).
    """

    def __init__(
        self,
        axes: list[SweepAxis],
        seed: int | None = None,
        direction: Literal["minimize", "maximize"] = "minimize",
        storage_path: Path | None = None,
        study_name: str | None = None,
    ) -> None:
        try:
            import optuna  # noqa: PLC0415
        except ImportError as exc:
            msg = (
                "TPE sweep mode requires optuna; install with "
                "`pip install lorahub[sweep]`"
            )
            raise SamplerUnavailableError(msg) from exc
        self._optuna = optuna
        self._axes = axes
        self._direction = direction
        sampler = optuna.samplers.TPESampler(seed=seed)
        # Silence Optuna's chatty per-trial INFO logger; we surface
        # progress through TrainingEvent / sweep_progress instead.
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        if storage_path is not None:
            storage_path.parent.mkdir(parents=True, exist_ok=True)
            # `sqlite:///abs_path` is what RDBStorage wants. On Windows
            # the absolute path starts with a drive letter, which Optuna
            # accepts as `sqlite:///C:/path/to.db`.
            url = f"sqlite:///{storage_path.as_posix()}"
            self._study = optuna.create_study(
                direction=direction,
                sampler=sampler,
                storage=url,
                study_name=study_name or "lorahub-sweep",
                load_if_exists=True,
            )
        else:
            self._study = optuna.create_study(
                direction=direction,
                sampler=sampler,
            )
        # Stash the in-flight Trial keyed by its frozen-axes tuple so
        # we can map a `report()` callback back to the right trial
        # even if the caller reports out of order. Restart wipes this
        # map; :meth:`report` falls back to a study-side scan in that
        # case, so the metric still lands on the right RUNNING trial.
        self._pending: dict[tuple[Any, ...], Any] = {}

    def suggest(self) -> dict[str, Any]:
        trial = self._study.ask()
        values: dict[str, Any] = {}
        for axis in self._axes:
            values[axis.path] = self._suggest_one(trial, axis)
        key = tuple(values[a.path] for a in self._axes)
        self._pending[key] = trial
        return values

    def report(self, axis_values: dict[str, Any], score: float) -> None:
        key = tuple(axis_values[a.path] for a in self._axes)
        trial = self._pending.pop(key, None)
        if trial is not None:
            self._study.tell(trial, score)
            return
        # Restart fallback: _pending is empty after a process restart,
        # but the RDB-backed study still has the dangling RUNNING trial.
        # Find it by params and tell by trial_id so the score lands on
        # the right row even though the live Trial object is gone.
        running = self._study.get_trials(
            states=(self._optuna.trial.TrialState.RUNNING,),
            deepcopy=False,
        )
        for t in running:
            if all(
                _params_match(t.params.get(a.path), axis_values.get(a.path))
                for a in self._axes
            ):
                self._study.tell(t.number, score)
                return
        # Truly orphan report (duplicate, or trial already completed):
        # nothing to do. Don't surface — the run is already done, this
        # is bookkeeping.

    def _suggest_one(self, trial: Any, axis: SweepAxis) -> Any:
        # Optuna API: each `suggest_*` keys off `axis.path` so a TPE
        # study sees a stable parameter name across trials.
        if axis.kind == "categorical":
            return trial.suggest_categorical(axis.path, axis.values)
        assert axis.low is not None and axis.high is not None
        if axis.kind == "int_uniform":
            return trial.suggest_int(axis.path, int(axis.low), int(axis.high), step=int(axis.step or 1))
        if axis.kind == "loguniform":
            return trial.suggest_float(axis.path, axis.low, axis.high, log=True)
        # uniform
        return trial.suggest_float(axis.path, axis.low, axis.high, step=axis.step)


def make_sampler(
    mode: SweepMode,
    axes: list[SweepAxis],
    *,
    seed: int | None = None,
    direction: Literal["minimize", "maximize"] = "minimize",
    storage_path: Path | None = None,
    study_name: str | None = None,
) -> Sampler:
    """Factory: pick the right Sampler for `mode`.

    Validation lives here (not in each Sampler's __init__) so the API
    layer hits a single error class when `mode` is bad. ``storage_path``
    + ``study_name`` are TPE-only — the other samplers ignore them.
    """
    if mode == "grid":
        return GridSampler(axes)
    if mode == "random":
        return RandomSampler(axes, seed=seed)
    if mode == "tpe":
        return OptunaTPESampler(
            axes,
            seed=seed,
            direction=direction,
            storage_path=storage_path,
            study_name=study_name,
        )
    msg = f"unknown sweep mode: {mode!r}"
    raise SweepError(msg)


def _params_match(a: Any, b: Any) -> bool:
    """True when an Optuna-stored param round-trips to the same value.

    RDBStorage stringifies floats via repr; reading back can land at
    e.g. 0.0001000000000001 vs 1e-4. Compare numerics with a relative
    tolerance and fall back to ``==`` for everything else.
    """
    if a is None or b is None:
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
    return bool(a == b)


# --------------------------------------------------------------------------- #
# Plan
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SweepPlan:
    """A search plan over ``base_config`` along the given ``axes``.

    `mode` selects the sampling strategy. `n_trials` caps the number
    of variants for non-grid modes; grid mode is bounded by the
    cartesian product (and ``SWEEP_MAX_VARIANTS``).

    `name_template` controls the per-variant ``output.name`` suffix.
    Two placeholders are recognised: ``{base}`` (the base config's
    ``output.name``) and ``{i}`` (the 1-based variant index, formatted
    with whatever spec the caller supplies — default is zero-padded
    three-digit).
    """

    base_config: dict[str, Any]
    axes: list[SweepAxis]
    name_template: str = "{base}-{i:03d}"
    mode: SweepMode = "grid"
    n_trials: int | None = None
    seed: int | None = None
    # Optional sqlite path for the TPE study. Reopening the same path
    # restores every prior trial; ignored for grid / random.
    storage_path: Path | None = None
    study_name: str | None = None

    # ------------------------------------------------------------- #
    # Static (offline) materialisation — used by grid + the legacy
    # "expand the whole batch up-front" workflow.
    # ------------------------------------------------------------- #

    def expand(self) -> list[tuple[str, dict[str, Any]]]:
        """Materialise variants up-front. Backwards-compatible alias.

        For ``mode="grid"`` (default) this enumerates the cartesian
        product. For ``"random"`` / ``"tpe"`` it draws ``n_trials``
        suggestions in one shot; TPE without feedback degrades to a
        single asks-then-no-tell sequence — the API layer should
        prefer :meth:`materialize` + :meth:`report_trial` so the
        sampler gets actual scores back.
        """
        return self._enumerate(self._effective_n_trials())

    def materialize(self, n_trials: int | None = None) -> MaterialisedSweep:
        """Stateful materialisation. Yields trials lazily.

        Callers iterate ``MaterialisedSweep`` to get one variant at a
        time, then push outcomes back via ``report_trial`` (only TPE
        cares; grid/random ignore the feedback). The returned object
        is what the API stores so subsequent /sweep webhooks can
        feed scores from finished jobs into the live study.
        """
        n = n_trials if n_trials is not None else self._effective_n_trials()
        return MaterialisedSweep(plan=self, n_trials=n)

    def axis_values_for(self, variant_index: int) -> dict[str, Any]:
        """Return the ``{axis.path: value}`` dict for the i-th GRID variant.

        Backwards-compat helper used by the API layer to record which
        combination each grid-spawned job corresponds to. Only valid
        for ``mode="grid"`` — random/tpe variants don't have a
        deterministic index → values mapping.
        """
        if self.mode != "grid":
            msg = "axis_values_for is only valid for grid sweeps"
            raise SweepError(msg)
        if variant_index < 1:
            msg = "variant_index is 1-based"
            raise ValueError(msg)
        idx = variant_index - 1
        value_grids = [axis.enumerate_grid() for axis in self.axes]
        combo = list(itertools.product(*value_grids))[idx]
        return {axis.path: value for axis, value in zip(self.axes, combo, strict=True)}

    # ------------------------------------------------------------- #
    # Internals
    # ------------------------------------------------------------- #

    def _effective_n_trials(self) -> int:
        if self.mode == "grid":
            total = 1
            for axis in self.axes:
                total *= len(axis.enumerate_grid())
            if total > SWEEP_MAX_VARIANTS:
                msg = (
                    f"sweep would produce {total} variants, exceeding the cap "
                    f"of {SWEEP_MAX_VARIANTS}; reduce axis sizes or split into "
                    "multiple sweeps"
                )
                raise SweepTooLargeError(msg)
            return total
        # random / tpe
        if self.n_trials is None or self.n_trials < 1:
            msg = f"sweep mode {self.mode!r} needs n_trials >= 1"
            raise SweepError(msg)
        if self.n_trials > SWEEP_MAX_VARIANTS:
            msg = (
                f"sweep n_trials={self.n_trials} exceeds cap of "
                f"{SWEEP_MAX_VARIANTS}; lower n_trials or raise the cap"
            )
            raise SweepTooLargeError(msg)
        return self.n_trials

    def _enumerate(self, n: int) -> list[tuple[str, dict[str, Any]]]:
        if not self.axes:
            msg = "sweep requires at least one axis"
            raise SweepError(msg)
        for axis in self.axes:
            _validate_path(self.base_config, axis.path)

        sampler = make_sampler(
            self.mode,
            list(self.axes),
            seed=self.seed,
            storage_path=self.storage_path,
            study_name=self.study_name,
        )
        base_name = (
            self.base_config.get("output", {}).get("name")
            if isinstance(self.base_config.get("output"), dict)
            else None
        ) or "sweep"
        out: list[tuple[str, dict[str, Any]]] = []
        for i in range(1, n + 1):
            axis_values = sampler.suggest()
            variant = copy.deepcopy(self.base_config)
            for path, value in axis_values.items():
                _set_by_path(variant, path, value)
            variant_name = self.name_template.format(base=base_name, i=i)
            variant.setdefault("output", {})
            if isinstance(variant["output"], dict):
                variant["output"]["name"] = variant_name
            out.append((variant_name, variant))
        return out


@dataclass
class MaterialisedSweep:
    """Stateful adapter around a SweepPlan + a live Sampler.

    The router uses this when running TPE: it calls ``next_variant``
    once per launch, then ``report_trial`` once each child job
    finishes so the sampler can ask better questions next time. For
    grid / random, ``report_trial`` still appends into
    ``reported_scores`` for introspection, but the sampler itself
    ignores the feedback (non-adaptive).

    Note on adaptive pruning (ASHA / Hyperband): each trial in the
    LoRA sweep is a *full* training subprocess, not an in-process
    iterator. We currently feed back the final score only; adopting
    Optuna's :class:`SuccessiveHalvingPruner` would require:

      1. Exposing the underlying ``optuna.trial.Trial`` through the
         Sampler protocol so the API layer can call
         ``trial.report(value, step)`` on every step / validation event.
      2. Plumbing ``trial.should_prune()`` checks into the sweep
         router and cancelling the corresponding child job mid-run.
      3. Deciding what to feed back when a trial is pruned —
         Optuna handles ``TrialState.PRUNED`` natively, but the
         current Sampler protocol only knows ``report(score)``.

    None of this is wired today; the architecture accepts only
    end-of-trial feedback. Step-level pruning is a TODO for a future
    cut once the subprocess-cancel pipeline is faster than launch
    overhead.
    """

    plan: SweepPlan
    n_trials: int
    _sampler: Sampler | None = None
    _emitted: int = 0
    # History of reported (axis_values, score) tuples — useful for
    # introspection from tests + the pareto endpoint, and for any
    # future restart-recovery hook that wants to replay the study.
    reported_scores: list[tuple[dict[str, Any], float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Validate paths once up front; running into a typo on trial 4
        # of a 50-trial study is much worse than rejecting at create
        # time.
        for axis in self.plan.axes:
            _validate_path(self.plan.base_config, axis.path)
        self._sampler = make_sampler(
            self.plan.mode,
            list(self.plan.axes),
            seed=self.plan.seed,
            storage_path=self.plan.storage_path,
            study_name=self.plan.study_name,
        )

    @property
    def sampler(self) -> Sampler:
        # Always exists after __post_init__.
        assert self._sampler is not None
        return self._sampler

    def remaining(self) -> int:
        return max(0, self.n_trials - self._emitted)

    def next_variant(self) -> tuple[str, dict[str, Any], dict[str, Any]] | None:
        """Yield (variant_name, config, axis_values) for the next trial.

        Returns ``None`` when the budget is exhausted; the API layer
        loops on this until exhausted, then closes the sweep record.
        """
        if self._emitted >= self.n_trials:
            return None
        axis_values = self.sampler.suggest()
        variant = copy.deepcopy(self.plan.base_config)
        for path, value in axis_values.items():
            _set_by_path(variant, path, value)
        base_name = (
            self.plan.base_config.get("output", {}).get("name")
            if isinstance(self.plan.base_config.get("output"), dict)
            else None
        ) or "sweep"
        variant.setdefault("output", {})
        idx = self._emitted + 1
        variant_name = self.plan.name_template.format(base=base_name, i=idx)
        if isinstance(variant["output"], dict):
            variant["output"]["name"] = variant_name
        self._emitted += 1
        return variant_name, variant, axis_values

    def report_trial(self, axis_values: dict[str, Any], score: float) -> None:
        """Feed a finished trial's outcome back to the sampler.

        Score semantics: lower is better (we minimise). Convert
        accuracy / IoU / etc. to a loss-shaped scalar before calling
        — the sampler doesn't know about your domain. ``float('inf')``
        is acceptable and signals a failed / metric-less trial; TPE
        treats it as a maximally bad sample and steers away from
        nearby regions.
        """
        self.reported_scores.append((dict(axis_values), float(score)))
        self.sampler.report(axis_values, score)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _set_by_path(target: dict[str, Any], dotted: str, value: Any) -> None:
    """Walk ``dotted`` into ``target`` and set the leaf to ``value``.

    Intermediate mappings are created on the fly; non-mapping intermediates
    raise :class:`SweepError`.
    """
    if not dotted:
        msg = "axis path must not be empty"
        raise SweepError(msg)
    parts = dotted.split(".")
    cursor: Any = target
    for part in parts[:-1]:
        if not isinstance(cursor, dict):
            msg = f"path {dotted!r} traverses non-mapping at {part!r}"
            raise SweepError(msg)
        nxt = cursor.get(part)
        if nxt is None:
            nxt = {}
            cursor[part] = nxt
        cursor = nxt
    if not isinstance(cursor, dict):
        msg = f"path {dotted!r} traverses non-mapping leaf"
        raise SweepError(msg)
    cursor[parts[-1]] = value


def _validate_path(base: dict[str, Any], dotted: str) -> None:
    """Confirm ``dotted`` resolves to a leaf in ``base``.

    Walks each segment and rejects paths that cross a non-mapping or
    name a key the base does not declare. Pure read-only — never
    mutates ``base``. Raises :class:`SweepError` on the first miss so
    the user sees exactly which axis is broken.
    """
    if not dotted:
        msg = "axis path must not be empty"
        raise SweepError(msg)
    cursor: Any = base
    parts = dotted.split(".")
    for i, part in enumerate(parts):
        if not isinstance(cursor, dict):
            head = ".".join(parts[:i]) or "<root>"
            msg = f"axis path {dotted!r} traverses non-mapping at {head!r}"
            raise SweepError(msg)
        if part not in cursor:
            head = ".".join(parts[: i + 1])
            msg = (
                f"axis path {dotted!r} does not resolve in base config "
                f"(missing key at {head!r})"
            )
            raise SweepError(msg)
        cursor = cursor[part]


__all__ = [
    "AxisKind",
    "GridSampler",
    "MaterialisedSweep",
    "OptunaTPESampler",
    "RandomSampler",
    "SWEEP_MAX_VARIANTS",
    "Sampler",
    "SamplerUnavailableError",
    "SweepAxis",
    "SweepError",
    "SweepMode",
    "SweepPlan",
    "SweepTooLargeError",
    "make_sampler",
]
