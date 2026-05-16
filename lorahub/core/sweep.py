"""Hyperparameter sweep — grid search over a base recipe.

A :class:`SweepPlan` takes a validated base recipe dict plus one or more
:class:`SweepAxis` declarations and materialises every cartesian-product
point as an independent recipe variant. Each variant carries a unique
``output.name`` suffix so the resulting workspaces, checkpoints, and
events.jsonl streams never collide when the API enqueues the batch.

Scope: this is the v1.0 first slice — pure cartesian grid only. Random /
Bayesian / TPE search are explicit follow-ups; see ``SWEEP_MAX_VARIANTS``
for the safety cap that protects users from an accidental 8-axis grid
detonation. The cap is intentionally conservative — the queue runs jobs
serially by default, so 256 cartesian points is already days of GPU time.
"""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass
from typing import Any

# Hard ceiling on the number of variants produced by a single SweepPlan.
# A single training job can take hours; the cap is a guard against an
# accidental "axis with 50 values times another with 20" detonation.
# Bumped only with care — the scheduler has no batch-cancel semantics yet.
SWEEP_MAX_VARIANTS = 256


class SweepError(ValueError):
    """Base class for sweep validation errors."""


class SweepTooLargeError(SweepError):
    """Raised when expand() would produce more than SWEEP_MAX_VARIANTS variants."""


@dataclass(frozen=True)
class SweepAxis:
    """One axis of the grid search.

    ``path`` is a dotted recipe path, e.g. ``optimizer.lr.unet`` or
    ``network.rank``. ``values`` are the values to enumerate. Each
    combination of axes' values produces one full recipe variant.
    """

    path: str
    values: list[Any]


@dataclass(frozen=True)
class SweepPlan:
    """A grid sweep over ``base_config`` along the given ``axes``.

    ``name_template`` controls the per-variant ``output.name`` suffix.
    Two placeholders are recognised: ``{base}`` (the base recipe's
    ``output.name``) and ``{i}`` (the 1-based variant index, formatted
    with whatever spec the caller supplies — default is zero-padded
    three-digit).
    """

    base_config: dict[str, Any]
    axes: list[SweepAxis]
    name_template: str = "{base}-{i:03d}"

    def expand(self) -> list[tuple[str, dict[str, Any]]]:
        """Return ``[(variant_name, recipe_dict), ...]`` for every grid point.

        Order is stable: the cartesian product walks axes in declaration
        order, with the last axis varying fastest (so axis 0 anchors the
        outer loop). Each returned recipe is an independent deep copy so
        callers can hand it straight to ``TrainingConfig.model_validate``
        without worrying about shared sub-dicts.

        The base ``output.name`` is read once from the validated base; if
        it is missing we fall back to ``"sweep"`` so the template still
        has something to interpolate.
        """
        if not self.axes:
            msg = "sweep requires at least one axis"
            raise SweepError(msg)
        for axis in self.axes:
            if not axis.values:
                msg = f"axis {axis.path!r} has no values"
                raise SweepError(msg)
            # Fail fast if a path can't be walked into the base recipe —
            # otherwise the user only learns of a typo after the API has
            # validated and enqueued the first N variants.
            _validate_path(self.base_config, axis.path)

        total = 1
        for axis in self.axes:
            total *= len(axis.values)
        if total > SWEEP_MAX_VARIANTS:
            msg = (
                f"sweep would produce {total} variants, exceeding the cap of "
                f"{SWEEP_MAX_VARIANTS}; reduce axis sizes or split into multiple sweeps"
            )
            raise SweepTooLargeError(msg)

        base_name = (
            self.base_config.get("output", {}).get("name")
            if isinstance(self.base_config.get("output"), dict)
            else None
        ) or "sweep"

        out: list[tuple[str, dict[str, Any]]] = []
        value_grids = [axis.values for axis in self.axes]
        for i, combo in enumerate(itertools.product(*value_grids), start=1):
            variant = copy.deepcopy(self.base_config)
            for axis, value in zip(self.axes, combo, strict=True):
                _set_by_path(variant, axis.path, value)
            variant_name = self.name_template.format(base=base_name, i=i)
            # Stamp the per-variant output.name so checkpoint files inherit
            # the variant suffix — otherwise N variants would clobber each
            # other when they share an output_dir.
            variant.setdefault("output", {})
            if isinstance(variant["output"], dict):
                variant["output"]["name"] = variant_name
            out.append((variant_name, variant))
        return out

    def axis_values_for(self, variant_index: int) -> dict[str, Any]:
        """Return the ``{axis.path: value}`` dict for the i-th variant (1-based).

        Used by the API layer to record which combination each spawned job
        corresponds to without having to re-diff the materialised recipe.
        """
        if variant_index < 1:
            msg = "variant_index is 1-based"
            raise ValueError(msg)
        idx = variant_index - 1
        value_grids = [axis.values for axis in self.axes]
        combo = list(itertools.product(*value_grids))[idx]
        return {axis.path: value for axis, value in zip(self.axes, combo, strict=True)}


# --------------------------------------------------------------------------- #
# Helpers
#
# `_set_by_path` mirrors `lorahub.api.recipe_templates._set_by_path` semantics
# but is duplicated here so `lorahub.core` does not have to import from
# `lorahub.api`. The two implementations should stay in sync; if one grows a
# feature, port it to the other.
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
                f"axis path {dotted!r} does not resolve in base recipe "
                f"(missing key at {head!r})"
            )
            raise SweepError(msg)
        cursor = cursor[part]


__all__ = [
    "SWEEP_MAX_VARIANTS",
    "SweepAxis",
    "SweepError",
    "SweepPlan",
    "SweepTooLargeError",
]
