"""Tests for the SweepPlan / SweepAxis grid-search expander."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from lorahub.core.config.schema import TrainingConfig
from lorahub.core.sweep import (
    SWEEP_MAX_VARIANTS,
    SweepAxis,
    SweepError,
    SweepPlan,
    SweepTooLargeError,
)


def _base_config(tmp_path: Path) -> dict[str, Any]:
    """Minimal but TrainingConfig-valid recipe used as the sweep base."""
    ckpt = tmp_path / "model.safetensors"
    ckpt.write_bytes(b"")
    data = tmp_path / "data"
    data.mkdir()
    return {
        "base_model": {"checkpoint": str(ckpt)},
        "dataset": {"source": str(data)},
        "schedule": {"epochs": 1, "batch_size": 1},
        "sampling": {"enabled": False},
        "optimizer": {"lr": {"unet": 1.0e-4, "text_encoder": 5.0e-5}},
        "network": {"rank": 32, "alpha": 16},
        "output": {"name": "demo"},
    }


def test_cartesian_expand_order(tmp_path: Path) -> None:
    """Axis 0 anchors the outer loop; the last axis varies fastest.

    With axes [(rank, [16,32]), (lr, [1e-4, 5e-4])] the order is
    (16,1e-4) (16,5e-4) (32,1e-4) (32,5e-4).
    """
    base = _base_config(tmp_path)
    plan = SweepPlan(
        base_config=base,
        axes=[
            SweepAxis(path="network.rank", values=[16, 32]),
            SweepAxis(path="optimizer.lr.unet", values=[1.0e-4, 5.0e-4]),
        ],
    )
    variants = plan.expand()
    assert len(variants) == 4
    diffs = [
        (v["network"]["rank"], v["optimizer"]["lr"]["unet"]) for _name, v in variants
    ]
    assert diffs == [(16, 1.0e-4), (16, 5.0e-4), (32, 1.0e-4), (32, 5.0e-4)]
    # Default name template renders the 1-based zero-padded index.
    names = [name for name, _v in variants]
    assert names == ["demo-001", "demo-002", "demo-003", "demo-004"]


def test_dotted_path_setter_preserves_neighbours(tmp_path: Path) -> None:
    """Setting optimizer.lr.unet must not clobber optimizer.lr.text_encoder."""
    base = _base_config(tmp_path)
    plan = SweepPlan(
        base_config=base,
        axes=[SweepAxis(path="optimizer.lr.unet", values=[1.0e-4, 2.0e-4])],
    )
    for _name, variant in plan.expand():
        assert variant["optimizer"]["lr"]["text_encoder"] == 5.0e-5
        # Sibling fields outside the axis path are untouched.
        assert variant["network"]["rank"] == 32
        assert variant["schedule"]["epochs"] == 1


def test_too_many_variants_raises(tmp_path: Path) -> None:
    base = _base_config(tmp_path)
    # Two axes whose product exceeds the cap (17 * 16 = 272 > 256).
    plan = SweepPlan(
        base_config=base,
        axes=[
            SweepAxis(path="network.rank", values=list(range(1, 18))),
            SweepAxis(path="schedule.epochs", values=list(range(1, 17))),
        ],
    )
    with pytest.raises(SweepTooLargeError) as exc_info:
        plan.expand()
    assert str(SWEEP_MAX_VARIANTS) in str(exc_info.value)


def test_unknown_axis_path_raises(tmp_path: Path) -> None:
    base = _base_config(tmp_path)
    plan = SweepPlan(
        base_config=base,
        axes=[SweepAxis(path="optimizer.does_not_exist", values=[1, 2])],
    )
    with pytest.raises(SweepError) as exc_info:
        plan.expand()
    assert "does not resolve" in str(exc_info.value)


def test_each_variant_validates_against_recipe_schema(tmp_path: Path) -> None:
    """Every materialised variant must round-trip through TrainingConfig."""
    base = _base_config(tmp_path)
    plan = SweepPlan(
        base_config=base,
        axes=[
            SweepAxis(path="network.rank", values=[16, 32, 64]),
            SweepAxis(path="optimizer.lr.unet", values=[1.0e-4, 5.0e-4]),
        ],
    )
    variants = plan.expand()
    assert len(variants) == 6
    for variant_name, variant in variants:
        cfg = TrainingConfig.model_validate(variant)
        # The variant suffix is stamped onto output.name so checkpoints don't collide.
        assert cfg.output.name == variant_name


# --------------------------------------------------------------------------- #
# Random + TPE coverage (B4 cut1)
# --------------------------------------------------------------------------- #


def test_random_mode_yields_n_distinct_trials(tmp_path: Path) -> None:
    """Random sampler must respect n_trials and emit unique variant names."""
    from lorahub.core.sweep import SweepAxis, SweepPlan

    base = _base_config(tmp_path)
    plan = SweepPlan(
        base_config=base,
        axes=[
            SweepAxis(path="network.rank", kind="int_uniform", low=8, high=64),
            SweepAxis(
                path="optimizer.lr.unet",
                kind="loguniform",
                low=1e-5,
                high=1e-3,
            ),
        ],
        mode="random",
        n_trials=8,
        seed=42,
    )
    variants = plan.expand()
    assert len(variants) == 8
    names = [n for n, _ in variants]
    assert len(set(names)) == 8  # variant names are unique by construction


def test_int_uniform_axis_grid_enumerates_inclusive_range(tmp_path: Path) -> None:
    """Grid mode over an int_uniform axis must include both endpoints."""
    from lorahub.core.sweep import SweepAxis, SweepPlan

    base = _base_config(tmp_path)
    plan = SweepPlan(
        base_config=base,
        axes=[SweepAxis(path="network.rank", kind="int_uniform", low=4, high=8, step=2)],
        mode="grid",
    )
    variants = plan.expand()
    ranks = [v["network"]["rank"] for _, v in variants]
    assert ranks == [4, 6, 8]


def test_loguniform_axis_grid_spreads_in_log_space(tmp_path: Path) -> None:
    """Loguniform without an explicit step gets a 5-point log-spaced grid."""
    from lorahub.core.sweep import SweepAxis, SweepPlan

    base = _base_config(tmp_path)
    plan = SweepPlan(
        base_config=base,
        axes=[
            SweepAxis(
                path="optimizer.lr.unet",
                kind="loguniform",
                low=1e-5,
                high=1e-1,
            ),
        ],
        mode="grid",
    )
    variants = plan.expand()
    lrs = [v["optimizer"]["lr"]["unet"] for _, v in variants]
    assert len(lrs) == 5
    # Endpoints land exactly on low / high; the rest are log-spaced
    # within an order of magnitude of each step.
    assert lrs[0] == pytest.approx(1e-5)
    assert lrs[-1] == pytest.approx(1e-1)


def test_random_mode_requires_n_trials(tmp_path: Path) -> None:
    from lorahub.core.sweep import SweepAxis, SweepError, SweepPlan

    base = _base_config(tmp_path)
    plan = SweepPlan(
        base_config=base,
        axes=[SweepAxis(path="network.rank", values=[16, 32])],
        mode="random",
        # n_trials missing
    )
    with pytest.raises(SweepError, match="n_trials"):
        plan.expand()


def test_materialised_sweep_yields_then_exhausts(tmp_path: Path) -> None:
    """`MaterialisedSweep.next_variant` returns None after the budget."""
    from lorahub.core.sweep import SweepAxis, SweepPlan

    base = _base_config(tmp_path)
    plan = SweepPlan(
        base_config=base,
        axes=[SweepAxis(path="network.rank", values=[16, 32])],
        mode="random",
        n_trials=3,
        seed=0,
    )
    mat = plan.materialize()
    assert mat.remaining() == 3
    seen = []
    while True:
        nxt = mat.next_variant()
        if nxt is None:
            break
        seen.append(nxt)
    assert len(seen) == 3
    assert mat.remaining() == 0
    assert mat.next_variant() is None


def test_tpe_mode_drives_optuna_study(tmp_path: Path) -> None:
    """TPE sampler must round-trip suggestions + report through optuna."""
    pytest.importorskip("optuna", reason="optuna is optional")
    from lorahub.core.sweep import SweepAxis, SweepPlan

    base = _base_config(tmp_path)
    plan = SweepPlan(
        base_config=base,
        axes=[
            SweepAxis(path="network.rank", kind="int_uniform", low=8, high=64),
            SweepAxis(
                path="optimizer.lr.unet",
                kind="loguniform",
                low=1e-5,
                high=1e-3,
            ),
        ],
        mode="tpe",
        n_trials=4,
        seed=7,
    )
    mat = plan.materialize()
    # Drive the loop manually so we can inject scores; this is the
    # shape the API layer will use once a job's metric stream lands.
    scores = [0.5, 0.3, 0.2, 0.4]
    for s in scores:
        nxt = mat.next_variant()
        assert nxt is not None
        _, _, axis_values = nxt
        # rank stays in range, lr stays in range
        assert 8 <= axis_values["network.rank"] <= 64
        assert 1e-5 <= axis_values["optimizer.lr.unet"] <= 1e-3
        mat.report_trial(axis_values, s)
    assert mat.next_variant() is None


def test_tpe_without_optuna_raises_sampler_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When optuna isn't importable, TPE mode raises a clear error."""
    import sys

    from lorahub.core.sweep import SamplerUnavailableError, SweepAxis, SweepPlan

    base = _base_config(tmp_path)
    plan = SweepPlan(
        base_config=base,
        axes=[SweepAxis(path="network.rank", values=[16, 32])],
        mode="tpe",
        n_trials=2,
    )
    # Force the import inside OptunaTPESampler to fail.
    monkeypatch.setitem(sys.modules, "optuna", None)
    with pytest.raises(SamplerUnavailableError, match="optuna"):
        plan.materialize()
