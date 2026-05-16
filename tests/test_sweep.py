"""Tests for the SweepPlan / SweepAxis grid-search expander."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from lorahub.core.config.schema import RecipeConfig
from lorahub.core.sweep import (
    SWEEP_MAX_VARIANTS,
    SweepAxis,
    SweepError,
    SweepPlan,
    SweepTooLargeError,
)


def _base_recipe(tmp_path: Path) -> dict[str, Any]:
    """Minimal but RecipeConfig-valid recipe used as the sweep base."""
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
    base = _base_recipe(tmp_path)
    plan = SweepPlan(
        base_recipe=base,
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
    base = _base_recipe(tmp_path)
    plan = SweepPlan(
        base_recipe=base,
        axes=[SweepAxis(path="optimizer.lr.unet", values=[1.0e-4, 2.0e-4])],
    )
    for _name, variant in plan.expand():
        assert variant["optimizer"]["lr"]["text_encoder"] == 5.0e-5
        # Sibling fields outside the axis path are untouched.
        assert variant["network"]["rank"] == 32
        assert variant["schedule"]["epochs"] == 1


def test_too_many_variants_raises(tmp_path: Path) -> None:
    base = _base_recipe(tmp_path)
    # Two axes whose product exceeds the cap (17 * 16 = 272 > 256).
    plan = SweepPlan(
        base_recipe=base,
        axes=[
            SweepAxis(path="network.rank", values=list(range(1, 18))),
            SweepAxis(path="schedule.epochs", values=list(range(1, 17))),
        ],
    )
    with pytest.raises(SweepTooLargeError) as exc_info:
        plan.expand()
    assert str(SWEEP_MAX_VARIANTS) in str(exc_info.value)


def test_unknown_axis_path_raises(tmp_path: Path) -> None:
    base = _base_recipe(tmp_path)
    plan = SweepPlan(
        base_recipe=base,
        axes=[SweepAxis(path="optimizer.does_not_exist", values=[1, 2])],
    )
    with pytest.raises(SweepError) as exc_info:
        plan.expand()
    assert "does not resolve" in str(exc_info.value)


def test_each_variant_validates_against_recipe_schema(tmp_path: Path) -> None:
    """Every materialised variant must round-trip through RecipeConfig."""
    base = _base_recipe(tmp_path)
    plan = SweepPlan(
        base_recipe=base,
        axes=[
            SweepAxis(path="network.rank", values=[16, 32, 64]),
            SweepAxis(path="optimizer.lr.unet", values=[1.0e-4, 5.0e-4]),
        ],
    )
    variants = plan.expand()
    assert len(variants) == 6
    for variant_name, variant in variants:
        cfg = RecipeConfig.model_validate(variant)
        # The variant suffix is stamped onto output.name so checkpoints don't collide.
        assert cfg.output.name == variant_name
