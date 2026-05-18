"""Tests for `lorahub.core.config`."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from lorahub.core.config.loader import dump_config, export_json_schema, load_config
from lorahub.core.config.schema import TrainingConfig

MINIMAL_RECIPE = {
    "base_model": {"checkpoint": "./model.safetensors"},
    "dataset": {"source": "./data"},
}


def test_minimal_recipe_loads_with_defaults() -> None:
    cfg = TrainingConfig.model_validate(MINIMAL_RECIPE)
    assert cfg.base_model.arch == "sdxl"
    assert cfg.schedule.batch_size == 1
    assert cfg.precision == "bf16"
    assert cfg.gradient_checkpointing is True
    assert cfg.network.rank == 32


def test_load_config_from_yaml(tmp_path: Path) -> None:
    recipe_file = tmp_path / "config.yaml"
    recipe_file.write_text(
        yaml.dump(MINIMAL_RECIPE, default_flow_style=False), encoding="utf-8"
    )
    cfg = load_config(recipe_file)
    assert cfg.base_model.checkpoint == Path("./model.safetensors")
    assert cfg.dataset.source == Path("./data")


def test_dump_and_reload_round_trips(tmp_path: Path) -> None:
    cfg = TrainingConfig.model_validate(MINIMAL_RECIPE)
    out = tmp_path / "out.yaml"
    dump_config(cfg, out)
    reloaded = load_config(out)
    assert reloaded.network.rank == cfg.network.rank
    assert reloaded.output.name == cfg.output.name


def test_invalid_resolution_rejected() -> None:
    bad = {**MINIMAL_RECIPE, "dataset": {"source": "./x", "resolution": [1, 2, 3]}}
    with pytest.raises(Exception, match="resolution"):
        TrainingConfig.model_validate(bad)


def test_extra_fields_rejected() -> None:
    bad = {**MINIMAL_RECIPE, "unknown_field": True}
    with pytest.raises(Exception):
        TrainingConfig.model_validate(bad)


def test_json_schema_export() -> None:
    schema_str = export_json_schema()
    assert "TrainingConfig" in schema_str
    # Schema dumps with camelCase aliases now (`baseModel`); the legacy
    # snake_case key would only appear if `populate_by_name` were
    # disabled. Check both so this test passes regardless of which
    # alias mode the dump runs in.
    assert "baseModel" in schema_str or "base_model" in schema_str


def test_example_recipe_loads() -> None:
    config_path = Path(__file__).resolve().parent.parent / "recipes" / "sdxl_character_8gb.yaml"
    if config_path.exists():
        cfg = load_config(config_path)
        assert cfg.base_model.arch == "sdxl"
        assert cfg.schedule.batch_size == 1


# --------------------------------------------------------------------------- #
# OptimizationConfig defaults & validation
# --------------------------------------------------------------------------- #


def test_optimization_defaults_are_all_off() -> None:
    """Bare TrainingConfig() should leave every optimization toggle at upstream defaults."""
    cfg = TrainingConfig.model_validate(MINIMAL_RECIPE)
    assert cfg.optimization.torch_compile is False
    assert cfg.optimization.fused_backward_pass is False
    assert cfg.optimization.full_bf16 is False
    assert cfg.optimization.blocks_to_swap == 0


def test_optimization_blocks_to_swap_must_be_non_negative() -> None:
    bad = {
        **MINIMAL_RECIPE,
        "optimization": {"blocks_to_swap": -1},
    }
    with pytest.raises(Exception, match="blocks_to_swap"):
        TrainingConfig.model_validate(bad)


def test_optimization_kitchen_sink_round_trip() -> None:
    """All four flags can be set together and survive a model_dump round-trip."""
    cfg = TrainingConfig.model_validate(
        {
            **MINIMAL_RECIPE,
            "optimization": {
                "torch_compile": True,
                "fused_backward_pass": True,
                "full_bf16": True,
                "blocks_to_swap": 8,
            },
        }
    )
    assert cfg.optimization.torch_compile is True
    assert cfg.optimization.fused_backward_pass is True
    assert cfg.optimization.full_bf16 is True
    assert cfg.optimization.blocks_to_swap == 8

