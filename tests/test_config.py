"""Tests for `lorahub.core.config`."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from lorahub.core.config.loader import dump_recipe, export_json_schema, load_recipe
from lorahub.core.config.schema import RecipeConfig

MINIMAL_RECIPE = {
    "base_model": {"checkpoint": "./model.safetensors"},
    "dataset": {"source": "./data"},
}


def test_minimal_recipe_loads_with_defaults() -> None:
    cfg = RecipeConfig.model_validate(MINIMAL_RECIPE)
    assert cfg.base_model.arch == "sdxl"
    assert cfg.schedule.batch_size == 1
    assert cfg.precision == "bf16"
    assert cfg.gradient_checkpointing is True
    assert cfg.network.rank == 32


def test_load_recipe_from_yaml(tmp_path: Path) -> None:
    recipe_file = tmp_path / "recipe.yaml"
    recipe_file.write_text(
        yaml.dump(MINIMAL_RECIPE, default_flow_style=False), encoding="utf-8"
    )
    cfg = load_recipe(recipe_file)
    assert cfg.base_model.checkpoint == Path("./model.safetensors")
    assert cfg.dataset.source == Path("./data")


def test_dump_and_reload_round_trips(tmp_path: Path) -> None:
    cfg = RecipeConfig.model_validate(MINIMAL_RECIPE)
    out = tmp_path / "out.yaml"
    dump_recipe(cfg, out)
    reloaded = load_recipe(out)
    assert reloaded.network.rank == cfg.network.rank
    assert reloaded.output.name == cfg.output.name


def test_invalid_resolution_rejected() -> None:
    bad = {**MINIMAL_RECIPE, "dataset": {"source": "./x", "resolution": [1, 2, 3]}}
    with pytest.raises(Exception, match="resolution"):
        RecipeConfig.model_validate(bad)


def test_extra_fields_rejected() -> None:
    bad = {**MINIMAL_RECIPE, "unknown_field": True}
    with pytest.raises(Exception):
        RecipeConfig.model_validate(bad)


def test_json_schema_export() -> None:
    schema_str = export_json_schema()
    assert "RecipeConfig" in schema_str
    assert "base_model" in schema_str


def test_example_recipe_loads() -> None:
    recipe_path = Path(__file__).resolve().parent.parent / "recipes" / "sdxl_character_8gb.yaml"
    if recipe_path.exists():
        cfg = load_recipe(recipe_path)
        assert cfg.base_model.arch == "sdxl"
        assert cfg.schedule.batch_size == 1
