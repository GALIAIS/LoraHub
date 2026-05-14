"""Tests for the kohya compiler."""

from __future__ import annotations

from pathlib import Path

import pytest

from lorahub.core.backends.kohya.compiler import CompilationError, compile_recipe
from lorahub.core.config.schema import RecipeConfig


def _recipe(**overrides: object) -> RecipeConfig:
    base = {
        "base_model": {"checkpoint": "/m/sdxl.safetensors"},
        "dataset": {"source": "/d/imgs"},
    }
    base.update(overrides)  # type: ignore[arg-type]
    return RecipeConfig.model_validate(base)


def _argv(recipe: RecipeConfig, ws: Path = Path("/ws")) -> list[str]:
    _, args = compile_recipe(recipe, ws)
    return args


def test_picks_correct_script_per_arch(tmp_path: Path) -> None:
    for arch, script in [
        ("sdxl", "sdxl_train_network.py"),
        ("sd15", "train_network.py"),
        ("flux", "flux_train_network.py"),
        ("sd3", "sd3_train_network.py"),
    ]:
        cfg = RecipeConfig.model_validate(
            {
                "base_model": {"arch": arch, "checkpoint": "/m.safetensors"},
                "dataset": {"source": "/d"},
            }
        )
        s, _ = compile_recipe(cfg, tmp_path)
        assert s == script


def test_dataset_resolution_single_value() -> None:
    cfg = _recipe(dataset={"source": "/d", "resolution": [768]})
    args = _argv(cfg)
    assert "--resolution=768" in args


def test_dataset_resolution_pair() -> None:
    cfg = _recipe(dataset={"source": "/d", "resolution": [1024, 768]})
    args = _argv(cfg)
    assert "--resolution=1024,768" in args


def test_bucket_args_when_enabled() -> None:
    args = _argv(_recipe())
    assert "--enable_bucket" in args
    assert any(a.startswith("--min_bucket_reso=") for a in args)
    assert any(a.startswith("--max_bucket_reso=") for a in args)


def test_bucket_args_omitted_when_disabled() -> None:
    cfg = _recipe(dataset={"source": "/d", "bucket": {"enabled": False}})
    args = _argv(cfg)
    assert "--enable_bucket" not in args


def test_network_lora_default() -> None:
    args = _argv(_recipe())
    assert "--network_module=networks.lora" in args
    assert "--network_dim=32" in args
    assert "--network_alpha=16" in args
    assert "--network_train_unet_only" in args


def test_network_locon_emits_algo() -> None:
    cfg = _recipe(network={"type": "locon", "rank": 16, "alpha": 8})
    args = _argv(cfg)
    assert "--network_module=lycoris.kohya" in args
    assert "--network_args" in args
    assert "algo=locon" in args


def test_optimizer_maps_adamw8bit() -> None:
    args = _argv(_recipe())
    assert "--optimizer_type=AdamW8bit" in args
    assert "--learning_rate=0.0001" in args
    assert "--unet_lr=0.0001" in args


def test_unknown_optimizer_rejected() -> None:
    cfg = _recipe(optimizer={"type": "made_up"})
    with pytest.raises(CompilationError):
        compile_recipe(cfg, Path("/ws"))


def test_precision_and_memory_flags() -> None:
    args = _argv(_recipe())
    assert "--mixed_precision=bf16" in args
    assert "--gradient_checkpointing" in args
    assert "--cache_latents" in args


def test_output_paths_use_workspace(tmp_path: Path) -> None:
    args = _argv(_recipe(), ws=tmp_path)
    assert f"--output_dir={tmp_path / 'output'}" in args
    assert f"--logging_dir={tmp_path / 'logs'}" in args
    assert "--save_model_as=safetensors" in args


def test_sampling_args_only_when_prompts_present() -> None:
    args = _argv(_recipe())
    assert not any(a.startswith("--sample_prompts=") for a in args)

    cfg = _recipe(sampling={"prompts_file": "/p/eval.txt"})
    args2 = _argv(cfg)
    assert "--sample_prompts=/p/eval.txt" in args2 or any(
        "eval.txt" in a for a in args2
    )


def test_extra_args_escape_hatch() -> None:
    cfg = _recipe(backend={"extra_args": {"seed": 1234, "noise_offset": 0.05, "xformers": True}})
    args = _argv(cfg)
    assert "--seed=1234" in args
    assert "--noise_offset=0.05" in args
    assert "--xformers" in args
