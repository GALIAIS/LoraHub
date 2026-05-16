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
    _, args, _files = compile_recipe(recipe, ws)
    return args


def _files(recipe: RecipeConfig, ws: Path = Path("/ws")) -> dict[Path, str]:
    _, _args, files = compile_recipe(recipe, ws)
    return files


def _dataset_toml(recipe: RecipeConfig, ws: Path = Path("/ws")) -> str:
    return next(iter(_files(recipe, ws).values()))


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
        s, _, _ = compile_recipe(cfg, tmp_path)
        assert s == script


def test_dataset_toml_emitted_with_dataset_config() -> None:
    args = _argv(_recipe())
    assert any(a.startswith("--dataset_config=") for a in args)
    assert not any(a.startswith("--train_data_dir=") for a in args)
    assert not any(a.startswith("--resolution=") for a in args)
    assert "--enable_bucket" not in args


def test_dataset_resolution_single_value() -> None:
    cfg = _recipe(dataset={"source": "/d", "resolution": [768]})
    toml = _dataset_toml(cfg)
    assert "resolution = 768" in toml


def test_dataset_resolution_pair() -> None:
    cfg = _recipe(dataset={"source": "/d", "resolution": [1024, 768]})
    toml = _dataset_toml(cfg)
    assert "resolution = [1024, 768]" in toml


def test_bucket_args_when_enabled() -> None:
    toml = _dataset_toml(_recipe())
    assert "enable_bucket = true" in toml
    assert "min_bucket_reso" in toml
    assert "max_bucket_reso" in toml


def test_bucket_args_omitted_when_disabled() -> None:
    cfg = _recipe(dataset={"source": "/d", "bucket": {"enabled": False}})
    toml = _dataset_toml(cfg)
    assert "enable_bucket" not in toml


def test_dataset_subset_includes_image_dir_and_repeats(tmp_path: Path) -> None:
    src = tmp_path / "imgs"
    src.mkdir()
    cfg = _recipe(dataset={"source": str(src), "num_repeats": 5})
    toml = _dataset_toml(cfg)
    # path is escaped for TOML; just confirm the basename appears and num_repeats lines up.
    assert "imgs" in toml
    assert "num_repeats = 5" in toml


def test_dataset_toml_path_is_under_workspace(tmp_path: Path) -> None:
    files = _files(_recipe(), ws=tmp_path)
    assert len(files) == 1
    toml_path = next(iter(files.keys()))
    assert toml_path.name == "dataset.toml"
    assert tmp_path.resolve() in toml_path.parents


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


def test_pony_variant_emits_clip_skip() -> None:
    cfg = _recipe(base_model={"arch": "sdxl", "arch_variant": "pony", "checkpoint": "/m.safetensors"})
    args = _argv(cfg)
    assert "--clip_skip=2" in args


def test_non_pony_variants_dont_emit_clip_skip() -> None:
    # Vanilla SDXL has no clip_skip flag.
    args = _argv(_recipe())
    assert not any(a.startswith("--clip_skip") for a in args)

    # Illustrious / NoobAI / Animagine intentionally don't add argv yet.
    for variant in ("illustrious", "noobai", "animagine"):
        cfg = _recipe(
            base_model={
                "arch": "sdxl",
                "arch_variant": variant,
                "checkpoint": "/m.safetensors",
            }
        )
        argv = _argv(cfg)
        assert not any(a.startswith("--clip_skip") for a in argv), variant


def test_validation_default_off() -> None:
    """val_split=0 (default) keeps validation argv off entirely."""
    args = _argv(_recipe())
    assert not any(a.startswith("--validation_split_percentage") for a in args)
    assert not any(a.startswith("--validate_every_n_epochs") for a in args)
    assert not any(a.startswith("--max_validation_steps") for a in args)


def test_validation_split_emits_kohya_flags() -> None:
    cfg = _recipe(
        dataset={"source": "/d", "val_split": 0.1},
        validation={"every_n_epochs": 2, "max_samples": 50},
    )
    args = _argv(cfg)
    assert "--validation_split_percentage=10" in args
    assert "--validate_every_n_epochs=2" in args
    assert "--max_validation_steps=50" in args


def test_validation_split_too_large_rejected() -> None:
    """val_split must stay strictly below 0.5 — pydantic rejects bigger values."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RecipeConfig.model_validate(
            {
                "base_model": {"checkpoint": "/m.safetensors"},
                "dataset": {"source": "/d", "val_split": 0.6},
            }
        )


def test_locon_emits_conv_dim_and_alpha() -> None:
    """locon recipes forward conv_dim/conv_alpha as `--network_args` keys."""
    cfg = _recipe(
        network={
            "type": "locon",
            "rank": 16,
            "alpha": 8,
            "conv_dim": 8,
            "conv_alpha": 4,
        }
    )
    args = _argv(cfg)
    assert "--network_args" in args
    idx = args.index("--network_args")
    network_args = args[idx + 1 :]
    assert "algo=locon" in network_args
    assert "conv_dim=8" in network_args
    assert "conv_alpha=4" in network_args


def test_loha_conv_alpha_optional() -> None:
    """conv_alpha unset means we don't emit the key (sd-scripts defaults it)."""
    cfg = _recipe(network={"type": "loha", "conv_dim": 8})
    args = _argv(cfg)
    idx = args.index("--network_args")
    network_args = args[idx + 1 :]
    assert "conv_dim=8" in network_args
    assert not any(a.startswith("conv_alpha=") for a in network_args)


def test_dropout_args_only_when_positive() -> None:
    """All three dropout knobs default to 0 and stay off the argv."""
    args_default = _argv(_recipe())
    if "--network_args" in args_default:
        idx = args_default.index("--network_args")
        rest = args_default[idx + 1 :]
        assert not any(a.startswith("dropout=") for a in rest)
        assert not any(a.startswith("rank_dropout=") for a in rest)
        assert not any(a.startswith("module_dropout=") for a in rest)

    cfg = _recipe(
        network={
            "type": "locon",
            "network_dropout": 0.1,
            "rank_dropout": 0.2,
            "module_dropout": 0.3,
        }
    )
    args = _argv(cfg)
    idx = args.index("--network_args")
    network_args = args[idx + 1 :]
    assert "dropout=0.1" in network_args
    assert "rank_dropout=0.2" in network_args
    assert "module_dropout=0.3" in network_args


def test_scale_weight_norms_is_top_level_flag() -> None:
    """scale_weight_norms goes on the top-level argv, not inside --network_args."""
    cfg = _recipe(network={"scale_weight_norms": 1.0})
    args = _argv(cfg)
    assert "--scale_weight_norms=1.0" in args
    # Make sure it isn't accidentally swept into --network_args
    if "--network_args" in args:
        idx = args.index("--network_args")
        rest = args[idx + 1 :]
        assert not any("scale_weight_norms" in a for a in rest)


def test_conv_dim_rejected_for_lora() -> None:
    """Plain `lora` doesn't have conv layers — schema rejects conv_dim."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RecipeConfig.model_validate(
            {
                "base_model": {"checkpoint": "/m.safetensors"},
                "dataset": {"source": "/d"},
                "network": {"type": "lora", "conv_dim": 8},
            }
        )


def test_conv_alpha_rejected_for_dora() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RecipeConfig.model_validate(
            {
                "base_model": {"checkpoint": "/m.safetensors"},
                "dataset": {"source": "/d"},
                "network": {"type": "dora", "conv_alpha": 4},
            }
        )
