"""Tests for the diffusion-pipe compiler.

The compiler is a pure function: take a RecipeConfig + workspace, give back
``(argv, files_to_write)``. We exercise that contract by inspecting the TOML
strings we'd write to disk; we never actually shell out to diffusion-pipe.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lorahub.core.backends.diffusion_pipe.compiler import (
    CompilationError,
    compile_recipe,
)
from lorahub.core.config.schema import RecipeConfig


def _recipe(**overrides: object) -> RecipeConfig:
    base = {
        "base_model": {"arch": "flux", "checkpoint": "/m/flux"},
        "dataset": {"source": "/d/imgs"},
    }
    base.update(overrides)  # type: ignore[arg-type]
    return RecipeConfig.model_validate(base)


def _toml_repr(p: str | Path) -> str:
    """Mirror the compiler's TOML escaping so cross-platform tests work."""
    raw = str(Path(p))
    return raw.replace("\\", "\\\\").replace('"', '\\"')


def _compile(recipe: RecipeConfig, ws: Path = Path("/ws")) -> tuple[list[str], dict[Path, str]]:
    return compile_recipe(recipe, ws)


def _main_toml(recipe: RecipeConfig, ws: Path = Path("/ws")) -> str:
    _argv, files = _compile(recipe, ws)
    main_path = ws.resolve() / "diffusion_pipe.toml"
    return files[main_path]


def _dataset_toml(recipe: RecipeConfig, ws: Path = Path("/ws")) -> str:
    _argv, files = _compile(recipe, ws)
    ds_path = ws.resolve() / "dataset.toml"
    return files[ds_path]


def test_argv_uses_deepspeed_and_workspace_config(tmp_path: Path) -> None:
    argv, files = _compile(_recipe(), tmp_path)
    assert argv[0] == "--deepspeed"
    assert "--config" in argv
    cfg_path = Path(argv[argv.index("--config") + 1])
    assert cfg_path.is_absolute()
    assert cfg_path.name == "diffusion_pipe.toml"
    assert cfg_path in files


def test_two_files_written_under_workspace(tmp_path: Path) -> None:
    _argv, files = _compile(_recipe(), tmp_path)
    assert set(p.name for p in files) == {"diffusion_pipe.toml", "dataset.toml"}
    for p in files:
        assert tmp_path.resolve() in p.parents


def test_flux_recipe_emits_diffusers_path() -> None:
    cfg = _recipe(base_model={"arch": "flux", "checkpoint": "/models/FLUX.1-dev"})
    main = _main_toml(cfg)
    assert "type = \"flux\"" in main
    assert f'diffusers_path = "{_toml_repr("/models/FLUX.1-dev")}"' in main
    assert "checkpoint_path" not in main


def test_sdxl_recipe_emits_checkpoint_path() -> None:
    cfg = _recipe(base_model={"arch": "sdxl", "checkpoint": "/models/sdxl.safetensors"})
    main = _main_toml(cfg)
    assert "type = \"sdxl\"" in main
    assert f'checkpoint_path = "{_toml_repr("/models/sdxl.safetensors")}"' in main
    assert "diffusers_path" not in main


def test_sd3_recipe_emits_diffusers_path() -> None:
    cfg = _recipe(base_model={"arch": "sd3", "checkpoint": "/models/sd3"})
    main = _main_toml(cfg)
    assert "type = \"sd3\"" in main
    assert f'diffusers_path = "{_toml_repr("/models/sd3")}"' in main


def test_sd15_rejected_with_actionable_error() -> None:
    cfg = _recipe(base_model={"arch": "sd15", "checkpoint": "/m/sd15"})
    with pytest.raises(CompilationError, match="does not support arch"):
        _compile(cfg)


def test_adapter_section_uses_lora_with_rank() -> None:
    cfg = _recipe(network={"type": "lora", "rank": 64, "alpha": 32})
    main = _main_toml(cfg)
    assert "[adapter]" in main
    assert "type = 'lora'" in main
    assert "rank = 64" in main
    # diffusion-pipe forbids `alpha` in the toml; it forces alpha=rank.
    assert "alpha" not in main


def test_non_lora_network_rejected() -> None:
    cfg = _recipe(network={"type": "locon"})
    with pytest.raises(CompilationError, match="only supports network.type='lora'"):
        _compile(cfg)


def test_optimizer_maps_known_names() -> None:
    cfg = _recipe(optimizer={"type": "adamw_optimi", "lr": {"unet": 2e-5}})
    main = _main_toml(cfg)
    assert 'type = "adamw_optimi"' in main
    assert "lr = 2e-05" in main


def test_optimizer_unknown_passes_through_for_pytorch_optimizer() -> None:
    # diffusion-pipe falls back to pytorch_optimizer for unknown types,
    # so we shouldn't reject them at compile time.
    cfg = _recipe(optimizer={"type": "Lamb"})
    main = _main_toml(cfg)
    assert 'type = "Lamb"' in main


def test_schedule_cosine_with_restarts_collapses_to_cosine() -> None:
    cfg = _recipe(optimizer={"schedule": "cosine_with_restarts"})
    main = _main_toml(cfg)
    assert 'lr_scheduler = "cosine"' in main


def test_schedule_constant_omits_lr_scheduler_field() -> None:
    cfg = _recipe(optimizer={"schedule": "constant"})
    main = _main_toml(cfg)
    # diffusion-pipe defaults to constant when the field is absent.
    assert "lr_scheduler" not in main


def test_schedule_max_steps_passthrough() -> None:
    cfg = _recipe(schedule={"epochs": 5, "max_steps": 1234})
    main = _main_toml(cfg)
    assert "max_steps = 1234" in main


def test_save_dtype_translates_to_diffusion_pipe_names() -> None:
    cfg = _recipe(output={"name": "x", "save_dtype": "bf16"})
    main = _main_toml(cfg)
    assert 'save_dtype = "bfloat16"' in main


def test_dataset_resolution_single_value() -> None:
    cfg = _recipe(dataset={"source": "/d", "resolution": [512]})
    ds = _dataset_toml(cfg)
    assert "resolutions = [512]" in ds


def test_dataset_resolution_pair_uses_nested_array() -> None:
    cfg = _recipe(dataset={"source": "/d", "resolution": [1024, 768]})
    ds = _dataset_toml(cfg)
    assert "resolutions = [[1024, 768]]" in ds


def test_dataset_directory_section_includes_path_and_repeats() -> None:
    cfg = _recipe(dataset={"source": "/d/imgs", "num_repeats": 3})
    ds = _dataset_toml(cfg)
    assert "[[directory]]" in ds
    assert f'path = "{_toml_repr("/d/imgs")}"' in ds
    assert "num_repeats = 3" in ds


def test_dataset_ar_bucket_toggle() -> None:
    on = _dataset_toml(_recipe())
    assert "enable_ar_bucket = true" in on
    assert "min_ar = 0.5" in on
    cfg = _recipe(dataset={"source": "/d", "bucket": {"enabled": False}})
    off = _dataset_toml(cfg)
    assert "enable_ar_bucket = false" in off
    assert "min_ar" not in off


def test_main_toml_points_at_dataset_toml(tmp_path: Path) -> None:
    main = _main_toml(_recipe(), ws=tmp_path)
    expected = tmp_path.resolve() / "dataset.toml"
    # Path is escaped for TOML; check the basename + parent appear.
    assert "dataset.toml" in main
    assert str(expected.parent).replace("\\", "\\\\") in main or str(
        expected.parent
    ) in main


def test_output_dir_defaults_under_workspace(tmp_path: Path) -> None:
    main = _main_toml(_recipe(), ws=tmp_path)
    assert "output_dir" in main
    assert "output" in main


def test_activation_checkpointing_follows_recipe_flag() -> None:
    on = _main_toml(_recipe(gradient_checkpointing=True))
    assert "activation_checkpointing = true" in on
    off = _main_toml(_recipe(gradient_checkpointing=False))
    assert "activation_checkpointing = false" in off


# --------------------------------------------------------------------------- #
# DiffusionPipeOptions: backwards compat snapshot + per-field coverage
# --------------------------------------------------------------------------- #


def test_default_options_preserve_legacy_general_fields() -> None:
    """Without `backend.diffusion_pipe`, the toml retains the old hard-coded knobs."""
    main = _main_toml(_recipe())
    assert "pipeline_stages = 1" in main
    assert "gradient_clipping = 1.0" in main
    assert 'partition_method = "parameters"' in main
    assert "caching_batch_size = 1" in main
    assert "steps_per_print = 1" in main
    # Optional knobs default to off and should not appear.
    assert "blocks_to_swap" not in main
    assert "compile = " not in main


def test_blocks_to_swap_emitted_when_positive() -> None:
    cfg = _recipe(backend={"type": "diffusion-pipe", "diffusion_pipe": {"blocks_to_swap": 20}})
    main = _main_toml(cfg)
    assert "blocks_to_swap = 20" in main


def test_compile_flag_emitted_when_true() -> None:
    cfg = _recipe(backend={"type": "diffusion-pipe", "diffusion_pipe": {"compile": True}})
    main = _main_toml(cfg)
    assert "compile = true" in main


def test_partition_method_uniform_overrides_default() -> None:
    cfg = _recipe(
        backend={"type": "diffusion-pipe", "diffusion_pipe": {"partition_method": "uniform"}}
    )
    main = _main_toml(cfg)
    assert 'partition_method = "uniform"' in main


def test_eval_section_absent_by_default() -> None:
    main = _main_toml(_recipe())
    assert "eval_every_n_epochs" not in main
    assert "eval_before_first_step" not in main


def test_eval_section_emitted_when_every_n_epochs_set() -> None:
    cfg = _recipe(
        backend={
            "type": "diffusion-pipe",
            "diffusion_pipe": {
                "eval_every_n_epochs": 2,
                "eval_before_first_step": True,
                "eval_micro_batch_size_per_gpu": 4,
            },
        }
    )
    main = _main_toml(cfg)
    assert "eval_every_n_epochs = 2" in main
    assert "eval_before_first_step = true" in main
    assert "eval_micro_batch_size_per_gpu = 4" in main


def test_monitoring_section_disabled_by_default() -> None:
    main = _main_toml(_recipe())
    assert "[monitoring]" in main
    assert "enable_wandb = false" in main
    # Optional sub-keys absent.
    assert "wandb_tracker_name" not in main
    assert "wandb_run_name" not in main


def test_monitoring_section_with_wandb_keys() -> None:
    cfg = _recipe(
        backend={
            "type": "diffusion-pipe",
            "diffusion_pipe": {
                "enable_wandb": True,
                "tracker_name": "lorahub_runs",
                "run_name": "exp-42",
            },
        }
    )
    main = _main_toml(cfg)
    assert "enable_wandb = true" in main
    assert 'wandb_tracker_name = "lorahub_runs"' in main
    assert 'wandb_run_name = "exp-42"' in main
    # Secret intentionally not in the recipe.
    assert "wandb_api_key" not in main


# --------------------------------------------------------------------------- #
# Dataset-level dp options
# --------------------------------------------------------------------------- #


def test_dataset_ar_defaults_match_legacy_values() -> None:
    ds = _dataset_toml(_recipe())
    assert "min_ar = 0.5" in ds
    assert "max_ar = 2.0" in ds
    assert "num_ar_buckets = 7" in ds


def test_dataset_ar_overrides_via_options() -> None:
    cfg = _recipe(
        backend={
            "type": "diffusion-pipe",
            "diffusion_pipe": {
                "min_ar": 0.25,
                "max_ar": 4.0,
                "num_ar_buckets": 9,
            },
        }
    )
    ds = _dataset_toml(cfg)
    assert "min_ar = 0.25" in ds
    assert "max_ar = 4.0" in ds
    assert "num_ar_buckets = 9" in ds


def test_dataset_cache_shuffle_and_skip_empty_caption_defaults() -> None:
    ds = _dataset_toml(_recipe())
    assert "cache_shuffle_num = 0" in ds
    assert "skip_empty_caption = true" in ds


def test_dataset_cache_shuffle_override() -> None:
    cfg = _recipe(
        backend={
            "type": "diffusion-pipe",
            "diffusion_pipe": {
                "cache_shuffle_num": 10,
                "skip_empty_caption": False,
            },
        }
    )
    ds = _dataset_toml(cfg)
    assert "cache_shuffle_num = 10" in ds
    assert "skip_empty_caption = false" in ds
def test_optimizer_betas_weight_decay_eps_render_to_toml() -> None:
    main = _main_toml(
        _recipe(optimizer={"betas": [0.95, 0.98], "weight_decay": 0.05, "eps": 1e-7})
    )
    assert "betas = [0.95, 0.98]" in main
    assert "weight_decay = 0.05" in main
    assert "eps = 1e-07" in main or "eps = 0.0000001" in main


def test_optimizer_args_extra_keys_render_to_toml() -> None:
    main = _main_toml(
        _recipe(optimizer={"optimizer_args": {"foreach": "true", "amsgrad": "false"}})
    )
    assert "foreach =" in main
    assert "amsgrad =" in main


# --------------------------------------------------------------------------- #
# Arch coverage: every supported dp arch renders a recognisable [model] type
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("arch", "dp_type"),
    [
        ("sdxl", "sdxl"),
        ("sd3", "sd3"),
        ("flux", "flux"),
        ("flux2", "flux2"),
        ("lumina", "lumina_2"),
        ("chroma", "chroma"),
        ("hidream", "hidream"),
        ("omnigen2", "omnigen2"),
        ("auraflow", "auraflow"),
        ("qwen_image", "qwen_image"),
        ("cosmos", "cosmos"),
        ("cosmos_predict2", "cosmos_predict2"),
        ("anima", "anima"),
        ("hunyuan_image", "hunyuan_image"),
        ("hunyuan_video", "hunyuan-video"),
        ("hunyuan_video_15", "hunyuan_video_15"),
        ("ltx_video", "ltx-video"),
        ("ltx2", "ltx2"),
        ("wan", "wan"),
        ("z_image", "z_image"),
        ("ernie_image", "ernie_image"),
    ],
)
def test_dp_arch_emits_correct_model_type(arch: str, dp_type: str) -> None:
    cfg = _recipe(base_model={"arch": arch, "checkpoint": "/m/ckpt"})
    main = _main_toml(cfg)
    assert f'type = "{dp_type}"' in main


def test_dp_rejects_kohya_only_arch_sd15() -> None:
    cfg = _recipe(base_model={"arch": "sd15", "checkpoint": "/m/sd15"})
    with pytest.raises(CompilationError, match="does not support arch"):
        _compile(cfg)


def test_dp_rejects_kohya_only_arch_sd2() -> None:
    cfg = _recipe(base_model={"arch": "sd2", "checkpoint": "/m/sd2"})
    with pytest.raises(CompilationError, match="does not support arch"):
        _compile(cfg)


# --------------------------------------------------------------------------- #
# DiffusionPipeOptions.model_paths -- arch-specific [model] path overrides
# --------------------------------------------------------------------------- #


def test_dp_model_paths_render_to_toml() -> None:
    """model_paths flatten into the [model] block as `key = "value"` lines."""
    cfg = _recipe(
        base_model={"arch": "anima", "checkpoint": "/m/anima"},
        backend={
            "type": "diffusion-pipe",
            "diffusion_pipe": {
                "model_paths": {
                    "transformer_path": "/x.safetensors",
                    "vae_path": "/v.safetensors",
                    "llm_path": "/llm",
                },
            },
        },
    )
    main = _main_toml(cfg)
    # The [model] block contains all three new path keys (TOML-escaped).
    assert 'transformer_path = "/x.safetensors"' in main
    assert 'vae_path = "/v.safetensors"' in main
    assert 'llm_path = "/llm"' in main


def test_dp_model_paths_override_default_diffusers_path() -> None:
    """Explicit `diffusers_path` in model_paths wins over the default."""
    cfg = _recipe(
        base_model={"arch": "flux", "checkpoint": "/auto/inferred"},
        backend={
            "type": "diffusion-pipe",
            "diffusion_pipe": {
                "model_paths": {"diffusers_path": "/explicit/override"},
            },
        },
    )
    main = _main_toml(cfg)
    # Only one diffusers_path line should remain, and it carries the override.
    assert main.count("diffusers_path =") == 1
    assert 'diffusers_path = "/explicit/override"' in main
    assert "/auto/inferred" not in main
