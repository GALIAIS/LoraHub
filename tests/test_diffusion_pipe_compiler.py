"""Tests for the diffusion-pipe compiler.

The compiler is a pure function: take a TrainingConfig + workspace, give back
``(argv, files_to_write)``. We exercise that contract by inspecting the TOML
strings we'd write to disk; we never actually shell out to diffusion-pipe.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lorahub.core.backends.diffusion_pipe.compiler import (
    CompilationError,
    compile_config,
)
from lorahub.core.config.schema import TrainingConfig


def _config(**overrides: object) -> TrainingConfig:
    base = {
        "base_model": {"arch": "flux", "checkpoint": "/m/flux"},
        "dataset": {"source": "/d/imgs"},
    }
    base.update(overrides)  # type: ignore[arg-type]
    return TrainingConfig.model_validate(base)


def _toml_repr(p: str | Path) -> str:
    """Mirror the compiler's TOML escaping so cross-platform tests work."""
    raw = str(Path(p))
    return raw.replace("\\", "\\\\").replace('"', '\\"')


def _compile(cfg: TrainingConfig, ws: Path = Path("/ws")) -> tuple[list[str], dict[Path, str]]:
    return compile_config(cfg, ws)


def _main_toml(cfg: TrainingConfig, ws: Path = Path("/ws")) -> str:
    _argv, files = _compile(cfg, ws)
    main_path = ws.resolve() / "diffusion_pipe.toml"
    return files[main_path]


def _dataset_toml(cfg: TrainingConfig, ws: Path = Path("/ws")) -> str:
    _argv, files = _compile(cfg, ws)
    ds_path = ws.resolve() / "dataset.toml"
    return files[ds_path]


def test_argv_uses_deepspeed_and_workspace_config(tmp_path: Path) -> None:
    argv, files = _compile(_config(), tmp_path)
    assert argv[0] == "--deepspeed"
    assert "--config" in argv
    cfg_path = Path(argv[argv.index("--config") + 1])
    assert cfg_path.is_absolute()
    assert cfg_path.name == "diffusion_pipe.toml"
    assert cfg_path in files


def test_two_files_written_under_workspace(tmp_path: Path) -> None:
    _argv, files = _compile(_config(), tmp_path)
    assert set(p.name for p in files) == {"diffusion_pipe.toml", "dataset.toml"}
    for p in files:
        assert tmp_path.resolve() in p.parents


def test_flux_config_emits_diffusers_path() -> None:
    cfg = _config(base_model={"arch": "flux", "checkpoint": "/models/FLUX.1-dev"})
    main = _main_toml(cfg)
    assert "type = \"flux\"" in main
    assert f'diffusers_path = "{_toml_repr("/models/FLUX.1-dev")}"' in main
    assert "checkpoint_path" not in main


def test_sdxl_config_emits_checkpoint_path() -> None:
    cfg = _config(base_model={"arch": "sdxl", "checkpoint": "/models/sdxl.safetensors"})
    main = _main_toml(cfg)
    assert "type = \"sdxl\"" in main
    assert f'checkpoint_path = "{_toml_repr("/models/sdxl.safetensors")}"' in main
    assert "diffusers_path" not in main


def test_sd3_config_emits_diffusers_path() -> None:
    cfg = _config(base_model={"arch": "sd3", "checkpoint": "/models/sd3"})
    main = _main_toml(cfg)
    assert "type = \"sd3\"" in main
    assert f'diffusers_path = "{_toml_repr("/models/sd3")}"' in main


def test_sd15_rejected_with_actionable_error() -> None:
    cfg = _config(base_model={"arch": "sd15", "checkpoint": "/m/sd15"})
    with pytest.raises(CompilationError, match="does not support arch"):
        _compile(cfg)


def test_adapter_section_uses_lora_with_rank() -> None:
    cfg = _config(network={"type": "lora", "rank": 64, "alpha": 32})
    main = _main_toml(cfg)
    assert "[adapter]" in main
    assert "type = 'lora'" in main
    assert "rank = 64" in main
    # diffusion-pipe forbids `alpha` in the toml; it forces alpha=rank.
    assert "alpha" not in main


def test_non_lora_network_rejected() -> None:
    cfg = _config(network={"type": "locon"})
    with pytest.raises(CompilationError, match="only supports network.type='lora'"):
        _compile(cfg)


def test_optimizer_maps_known_names() -> None:
    cfg = _config(optimizer={"type": "adamw_optimi", "lr": {"unet": 2e-5}})
    main = _main_toml(cfg)
    assert 'type = "adamw_optimi"' in main
    assert "lr = 2e-05" in main


def test_optimizer_unknown_passes_through_for_pytorch_optimizer() -> None:
    # diffusion-pipe falls back to pytorch_optimizer for unknown types,
    # so we shouldn't reject them at compile time.
    cfg = _config(optimizer={"type": "Lamb"})
    main = _main_toml(cfg)
    assert 'type = "Lamb"' in main


def test_schedule_cosine_with_restarts_collapses_to_cosine() -> None:
    cfg = _config(optimizer={"schedule": "cosine_with_restarts"})
    main = _main_toml(cfg)
    assert 'lr_scheduler = "cosine"' in main


def test_schedule_constant_omits_lr_scheduler_field() -> None:
    cfg = _config(optimizer={"schedule": "constant"})
    main = _main_toml(cfg)
    # diffusion-pipe defaults to constant when the field is absent.
    assert "lr_scheduler" not in main


def test_schedule_max_steps_passthrough() -> None:
    cfg = _config(schedule={"epochs": 5, "max_steps": 1234})
    main = _main_toml(cfg)
    assert "max_steps = 1234" in main


def test_save_dtype_translates_to_diffusion_pipe_names() -> None:
    cfg = _config(output={"name": "x", "save_dtype": "bf16"})
    main = _main_toml(cfg)
    assert 'save_dtype = "bfloat16"' in main


def test_dataset_resolution_single_value() -> None:
    cfg = _config(dataset={"source": "/d", "resolution": [512]})
    ds = _dataset_toml(cfg)
    assert "resolutions = [512]" in ds


def test_dataset_resolution_pair_uses_nested_array() -> None:
    cfg = _config(dataset={"source": "/d", "resolution": [1024, 768]})
    ds = _dataset_toml(cfg)
    assert "resolutions = [[1024, 768]]" in ds


def test_dataset_directory_section_includes_path_and_repeats() -> None:
    cfg = _config(dataset={"source": "/d/imgs", "num_repeats": 3})
    ds = _dataset_toml(cfg)
    assert "[[directory]]" in ds
    assert f'path = "{_toml_repr("/d/imgs")}"' in ds
    assert "num_repeats = 3" in ds


def test_dataset_ar_bucket_toggle() -> None:
    on = _dataset_toml(_config())
    assert "enable_ar_bucket = true" in on
    assert "min_ar = 0.5" in on
    cfg = _config(dataset={"source": "/d", "bucket": {"enabled": False}})
    off = _dataset_toml(cfg)
    assert "enable_ar_bucket = false" in off
    assert "min_ar" not in off


def test_main_toml_points_at_dataset_toml(tmp_path: Path) -> None:
    main = _main_toml(_config(), ws=tmp_path)
    expected = tmp_path.resolve() / "dataset.toml"
    # Path is escaped for TOML; check the basename + parent appear.
    assert "dataset.toml" in main
    assert str(expected.parent).replace("\\", "\\\\") in main or str(
        expected.parent
    ) in main


def test_output_dir_defaults_under_workspace(tmp_path: Path) -> None:
    main = _main_toml(_config(), ws=tmp_path)
    assert "output_dir" in main
    assert "output" in main


def test_activation_checkpointing_follows_config_flag() -> None:
    on = _main_toml(_config(gradient_checkpointing=True))
    assert "activation_checkpointing = true" in on
    off = _main_toml(_config(gradient_checkpointing=False))
    assert "activation_checkpointing = false" in off


# --------------------------------------------------------------------------- #
# DiffusionPipeOptions: backwards compat snapshot + per-field coverage
# --------------------------------------------------------------------------- #


def test_default_options_preserve_legacy_general_fields() -> None:
    """Without `backend.diffusion_pipe`, the toml retains the old hard-coded knobs."""
    main = _main_toml(_config())
    assert "pipeline_stages = 1" in main
    assert "gradient_clipping = 1.0" in main
    assert 'partition_method = "parameters"' in main
    assert "caching_batch_size = 1" in main
    assert "steps_per_print = 1" in main
    # Optional knobs default to off and should not appear.
    assert "blocks_to_swap" not in main
    assert "compile = " not in main


def test_blocks_to_swap_emitted_when_positive() -> None:
    cfg = _config(backend={"type": "diffusion-pipe", "diffusion_pipe": {"blocks_to_swap": 20}})
    main = _main_toml(cfg)
    assert "blocks_to_swap = 20" in main


def test_compile_flag_emitted_when_true() -> None:
    cfg = _config(backend={"type": "diffusion-pipe", "diffusion_pipe": {"compile": True}})
    main = _main_toml(cfg)
    assert "compile = true" in main


def test_partition_method_uniform_overrides_default() -> None:
    cfg = _config(
        backend={"type": "diffusion-pipe", "diffusion_pipe": {"partition_method": "uniform"}}
    )
    main = _main_toml(cfg)
    assert 'partition_method = "uniform"' in main


def test_eval_section_absent_by_default() -> None:
    main = _main_toml(_config())
    assert "eval_every_n_epochs" not in main
    assert "eval_before_first_step" not in main


def test_eval_section_emitted_when_every_n_epochs_set() -> None:
    cfg = _config(
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
    main = _main_toml(_config())
    assert "[monitoring]" in main
    assert "enable_wandb = false" in main
    # Optional sub-keys absent.
    assert "wandb_tracker_name" not in main
    assert "wandb_run_name" not in main


def test_monitoring_section_with_wandb_keys() -> None:
    cfg = _config(
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
    # Secret intentionally not in the config.
    assert "wandb_api_key" not in main


# --------------------------------------------------------------------------- #
# Dataset-level dp options
# --------------------------------------------------------------------------- #


def test_dataset_ar_defaults_match_legacy_values() -> None:
    ds = _dataset_toml(_config())
    assert "min_ar = 0.5" in ds
    assert "max_ar = 2.0" in ds
    assert "num_ar_buckets = 7" in ds


def test_dataset_ar_overrides_via_options() -> None:
    cfg = _config(
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
    ds = _dataset_toml(_config())
    assert "cache_shuffle_num = 0" in ds
    assert "skip_empty_caption = true" in ds


def test_dataset_cache_shuffle_override() -> None:
    cfg = _config(
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
        _config(optimizer={"betas": [0.95, 0.98], "weight_decay": 0.05, "eps": 1e-7})
    )
    assert "betas = [0.95, 0.98]" in main
    assert "weight_decay = 0.05" in main
    assert "eps = 1e-07" in main or "eps = 0.0000001" in main


def test_optimizer_args_extra_keys_render_to_toml() -> None:
    main = _main_toml(
        _config(optimizer={"optimizer_args": {"foreach": "true", "amsgrad": "false"}})
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
    cfg = _config(base_model={"arch": arch, "checkpoint": "/m/ckpt"})
    main = _main_toml(cfg)
    assert f'type = "{dp_type}"' in main


def test_dp_rejects_kohya_only_arch_sd15() -> None:
    cfg = _config(base_model={"arch": "sd15", "checkpoint": "/m/sd15"})
    with pytest.raises(CompilationError, match="does not support arch"):
        _compile(cfg)


def test_dp_rejects_kohya_only_arch_sd2() -> None:
    cfg = _config(base_model={"arch": "sd2", "checkpoint": "/m/sd2"})
    with pytest.raises(CompilationError, match="does not support arch"):
        _compile(cfg)


# --------------------------------------------------------------------------- #
# DiffusionPipeOptions.model_paths -- arch-specific [model] path overrides
# --------------------------------------------------------------------------- #


def test_dp_model_paths_render_to_toml() -> None:
    """model_paths flatten into the [model] block as `key = "value"` lines."""
    cfg = _config(
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
    cfg = _config(
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


# --------------------------------------------------------------------------- #
# OptimizationConfig: torch_compile / fused_backward_pass are no-ops on dp;
# full_bf16 maps to optim_dtype="bf16"; blocks_to_swap top-level wins over
# the legacy `backend.diffusion_pipe.blocks_to_swap`.
# --------------------------------------------------------------------------- #


def test_dp_full_bf16_emits_optim_dtype() -> None:
    # Plain torch AdamW accepts optim_dtype, so full_bf16 must surface
    # it in the [optimizer] block.
    cfg = _config(
        optimization={"full_bf16": True},
        optimizer={"type": "adamw"},
    )
    main = _main_toml(cfg)
    assert 'optim_dtype = "bf16"' in main


def test_dp_full_bf16_default_omits_optim_dtype() -> None:
    main = _main_toml(_config())
    assert "optim_dtype" not in main


def test_dp_full_bf16_with_8bit_optimizer_omits_optim_dtype() -> None:
    # bitsandbytes AdamW8bit / its 4bit / lion_8bit cousins refuse the
    # optim_dtype kwarg (state is fp32 by design). full_bf16 must be a
    # no-op for these so dp doesn't crash with a TypeError.
    for q_type in ("adamw8bit", "lion8bit", "paged_adamw_8bit", "adamw8bitkahan"):
        cfg = _config(
            optimization={"full_bf16": True},
            optimizer={"type": q_type},
        )
        main = _main_toml(cfg)
        assert "optim_dtype" not in main, (
            f"{q_type} must not receive optim_dtype but compiler emitted it"
        )


def test_dp_blocks_to_swap_top_level_wins_over_dp_options() -> None:
    """`cfg.optimization.blocks_to_swap` overrides the dp-specific knob."""
    cfg = _config(
        optimization={"blocks_to_swap": 12},
        backend={
            "type": "diffusion-pipe",
            "diffusion_pipe": {"blocks_to_swap": 3},
        },
    )
    main = _main_toml(cfg)
    assert "blocks_to_swap = 12" in main
    assert "blocks_to_swap = 3" not in main


def test_dp_blocks_to_swap_legacy_field_still_works() -> None:
    """Old configs setting only `backend.diffusion_pipe.blocks_to_swap` still emit."""
    cfg = _config(
        backend={
            "type": "diffusion-pipe",
            "diffusion_pipe": {"blocks_to_swap": 7},
        },
    )
    main = _main_toml(cfg)
    assert "blocks_to_swap = 7" in main


def test_dp_blocks_to_swap_emitted_from_top_level() -> None:
    cfg = _config(optimization={"blocks_to_swap": 5})
    main = _main_toml(cfg)
    assert "blocks_to_swap = 5" in main


def test_dp_torch_compile_optimization_field_is_noop() -> None:
    """dp ignores `cfg.optimization.torch_compile` (its own `compile` flag still works)."""
    cfg = _config(optimization={"torch_compile": True})
    main = _main_toml(cfg)
    # No `compile = true` from the optimization knob alone.
    assert "compile = true" not in main


def test_dp_fused_backward_pass_optimization_field_is_noop() -> None:
    """dp has no fused-backward concept; the field shouldn't fail compilation."""
    cfg = _config(optimization={"fused_backward_pass": True})
    # Just verify it compiles cleanly without crashing.
    main = _main_toml(cfg)
    assert "fused_backward" not in main


def test_dp_optimization_kitchen_sink() -> None:
    # full_bf16 surfaces optim_dtype only when the optimizer accepts it —
    # explicit adamw here so the kitchen-sink assertion stays meaningful.
    cfg = _config(
        optimization={
            "torch_compile": True,
            "fused_backward_pass": True,
            "full_bf16": True,
            "blocks_to_swap": 9,
        },
        optimizer={"type": "adamw"},
    )
    main = _main_toml(cfg)
    assert "blocks_to_swap = 9" in main
    assert 'optim_dtype = "bf16"' in main
    # torch_compile / fused_backward_pass remain dp no-ops.
    assert "fused_backward" not in main


def test_sampling_attention_legacy_field_silently_ignored() -> None:
    """``sampling.attention`` was removed (schema-only knob, dp's
    eval/sample reuses the training attention kernel). Legacy YAML
    files carrying it must still load — pydantic's default
    ``extra="ignore"`` policy on SamplingConfig drops the unknown
    key. Resulting TOML stays byte-identical to a config that omits
    the field."""
    baseline = _main_toml(_config())
    cfg = _config(sampling={"attention": "sageattn"})
    main = _main_toml(cfg)
    assert main == baseline


# attention.training: dp auto-detects, so it's mostly advisory
# --------------------------------------------------------------------------- #


def test_attention_default_emits_no_marker_comment() -> None:
    """Vanilla configs shouldn't grow an attention comment."""
    main = _main_toml(_config())
    assert "attention.training" not in main


def test_attention_flash3_compiles_without_error() -> None:
    """`flash3` is purely advisory on dp — compile_config must succeed."""
    cfg = _config(attention={"training": "flash3"})
    argv, files = _compile(cfg, Path("/ws"))
    # argv shape stays unchanged
    assert argv[0] == "--deepspeed"
    assert "--config" in argv
    main_path = Path("/ws").resolve() / "diffusion_pipe.toml"
    assert "flash3" in files[main_path]


def test_attention_flash4_compiles_without_error() -> None:
    cfg = _config(attention={"training": "flash4"})
    argv, files = _compile(cfg, Path("/ws"))
    assert "--deepspeed" in argv
    main_path = Path("/ws").resolve() / "diffusion_pipe.toml"
    assert "flash4" in files[main_path]


def test_attention_xformers_logs_but_compiles(caplog: pytest.LogCaptureFixture) -> None:
    """dp doesn't honour xformers; log a warning but never error."""
    import logging

    cfg = _config(attention={"training": "xformers"})
    with caplog.at_level(logging.WARNING):
        argv, _files = _compile(cfg, Path("/ws"))
    assert "--deepspeed" in argv
    assert any("xformers" in rec.message for rec in caplog.records)


# --------------------------------------------------------------------------- #
# Batch B2: full coverage of every dp upstream TOML key the schema exposes.
# Each section keeps a "default omits" + "set value emits" pair so the
# byte-identical-default contract is regression-tested.
# --------------------------------------------------------------------------- #


# DEFAULT_KEYS_TO_NEVER_EMIT serves both as a regression check and a
# documentation of fields whose dp-only knob should NOT appear in the
# baseline TOML. If you add a new emit and intentionally make it default-on,
# remove the relevant entry here together with the test below.
DEFAULT_OMITTED_KEYS: list[str] = [
    "partition_split",
    "reentrant_activation_checkpointing",
    "disable_block_swap_for_eval",
    "image_micro_batch_size_per_gpu",
    "image_eval_micro_batch_size_per_gpu",
    "force_constant_lr",
    "uncond_fraction",
    "x_axis_examples",
    "logging_steps",
    "video_clip_mode",
    "map_num_proc",
    "save_every_n_steps",
    "save_every_n_examples",
    "checkpoint_every_n_epochs",
    "checkpoint_every_n_minutes",
    "pseudo_huber_c",
    "eval_every_n_steps",
    "eval_every_n_examples",
    "eval_gradient_accumulation_steps",
    "eval_datasets",
    "transformer_dtype",
    "diffusion_model_dtype",
    "timestep_sample_method",
    "init_from_existing",
    "fuse_adapters",
    "gradient_release",
]


def test_default_config_omits_every_optional_key() -> None:
    main = _main_toml(_config())
    for key in DEFAULT_OMITTED_KEYS:
        assert f"{key} =" not in main, f"unexpected {key} in default TOML"


# ---- Top-level main TOML new emits ---- #


def test_partition_split_emitted() -> None:
    cfg = _config(
        backend={
            "type": "diffusion-pipe",
            "diffusion_pipe": {"partition_split": [10, 20]},
        }
    )
    main = _main_toml(cfg)
    assert "partition_split = [10, 20]" in main


def test_reentrant_activation_checkpointing_emitted() -> None:
    cfg = _config(
        backend={
            "type": "diffusion-pipe",
            "diffusion_pipe": {"reentrant_activation_checkpointing": True},
        }
    )
    main = _main_toml(cfg)
    assert "reentrant_activation_checkpointing = true" in main


def test_disable_block_swap_for_eval_emitted() -> None:
    cfg = _config(
        backend={
            "type": "diffusion-pipe",
            "diffusion_pipe": {"disable_block_swap_for_eval": True},
        }
    )
    main = _main_toml(cfg)
    assert "disable_block_swap_for_eval = true" in main


def test_image_micro_batch_size_per_gpu_emitted() -> None:
    cfg = _config(
        backend={
            "type": "diffusion-pipe",
            "diffusion_pipe": {
                "image_micro_batch_size_per_gpu": 4,
                "image_eval_micro_batch_size_per_gpu": 2,
            },
        }
    )
    main = _main_toml(cfg)
    assert "image_micro_batch_size_per_gpu = 4" in main
    assert "image_eval_micro_batch_size_per_gpu = 2" in main


def test_force_constant_lr_emitted() -> None:
    cfg = _config(
        backend={
            "type": "diffusion-pipe",
            "diffusion_pipe": {"force_constant_lr": 1e-5},
        }
    )
    main = _main_toml(cfg)
    assert "force_constant_lr = 1e-05" in main or "force_constant_lr = 0.00001" in main


def test_uncond_fraction_emitted_only_when_positive() -> None:
    main = _main_toml(_config())
    assert "uncond_fraction" not in main
    cfg = _config(
        backend={
            "type": "diffusion-pipe",
            "diffusion_pipe": {"uncond_fraction": 0.1},
        }
    )
    main2 = _main_toml(cfg)
    assert "uncond_fraction = 0.1" in main2


def test_x_axis_examples_emitted() -> None:
    cfg = _config(
        backend={
            "type": "diffusion-pipe",
            "diffusion_pipe": {"x_axis_examples": True},
        }
    )
    main = _main_toml(cfg)
    assert "x_axis_examples = true" in main


def test_logging_steps_emitted_only_when_overridden() -> None:
    main = _main_toml(_config())
    assert "logging_steps" not in main  # default 1 is dp default
    cfg = _config(
        backend={
            "type": "diffusion-pipe",
            "diffusion_pipe": {"logging_steps": 10},
        }
    )
    main2 = _main_toml(cfg)
    assert "logging_steps = 10" in main2


def test_video_clip_mode_emitted_only_when_non_default() -> None:
    main = _main_toml(_config())
    assert "video_clip_mode" not in main  # default single_beginning is dp default
    cfg = _config(
        backend={
            "type": "diffusion-pipe",
            "diffusion_pipe": {"video_clip_mode": "single_middle"},
        }
    )
    main2 = _main_toml(cfg)
    assert 'video_clip_mode = "single_middle"' in main2


def test_map_num_proc_emitted_from_dataloader() -> None:
    cfg = _config(dataloader={"map_num_proc": 32})
    main = _main_toml(cfg)
    assert "map_num_proc = 32" in main


def test_save_every_n_steps_and_examples_emitted() -> None:
    cfg = _config(output={"save_every_n_steps": 100, "save_every_n_examples": 1000})
    main = _main_toml(cfg)
    assert "save_every_n_steps = 100" in main
    assert "save_every_n_examples = 1000" in main


def test_pseudo_huber_c_emitted() -> None:
    cfg = _config(loss={"pseudo_huber_c": 0.5})
    main = _main_toml(cfg)
    assert "pseudo_huber_c = 0.5" in main


def test_checkpoint_cadence_emitted() -> None:
    cfg = _config(
        backend={
            "type": "diffusion-pipe",
            "diffusion_pipe": {
                "checkpoint_every_n_epochs": 1,
                "checkpoint_every_n_minutes": 60,
            },
        }
    )
    main = _main_toml(cfg)
    assert "checkpoint_every_n_epochs = 1" in main
    assert "checkpoint_every_n_minutes = 60" in main


def test_eval_section_with_steps_and_examples() -> None:
    cfg = _config(
        backend={
            "type": "diffusion-pipe",
            "diffusion_pipe": {
                "eval_every_n_steps": 100,
                "eval_every_n_examples": 1000,
                "eval_gradient_accumulation_steps": 2,
            },
        }
    )
    main = _main_toml(cfg)
    assert "eval_every_n_steps = 100" in main
    assert "eval_every_n_examples = 1000" in main
    assert "eval_gradient_accumulation_steps = 2" in main


def test_eval_datasets_emitted_as_inline_table_array() -> None:
    cfg = _config(
        backend={
            "type": "diffusion-pipe",
            "diffusion_pipe": {
                "eval_datasets": [
                    {"name": "small", "config_path": "/eval/small.toml"},
                    {"name": "anime", "config_path": "/eval/anime.toml"},
                ],
            },
        }
    )
    main = _main_toml(cfg)
    assert (
        'eval_datasets = [{ name = "small", config = "/eval/small.toml" }, '
        '{ name = "anime", config = "/eval/anime.toml" }]' in main
    )


# ---- [model] section new emits ---- #


def test_transformer_dtype_emitted() -> None:
    cfg = _config(
        backend={
            "type": "diffusion-pipe",
            "diffusion_pipe": {"transformer_dtype": "float8_e4m3fn"},
        }
    )
    main = _main_toml(cfg)
    assert 'transformer_dtype = "float8_e4m3fn"' in main


def test_diffusion_model_dtype_emitted() -> None:
    cfg = _config(
        backend={
            "type": "diffusion-pipe",
            "diffusion_pipe": {"diffusion_model_dtype": "float8_e4m3fn"},
        }
    )
    main = _main_toml(cfg)
    assert 'diffusion_model_dtype = "float8_e4m3fn"' in main


def test_timestep_sample_method_emitted() -> None:
    cfg = _config(
        backend={
            "type": "diffusion-pipe",
            "diffusion_pipe": {"timestep_sample_method": "logit_normal"},
        }
    )
    main = _main_toml(cfg)
    assert 'timestep_sample_method = "logit_normal"' in main


# ---- ArchPathsConfig coverage ---- #


def test_arch_paths_render_to_model_section() -> None:
    cfg = _config(
        base_model={
            "arch": "anima",
            "checkpoint": "/m/anima",
            "arch_paths": {
                "transformer": "/p/transformer.safetensors",
                "llm": "/p/qwen3.safetensors",
                "qwen3": "/p/qwen3-base.safetensors",
                "t5_tokenizer": "/p/t5_tok",
                "llm_adapter": "/p/llm_adapter.safetensors",
            },
        }
    )
    main = _main_toml(cfg)
    assert f'transformer_path = "{_toml_repr("/p/transformer.safetensors")}"' in main
    assert f'llm_path = "{_toml_repr("/p/qwen3.safetensors")}"' in main
    assert f'qwen3_path = "{_toml_repr("/p/qwen3-base.safetensors")}"' in main
    assert f't5_tokenizer_path = "{_toml_repr("/p/t5_tok")}"' in main
    assert f'llm_adapter_path = "{_toml_repr("/p/llm_adapter.safetensors")}"' in main


def test_arch_paths_clip_t5_ae_render() -> None:
    cfg = _config(
        base_model={
            "arch": "flux",
            "checkpoint": "/m/flux",
            "arch_paths": {
                "clip_l": "/p/clip_l.safetensors",
                "clip_g": "/p/clip_g.safetensors",
                "t5xxl": "/p/t5xxl.safetensors",
                "ae": "/p/ae.safetensors",
                "byt5": "/p/byt5.safetensors",
                "text_encoder": "/p/te.safetensors",
            },
        }
    )
    main = _main_toml(cfg)
    assert f'clip_l_path = "{_toml_repr("/p/clip_l.safetensors")}"' in main
    assert f'clip_g_path = "{_toml_repr("/p/clip_g.safetensors")}"' in main
    assert f't5xxl_path = "{_toml_repr("/p/t5xxl.safetensors")}"' in main
    assert f'ae_path = "{_toml_repr("/p/ae.safetensors")}"' in main
    assert f'byt5_path = "{_toml_repr("/p/byt5.safetensors")}"' in main
    assert f'text_encoder_path = "{_toml_repr("/p/te.safetensors")}"' in main


def test_arch_paths_legacy_vae_renders_as_vae_path() -> None:
    cfg = _config(
        base_model={
            "arch": "flux",
            "checkpoint": "/m/flux",
            "vae": "/p/vae.safetensors",
        }
    )
    main = _main_toml(cfg)
    assert f'vae_path = "{_toml_repr("/p/vae.safetensors")}"' in main


def test_arch_paths_token_caps_emitted() -> None:
    cfg = _config(
        base_model={
            "arch": "flux",
            "checkpoint": "/m/flux",
            "arch_paths": {
                "t5xxl_max_token_length": 256,
                "qwen3_max_token_length": 1024,
                "t5_max_token_length": 512,
            },
        }
    )
    main = _main_toml(cfg)
    assert "t5xxl_max_token_length = 256" in main
    assert "qwen3_max_token_length = 1024" in main
    assert "t5_max_token_length = 512" in main


def test_arch_paths_attn_masks_and_dropouts() -> None:
    cfg = _config(
        base_model={
            "arch": "flux",
            "checkpoint": "/m/flux",
            "arch_paths": {
                "apply_t5_attn_mask": True,
                "apply_lg_attn_mask": True,
                "t5_dropout_rate": 0.1,
                "clip_l_dropout_rate": 0.05,
                "clip_g_dropout_rate": 0.07,
            },
        }
    )
    main = _main_toml(cfg)
    assert "apply_t5_attn_mask = true" in main
    assert "apply_lg_attn_mask = true" in main
    assert "t5_dropout_rate = 0.1" in main
    assert "clip_l_dropout_rate = 0.05" in main
    assert "clip_g_dropout_rate = 0.07" in main


def test_arch_paths_guidance_scale_and_vae_tweaks() -> None:
    cfg = _config(
        base_model={
            "arch": "flux",
            "checkpoint": "/m/flux",
            "arch_paths": {
                "guidance_scale": 1.0,
                "vae_chunk_size": 8,
                "text_encoder_cpu": True,
            },
        }
    )
    main = _main_toml(cfg)
    assert "guidance_scale = 1.0" in main
    assert "vae_chunk_size = 8" in main
    assert "text_encoder_cpu = true" in main


def test_arch_paths_default_unchanged_byte_identical() -> None:
    """A config with no arch_paths fields should produce zero new keys."""
    main = _main_toml(_config())
    assert "transformer_path" not in main
    assert "llm_path" not in main
    assert "vae_path" not in main
    assert "guidance_scale" not in main


def test_model_paths_legacy_overrides_arch_paths_collisions() -> None:
    """`model_paths` (free dict) wins over `arch_paths` for shared keys."""
    cfg = _config(
        base_model={
            "arch": "flux",
            "checkpoint": "/m/flux",
            "arch_paths": {"transformer": "/from/arch.safetensors"},
        },
        backend={
            "type": "diffusion-pipe",
            "diffusion_pipe": {
                "model_paths": {"transformer_path": "/from/legacy.safetensors"},
            },
        },
    )
    main = _main_toml(cfg)
    assert main.count("transformer_path =") == 1
    assert 'transformer_path = "/from/legacy.safetensors"' in main
    assert "/from/arch.safetensors" not in main


def test_model_paths_and_arch_paths_dont_clobber_distinct_keys() -> None:
    """When keys differ, both arch_paths and model_paths render."""
    cfg = _config(
        base_model={
            "arch": "flux",
            "checkpoint": "/m/flux",
            "arch_paths": {"transformer": "/p/transformer.safetensors"},
        },
        backend={
            "type": "diffusion-pipe",
            "diffusion_pipe": {
                "model_paths": {"single_file_path": "/p/single.safetensors"},
            },
        },
    )
    main = _main_toml(cfg)
    assert f'transformer_path = "{_toml_repr("/p/transformer.safetensors")}"' in main
    assert 'single_file_path = "/p/single.safetensors"' in main


# ---- [adapter] section new emits ---- #


def test_adapter_dtype_emitted() -> None:
    cfg = _config(network={"dtype": "bf16"})
    main = _main_toml(cfg)
    # Adapter block dtype key is plain `dtype`, not `lora_dtype`.
    assert "[adapter]" in main
    adapter_section = main.split("[adapter]", 1)[1].split("\n\n", 1)[0]
    assert 'dtype = "bfloat16"' in adapter_section


def test_adapter_init_from_existing_emitted() -> None:
    cfg = _config(network={"init_from": "/runs/prev/epoch5"})
    main = _main_toml(cfg)
    assert f'init_from_existing = "{_toml_repr("/runs/prev/epoch5")}"' in main


def test_adapter_fuse_adapters_inline_table_array() -> None:
    cfg = _config(
        network={
            "fuse_adapters": [
                {"path": "/loras/a.safetensors", "weight": 1.0},
                {"path": "/loras/b.safetensors", "multiplier": 0.5},
            ]
        }
    )
    main = _main_toml(cfg)
    assert (
        'fuse_adapters = [{ path = "/loras/a.safetensors", weight = 1.0 }, '
        '{ path = "/loras/b.safetensors", weight = 0.5 }]' in main
    )


# ---- [optimizer] section new emit ---- #


def test_optimizer_gradient_release_emitted() -> None:
    cfg = _config(optimizer={"gradient_release": True})
    main = _main_toml(cfg)
    assert "gradient_release = true" in main


# ---- Dataset TOML new emits ---- #


def test_dataset_frame_buckets_default_unchanged() -> None:
    ds = _dataset_toml(_config())
    assert "frame_buckets = [1]" in ds


def test_dataset_frame_buckets_video_emitted() -> None:
    cfg = _config(dataset={"source": "/d", "frame_buckets": [1, 33, 65]})
    ds = _dataset_toml(cfg)
    assert "frame_buckets = [1, 33, 65]" in ds


def test_dataset_subsets_emit_multiple_directory_blocks() -> None:
    cfg = _config(
        dataset={
            "source": "/d",  # ignored when subsets is non-empty
            "num_repeats": 99,  # also ignored
            "subsets": [
                {
                    "path": "/d/imgs",
                    "num_repeats": 3,
                    "mask_path": "/d/masks",
                    "caption_prefix": "anime style, ",
                },
                {
                    "path": "/d/extra",
                    "num_repeats": 1,
                    "ar_buckets": [1.0, 1.5],
                },
            ],
        }
    )
    ds = _dataset_toml(cfg)
    # Two [[directory]] blocks, no leftover single-source path.
    assert ds.count("[[directory]]") == 2
    assert _toml_repr("/d/imgs") in ds
    assert "num_repeats = 3" in ds
    assert _toml_repr("/d/masks") in ds
    assert 'caption_prefix = "anime style, "' in ds
    assert _toml_repr("/d/extra") in ds
    assert "ar_buckets = [1.0, 1.5]" in ds
    # Single-source legacy fields suppressed when subsets is set.
    assert "num_repeats = 99" not in ds


def test_dataset_explicit_ar_buckets_overrides_min_max() -> None:
    cfg = _config(
        dataset={"source": "/d", "bucket": {"ar_buckets": [1.0, 1.5, 2.0]}}
    )
    ds = _dataset_toml(cfg)
    assert "ar_buckets = [1.0, 1.5, 2.0]" in ds
    assert "min_ar" not in ds
    assert "max_ar" not in ds
    assert "num_ar_buckets" not in ds


def test_dataset_caption_shuffle_delimiter_renders_dp_key() -> None:
    cfg = _config(dataset={"source": "/d", "caption": {"shuffle_delimiter": " | "}})
    ds = _dataset_toml(cfg)
    # Note: dp uses `cache_shuffle_delimiter`, not `shuffle_delimiter`.
    assert 'cache_shuffle_delimiter = " | "' in ds
    assert "shuffle_delimiter =" not in ds.replace("cache_shuffle_delimiter", "")


def test_dataset_caption_shuffle_tags_legacy_flag() -> None:
    cfg = _config(dataset={"source": "/d", "caption": {"shuffle_tags": True}})
    ds = _dataset_toml(cfg)
    assert "shuffle_tags = true" in ds


# ---- Kohya-only field debug logging ---- #


def test_kohya_only_fields_logged_at_debug(caplog: pytest.LogCaptureFixture) -> None:
    """Setting a kohya-only field should produce a single debug log entry,
    no warnings, and no TOML drift."""
    import logging

    baseline = _main_toml(_config())
    cfg = _config(
        loss={"min_snr_gamma": 5.0},  # kohya-only
        augmentation={"flip": True},  # kohya-only
        optimization={"fp8_base": True},  # kohya-only
    )
    with caplog.at_level(logging.DEBUG, logger="lorahub.core.backends.diffusion_pipe.compiler"):
        main = _main_toml(cfg)
    # Kohya-only inputs shouldn't shape the dp TOML.
    assert main == baseline
    # And we shouldn't see a warning about them, only debug.
    assert not any(rec.levelno >= logging.WARNING for rec in caplog.records)
    debug_msgs = [rec.message for rec in caplog.records if rec.levelno == logging.DEBUG]
    audit_msg = next(
        (m for m in debug_msgs if "ignored" in m and "kohya-only" in m), None
    )
    assert audit_msg is not None, debug_msgs
    assert "loss.min_snr_gamma" in audit_msg
    assert "augmentation.flip" in audit_msg
    assert "optimization.fp8_base" in audit_msg

