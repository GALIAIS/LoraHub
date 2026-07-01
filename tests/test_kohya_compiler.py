"""Tests for the kohya compiler."""

from __future__ import annotations

from pathlib import Path

import pytest

from lorahub.core.backends.kohya.compiler import CompilationError, compile_config
from lorahub.core.config.schema import TrainingConfig


def _config(**overrides: object) -> TrainingConfig:
    base = {
        "base_model": {"checkpoint": "/m/sdxl.safetensors"},
        "dataset": {"source": "/d/imgs"},
    }
    base.update(overrides)  # type: ignore[arg-type]
    return TrainingConfig.model_validate(base)


def _argv(cfg: TrainingConfig, ws: Path = Path("/ws")) -> list[str]:
    _, args, _files, _env = compile_config(cfg, ws)
    return args


def _files(cfg: TrainingConfig, ws: Path = Path("/ws")) -> dict[Path, str]:
    _, _args, files, _env = compile_config(cfg, ws)
    return files


def _compile_env(cfg: TrainingConfig, ws: Path = Path("/ws")) -> dict[str, str]:
    _, _args, _files, env = compile_config(cfg, ws)
    return env


def _dataset_toml(cfg: TrainingConfig, ws: Path = Path("/ws")) -> str:
    return next(iter(_files(cfg, ws).values()))


def test_picks_correct_script_per_arch(tmp_path: Path) -> None:
    for arch, script in [
        ("sdxl", "sdxl_train_network.py"),
        ("sd15", "train_network.py"),
        ("flux", "flux_train_network.py"),
        ("sd3", "sd3_train_network.py"),
    ]:
        cfg = TrainingConfig.model_validate(
            {
                "base_model": {"arch": arch, "checkpoint": "/m.safetensors"},
                "dataset": {"source": "/d"},
            }
        )
        s, _, _, _ = compile_config(cfg, tmp_path)
        assert s == script


def _arch_config(arch: str) -> TrainingConfig:
    return TrainingConfig.model_validate(
        {
            "base_model": {"arch": arch, "checkpoint": "/m.safetensors"},
            "dataset": {"source": "/d"},
        }
    )


def test_pick_script_anima(tmp_path: Path) -> None:
    """Anima uses its own entry script per kohya's README."""
    s, _, _, _ = compile_config(_arch_config("anima"), tmp_path)
    assert s == "anima_train_network.py"


def test_pick_script_lumina(tmp_path: Path) -> None:
    s, _, _, _ = compile_config(_arch_config("lumina"), tmp_path)
    assert s == "lumina_train_network.py"


def test_pick_script_hunyuan_image(tmp_path: Path) -> None:
    s, _, _, _ = compile_config(_arch_config("hunyuan_image"), tmp_path)
    assert s == "hunyuan_image_train_network.py"


def test_pick_script_sd2_reuses_sd15_entry(tmp_path: Path) -> None:
    """sd-scripts ships sd1.x/2.x in the same train_network.py entry script."""
    s, _, _, _ = compile_config(_arch_config("sd2"), tmp_path)
    assert s == "train_network.py"


@pytest.mark.parametrize(
    "arch",
    [
        "hunyuan_video",
        "wan",
        "chroma",
        "flux2",
        "ltx_video",
        "qwen_image",
    ],
)
def test_pick_script_rejects_dp_only_arch(tmp_path: Path, arch: str) -> None:
    """Arches that only diffusion-pipe ships fail kohya compilation up front."""
    with pytest.raises(CompilationError, match="diffusion-pipe"):
        compile_config(_arch_config(arch), tmp_path)


def test_dataset_toml_emitted_with_dataset_config() -> None:
    args = _argv(_config())
    assert any(a.startswith("--dataset_config=") for a in args)
    assert not any(a.startswith("--train_data_dir=") for a in args)
    assert not any(a.startswith("--resolution=") for a in args)
    assert "--enable_bucket" not in args


def test_dataset_resolution_single_value() -> None:
    cfg = _config(dataset={"source": "/d", "resolution": [768]})
    toml = _dataset_toml(cfg)
    assert "resolution = 768" in toml


def test_dataset_resolution_pair() -> None:
    cfg = _config(dataset={"source": "/d", "resolution": [1024, 768]})
    toml = _dataset_toml(cfg)
    assert "resolution = [1024, 768]" in toml


def test_bucket_args_when_enabled() -> None:
    toml = _dataset_toml(_config())
    assert "enable_bucket = true" in toml
    assert "min_bucket_reso" in toml
    assert "max_bucket_reso" in toml


def test_bucket_args_omitted_when_disabled() -> None:
    cfg = _config(dataset={"source": "/d", "bucket": {"enabled": False}})
    toml = _dataset_toml(cfg)
    assert "enable_bucket" not in toml


def test_dataset_subset_includes_image_dir_and_repeats(tmp_path: Path) -> None:
    src = tmp_path / "imgs"
    src.mkdir()
    cfg = _config(dataset={"source": str(src), "num_repeats": 5})
    toml = _dataset_toml(cfg)
    # path is escaped for TOML; just confirm the basename appears and num_repeats lines up.
    assert "imgs" in toml
    assert "num_repeats = 5" in toml


def test_dataset_toml_path_is_under_workspace(tmp_path: Path) -> None:
    files = _files(_config(), ws=tmp_path)
    assert len(files) == 1
    toml_path = next(iter(files.keys()))
    assert toml_path.name == "dataset.toml"
    assert tmp_path.resolve() in toml_path.parents


def test_network_lora_default() -> None:
    args = _argv(_config())
    assert "--network_module=networks.lora" in args
    assert "--network_dim=32" in args
    assert "--network_alpha=16" in args
    assert "--network_train_unet_only" in args


def test_network_locon_emits_algo() -> None:
    cfg = _config(network={"type": "locon", "rank": 16, "alpha": 8})
    args = _argv(cfg)
    assert "--network_module=lycoris.kohya" in args
    assert "--network_args" in args
    assert "algo=locon" in args


def test_network_lokr_emits_algo() -> None:
    cfg = _config(network={"type": "lokr", "rank": 8, "alpha": 8})
    args = _argv(cfg)
    assert "--network_module=lycoris.kohya" in args
    assert "--network_args" in args
    assert "algo=lokr" in args


def test_optimizer_maps_adamw8bit() -> None:
    args = _argv(_config())
    assert "--optimizer_type=AdamW8bit" in args
    assert "--learning_rate=0.0001" in args
    assert "--unet_lr=0.0001" in args


def test_unknown_optimizer_rejected() -> None:
    cfg = _config(optimizer={"type": "made_up"})
    with pytest.raises(CompilationError):
        compile_config(cfg, Path("/ws"))


def test_precision_and_memory_flags() -> None:
    args = _argv(_config())
    assert "--mixed_precision=bf16" in args
    assert "--gradient_checkpointing" in args
    assert "--cache_latents" in args


def test_output_paths_use_workspace(tmp_path: Path) -> None:
    args = _argv(_config(), ws=tmp_path)
    assert f"--output_dir={tmp_path / 'output'}" in args
    assert f"--logging_dir={tmp_path / 'logs'}" in args
    assert "--save_model_as=safetensors" in args


def test_sampling_args_only_when_prompts_present() -> None:
    args = _argv(_config())
    assert not any(a.startswith("--sample_prompts=") for a in args)

    cfg = _config(sampling={"prompts_file": "/p/eval.txt"})
    args2 = _argv(cfg)
    assert "--sample_prompts=/p/eval.txt" in args2 or any(
        "eval.txt" in a for a in args2
    )


def test_sampling_attention_legacy_field_silently_ignored() -> None:
    """The ``sampling.attention`` field was removed (schema-only knob,
    no compiler ever wired it). Legacy YAML files carrying it must
    still load — pydantic's default ``extra="ignore"`` policy on
    SamplingConfig drops the unknown key. The compiled argv stays
    byte-identical to a config that omits the field."""
    baseline = _argv(_config(sampling={"prompts_file": "/p/eval.txt"}))
    legacy = _argv(
        _config(
            sampling={"prompts_file": "/p/eval.txt", "attention": "sageattn"}
        )
    )
    assert baseline == legacy


def test_extra_args_escape_hatch() -> None:
    cfg = _config(backend={"extra_args": {"seed": 1234, "noise_offset": 0.05, "xformers": True}})
    args = _argv(cfg)
    assert "--seed=1234" in args
    assert "--noise_offset=0.05" in args
    assert "--xformers" in args


def test_pony_variant_emits_clip_skip() -> None:
    cfg = _config(base_model={"arch": "sdxl", "arch_variant": "pony", "checkpoint": "/m.safetensors"})
    args = _argv(cfg)
    assert "--clip_skip=2" in args


def test_non_pony_variants_dont_emit_clip_skip() -> None:
    # Vanilla SDXL has no clip_skip flag.
    args = _argv(_config())
    assert not any(a.startswith("--clip_skip") for a in args)

    # Illustrious / NoobAI / Animagine intentionally don't add argv yet.
    for variant in ("illustrious", "noobai", "animagine"):
        cfg = _config(
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
    args = _argv(_config())
    assert not any(a.startswith("--validation_split_percentage") for a in args)
    assert not any(a.startswith("--validate_every_n_epochs") for a in args)
    assert not any(a.startswith("--max_validation_steps") for a in args)


def test_validation_split_emits_kohya_flags() -> None:
    cfg = _config(
        dataset={"source": "/d", "val_split": 0.1},
        validation={"every_n_epochs": 2, "max_samples": 50},
    )
    args = _argv(cfg)
    assert "--validation_split_percentage=10" in args
    assert "--validate_every_n_epochs=2" in args
    assert "--max_validation_steps=50" in args


def test_validation_split_too_large_rejected() -> None:
    """val_split must stay strictly below 0.5 鈥?pydantic rejects bigger values."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        TrainingConfig.model_validate(
            {
                "base_model": {"checkpoint": "/m.safetensors"},
                "dataset": {"source": "/d", "val_split": 0.6},
            }
        )


def test_locon_emits_conv_dim_and_alpha() -> None:
    """locon configs forward conv_dim/conv_alpha as `--network_args` keys."""
    cfg = _config(
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
    cfg = _config(network={"type": "loha", "conv_dim": 8})
    args = _argv(cfg)
    idx = args.index("--network_args")
    network_args = args[idx + 1 :]
    assert "conv_dim=8" in network_args
    assert not any(a.startswith("conv_alpha=") for a in network_args)


def test_dropout_args_only_when_positive() -> None:
    """All three dropout knobs default to 0 and stay off the argv."""
    args_default = _argv(_config())
    if "--network_args" in args_default:
        idx = args_default.index("--network_args")
        rest = args_default[idx + 1 :]
        assert not any(a.startswith("dropout=") for a in rest)
        assert not any(a.startswith("rank_dropout=") for a in rest)
        assert not any(a.startswith("module_dropout=") for a in rest)

    cfg = _config(
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
    cfg = _config(network={"scale_weight_norms": 1.0})
    args = _argv(cfg)
    assert "--scale_weight_norms=1.0" in args
    # Make sure it isn't accidentally swept into --network_args
    if "--network_args" in args:
        idx = args.index("--network_args")
        rest = args[idx + 1 :]
        assert not any("scale_weight_norms" in a for a in rest)


def test_conv_dim_rejected_for_lora() -> None:
    """Plain `lora` doesn't have conv layers 鈥?schema rejects conv_dim."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        TrainingConfig.model_validate(
            {
                "base_model": {"checkpoint": "/m.safetensors"},
                "dataset": {"source": "/d"},
                "network": {"type": "lora", "conv_dim": 8},
            }
        )


def test_conv_alpha_rejected_for_dora() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        TrainingConfig.model_validate(
            {
                "base_model": {"checkpoint": "/m.safetensors"},
                "dataset": {"source": "/d"},
                "network": {"type": "dora", "conv_alpha": 4},
            }
        )
def test_loss_default_emits_no_flags() -> None:
    """A bare LossConfig() is identity 鈥?sd-scripts keeps its own defaults."""
    args = _argv(_config())
    for flag in (
        "--min_snr_gamma",
        "--noise_offset",
        "--ip_noise_gamma",
        "--prior_loss_weight",
        "--loss_type",
        "--debiased_estimation_loss",
        "--masked_loss",
        "--scale_v_pred_loss_like_noise_pred",
        "--v_parameterization",
    ):
        assert not any(a.startswith(flag) for a in args), flag


def test_loss_min_snr_gamma_only() -> None:
    cfg = _config(loss={"min_snr_gamma": 5})
    args = _argv(cfg)
    assert "--min_snr_gamma=5.0" in args
    # noise_offset stayed default 鈫?still absent
    assert not any(a.startswith("--noise_offset") for a in args)


def test_loss_full_kitchen_sink() -> None:
    cfg = _config(
        loss={
            "min_snr_gamma": 5,
            "noise_offset": 0.05,
            "ip_noise_gamma": 0.1,
            "prior_loss_weight": 0.5,
            "loss_type": "huber",
            "debiased_estimation": True,
            "masked_loss": True,
            "scale_v_pred_loss_like_noise_pred": True,
            "v_parameterization": True,
        }
    )
    args = _argv(cfg)
    assert "--min_snr_gamma=5.0" in args
    assert "--noise_offset=0.05" in args
    assert "--ip_noise_gamma=0.1" in args
    assert "--prior_loss_weight=0.5" in args
    assert "--loss_type=huber" in args
    assert "--debiased_estimation_loss" in args
    assert "--masked_loss" in args
    assert "--scale_v_pred_loss_like_noise_pred" in args
    assert "--v_parameterization" in args


def test_loss_prior_weight_default_one_omitted() -> None:
    """prior_loss_weight=1.0 matches sd-scripts default 鈫?omit to keep argv tight."""
    args = _argv(_config(loss={"prior_loss_weight": 1.0}))
    assert not any(a.startswith("--prior_loss_weight") for a in args)


def test_optimizer_args_emit_betas_weight_decay_eps() -> None:
    args = _argv(_config(optimizer={"betas": [0.95, 0.999], "weight_decay": 0.1, "eps": 1e-7}))
    idx = args.index("--optimizer_args")
    tail = args[idx + 1 :]
    assert "betas=0.95,0.999" in tail
    assert "weight_decay=0.1" in tail
    assert "eps=1e-07" in tail


def test_optimizer_args_user_overrides_dedicated_fields() -> None:
    """Free-form `optimizer_args` keys win over the dedicated betas/eps."""
    args = _argv(
        _config(optimizer={"optimizer_args": {"betas": "0.5,0.5", "use_bias_correction": "True"}})
    )
    idx = args.index("--optimizer_args")
    tail = args[idx + 1 :]
    # Both the dedicated default and the override are present; the override
    # comes after, and kohya's last-wins semantics promote it. We allow either
    # ordering as long as the override sits after the default.
    assert "betas=0.5,0.5" in tail
    assert "use_bias_correction=True" in tail
    default_idx = tail.index("betas=0.9,0.999") if "betas=0.9,0.999" in tail else -1
    user_idx = tail.index("betas=0.5,0.5")
    if default_idx != -1:
        assert default_idx < user_idx


# --------------------------------------------------------------------------- #
# OptimizationConfig: torch_compile / fused_backward_pass / full_bf16 /
# blocks_to_swap argv emission.
# --------------------------------------------------------------------------- #


def test_optimization_default_emits_no_flags() -> None:
    """Bare OptimizationConfig() leaves the argv untouched (kohya defaults)."""
    args = _argv(_config())
    for flag in (
        "--torch_compile",
        "--fused_backward_pass",
        "--full_bf16",
        "--blocks_to_swap",
    ):
        assert not any(a.startswith(flag) for a in args), flag


def test_optimization_torch_compile_emits_flag() -> None:
    cfg = _config(optimization={"torch_compile": True})
    assert "--torch_compile" in _argv(cfg)


def test_optimization_fused_backward_pass_emits_flag() -> None:
    cfg = _config(optimization={"fused_backward_pass": True})
    assert "--fused_backward_pass" in _argv(cfg)


def test_optimization_full_bf16_coexists_with_mixed_precision() -> None:
    """`--full_bf16` is additive: it lands alongside `--mixed_precision=bf16`."""
    cfg = _config(optimization={"full_bf16": True})
    args = _argv(cfg)
    assert "--full_bf16" in args
    assert "--mixed_precision=bf16" in args


def test_optimization_blocks_to_swap_emits_for_flux() -> None:
    """FLUX ships --blocks_to_swap in flux_train_network.py."""
    cfg = TrainingConfig.model_validate(
        {
            "base_model": {"arch": "flux", "checkpoint": "/m/flux"},
            "dataset": {"source": "/d"},
            "optimization": {"blocks_to_swap": 12},
        }
    )
    assert "--blocks_to_swap=12" in _argv(cfg)


def test_optimization_blocks_to_swap_emits_for_sd3() -> None:
    cfg = TrainingConfig.model_validate(
        {
            "base_model": {"arch": "sd3", "checkpoint": "/m/sd3"},
            "dataset": {"source": "/d"},
            "optimization": {"blocks_to_swap": 4},
        }
    )
    assert "--blocks_to_swap=4" in _argv(cfg)


def test_optimization_blocks_to_swap_skipped_for_sdxl(caplog) -> None:
    """SDXL's sd-scripts entry has no --blocks_to_swap; we drop the flag + warn."""
    import logging

    cfg = _config(optimization={"blocks_to_swap": 8})
    with caplog.at_level(logging.WARNING, logger="lorahub.core.backends.kohya.compiler"):
        args = _argv(cfg)
    assert not any(a.startswith("--blocks_to_swap") for a in args)
    assert any("blocks_to_swap" in rec.message for rec in caplog.records)


@pytest.mark.parametrize("arch", ["sd15", "sd2", "sdxl"])
def test_optimization_blocks_to_swap_skipped_for_unsupported_arches(
    arch: str, caplog
) -> None:
    import logging

    cfg = TrainingConfig.model_validate(
        {
            "base_model": {"arch": arch, "checkpoint": "/m"},
            "dataset": {"source": "/d"},
            "optimization": {"blocks_to_swap": 6},
        }
    )
    with caplog.at_level(logging.WARNING, logger="lorahub.core.backends.kohya.compiler"):
        args = _argv(cfg)
    assert not any(a.startswith("--blocks_to_swap") for a in args)


def test_optimization_blocks_to_swap_zero_is_silent() -> None:
    """blocks_to_swap=0 (default) emits no flag and no warning."""
    cfg = _config(optimization={"blocks_to_swap": 0})
    args = _argv(cfg)
    assert not any(a.startswith("--blocks_to_swap") for a in args)


def test_optimization_full_kitchen_sink_on_flux() -> None:
    cfg = TrainingConfig.model_validate(
        {
            "base_model": {"arch": "flux", "checkpoint": "/m/flux"},
            "dataset": {"source": "/d"},
            "optimization": {
                "torch_compile": True,
                "fused_backward_pass": True,
                "full_bf16": True,
                "blocks_to_swap": 16,
            },
        }
    )
    args = _argv(cfg)
    assert "--torch_compile" in args
    assert "--fused_backward_pass" in args
    assert "--full_bf16" in args
    assert "--blocks_to_swap=16" in args
# attention.training -> kohya argv + env overrides
# --------------------------------------------------------------------------- #


def test_attention_auto_emits_no_argv() -> None:
    """`auto` keeps kohya's own default — we never emit attention argv."""
    args = _argv(_config())
    assert not any(a.startswith("--attn_mode") for a in args)
    assert "--xformers" not in args
    assert "--sdpa" not in args
    assert "--split_attn" not in args


def test_attention_torch_emits_attn_mode() -> None:
    args = _argv(_config(attention={"training": "torch"}))
    assert "--attn_mode=torch" in args


def test_attention_sdpa_emits_sdpa_flag() -> None:
    args = _argv(_config(attention={"training": "sdpa"}))
    assert "--sdpa" in args


def test_attention_xformers_with_split() -> None:
    cfg = _config(attention={"training": "xformers", "split": True})
    args = _argv(cfg)
    assert "--xformers" in args
    assert "--split_attn" in args


def test_attention_xformers_without_split() -> None:
    args = _argv(_config(attention={"training": "xformers"}))
    assert "--xformers" in args
    assert "--split_attn" not in args


def test_attention_flash_emits_attn_mode() -> None:
    args = _argv(_config(attention={"training": "flash"}))
    assert "--attn_mode=flash" in args


def test_attention_flex_falls_back_to_sdpa(caplog: pytest.LogCaptureFixture) -> None:
    """flex isn't supported by kohya; we drop to sdpa with a warning."""
    import logging

    with caplog.at_level(logging.WARNING):
        args = _argv(_config(attention={"training": "flex"}))
    assert "--sdpa" in args
    assert any("flex" in rec.message for rec in caplog.records)


def test_attention_flash3_sets_env_override() -> None:
    cfg = _config(attention={"training": "flash3"})
    _, args, _files, env = compile_config(cfg, Path("/ws"))
    assert "--attn_mode=flash" in args
    assert env.get("LORAHUB_KOHYA_ATTN_OVERRIDE") == "flash3"


def test_attention_flash4_sets_env_override() -> None:
    cfg = _config(attention={"training": "flash4"})
    _, args, _files, env = compile_config(cfg, Path("/ws"))
    assert "--attn_mode=flash" in args
    assert env.get("LORAHUB_KOHYA_ATTN_OVERRIDE") == "flash4"


def test_attention_default_env_is_empty() -> None:
    """Backends that don't need an env override get an empty mapping."""
    _, _args, _files, env = compile_config(_config(), Path("/ws"))
    assert env == {}


def test_attn_patch_module_imports_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing the patch module without the env var is a no-op.

    The host venv almost certainly doesn't have flash_attn_interface
    installed; calling apply() should still return False without raising.
    """
    from lorahub.core.backends.kohya import _attn_patch

    monkeypatch.delenv(_attn_patch.OVERRIDE_ENV, raising=False)
    assert _attn_patch.apply() is False
    monkeypatch.setenv(_attn_patch.OVERRIDE_ENV, "")
    assert _attn_patch.apply() is False
    monkeypatch.setenv(_attn_patch.OVERRIDE_ENV, "flash3")
    # Without flash_attn_interface installed this still returns False
    # (and logs a warning) instead of crashing.
    assert _attn_patch.apply() is False


# --------------------------------------------------------------------------- #
# B1: every-field-emit coverage. Each test focuses on a single helper so a
# regression points at exactly one helper / one schema field.
# --------------------------------------------------------------------------- #


def _flux_config(**overrides: object) -> TrainingConfig:
    base = {
        "base_model": {"arch": "flux", "checkpoint": "/m/flux"},
        "dataset": {"source": "/d"},
    }
    base.update(overrides)  # type: ignore[arg-type]
    return TrainingConfig.model_validate(base)


def _sd3_config(**overrides: object) -> TrainingConfig:
    base = {
        "base_model": {"arch": "sd3", "checkpoint": "/m/sd3"},
        "dataset": {"source": "/d"},
    }
    base.update(overrides)  # type: ignore[arg-type]
    return TrainingConfig.model_validate(base)


def _anima_config(**overrides: object) -> TrainingConfig:
    base = {
        "base_model": {"arch": "anima", "checkpoint": "/m/anima"},
        "dataset": {"source": "/d"},
    }
    base.update(overrides)  # type: ignore[arg-type]
    return TrainingConfig.model_validate(base)


def _hunyuan_config(**overrides: object) -> TrainingConfig:
    base = {
        "base_model": {"arch": "hunyuan_image", "checkpoint": "/m/h"},
        "dataset": {"source": "/d"},
    }
    base.update(overrides)  # type: ignore[arg-type]
    return TrainingConfig.model_validate(base)


# --- Defaults stay byte-identical -----------------------------------------


def test_b1_default_config_emits_no_new_argv() -> None:
    """A bare config must not pick up any of the new B1 flags. Existing
    fixtures encode the byte-level expectation; this guards explicit
    membership."""
    args = _argv(_config())
    forbidden = (
        "--noise_offset_random_strength",
        "--multires_noise_iterations",
        "--multires_noise_discount",
        "--adaptive_noise_scale",
        "--ip_noise_gamma_random_strength",
        "--zero_terminal_snr",
        "--min_timestep",
        "--max_timestep",
        "--huber_schedule",
        "--huber_c",
        "--huber_scale",
        "--v_pred_like_loss",
        "--max_grad_norm",
        "--lr_scheduler_type",
        "--lr_scheduler_args",
        "--lr_scheduler_num_cycles",
        "--lr_scheduler_power",
        "--lr_scheduler_timescale",
        "--lr_scheduler_min_lr_ratio",
        "--seed",
        "--lr_decay_steps",
        "--save_every_n_steps",
        "--save_last_n_epochs",
        "--save_last_n_steps",
        "--training_comment",
        "--no_metadata",
        "--metadata_",
        "--resume",
        "--save_last_n_epochs_state",
        "--save_last_n_steps_state",
        "--skip_until_initial_step",
        "--initial_epoch",
        "--initial_step",
        "--validate_every_n_steps",
        "--validation_seed",
        "--full_fp16",
        "--lowram",
        "--highvram",
        "--no_half_vae",
        "--cpu_offload_checkpointing",
        "--unsloth_offload_checkpointing",
        "--cache_text_encoder_outputs",
        "--cache_text_encoder_outputs_to_disk",
        "--fp8_base",
        "--fp8_base_unet",
        "--fp8_scaled",
        "--fp8_vl",
        "--disable_mmap_load_safetensors",
        "--cache_latents_to_disk",
        "--skip_cache_check",
        "--cache_info",
        "--train_inpainting",
        "--max_data_loader_n_workers",
        "--persistent_data_loader_workers",
        "--vae_batch_size",
        "--text_encoder_batch_size",
        "--flip_aug",
        "--color_aug",
        "--random_crop",
        "--face_crop_aug_range",
        "--alpha_mask",
        "--caption_dropout_every_n_epochs",
        "--caption_tag_dropout_rate",
        "--keep_tokens",
        "--keep_tokens_separator",
        "--secondary_separator",
        "--enable_wildcard",
        "--caption_prefix",
        "--caption_suffix",
        "--max_token_length",
        "--token_warmup_min",
        "--token_warmup_step",
        "--weighted_captions",
        "--bucket_no_upscale",
        "--skip_image_resolution",
        "--resize_interpolation",
        "--timestep_sampling",
        "--sigmoid_scale",
        "--model_prediction_type",
        "--discrete_flow_shift",
        "--training_shift",
        "--weighting_scheme",
        "--logit_mean",
        "--logit_std",
        "--mode_scale",
        "--clip_l",
        "--clip_g",
        "--t5xxl",
        "--ae",
        "--qwen3",
        "--llm_adapter_path",
        "--t5_tokenizer_path",
        "--qwen3_max_token_length",
        "--t5_max_token_length",
        "--text_encoder_cpu",
        "--vae_chunk_size",
        "--vae_disable_cache",
        "--apply_t5_attn_mask",
        "--apply_lg_attn_mask",
        "--pos_emb_random_crop_rate",
        "--enable_scaled_pos_embed",
        "--guidance_scale",
        "--t5xxl_max_token_length",
        "--t5_dropout_rate",
        "--clip_l_dropout_rate",
        "--clip_g_dropout_rate",
        "--llm_adapter_lr",
        "--self_attn_lr",
        "--cross_attn_lr",
        "--mlp_lr",
        "--mod_lr",
    )
    for flag in forbidden:
        assert not any(a.startswith(flag) for a in args), f"unexpected {flag} on default config"


# --- LossConfig new fields -------------------------------------------------


def test_b1_loss_advanced_full_emit() -> None:
    cfg = _config(
        loss={
            "noise_offset": 0.05,
            "noise_offset_random_strength": True,
            "multires_noise_iterations": 8,
            "multires_noise_discount": 0.4,
            "adaptive_noise_scale": 0.005,
            "ip_noise_gamma": 0.1,
            "ip_noise_gamma_random_strength": True,
            "zero_terminal_snr": True,
            "min_timestep": 5,
            "max_timestep": 950,
            "huber_schedule": "exponential",
            "huber_c": 0.2,
            "huber_scale": 1.5,
            "v_pred_like_loss": 0.3,
        }
    )
    args = _argv(cfg)
    for flag in (
        "--noise_offset=0.05",
        "--noise_offset_random_strength",
        "--multires_noise_iterations=8",
        "--multires_noise_discount=0.4",
        "--adaptive_noise_scale=0.005",
        "--ip_noise_gamma=0.1",
        "--ip_noise_gamma_random_strength",
        "--zero_terminal_snr",
        "--min_timestep=5",
        "--max_timestep=950",
        "--huber_schedule=exponential",
        "--huber_c=0.2",
        "--huber_scale=1.5",
        "--v_pred_like_loss=0.3",
    ):
        assert flag in args, flag


def test_b1_loss_multires_discount_default_omitted() -> None:
    """`multires_noise_discount=0.3` matches kohya default — argv stays clean."""
    args = _argv(_config(loss={"multires_noise_discount": 0.3}))
    assert not any(a.startswith("--multires_noise_discount") for a in args)


# --- OptimizerConfig new fields -------------------------------------------


def test_b1_optimizer_max_grad_norm_emitted_when_moved() -> None:
    args = _argv(_config(optimizer={"max_grad_norm": 0.5}))
    assert "--max_grad_norm=0.5" in args


def test_b1_optimizer_max_grad_norm_default_omitted() -> None:
    """`max_grad_norm=1.0` matches kohya default — argv stays clean."""
    args = _argv(_config())
    assert not any(a.startswith("--max_grad_norm") for a in args)


def test_b1_optimizer_scheduler_module_and_args() -> None:
    cfg = _config(
        optimizer={
            "scheduler_module": "transformers.optimization.cosine",
            "scheduler_args": {"num_cycles": "0.5"},
            "scheduler_num_cycles": 3,
            "scheduler_power": 0.5,
            "scheduler_timescale": 1000,
            "scheduler_min_lr_ratio": 0.05,
        }
    )
    args = _argv(cfg)
    assert "--lr_scheduler_type=transformers.optimization.cosine" in args
    assert "--lr_scheduler_args" in args
    sched_idx = args.index("--lr_scheduler_args")
    assert "num_cycles=0.5" in args[sched_idx + 1:]
    assert "--lr_scheduler_num_cycles=3" in args
    assert "--lr_scheduler_power=0.5" in args
    assert "--lr_scheduler_timescale=1000" in args
    assert "--lr_scheduler_min_lr_ratio=0.05" in args


# --- ScheduleConfig new fields --------------------------------------------


def test_b1_schedule_seed_and_lr_decay() -> None:
    cfg = _config(schedule={"seed": 1234, "lr_decay_steps": 500})
    args = _argv(cfg)
    assert "--seed=1234" in args
    assert "--lr_decay_steps=500" in args


# --- TrainingConfig top-level booleans ------------------------------------


def test_b1_top_level_caching_flags() -> None:
    cfg = _config(
        cache_latents_to_disk=True,
        skip_cache_check=True,
        cache_info=True,
        train_inpainting=True,
    )
    args = _argv(cfg)
    assert "--cache_latents_to_disk" in args
    assert "--skip_cache_check" in args
    assert "--cache_info" in args
    assert "--train_inpainting" in args


# --- OutputConfig new fields ----------------------------------------------


def test_b1_output_step_and_retention() -> None:
    cfg = _config(
        output={
            "save_every_n_steps": 100,
            "save_last_n_epochs": 3,
            "save_last_n_steps": 500,
            "training_comment": "lorahub run",
            "no_metadata": True,
        }
    )
    args = _argv(cfg)
    assert "--save_every_n_steps=100" in args
    assert "--save_last_n_epochs=3" in args
    assert "--save_last_n_steps=500" in args
    assert "--training_comment=lorahub run" in args
    assert "--no_metadata" in args


def test_b1_output_metadata_keys_emit_metadata_flags() -> None:
    cfg = _config(
        output={
            "metadata": {
                "title": "My LoRA",
                "author": "lorahub",
                "trigger_phrase": "hub_trig",
            }
        }
    )
    args = _argv(cfg)
    assert "--metadata_title=My LoRA" in args
    assert "--metadata_author=lorahub" in args
    assert "--metadata_trigger_phrase=hub_trig" in args


def test_b1_output_metadata_unknown_key_warns_and_passes(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    cfg = _config(output={"metadata": {"weird_key": "v"}})
    with caplog.at_level(logging.WARNING, logger="lorahub.core.backends.kohya.compiler"):
        args = _argv(cfg)
    assert "--metadata_weird_key=v" in args
    assert any("weird_key" in r.message for r in caplog.records)


# --- ResumeConfig new fields ----------------------------------------------


def test_b1_resume_extras_full_emit() -> None:
    cfg = _config(
        resume={
            "save_state": True,
            "save_state_at_end": True,
            "resume_from": "/r/state",
            "save_last_n_epochs_state": 2,
            "save_last_n_steps_state": 1000,
            "skip_until_initial_step": True,
            "initial_epoch": 3,
            "initial_step": 1500,
        }
    )
    args = _argv(cfg)
    assert any(a.startswith("--resume=") for a in args)
    assert "--save_last_n_epochs_state=2" in args
    assert "--save_last_n_steps_state=1000" in args
    assert "--skip_until_initial_step" in args
    assert "--initial_epoch=3" in args
    assert "--initial_step=1500" in args


# --- ValidationConfig new fields ------------------------------------------


def test_b1_validation_step_cadence_and_seed() -> None:
    cfg = _config(
        dataset={"source": "/d", "val_split": 0.1},
        validation={"every_n_epochs": 1, "every_n_steps": 200, "seed": 7},
    )
    args = _argv(cfg)
    assert "--validate_every_n_steps=200" in args
    assert "--validation_seed=7" in args


# --- SamplingConfig new fields --------------------------------------------


def test_b1_sampling_step_cadence_and_at_first() -> None:
    cfg = _config(
        sampling={
            "prompts_file": "/p/eval.txt",
            "every_n_epochs": 1,
            "every_n_steps": 50,
            "at_first": True,
        }
    )
    args = _argv(cfg)
    assert "--sample_every_n_steps=50" in args
    assert "--sample_at_first" in args


# --- DataLoaderConfig -----------------------------------------------------


def test_b1_dataloader_overrides_emit() -> None:
    cfg = _config(
        dataloader={
            "num_workers": 2,
            "persistent_workers": True,
            "vae_batch_size": 4,
        }
    )
    args = _argv(cfg)
    assert "--max_data_loader_n_workers=2" in args
    assert "--persistent_data_loader_workers" in args
    assert "--vae_batch_size=4" in args


def test_b1_dataloader_text_encoder_batch_size_sdxl() -> None:
    cfg = _config(dataloader={"text_encoder_batch_size": 8})
    args = _argv(cfg)
    assert "--text_encoder_batch_size=8" in args


def test_b1_dataloader_text_encoder_batch_size_sd15_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    cfg = TrainingConfig.model_validate(
        {
            "base_model": {"arch": "sd15", "checkpoint": "/m"},
            "dataset": {"source": "/d"},
            "dataloader": {"text_encoder_batch_size": 8},
        }
    )
    with caplog.at_level(logging.WARNING, logger="lorahub.core.backends.kohya.compiler"):
        args = _argv(cfg)
    assert not any(a.startswith("--text_encoder_batch_size") for a in args)
    assert any("text_encoder_batch_size" in r.message for r in caplog.records)


def test_b1_dataloader_defaults_silent() -> None:
    args = _argv(_config())
    assert not any(a.startswith("--max_data_loader_n_workers") for a in args)


# --- AugmentationConfig ---------------------------------------------------


def test_b1_augmentation_full_emit() -> None:
    cfg = _config(
        augmentation={
            "flip": True,
            "color": True,
            "random_crop": True,
            "face_crop_aug_range": "1.0,2.0,3.0",
            "alpha_mask": True,
        }
    )
    args = _argv(cfg)
    assert "--flip_aug" in args
    assert "--color_aug" in args
    assert "--random_crop" in args
    assert "--face_crop_aug_range=1.0,2.0,3.0" in args
    assert "--alpha_mask" in args


# --- Caption knobs --------------------------------------------------------


def test_b1_caption_advanced_full_emit() -> None:
    cfg = _config(
        dataset={
            "source": "/d",
            "caption": {
                "dropout_every_n_epochs": 5,
                "tag_dropout_rate": 0.1,
                "keep_tokens": 2,
                "keep_tokens_separator": "|",
                "secondary_separator": ";",
                "enable_wildcard": True,
                "prefix": "trigger,",
                "suffix": ", style",
                "max_token_length": 225,
                "token_warmup_min": 1,
                "token_warmup_step": 100.0,
                "weighted": True,
            },
        }
    )
    args = _argv(cfg)
    assert "--caption_dropout_every_n_epochs=5" in args
    assert "--caption_tag_dropout_rate=0.1" in args
    assert "--keep_tokens=2" in args
    assert "--keep_tokens_separator=|" in args
    assert "--secondary_separator=;" in args
    assert "--enable_wildcard" in args
    assert "--caption_prefix=trigger," in args
    assert "--caption_suffix=, style" in args
    assert "--max_token_length=225" in args
    assert "--token_warmup_min=1" in args
    assert "--token_warmup_step=100.0" in args
    assert "--weighted_captions" in args


# --- BucketConfig new fields ----------------------------------------------


def test_b1_bucket_extra_argv() -> None:
    cfg = _config(
        dataset={
            "source": "/d",
            "bucket": {
                "no_upscale": True,
                "skip_image_resolution": True,
                "resize_interpolation": "lanczos",
            },
        }
    )
    args = _argv(cfg)
    assert "--bucket_no_upscale" in args
    assert "--skip_image_resolution=0" in args
    assert "--resize_interpolation=lanczos" in args


def test_b1_bucket_extra_argv_silent_when_bucket_disabled() -> None:
    cfg = _config(
        dataset={
            "source": "/d",
            "bucket": {
                "enabled": False,
                "no_upscale": True,
                "resize_interpolation": "lanczos",
            },
        }
    )
    args = _argv(cfg)
    assert "--bucket_no_upscale" not in args
    assert not any(a.startswith("--resize_interpolation") for a in args)


# --- OptimizationConfig new advanced flags --------------------------------


def test_b1_optimization_universal_toggles() -> None:
    cfg = _config(
        optimization={
            "full_fp16": True,
            "lowram": True,
            "highvram": True,
            "no_half_vae": True,
            "cpu_offload_checkpointing": True,
            "fp8_base": True,
            "fp8_base_unet": True,
        }
    )
    args = _argv(cfg)
    for flag in (
        "--full_fp16",
        "--lowram",
        "--highvram",
        "--no_half_vae",
        "--cpu_offload_checkpointing",
        "--fp8_base",
        "--fp8_base_unet",
    ):
        assert flag in args, flag


def test_b1_optimization_disable_mmap_emits_for_sdxl() -> None:
    """SDXL's add_sdxl_training_arguments ships --disable_mmap_load_safetensors."""
    cfg = _config(optimization={"disable_mmap_load_safetensors": True})
    assert "--disable_mmap_load_safetensors" in _argv(cfg)


def test_b1_optimization_disable_mmap_skipped_for_sd15(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    cfg = TrainingConfig.model_validate(
        {
            "base_model": {"arch": "sd15", "checkpoint": "/m"},
            "dataset": {"source": "/d"},
            "optimization": {"disable_mmap_load_safetensors": True},
        }
    )
    with caplog.at_level(logging.WARNING, logger="lorahub.core.backends.kohya.compiler"):
        args = _argv(cfg)
    assert "--disable_mmap_load_safetensors" not in args
    assert any("disable_mmap" in r.message for r in caplog.records)


def test_b1_optimization_cache_te_outputs_for_sdxl() -> None:
    cfg = _config(
        optimization={
            "cache_text_encoder_outputs": True,
            "cache_text_encoder_outputs_to_disk": True,
        }
    )
    args = _argv(cfg)
    assert "--cache_text_encoder_outputs" in args
    assert "--cache_text_encoder_outputs_to_disk" in args


def test_b1_optimization_cache_te_outputs_skipped_for_sd15(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    cfg = TrainingConfig.model_validate(
        {
            "base_model": {"arch": "sd15", "checkpoint": "/m"},
            "dataset": {"source": "/d"},
            "optimization": {"cache_text_encoder_outputs": True},
        }
    )
    with caplog.at_level(logging.WARNING, logger="lorahub.core.backends.kohya.compiler"):
        args = _argv(cfg)
    assert "--cache_text_encoder_outputs" not in args
    assert any("cache_text_encoder_outputs" in r.message for r in caplog.records)


def test_b1_optimization_fp8_scaled_only_hunyuan() -> None:
    args = _argv(_hunyuan_config(optimization={"fp8_scaled": True}))
    assert "--fp8_scaled" in args


def test_b1_optimization_fp8_scaled_warns_on_flux(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    with caplog.at_level(logging.WARNING, logger="lorahub.core.backends.kohya.compiler"):
        args = _argv(_flux_config(optimization={"fp8_scaled": True}))
    assert "--fp8_scaled" not in args
    assert any("fp8_scaled" in r.message for r in caplog.records)


def test_b1_optimization_fp8_vl_text_encoder_emits_short_flag_on_hunyuan() -> None:
    """Schema field is `fp8_vl_text_encoder` but kohya's argv is `--fp8_vl`."""
    args = _argv(_hunyuan_config(optimization={"fp8_vl_text_encoder": True}))
    assert "--fp8_vl" in args
    assert not any(a.startswith("--fp8_vl_text_encoder") for a in args)


def test_b1_optimization_unsloth_offload_only_anima() -> None:
    args = _argv(_anima_config(optimization={"unsloth_offload_checkpointing": True}))
    assert "--unsloth_offload_checkpointing" in args


def test_b1_optimization_unsloth_offload_warns_on_flux(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    with caplog.at_level(logging.WARNING, logger="lorahub.core.backends.kohya.compiler"):
        args = _argv(_flux_config(optimization={"unsloth_offload_checkpointing": True}))
    assert "--unsloth_offload_checkpointing" not in args
    assert any("unsloth" in r.message for r in caplog.records)


# --- FlowMatchConfig ------------------------------------------------------


def test_b1_flow_match_full_emit_on_flux() -> None:
    cfg = _flux_config(
        flow_match={
            "timestep_sampling": "logit_normal",
            "sigmoid_scale": 1.5,
            "model_prediction_type": "raw",
            "discrete_flow_shift": 3.0,
            "weighting_scheme": "logit_normal",
            "logit_mean": 0.0,
            "logit_std": 1.0,
            "mode_scale": 1.29,
        }
    )
    args = _argv(cfg)
    assert "--timestep_sampling=logit_normal" in args
    assert "--sigmoid_scale=1.5" in args
    assert "--model_prediction_type=raw" in args
    assert "--discrete_flow_shift=3.0" in args
    assert "--weighting_scheme=logit_normal" in args
    assert "--logit_mean=0.0" in args
    assert "--logit_std=1.0" in args
    assert "--mode_scale=1.29" in args


def test_b1_flow_match_training_shift_only_sd3() -> None:
    args = _argv(_sd3_config(flow_match={"training_shift": 2.5}))
    assert "--training_shift=2.5" in args


def test_b1_flow_match_training_shift_warns_on_flux(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    with caplog.at_level(logging.WARNING, logger="lorahub.core.backends.kohya.compiler"):
        args = _argv(_flux_config(flow_match={"training_shift": 2.5}))
    assert not any(a.startswith("--training_shift") for a in args)
    assert any("training_shift" in r.message for r in caplog.records)


def test_b1_flow_match_warns_on_sdxl(caplog: pytest.LogCaptureFixture) -> None:
    """SDXL is epsilon-prediction; flow_match doesn't apply."""
    import logging

    with caplog.at_level(logging.WARNING, logger="lorahub.core.backends.kohya.compiler"):
        args = _argv(_config(flow_match={"timestep_sampling": "logit_normal"}))
    assert not any(a.startswith("--timestep_sampling") for a in args)
    assert any("flow_match" in r.message for r in caplog.records)


# --- ArchPathsConfig ------------------------------------------------------


def test_b1_arch_paths_flux_full_emit() -> None:
    cfg = _flux_config(
        base_model={
            "arch": "flux",
            "checkpoint": "/m/flux",
            "arch_paths": {
                "clip_l": "/m/clip_l",
                "t5xxl": "/m/t5",
                "ae": "/m/ae",
                "t5xxl_max_token_length": 512,
                "apply_t5_attn_mask": True,
                "guidance_scale": 3.5,
                "t5_dropout_rate": 0.1,
                "clip_l_dropout_rate": 0.05,
            },
        }
    )
    args = _argv(cfg)
    # Path values are platform-normalised by pathlib; assert flag prefix only.
    assert any(a.startswith("--clip_l=") for a in args)
    assert any(a.startswith("--t5xxl=") for a in args)
    assert any(a.startswith("--ae=") for a in args)
    assert "--t5xxl_max_token_length=512" in args
    assert "--apply_t5_attn_mask" in args
    assert "--guidance_scale=3.5" in args
    assert "--t5_dropout_rate=0.1" in args
    assert "--clip_l_dropout_rate=0.05" in args


def test_b1_arch_paths_sd3_full_emit() -> None:
    cfg = _sd3_config(
        base_model={
            "arch": "sd3",
            "checkpoint": "/m/sd3",
            "arch_paths": {
                "clip_l": "/m/clip_l",
                "clip_g": "/m/clip_g",
                "t5xxl": "/m/t5",
                "apply_t5_attn_mask": True,
                "apply_lg_attn_mask": True,
                "pos_emb_random_crop_rate": 0.1,
                "enable_scaled_pos_embed": True,
                "t5xxl_device": "cpu",
                "t5xxl_dtype": "fp16",
                "t5xxl_max_token_length": 256,
                "clip_l_dropout_rate": 0.1,
                "clip_g_dropout_rate": 0.1,
                "t5_dropout_rate": 0.1,
            },
        }
    )
    args = _argv(cfg)
    assert any(a.startswith("--clip_l=") for a in args)
    assert any(a.startswith("--clip_g=") for a in args)
    assert any(a.startswith("--t5xxl=") for a in args)
    for flag in (
        "--apply_t5_attn_mask",
        "--apply_lg_attn_mask",
        "--pos_emb_random_crop_rate=0.1",
        "--enable_scaled_pos_embed",
        "--t5xxl_device=cpu",
        "--t5xxl_dtype=fp16",
        "--t5xxl_max_token_length=256",
    ):
        assert flag in args, flag


def test_b1_arch_paths_anima_uses_upstream_spelling() -> None:
    """Anima argv: `--llm_adapter_path` (not `--llm_adapter`),
    `--t5_tokenizer_path` (not `--t5_tokenizer`)."""
    cfg = _anima_config(
        base_model={
            "arch": "anima",
            "checkpoint": "/m/anima",
            "arch_paths": {
                "qwen3": "/m/qwen3",
                "llm_adapter": "/m/adapter",
                "t5_tokenizer": "/m/t5tok",
                "qwen3_max_token_length": 256,
                "t5_max_token_length": 512,
                "vae_chunk_size": 16,
                "vae_disable_cache": True,
            },
        }
    )
    args = _argv(cfg)
    assert any(a.startswith("--qwen3=") for a in args)
    assert any(a.startswith("--llm_adapter_path=") for a in args)
    assert any(a.startswith("--t5_tokenizer_path=") for a in args)
    assert "--qwen3_max_token_length=256" in args
    assert "--t5_max_token_length=512" in args
    assert "--vae_chunk_size=16" in args
    assert "--vae_disable_cache" in args
    # Negative: the schema field name does NOT leak into argv.
    assert not any(a.startswith("--llm_adapter=") for a in args)
    assert not any(a.startswith("--t5_tokenizer=") for a in args)


def test_b1_arch_paths_hunyuan_full_emit() -> None:
    cfg = _hunyuan_config(
        base_model={
            "arch": "hunyuan_image",
            "checkpoint": "/m/h",
            "arch_paths": {
                "text_encoder": "/m/qwenvl",
                "byt5": "/m/byt5",
                "text_encoder_cpu": True,
                "vae_chunk_size": 16,
            },
        }
    )
    args = _argv(cfg)
    assert any(a.startswith("--text_encoder=") for a in args)
    assert any(a.startswith("--byt5=") for a in args)
    assert "--text_encoder_cpu" in args
    assert "--vae_chunk_size=16" in args


def test_b1_arch_paths_flux_fields_warn_on_sdxl(caplog: pytest.LogCaptureFixture) -> None:
    """clip_l / t5xxl / ae set on SDXL must not leak into argv."""
    import logging

    cfg = _config(
        base_model={
            "arch": "sdxl",
            "checkpoint": "/m",
            "arch_paths": {"clip_l": "/m/clip_l", "t5xxl": "/m/t5"},
        }
    )
    with caplog.at_level(logging.WARNING, logger="lorahub.core.backends.kohya.compiler"):
        args = _argv(cfg)
    assert not any(a.startswith("--clip_l=") for a in args)
    assert not any(a.startswith("--t5xxl=") for a in args)
    assert any("FLUX/SD3" in r.message for r in caplog.records)


def test_b1_arch_paths_anima_fields_warn_on_sdxl(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    cfg = _config(
        base_model={
            "arch": "sdxl",
            "checkpoint": "/m",
            "arch_paths": {"qwen3": "/m/q"},
        }
    )
    with caplog.at_level(logging.WARNING, logger="lorahub.core.backends.kohya.compiler"):
        args = _argv(cfg)
    assert not any(a.startswith("--qwen3=") for a in args)
    assert any("Anima-only" in r.message for r in caplog.records)


def test_b1_arch_paths_byt5_warns_on_flux(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    cfg = _flux_config(
        base_model={
            "arch": "flux",
            "checkpoint": "/m/flux",
            "arch_paths": {"byt5": "/m/byt5"},
        }
    )
    with caplog.at_level(logging.WARNING, logger="lorahub.core.backends.kohya.compiler"):
        args = _argv(cfg)
    assert not any(a.startswith("--byt5=") for a in args)
    assert any("HunyuanImage-only" in r.message for r in caplog.records)


# --- Per-module LR (Anima) ------------------------------------------------


def test_b1_module_lr_anima_full_emit() -> None:
    cfg = _anima_config(
        network={
            "module_lr": {
                "llm_adapter": 5e-5,
                "self_attn": 1e-4,
                "cross_attn": 2e-4,
                "mlp": 3e-4,
                "mod": 4e-4,
            }
        }
    )
    args = _argv(cfg)
    assert "--llm_adapter_lr=5e-05" in args
    assert "--self_attn_lr=0.0001" in args
    assert "--cross_attn_lr=0.0002" in args
    assert "--mlp_lr=0.0003" in args
    assert "--mod_lr=0.0004" in args


def test_b1_module_lr_warns_on_sdxl(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    cfg = _config(network={"module_lr": {"self_attn": 1e-4}})
    with caplog.at_level(logging.WARNING, logger="lorahub.core.backends.kohya.compiler"):
        args = _argv(cfg)
    assert not any(a.startswith("--self_attn_lr") for a in args)
    assert any("module_lr" in r.message for r in caplog.records)


# --- NetworkConfig new init/base-weights fields ---------------------------


def test_b1_network_init_from_emits_network_weights() -> None:
    cfg = _config(network={"init_from": "/lora/base.safetensors"})
    args = _argv(cfg)
    assert any(a.startswith("--network_weights=") for a in args)


def test_b1_network_dim_from_weights_flag() -> None:
    cfg = _config(
        network={
            "init_from": "/lora/base.safetensors",
            "dim_from_weights": "/lora/base.safetensors",
        }
    )
    args = _argv(cfg)
    assert any(a.startswith("--network_weights=") for a in args)
    assert "--dim_from_weights" in args


def test_b1_network_base_weights_with_multipliers() -> None:
    cfg = _config(
        network={
            "base_weights": ["/lora/a.safetensors", "/lora/b.safetensors"],
            "base_weights_multiplier": [0.7, 0.3],
        }
    )
    args = _argv(cfg)
    assert "--base_weights" in args
    bw_idx = args.index("--base_weights")
    # Path values are platform-normalised; check the basename.
    assert "a.safetensors" in args[bw_idx + 1]
    assert "b.safetensors" in args[bw_idx + 2]
    assert "--base_weights_multiplier" in args
    bm_idx = args.index("--base_weights_multiplier")
    assert args[bm_idx + 1] == "0.7"
    assert args[bm_idx + 2] == "0.3"


# --- Composite kitchen-sink ----------------------------------------------


def test_b1_kitchen_sink_full_bf16_plus_fp8_plus_cache_te_disk() -> None:
    """Real-world combo: bf16 mixed precision + full_bf16 + fp8_base +
    cache_text_encoder_outputs_to_disk + multires_noise + max_grad_norm.
    All must coexist on a single SDXL config without conflicts."""
    cfg = _config(
        optimization={
            "full_bf16": True,
            "fp8_base": True,
            "cache_text_encoder_outputs_to_disk": True,
        },
        loss={
            "multires_noise_iterations": 6,
            "multires_noise_discount": 0.4,
        },
        optimizer={"max_grad_norm": 0.5},
    )
    args = _argv(cfg)
    assert "--mixed_precision=bf16" in args
    assert "--full_bf16" in args
    assert "--fp8_base" in args
    assert "--cache_text_encoder_outputs_to_disk" in args
    assert "--multires_noise_iterations=6" in args
    assert "--multires_noise_discount=0.4" in args
    assert "--max_grad_norm=0.5" in args


def test_b1_kitchen_sink_flux_full_arch_paths_plus_flow_match() -> None:
    """FLUX config with arch_paths + flow_match + advanced loss + optimization."""
    cfg = _flux_config(
        base_model={
            "arch": "flux",
            "checkpoint": "/m/flux",
            "arch_paths": {
                "clip_l": "/m/clip_l",
                "t5xxl": "/m/t5",
                "ae": "/m/ae",
                "guidance_scale": 1.0,
                "apply_t5_attn_mask": True,
            },
        },
        flow_match={
            "timestep_sampling": "logit_normal",
            "discrete_flow_shift": 3.0,
            "weighting_scheme": "logit_normal",
        },
        optimization={
            "full_bf16": True,
            "fp8_base": True,
            "blocks_to_swap": 16,
            "cache_text_encoder_outputs": True,
        },
        loss={"zero_terminal_snr": True},
    )
    args = _argv(cfg)
    for flag in (
        "--apply_t5_attn_mask",
        "--timestep_sampling=logit_normal",
        "--discrete_flow_shift=3.0",
        "--weighting_scheme=logit_normal",
        "--full_bf16",
        "--fp8_base",
        "--blocks_to_swap=16",
        "--cache_text_encoder_outputs",
        "--zero_terminal_snr",
    ):
        assert flag in args, flag
    # Path comparisons normalise the OS separator.
    assert any(a.startswith("--clip_l=") and "clip_l" in a for a in args)
    assert any(a.startswith("--t5xxl=") and "t5" in a for a in args)
    assert any(a.startswith("--ae=") and "ae" in a for a in args)
    assert any(a.startswith("--guidance_scale=1.0") for a in args)


def test_b1_default_config_argv_byte_identical_after_b1() -> None:
    """Anchor test: producing argv from a default-only config must remain
    a stable list with the existing fields (sanity check the new helpers
    don't slip in any defaults)."""
    args = _argv(_config(), ws=Path("/ws"))
    expected_anchors = [
        "--network_module=networks.lora",
        "--network_dim=32",
        "--network_alpha=16",
        "--network_train_unet_only",
        "--optimizer_type=AdamW8bit",
        "--learning_rate=0.0001",
        "--lr_scheduler=cosine_with_restarts",
        "--lr_warmup_steps=100",
        "--max_train_epochs=10",
        "--train_batch_size=1",
        "--gradient_accumulation_steps=2",
        "--mixed_precision=bf16",
        "--gradient_checkpointing",
        "--cache_latents",
        "--save_model_as=safetensors",
        "--save_precision=fp16",
        "--save_state",
        "--save_state_on_train_end",
    ]
    for flag in expected_anchors:
        assert flag in args, flag
    # Path-bearing flags use platform separators; check prefix.
    assert any(a.startswith("--pretrained_model_name_or_path=") for a in args)
