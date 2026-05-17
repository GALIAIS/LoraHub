"""Tests for the kohya compiler."""

from __future__ import annotations

from pathlib import Path

import pytest

from lorahub.core.backends.kohya.compiler import CompilationError, compile_config
from lorahub.core.config.schema import TrainingConfig


def _recipe(**overrides: object) -> TrainingConfig:
    base = {
        "base_model": {"checkpoint": "/m/sdxl.safetensors"},
        "dataset": {"source": "/d/imgs"},
    }
    base.update(overrides)  # type: ignore[arg-type]
    return TrainingConfig.model_validate(base)


def _argv(cfg: TrainingConfig, ws: Path = Path("/ws")) -> list[str]:
    _, args, _files = compile_config(cfg, ws)
    return args


def _files(cfg: TrainingConfig, ws: Path = Path("/ws")) -> dict[Path, str]:
    _, _args, files = compile_config(cfg, ws)
    return files


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
        s, _, _ = compile_config(cfg, tmp_path)
        assert s == script


def _arch_recipe(arch: str) -> TrainingConfig:
    return TrainingConfig.model_validate(
        {
            "base_model": {"arch": arch, "checkpoint": "/m.safetensors"},
            "dataset": {"source": "/d"},
        }
    )


def test_pick_script_anima(tmp_path: Path) -> None:
    """Anima uses its own entry script per kohya's README."""
    s, _, _ = compile_config(_arch_recipe("anima"), tmp_path)
    assert s == "anima_train_network.py"


def test_pick_script_lumina(tmp_path: Path) -> None:
    s, _, _ = compile_config(_arch_recipe("lumina"), tmp_path)
    assert s == "lumina_train_network.py"


def test_pick_script_hunyuan_image(tmp_path: Path) -> None:
    s, _, _ = compile_config(_arch_recipe("hunyuan_image"), tmp_path)
    assert s == "hunyuan_image_train_network.py"


def test_pick_script_sd2_reuses_sd15_entry(tmp_path: Path) -> None:
    """sd-scripts ships sd1.x/2.x in the same train_network.py entry script."""
    s, _, _ = compile_config(_arch_recipe("sd2"), tmp_path)
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
        compile_config(_arch_recipe(arch), tmp_path)


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
        compile_config(cfg, Path("/ws"))


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


def test_sampling_attention_default_emits_no_extra_argv() -> None:
    """The schema default (`"default"`) must keep argv byte-identical to recipes
    that omit the field entirely. Otherwise existing recipes would drift."""
    baseline = _argv(_recipe(sampling={"prompts_file": "/p/eval.txt"}))
    explicit = _argv(
        _recipe(
            sampling={"prompts_file": "/p/eval.txt", "attention": "default"}
        )
    )
    assert baseline == explicit


def test_sampling_attention_non_default_does_not_emit_attn_mode(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """SageAttention has no working backward; the compiler must NOT inject
    ``--attn_mode`` into argv (which kohya wires into the *training* forward
    on anima/hunyuan_image), and must surface a warning so the user knows the
    field is parked until the wrapper lands."""
    cfg = _recipe(
        sampling={"prompts_file": "/p/eval.txt", "attention": "sageattn"}
    )
    with caplog.at_level("WARNING", logger="lorahub.core.backends.kohya.compiler"):
        args = _argv(cfg)

    assert not any(a.startswith("--attn_mode") for a in args)
    assert any(
        "sampling.attention" in rec.message and "sageattn" in rec.message
        for rec in caplog.records
    ), [rec.message for rec in caplog.records]


def test_sampling_attention_non_default_compiles_for_every_kohya_arch(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Picking a non-default attention must never raise CompilationError --
    the field is purely additive. Cover every kohya entry script so we catch
    accidental coupling later."""
    archs = [
        "sd15", "sd2", "sdxl", "sd3", "flux", "lumina",
        "hunyuan_image", "anima",
    ]
    for arch in archs:
        cfg = TrainingConfig.model_validate(
            {
                "base_model": {"arch": arch, "checkpoint": "/m.safetensors"},
                "dataset": {"source": "/d"},
                "sampling": {"prompts_file": "/p/eval.txt", "attention": "xformers"},
            }
        )
        # Should not raise; argv stays free of `--attn_mode` regardless of arch.
        _, args, _ = compile_config(cfg, Path("/ws"))
        assert not any(a.startswith("--attn_mode") for a in args), arch


def test_sampling_attention_silent_when_sampling_disabled(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """If sampling is disabled the attention preference is moot, no warning."""
    cfg = _recipe(sampling={"enabled": False, "attention": "sageattn"})
    with caplog.at_level("WARNING", logger="lorahub.core.backends.kohya.compiler"):
        _argv(cfg)
    assert not any(
        "sampling.attention" in rec.message for rec in caplog.records
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
    args = _argv(_recipe())
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
    cfg = _recipe(loss={"min_snr_gamma": 5})
    args = _argv(cfg)
    assert "--min_snr_gamma=5.0" in args
    # noise_offset stayed default 鈫?still absent
    assert not any(a.startswith("--noise_offset") for a in args)


def test_loss_full_kitchen_sink() -> None:
    cfg = _recipe(
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
    args = _argv(_recipe(loss={"prior_loss_weight": 1.0}))
    assert not any(a.startswith("--prior_loss_weight") for a in args)


def test_optimizer_args_emit_betas_weight_decay_eps() -> None:
    args = _argv(_recipe(optimizer={"betas": [0.95, 0.999], "weight_decay": 0.1, "eps": 1e-7}))
    idx = args.index("--optimizer_args")
    tail = args[idx + 1 :]
    assert "betas=0.95,0.999" in tail
    assert "weight_decay=0.1" in tail
    assert "eps=1e-07" in tail


def test_optimizer_args_user_overrides_dedicated_fields() -> None:
    """Free-form `optimizer_args` keys win over the dedicated betas/eps."""
    args = _argv(
        _recipe(optimizer={"optimizer_args": {"betas": "0.5,0.5", "use_bias_correction": "True"}})
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
    args = _argv(_recipe())
    for flag in (
        "--torch_compile",
        "--fused_backward_pass",
        "--full_bf16",
        "--blocks_to_swap",
    ):
        assert not any(a.startswith(flag) for a in args), flag


def test_optimization_torch_compile_emits_flag() -> None:
    cfg = _recipe(optimization={"torch_compile": True})
    assert "--torch_compile" in _argv(cfg)


def test_optimization_fused_backward_pass_emits_flag() -> None:
    cfg = _recipe(optimization={"fused_backward_pass": True})
    assert "--fused_backward_pass" in _argv(cfg)


def test_optimization_full_bf16_coexists_with_mixed_precision() -> None:
    """`--full_bf16` is additive: it lands alongside `--mixed_precision=bf16`."""
    cfg = _recipe(optimization={"full_bf16": True})
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

    cfg = _recipe(optimization={"blocks_to_swap": 8})
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
    cfg = _recipe(optimization={"blocks_to_swap": 0})
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

