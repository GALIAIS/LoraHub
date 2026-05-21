"""anima_lora compiler tests — argv translation + constraint enforcement.

Each method gets a snapshot test that verifies the emitted CLI flags
include the right ``--method`` / ``--preset`` and the method-specific
overrides land where upstream's argparse expects them.

We don't snapshot the full argv list (it'd be brittle against schema
churn) — instead we assert on key/value membership which catches the
"this flag stopped being emitted" regressions without trapping
ourselves into rewriting the snapshot every time we add a knob.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lorahub.core.backends.anima_lora import CompilationError, compile_config
from lorahub.core.config.schema import (
    AnimaLoraMethodChimeraConfig,
    AnimaLoraMethodEasyControlConfig,
    AnimaLoraMethodIPAdapterConfig,
    AnimaLoraMethodPostfixConfig,
    AnimaLoraOptions,
    TrainingConfig,
)


def _recipe(tmp_path: Path, opts: AnimaLoraOptions) -> TrainingConfig:
    ckpt = tmp_path / "m.safetensors"
    ckpt.write_bytes(b"")
    data = tmp_path / "data"
    data.mkdir()
    return TrainingConfig.model_validate(
        {
            "base_model": {"checkpoint": str(ckpt), "arch": "anima"},
            "dataset": {"source": str(data)},
            "schedule": {"epochs": 1, "batch_size": 1},
            "sampling": {"enabled": False},
            "optimizer": {"lr": {"unet": 1e-4, "text_encoder": 5e-5}},
            "network": {"rank": 16, "alpha": 8},
            "output": {"name": "x"},
            "backend": {"type": "anima_lora", "animaLora": opts.model_dump(by_alias=True)},
        }
    )


def _argv_pairs(argv: list[str]) -> dict[str, list[str]]:
    """Group argv into a flag → values map.

    ``["--a", "1", "--b", "--c", "2"]`` →
    ``{"--a": ["1"], "--b": [""], "--c": ["2"]}``. Repeated flags
    accumulate (matters for ``--network_args``).
    """
    out: dict[str, list[str]] = {}
    i = 0
    while i < len(argv):
        flag = argv[i]
        assert flag.startswith("--"), f"expected flag at {i}, got {flag!r}"
        # Look ahead — store-true flags have no value.
        if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            out.setdefault(flag, []).append(argv[i + 1])
            i += 2
        else:
            out.setdefault(flag, []).append("")
            i += 1
    return out


# --------------------------------------------------------------------------- #
# Method routing — every method must select itself + emit core knobs
# --------------------------------------------------------------------------- #


def test_max_steps_emits_zero_max_train_epochs(tmp_path: Path) -> None:
    """schedule.max_steps must produce an explicit ``--max_train_epochs 0``.

    Without the zero, configs/methods/lora.toml's ``max_train_epochs = 8``
    sneaks into args via the TOML merge chain and train.py:1622 silently
    rewrites max_train_steps to ``epochs × steps_per_epoch``. The zero
    is consumed by our local train.py patch as the "ignore epochs"
    sentinel; if either side drifts, runs revert to the 1984-step bug.
    """
    opts = AnimaLoraOptions()
    cfg = _recipe(tmp_path, opts)
    cfg.schedule.max_steps = 4000
    argv, _ = compile_config(cfg, tmp_path / "ws")
    pairs = _argv_pairs(argv)

    assert pairs["--max_train_steps"] == ["4000"]
    assert pairs["--max_train_epochs"] == ["0"]


def test_max_steps_unset_emits_method_max_train_epochs(tmp_path: Path) -> None:
    """When the user leaves max_steps unset, send the method's epoch budget."""
    opts = AnimaLoraOptions()
    cfg = _recipe(tmp_path, opts)
    cfg.schedule.max_steps = None
    argv, _ = compile_config(cfg, tmp_path / "ws")
    pairs = _argv_pairs(argv)

    assert "--max_train_steps" not in pairs
    assert pairs["--max_train_epochs"] == [str(opts.max_train_epochs)]


def test_lora_method_emits_default_stack(tmp_path: Path) -> None:
    """method='lora' default stacks OrthoLoRA + T-LoRA per upstream lora.toml.

    All four LoRA toggles flow through ``--network_args`` (kwargs read by
    ``networks/lora_anima/config.py``), not as discrete argparse flags —
    upstream's train.py never declared them.
    """
    opts = AnimaLoraOptions()
    cfg = _recipe(tmp_path, opts)
    argv, files = compile_config(cfg, tmp_path / "ws")
    pairs = _argv_pairs(argv)

    assert pairs["--method"] == ["lora"]
    assert pairs["--preset"] == ["default"]
    network_args = pairs["--network_args"]
    assert "use_ortho=true" in network_args
    assert "use_timestep_mask=true" in network_args
    assert "min_rank=8" in network_args
    assert any(p.startswith("alpha_rank_scale=") for p in network_args)
    # files-to-write contains exactly the generated dataset_config TOML
    # — upstream's argparse has no flag for the three data path keys
    # (source / resized / cache), so we deliver them via
    # --dataset_config <path>. Method / preset merge stays
    # upstream-owned through configs/base.toml + configs/methods/<x>.toml.
    assert len(files) == 1
    [only_path] = list(files.keys())
    assert only_path.name == "_lorahub_anima_dataset.toml"


def test_postfix_method_emits_network_args(tmp_path: Path) -> None:
    opts = AnimaLoraOptions(
        method="postfix",
        postfix=AnimaLoraMethodPostfixConfig(),
    )
    cfg = _recipe(tmp_path, opts)
    argv, _ = compile_config(cfg, tmp_path / "ws")
    pairs = _argv_pairs(argv)

    assert pairs["--method"] == ["postfix"]
    # Each network_args k=v lands as a repeated --network_args flag.
    network_args = pairs["--network_args"]
    assert any(p.startswith("mode=") for p in network_args)
    assert any(p.startswith("ortho_basis=svd_te") for p in network_args)
    assert any(p.startswith("lambda_init=") for p in network_args)


def test_chimera_method_emits_balance_weights(tmp_path: Path) -> None:
    """ChimeraHydra knobs flow through --network_args, not as argparse flags.

    ``use_chimera_hydra`` / ``balance_w_*`` / ``fei_feature_dim`` are all
    read off ``kwargs`` in ``networks/lora_anima/config.py:LoRAConfig.from_kwargs``.
    Emitting them as discrete CLI flags trips train.py's argparse.
    """
    opts = AnimaLoraOptions(
        method="chimera",
        chimera=AnimaLoraMethodChimeraConfig(),
    )
    cfg = _recipe(tmp_path, opts)
    argv, _ = compile_config(cfg, tmp_path / "ws")
    pairs = _argv_pairs(argv)

    assert pairs["--method"] == ["chimera"]
    network_args = pairs["--network_args"]
    assert "use_chimera_hydra=true" in network_args
    assert any(p.startswith("balance_w_content=") for p in network_args)
    assert any(p.startswith("balance_w_freq=") for p in network_args)
    assert any(p.startswith("fei_feature_dim=") for p in network_args)


def test_easycontrol_method_emits_b_cond_init(tmp_path: Path) -> None:
    """EasyControl mixes argparse flags + network_args.

    ``--use_easycontrol`` / ``--easycontrol_drop_p`` / ``--easycontrol_cond_noise_max``
    are real argparse flags. ``b_cond_init`` / ``cond_scale`` / ``apply_ffn_lora``
    / ``cond_token_count`` are read from kwargs in
    ``networks/methods/easycontrol.py``.
    """
    opts = AnimaLoraOptions(
        method="easycontrol",
        easycontrol=AnimaLoraMethodEasyControlConfig(),
    )
    cfg = _recipe(tmp_path, opts)
    argv, _ = compile_config(cfg, tmp_path / "ws")
    pairs = _argv_pairs(argv)

    assert pairs["--method"] == ["easycontrol"]
    assert "--use_easycontrol" in pairs
    assert "--easycontrol_drop_p" in pairs
    network_args = pairs["--network_args"]
    # b_cond_init = -10 zeros the gate at step 0.
    assert any(p.startswith("b_cond_init=-10") for p in network_args)
    assert "cond_token_count=4096" in network_args


def test_ip_adapter_method_emits_pe_encoder(tmp_path: Path) -> None:
    """IP-Adapter mixes argparse flags + network_args (resampler dims, ip_scale, gate_lr)."""
    opts = AnimaLoraOptions(
        method="ip_adapter",
        ip_adapter=AnimaLoraMethodIPAdapterConfig(),
    )
    cfg = _recipe(tmp_path, opts)
    argv, _ = compile_config(cfg, tmp_path / "ws")
    pairs = _argv_pairs(argv)

    assert pairs["--method"] == ["ip_adapter"]
    assert "--use_ip_adapter" in pairs
    assert pairs["--ip_encoder"] == ["PE-Core-L14-336"]
    # gate_lr is 10x global LR per upstream rationale; rides --network_args.
    network_args = pairs["--network_args"]
    assert any(p.startswith("gate_lr=") for p in network_args)
    assert any(p.startswith("ip_scale=") for p in network_args)


# --------------------------------------------------------------------------- #
# Preset routing
# --------------------------------------------------------------------------- #


def test_preset_low_vram_passed_through(tmp_path: Path) -> None:
    """`preset=low_vram` selects upstream's [low_vram] section verbatim."""
    opts = AnimaLoraOptions(preset="low_vram")
    cfg = _recipe(tmp_path, opts)
    argv, _ = compile_config(cfg, tmp_path / "ws")
    pairs = _argv_pairs(argv)
    assert pairs["--preset"] == ["low_vram"]


def test_preset_debug_passed_through(tmp_path: Path) -> None:
    opts = AnimaLoraOptions(preset="debug")
    cfg = _recipe(tmp_path, opts)
    argv, _ = compile_config(cfg, tmp_path / "ws")
    pairs = _argv_pairs(argv)
    assert pairs["--preset"] == ["debug"]


# --------------------------------------------------------------------------- #
# Compile-mode constraint
# --------------------------------------------------------------------------- #


def test_compile_full_with_gradient_checkpointing_rejected(tmp_path: Path) -> None:
    """Per upstream CLAUDE.md, compile_mode='full' ⊥ gradient_checkpointing."""
    opts = AnimaLoraOptions(
        compile_mode="full",
        compile_inductor_mode="reduce-overhead",
        gradient_checkpointing=True,
    )
    cfg = _recipe(tmp_path, opts)
    with pytest.raises(CompilationError, match="compile_mode"):
        compile_config(cfg, tmp_path / "ws")


def test_compile_full_with_blocks_to_swap_rejected(tmp_path: Path) -> None:
    opts = AnimaLoraOptions(
        compile_mode="full",
        blocks_to_swap=4,
    )
    cfg = _recipe(tmp_path, opts)
    with pytest.raises(CompilationError, match="blocks_to_swap"):
        compile_config(cfg, tmp_path / "ws")


def test_compile_blocks_with_gradient_checkpointing_allowed(tmp_path: Path) -> None:
    """compile_mode='blocks' is the per-block compile path — allowed alongside grad ckpt."""
    opts = AnimaLoraOptions(
        compile_mode="blocks",
        gradient_checkpointing=True,
    )
    cfg = _recipe(tmp_path, opts)
    # No raise.
    argv, _ = compile_config(cfg, tmp_path / "ws")
    pairs = _argv_pairs(argv)
    assert pairs["--compile_mode"] == ["blocks"]
    assert "--gradient_checkpointing" in pairs


# --------------------------------------------------------------------------- #
# Output dir + output name
# --------------------------------------------------------------------------- #


def test_workspace_drives_output_dir(tmp_path: Path) -> None:
    """`workspace/ckpt` is what anima_lora writes safetensors into."""
    opts = AnimaLoraOptions(output_name="my_run")
    cfg = _recipe(tmp_path, opts)
    ws = tmp_path / "ws"
    argv, _ = compile_config(cfg, ws)
    pairs = _argv_pairs(argv)
    expected = (ws / "ckpt").resolve()
    assert Path(pairs["--output_dir"][0]) == expected
    assert pairs["--output_name"] == ["my_run"]


# --------------------------------------------------------------------------- #
# Wrong backend type guard
# --------------------------------------------------------------------------- #


def test_compile_rejects_non_anima_lora_recipe(tmp_path: Path) -> None:
    """Defence-in-depth: dispatch should never give us a non-anima_lora recipe."""
    ckpt = tmp_path / "m.safetensors"
    ckpt.write_bytes(b"")
    data = tmp_path / "data"
    data.mkdir()
    cfg = TrainingConfig.model_validate(
        {
            "base_model": {"checkpoint": str(ckpt)},
            "dataset": {"source": str(data)},
            "schedule": {"epochs": 1, "batch_size": 1},
            "sampling": {"enabled": False},
            "optimizer": {"lr": {"unet": 1e-4, "text_encoder": 5e-5}},
            "network": {"rank": 16, "alpha": 8},
            "output": {"name": "x"},
            "backend": {"type": "kohya"},  # wrong on purpose
        }
    )
    with pytest.raises(CompilationError, match="anima_lora"):
        compile_config(cfg, tmp_path / "ws")


def test_compile_rejects_anima_lora_with_no_options(tmp_path: Path) -> None:
    """type='anima_lora' but animaLora field absent — clear error, not crash."""
    ckpt = tmp_path / "m.safetensors"
    ckpt.write_bytes(b"")
    data = tmp_path / "data"
    data.mkdir()
    cfg = TrainingConfig.model_validate(
        {
            "base_model": {"checkpoint": str(ckpt)},
            "dataset": {"source": str(data)},
            "schedule": {"epochs": 1, "batch_size": 1},
            "sampling": {"enabled": False},
            "optimizer": {"lr": {"unet": 1e-4, "text_encoder": 5e-5}},
            "network": {"rank": 16, "alpha": 8},
            "output": {"name": "x"},
            "backend": {"type": "anima_lora"},  # animaLora omitted
        }
    )
    with pytest.raises(CompilationError, match="anima_lora"):
        compile_config(cfg, tmp_path / "ws")


# --------------------------------------------------------------------------- #
# Shared overrides — sanity sweeps
# --------------------------------------------------------------------------- #


def test_default_options_emit_seeded_argv(tmp_path: Path) -> None:
    """Even without explicit cfg.seed, the compiler emits a deterministic seed."""
    opts = AnimaLoraOptions()
    cfg = _recipe(tmp_path, opts)
    argv, _ = compile_config(cfg, tmp_path / "ws")
    pairs = _argv_pairs(argv)
    assert "--seed" in pairs
    # Default fallback is 42 — let upstream CLAUDE.md drift catch us if
    # they ever change the convention.
    assert pairs["--seed"] == ["42"]


def test_caching_flags_emit_when_enabled(tmp_path: Path) -> None:
    """The four cache_* booleans default True and must all surface as flags."""
    opts = AnimaLoraOptions()
    cfg = _recipe(tmp_path, opts)
    argv, _ = compile_config(cfg, tmp_path / "ws")
    pairs = _argv_pairs(argv)
    for flag in (
        "--cache_latents",
        "--cache_latents_to_disk",
        "--cache_text_encoder_outputs",
        "--cache_text_encoder_outputs_to_disk",
        "--cache_llm_adapter_outputs",
    ):
        assert flag in pairs, f"missing default cache flag {flag}"


def test_attn_mode_default_is_torch(tmp_path: Path) -> None:
    """Default attn_mode is ``torch`` (PyTorch SDPA) for portability.

    Upstream's base.toml default is ``flash``, but flash-attn is an
    optional, compute-capability-sensitive build many environments
    don't have (and trips a RuntimeError at DiT load time when
    missing). LoRaHub overrides the default to ``torch`` so a fresh
    install runs out of the box; users with flash-attn flip it back.
    """
    opts = AnimaLoraOptions()
    cfg = _recipe(tmp_path, opts)
    argv, _ = compile_config(cfg, tmp_path / "ws")
    pairs = _argv_pairs(argv)
    assert pairs["--attn_mode"] == ["torch"]


# --------------------------------------------------------------------------- #
# Dataset path overrides — kohya / dp parity (cfg.dataset.source = raw images)
# --------------------------------------------------------------------------- #


def test_compile_emits_source_resized_lora_cache_paths(tmp_path: Path) -> None:
    """The compiler pins source / resized / cache to absolute LoraHub paths.

    cfg.dataset.source is the user-facing raw image directory (kohya /
    dp parity); the resized + cache dirs are LoRaHub-managed under
    ``<workspace>/post_image_dataset/{resized,lora}``. The three are
    surfaced via a generated ``--dataset_config <path>`` TOML
    (train.py has no CLI flag for the three keys — they live as
    ``configs/base.toml`` top-level scalars). The TOML must contain
    absolute paths for the resized + cache dirs so anima_lora's own
    ``configs/base.toml`` defaults (relative to the vendored repo
    root) are bypassed.
    """
    opts = AnimaLoraOptions()
    cfg = _recipe(tmp_path, opts)
    ws = tmp_path / "ws"
    argv, files = compile_config(cfg, ws)
    pairs = _argv_pairs(argv)

    # Dataset config is delivered as a single --dataset_config flag
    # whose value is a path to a TOML written under the workspace.
    cfg_path = Path(pairs["--dataset_config"][0])
    assert cfg_path.is_absolute()
    assert cfg_path.parent == ws.resolve()
    assert cfg_path in files, "compile_config must hand the TOML body back via the files dict"

    body = files[cfg_path]
    resized_expected = str((ws / "post_image_dataset" / "resized").resolve())
    cache_expected = str((ws / "post_image_dataset" / "lora").resolve())
    # TOML strings are double-quoted with backslashes escaped on
    # Windows; do a substring check that survives both forms.
    assert resized_expected.replace("\\", "\\\\") in body or resized_expected in body
    assert cache_expected.replace("\\", "\\\\") in body or cache_expected in body


def test_extra_args_pass_through_verbatim(tmp_path: Path) -> None:
    """``backend.extra_args`` must reach argv as ``--flag``/``--flag=val``.

    This is the escape hatch new train.py flags ride before they're
    promoted into AnimaLoraOptions: EMA, nan_guard, min_snr_gamma,
    sample_grid all live here in the recipe.
    """
    opts = AnimaLoraOptions()
    cfg = _recipe(tmp_path, opts)
    cfg.backend.extra_args = {
        "ema": True,
        "ema_decay": 0.9999,
        "ema_use_num_updates": True,
        "nan_guard": True,
        "nan_guard_recover": True,
        "min_snr_gamma": 5,
        "sample_grid": True,
        "weighting_scheme": "min_snr_rf",
        "should_be_dropped": False,
        "also_dropped": None,
    }
    argv, _ = compile_config(cfg, tmp_path / "ws")

    assert "--ema" in argv
    assert "--ema_decay=0.9999" in argv
    assert "--ema_use_num_updates" in argv
    assert "--nan_guard" in argv
    assert "--nan_guard_recover" in argv
    assert "--min_snr_gamma=5" in argv
    assert "--sample_grid" in argv
    # extra_args last-write-wins for any flag also emitted by the
    # shared layer — weighting_scheme defaults to None in opts so this
    # one only gets emitted from extra_args, but the relative order
    # still has the extra_args copy last.
    assert "--weighting_scheme=min_snr_rf" in argv
    assert "--should_be_dropped" not in argv
    assert "--also_dropped" not in argv


def test_ema_forces_inductor_mode_default_when_unset(tmp_path: Path) -> None:
    """``ema=true`` + no compile_inductor_mode → forced to ``default``.

    anima base.toml ships ``compile_inductor_mode = "reduce-overhead"``,
    which enables cudagraph_trees. EMA's per-step ``shadow.copy_(live...)``
    breaks the input-tensor liveness invariant cudagraph_trees checks,
    so step 2 crashes with ``RuntimeError: graph recording observed an
    input tensor deallocate during graph recording that did not occur
    during replay``. The compiler force-overrides to silence this trap.
    """
    opts = AnimaLoraOptions()  # compile_inductor_mode left None
    cfg = _recipe(tmp_path, opts)
    cfg.backend.extra_args = {"ema": True, "ema_decay": 0.9999}
    argv, _ = compile_config(cfg, tmp_path / "ws")

    assert "--compile_inductor_mode=default" in argv
    # The shared-layer emit only fires when opts has it set, so with
    # opts=None we should not see the long-form ``--compile_inductor_mode default``.
    assert "--compile_inductor_mode" not in argv  # the space-separated form


def test_ema_overrides_reduce_overhead_from_opts(tmp_path: Path) -> None:
    """``ema=true`` overrides an explicit reduce-overhead from opts."""
    opts = AnimaLoraOptions(compile_inductor_mode="reduce-overhead")
    cfg = _recipe(tmp_path, opts)
    cfg.backend.extra_args = {"ema": True}
    argv, _ = compile_config(cfg, tmp_path / "ws")

    # Shared layer still emits the reduce-overhead pair, but the
    # extra_args-driven default lands later and last-write-wins for
    # the trainer's argparse.
    assert argv.count("--compile_inductor_mode") == 1  # space form
    assert "reduce-overhead" in argv
    assert "--compile_inductor_mode=default" in argv
    # Order matters: the override must appear AFTER the reduce-overhead pair.
    idx_ro = argv.index("reduce-overhead")
    idx_def = argv.index("--compile_inductor_mode=default")
    assert idx_def > idx_ro


def test_ema_overrides_reduce_overhead_from_extra_args(tmp_path: Path) -> None:
    """``ema=true`` overrides reduce-overhead even when set via extra_args.

    extra_args is the legacy path; the cross-check still recognises ``ema``
    set there. The override is appended after extra_args so argparse
    last-write-wins picks ``default``.
    """
    opts = AnimaLoraOptions()
    cfg = _recipe(tmp_path, opts)
    cfg.backend.extra_args = {
        "ema": True,
        "compile_inductor_mode": "reduce-overhead",
    }
    argv, _ = compile_config(cfg, tmp_path / "ws")

    assert "--compile_inductor_mode=default" in argv
    # Both values are present (extra_args verbatim + cross-check append),
    # but the default override appears strictly after the reduce-overhead
    # value — argparse takes the last one.
    idx_ro = argv.index("--compile_inductor_mode=reduce-overhead")
    idx_def = argv.index("--compile_inductor_mode=default")
    assert idx_def > idx_ro


def test_ema_leaves_default_mode_alone(tmp_path: Path) -> None:
    """``ema=true`` + already ``default`` → no surprise mutation."""
    opts = AnimaLoraOptions(compile_inductor_mode="default")
    cfg = _recipe(tmp_path, opts)
    cfg.backend.extra_args = {"ema": True}
    argv, _ = compile_config(cfg, tmp_path / "ws")

    assert "--compile_inductor_mode" in argv  # from shared layer
    assert "default" in argv  # value emitted by shared layer
    # No second emit of ``=default`` from extra_args injection.
    assert "--compile_inductor_mode=default" not in argv


def test_no_ema_leaves_reduce_overhead_alone(tmp_path: Path) -> None:
    """Without EMA, reduce-overhead is a legitimate fast path — don't touch it."""
    opts = AnimaLoraOptions(compile_inductor_mode="reduce-overhead")
    cfg = _recipe(tmp_path, opts)
    cfg.backend.extra_args = {}
    argv, _ = compile_config(cfg, tmp_path / "ws")

    assert "reduce-overhead" in argv
    assert "--compile_inductor_mode=default" not in argv


def test_ema_schema_field_emits_full_flag_set(tmp_path: Path) -> None:
    """Schema-level ema=true emits --ema --ema_decay --ema_use_num_updates.

    Mirrors the train.py CLI surface: the trainer reads each flag
    independently, so missing any one leaves a knob at its argparse
    default rather than the user's intent.
    """
    opts = AnimaLoraOptions(
        compile_inductor_mode="default",  # avoid the cross-check append
        ema=True,
        ema_decay=0.999,
        ema_use_num_updates=False,
    )
    cfg = _recipe(tmp_path, opts)
    argv, _ = compile_config(cfg, tmp_path / "ws")

    assert "--ema" in argv
    # Decay is emitted as space-separated value pair so it matches the
    # shared-layer style (the rest of the file uses _argv_pairs which
    # expects this shape).
    pairs = _argv_pairs(argv)
    assert pairs["--ema_decay"] == ["0.999"]
    assert "--ema_use_num_updates" not in argv  # toggled off


def test_nan_guard_schema_field_emits_recovery_flags(tmp_path: Path) -> None:
    """Schema-level nan_guard=true emits the full recovery surface."""
    opts = AnimaLoraOptions(
        compile_inductor_mode="default",
        nan_guard=True,
        nan_guard_recover=True,
        nan_guard_max_consecutive=10,
    )
    cfg = _recipe(tmp_path, opts)
    argv, _ = compile_config(cfg, tmp_path / "ws")
    pairs = _argv_pairs(argv)

    assert "--nan_guard" in argv
    assert pairs["--nan_guard_max_consecutive"] == ["10"]
    assert "--nan_guard_recover" in argv


def test_min_snr_rf_emits_gamma_when_provided(tmp_path: Path) -> None:
    """min_snr_rf is the rectified-flow Min-SNR-γ weighting; emit γ pair."""
    opts = AnimaLoraOptions(
        weighting_scheme="min_snr_rf",
        min_snr_gamma=5.0,
    )
    cfg = _recipe(tmp_path, opts)
    argv, _ = compile_config(cfg, tmp_path / "ws")
    pairs = _argv_pairs(argv)

    assert pairs["--weighting_scheme"] == ["min_snr_rf"]
    assert pairs["--min_snr_gamma"] == ["5.0"]


def test_min_snr_rf_without_gamma_falls_back_uniform(tmp_path: Path, caplog) -> None:
    """min_snr_rf without γ is a no-op in the trainer; warn at compile time."""
    import logging

    opts = AnimaLoraOptions(weighting_scheme="min_snr_rf", min_snr_gamma=None)
    cfg = _recipe(tmp_path, opts)
    with caplog.at_level(logging.WARNING):
        argv, _ = compile_config(cfg, tmp_path / "ws")

    assert "--weighting_scheme" in argv
    assert "--min_snr_gamma" not in argv
    assert any("min_snr_gamma" in r.message for r in caplog.records)


def test_sample_grid_schema_field_emits_flag(tmp_path: Path) -> None:
    """sample_grid=true emits the bare ``--sample_grid`` flag."""
    opts = AnimaLoraOptions(
        compile_inductor_mode="default",
        sample_grid=True,
    )
    cfg = _recipe(tmp_path, opts)
    argv, _ = compile_config(cfg, tmp_path / "ws")
    assert "--sample_grid" in argv


def test_dora_emits_use_dora_network_arg(tmp_path: Path) -> None:
    """``backend.animaLora.lora.useDora=True`` reaches train.py via network_args.

    DoRA is selected at the network factory by ``use_dora=true`` — the
    flag is read off ``--network_args`` (not as a top-level CLI arg)
    because anima_lora's argparse doesn't define one. The compiler must
    emit it alongside ``use_ortho`` so the variant resolver picks it up.
    """
    from lorahub.core.config.schema import AnimaLoraMethodLoraConfig

    opts = AnimaLoraOptions(
        lora=AnimaLoraMethodLoraConfig(use_ortho=False, use_dora=True),
    )
    cfg = _recipe(tmp_path, opts)
    argv, _ = compile_config(cfg, tmp_path / "ws")
    assert "use_dora=true" in argv
    assert "use_ortho=false" in argv


def test_dora_and_ortho_mutex_rejected_at_validation(tmp_path: Path) -> None:
    """``use_dora`` with ``use_ortho`` must fail pydantic validation.

    The two are incompatible: DoRA stores per-Linear ``.magnitude``,
    OrthoLoRA writes Cayley distill keys, and the standard save layout
    can't carry both. Catching this at schema-validation time gives a
    clean error before the trainer ever spawns.
    """
    from lorahub.core.config.schema import AnimaLoraMethodLoraConfig

    with pytest.raises(ValueError, match="use_ortho=True"):
        AnimaLoraMethodLoraConfig(use_ortho=True, use_dora=True)


def test_ia3_emits_use_ia3_network_arg(tmp_path: Path) -> None:
    """``backend.animaLora.lora.useIa3=True`` reaches train.py via network_args."""
    from lorahub.core.config.schema import AnimaLoraMethodLoraConfig

    opts = AnimaLoraOptions(
        lora=AnimaLoraMethodLoraConfig(use_ortho=False, use_ia3=True),
    )
    cfg = _recipe(tmp_path, opts)
    argv, _ = compile_config(cfg, tmp_path / "ws")
    assert "use_ia3=true" in argv


def test_ia3_mutex_with_dora_or_ortho_rejected(tmp_path: Path) -> None:
    """IA3 has no LoRA legs to compose with — pydantic rejects the combo."""
    from lorahub.core.config.schema import AnimaLoraMethodLoraConfig

    with pytest.raises(ValueError, match="use_ia3"):
        AnimaLoraMethodLoraConfig(use_ortho=False, use_ia3=True, use_dora=True)
    with pytest.raises(ValueError, match="use_ia3"):
        AnimaLoraMethodLoraConfig(use_ortho=True, use_ia3=True)
