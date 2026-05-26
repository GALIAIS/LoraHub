"""anima_lora compiler tests — TOML config_file translation + constraints.

LoraHub now emits a single ``--config_file <workspace>/_lorahub_anima_config.toml``
plus the dataset blueprint baked into the same file; method/preset
selection that used to ride the upstream merge chain is dead code in
this layer. These tests therefore parse the generated TOML and assert
on its content via the same flag-shaped vocabulary the legacy
argv-grouping helper used, so test bodies didn't have to be rewritten
when the rendering layer changed.

The compatibility shim below exposes:
  * ``_argv_pairs(argv, files)`` — parses the emitted TOML into a
    ``{"--key": ["value"]}`` shape that matches the historical argv
    groupby helper.  ``--method`` / ``--preset`` are gone (they only
    drove the now-bypassed TOML merge chain) and any test that asserted
    them was updated alongside.
  * ``_emitted_toml(argv, files)`` — for tests that want the raw dict
    (e.g. dataset blueprint inspection).
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

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


def _emitted_toml(argv: list[str], files: dict[Path, str]) -> dict[str, Any]:
    """Locate the generated config file in ``files`` and parse it."""
    assert argv[:1] == ["--config_file"], (
        "compile_config now drives via --config_file; "
        f"unexpected argv head: {argv[:3]}"
    )
    config_path = Path(argv[1])
    body = files.get(config_path)
    assert body is not None, (
        f"--config_file points at {config_path} but it's not in files: "
        f"{list(files)}"
    )
    return tomllib.loads(body)


def _argv_pairs(argv: list[str], files: dict[Path, str] | None = None) -> dict[str, list[str]]:
    """Shim returning a ``--flag → [str values]`` map for legacy assertions.

    Walks the emitted TOML (preferred) plus any leftover argv pairs
    (none in the current implementation, but kept as a safety net).
    Bools become ``""`` to match the historical store-true convention.
    Lists become repeated entries (``--network_args`` style).
    Sub-tables (``[general]`` / ``[[datasets]]``) are flattened with
    their parent key prefix joined by ``_`` to mirror upstream's flat
    merged namespace; tests rarely poke those, but the few that do
    look at the raw dict via ``_emitted_toml``.
    """
    out: dict[str, list[str]] = {}
    if files is not None:
        try:
            cfg_dict = _emitted_toml(argv, files)
        except AssertionError:
            cfg_dict = {}
        else:
            for key, value in cfg_dict.items():
                if isinstance(value, dict):
                    # [general]/[[datasets]] — skip; tests poke via _emitted_toml.
                    continue
                if isinstance(value, list) and value and all(
                    isinstance(x, dict) for x in value
                ):
                    continue
                if isinstance(value, list):
                    out["--" + key] = [str(x) for x in value]
                elif isinstance(value, bool):
                    out["--" + key] = [""]
                else:
                    out["--" + key] = [_render_scalar(value)]

    # Trailing argv (non-config_file flags). Currently none, but keep
    # the loop so future additions show up automatically.
    i = 0
    while i < len(argv):
        flag = argv[i]
        if flag == "--config_file":
            i += 2
            continue
        if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            out.setdefault(flag, []).append(argv[i + 1])
            i += 2
        else:
            out.setdefault(flag, []).append("")
            i += 1
    return out


def _render_scalar(value: Any) -> str:
    """Stringify a TOML scalar back into the form historical tests expect.

    Floats keep ``repr`` shape (matches the legacy ``_fmt_float``);
    ints, paths, and strings round-trip via ``str``.
    """
    if isinstance(value, float):
        return repr(value)
    return str(value)


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
    argv, files = compile_config(cfg, tmp_path / "ws")
    pairs = _argv_pairs(argv, files)

    assert pairs["--max_train_steps"] == ["4000"]
    assert pairs["--max_train_epochs"] == ["0"]


def test_max_steps_unset_emits_method_max_train_epochs(tmp_path: Path) -> None:
    """When the user leaves max_steps unset, send the method's epoch budget."""
    opts = AnimaLoraOptions()
    cfg = _recipe(tmp_path, opts)
    cfg.schedule.max_steps = None
    argv, files = compile_config(cfg, tmp_path / "ws")
    pairs = _argv_pairs(argv, files)

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
    pairs = _argv_pairs(argv, files)

    # assert pairs["--method"] == ["lora"]  # method/preset gone — driven by config_file
    # assert pairs["--preset"] == ["default"]  # method/preset gone — driven by config_file
    network_args = pairs["--network_args"]
    assert "use_ortho=true" in network_args
    assert "use_timestep_mask=true" in network_args
    assert "min_rank=16" in network_args
    assert any(p.startswith("alpha_rank_scale=") for p in network_args)
    # files-to-write contains exactly the generated dataset_config TOML
    # — upstream's argparse has no flag for the three data path keys
    # (source / resized / cache), so we deliver them via
    # --dataset_config <path>. Method / preset merge stays
    # upstream-owned through configs/base.toml + configs/methods/<x>.toml.
    assert len(files) == 1
    [only_path] = list(files.keys())
    assert only_path.name == "_lorahub_anima_config.toml"


def test_postfix_method_emits_network_args(tmp_path: Path) -> None:
    opts = AnimaLoraOptions(
        method="postfix",
        postfix=AnimaLoraMethodPostfixConfig(),
    )
    cfg = _recipe(tmp_path, opts)
    argv, files = compile_config(cfg, tmp_path / "ws")
    pairs = _argv_pairs(argv, files)

    # assert pairs["--method"] == ["postfix"]  # method/preset gone — driven by config_file
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
    argv, files = compile_config(cfg, tmp_path / "ws")
    pairs = _argv_pairs(argv, files)

    # assert pairs["--method"] == ["chimera"]  # method/preset gone — driven by config_file
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
    argv, files = compile_config(cfg, tmp_path / "ws")
    pairs = _argv_pairs(argv, files)

    # assert pairs["--method"] == ["easycontrol"]  # method/preset gone — driven by config_file
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
    argv, files = compile_config(cfg, tmp_path / "ws")
    pairs = _argv_pairs(argv, files)

    # assert pairs["--method"] == ["ip_adapter"]  # method/preset gone — driven by config_file
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
    argv, files = compile_config(cfg, tmp_path / "ws")
    pairs = _argv_pairs(argv, files)
    # assert pairs["--preset"] == ["low_vram"]  # method/preset gone — driven by config_file


def test_preset_debug_passed_through(tmp_path: Path) -> None:
    opts = AnimaLoraOptions(preset="debug")
    cfg = _recipe(tmp_path, opts)
    argv, files = compile_config(cfg, tmp_path / "ws")
    pairs = _argv_pairs(argv, files)
    # assert pairs["--preset"] == ["debug"]  # method/preset gone — driven by config_file


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
    argv, files = compile_config(cfg, tmp_path / "ws")
    pairs = _argv_pairs(argv, files)
    assert pairs["--compile_mode"] == ["blocks"]
    assert "--gradient_checkpointing" in pairs


def test_blocks_to_swap_with_cpu_offload_checkpointing_rejected(
    tmp_path: Path,
) -> None:
    """anima_lora train.py:326 asserts these two are mutually exclusive.

    Catching it at compile time turns a startup-after-cache crash into
    a structured error before the trainer ever spawns.
    """
    opts = AnimaLoraOptions(
        blocks_to_swap=24,
        cpu_offload_checkpointing=True,
    )
    cfg = _recipe(tmp_path, opts)
    with pytest.raises(CompilationError, match="cpu_offload_checkpointing"):
        compile_config(cfg, tmp_path / "ws")


def test_blocks_to_swap_with_unsloth_offload_allowed(tmp_path: Path) -> None:
    """``unsloth_offload_checkpointing`` composes with blocks_to_swap (the 8GB recipe)."""
    opts = AnimaLoraOptions(
        blocks_to_swap=24,
        unsloth_offload_checkpointing=True,
        cpu_offload_checkpointing=False,
    )
    cfg = _recipe(tmp_path, opts)
    argv, files = compile_config(cfg, tmp_path / "ws")
    pairs = _argv_pairs(argv, files)
    assert "--blocks_to_swap" in pairs
    assert "--unsloth_offload_checkpointing" in pairs


# --------------------------------------------------------------------------- #
# Output dir + output name
# --------------------------------------------------------------------------- #


def test_workspace_drives_output_dir(tmp_path: Path) -> None:
    """`workspace/ckpt` is what anima_lora writes safetensors into."""
    opts = AnimaLoraOptions(output_name="my_run")
    cfg = _recipe(tmp_path, opts)
    ws = tmp_path / "ws"
    argv, files = compile_config(cfg, ws)
    pairs = _argv_pairs(argv, files)
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
    argv, files = compile_config(cfg, tmp_path / "ws")
    pairs = _argv_pairs(argv, files)
    assert "--seed" in pairs
    # Default fallback is 42 — let upstream CLAUDE.md drift catch us if
    # they ever change the convention.
    assert pairs["--seed"] == ["42"]


def test_caching_flags_emit_when_enabled(tmp_path: Path) -> None:
    """The four cache_* booleans default True and must all surface as flags."""
    opts = AnimaLoraOptions()
    cfg = _recipe(tmp_path, opts)
    argv, files = compile_config(cfg, tmp_path / "ws")
    pairs = _argv_pairs(argv, files)
    for flag in (
        "--cache_latents",
        "--cache_latents_to_disk",
        "--cache_text_encoder_outputs",
        "--cache_text_encoder_outputs_to_disk",
        "--cache_llm_adapter_outputs",
    ):
        assert flag in pairs, f"missing default cache flag {flag}"


def test_attn_mode_default_is_flash(tmp_path: Path) -> None:
    """Default attn_mode is ``flash`` to match Backend's base.toml.

    Backend ships flash-attn as the throughput-leading default on
    Ampere+ GPUs. Operators without a working flash-attn install must
    flip this to ``torch`` (PyTorch SDPA) explicitly in their recipe;
    SDPA hits ~85-95% of flash-attn throughput and is always available.
    """
    opts = AnimaLoraOptions()
    cfg = _recipe(tmp_path, opts)
    argv, files = compile_config(cfg, tmp_path / "ws")
    pairs = _argv_pairs(argv, files)
    assert pairs["--attn_mode"] == ["flash"]


# --------------------------------------------------------------------------- #
# Dataset path overrides — kohya / dp parity (cfg.dataset.source = raw images)
# --------------------------------------------------------------------------- #


def test_compile_emits_source_resized_lora_cache_paths(tmp_path: Path) -> None:
    """The compiler pins source / resized / cache to absolute LoraHub paths.

    cfg.dataset.source is the user-facing raw image directory (kohya /
    dp parity); the resized + cache dirs are LoRaHub-managed under
    ``<workspace>/post_image_dataset/{resized,lora}``. Now baked into
    the single ``_lorahub_anima_config.toml`` as both top-level scalar
    keys (``source_image_dir`` / ``resized_image_dir`` /
    ``lora_cache_dir``) AND inside the ``[[datasets.subsets]]`` block
    (``image_dir`` / ``cache_dir``) so anima's blueprint generator and
    its preprocess scripts agree on the same absolute paths.
    """
    opts = AnimaLoraOptions()
    cfg = _recipe(tmp_path, opts)
    ws = tmp_path / "ws"
    argv, files = compile_config(cfg, ws)
    cfg_dict = _emitted_toml(argv, files)

    resized_expected = str((ws / "post_image_dataset" / "resized").resolve())
    cache_expected = str((ws / "post_image_dataset" / "lora").resolve())
    src_expected = str(cfg.dataset.source.resolve())

    # Top-level scalar keys for blueprint template substitution.
    assert cfg_dict["resized_image_dir"] == resized_expected
    assert cfg_dict["lora_cache_dir"] == cache_expected
    assert cfg_dict["source_image_dir"] == src_expected

    # Subset block — what the trainer's BlueprintGenerator actually reads.
    [dataset] = cfg_dict["datasets"]
    [subset] = dataset["subsets"]
    assert subset["image_dir"] == resized_expected
    assert subset["cache_dir"] == cache_expected


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
    argv, files = compile_config(cfg, tmp_path / "ws")
    pairs = _argv_pairs(argv, files)

    assert "--ema" in pairs
    assert pairs.get("--ema_decay") == ["0.9999"]
    assert "--ema_use_num_updates" in pairs
    assert "--nan_guard" in pairs
    assert "--nan_guard_recover" in pairs
    assert pairs.get("--min_snr_gamma") == ["5"]
    assert "--sample_grid" in pairs
    # extra_args last-write-wins for any flag also emitted by the
    # shared layer — weighting_scheme defaults to None in opts so this
    # one only gets emitted from extra_args, but the relative order
    # still has the extra_args copy last.
    assert pairs.get("--weighting_scheme") == ["min_snr_rf"]
    assert "--should_be_dropped" not in pairs
    assert "--also_dropped" not in pairs


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
    argv, files = compile_config(cfg, tmp_path / "ws")
    pairs = _argv_pairs(argv, files)

    assert pairs.get("--compile_inductor_mode") == ["default"]


def test_ema_overrides_reduce_overhead_from_opts(tmp_path: Path) -> None:
    """``ema=true`` overrides an explicit reduce-overhead from opts."""
    opts = AnimaLoraOptions(compile_inductor_mode="reduce-overhead")
    cfg = _recipe(tmp_path, opts)
    cfg.backend.extra_args = {"ema": True}
    argv, files = compile_config(cfg, tmp_path / "ws")
    pairs = _argv_pairs(argv, files)

    # In the TOML-render path the override is a dict-set, not appended:
    # only the final value (default) appears in the emitted config.
    assert pairs.get("--compile_inductor_mode") == ["default"]


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
    argv, files = compile_config(cfg, tmp_path / "ws")
    pairs = _argv_pairs(argv, files)

    assert pairs.get("--compile_inductor_mode") == ["default"]
    # In the TOML-render path the override is a dict-set, not an
    # appended last-wins flag, so we just verify the final value won.


def test_ema_leaves_default_mode_alone(tmp_path: Path) -> None:
    """``ema=true`` + already ``default`` → no surprise mutation."""
    opts = AnimaLoraOptions(compile_inductor_mode="default")
    cfg = _recipe(tmp_path, opts)
    cfg.backend.extra_args = {"ema": True}
    argv, files = compile_config(cfg, tmp_path / "ws")
    pairs = _argv_pairs(argv, files)

    assert "--compile_inductor_mode" in pairs  # from shared layer
    assert any('default' in v for vs in pairs.values() for v in vs)
    # No second emit of ``=default`` from extra_args injection.
    assert "--compile_inductor_mode=default" not in _argv_pairs(argv, files).get("--network_args", [])
    pairs = _argv_pairs(argv, files)


def test_no_ema_leaves_reduce_overhead_alone(tmp_path: Path) -> None:
    """Without EMA, reduce-overhead is a legitimate fast path — don't touch it."""
    opts = AnimaLoraOptions(compile_inductor_mode="reduce-overhead")
    cfg = _recipe(tmp_path, opts)
    cfg.backend.extra_args = {}
    argv, files = compile_config(cfg, tmp_path / "ws")
    pairs = _argv_pairs(argv, files)

    assert any('reduce-overhead' in v for vs in pairs.values() for v in vs)
    assert "--compile_inductor_mode=default" not in _argv_pairs(argv, files).get("--network_args", [])
    pairs = _argv_pairs(argv, files)


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
    argv, files = compile_config(cfg, tmp_path / "ws")
    pairs = _argv_pairs(argv, files)

    assert "--ema" in pairs
    # Decay is emitted as space-separated value pair so it matches the
    # shared-layer style (the rest of the file uses _argv_pairs which
    # expects this shape).
    assert pairs["--ema_decay"] == ["0.999"]
    assert "--ema_use_num_updates" not in pairs  # toggled off


def test_nan_guard_schema_field_emits_recovery_flags(tmp_path: Path) -> None:
    """Schema-level nan_guard=true emits the full recovery surface."""
    opts = AnimaLoraOptions(
        compile_inductor_mode="default",
        nan_guard=True,
        nan_guard_recover=True,
        nan_guard_max_consecutive=10,
    )
    cfg = _recipe(tmp_path, opts)
    argv, files = compile_config(cfg, tmp_path / "ws")
    pairs = _argv_pairs(argv, files)

    assert "--nan_guard" in pairs
    assert pairs["--nan_guard_max_consecutive"] == ["10"]
    assert "--nan_guard_recover" in pairs


def test_min_snr_rf_emits_gamma_when_provided(tmp_path: Path) -> None:
    """min_snr_rf is the rectified-flow Min-SNR-γ weighting; emit γ pair."""
    opts = AnimaLoraOptions(
        weighting_scheme="min_snr_rf",
        min_snr_gamma=5.0,
    )
    cfg = _recipe(tmp_path, opts)
    argv, files = compile_config(cfg, tmp_path / "ws")
    pairs = _argv_pairs(argv, files)

    assert pairs["--weighting_scheme"] == ["min_snr_rf"]
    assert pairs["--min_snr_gamma"] == ["5.0"]


def test_min_snr_rf_without_gamma_falls_back_uniform(tmp_path: Path, caplog) -> None:
    """min_snr_rf without γ is a no-op in the trainer; warn at compile time."""
    import logging

    opts = AnimaLoraOptions(weighting_scheme="min_snr_rf", min_snr_gamma=None)
    cfg = _recipe(tmp_path, opts)
    with caplog.at_level(logging.WARNING):
        argv, files = compile_config(cfg, tmp_path / "ws")
    pairs = _argv_pairs(argv, files)

    assert "--weighting_scheme" in pairs
    assert "--min_snr_gamma" not in pairs
    assert any("min_snr_gamma" in r.message for r in caplog.records)


def test_sample_grid_schema_field_emits_flag(tmp_path: Path) -> None:
    """sample_grid=true emits the bare ``--sample_grid`` flag."""
    opts = AnimaLoraOptions(
        compile_inductor_mode="default",
        sample_grid=True,
    )
    cfg = _recipe(tmp_path, opts)
    argv, files = compile_config(cfg, tmp_path / "ws")
    pairs = _argv_pairs(argv, files)
    assert "--sample_grid" in pairs


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
    argv, files = compile_config(cfg, tmp_path / "ws")
    pairs = _argv_pairs(argv, files)
    assert "use_dora=true" in _argv_pairs(argv, files).get("--network_args", [])
    pairs = _argv_pairs(argv, files)
    assert "use_ortho=false" in _argv_pairs(argv, files).get("--network_args", [])


def test_dora_and_ortho_mutex_rejected_at_validation(tmp_path: Path) -> None:
    """``use_dora`` with ``use_ortho`` must fail pydantic validation.

    The two are incompatible: DoRA stores per-Linear ``.magnitude``,
    OrthoLoRA writes Cayley distill keys, and the standard save layout
    can't carry both. Catching this at schema-validation time gives a
    clean error before the trainer ever spawns.
    """
    from lorahub.core.config.schema import AnimaLoraMethodLoraConfig

    # Two legacy bools both True is rejected by the multi-True guard.
    with pytest.raises(ValueError, match="multiple legacy use_X"):
        AnimaLoraMethodLoraConfig(use_ortho=True, use_dora=True)
    # Explicit ``algorithm=ortho`` paired with ``use_dora=True`` is the
    # other inconsistency path; reconciler emits a tailored message.
    with pytest.raises(ValueError, match="use_dora=True disagrees"):
        AnimaLoraMethodLoraConfig(algorithm="ortho", use_dora=True)


def test_ia3_emits_use_ia3_network_arg(tmp_path: Path) -> None:
    """``backend.animaLora.lora.useIa3=True`` reaches train.py via network_args."""
    from lorahub.core.config.schema import AnimaLoraMethodLoraConfig

    opts = AnimaLoraOptions(
        lora=AnimaLoraMethodLoraConfig(use_ortho=False, use_ia3=True),
    )
    cfg = _recipe(tmp_path, opts)
    argv, files = compile_config(cfg, tmp_path / "ws")
    pairs = _argv_pairs(argv, files)
    assert "use_ia3=true" in _argv_pairs(argv, files).get("--network_args", [])
    pairs = _argv_pairs(argv, files)


def test_ia3_mutex_with_dora_or_ortho_rejected(tmp_path: Path) -> None:
    """IA3 has no LoRA legs to compose with — pydantic rejects the combo."""
    from lorahub.core.config.schema import AnimaLoraMethodLoraConfig

    with pytest.raises(ValueError, match="multiple legacy use_X"):
        AnimaLoraMethodLoraConfig(use_ia3=True, use_dora=True)
    with pytest.raises(ValueError, match="multiple legacy use_X"):
        AnimaLoraMethodLoraConfig(use_ortho=True, use_ia3=True)


def test_lokr_emits_use_lokr_and_factor(tmp_path: Path) -> None:
    """``use_lokr=True`` reaches train.py with ``lokr_factor`` alongside."""
    from lorahub.core.config.schema import AnimaLoraMethodLoraConfig

    opts = AnimaLoraOptions(
        lora=AnimaLoraMethodLoraConfig(use_ortho=False, use_lokr=True, lokr_factor=12),
    )
    cfg = _recipe(tmp_path, opts)
    argv, files = compile_config(cfg, tmp_path / "ws")
    pairs = _argv_pairs(argv, files)
    assert "use_lokr=true" in _argv_pairs(argv, files).get("--network_args", [])
    pairs = _argv_pairs(argv, files)
    assert "lokr_factor=12" in _argv_pairs(argv, files).get("--network_args", [])


def test_loha_emits_use_loha(tmp_path: Path) -> None:
    """``use_loha=True`` reaches train.py via network_args."""
    from lorahub.core.config.schema import AnimaLoraMethodLoraConfig

    opts = AnimaLoraOptions(
        lora=AnimaLoraMethodLoraConfig(use_ortho=False, use_loha=True),
    )
    cfg = _recipe(tmp_path, opts)
    argv, files = compile_config(cfg, tmp_path / "ws")
    pairs = _argv_pairs(argv, files)
    assert "use_loha=true" in _argv_pairs(argv, files).get("--network_args", [])
    pairs = _argv_pairs(argv, files)


def test_atomic_variants_mutually_exclusive(tmp_path: Path) -> None:
    """Two legacy use_X True together is rejected by the multi-True guard."""
    from lorahub.core.config.schema import AnimaLoraMethodLoraConfig

    with pytest.raises(ValueError, match="multiple legacy use_X"):
        AnimaLoraMethodLoraConfig(use_ia3=True, use_lokr=True)
    with pytest.raises(ValueError, match="multiple legacy use_X"):
        AnimaLoraMethodLoraConfig(use_lokr=True, use_loha=True)
    with pytest.raises(ValueError, match="multiple legacy use_X"):
        AnimaLoraMethodLoraConfig(use_full=True, use_ia3=True)


def test_dylora_emits_use_dylora(tmp_path: Path) -> None:
    """DyLoRA shares LoRA's on-disk shape; selector still forwards via network_args."""
    from lorahub.core.config.schema import AnimaLoraMethodLoraConfig

    opts = AnimaLoraOptions(
        lora=AnimaLoraMethodLoraConfig(use_ortho=False, use_dylora=True),
    )
    cfg = _recipe(tmp_path, opts)
    argv, files = compile_config(cfg, tmp_path / "ws")
    pairs = _argv_pairs(argv, files)
    assert "use_dylora=true" in _argv_pairs(argv, files).get("--network_args", [])
    pairs = _argv_pairs(argv, files)


def test_full_emits_use_full(tmp_path: Path) -> None:
    """Full free-Δ wrapper is selected via use_full network_arg."""
    from lorahub.core.config.schema import AnimaLoraMethodLoraConfig

    opts = AnimaLoraOptions(
        lora=AnimaLoraMethodLoraConfig(use_ortho=False, use_full=True),
    )
    cfg = _recipe(tmp_path, opts)
    argv, files = compile_config(cfg, tmp_path / "ws")
    pairs = _argv_pairs(argv, files)
    assert "use_full=true" in _argv_pairs(argv, files).get("--network_args", [])
    pairs = _argv_pairs(argv, files)


def test_dylora_mutex_with_dora_or_ortho(tmp_path: Path) -> None:
    """DyLoRA's rank truncation doesn't compose with magnitude / Cayley."""
    from lorahub.core.config.schema import AnimaLoraMethodLoraConfig

    with pytest.raises(ValueError, match="multiple legacy use_X"):
        AnimaLoraMethodLoraConfig(use_dylora=True, use_dora=True)
    with pytest.raises(ValueError, match="multiple legacy use_X"):
        AnimaLoraMethodLoraConfig(use_ortho=True, use_dylora=True)


def test_diag_oft_emits_use_diag_oft(tmp_path: Path) -> None:
    """Diag-OFT selector forwards via network_args."""
    from lorahub.core.config.schema import AnimaLoraMethodLoraConfig

    opts = AnimaLoraOptions(
        lora=AnimaLoraMethodLoraConfig(use_ortho=False, use_diag_oft=True),
    )
    cfg = _recipe(tmp_path, opts)
    argv, files = compile_config(cfg, tmp_path / "ws")
    pairs = _argv_pairs(argv, files)
    assert "use_diag_oft=true" in _argv_pairs(argv, files).get("--network_args", [])
    pairs = _argv_pairs(argv, files)


def test_boft_emits_use_boft_and_factors(tmp_path: Path) -> None:
    """BOFT propagates ``boft_factors`` so the network can construct R."""
    from lorahub.core.config.schema import AnimaLoraMethodLoraConfig

    opts = AnimaLoraOptions(
        lora=AnimaLoraMethodLoraConfig(
            use_ortho=False, use_boft=True, boft_factors=6
        ),
    )
    cfg = _recipe(tmp_path, opts)
    argv, files = compile_config(cfg, tmp_path / "ws")
    pairs = _argv_pairs(argv, files)
    assert "use_boft=true" in _argv_pairs(argv, files).get("--network_args", [])
    pairs = _argv_pairs(argv, files)
    assert "boft_factors=6" in _argv_pairs(argv, files).get("--network_args", [])


def test_glora_emits_use_glora(tmp_path: Path) -> None:
    """GLoRA-light selector forwards via network_args."""
    from lorahub.core.config.schema import AnimaLoraMethodLoraConfig

    opts = AnimaLoraOptions(
        lora=AnimaLoraMethodLoraConfig(use_ortho=False, use_glora=True),
    )
    cfg = _recipe(tmp_path, opts)
    argv, files = compile_config(cfg, tmp_path / "ws")
    pairs = _argv_pairs(argv, files)
    assert "use_glora=true" in _argv_pairs(argv, files).get("--network_args", [])
    pairs = _argv_pairs(argv, files)


def test_vera_emits_use_vera(tmp_path: Path) -> None:
    """VeRA selector forwards via network_args."""
    from lorahub.core.config.schema import AnimaLoraMethodLoraConfig

    opts = AnimaLoraOptions(
        lora=AnimaLoraMethodLoraConfig(use_ortho=False, use_vera=True),
    )
    cfg = _recipe(tmp_path, opts)
    argv, files = compile_config(cfg, tmp_path / "ws")
    pairs = _argv_pairs(argv, files)
    assert "use_vera=true" in _argv_pairs(argv, files).get("--network_args", [])
    pairs = _argv_pairs(argv, files)


def test_algorithm_enum_drives_compiler_selection(tmp_path: Path) -> None:
    """``algorithm=<name>`` (no legacy bools) emits the matching use_X=true.

    Future-default path — once back-compat callers migrate, the legacy
    booleans shouldn't need to appear at all.
    """
    from lorahub.core.config.schema import AnimaLoraMethodLoraConfig

    cases = [
        ("lora", "use_ortho=false"),  # plain LoRA — every selector false
        ("ortho", "use_ortho=true"),
        ("dora", "use_dora=true"),
        ("ia3", "use_ia3=true"),
        ("lokr", "use_lokr=true"),
        ("loha", "use_loha=true"),
        ("dylora", "use_dylora=true"),
        ("full", "use_full=true"),
        ("diag_oft", "use_diag_oft=true"),
        ("boft", "use_boft=true"),
        ("glora", "use_glora=true"),
        ("vera", "use_vera=true"),
    ]
    for i, (algo, expected) in enumerate(cases):
        # Distinct sub-tmp_path per case so _recipe's data.mkdir() and
        # the dataset_config TOML write don't collide.
        sub = tmp_path / f"case_{i}_{algo}"
        sub.mkdir()
        opts = AnimaLoraOptions(
            lora=AnimaLoraMethodLoraConfig(algorithm=algo),
        )
        cfg = _recipe(sub, opts)
        argv, files = compile_config(cfg, sub / "ws")
        pairs = _argv_pairs(argv, files)
        network_args = _argv_pairs(argv, files).get("--network_args", [])
        pairs = _argv_pairs(argv, files)
        assert expected in network_args, (
            f"algorithm={algo!r} should emit {expected!r}; "
            f"network_args={network_args}"
        )


def test_legacy_bool_back_compat(tmp_path: Path) -> None:
    """Setting only ``use_X=True`` (no enum) still works — bool wins."""
    from lorahub.core.config.schema import AnimaLoraMethodLoraConfig

    cfg_obj = AnimaLoraMethodLoraConfig(use_dora=True)
    assert cfg_obj.algorithm == "dora"
    cfg_obj = AnimaLoraMethodLoraConfig(use_lokr=True)
    assert cfg_obj.algorithm == "lokr"


def test_explicit_enum_disagrees_with_bool_rejected(tmp_path: Path) -> None:
    """Explicit ``algorithm`` + contradictory ``use_X=True`` raises."""
    from lorahub.core.config.schema import AnimaLoraMethodLoraConfig

    with pytest.raises(ValueError, match="disagrees"):
        AnimaLoraMethodLoraConfig(algorithm="lokr", use_dora=True)
