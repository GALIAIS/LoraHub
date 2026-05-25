"""Compile a TrainingConfig into anima_lora launch argv.

LoraHub now bypasses the upstream four-layer TOML chain
(``base.toml -> presets.toml -> methods/<method>.toml -> CLI``)
entirely. We materialise a single ``_lorahub_anima_config.toml`` under
the workspace containing every key LoraHub knows about, then launch
``train.py --config_file <path>``.  Upstream's ``read_config_from_file``
takes that branch and skips ``--method/--preset`` merging, so:

  * Every value the user sees in the LoraHub UI is the value the
    trainer will actually use — no more "I changed the LR but the
    method TOML still wins because CLI is parsed before TOML merge".
  * The vendored ``configs/`` tree becomes a documentation reference,
    not load-bearing: nothing in there is read at training time.

This module is a pure function: callers pass the recipe and a workspace,
and get back ``(argv, files_to_write)``. ``files_to_write`` carries the
generated TOML keyed by absolute path; ``backend.launch`` writes them
to disk before spawning ``train.py``.

Constraints we still enforce at compile time (vs letting upstream
crash mid-launch):

  * ``compile_mode='full'`` is incompatible with grad checkpointing /
    unsloth offload / ``blocks_to_swap > 0``.
  * ``blocks_to_swap > 0`` is incompatible with
    ``cpu_offload_checkpointing=true``.

Locked fields (``LOCKED_FIELDS``) are still surfaced as warnings rather
than hard errors — most are advisory now that LoraHub owns the entire
config; we keep them for the UI's 🔒 badges.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from lorahub.core.config.schema import (
    AnimaLoraOptions,
    TrainingConfig,
)

__all__ = [
    "CompilationError",
    "LOCKED_FIELDS",
    "compile_config",
    "compile_turbo_config",
]


# Fields LoraHub exposes for completeness but anima_lora's upstream
# argparse can't really change. Each entry: ``field name → (kind, why)``.
# The compiler logs a warning when the user overrides one of these to a
# value the trainer would silently ignore. The frontend uses the same
# table to render the 🔒 / ⚠️ badges.
LOCKED_FIELDS: dict[str, dict[str, str]] = {
    # — pipeline-locked booleans (base.toml ``= true``, no reverse flag) —
    "masked_loss": {
        "kind": "locked_true",
        "reason": "Anima 训练管线硬依赖 masked loss;关掉是无效操作。",
    },
    "torch_compile": {
        "kind": "locked_true",
        "reason": "torch.compile 是 static_token_count 性能收益的前提;upstream 训练循环假定开启。",
    },
    "skip_cache_check": {
        "kind": "locked_true",
        "reason": "缓存校验跳过对训练正确性无影响,只影响启动速度;关掉无意义。",
    },
    "dataloader_pin_memory": {
        "kind": "locked_true",
        "reason": "DataLoader pin_memory 一直开;upstream 没提供反向 flag。",
    },
    "enable_bucket": {
        "kind": "locked_true",
        "reason": "constant-token bucketing 是 Anima static-shape compile 的硬约束。",
    },
    "cache_latents": {
        "kind": "locked_true",
        "reason": "anima_lora 训练流程依赖预计算 latent 缓存,关掉训练会失败。",
    },
    "cache_latents_to_disk": {
        "kind": "locked_true",
        "reason": "缓存必须落盘以避免每次 epoch 重算。",
    },
    "cache_text_encoder_outputs": {
        "kind": "locked_true",
        "reason": "TE 输出必缓存,否则训练时拖累 Qwen3 的 forward。",
    },
    "cache_text_encoder_outputs_to_disk": {
        "kind": "locked_true",
        "reason": "TE 缓存必须落盘。",
    },
    "cache_llm_adapter_outputs": {
        "kind": "locked_true",
        "reason": "LLM adapter 输出必缓存。",
    },
    # — value-locked 整数 —
    "static_token_count": {
        "kind": "locked_value",
        "reason": "Anima DiT torch.compile 路径锁死 4096(constant-token bucket map)。其它值会引发每个分辨率重新编译。",
    },
    "vae_chunk_size": {
        "kind": "locked_value",
        "reason": "QwenImage VAE memory layout 锁死 64;改了多半 OOM 或无收益。",
    },
    "caption_extension": {
        "kind": "locked_value",
        "reason": "数据 pipeline 写死 .txt 后缀;改了所有图片会被跳过且无警告。",
    },
    "save_model_as": {
        "kind": "locked_value",
        "reason": "Anima 只能加载 safetensors;其它格式无法 round-trip。",
    },
    # — risky (能改但有副作用) —
    "vae_disable_cache": {
        "kind": "risky",
        "reason": "改成 false 会拖慢 VAE encode ~30%,但与官方 VAE 行为一致。",
    },
    "no_half_vae": {
        "kind": "risky",
        "reason": "true 半精度 VAE 省显存,但偶尔在边缘数据集上产生 NaN。",
    },
    "trim_crossattn_kv": {
        "kind": "risky",
        "reason": "true 启用 KV trimming(短 caption 加速 ~10-15%),但需匹配 caption 长度分布。",
    },
    "save_precision": {
        "kind": "risky",
        "reason": "fp32 是双倍体积无质量收益;fp16 略小但偶有量化损失;bf16 是 upstream 默认。",
    },
    "persistent_data_loader_workers": {
        "kind": "risky",
        "reason": "true 减少 epoch 边界 stall,但长跑可能泄漏 file handle。",
    },
    "keep_tokens": {
        "kind": "risky",
        "reason": "anima 训练模板把 trigger / character / character-feature 三件放前 3 位;改了 trigger word 不再可靠。",
    },
    "validation_split_num": {
        "kind": "risky",
        "reason": "0 = 关 CMMD 验证(val_loss 不再更新)。",
    },
}

_log = logging.getLogger(__name__)


# Filename used when LoraHub auto-generates a fallback prompts file
# under the job workspace because the user enabled sampling but didn't
# point at one. backend.launch materialises the file before spawning
# train.py — see ``_ensure_sample_prompts_file`` in backend.py.
DEFAULT_SAMPLE_PROMPTS_FILENAME = "_lorahub_sample_prompts.txt"


class CompilationError(ValueError):
    """Raised when an AnimaLoraOptions config can't be compiled."""


def compile_config(
    cfg: TrainingConfig,
    workspace: Path,
) -> tuple[list[str], dict[Path, str]]:
    """Translate a recipe into ``(argv, files_to_write)`` for anima_lora.

    Returns the argv to append after ``python <repo>/train.py``
    (or after ``accelerate launch``) plus a ``files_to_write`` dict
    the launcher writes to disk before spawn.

    The argv is intentionally short — every training knob lives in the
    generated ``_lorahub_anima_config.toml`` under the workspace.
    Only ``--config_file`` plus a tiny wrapper for ``cfg.backend.extra_args``
    (last-write-wins escape hatch) goes on the command line.
    """
    if cfg.backend.type != "anima_lora":
        msg = (
            f"anima_lora compiler invoked on backend.type={cfg.backend.type!r}; "
            "this is a programming error in the dispatch path"
        )
        raise CompilationError(msg)
    if cfg.backend.anima_lora is None:
        msg = (
            "backend.type='anima_lora' requires backend.animaLora to be set "
            "with at least default options; populate it in the recipe"
        )
        raise CompilationError(msg)

    opts = cfg.backend.anima_lora
    if opts.turbo is not None:
        msg = (
            "AnimaLoraOptions.turbo is set — use compile_turbo_config() "
            "instead of compile_config(). The backend.launch dispatch picks "
            "the right path automatically; if you're calling compile_config "
            "directly, drop opts.turbo first."
        )
        raise CompilationError(msg)

    _enforce_compile_constraints(opts)
    _warn_locked_fields_changed(opts)

    workspace = workspace.resolve()
    output_dir = workspace / "ckpt"

    config_dict = _render_full_config(cfg, opts, workspace, output_dir)

    # Last-write-wins extra_args overlay: lets a recipe poke any
    # train.py CLI flag without forcing every new knob into
    # AnimaLoraOptions before it stabilises. Keys may include the
    # leading ``--``; both forms work. ``False`` / ``None`` removes.
    # We overlay BEFORE the EMA cross-check so the override stays
    # on top.
    extra_args_dict = dict(cfg.backend.extra_args)
    _overlay_extra_args(extra_args_dict, config_dict)

    # cudagraph_trees × EMA cross-check — applies whether the user set
    # compile_inductor_mode explicitly in opts, in TOML render, or via
    # the extra_args escape hatch. When EMA is on we force ``default``
    # so the trainer doesn't crash mid-step.
    _apply_ema_compile_override(opts, config_dict, extra_args_dict)

    config_path = workspace / "_lorahub_anima_config.toml"
    files: dict[Path, str] = {config_path: _dump_toml(config_dict)}

    argv: list[str] = ["--config_file", str(config_path)]
    return argv, files


# --------------------------------------------------------------------------- #
# Config rendering
# --------------------------------------------------------------------------- #


def _render_full_config(
    cfg: TrainingConfig,
    opts: AnimaLoraOptions,
    workspace: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Build the complete TOML config tree.

    Returns a dict shaped like upstream's flattened argparse namespace
    plus the dataset blueprint sections (``[general]`` /
    ``[[datasets]]`` / ``[[datasets.subsets]]``) that the
    ``BlueprintGenerator`` consumes. Keys are upstream's snake_case
    flag names; values are TOML-native scalars / arrays / tables.
    """
    cfg_dict: dict[str, Any] = {}

    # ---- Output ----
    cfg_dict["output_dir"] = str(output_dir)
    cfg_dict["output_name"] = opts.output_name

    # ---- Model paths ----
    bm = cfg.base_model
    if bm.checkpoint:
        cfg_dict["pretrained_model_name_or_path"] = str(bm.checkpoint)
    if bm.arch_paths.qwen3 is not None:
        cfg_dict["qwen3"] = str(bm.arch_paths.qwen3)
    if bm.arch_paths.ae is not None:
        # anima_lora calls the VAE flag --vae upstream
        cfg_dict["vae"] = str(bm.arch_paths.ae)

    # ---- Network ----
    cfg_dict["network_module"] = opts.network_module
    cfg_dict["network_dim"] = int(opts.network_dim)
    cfg_dict["network_alpha"] = float(opts.network_alpha)
    if opts.network_train_unet_only:
        cfg_dict["network_train_unet_only"] = True

    # ---- Optimizer / schedule ----
    cfg_dict["optimizer_type"] = opts.optimizer_type
    cfg_dict["lr_scheduler"] = opts.lr_scheduler
    cfg_dict["learning_rate"] = float(opts.learning_rate)

    if cfg.schedule.max_steps is not None and cfg.schedule.max_steps > 0:
        # See compiler history for the long version: train.py
        # unconditionally recomputes max_train_steps when
        # max_train_epochs is non-zero, so we pin epochs to 0 to make
        # the user's explicit step cap stick.
        cfg_dict["max_train_steps"] = int(cfg.schedule.max_steps)
        cfg_dict["max_train_epochs"] = 0
    else:
        cfg_dict["max_train_epochs"] = int(opts.max_train_epochs)
    cfg_dict["save_every_n_epochs"] = int(opts.save_every_n_epochs)
    if opts.save_every_n_steps is not None and opts.save_every_n_steps > 0:
        cfg_dict["save_every_n_steps"] = int(opts.save_every_n_steps)
    cfg_dict["checkpointing_epochs"] = int(opts.checkpointing_epochs)
    if opts.caption_dropout_rate > 0:
        cfg_dict["caption_dropout_rate"] = float(opts.caption_dropout_rate)

    # ---- Sampling / loss (flow-matching) ----
    cfg_dict["timestep_sampling"] = opts.timestep_sampling
    cfg_dict["sigmoid_scale"] = float(opts.sigmoid_scale)
    cfg_dict["discrete_flow_shift"] = float(opts.discrete_flow_shift)
    if opts.weighting_scheme is not None:
        cfg_dict["weighting_scheme"] = opts.weighting_scheme
    if opts.min_snr_gamma is not None:
        cfg_dict["min_snr_gamma"] = float(opts.min_snr_gamma)
    elif opts.weighting_scheme == "min_snr_rf":
        _log.warning(
            "anima_lora: weighting_scheme='min_snr_rf' set without "
            "min_snr_gamma — the trainer falls back to uniform "
            "weighting; set min_snr_gamma (recommended 5.0) to enable.",
        )
    if opts.logit_mean is not None:
        cfg_dict["logit_mean"] = float(opts.logit_mean)
    if opts.logit_std is not None:
        cfg_dict["logit_std"] = float(opts.logit_std)
    if opts.mode_scale is not None:
        cfg_dict["mode_scale"] = float(opts.mode_scale)
    if opts.vr_loss_weight is not None:
        cfg_dict["vr_loss_weight"] = float(opts.vr_loss_weight)

    # ---- EMA / NaN guard / sample grid ----
    if opts.ema:
        cfg_dict["ema"] = True
        cfg_dict["ema_decay"] = float(opts.ema_decay)
        if opts.ema_use_num_updates:
            cfg_dict["ema_use_num_updates"] = True
    if opts.nan_guard:
        cfg_dict["nan_guard"] = True
        cfg_dict["nan_guard_max_consecutive"] = int(opts.nan_guard_max_consecutive)
        if opts.nan_guard_recover:
            cfg_dict["nan_guard_recover"] = True
    if opts.sample_grid:
        cfg_dict["sample_grid"] = True

    # ---- Caching / data ----
    if opts.cache_latents:
        cfg_dict["cache_latents"] = True
    if opts.cache_latents_to_disk:
        cfg_dict["cache_latents_to_disk"] = True
    if opts.cache_text_encoder_outputs:
        cfg_dict["cache_text_encoder_outputs"] = True
    if opts.cache_text_encoder_outputs_to_disk:
        cfg_dict["cache_text_encoder_outputs_to_disk"] = True
    if opts.cache_llm_adapter_outputs:
        cfg_dict["cache_llm_adapter_outputs"] = True
    if opts.use_shuffled_caption_variants:
        cfg_dict["use_shuffled_caption_variants"] = True
    if opts.sample_ratio is not None:
        cfg_dict["sample_ratio"] = float(opts.sample_ratio)
    if opts.static_token_count is not None:
        cfg_dict["static_token_count"] = int(opts.static_token_count)
    cfg_dict["vae_chunk_size"] = int(opts.vae_chunk_size)
    if opts.vae_disable_cache:
        cfg_dict["vae_disable_cache"] = True
    if opts.no_half_vae:
        cfg_dict["no_half_vae"] = True

    # ---- Attention / compile ----
    cfg_dict["attn_mode"] = opts.attn_mode
    if opts.xformers:
        cfg_dict["xformers"] = True
    if opts.split_attn:
        cfg_dict["split_attn"] = True
    if opts.compile_mode is not None:
        cfg_dict["compile_mode"] = opts.compile_mode
    if opts.compile_inductor_mode is not None:
        cfg_dict["compile_inductor_mode"] = opts.compile_inductor_mode
    if opts.enable_native_flatten:
        cfg_dict["enable_native_flatten"] = True
    if opts.bucket_table is not None and opts.bucket_table != "default":
        cfg_dict["bucket_table"] = opts.bucket_table

    # ---- Memory / offload ----
    if opts.blocks_to_swap > 0:
        cfg_dict["blocks_to_swap"] = int(opts.blocks_to_swap)
    if opts.gradient_checkpointing:
        cfg_dict["gradient_checkpointing"] = True
    if opts.unsloth_offload_checkpointing:
        cfg_dict["unsloth_offload_checkpointing"] = True
    if opts.cpu_offload_checkpointing:
        cfg_dict["cpu_offload_checkpointing"] = True
    cfg_dict["mixed_precision"] = opts.mixed_precision

    # ---- Validation ----
    if opts.use_cmmd:
        cfg_dict["use_cmmd"] = True
    if opts.validation_seed is not None:
        cfg_dict["validation_seed"] = int(opts.validation_seed)
    if opts.validation_sample_steps is not None:
        cfg_dict["validation_sample_steps"] = int(opts.validation_sample_steps)
    if opts.validation_cfg_scale is not None:
        cfg_dict["validation_cfg_scale"] = float(opts.validation_cfg_scale)

    # ---- Locked / risky cluster ----
    if opts.masked_loss:
        cfg_dict["masked_loss"] = True
    if opts.torch_compile:
        cfg_dict["torch_compile"] = True
    if opts.skip_cache_check:
        cfg_dict["skip_cache_check"] = True
    if opts.dataloader_pin_memory:
        cfg_dict["dataloader_pin_memory"] = True
    if opts.persistent_data_loader_workers:
        cfg_dict["persistent_data_loader_workers"] = True
    if opts.trim_crossattn_kv:
        cfg_dict["trim_crossattn_kv"] = True
    cfg_dict["save_model_as"] = opts.save_model_as
    cfg_dict["save_precision"] = opts.save_precision
    cfg_dict["log_every_n_steps"] = int(opts.log_every_n_steps)

    # ---- Seed ----
    seed = cfg.schedule.seed if cfg.schedule.seed is not None else 42
    cfg_dict["seed"] = int(seed)

    # ---- Resume / state writing ----
    if cfg.resume.save_state:
        cfg_dict["save_state"] = True
    if cfg.resume.save_state_at_end:
        cfg_dict["save_state_on_train_end"] = True
    if cfg.resume.save_last_n_epochs_state is not None:
        cfg_dict["save_last_n_epochs_state"] = int(
            cfg.resume.save_last_n_epochs_state,
        )
    if cfg.resume.save_last_n_steps_state is not None:
        cfg_dict["save_last_n_steps_state"] = int(
            cfg.resume.save_last_n_steps_state,
        )

    # ---- Sampling preview ----
    _render_sampling(cfg, workspace, cfg_dict)

    # ---- Method-specific (network_args + named flags) ----
    _render_method(opts, cfg_dict)

    # ---- Dataset blueprint ([general] / [[datasets]] / subsets) ----
    _render_dataset(cfg, opts, workspace, cfg_dict)

    return cfg_dict


def _render_sampling(
    cfg: TrainingConfig,
    workspace: Path,
    cfg_dict: dict[str, Any],
) -> None:
    """Translate ``cfg.sampling`` into upstream's ``--sample_*`` keys.

    Behaviour mirrors the historical CLI emitter: when sampling is
    disabled we emit nothing; otherwise we forward at_first / cadence /
    prompts_file as TOML scalars. The fallback prompts file under the
    workspace is materialised by ``backend.launch`` just before spawn.
    """
    sampling = cfg.sampling
    if not sampling.enabled:
        return
    if sampling.at_first:
        cfg_dict["sample_at_first"] = True
    if sampling.every_n_epochs and sampling.every_n_epochs > 0:
        cfg_dict["sample_every_n_epochs"] = int(sampling.every_n_epochs)
    if sampling.every_n_steps and sampling.every_n_steps > 0:
        cfg_dict["sample_every_n_steps"] = int(sampling.every_n_steps)
    if sampling.prompts_file is not None:
        prompts_path = Path(str(sampling.prompts_file))
    else:
        prompts_path = workspace / DEFAULT_SAMPLE_PROMPTS_FILENAME
    cfg_dict["sample_prompts"] = str(prompts_path)


def _render_method(opts: AnimaLoraOptions, cfg_dict: dict[str, Any]) -> None:
    """Add method-specific keys (network_args list + named bool flags).

    Upstream's ``--network_args`` is ``nargs="*"`` so in TOML it
    becomes ``network_args = ["use_ortho=true", "min_rank=8", ...]``.
    Method-named CLI flags (``use_easycontrol`` / ``use_ip_adapter``)
    are bool keys at the top level — io.py's flat-merge picks them up
    the same way it would pick up an ``--use_easycontrol`` argv.

    ``cfg_dict["use_custom_down_autograd"]`` is *not* a real flag — it
    rides ``network_args`` because the LoRA factory reads it out of the
    kwargs bag. Same trick the legacy CLI emitter used.
    """
    method = opts.method
    pieces: list[str] = []

    if method == "lora":
        pieces.extend(_lora_network_args(opts))
    elif method == "postfix":
        pieces.extend(_postfix_network_args(opts))
    elif method == "chimera":
        pieces.extend(_chimera_network_args(opts))
    elif method == "easycontrol":
        cfg_dict["use_easycontrol"] = True
        sub = opts.easycontrol
        if sub is None:
            msg = "method='easycontrol' missing sub-config"
            raise CompilationError(msg)
        cfg_dict["easycontrol_drop_p"] = float(sub.drop_p)
        cfg_dict["easycontrol_cond_noise_max"] = float(sub.cond_noise_max)
        pieces.extend(_easycontrol_network_args(opts))
    elif method == "ip_adapter":
        cfg_dict["use_ip_adapter"] = True
        sub = opts.ip_adapter
        if sub is None:
            msg = "method='ip_adapter' missing sub-config"
            raise CompilationError(msg)
        cfg_dict["ip_encoder"] = sub.encoder
        cfg_dict["ip_image_drop_p"] = float(sub.image_drop_p)
        if sub.features_cache_to_disk:
            cfg_dict["ip_features_cache_to_disk"] = True
        pieces.extend(_ip_adapter_network_args(opts))
    else:
        msg = f"unhandled method {opts.method!r} (schema enum drift?)"
        raise CompilationError(msg)

    # Universal — the LoRA factory reads this kwarg regardless of method.
    if opts.use_custom_down_autograd:
        pieces.append("use_custom_down_autograd=true")

    if pieces:
        cfg_dict["network_args"] = pieces


def _render_dataset(
    cfg: TrainingConfig,
    opts: AnimaLoraOptions,
    workspace: Path,
    cfg_dict: dict[str, Any],
) -> None:
    """Add the ``[general]`` / ``[[datasets]]`` / subset blueprint.

    Replaces the legacy ``--dataset_config`` separate-file approach;
    everything lives in the single ``_lorahub_anima_config.toml`` now.
    Path keys (``source_image_dir`` / ``resized_image_dir`` /
    ``lora_cache_dir``) are also written as top-level scalars so
    template substitution (``{resized_image_dir}`` etc.) inside the
    blueprint resolves to the LoraHub-managed dirs.
    """
    src = cfg.dataset.source.resolve()
    resized = (workspace / "post_image_dataset" / "resized").resolve()
    cache = (workspace / "post_image_dataset" / "lora").resolve()

    cfg_dict["source_image_dir"] = str(src)
    cfg_dict["resized_image_dir"] = str(resized)
    cfg_dict["lora_cache_dir"] = str(cache)
    cfg_dict["path_pattern"] = opts.path_pattern

    res = cfg.dataset.resolution
    if isinstance(res, (list, tuple)) and len(res) == 2:
        resolution: Any = [int(res[0]), int(res[1])]
    elif isinstance(res, (list, tuple)) and len(res) == 1:
        resolution = int(res[0])
    else:
        resolution = 1024

    batch_size = max(1, int(cfg.schedule.batch_size or 1))
    keep_tokens = int(opts.keep_tokens)
    caption_ext = (opts.caption_extension or ".txt").strip() or ".txt"
    num_repeats = max(1, int(getattr(cfg.dataset, "num_repeats", 1) or 1))

    cfg_dict["general"] = {
        "caption_extension": caption_ext,
        "keep_tokens": keep_tokens,
    }

    dataset_entry: dict[str, Any] = {
        "resolution": resolution,
        "batch_size": batch_size,
        "enable_bucket": bool(opts.enable_bucket),
        "validation_seed": int(opts.validation_seed) if opts.validation_seed is not None else 42,
        "validation_split_num": int(opts.validation_split_num),
        "subsets": [
            {
                "image_dir": str(resized),
                "cache_dir": str(cache),
                "num_repeats": num_repeats,
                "recursive": True,
            },
        ],
    }
    cfg_dict["datasets"] = [dataset_entry]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _apply_ema_compile_override(
    opts: AnimaLoraOptions,
    cfg_dict: dict[str, Any],
    extra_args: dict[str, Any],
) -> None:
    """When EMA is on, force ``compile_inductor_mode = "default"``.

    Background: ``reduce-overhead`` enables cudagraph_trees, whose
    liveness check fails on EMA's per-step shadow.copy_(...). Upstream
    base.toml ships ``reduce-overhead`` as default; without an explicit
    override the run crashes mid-step. We log + force ``default`` here.
    """
    ema_on = bool(opts.ema) or bool(extra_args.get("ema") is True)
    if not ema_on:
        return
    explicit_mode = cfg_dict.get("compile_inductor_mode")
    if extra_args.get("compile_inductor_mode") is not None:
        explicit_mode = extra_args["compile_inductor_mode"]
    if explicit_mode not in (None, "reduce-overhead"):
        return
    _log.warning(
        "anima_lora: ema=True with compile_inductor_mode=%r would "
        "trigger cudagraph_trees liveness check failure mid-step "
        "(EMA mutates LoRA params via detach/copy). Forcing "
        "compile_inductor_mode='default'. Set compile_inductor_mode "
        "explicitly to a non-reduce-overhead value to silence this.",
        explicit_mode if explicit_mode is not None else "<default reduce-overhead>",
    )
    cfg_dict["compile_inductor_mode"] = "default"


def _overlay_extra_args(
    extra_args: dict[str, Any],
    cfg_dict: dict[str, Any],
) -> None:
    """Merge ``cfg.backend.extra_args`` into the TOML dict, last-write-wins.

    The escape hatch for any train.py flag that hasn't (yet) been
    promoted to a typed AnimaLoraOptions field. Keys may include the
    leading ``--``; both forms are accepted.
    Boolean ``False`` / ``None`` removes the key.
    """
    for raw_key, value in extra_args.items():
        key = raw_key.lstrip("-")
        if value is False or value is None:
            cfg_dict.pop(key, None)
            continue
        cfg_dict[key] = value


# --------------------------------------------------------------------------- #
# Network-args generators (per method)
# --------------------------------------------------------------------------- #


def _lora_network_args(opts: AnimaLoraOptions) -> list[str]:
    """Map the LoRA family enum + knobs onto ``network_args`` pieces."""
    sub = opts.lora
    flag_for_algorithm: dict[str, str | None] = {
        "lora": None,
        "ortho": "use_ortho",
        "dora": "use_dora",
        "ia3": "use_ia3",
        "lokr": "use_lokr",
        "loha": "use_loha",
        "dylora": "use_dylora",
        "full": "use_full",
        "diag_oft": "use_diag_oft",
        "boft": "use_boft",
        "glora": "use_glora",
        "vera": "use_vera",
    }
    pieces: list[str] = []
    chosen = sub.algorithm
    for algo, flag in flag_for_algorithm.items():
        if flag is None:
            continue
        pieces.append(f"{flag}={'true' if algo == chosen else 'false'}")
    pieces.append(f"lokr_factor={sub.lokr_factor}")
    pieces.append(f"boft_factors={sub.boft_factors}")
    pieces.append(
        f"use_timestep_mask={'true' if sub.use_timestep_mask else 'false'}",
    )
    pieces.append(f"min_rank={sub.min_rank}")
    pieces.append(f"alpha_rank_scale={_fmt_float(sub.alpha_rank_scale)}")
    return pieces


def _postfix_network_args(opts: AnimaLoraOptions) -> list[str]:
    sub = opts.postfix
    if sub is None:
        msg = "method='postfix' missing sub-config (validator should have caught this)"
        raise CompilationError(msg)
    pieces = [
        f"mode={sub.mode}",
        f"cond_hidden_dim={sub.cond_hidden_dim}",
        f"splice_position={sub.splice_position}",
        f"ortho_basis={sub.ortho_basis}",
        f"svd_num_files={sub.svd_num_files}",
        f"ortho_basis_seed={sub.ortho_basis_seed}",
        f"lambda_init={sub.lambda_init}",
    ]
    if sub.te_cache_dir is not None:
        pieces.append(f"te_cache_dir={sub.te_cache_dir}")
    return pieces


def _chimera_network_args(opts: AnimaLoraOptions) -> list[str]:
    sub = opts.chimera
    if sub is None:
        msg = "method='chimera' missing sub-config"
        raise CompilationError(msg)
    return [
        "use_chimera_hydra=true",
        f"balance_w_content={_fmt_float(sub.balance_w_content)}",
        f"balance_w_freq={_fmt_float(sub.balance_w_freq)}",
        f"balance_loss_warmup_ratio={_fmt_float(sub.balance_loss_warmup_ratio)}",
        f"fei_feature_dim={sub.fei_feature_dim}",
        f"sigma_feature_dim={sub.sigma_feature_dim}",
    ]


def _easycontrol_network_args(opts: AnimaLoraOptions) -> list[str]:
    sub = opts.easycontrol
    if sub is None:
        msg = "method='easycontrol' missing sub-config"
        raise CompilationError(msg)
    return [
        f"b_cond_init={_fmt_float(sub.b_cond_init)}",
        f"cond_scale={_fmt_float(sub.cond_scale)}",
        f"apply_ffn_lora={'1' if sub.apply_ffn_lora else '0'}",
        f"cond_token_count={sub.cond_token_count}",
    ]


def _ip_adapter_network_args(opts: AnimaLoraOptions) -> list[str]:
    sub = opts.ip_adapter
    if sub is None:
        msg = "method='ip_adapter' missing sub-config"
        raise CompilationError(msg)
    return [
        f"ip_resampler_layers={sub.resampler_layers}",
        f"ip_resampler_heads={sub.resampler_heads}",
        f"ip_scale={_fmt_float(sub.ip_scale)}",
        f"gate_lr={_fmt_float(sub.gate_lr)}",
    ]


# --------------------------------------------------------------------------- #
# Minimal TOML writer (we only need scalars / arrays / nested tables /
# arrays of inline tables; the stdlib has tomllib for reading but no
# writer, and we don't want to take on tomli_w as a dependency for the
# narrow set of types this module actually emits).
# --------------------------------------------------------------------------- #


def _dump_toml(d: dict[str, Any]) -> str:
    """Render ``d`` as a TOML document.

    Layout:
      * Top-level scalars are emitted first.
      * Then any top-level tables (e.g. ``[general]``).
      * Then any top-level arrays-of-tables (e.g. ``[[datasets]]``).
        Each entry's ``subsets`` array (if present) becomes its own
        ``[[datasets.subsets]]`` block.

    Missing on purpose: TOML date types, deeply-nested arrays-of-arrays
    of arbitrary tables, multi-line strings. We don't need any of those
    for an anima_lora config.
    """
    lines: list[str] = []
    lines.append(
        "# LoraHub-generated anima_lora config — pre-merged base/preset/method.",
    )
    lines.append(
        "# Regenerated on every launch; hand-edits are wiped at next compile.",
    )
    lines.append("")

    scalars: list[tuple[str, Any]] = []
    tables: list[tuple[str, dict[str, Any]]] = []
    arrays_of_tables: list[tuple[str, list[dict[str, Any]]]] = []
    for key, value in d.items():
        if isinstance(value, dict):
            tables.append((key, value))
        elif (
            isinstance(value, list)
            and value
            and all(isinstance(x, dict) for x in value)
        ):
            arrays_of_tables.append((key, value))
        else:
            scalars.append((key, value))

    for key, value in scalars:
        lines.append(f"{key} = {_toml_value(value)}")

    for key, table in tables:
        lines.append("")
        lines.append(f"[{key}]")
        for sub_key, sub_value in table.items():
            lines.append(f"{sub_key} = {_toml_value(sub_value)}")

    for key, entries in arrays_of_tables:
        for entry in entries:
            lines.append("")
            lines.append(f"[[{key}]]")
            nested_arrays: list[tuple[str, list[dict[str, Any]]]] = []
            for sub_key, sub_value in entry.items():
                if (
                    isinstance(sub_value, list)
                    and sub_value
                    and all(isinstance(x, dict) for x in sub_value)
                ):
                    nested_arrays.append((sub_key, sub_value))
                else:
                    lines.append(f"{sub_key} = {_toml_value(sub_value)}")
            for sub_key, sub_entries in nested_arrays:
                for sub_entry in sub_entries:
                    lines.append("")
                    lines.append(f"  [[{key}.{sub_key}]]")
                    for k2, v2 in sub_entry.items():
                        lines.append(f"  {k2} = {_toml_value(v2)}")

    return "\n".join(lines) + "\n"


def _toml_value(v: Any) -> str:
    """Render a single TOML-native value (scalar or homogeneous list)."""
    if v is True:
        return "true"
    if v is False:
        return "false"
    if isinstance(v, str):
        # Escape backslashes + double quotes; we never need multi-line.
        escaped = v.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(v, int) and not isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return _fmt_float(v)
    if isinstance(v, list):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    if v is None:
        # Should never happen — _render_full_config skips None values.
        msg = f"cannot serialise None to TOML (key dropped upstream?)"
        raise CompilationError(msg)
    msg = f"unsupported TOML value type: {type(v).__name__} = {v!r}"
    raise CompilationError(msg)


def _enforce_compile_constraints(opts: AnimaLoraOptions) -> None:
    """Reject combos anima_lora's ``train.py`` asserts against at startup.

    Catching these at compile time turns an opaque mid-launch crash
    into a structured error the UI can render before the user waits
    through model load + dataset cache.

    Constraints enforced:

      * ``compile_mode='full'`` is incompatible with grad
        checkpointing / unsloth offload / ``blocks_to_swap > 0``
        (upstream's CLAUDE.md).
      * ``blocks_to_swap > 0`` is incompatible with
        ``cpu_offload_checkpointing=true``
        (``train.py:326`` AssertionError).
    """
    bad: list[str] = []
    if opts.compile_mode == "full":
        if opts.gradient_checkpointing:
            bad.append("gradient_checkpointing=true")
        if opts.unsloth_offload_checkpointing:
            bad.append("unsloth_offload_checkpointing=true")
        if opts.blocks_to_swap > 0:
            bad.append(f"blocks_to_swap={opts.blocks_to_swap}")
        if bad:
            msg = (
                f"compile_mode='full' is incompatible with: {', '.join(bad)}. "
                "Either drop compile_mode (set to None or 'blocks') or disable "
                "the conflicting offload knobs."
            )
            raise CompilationError(msg)

    if opts.blocks_to_swap > 0 and opts.cpu_offload_checkpointing:
        msg = (
            f"blocks_to_swap={opts.blocks_to_swap} is incompatible with "
            "cpu_offload_checkpointing=true (anima_lora train.py:326). "
            "Pick one: keep blocks_to_swap (the bigger memory win) and "
            "set cpu_offload_checkpointing=false, or vice versa. "
            "unsloth_offload_checkpointing composes with blocks_to_swap."
        )
        raise CompilationError(msg)


def _warn_locked_fields_changed(opts: AnimaLoraOptions) -> None:
    """Log a warning for every locked field the user moved off its default.

    LoraHub exposes every ``base.toml`` field for editor completeness, but
    upstream's argparse can't actually flip several of them off (no
    reverse ``--no_<x>`` flag, or the value is hard-baked into the
    static-shape compile path). When the user changes a locked-True
    field to False, or a locked-value field to a non-default, we emit a
    warning so the operator notices instead of being silently ignored
    by the trainer. See ``LOCKED_FIELDS`` for the per-field rationale.
    """
    # Defaults match the schema constructor's annotations. Hardcoded so
    # the warning is honest about what "the default" means even if a
    # caller passes ``opts`` from a non-validated source.
    locked_defaults: dict[str, object] = {
        "masked_loss": True,
        "torch_compile": True,
        "skip_cache_check": True,
        "dataloader_pin_memory": True,
        "enable_bucket": True,
        "cache_latents": True,
        "cache_latents_to_disk": True,
        "cache_text_encoder_outputs": True,
        "cache_text_encoder_outputs_to_disk": True,
        "cache_llm_adapter_outputs": True,
        "static_token_count": 4096,
        "vae_chunk_size": 64,
        "caption_extension": ".txt",
        "save_model_as": "safetensors",
    }
    for field, default in locked_defaults.items():
        if not hasattr(opts, field):
            continue
        actual = getattr(opts, field)
        # ``static_token_count`` is intentionally None on the
        # native-flatten path (mutually exclusive with the static-pad
        # 4096 path), so don't warn when the user opted into that.
        if (
            field == "static_token_count"
            and getattr(opts, "enable_native_flatten", False)
        ):
            continue
        if actual != default:
            meta = LOCKED_FIELDS.get(field, {})
            kind = meta.get("kind", "locked")
            reason = meta.get("reason", "upstream cannot change this field")
            _log.warning(
                "anima_lora: %s set to %r (default %r, %s) — %s",
                field,
                actual,
                default,
                kind,
                reason,
            )


def _fmt_float(v: float) -> str:
    """Format a float for argparse without scientific notation surprises.

    argparse handles ``5e-05`` fine, but readability of ``--learning_rate
    5e-05`` in logs is worse than ``--learning_rate 5e-05`` vs
    ``--learning_rate 0.00005``. We pick ``repr()``-style which keeps
    short-form for round numbers and falls back to e-notation for very
    small values — same shape upstream uses in their own argparse
    examples.
    """
    return repr(v)


def compile_turbo_config(
    cfg: TrainingConfig,
    workspace: Path,
) -> tuple[list[str], dict[Path, str]]:
    """Translate a TurboConfig into ``scripts/distill_turbo.py`` argv.

    Distinct from :func:`compile_config` because turbo distillation
    runs a **bespoke** trainer (Liu et al. arXiv:2511.22677 Decoupled-
    Hybrid DMD2), not ``train.py``. No accelerate launcher, no
    method/preset merge chain — every value flows through the script's
    own argparse.

    Returns the same ``(argv, files_to_write)`` shape as the regular
    compiler so the backend.launch dispatch can swap runners cleanly.
    ``files_to_write`` is always empty: distill_turbo.py owns its own
    ``configs/methods/turbo.toml`` defaults; we only override via CLI
    flags. The first argv element is **not** ``--config`` because
    we'd rather have every value explicit at the CLI than rely on
    upstream's TOML defaults that might drift.
    """
    if cfg.backend.type != "anima_lora":
        msg = (
            f"anima_lora compiler invoked on backend.type={cfg.backend.type!r}; "
            "this is a programming error in the dispatch path"
        )
        raise CompilationError(msg)
    if cfg.backend.anima_lora is None or cfg.backend.anima_lora.turbo is None:
        msg = (
            "compile_turbo_config requires backend.animaLora.turbo to be set; "
            "use compile_config for the standard training path"
        )
        raise CompilationError(msg)

    opts = cfg.backend.anima_lora
    turbo = opts.turbo

    workspace = workspace.resolve()
    output_dir = workspace / "ckpt"
    log_dir = workspace / "logs" / "turbo"

    argv: list[str] = []

    # ---- Top-level scalars ----
    bm = cfg.base_model
    if bm.checkpoint:
        argv += ["--dit_path", str(bm.checkpoint)]
    # Upstream's `data_dir` points at the LoRA cache folder
    # (post_image_dataset/lora). With LoraHub's auto-preprocess flow,
    # the cache is written under the workspace at
    # ``<workspace>/post_image_dataset/lora`` regardless of where
    # ``cfg.dataset.source`` (the raw images) lives. cfg.dataset.source
    # itself stays the user-facing raw image dir for shape parity with
    # kohya / dp recipes.
    argv += [
        "--data_dir",
        str((workspace / "post_image_dataset" / "lora").resolve()),
    ]
    argv += ["--output_dir", str(output_dir)]
    argv += ["--output_name", opts.output_name]
    argv += ["--iterations", str(turbo.iterations)]
    argv += ["--batch_size", str(turbo.batch_size)]
    argv += ["--seed", str(turbo.seed)]
    if turbo.use_custom_down_autograd:
        argv += ["--use_custom_down_autograd"]
    else:
        argv += ["--no_use_custom_down_autograd"]

    # ---- Network ----
    argv += ["--student_rank", str(turbo.student_rank)]
    argv += ["--fake_rank", str(turbo.fake_rank)]
    # student_alpha / fake_alpha aren't standalone CLI flags upstream
    # (see distill_turbo.py:140-189) — they're TOML-only. The script
    # reads them through `pick("network.student_alpha", default=rank)`
    # so passing them as a [network] override would need a config TOML
    # write. For cut4 we skip the override path: defaulting alpha=rank
    # matches our schema defaults (48/48, 64/64). Future cut4.B can
    # emit a turbo override TOML if users start needing alpha != rank.

    # ---- Optimization ----
    argv += ["--student_lr", _fmt_float(turbo.student_lr)]
    argv += ["--fake_lr", _fmt_float(turbo.fake_lr)]
    argv += ["--fake_steps_per_student_step", str(turbo.fake_steps_per_student_step)]
    argv += ["--alpha_warmup_steps", str(turbo.alpha_warmup_steps)]
    # `--alpha` overrides dmd.teacher_cfg per the script's argparse help.
    argv += ["--alpha", _fmt_float(turbo.teacher_cfg)]

    # ---- DMD schedule ----
    argv += ["--student_steps", str(turbo.student_steps)]

    # ---- Memory / kernels ----
    argv += ["--blocks_to_swap", str(0)]  # turbo defaults to 0; no schema knob yet
    argv += ["--attn_mode", turbo.attn_mode]

    # ---- I/O cadence ----
    argv += ["--save_every", str(turbo.save_every)]
    argv += ["--log_interval", str(turbo.log_interval)]
    argv += ["--log_dir", str(log_dir)]

    files: dict[Path, str] = {}
    return argv, files
