"""Compile a TrainingConfig into anima_lora CLI argv.

anima_lora is config-driven via a four-layer chain:

    base.toml -> presets.toml[<preset>] -> methods/<method>.toml -> CLI

Upstream owns the first three layers (vendored under `external/anima_lora/`).
LoraHub contributes only the CLI override layer — translating
:class:`AnimaLoraOptions` into ``--key value`` pairs that ``train.py``
re-parses on top of the merged TOML chain.

This module is a pure function: callers pass in the recipe and a
workspace, and get back ``(argv, files_to_write)``. We do not emit a
TOML file — keeping the upstream merge chain intact is the whole
point of the vendored layout. The `files_to_write` dict is always
empty; the contract matches kohya / dp compiler return shape so the
launcher dispatch is uniform.

Method routing: the value of ``opts.method`` becomes ``--method <X>``
plus the matching sub-config's fields are mapped to upstream's CLI
flag names. Sub-configs whose method isn't selected are silently
ignored (callers might keep them around for fast switching).

Compile-mode constraint: ``compile_mode = "full"`` is incompatible
with ``gradient_checkpointing`` and ``blocks_to_swap > 0`` per
upstream `CLAUDE.md` ("compile_mode = 'full' is incompatible with
gradient_checkpointing / blocks_to_swap"). We catch this at compile
time so the user gets a clear error before launch instead of an
opaque torch.compile traceback.
"""

from __future__ import annotations

import logging
from pathlib import Path

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


class CompilationError(ValueError):
    """Raised when an AnimaLoraOptions config can't be compiled."""


def compile_config(
    cfg: TrainingConfig,
    workspace: Path,
) -> tuple[list[str], dict[Path, str]]:
    """Translate a recipe into ``(argv, files_to_write)`` for anima_lora.

    ``argv`` is what to append after ``python <repo>/train.py`` (or after
    ``accelerate launch`` if cut2 wraps it). ``files_to_write`` is always
    empty — upstream owns its own config layout under
    ``external/anima_lora/configs/`` and we don't emit anything.
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

    argv: list[str] = []
    # Layer 1: method + preset selection. anima_lora reads these first
    # to drive its TOML merge chain before any CLI override applies.
    argv += ["--method", opts.method, "--preset", opts.preset]

    # Layer 2: shared overrides (apply to every method).
    argv += _shared_overrides(cfg, opts, output_dir)

    # Layer 3: dataset path overrides — pin source / resized / cache to
    # absolute paths under the LoRaHub workspace so anima_lora's own
    # ``base.toml`` defaults (which are relative to the vendored repo
    # root) are bypassed. ``cfg.dataset.source`` is the user-facing
    # raw image dir (same shape kohya / dp use), and the resized + cache
    # dirs are LoRaHub-managed under the workspace. This is delivered
    # as a generated ``--dataset_config <path>`` TOML because train.py
    # has no CLI flag for the three path keys (they live as
    # ``configs/base.toml`` top-level scalars, not argparse).
    ds_argv, ds_files = _dataset_config_override(cfg, workspace)
    argv += ds_argv

    # Layer 4: method-specific sub-config overrides.
    argv += _method_overrides(opts)

    # Files: a single dataset_config TOML that pins the three data
    # paths. Written under the workspace by ``backend.launch`` before
    # spawning train.py.
    files: dict[Path, str] = dict(ds_files)
    return argv, files


def _enforce_compile_constraints(opts: AnimaLoraOptions) -> None:
    """``compile_mode = 'full'`` is incompatible with checkpointing / swap.

    Documented in upstream's `CLAUDE.md`. Raise at compile time so the
    user sees the conflict instead of an opaque torch.compile error.
    """
    if opts.compile_mode != "full":
        return
    bad: list[str] = []
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


def _shared_overrides(
    cfg: TrainingConfig,
    opts: AnimaLoraOptions,
    output_dir: Path,
) -> list[str]:
    """Method-agnostic CLI overrides — applied for every method.

    Upstream's flag names are snake_case (matches the TOML keys); the
    AnimaLoraOptions schema is also snake_case at the python level so
    most fields map 1:1.
    """
    out: list[str] = []
    # ---- Output ----
    out += ["--output_dir", str(output_dir)]
    out += ["--output_name", opts.output_name]

    # ---- Model paths (from BaseModelConfig — anima_lora reads its own
    # base.toml model paths but we may override per-recipe) ----
    bm = cfg.base_model
    if bm.checkpoint:
        out += ["--pretrained_model_name_or_path", str(bm.checkpoint)]
    paths = bm.arch_paths
    if paths.qwen3 is not None:
        out += ["--qwen3", str(paths.qwen3)]
    if paths.ae is not None:
        # anima_lora calls the VAE flag --vae upstream
        out += ["--vae", str(paths.ae)]

    # ---- Network ----
    out += ["--network_module", opts.network_module]
    out += ["--network_dim", str(opts.network_dim)]
    out += ["--network_alpha", str(opts.network_alpha)]
    if opts.network_train_unet_only:
        out += ["--network_train_unet_only"]

    # ---- Optim / schedule ----
    out += ["--optimizer_type", opts.optimizer_type]
    out += ["--lr_scheduler", opts.lr_scheduler]
    out += ["--learning_rate", _fmt_float(opts.learning_rate)]
    out += ["--max_train_epochs", str(opts.max_train_epochs)]
    # ``cfg.schedule.max_steps`` is the user-facing "训练总步数" override
    # — the form widgets all read from there. anima_lora's train.py
    # honours ``--max_train_steps`` (sd-scripts inheritance), and when
    # both flags are set the trainer stops at whichever comes first.
    # We forward the schedule cap when present so the UI's number
    # actually drives training.
    if cfg.schedule.max_steps is not None and cfg.schedule.max_steps > 0:
        out += ["--max_train_steps", str(int(cfg.schedule.max_steps))]
    out += ["--save_every_n_epochs", str(opts.save_every_n_epochs)]
    if opts.save_every_n_steps is not None and opts.save_every_n_steps > 0:
        out += ["--save_every_n_steps", str(int(opts.save_every_n_steps))]
    out += ["--checkpointing_epochs", str(opts.checkpointing_epochs)]
    if opts.caption_dropout_rate > 0:
        out += ["--caption_dropout_rate", _fmt_float(opts.caption_dropout_rate)]

    # ---- Sampling / loss (flow-matching) ----
    out += ["--timestep_sampling", opts.timestep_sampling]
    out += ["--sigmoid_scale", _fmt_float(opts.sigmoid_scale)]
    out += ["--discrete_flow_shift", _fmt_float(opts.discrete_flow_shift)]
    if opts.weighting_scheme is not None:
        out += ["--weighting_scheme", opts.weighting_scheme]
    if opts.logit_mean is not None:
        out += ["--logit_mean", _fmt_float(opts.logit_mean)]
    if opts.logit_std is not None:
        out += ["--logit_std", _fmt_float(opts.logit_std)]
    if opts.mode_scale is not None:
        out += ["--mode_scale", _fmt_float(opts.mode_scale)]
    if opts.vr_loss_weight is not None:
        out += ["--vr_loss_weight", _fmt_float(opts.vr_loss_weight)]

    # ---- Caching / data ----
    if opts.cache_latents:
        out += ["--cache_latents"]
    if opts.cache_latents_to_disk:
        out += ["--cache_latents_to_disk"]
    if opts.cache_text_encoder_outputs:
        out += ["--cache_text_encoder_outputs"]
    if opts.cache_text_encoder_outputs_to_disk:
        out += ["--cache_text_encoder_outputs_to_disk"]
    if opts.cache_llm_adapter_outputs:
        out += ["--cache_llm_adapter_outputs"]
    if opts.use_shuffled_caption_variants:
        out += ["--use_shuffled_caption_variants"]
    if opts.sample_ratio is not None:
        out += ["--sample_ratio", _fmt_float(opts.sample_ratio)]
    out += ["--static_token_count", str(opts.static_token_count)]
    out += ["--vae_chunk_size", str(opts.vae_chunk_size)]
    if opts.vae_disable_cache:
        out += ["--vae_disable_cache"]
    if opts.no_half_vae:
        out += ["--no_half_vae"]

    # ---- Attention / compile ----
    out += ["--attn_mode", opts.attn_mode]
    if opts.xformers:
        out += ["--xformers"]
    if opts.split_attn:
        out += ["--split_attn"]
    if opts.compile_mode is not None:
        out += ["--compile_mode", opts.compile_mode]
    if opts.compile_inductor_mode is not None:
        out += ["--compile_inductor_mode", opts.compile_inductor_mode]
    if opts.use_custom_down_autograd:
        # Upstream consumes this as a network kwarg, not an argparse flag.
        # See ``networks/lora_anima/factory.py`` line 120 — the value
        # is read off ``--network_args`` and dispatched onto each LoRA
        # module's ``use_custom_down_autograd`` attribute. train.py's
        # argparse only knows ``--network_args``; emitting
        # ``--use_custom_down_autograd`` directly trips
        # "unrecognized arguments".
        out += ["--network_args", "use_custom_down_autograd=true"]

    # ---- Memory / offload ----
    if opts.blocks_to_swap > 0:
        out += ["--blocks_to_swap", str(opts.blocks_to_swap)]
    if opts.gradient_checkpointing:
        out += ["--gradient_checkpointing"]
    if opts.unsloth_offload_checkpointing:
        out += ["--unsloth_offload_checkpointing"]
    if opts.cpu_offload_checkpointing:
        out += ["--cpu_offload_checkpointing"]
    out += ["--mixed_precision", opts.mixed_precision]

    # ---- Validation ----
    if opts.use_cmmd:
        out += ["--use_cmmd"]
    if opts.validation_seed is not None:
        out += ["--validation_seed", str(opts.validation_seed)]
    if opts.validation_sample_steps is not None:
        out += ["--validation_sample_steps", str(opts.validation_sample_steps)]
    if opts.validation_cfg_scale is not None:
        out += ["--validation_cfg_scale", _fmt_float(opts.validation_cfg_scale)]

    # ---- Upstream-locked / risky fields (B5 cut-locks) ----
    # Most of these are store_true and base.toml already pins them on,
    # so we emit the flag whenever opts.* is True. When the user sets
    # a locked-True field to False the emit is skipped — the compiler
    # also logs a warning above so the operator notices the no-op.
    if opts.masked_loss:
        out += ["--masked_loss"]
    if opts.torch_compile:
        out += ["--torch_compile"]
    if opts.skip_cache_check:
        out += ["--skip_cache_check"]
    if opts.dataloader_pin_memory:
        out += ["--dataloader_pin_memory"]
    if opts.persistent_data_loader_workers:
        out += ["--persistent_data_loader_workers"]
    if opts.trim_crossattn_kv:
        out += ["--trim_crossattn_kv"]
    out += ["--save_model_as", opts.save_model_as]
    out += ["--save_precision", opts.save_precision]
    out += ["--log_every_n_steps", str(opts.log_every_n_steps)]
    # keep_tokens / caption_extension / validation_split_num /
    # enable_bucket / path_pattern live in the dataset blueprint, not
    # the argparse namespace. We surface them in the LoraHub schema for
    # editor-side warnings; the actual dataset_config TOML is whatever
    # base.toml ships with. cut B5 follow-up will materialise a
    # per-recipe dataset.toml override if a user actually changes one.

    # ---- Seed ----
    seed = cfg.schedule.seed if cfg.schedule.seed is not None else 42
    out += ["--seed", str(seed)]

    # ---- Resume / state writing ----
    # Mirror kohya's behaviour: write optimizer/scheduler state next to
    # checkpoints so a later /resume can re-attach. Without this, a
    # cancelled run can resume the LoRA weights but loses the optimizer
    # momentum + lr schedule position.
    if cfg.resume.save_state:
        out += ["--save_state"]
    if cfg.resume.save_state_at_end:
        out += ["--save_state_on_train_end"]
    if cfg.resume.save_last_n_epochs_state is not None:
        out += [
            "--save_last_n_epochs_state",
            str(cfg.resume.save_last_n_epochs_state),
        ]
    if cfg.resume.save_last_n_steps_state is not None:
        out += [
            "--save_last_n_steps_state",
            str(cfg.resume.save_last_n_steps_state),
        ]

    return out


def _dataset_config_override(
    cfg: TrainingConfig,
    workspace: Path,
) -> tuple[list[str], dict[Path, str]]:
    """Pin dataset paths via a generated ``--dataset_config`` TOML.

    Upstream's CLI doesn't expose the three relevant path keys
    (``source_image_dir`` / ``resized_image_dir`` / ``lora_cache_dir``)
    as argparse flags — they live as top-level scalars in
    ``configs/base.toml`` and feed the dataset blueprint via
    ``{...}`` template substitution. Trying to emit them as
    ``--source_image_dir <path>`` etc. trips
    "unrecognized arguments" against ``train.py``.

    The clean injection point is ``--dataset_config <path>``, which
    train.py honours by loading the supplied TOML and skipping the
    base blueprint entirely. We materialise a minimal blueprint
    pointing the resized + cache dirs at the LoraHub workspace's
    ``post_image_dataset/`` (where the auto-preprocess step writes),
    and the source image dir at ``cfg.dataset.source`` — same shape
    kohya / dp use across the rest of LoraHub.

    Returns ``(argv, files)`` so the caller can fold both into the
    final compile result; ``files`` is a single ``{path: content}``
    pair that ``backend.launch`` writes to disk before spawning
    ``train.py``.
    """
    opts = cfg.backend.anima_lora
    assert opts is not None  # narrowed by compile_config

    # Match upstream's default blueprint shape exactly — same keys,
    # same nesting. Only the path fields are LoraHub-specific.
    src = cfg.dataset.source.resolve()
    resized = (workspace / "post_image_dataset" / "resized").resolve()
    cache = (workspace / "post_image_dataset" / "lora").resolve()

    # Quote paths defensively. anima_lora's TOML parser uses tomllib
    # (PEP 680) which accepts double-quoted strings with a fixed
    # escape table; backslashes (Windows) and embedded double-quotes
    # both need escaping.
    def _q(p: Path | str) -> str:
        s = str(p).replace("\\", "\\\\").replace('"', '\\"')
        return f'"{s}"'

    # Resolution: take the first dim of cfg.dataset.resolution.
    # anima_lora's blueprint uses a single int (square) when given
    # one number, or "[H, W]" pair when given two.
    res = cfg.dataset.resolution
    if isinstance(res, (list, tuple)) and len(res) == 2:
        res_token = f"[{int(res[0])}, {int(res[1])}]"
    elif isinstance(res, (list, tuple)) and len(res) == 1:
        res_token = str(int(res[0]))
    else:
        res_token = "1024"

    batch_size = max(1, int(cfg.schedule.batch_size or 1))
    keep_tokens = int(opts.keep_tokens)
    caption_ext = (opts.caption_extension or ".txt").strip() or ".txt"
    # num_repeats: same field kohya / dp read from ``cfg.dataset.num_repeats``
    # (each image is sampled this many times per epoch). Write it into the
    # generated dataset blueprint so anima_lora respects the same knob the
    # rest of LoRaHub exposes; defaults to 1 when the recipe doesn't set it.
    num_repeats = max(1, int(getattr(cfg.dataset, "num_repeats", 1) or 1))

    body = (
        "# LoRaHub-generated dataset_config — pins anima_lora's three\n"
        "# data paths (source / resized / cache) at LoRaHub-managed\n"
        "# absolute locations. Regenerated on every launch.\n"
        "[general]\n"
        f"caption_extension = {_q(caption_ext)}\n"
        f"keep_tokens = {keep_tokens}\n"
        "\n"
        "[[datasets]]\n"
        f"resolution = {res_token}\n"
        f"batch_size = {batch_size}\n"
        "enable_bucket = true\n"
        "\n"
        "  [[datasets.subsets]]\n"
        f"  image_dir = {_q(resized)}\n"
        f"  cache_dir = {_q(cache)}\n"
        f"  num_repeats = {num_repeats}\n"
    )

    # Tag with a sub-folder so multiple parallel jobs writing to
    # different workspaces don't collide on filename.
    target = workspace / "_lorahub_anima_dataset.toml"

    # We additionally export the three path keys via env-side
    # config so the auto-preprocess step (which we already write to
    # the workspace, see preprocess.py) and train.py see the same
    # values. The argv side is just the --dataset_config pointer.
    argv = ["--dataset_config", str(target)]
    files = {target: body}
    # Stash source path as an env hint for any downstream tooling
    # (e.g. the GUI) that wants to know which raw dir backed this
    # generated blueprint. Not consumed by train.py itself.
    _ = src
    return argv, files


def _method_overrides(opts: AnimaLoraOptions) -> list[str]:
    """Method-specific CLI overrides — only the selected method's sub-config.

    The other sub-configs are intentionally ignored (we keep them in the
    schema so the user can flip ``opts.method`` between switches without
    losing per-method tuning).

    Most flag names match upstream's TOML keys 1:1 — see
    ``library/training/cli_args.py`` in the vendored copy.
    """
    if opts.method == "lora":
        return _lora_overrides(opts)
    if opts.method == "postfix":
        return _postfix_overrides(opts)
    if opts.method == "chimera":
        return _chimera_overrides(opts)
    if opts.method == "easycontrol":
        return _easycontrol_overrides(opts)
    if opts.method == "ip_adapter":
        return _ip_adapter_overrides(opts)
    msg = f"unhandled method {opts.method!r} (schema enum drift?)"
    raise CompilationError(msg)


def _network_args(*pairs: str) -> list[str]:
    """Format ``key=value`` pieces as repeated ``--network_args`` argv.

    train.py only exposes ``--network_args [NETWORK_ARGS ...]`` (nargs="*");
    each adapter family reads its own kwargs out of that bag (see
    ``networks/lora_anima/factory.py:120`` and the ``kwargs.get(...)``
    sites under ``networks/methods/``). Emitting ``--use_ortho`` or
    ``--b_cond_init`` etc. directly trips "unrecognized arguments" —
    upstream's argparse never declared them as flags.
    """
    out: list[str] = []
    for piece in pairs:
        out += ["--network_args", piece]
    return out


def _lora_overrides(opts: AnimaLoraOptions) -> list[str]:
    """LoRA / OrthoLoRA / T-LoRA stack — the default anima_lora behaviour.

    All four knobs feed ``networks/lora_anima/config.py``'s ``LoRAConfig.from_kwargs``
    via ``--network_args`` k=v pairs; none of them is an argparse flag.
    """
    sub = opts.lora
    pieces: list[str] = [
        f"use_ortho={'true' if sub.use_ortho else 'false'}",
        f"use_timestep_mask={'true' if sub.use_timestep_mask else 'false'}",
        f"min_rank={sub.min_rank}",
        f"alpha_rank_scale={_fmt_float(sub.alpha_rank_scale)}",
    ]
    return _network_args(*pieces)


def _postfix_overrides(opts: AnimaLoraOptions) -> list[str]:
    """Postfix tuning — see ``networks/methods/postfix.py``."""
    sub = opts.postfix
    if sub is None:  # validated upstream by AnimaLoraOptions model_validator
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
    return _network_args(*pieces)


def _chimera_overrides(opts: AnimaLoraOptions) -> list[str]:
    """ChimeraHydra dual-pool MoE — pinned router knobs.

    All keys (use_chimera_hydra / balance_* / fei_feature_dim /
    sigma_feature_dim) are read out of ``kwargs`` in
    ``networks/lora_anima/config.py``'s ``LoRAConfig.from_kwargs`` and
    ``networks/__init__.py::_parse_bool_flag``; emit them through
    ``--network_args``.
    """
    sub = opts.chimera
    if sub is None:
        msg = "method='chimera' missing sub-config"
        raise CompilationError(msg)
    pieces = [
        "use_chimera_hydra=true",
        f"balance_w_content={_fmt_float(sub.balance_w_content)}",
        f"balance_w_freq={_fmt_float(sub.balance_w_freq)}",
        f"balance_loss_warmup_ratio={_fmt_float(sub.balance_loss_warmup_ratio)}",
        f"fei_feature_dim={sub.fei_feature_dim}",
        f"sigma_feature_dim={sub.sigma_feature_dim}",
    ]
    return _network_args(*pieces)


def _easycontrol_overrides(opts: AnimaLoraOptions) -> list[str]:
    """EasyControl per-block conditioning LoRA + softmax gate.

    ``--use_easycontrol`` / ``--easycontrol_drop_p`` /
    ``--easycontrol_cond_noise_max`` ARE real argparse flags (see
    ``library/anima/training.py``). The gate / scaling knobs live in
    ``networks/methods/easycontrol.py:make_easycontrol_network`` which
    reads them from kwargs, so they must go through ``--network_args``.
    """
    sub = opts.easycontrol
    if sub is None:
        msg = "method='easycontrol' missing sub-config"
        raise CompilationError(msg)
    out: list[str] = ["--use_easycontrol"]
    out += ["--easycontrol_drop_p", _fmt_float(sub.drop_p)]
    out += ["--easycontrol_cond_noise_max", _fmt_float(sub.cond_noise_max)]
    pieces = [
        f"b_cond_init={_fmt_float(sub.b_cond_init)}",
        f"cond_scale={_fmt_float(sub.cond_scale)}",
        f"apply_ffn_lora={'1' if sub.apply_ffn_lora else '0'}",
        f"cond_token_count={sub.cond_token_count}",
    ]
    out += _network_args(*pieces)
    return out


def _ip_adapter_overrides(opts: AnimaLoraOptions) -> list[str]:
    """IP-Adapter — PE-Core encoder + resampler + per-block KV.

    ``--use_ip_adapter`` / ``--ip_encoder`` / ``--ip_image_drop_p`` /
    ``--ip_features_cache_to_disk`` ARE argparse flags. The resampler
    sizing + IP scale + gate LR live in the network factory's kwargs.
    """
    sub = opts.ip_adapter
    if sub is None:
        msg = "method='ip_adapter' missing sub-config"
        raise CompilationError(msg)
    out: list[str] = ["--use_ip_adapter"]
    out += ["--ip_encoder", sub.encoder]
    out += ["--ip_image_drop_p", _fmt_float(sub.image_drop_p)]
    if sub.features_cache_to_disk:
        out += ["--ip_features_cache_to_disk"]
    pieces = [
        f"ip_resampler_layers={sub.resampler_layers}",
        f"ip_resampler_heads={sub.resampler_heads}",
        f"ip_scale={_fmt_float(sub.ip_scale)}",
        f"gate_lr={_fmt_float(sub.gate_lr)}",
    ]
    out += _network_args(*pieces)
    return out


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
