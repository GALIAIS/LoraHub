"""anima_lora backend specific configs (methods + turbo + options)."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, Field, model_validator

from .._shared import _CAMEL_CONFIG


class AnimaLoraMethodLoraConfig(BaseModel):
    """Defaults for ``method = "lora"`` on the anima_lora backend.

    The ``method = "lora"`` path on anima_lora is *not* the bare LoRA you'd
    get on kohya — upstream stacks OrthoLoRA + T-LoRA on top by default.
    Tracking those knobs here keeps the LoraHub schema explicit about the
    stack instead of hiding it in the compiler.

    Algorithm selection: prefer ``algorithm`` (string enum) — it's the
    single authoritative knob. The legacy ``use_X`` booleans are kept
    for back-compat and resolved into ``algorithm`` by ``_normalise``;
    setting both an inconsistent enum + bool pair is rejected.
    """

    model_config = _CAMEL_CONFIG

    # Authoritative algorithm selector. Default ``ortho`` preserves
    # anima's upstream behaviour (OrthoLoRA stack on top of LoRA);
    # set ``algorithm="lora"`` for the bare-LoRA baseline. Enum values
    # match the keys in
    # ``external/anima_lora/networks/__init__.py::NETWORK_REGISTRY``.
    algorithm: Literal[
        "lora",
        "ortho",
        "dora",
        "ia3",
        "lokr",
        "loha",
        "dylora",
        "full",
        "diag_oft",
        "boft",
        "glora",
        "vera",
    ] = "ortho"

    # ---- Legacy boolean toggles (deprecated) ----
    # Kept as optional shadows of ``algorithm`` for back-compat with
    # YAML / API consumers written before the enum landed. ``None``
    # means "user didn't touch this knob, use the enum"; ``True`` /
    # ``False`` is reconciled against the enum in ``_normalise`` and
    # an inconsistency raises.
    use_ortho: bool | None = None
    use_dora: bool | None = None
    use_ia3: bool | None = None
    use_lokr: bool | None = None
    use_loha: bool | None = None
    use_dylora: bool | None = None
    use_full: bool | None = None
    use_diag_oft: bool | None = None
    use_boft: bool | None = None
    use_glora: bool | None = None
    use_vera: bool | None = None

    # Algorithm-specific knobs (only consulted by the matching
    # algorithm; ignored otherwise).
    lokr_factor: int = Field(8, ge=1)
    boft_factors: int = Field(4, ge=1)

    # T-LoRA timestep mask: high noise → low rank, low noise → full rank.
    # Composes with any LoRA-leg algorithm; a no-op for atomic variants
    # (ia3 / lokr / loha / full / diag_oft / boft / glora / vera) but
    # keeping the field on universally simplifies the front-end.
    use_timestep_mask: bool = True
    min_rank: int = Field(16, ge=1)
    alpha_rank_scale: float = Field(1.0, gt=0)

    # Mapping legacy ``use_X`` booleans → algorithm enum value. Ordered
    # so the first True one wins when two are accidentally co-enabled
    # (the validator reports the conflict explicitly anyway).
    _BOOL_TO_ALGORITHM: ClassVar[tuple[tuple[str, str], ...]] = (
        ("use_dora", "dora"),
        ("use_ia3", "ia3"),
        ("use_lokr", "lokr"),
        ("use_loha", "loha"),
        ("use_dylora", "dylora"),
        ("use_full", "full"),
        ("use_diag_oft", "diag_oft"),
        ("use_boft", "boft"),
        ("use_glora", "glora"),
        ("use_vera", "vera"),
        # ``use_ortho`` last so DoRA / atomic / etc. take priority over
        # the ortho stack default — the bool reads "stack OrthoLoRA on
        # top" and pre-enum YAML wrote ``use_ortho=true`` as the default.
        ("use_ortho", "ortho"),
    )

    @model_validator(mode="after")
    def _normalise(self) -> AnimaLoraMethodLoraConfig:
        """Reconcile ``algorithm`` with the legacy ``use_X`` shadows.

        Three cases:
          1. User set only ``algorithm``: leave ``use_X`` shadows as
             ``None`` (front-end and compiler both read enum first).
          2. User set only ``use_X`` toggles: derive ``algorithm`` from
             the first True bool, leaving the explicit ``use_X = False``
             shadows in place so a round-trip preserves intent.
          3. User set both: reject if they disagree; otherwise pass.

        ``use_ortho=True`` + a non-LoRA-leg algorithm (e.g. dora) is
        rejected too — these don't compose for save-layout reasons.
        """
        # Collect explicitly-True legacy bools.
        explicit_true = [
            field for field, _ in self._BOOL_TO_ALGORITHM
            if getattr(self, field) is True
        ]
        if len(explicit_true) > 1:
            raise ValueError(
                f"AnimaLoraMethodLoraConfig: multiple legacy use_X toggles set "
                f"to True ({explicit_true}); set at most one, or use the "
                f"``algorithm`` enum instead."
            )

        # Did the user explicitly set ``algorithm``? If not, the legacy
        # bool wins (back-compat: callers that pre-date the enum still
        # work). Pydantic v2 exposes ``model_fields_set`` (via the
        # internal ``__pydantic_fields_set__``) to disambiguate.
        algorithm_explicit = "algorithm" in self.__pydantic_fields_set__

        if explicit_true:
            field = explicit_true[0]
            mapped = dict(self._BOOL_TO_ALGORITHM)[field]
            if algorithm_explicit and self.algorithm != mapped:
                # ``use_ortho=True`` paired with another algorithm is the
                # most likely "user added DoRA but forgot to drop the
                # legacy ortho default" path; help them fix it.
                if field == "use_ortho":
                    raise ValueError(
                        f"AnimaLoraMethodLoraConfig: use_ortho=True with "
                        f"algorithm={self.algorithm!r} is not supported. "
                        f"OrthoLoRA's Cayley distill keys don't compose "
                        f"with the {self.algorithm} save layout. Pick one."
                    )
                raise ValueError(
                    f"AnimaLoraMethodLoraConfig: legacy {field}=True "
                    f"disagrees with algorithm={self.algorithm!r}. Drop "
                    f"the bool, or set algorithm={mapped!r}."
                )
            # Legacy bool wins when algorithm is the default.
            object.__setattr__(self, "algorithm", mapped)

        # No-op: ``use_X=False`` is just confirming the algorithm not
        # being chosen.
        return self


class AnimaLoraMethodPostfixConfig(BaseModel):
    """Postfix tuning knobs (``networks/methods/postfix.py``).

    Two modes: ``postfix`` is a free K×D tensor appended to the cached
    adapter output; ``cond`` is caption-conditional with structural
    orthogonality (Cayley rotation over an orthonormal basis).
    """

    model_config = _CAMEL_CONFIG

    mode: Literal["postfix", "cond"] = "cond"
    cond_hidden_dim: int = Field(1024, ge=1)
    splice_position: Literal["front_of_padding", "after_padding"] = "front_of_padding"
    ortho_basis: Literal["svd_te", "random", "identity"] = "svd_te"
    te_cache_dir: Path | None = None
    svd_num_files: int = Field(1024, ge=1)
    ortho_basis_seed: int = 0
    lambda_init: float = Field(0.3, gt=0)


class AnimaLoraMethodChimeraConfig(BaseModel):
    """ChimeraHydra dual-pool routing config.

    Content pool (router on pooled text features) + frequency pool
    (FreqRouter on FEI bands) sum additively into one A matrix per
    Linear. ``balance_w_*`` are per-router load-balance loss weights.
    """

    model_config = _CAMEL_CONFIG

    balance_w_content: float = Field(2e-7, ge=0)
    balance_w_freq: float = Field(5e-7, ge=0)
    balance_loss_warmup_ratio: float = Field(0.4, ge=0.0, le=1.0)
    fei_feature_dim: int = Field(2, ge=1)
    sigma_feature_dim: int = Field(16, ge=1)


class AnimaLoraMethodEasyControlConfig(BaseModel):
    """EasyControl Phase 1: per-block conditioning LoRA + softmax gate.

    ``b_cond_init = -10`` keeps step 0 identical to baseline DiT;
    ``cond_token_count`` is the static pad length (lower for tight VRAM).
    """

    model_config = _CAMEL_CONFIG

    b_cond_init: float = -10.0
    cond_scale: float = Field(1.0, gt=0)
    apply_ffn_lora: bool = True
    cond_token_count: int = Field(4096, ge=1)
    drop_p: float = Field(0.1, ge=0.0, le=1.0)
    cond_noise_max: float = Field(0.3, ge=0.0)


class AnimaLoraMethodIPAdapterConfig(BaseModel):
    """IP-Adapter: PE-Core encoder + resampler + per-block KV.

    ``gate_lr`` is intentionally ~10× the global LR — upstream noted
    that 8 epochs at 1e-4 only reached gate ``abs_max ~0.004`` without
    the boost.
    """

    model_config = _CAMEL_CONFIG

    encoder: Literal["PE-Core-L14-336", "PE-Core-G14-448"] = "PE-Core-L14-336"
    resampler_layers: int = Field(2, ge=1)
    resampler_heads: int = Field(8, ge=1)
    ip_scale: float = Field(1.0, gt=0)
    image_drop_p: float = Field(0.05, ge=0.0, le=1.0)
    gate_lr: float = Field(1e-3, gt=0)
    features_cache_to_disk: bool = True


class AnimaLoraTurboConfig(BaseModel):
    """DMD turbo distillation knobs (``scripts/distill_turbo.py``).

    Decoupled-Hybrid DMD2 (Liu et al. arXiv:2511.22677): trains a
    student LoRA + a fake LoRA on a frozen Anima DiT to bake CFG into
    the student so a 4-step Euler sample matches the 28-step teacher
    output at CFG=4. Output is a regular LoRA loaded via the standard
    inference path with ``--infer_steps 4 --cfg 1.0``.

    Mirrors the upstream ``configs/methods/turbo.toml`` schema. When
    set on ``AnimaLoraOptions.turbo``, the compiler routes through
    ``scripts/distill_turbo.py`` instead of ``train.py`` (no
    accelerate launch, bespoke CLI surface).
    """

    model_config = _CAMEL_CONFIG

    # Top-level
    iterations: int = Field(1000, ge=1)
    batch_size: int = Field(1, ge=1)
    # ComfyUI-style sentinel: -1 picks a fresh random seed at run start.
    seed: int = -1
    use_custom_down_autograd: bool = True

    # Network — student + fake LoRA capacities.
    student_rank: int = Field(48, ge=1)
    student_alpha: float = Field(48, gt=0)
    fake_rank: int = Field(64, ge=1)
    fake_alpha: float = Field(64, gt=0)
    # See main attn_mode: default to PyTorch SDPA so users without
    # flash-attn installed don't immediately bounce off a RuntimeError.
    attn_mode: Literal["flash", "torch", "flex", "sageattn", "xformers"] = "torch"

    # DMD2 schedule (proposal §Schedule, paper Table 1 row 4).
    student_steps: int = Field(4, ge=1)
    teacher_cfg: float = Field(4.0, gt=0)
    tau_ca_strategy: Literal["above_t", "uniform"] = "above_t"
    tau_dm_strategy: Literal["uniform", "above_t"] = "uniform"
    tau_ca_min_gap: float = Field(0.05, ge=0.0, lt=1.0)
    tau_ca_skip_above_t: float = Field(0.95, gt=0.0, le=1.0)

    # Optimization.
    student_lr: float = Field(5e-6, gt=0)
    fake_lr: float = Field(5e-5, gt=0)
    fake_steps_per_student_step: int = Field(2, ge=1)
    alpha_warmup_steps: int = Field(100, ge=0)
    weight_decay: float = Field(0.0, ge=0.0)
    grad_clip: float = Field(1.0, gt=0)

    # Sampling.
    t_distribution: Literal["uniform", "sigmoid"] = "uniform"
    sigmoid_scale: float = Field(1.0, gt=0)

    # I/O cadence.
    save_every: int = Field(250, ge=1)
    log_interval: int = Field(5, ge=1)


class AnimaLoraOptions(BaseModel):
    """anima_lora backend specific knobs.

    Independent from kohya / diffusion-pipe. Mirrors the upstream
    ``base.toml`` + ``methods/<method>.toml`` + ``presets.toml`` chain
    but presents one flat schema — the LoraHub compiler emits a
    pre-merged anima_lora.toml so we don't replay the upstream merge
    layering at runtime.

    Method axis: ``method`` selects which of the five sub-configs (lora,
    postfix, chimera, easycontrol, ip_adapter) is consumed by the
    compiler. The other sub-configs may be populated and will simply
    not surface in the emitted TOML — useful for keeping per-method
    presets around without losing them across method switches.
    """

    model_config = _CAMEL_CONFIG

    # ---- Method + preset axes ----
    method: Literal[
        "lora", "postfix", "chimera", "easycontrol", "ip_adapter"
    ] = "lora"
    preset: Literal[
        "default", "low_vram", "graft", "half", "quarter", "tenth", "debug"
    ] = "default"

    # ---- Output ----
    output_name: str = "anima_lora"

    # ---- Conditioning training (差异训练) ----
    # When True, ``--conditioning`` is forwarded to train.py and each
    # subset's ``conditioning_data_dir`` (on DatasetSubsetConfig) is
    # written into the generated dataset_config.toml so train.py pairs
    # target images with same-stem references and exposes
    # ``batch['conditioning_images']`` to downstream losses.
    # Mirumo fork docs:
    #   https://github.com/Mirumo0u0/sd-scripts/blob/main/docs/anima_conditioning_training-zh.md
    conditioning: bool = False

    # ---- Network ----
    network_module: str = "networks.lora_anima"
    network_dim: int = Field(32, ge=1)
    network_alpha: float = Field(32, gt=0)
    network_train_unet_only: bool = True
    # OrthoLoRA / LoRA channel-wise gain scaling (network kwarg, not a CLI
    # flag). Controls the magnitude of the per-channel scale applied on
    # the output of the LoRA branch — Backend's ``methods/lora.toml``
    # ships 0.5 as the numerically-stable default. Lower values shrink
    # the LoRA contribution at init (slower learn, less collapse risk);
    # higher values let the LoRA leg pull harder against the base.
    channel_scaling_alpha: float = Field(0.5, gt=0, le=1)

    # ---- Optimizer / schedule ----
    optimizer_type: Literal["AdamW", "AdamW8bit", "Lion", "Prodigy", "CAME"] = "AdamW"
    lr_scheduler: Literal[
        "constant", "cosine", "cosine_with_restarts", "linear", "polynomial"
    ] = "cosine"
    learning_rate: float = Field(2.0e-5, gt=0)
    # Warmup as a *ratio* of total steps. Upstream's ``--lr_warmup_steps``
    # is dual-typed: float < 1 → ratio, int >= 1 → absolute step count.
    # When this field is non-None it wins over the absolute-steps path
    # (cfg.optimizer.warmup_steps) and the compiler emits the value as a
    # float so argparse takes the ratio branch. None → fall back to the
    # absolute-steps path; the trainer then uses Backend's base.toml
    # default if neither is set.
    lr_warmup_ratio: float | None = Field(default=0.05, ge=0.0, le=1.0)
    max_train_epochs: int = Field(3, ge=1)
    save_every_n_epochs: int = Field(3, ge=1)
    # Optional step-based checkpoint cadence. When set, the trainer
    # also writes a ckpt every N steps (in addition to the epoch
    # cadence above). Useful for short epochs / small datasets where
    # a full epoch is the only thing that triggers a state dir, and
    # the pause-resume workflow wants finer granularity.
    save_every_n_steps: int | None = Field(default=None, ge=1)
    checkpointing_epochs: int = Field(3, ge=1)
    caption_dropout_rate: float = Field(0.1, ge=0.0, le=1.0)

    # ---- Sampling / loss (flow-matching for Anima DiT) ----
    timestep_sampling: Literal["sigmoid", "uniform", "logit_normal"] = "sigmoid"
    sigmoid_scale: float = Field(1.0, gt=0)
    discrete_flow_shift: float = Field(1.0, gt=0)
    # ``min_snr_rf`` is a rectified-flow-aware Min-SNR-γ weighting (Hang et
    # al. ICCV'23 adapted to RF). Enable by selecting it here AND setting
    # ``min_snr_gamma`` below — leaving ``min_snr_gamma`` None reduces to
    # uniform weighting and the trainer logs a warning.
    weighting_scheme: Literal[
        "sigma_sqrt", "logit_normal", "mode", "cosmap", "min_snr_rf"
    ] | None = None
    # γ for ``weighting_scheme = "min_snr_rf"``. Recommended 5.0; ignored
    # for any other weighting_scheme value.
    min_snr_gamma: float | None = Field(default=None, gt=0)
    logit_mean: float | None = None
    logit_std: float | None = None
    mode_scale: float | None = None
    # Variance-reduced flow-matching loss (AsymFlow §5.2). +40% step compute
    # when enabled; leave None to skip.
    vr_loss_weight: float | None = Field(default=None, ge=0)

    # ---- Training stabilisers ----
    # Exponential Moving Average over the LoRA network's trainable params.
    # Adds ~0% throughput overhead (one extra mul_/add_ per step), uses
    # roughly 2x adapter VRAM for the shadow copy. The shadow weights are
    # written next to every checkpoint as ``{name}_ema.safetensors`` and
    # are usually higher-quality than the live weights at inference time.
    ema: bool = False
    # Decay factor — 0.9999 is a sane default for typical LoRA training
    # (~10k step half-life). Lower it (0.999 / 0.99) for short runs so the
    # shadow doesn't lag too far behind the live params.
    ema_decay: float = Field(0.9999, gt=0, lt=1)
    # When True, scale decay during warmup as min(decay, (1+t)/(10+t)) so
    # the first ~hundred steps don't bake noise into the shadow.
    ema_use_num_updates: bool = True
    # Pre-backward + post-clip NaN/Inf guard on loss and gradients. When
    # nan_guard_recover is also on, the trainer halves every param group's
    # LR + restores the live params from EMA shadow after
    # ``nan_guard_max_consecutive`` strikes; otherwise it logs and aborts.
    nan_guard: bool = False
    nan_guard_recover: bool = False
    nan_guard_max_consecutive: int = Field(default=5, ge=1)
    # Compose every epoch's sample images into a contact-sheet PNG so
    # progression is visible in one click. Samples are still written
    # individually too — this only adds the grid sibling.
    sample_grid: bool = False

    # ---- Caching / data ----
    cache_latents: bool = True
    cache_latents_to_disk: bool = True
    cache_text_encoder_outputs: bool = True
    cache_text_encoder_outputs_to_disk: bool = True
    cache_llm_adapter_outputs: bool = True
    use_shuffled_caption_variants: bool = True
    # Subset sampling — per-preset override (debug=0.001, tenth=0.1 etc).
    sample_ratio: float | None = Field(default=None, gt=0.0, le=1.0)
    # Pad sequence length to this token count (legacy 4096-pad path).
    # Set to ``None`` (or omit) when ``enable_native_flatten=true`` —
    # the two paths are mutually exclusive (vendored ``compile_blocks``
    # asserts) and policies.py rejects the combo at validate-time.
    static_token_count: int | None = Field(default=4096, ge=1)
    vae_chunk_size: int = Field(64, ge=1)
    vae_disable_cache: bool = False
    no_half_vae: bool = False

    # ---- Attention / compile ----
    # Default to ``flash`` to match Backend's base.toml — flash-attn
    # delivers the best throughput on Ampere+ when the build is
    # available. Operators without a working flash-attn install should
    # flip this to ``torch`` (PyTorch SDPA) in their config; SDPA hits
    # ~85-95% of flash-attn throughput and is always available.
    attn_mode: Literal["flash", "torch", "flex", "sageattn", "xformers"] = "flash"
    xformers: bool = False
    split_attn: bool = False
    # ``compile_mode = "full"`` enables CUDAGraph capture via
    # ``compile_inductor_mode = "reduce-overhead"``. Incompatible with
    # gradient_checkpointing / blocks_to_swap.
    compile_mode: Literal["blocks", "full"] | None = None
    compile_inductor_mode: Literal[
        "default", "reduce-overhead", "max-autotune"
    ] | None = None
    # Native-shape flattening (compile_blocks(native_flatten=True) on the
    # vendored DiT). When True the trainer:
    #   * switches to the 4032+4200 two-family bucket table (every bucket
    #     exactly fills its token count, zero intra-bucket padding)
    #   * compiles the block stack to TWO graphs keyed on token count
    #     alone (vs ~24 graphs per resolution in the legacy static_pad
    #     path) — ~2x training throughput on RTX Pro 6000 / 4090 class
    #     GPUs that bottleneck on dynamo guard checks rather than FLOPs
    #   * is mutually exclusive with non-zero static_token_count; the
    #     vendored ``compile_blocks`` raises if both are set
    # Off by default so existing configs keep their behaviour. Switching
    # ON requires re-caching the dataset (the bucket table changes).
    enable_native_flatten: bool = False
    # Bucket-resolution table override. ``"1536"`` selects the
    # 9216+9240 two-family table for Anima v1.0's native 1536x1536
    # training (12 entries covering ar 0.44-2.25). Pair with
    # ``enable_native_flatten=true`` (recommended) or
    # ``static_token_count >= 9240``. ``None`` / "default" uses the
    # table the legacy / native-flatten flags would otherwise pick.
    bucket_table: Literal["default", "1536"] | None = None
    use_custom_down_autograd: bool = True

    # ---- Memory / offload ----
    blocks_to_swap: int = Field(0, ge=0)
    gradient_checkpointing: bool = False
    unsloth_offload_checkpointing: bool = False
    cpu_offload_checkpointing: bool = False
    mixed_precision: Literal["bf16", "fp16", "fp32"] = "bf16"

    # ---- Validation (CMMD + sample-time) ----
    use_cmmd: bool = False
    validation_seed: int | None = None
    validation_sample_steps: int | None = Field(default=None, ge=1)
    validation_cfg_scale: float | None = Field(default=None, gt=0)

    # ---- Upstream-default fields (B5 cut-locks) ----
    #
    # These mirror keys from anima_lora's vendored ``configs/base.toml``
    # and ``configs/methods/lora.toml`` ``[[datasets]]`` block. Many of
    # them are "pipeline-locked" in upstream — base.toml hard-codes them
    # to true and the train.py argparse offers no ``--no_<x>`` reverse
    # flag, so flipping them off in LoraHub silently does nothing. The
    # frontend renders a 🔒 badge on each one and the compiler logs a
    # warning (not an error) when the user sets a locked field to a
    # value the upstream would ignore.
    #
    # Lock taxonomy:
    #   * LOCKED_TRUE    — base.toml has it = true and there's no
    #     reverse CLI flag. Setting False in LoraHub is a no-op.
    #   * LOCKED_VALUE   — base.toml or method file pins a specific
    #     non-bool value the trainer's static-shape compile chain
    #     depends on (e.g. static_token_count = 4096 for Anima DiT).
    #   * RISKY          — can be changed but breaks something obvious.
    # See ``lorahub.core.backends.anima_lora.compiler.LOCKED_FIELDS``.

    # ⚠️ RISKY — masked loss. 上游来自 kohya/sd-scripts 的 --masked_loss
    # (store_true,默认 False),Anima 沿用同一开关。开启后 apply_masked_loss
    # 在每步 loss 上贴 mask:优先取 batch["conditioning_images"] 的 R 通道,
    # 其次取 alpha_masks。若两者皆无则 no-op,但若 batch 把
    # conditioning_images=None 显式塞进去就会撞 NoneType.to。
    # 实际生效路径需要差异训练资料: conditioning=True 且至少一个 subset
    # 配了 conditioning_data_dir;否则保持默认 False 即可。
    masked_loss: bool = False
    # 🔒 LOCKED_TRUE — torch.compile is required for the static-shape
    # constant-token bucketing to pay off; upstream's loop assumes it.
    torch_compile: bool = True
    # 🔒 LOCKED_TRUE — skip_cache_check trades safety for startup speed
    # (skips per-image hash verification before training). On by
    # default; turning off would force a 248-image dataset to re-hash
    # every run with no functional benefit.
    skip_cache_check: bool = True
    # 🔒 LOCKED_TRUE — DataLoader pin_memory; upstream sets it true and
    # offers no off-switch. Off would slow down host→GPU transfers.
    dataloader_pin_memory: bool = True
    # ⚠️ RISKY — persistent dataloader workers; Backend's base.toml ships
    # this true (workers stay alive across epochs, reducing epoch-boundary
    # stalls). Setting false reverts to per-epoch worker spawn — slower
    # but avoids long-run file-handle leaks.
    persistent_data_loader_workers: bool = True
    # 🔒 LOCKED_TRUE — base.toml writes ``trim_crossattn_kv = false``,
    # but the corresponding flag is store_true so users *can* turn it
    # on via CLI. Setting True enables KV trimming for short captions
    # (~10-15% throughput gain). Default false is upstream-faithful.
    trim_crossattn_kv: bool = False

    # ⚠️ RISKY — save format. Always safetensors for Anima (other
    # formats are kohya legacies that don't round-trip through Anima's
    # weight loading). LoraHub doesn't expose alternatives.
    save_model_as: Literal["safetensors"] = "safetensors"
    # ⚠️ RISKY — save dtype. fp16 is smaller, bf16 is upstream-default
    # and matches the training compute dtype on Ampere+; fp32 doubles
    # disk usage with no quality benefit.
    save_precision: Literal["bf16", "fp16", "fp32"] = "bf16"

    # log_every_n_steps — how often the trainer flushes step events to
    # tensorboard. Cosmetic; default 2 ≈ 0.5Hz.
    log_every_n_steps: int = Field(2, ge=1)

    # ---- Dataset blueprint fields (under [[datasets]] / [general]) ----
    # These live in the dataset blueprint section of base.toml and
    # aren't argparse-driven — LoraHub emits them through a separate
    # ``--dataset_config`` override or a method-TOML shallow-merge.
    # Default values mirror upstream.

    # ⚠️ RISKY — caption-shuffle "keep first N tags" knob. base.toml
    # uses 3 (matching Anima's training-time T5 caption format with
    # the trigger / character / character-feature triple at front).
    # Changing this can degrade trigger-word reliability.
    keep_tokens: int = Field(3, ge=0)
    # 🔒 LOCKED_VALUE — caption file extension (``.txt``). upstream's
    # data pipeline scans for this exact suffix; changing it skips
    # every image with no warning. Exposed for completeness.
    caption_extension: str = ".txt"
    # ⚠️ RISKY — held-out validation set count. Backend's base.toml uses
    # 0 (no held-out images; CMMD validation disabled by default; the
    # method TOML can override). Bump this to enable val_loss tracking.
    validation_split_num: int = Field(0, ge=0)
    # Bucketing is a hard requirement of Anima's static-shape compile;
    # 🔒 LOCKED_TRUE here too.
    enable_bucket: bool = True
    # ⚠️ RISKY — fnmatch glob for image discovery; ``*`` trains on
    # everything. Use ``char_a/*|char_b/*`` to OR-combine subfolders.
    path_pattern: str = "*"

    # ---- Method-specific sub-configs ----
    # Only the sub-config matching `method` is consumed by the compiler;
    # populating the others is allowed (lets users keep per-method
    # presets across method switches) but harmless until selected.
    lora: AnimaLoraMethodLoraConfig = Field(default_factory=AnimaLoraMethodLoraConfig)
    postfix: AnimaLoraMethodPostfixConfig | None = None
    chimera: AnimaLoraMethodChimeraConfig | None = None
    easycontrol: AnimaLoraMethodEasyControlConfig | None = None
    ip_adapter: AnimaLoraMethodIPAdapterConfig | None = None
    # DMD turbo distillation — orthogonal to method/preset axes. When
    # set, the compiler routes through scripts/distill_turbo.py instead
    # of train.py and the method/preset values are ignored. Output is
    # still a regular LoRA loaded via the standard inference path.
    turbo: AnimaLoraTurboConfig | None = None

    @model_validator(mode="after")
    def _check_method_subconfig_present(self) -> AnimaLoraOptions:
        """When `method` ≠ 'lora', the matching sub-config must be set.

        Lets us catch the "selected `method=postfix` but forgot to fill
        the sub-config" error at validation time instead of crashing
        the compiler with an attribute error.
        """
        if self.method == "postfix" and self.postfix is None:
            msg = "method='postfix' requires the `postfix` sub-config to be set"
            raise ValueError(msg)
        if self.method == "chimera" and self.chimera is None:
            msg = "method='chimera' requires the `chimera` sub-config to be set"
            raise ValueError(msg)
        if self.method == "easycontrol" and self.easycontrol is None:
            msg = "method='easycontrol' requires the `easycontrol` sub-config to be set"
            raise ValueError(msg)
        if self.method == "ip_adapter" and self.ip_adapter is None:
            msg = "method='ip_adapter' requires the `ipAdapter` sub-config to be set"
            raise ValueError(msg)
        return self


