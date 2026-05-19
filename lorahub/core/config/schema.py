"""Semantic recipe schema for LoRA training.

Users write a single YAML file describing *what* they want to train. The schema
validates it, fills defaults (tuned for 8GB VRAM on SDXL), and later a
backend-specific compiler translates it into the backend's native arguments.

The schema deliberately mirrors the union of kohya-ss/sd-scripts argv and
tdrussell/diffusion-pipe TOML keys. Each backend's compiler emits whichever
fields its upstream understands and silently ignores the rest, so the same
recipe can be retargeted between backends with minimal edits.

Field consumption is split across three layers, which can be confusing when
spelunking a single field "is anyone reading this?":

  1. **Compiler-level fields** (``schedule.epochs``, ``optimizer.lr.unet``,
     ``loss.min_snr_gamma``, ...). Read by ``lorahub.core.backends.<kohya|
     diffusion_pipe>.compiler`` and emitted to argv / TOML. Most fields
     live here.

  2. **Runtime-level fields** (``sampling.enable_live_inference``,
     ``sampling.inference_steps`` / ``inference_cfg``,
     ``backend.sd_scripts_path``, ``backend.python_executable``,
     ``backend.pin_version``). The compiler doesn't read these; they're
     consumed by ``lorahub.api.jobs_helpers`` (live preview worker) or
     ``lorahub.api.routers.bootstrap`` (which kohya checkout to install).

  3. **UI-only fields** (``schema_version``, ``dataset.caption.strategy``).
     Frontend form gates UI controls on these; backends ignore them.
     Kept in the schema so YAML files round-trip cleanly through the UI.

When a backend declares a field unsupported, it should ``_track`` it via
the compiler's `dropped` audit list rather than failing — so a config
written for kohya can still validate (and partially compile) under
diffusion-pipe.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.alias_generators import to_camel


# Shared model_config: every YAML field is accepted both in its Python
# snake_case form and in camelCase (the canonical wire form going forward).
# `populate_by_name=True` keeps existing recipes valid; `extra="forbid"` is
# applied per-model where appropriate.
_CAMEL_CONFIG = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class BaseModelConfig(BaseModel):
    model_config = _CAMEL_CONFIG

    # The arch literal mirrors the union of upstream-supported model families
    # across kohya sd-scripts and diffusion-pipe. Backends are responsible for
    # rejecting arches they do not implement (kohya rejects dp-only entries
    # like `wan` or `hunyuan_video`; dp rejects kohya-only entries like sd15).
    # Old recipe values stay valid; new values follow each upstream's docs.
    arch: Literal[
        # kohya sd-scripts (README "Supported Models")
        "sd15",
        "sd2",
        "sdxl",
        "sd3",
        "flux",
        "lumina",
        "hunyuan_image",
        "anima",
        # diffusion-pipe (docs/supported_models.md) -- additional entries
        "flux2",
        "chroma",
        "hidream",
        "omnigen2",
        "auraflow",
        "qwen_image",
        "cosmos",
        "cosmos_predict2",
        "hunyuan_video",
        "hunyuan_video_15",
        "ltx_video",
        "ltx2",
        "wan",
        "z_image",
        "ernie_image",
    ] = "sdxl"
    # SDXL sub-architectures sharing the SDXL backbone but trained on
    # different finetune lineages (Pony/Illustrious/NoobAI/Animagine).
    # Backends still treat these as SDXL; the variant only nudges
    # default learning rates and a couple of CLI flags.
    arch_variant: Literal["", "pony", "illustrious", "noobai", "animagine"] = ""
    checkpoint: Path
    vae: Path | None = None
    # Per-component checkpoint paths for arches that ship as multi-file
    # bundles (FLUX = clip_l + t5xxl + ae; SD3 = clip_l/g + t5xxl;
    # Anima = qwen3 + qwen_image_vae + transformer; HunyuanImage = byt5 +
    # text_encoder; ...). Free-form bag because upstream keeps adding
    # entries. The kohya/dp compilers each render the keys their
    # respective upstream understands.
    arch_paths: ArchPathsConfig = Field(default_factory=lambda: ArchPathsConfig())


class ArchPathsConfig(BaseModel):
    """Arch-specific component paths shared by both backends.

    Empty by default — only set the fields your model actually uses.
    Both compilers consume the same fields and emit them under the names
    the corresponding upstream expects (kohya argv, dp TOML keys).
    """

    model_config = _CAMEL_CONFIG

    # FLUX / SD3 / FLUX2
    clip_l: Path | None = None
    clip_g: Path | None = None
    t5xxl: Path | None = None
    ae: Path | None = None  # FLUX autoencoder

    # Generic (Anima / Wan / HunyuanImage / chroma transformer-style)
    transformer: Path | None = None
    text_encoder: Path | None = None
    llm: Path | None = None  # Anima Qwen3, HunyuanVideo LLM
    byt5: Path | None = None  # HunyuanImage byT5

    # Anima-specific
    qwen3: Path | None = None
    t5_tokenizer: Path | None = None
    llm_adapter: Path | None = None

    # Token length caps
    t5xxl_max_token_length: int | None = Field(default=None, ge=1)
    qwen3_max_token_length: int | None = Field(default=None, ge=1)
    t5_max_token_length: int | None = Field(default=None, ge=1)

    # Attention masking + dropout — FLUX/SD3
    apply_t5_attn_mask: bool = False
    apply_lg_attn_mask: bool = False
    t5_dropout_rate: float = Field(0.0, ge=0.0, lt=1.0)
    clip_l_dropout_rate: float = Field(0.0, ge=0.0, lt=1.0)
    clip_g_dropout_rate: float = Field(0.0, ge=0.0, lt=1.0)

    # SD3 positional-embed crop
    pos_emb_random_crop_rate: float = Field(0.0, ge=0.0, lt=1.0)
    enable_scaled_pos_embed: bool = False

    # FLUX dev distilled guidance scale baked into the LoRA
    guidance_scale: float | None = Field(default=None, gt=0)

    # Place TE on a specific device / dtype (SD3 separates TEs from UNet)
    t5xxl_device: str | None = None
    t5xxl_dtype: Literal["fp16", "bf16", "fp32", "fp8"] | None = None

    # VAE memory tweaks (Anima / HunyuanImage / Wan)
    vae_chunk_size: int | None = Field(default=None, ge=1)
    vae_disable_cache: bool = False
    text_encoder_cpu: bool = False


class BucketConfig(BaseModel):
    enabled: bool = True
    min_size: int = Field(256, alias="min")
    max_size: int = Field(2048, alias="max")
    step: int = 64
    # Don't upscale images smaller than the bucket; clamps tiny inputs
    # instead of stretching them. kohya: --bucket_no_upscale.
    no_upscale: bool = False
    # Skip the resolution sanity check (kohya: --skip_image_resolution).
    # Useful for datasets with unusual aspect ratios.
    skip_image_resolution: bool = False
    # PIL resampling kernel. None lets the trainer pick its default.
    # Mirrors kohya's --bucket_reso_steps companion flag's accepted set.
    resize_interpolation: Literal[
        "lanczos", "nearest", "bilinear", "linear", "bicubic", "cubic", "area"
    ] | None = None
    # diffusion-pipe accepts an explicit AR list overriding min/max/num.
    # Each entry is a width/height ratio; only consumed by the dp compiler.
    ar_buckets: list[float] | None = None

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
    )


class CaptionConfig(BaseModel):
    model_config = _CAMEL_CONFIG

    # NOTE: ``strategy`` is consumed by the front-end form (it gates
    # which UI controls render). The kohya / diffusion-pipe compilers
    # don't read it — they always look for an ``<image>.txt`` companion
    # file regardless of strategy. Keep it in the schema so configs
    # round-trip cleanly through the UI, but treat it as documentation
    # of intent rather than a backend-driving knob.
    strategy: Literal["tag_file", "filename", "none"] = "tag_file"
    ext: str = ".txt"
    shuffle: bool = True
    drop_rate: float = Field(0.0, ge=0.0, le=1.0)
    # Per-epoch caption dropout (kohya: --caption_dropout_every_n_epochs).
    # Different from drop_rate which is per-step.
    dropout_every_n_epochs: int = Field(0, ge=0)
    # Per-tag dropout — drops individual tags within a caption.
    tag_dropout_rate: float = Field(0.0, ge=0.0, le=1.0)
    # First N comma-separated tokens are NEVER shuffled away; pinning
    # the trigger word at index 0 is the typical use.
    keep_tokens: int = Field(0, ge=0)
    # Hard-drop list — every entry that appears in a caption is
    # removed verbatim before training (case-insensitive substring
    # match, then comma-list cleanup). Entries can be either tag-style
    # (``"1girl"``, ``"looking at viewer"``) or natural-language
    # phrases (``"a person standing in front of a window"``). Applied
    # at compile time to a sanitised mirror of the dataset under
    # ``<workspace>/captions_sanitized/`` so the trainer reads the
    # filtered text, but the user's source ``.txt`` files are left
    # untouched. Empty list = no-op (the mirror step is also skipped).
    drop_tokens: list[str] = Field(default_factory=list)
    # Custom separator between "kept" and shufflable tokens; default ","
    keep_tokens_separator: str | None = None
    # Secondary separator within a token group (e.g. " ,"). kohya only.
    secondary_separator: str | None = None
    # `{a|b|c}` wildcard support in captions (kohya: --enable_wildcard).
    enable_wildcard: bool = False
    # Compose-time prefix/suffix prepended/appended to every caption.
    prefix: str | None = None
    suffix: str | None = None
    # Max tokenizer length (kohya: --max_token_length, valid: 150/225;
    # 75 is the implicit default when the flag is absent so it's
    # represented as None here rather than an explicit value).
    max_token_length: Literal[150, 225] | None = None
    # Token warmup (slow-start the tag count). kohya only.
    token_warmup_min: int | None = Field(default=None, ge=1)
    token_warmup_step: float | None = Field(default=None, ge=0)
    # Weighted captions (lpw-style `(token:1.5)`). kohya: --weighted_captions.
    weighted: bool = False
    # dp-only: tag shuffle delimiter (default ", ") and legacy whole-caption shuffle.
    shuffle_delimiter: str | None = None
    shuffle_tags: bool = False


class DatasetSubsetConfig(BaseModel):
    """One [[directory]] entry on the dp side; kohya squashes these into
    `--train_data_dir` semantics via per-subset toml."""

    model_config = _CAMEL_CONFIG

    path: Path
    num_repeats: int = Field(1, ge=1)
    # Optional mask directory, mirrors the image dir layout.
    mask_path: Path | None = None
    # Per-subset bucket override (dp).
    ar_buckets: list[float] | None = None
    # Per-subset caption override.
    caption_prefix: str | None = None


class DatasetConfig(BaseModel):
    model_config = _CAMEL_CONFIG

    source: Path
    resolution: list[int] = Field(default_factory=lambda: [1024, 1024])
    bucket: BucketConfig = Field(default_factory=lambda: BucketConfig())
    caption: CaptionConfig = Field(default_factory=lambda: CaptionConfig())
    num_repeats: int = Field(1, ge=1)
    # Fraction of the dataset reserved for held-out validation. `0.0` disables
    # validation entirely (the previous behaviour); upper bound stays under
    # 0.5 because anything more would be a strange split. sd-scripts' flag
    # `--validation_split_percentage` takes an integer percent — we convert
    # at compile time.
    val_split: float = Field(0.0, ge=0.0, lt=0.5)
    # Multi-directory support — when populated, OVERRIDES `source`.
    # dp emits one [[directory]] block per entry; kohya synthesises an
    # equivalent dataset toml.
    subsets: list[DatasetSubsetConfig] = Field(default_factory=list)
    # Video training: list of frame counts (e.g. `[1, 33, 65]`).
    # Default `[1]` means image-only.
    frame_buckets: list[int] = Field(default_factory=lambda: [1])
    # ControlNet / inpainting conditioning images (kohya: --conditioning_data_dir).
    conditioning_dir: Path | None = None
    # DreamBooth regularisation set (kohya: --reg_data_dir).
    reg_source: Path | None = None

    @model_validator(mode="after")
    def _validate_resolution(self) -> DatasetConfig:
        if len(self.resolution) not in (1, 2):
            msg = "resolution must be [size] or [width, height]"
            raise ValueError(msg)
        return self


class PerModuleLRConfig(BaseModel):
    """Per-submodule LR overrides for arches that train multiple components.

    Anima exposes llm_adapter / self_attn / cross_attn / mlp / mod LRs
    independently; SD3 separates TE from UNet; etc. None means "inherit
    the global unet LR".
    """

    model_config = _CAMEL_CONFIG

    llm_adapter: float | None = Field(default=None, gt=0)
    self_attn: float | None = Field(default=None, gt=0)
    cross_attn: float | None = Field(default=None, gt=0)
    mlp: float | None = Field(default=None, gt=0)
    mod: float | None = Field(default=None, gt=0)


class NetworkConfig(BaseModel):
    model_config = _CAMEL_CONFIG

    type: Literal["lora", "locon", "loha", "dora"] = "lora"
    rank: int = Field(32, ge=1, le=512)
    alpha: int = Field(16, ge=1)
    target_unet: bool = True
    target_text_encoder: bool = False
    # Convolutional rank/alpha for locon/loha. Plain `lora` doesn't touch
    # conv layers, so these only make sense on lycoris flavours and the
    # validator below rejects them otherwise. `conv_alpha=None` means
    # "let sd-scripts default it (commonly mirrors `alpha`)".
    conv_dim: int | None = Field(default=None, ge=1, le=512)
    conv_alpha: int | None = Field(default=None, ge=1)
    # Regularisation knobs forwarded to sd-scripts as `--network_args`. All
    # default to 0 / None so existing recipes keep emitting identical argv.
    network_dropout: float = Field(0.0, ge=0.0, lt=1.0)
    rank_dropout: float = Field(0.0, ge=0.0, lt=1.0)
    module_dropout: float = Field(0.0, ge=0.0, lt=1.0)
    # Top-level `--scale_weight_norms` max-norm scalar. None disables it.
    scale_weight_norms: float | None = Field(default=None, gt=0)
    # Continue training from an existing LoRA. kohya: --network_weights;
    # dp: [adapter] init_from_existing.
    init_from: Path | None = None
    # Tell kohya to read rank from the loaded weights file rather than CLI.
    dim_from_weights: Path | None = None
    # Merge a list of LoRA bases into the model before training (kohya).
    base_weights: list[Path] = Field(default_factory=list)
    base_weights_multiplier: list[float] = Field(default_factory=list)
    # dp: list of `{path, multiplier}` dicts to fuse before training.
    fuse_adapters: list[dict[str, Any]] = Field(default_factory=list)
    # Per-module LR overrides (Anima/Wan multi-component models).
    module_lr: PerModuleLRConfig | None = None
    # LoRA training dtype on dp. kohya is always fp32 for LoRA params.
    dtype: Literal["fp16", "bf16", "fp32"] | None = None

    @model_validator(mode="after")
    def _validate_conv_for_lora(self) -> NetworkConfig:
        """`lora` and `dora` don't expose conv layers in sd-scripts, so
        rejecting `conv_dim` / `conv_alpha` upfront avoids a confusing
        runtime crash inside the trainer."""
        if self.type in ("lora", "dora"):
            if self.conv_dim is not None:
                msg = (
                    f"network.conv_dim is only valid for locon/loha "
                    f"(got network.type={self.type!r})"
                )
                raise ValueError(msg)
            if self.conv_alpha is not None:
                msg = (
                    f"network.conv_alpha is only valid for locon/loha "
                    f"(got network.type={self.type!r})"
                )
                raise ValueError(msg)
        if len(self.base_weights) != len(self.base_weights_multiplier):
            if self.base_weights and self.base_weights_multiplier:
                msg = (
                    "network.base_weights and base_weights_multiplier "
                    "must have the same length"
                )
                raise ValueError(msg)
        return self


class LRConfig(BaseModel):
    model_config = _CAMEL_CONFIG

    unet: float = 1.0e-4
    text_encoder: float = 5.0e-5


class OptimizerConfig(BaseModel):
    """Optimizer hyperparameters consumed by both kohya and diffusion-pipe.

    `betas`, `weight_decay`, and `eps` map to the standard AdamW-style knobs;
    `optimizer_args` is a free-form `key=value` bag for backend-specific
    extensions (e.g. Lion's `momentum`, Prodigy's `decouple`). User-provided
    `optimizer_args` keys win over the dedicated `betas`/`weight_decay`/`eps`
    when names collide on the kohya `--optimizer_args` line.
    """

    model_config = _CAMEL_CONFIG

    type: str = "adamw8bit"
    lr: LRConfig = Field(default_factory=lambda: LRConfig())
    schedule: str = "cosine_with_restarts"
    warmup_steps: int = 100
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = Field(0.0, ge=0.0)
    eps: float = Field(1e-8, gt=0)
    optimizer_args: dict[str, str] = Field(default_factory=dict)
    # Gradient clipping max-norm. kohya: --max_grad_norm; dp: gradient_clipping
    # (already on DiffusionPipeOptions, this top-level mirror lets recipes
    # share). 0 disables clipping.
    max_grad_norm: float = Field(1.0, ge=0)
    # Custom LR scheduler module (kohya: --lr_scheduler_type).
    scheduler_module: str | None = None
    # Free-form scheduler-specific kwargs (kohya: --lr_scheduler_args).
    scheduler_args: dict[str, str] = Field(default_factory=dict)
    # cosine_with_restarts cycle count.
    scheduler_num_cycles: int = Field(1, ge=1)
    # polynomial decay power.
    scheduler_power: float = Field(1.0, gt=0)
    # inverse_sqrt timescale.
    scheduler_timescale: int | None = Field(default=None, ge=1)
    # cosine min-LR ratio (kohya: --lr_scheduler_min_lr_ratio).
    scheduler_min_lr_ratio: float | None = Field(default=None, ge=0, le=1)
    # dp gradient_release: chunk-wise grad release for memory savings.
    gradient_release: bool = False


class LossConfig(BaseModel):
    """Core loss-shaping hyperparameters.

    Per-step noise / SNR weighting / huber. Currently kohya consumes most
    of these; dp ignores all but `pseudo_huber_c`. Advanced flow-matching
    knobs live on `FlowMatchConfig`.
    """

    model_config = _CAMEL_CONFIG

    min_snr_gamma: float | None = Field(default=None, gt=0)
    noise_offset: float = Field(0.0, ge=0)
    noise_offset_random_strength: bool = False
    multires_noise_iterations: int | None = Field(default=None, ge=1)
    multires_noise_discount: float = Field(0.3, ge=0.0, le=1.0)
    adaptive_noise_scale: float | None = None
    ip_noise_gamma: float | None = Field(default=None, gt=0)
    ip_noise_gamma_random_strength: bool = False
    zero_terminal_snr: bool = False
    min_timestep: int | None = Field(default=None, ge=0)
    max_timestep: int | None = Field(default=None, ge=0)
    prior_loss_weight: float = Field(1.0, ge=0)
    loss_type: Literal["l2", "huber", "smooth_l1"] = "l2"
    huber_schedule: Literal["constant", "exponential", "snr"] | None = None
    huber_c: float | None = Field(default=None, gt=0)
    huber_scale: float | None = Field(default=None, gt=0)
    debiased_estimation: bool = False
    masked_loss: bool = False
    scale_v_pred_loss_like_noise_pred: bool = False
    v_parameterization: bool = False
    v_pred_like_loss: float | None = Field(default=None, gt=0)
    # dp: pseudo Huber loss constant (top-level TOML).
    pseudo_huber_c: float | None = Field(default=None, gt=0)


class FlowMatchConfig(BaseModel):
    """Flow-matching hyperparameters used by FLUX / SD3 / Lumina / Anima /
    HunyuanImage / chroma. These are entirely separate from the SD-style
    epsilon-prediction loss in `LossConfig`.

    None values mean "use the trainer's default for the chosen arch".
    """

    model_config = _CAMEL_CONFIG

    # logit_normal / uniform / sigma_uniform / mode / cosmap. kohya/dp arch-specific.
    timestep_sampling: Literal[
        "logit_normal", "uniform", "sigma_uniform", "mode", "cosmap"
    ] | None = None
    sigmoid_scale: float | None = Field(default=None, gt=0)
    model_prediction_type: Literal["raw", "additive", "sigma_scaled"] | None = None
    # Discrete flow timestep shift (FLUX/Anima).
    discrete_flow_shift: float | None = Field(default=None, gt=0)
    # SD3 training-time shift.
    training_shift: float | None = Field(default=None, gt=0)
    # FLUX/SD3 timestep weighting scheme.
    weighting_scheme: Literal[
        "sigma_sqrt", "logit_normal", "mode", "cosmap", "none"
    ] | None = None
    logit_mean: float | None = None
    logit_std: float | None = Field(default=None, gt=0)
    mode_scale: float | None = Field(default=None, gt=0)


class ScheduleConfig(BaseModel):
    model_config = _CAMEL_CONFIG

    epochs: int = Field(10, ge=1)
    batch_size: int = Field(1, ge=1)
    grad_accum: int = Field(2, ge=1)
    max_steps: int | None = None
    # Random seed (kohya: --seed).
    seed: int | None = None
    # cosine/linear decay window if you want it shorter than the full run.
    lr_decay_steps: int | None = Field(default=None, ge=1)


class SamplingConfig(BaseModel):
    model_config = _CAMEL_CONFIG

    enabled: bool = True
    every_n_epochs: int = Field(1, ge=1)
    # Step-level sampling cadence (kohya: --sample_every_n_steps).
    every_n_steps: int | None = Field(default=None, ge=1)
    # Generate a baseline before training starts (kohya: --sample_at_first).
    at_first: bool = False
    prompts_file: Path | None = None
    resolution: list[int] = Field(default_factory=lambda: [1024, 1024])
    seed: int = 42
    # Attention backend used ONLY for sample image generation. Training
    # forward stays on `cfg.attention.training`. SageAttention is the
    # main motivator: its INT8 forward kernel has no matching backward
    # so it would corrupt LoRA gradients in training, but it's safe and
    # fast in the sample/validation pipeline (long-video previews on
    # Wan / HunyuanVideo are where it pays off most).
    # `default` reuses the training backend; explicit choice overrides.
    attention: Literal["default", "torch", "sdpa", "xformers", "flash", "sageattn"] = "default"

    # diffusion-pipe doesn't generate preview images on its own. When
    # `enable_live_inference` is on, the lorahub job runner starts a
    # background watcher that polls the workspace `output/step*` dirs
    # and runs an in-process Anima inference for every new checkpoint
    # using the prompt list at `prompts_file`. The PNGs land under
    # `workspace/samples/` and a `sample_ready` event is emitted so
    # the analysis-tab gallery picks them up live.
    #
    # Off by default — turning it on adds GPU pressure during the
    # narrow window between checkpoints; only useful with the dp
    # backend (kohya already produces previews via --sample_prompts).
    enable_live_inference: bool = False
    inference_steps: int = Field(24, ge=1)
    inference_cfg: float = Field(5.0, gt=0)


class AttentionConfig(BaseModel):
    """Selects the attention kernel for the training forward+backward pass.

    Backends with no working backward (every flavour of SageAttention) are
    intentionally absent from this enum — they belong on
    ``sampling.attention``. ``flash3`` / ``flash4`` require Hopper /
    Blackwell hardware respectively; the runtime gates them by
    compute-capability and falls back to ``flash`` (FlashAttention 2)
    when the host can't run the chosen kernel.
    """

    model_config = _CAMEL_CONFIG

    training: Literal[
        "auto",     # pick the best available kernel for this GPU
        "torch",    # naive torch attention — debugging only
        "sdpa",     # F.scaled_dot_product_attention (PyTorch native)
        "flex",     # torch.nn.attention.flex_attention (PyTorch 2.5+)
        "xformers",
        "flash",    # FlashAttention 2 (Ampere/Ada/Hopper)
        "flash3",   # FlashAttention 3 (Hopper-only)
        "flash4",   # FlashAttention 4 beta (Hopper/Blackwell)
    ] = "auto"
    # Some backends require a memory-split attention path on kohya
    # (notably xformers). Mirrors --split_attn.
    split: bool = False


class DataLoaderConfig(BaseModel):
    """DataLoader / cache pipeline knobs.

    kohya: --max_data_loader_n_workers, --persistent_data_loader_workers,
    --vae_batch_size, --text_encoder_batch_size. dp: caching_batch_size,
    map_num_proc.
    """

    model_config = _CAMEL_CONFIG

    num_workers: int = Field(8, ge=0)
    persistent_workers: bool = False
    vae_batch_size: int = Field(1, ge=1)
    text_encoder_batch_size: int | None = Field(default=None, ge=1)
    cache_shuffle_num: int = Field(0, ge=0)
    map_num_proc: int | None = Field(default=None, ge=1)


class AugmentationConfig(BaseModel):
    """Image augmentation (kohya only). dp doesn't currently consume any of these."""

    model_config = _CAMEL_CONFIG

    flip: bool = False
    color: bool = False
    random_crop: bool = False
    # `min_face_size,target_size,max_face_size` triple, kohya format.
    face_crop_aug_range: str | None = None
    # Use image alpha channel as masked-loss mask.
    alpha_mask: bool = False


class OptimizationConfig(BaseModel):
    """Training-time speed / VRAM knobs that sit alongside the optimiser.

    Each field lines up with an upstream argv or TOML key the per-backend
    compilers already know how to emit. Defaults match upstream defaults
    so existing recipes keep producing identical commands.
    """

    model_config = _CAMEL_CONFIG

    # PyTorch 2 graph compilation. kohya: --torch_compile. dp:
    # `pipeline_model.compile(dynamic=True)` is currently unconditional
    # in upstream's train.py, so dp ignores this knob (kept for parity
    # of UI/recipe shape).
    torch_compile: bool = False
    # Fused LoRA backward + optimizer step. kohya: --fused_backward_pass.
    # Saves one gradient buffer; LoRA-compatible. dp does not have an
    # equivalent argv yet (its DeepSpeed pipeline orders bwd/step
    # internally) — passing this through dp is a no-op.
    fused_backward_pass: bool = False
    # Train all parameters in bf16 (model + grads + optimizer states).
    # kohya: --full_bf16. dp: optim_dtype="bf16".
    full_bf16: bool = False
    # Same idea, fp16 path (older GPUs).
    full_fp16: bool = False
    # Number of transformer blocks temporarily offloaded to CPU during
    # the forward pass. kohya FLUX/SD3: --blocks_to_swap. dp:
    # `blocks_to_swap` in TOML's [general] block (already supported via
    # backend.diffusion_pipe.blocks_to_swap; this top-level mirror lets
    # us share recipes across backends without two source-of-truth keys).
    blocks_to_swap: int = Field(0, ge=0)
    # FP8 base model weight load (FLUX / SD3 / HunyuanImage). VRAM -40%.
    fp8_base: bool = False
    fp8_base_unet: bool = False
    # HunyuanImage scaled FP8 (different math from --fp8_base).
    fp8_scaled: bool = False
    # HunyuanImage VL text encoder in FP8.
    fp8_vl_text_encoder: bool = False
    # Memory-strategy hints (kohya).
    lowram: bool = False
    highvram: bool = False
    # SDXL VAE in fp32 (the half-precision VAE corrupts colours on some
    # SDXL finetunes; this is the canonical workaround).
    no_half_vae: bool = False
    # safetensors loading without mmap (NFS / network filesystems).
    disable_mmap_load_safetensors: bool = False
    # Gradient checkpointing offloaded to CPU (kohya).
    cpu_offload_checkpointing: bool = False
    # Anima-specific unsloth-flavoured offload.
    unsloth_offload_checkpointing: bool = False
    # Cache text encoder outputs to RAM / disk (kohya). Disk version frees
    # VRAM completely and lets 6GB cards train SDXL.
    cache_text_encoder_outputs: bool = False
    cache_text_encoder_outputs_to_disk: bool = False


class OutputConfig(BaseModel):
    model_config = _CAMEL_CONFIG

    name: str = "lora_output"
    save_every_n_epochs: int = Field(1, ge=1)
    # Step-level save cadence (kohya / dp).
    save_every_n_steps: int | None = Field(default=None, ge=1)
    # dp: examples-level save cadence.
    save_every_n_examples: int | None = Field(default=None, ge=1)
    # Retain only the most recent N checkpoints (kohya).
    save_last_n_epochs: int | None = Field(default=None, ge=1)
    save_last_n_steps: int | None = Field(default=None, ge=1)
    save_dtype: Literal["fp16", "bf16", "float"] = "fp16"
    output_dir: Path | None = None
    # Free-form metadata stamped onto the LoRA file. kohya: --metadata_*.
    training_comment: str | None = None
    no_metadata: bool = False
    metadata: dict[str, str] = Field(default_factory=dict)


class MultiNodeConfig(BaseModel):
    """Multi-node DeepSpeed launcher knobs (forwarded to ``deepspeed`` CLI).

    DeepSpeed itself reads the hostfile to discover workers and rsyncs
    code. LoraHub doesn't manage the rsync — the user is responsible for
    keeping the diffusion-pipe checkout + venv on every node. The
    configured ``master_addr`` must be reachable from every worker; if
    omitted DeepSpeed picks the first hostfile entry's hostname, which
    is fine for tightly-coupled clusters.
    """

    model_config = _CAMEL_CONFIG

    # Path to the DeepSpeed-format hostfile. Each line: ``host slots=N``.
    # Resolved relative to cwd if not absolute.
    hostfile: Path
    # Total node count. DeepSpeed cross-checks against the hostfile and
    # raises if they disagree, so this is mostly a safety check + a
    # sanity gate before launch.
    num_nodes: int = Field(ge=2)
    # Optional explicit master address for rendezvous. Leave None to let
    # DeepSpeed auto-discover from the hostfile's first host.
    master_addr: str | None = None
    # Optional master port. DeepSpeed default is 29500.
    master_port: int | None = Field(default=None, ge=1024, le=65535)


class DiffusionPipeOptions(BaseModel):
    """diffusion-pipe specific knobs not represented anywhere else.

    Most dp options now live on shared sections (LossConfig.pseudo_huber_c,
    BucketConfig.ar_buckets, NetworkConfig.fuse_adapters, ...). What
    remains here is genuinely dp-only or doesn't make sense to expose
    cross-backend.
    """

    model_config = _CAMEL_CONFIG

    # ---- Top-level [general] knobs ----
    pipeline_stages: int = Field(1, ge=1)
    gradient_clipping: float = Field(1.0, gt=0)
    partition_method: Literal[
        "parameters", "uniform", "type:transformer_layer"
    ] = "parameters"
    # Manual layer-split when partition_method=manual; len = pipeline_stages-1.
    partition_split: list[int] | None = None
    caching_batch_size: int = Field(1, ge=1)
    steps_per_print: int = Field(1, ge=1)
    blocks_to_swap: int = Field(0, ge=0)
    compile: bool = False
    # Pipeline parallelism + reentrant grad ckpt requirement.
    reentrant_activation_checkpointing: bool = False
    # Skip block_swap during eval (eval uses less memory).
    disable_block_swap_for_eval: bool = False
    # Mixed image+video training.
    image_micro_batch_size_per_gpu: int | None = Field(default=None, ge=1)
    image_eval_micro_batch_size_per_gpu: int | None = Field(default=None, ge=1)
    eval_gradient_accumulation_steps: int = Field(1, ge=1)
    # Force a flat LR regardless of scheduler (resume tweak).
    force_constant_lr: float | None = Field(default=None, gt=0)
    # CFG-style training: drop captions for `uncond_fraction` of steps.
    uncond_fraction: float = Field(0.0, ge=0.0, le=1.0)
    # Tensorboard X-axis: `steps` or `examples`.
    x_axis_examples: bool = False
    logging_steps: int = Field(1, ge=1)
    # Transformer dtype (HunyuanVideo float8 etc).
    transformer_dtype: Literal["bfloat16", "float16", "float8_e4m3fn", "float8_e5m2"] | None = None
    diffusion_model_dtype: Literal["bfloat16", "float16", "float8_e4m3fn"] | None = None
    timestep_sample_method: Literal["logit_normal", "uniform"] | None = None
    # Independent eval datasets — each entry: {name, config_path}.
    eval_datasets: list[dict[str, str]] = Field(default_factory=list)
    # Video clip extraction strategy.
    video_clip_mode: Literal["single_beginning", "single_middle"] = "single_beginning"

    # ---- [eval] section ----
    eval_every_n_epochs: int | None = Field(default=None, ge=1)
    eval_every_n_steps: int | None = Field(default=None, ge=1)
    eval_every_n_examples: int | None = Field(default=None, ge=1)
    eval_before_first_step: bool = False
    eval_micro_batch_size_per_gpu: int = Field(1, ge=1)

    # ---- Checkpoint cadence (DeepSpeed state, separate from save_*) ----
    checkpoint_every_n_epochs: int | None = Field(default=None, ge=1)
    checkpoint_every_n_minutes: int | None = Field(default=None, ge=1)

    # ---- [monitoring] section ----
    enable_wandb: bool = False
    tracker_name: str | None = None
    run_name: str | None = None

    # ---- Dataset bucketing knobs (dp only) ----
    min_ar: float = Field(0.5, gt=0)
    max_ar: float = Field(2.0, gt=0)
    num_ar_buckets: int = Field(7, ge=1)
    cache_shuffle_num: int = Field(0, ge=0)  # 0 = preserve original order
    skip_empty_caption: bool = True

    # ---- Free-form per-arch path bag for the [model] section ----
    # Legacy escape hatch for arch-specific keys not represented on
    # ArchPathsConfig. Most arches should now use cfg.base_model.arch_paths
    # instead; this remains for upstream additions we haven't typed yet.
    model_paths: dict[str, str] = Field(default_factory=dict)

    # ---- Multi-node DeepSpeed launcher (B8) ----
    # When set, the dp runner forwards ``--hostfile`` / ``--num_nodes`` /
    # ``--master_addr`` to the deepspeed launcher so a job spans multiple
    # machines. Single-node training (the common case) leaves this None.
    # The hostfile is the standard DeepSpeed shape — one line per host
    # in the form ``hostname slots=N``.
    multi_node: MultiNodeConfig | None = None


class AnimaLoraMethodLoraConfig(BaseModel):
    """Defaults for ``method = "lora"`` on the anima_lora backend.

    The ``method = "lora"`` path on anima_lora is *not* the bare LoRA you'd
    get on kohya — upstream stacks OrthoLoRA + T-LoRA on top by default.
    Tracking those knobs here keeps the LoraHub schema explicit about the
    stack instead of hiding it in the compiler.
    """

    model_config = _CAMEL_CONFIG

    use_ortho: bool = True
    # T-LoRA timestep mask: high noise → low rank, low noise → full rank.
    use_timestep_mask: bool = True
    min_rank: int = Field(8, ge=1)
    alpha_rank_scale: float = Field(1.0, gt=0)


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
    seed: int = 42
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

    # ---- Network ----
    network_module: str = "networks.lora_anima"
    network_dim: int = Field(16, ge=1)
    network_alpha: float = Field(16, gt=0)
    network_train_unet_only: bool = True

    # ---- Optimizer / schedule ----
    optimizer_type: Literal["AdamW", "AdamW8bit", "Lion", "Prodigy"] = "AdamW"
    lr_scheduler: Literal[
        "constant", "cosine", "cosine_with_restarts", "linear", "polynomial"
    ] = "constant"
    learning_rate: float = Field(5.0e-5, gt=0)
    max_train_epochs: int = Field(8, ge=1)
    save_every_n_epochs: int = Field(1, ge=1)
    # Optional step-based checkpoint cadence. When set, the trainer
    # also writes a ckpt every N steps (in addition to the epoch
    # cadence above). Useful for short epochs / small datasets where
    # a full epoch is the only thing that triggers a state dir, and
    # the pause-resume workflow wants finer granularity.
    save_every_n_steps: int | None = Field(default=None, ge=1)
    checkpointing_epochs: int = Field(1, ge=1)
    caption_dropout_rate: float = Field(0.1, ge=0.0, le=1.0)

    # ---- Sampling / loss (flow-matching for Anima DiT) ----
    timestep_sampling: Literal["sigmoid", "uniform", "logit_normal"] = "sigmoid"
    sigmoid_scale: float = Field(1.0, gt=0)
    discrete_flow_shift: float = Field(1.0, gt=0)
    weighting_scheme: Literal["sigma_sqrt", "logit_normal", "mode", "cosmap"] | None = None
    logit_mean: float | None = None
    logit_std: float | None = None
    mode_scale: float | None = None
    # Variance-reduced flow-matching loss (AsymFlow §5.2). +40% step compute
    # when enabled; leave None to skip.
    vr_loss_weight: float | None = Field(default=None, ge=0)

    # ---- Caching / data ----
    cache_latents: bool = True
    cache_latents_to_disk: bool = True
    cache_text_encoder_outputs: bool = True
    cache_text_encoder_outputs_to_disk: bool = True
    cache_llm_adapter_outputs: bool = True
    use_shuffled_caption_variants: bool = True
    # Subset sampling — per-preset override (debug=0.001, tenth=0.1 etc).
    sample_ratio: float | None = Field(default=None, gt=0.0, le=1.0)
    static_token_count: int = Field(4096, ge=1)
    vae_chunk_size: int = Field(64, ge=1)
    vae_disable_cache: bool = False
    no_half_vae: bool = False

    # ---- Attention / compile ----
    # Default to ``torch`` (PyTorch SDPA) instead of upstream's ``flash``
    # because flash-attn is an optional, compute-capability-sensitive
    # build that many environments don't have. SDPA is always available
    # and on Ampere+ runs at ~85-95% of flash-attn's throughput. Users
    # with flash-attn installed flip this to ``flash`` for the last
    # 5-15% perf gain.
    attn_mode: Literal["flash", "torch", "flex", "sageattn", "xformers"] = "torch"
    xformers: bool = False
    split_attn: bool = False
    # ``compile_mode = "full"`` enables CUDAGraph capture via
    # ``compile_inductor_mode = "reduce-overhead"``. Incompatible with
    # gradient_checkpointing / blocks_to_swap.
    compile_mode: Literal["blocks", "full"] | None = None
    compile_inductor_mode: Literal[
        "default", "reduce-overhead", "max-autotune"
    ] | None = None
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

    # 🔒 LOCKED_TRUE — masked loss is part of the Anima training pipeline
    # contract; upstream's _compute_loss path branches on it without a
    # backward edge. Disabling is a silent no-op.
    masked_loss: bool = True
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
    # ⚠️ RISKY — persistent dataloader workers; upstream default is
    # false (workers re-spawn each epoch). Setting true reduces epoch
    # boundary stalls but may leak file handles on long runs.
    persistent_data_loader_workers: bool = False
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
    # ⚠️ RISKY — held-out validation set count. base.toml uses 16.
    # 0 disables CMMD val (val_loss series stops updating).
    validation_split_num: int = Field(16, ge=0)
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


class BackendConfig(BaseModel):
    model_config = _CAMEL_CONFIG

    type: Literal["kohya", "diffusion-pipe", "anima_lora"] = "kohya"
    pin_version: str | None = None
    sd_scripts_path: Path | None = None
    python_executable: Path | None = None
    extra_args: dict[str, Any] = Field(default_factory=dict)
    # Optional, dp-specific knobs. None means "use library defaults" so kohya
    # users never need to touch this field.
    diffusion_pipe: DiffusionPipeOptions | None = None
    # Optional, anima_lora-specific knobs. None means "use anima_lora's own
    # base.toml defaults so kohya / dp users never need to touch this.
    anima_lora: AnimaLoraOptions | None = None


class ResumeConfig(BaseModel):
    """Checkpoint state writing for resume support.

    When `save_state=True`, kohya writes optimizer + scheduler state next
    to the safetensors so a later run can pick up exactly where the
    interrupted one left off. State directories are large; use
    `save_state_every_n_epochs` to throttle writes if disk is tight.
    """

    model_config = _CAMEL_CONFIG

    save_state: bool = True
    save_state_at_end: bool = True
    save_state_every_n_epochs: int | None = Field(default=None, ge=1)
    # Local resume path (kohya: --resume).
    resume_from: Path | None = None
    # Retain only the most recent N state directories.
    save_last_n_epochs_state: int | None = Field(default=None, ge=1)
    save_last_n_steps_state: int | None = Field(default=None, ge=1)
    # Skip ahead to a specific step on resume (kohya).
    skip_until_initial_step: bool = False
    initial_epoch: int | None = Field(default=None, ge=1)
    initial_step: int | None = Field(default=None, ge=0)


class ValidationConfig(BaseModel):
    """Validation-loss cadence for overfit detection.

    Only takes effect when `dataset.val_split > 0`; otherwise the compiler
    skips emitting validation argv entirely. `max_samples` caps how many
    validation steps sd-scripts will run per evaluation pass — handy when
    the held-out split is large and you only want a quick signal.
    """

    model_config = _CAMEL_CONFIG

    every_n_epochs: int = Field(1, ge=1)
    every_n_steps: int | None = Field(default=None, ge=1)
    max_samples: int | None = Field(default=None, ge=1)
    seed: int | None = None


class TrainingConfig(BaseModel):
    """Top-level recipe configuration. One YAML file = one TrainingConfig."""

    schema_version: str = "1.0"
    base_model: BaseModelConfig
    dataset: DatasetConfig
    network: NetworkConfig = Field(default_factory=lambda: NetworkConfig())
    optimizer: OptimizerConfig = Field(default_factory=lambda: OptimizerConfig())
    loss: LossConfig = Field(default_factory=lambda: LossConfig())
    flow_match: FlowMatchConfig = Field(default_factory=lambda: FlowMatchConfig())
    schedule: ScheduleConfig = Field(default_factory=lambda: ScheduleConfig())
    precision: Literal["fp16", "bf16", "fp32"] = "bf16"
    gradient_checkpointing: bool = True
    cache_latents: bool = True
    cache_latents_to_disk: bool = False
    skip_cache_check: bool = False
    cache_info: bool = False
    train_inpainting: bool = False
    sampling: SamplingConfig = Field(default_factory=lambda: SamplingConfig())
    output: OutputConfig = Field(default_factory=lambda: OutputConfig())
    backend: BackendConfig = Field(default_factory=lambda: BackendConfig())
    resume: ResumeConfig = Field(default_factory=lambda: ResumeConfig())
    validation: ValidationConfig = Field(default_factory=lambda: ValidationConfig())
    attention: AttentionConfig = Field(default_factory=lambda: AttentionConfig())
    optimization: OptimizationConfig = Field(default_factory=lambda: OptimizationConfig())
    dataloader: DataLoaderConfig = Field(default_factory=lambda: DataLoaderConfig())
    augmentation: AugmentationConfig = Field(default_factory=lambda: AugmentationConfig())

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="forbid",
    )

    @model_validator(mode="after")
    def _validate_arch_variant(self) -> TrainingConfig:
        """SDXL sub-variants only make sense on the SDXL backbone."""
        if self.base_model.arch_variant and self.base_model.arch != "sdxl":
            msg = (
                "base_model.arch_variant requires base_model.arch == 'sdxl' "
                f"(got arch={self.base_model.arch!r}, "
                f"arch_variant={self.base_model.arch_variant!r})"
            )
            raise ValueError(msg)
        return self
