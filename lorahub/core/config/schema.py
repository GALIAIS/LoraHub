"""Semantic recipe schema for LoRA training.

Users write a single YAML file describing *what* they want to train. The schema
validates it, fills defaults (tuned for 8GB VRAM on SDXL), and later a
backend-specific compiler translates it into the backend's native arguments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class BaseModelConfig(BaseModel):
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


class BucketConfig(BaseModel):
    enabled: bool = True
    min_size: int = Field(256, alias="min")
    max_size: int = Field(2048, alias="max")
    step: int = 64

    model_config = {"populate_by_name": True}


class CaptionConfig(BaseModel):
    strategy: Literal["tag_file", "filename", "none"] = "tag_file"
    ext: str = ".txt"
    shuffle: bool = True
    drop_rate: float = Field(0.0, ge=0.0, le=1.0)


class DatasetConfig(BaseModel):
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

    @model_validator(mode="after")
    def _validate_resolution(self) -> DatasetConfig:
        if len(self.resolution) not in (1, 2):
            msg = "resolution must be [size] or [width, height]"
            raise ValueError(msg)
        return self


class NetworkConfig(BaseModel):
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
        return self


class LRConfig(BaseModel):
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

    type: str = "adamw8bit"
    lr: LRConfig = Field(default_factory=lambda: LRConfig())
    schedule: str = "cosine_with_restarts"
    warmup_steps: int = 100
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = Field(0.0, ge=0.0)
    eps: float = Field(1e-8, gt=0)
    optimizer_args: dict[str, str] = Field(default_factory=dict)


class LossConfig(BaseModel):
    """Loss-shaping hyperparameters for diffusion training.

    Currently only the kohya backend consumes this section; the diffusion-pipe
    backend ignores it because its loss settings live under that backend's own
    `[model]` toml block and are not generally portable. None-valued fields are
    omitted from kohya argv entirely so sd-scripts falls back to its defaults.
    """

    min_snr_gamma: float | None = Field(default=None, gt=0)
    noise_offset: float = Field(0.0, ge=0)
    ip_noise_gamma: float | None = Field(default=None, gt=0)
    prior_loss_weight: float = Field(1.0, ge=0)
    loss_type: Literal["l2", "huber", "smooth_l1"] = "l2"
    debiased_estimation: bool = False
    masked_loss: bool = False
    scale_v_pred_loss_like_noise_pred: bool = False
    v_parameterization: bool = False


class ScheduleConfig(BaseModel):
    epochs: int = Field(10, ge=1)
    batch_size: int = Field(1, ge=1)
    grad_accum: int = Field(2, ge=1)
    max_steps: int | None = None


class SamplingConfig(BaseModel):
    enabled: bool = True
    every_n_epochs: int = Field(1, ge=1)
    prompts_file: Path | None = None
    resolution: list[int] = Field(default_factory=lambda: [1024, 1024])
    seed: int = 42


class OutputConfig(BaseModel):
    name: str = "lora_output"
    save_every_n_epochs: int = Field(1, ge=1)
    save_dtype: Literal["fp16", "bf16", "float"] = "fp16"
    output_dir: Path | None = None


class DiffusionPipeOptions(BaseModel):
    """diffusion-pipe specific knobs.

    Only consumed when ``backend.type == "diffusion-pipe"``; other backends
    (kohya) ignore this field entirely. Defaults reproduce the previous
    hard-coded values in the dp compiler so existing recipes keep producing
    the same TOML.
    """

    # ---- Top-level [general] knobs ----
    pipeline_stages: int = Field(1, ge=1)
    gradient_clipping: float = Field(1.0, gt=0)
    partition_method: Literal[
        "parameters", "uniform", "type:transformer_layer"
    ] = "parameters"
    caching_batch_size: int = Field(1, ge=1)
    steps_per_print: int = Field(1, ge=1)
    blocks_to_swap: int = Field(0, ge=0)
    compile: bool = False

    # ---- [eval] section ----
    eval_every_n_epochs: int | None = Field(default=None, ge=1)
    eval_before_first_step: bool = False
    eval_micro_batch_size_per_gpu: int = Field(1, ge=1)

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
    # diffusion-pipe's [model] block accepts arch-specific path keys like
    # `transformer_path`, `vae_path`, `llm_path`, `clip_l_path`, `t5_path`,
    # etc. The set varies by arch (Anima needs transformer/vae/llm; Hunyuan
    # Video needs transformer/vae/llm/clip; ...) and upstream keeps adding
    # new entries. Rather than encode every variant in pydantic, we accept
    # an opaque `key=value` mapping that the compiler flattens into the
    # `[model]` section verbatim. Empty by default so existing SDXL/Flux/SD3
    # recipes keep producing identical TOML.
    model_paths: dict[str, str] = Field(default_factory=dict)


class BackendConfig(BaseModel):
    type: Literal["kohya", "diffusion-pipe"] = "kohya"
    pin_version: str | None = None
    sd_scripts_path: Path | None = None
    python_executable: Path | None = None
    extra_args: dict[str, Any] = Field(default_factory=dict)
    # Optional, dp-specific knobs. None means "use library defaults" so kohya
    # users never need to touch this field.
    diffusion_pipe: DiffusionPipeOptions | None = None


class ResumeConfig(BaseModel):
    """Checkpoint state writing for resume support.

    When `save_state=True`, kohya writes optimizer + scheduler state next
    to the safetensors so a later run can pick up exactly where the
    interrupted one left off. State directories are large; use
    `save_state_every_n_epochs` to throttle writes if disk is tight.
    """

    save_state: bool = True
    save_state_at_end: bool = True
    save_state_every_n_epochs: int | None = Field(default=None, ge=1)


class ValidationConfig(BaseModel):
    """Validation-loss cadence for overfit detection.

    Only takes effect when `dataset.val_split > 0`; otherwise the compiler
    skips emitting validation argv entirely. `max_samples` caps how many
    validation steps sd-scripts will run per evaluation pass — handy when
    the held-out split is large and you only want a quick signal.
    """

    every_n_epochs: int = Field(1, ge=1)
    max_samples: int | None = Field(default=None, ge=1)


class TrainingConfig(BaseModel):
    """Top-level recipe configuration. One YAML file = one TrainingConfig."""

    schema_version: str = "1.0"
    base_model: BaseModelConfig
    dataset: DatasetConfig
    network: NetworkConfig = Field(default_factory=lambda: NetworkConfig())
    optimizer: OptimizerConfig = Field(default_factory=lambda: OptimizerConfig())
    loss: LossConfig = Field(default_factory=lambda: LossConfig())
    schedule: ScheduleConfig = Field(default_factory=lambda: ScheduleConfig())
    precision: Literal["fp16", "bf16", "fp32"] = "bf16"
    gradient_checkpointing: bool = True
    cache_latents: bool = True
    sampling: SamplingConfig = Field(default_factory=lambda: SamplingConfig())
    output: OutputConfig = Field(default_factory=lambda: OutputConfig())
    backend: BackendConfig = Field(default_factory=lambda: BackendConfig())
    resume: ResumeConfig = Field(default_factory=lambda: ResumeConfig())
    validation: ValidationConfig = Field(default_factory=lambda: ValidationConfig())

    model_config = {"extra": "forbid"}

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
