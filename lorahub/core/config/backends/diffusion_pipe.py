"""diffusion-pipe specific options."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from .._shared import _CAMEL_CONFIG
from ..monitoring import MultiNodeConfig


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
        "parameters", "uniform", "manual", "type:transformer_layer"
    ] = "parameters"
    # Manual layer-split when partition_method=manual; len = pipeline_stages-1.
    partition_split: list[int] | None = None

    @field_validator("partition_split")
    @classmethod
    def _validate_partition_split(cls, value: list[int] | None) -> list[int] | None:
        if value is None:
            return None
        if any(item < 1 for item in value):
            raise ValueError("partition_split values must be positive")
        if any(
            right <= left for left, right in zip(value, value[1:], strict=False)
        ):
            raise ValueError("partition_split values must be strictly increasing")
        return value
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
    video_clip_mode: Literal[
        "single_beginning", "single_middle", "multiple_overlapping"
    ] = "single_beginning"

    # ---- [eval] section ----
    eval_every_n_epochs: int | None = Field(default=None, ge=1)
    eval_every_n_steps: int | None = Field(default=None, ge=1)
    eval_every_n_examples: int | None = Field(default=None, ge=1)
    eval_before_first_step: bool = False
    eval_micro_batch_size_per_gpu: int = Field(1, ge=1)

    # ---- Checkpoint cadence (DeepSpeed state, separate from save_*) ----
    checkpoint_every_n_epochs: int | None = Field(default=None, ge=1)
    checkpoint_every_n_minutes: int | None = Field(default=None, ge=1)

    # ---- [monitoring] section (DEPRECATED — see top-level monitoring) ----
    # Kept for back-compat with configs saved before MonitoringConfig was
    # promoted to a top-level section. Top-level ``monitoring.*`` wins
    # when present; these fields are only consulted as a fallback.
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
