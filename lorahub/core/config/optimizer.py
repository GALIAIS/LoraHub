"""Optimizer, LR, and schedule configs."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ._shared import _CAMEL_CONFIG


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
    # (already on DiffusionPipeOptions, this top-level mirror lets configs
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
