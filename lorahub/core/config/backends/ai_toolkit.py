"""ai-toolkit options used by the vendored Krea2 training backend."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .._shared import _CAMEL_CONFIG


class AiToolkitModelOptions(BaseModel):
    model_config = _CAMEL_CONFIG

    quantize: bool = True
    qtype: Literal["qfloat8", "float8", "int8", "uint8", "uint4"] = "qfloat8"
    quantize_text_encoder: bool = True
    qtype_text_encoder: Literal["qfloat8", "qint8", "qint4"] = "qfloat8"
    low_vram: bool = False
    layer_offloading: bool = False
    layer_offloading_transformer_percent: float = Field(1.0, ge=0.0, le=1.0)
    layer_offloading_text_encoder_percent: float = Field(1.0, ge=0.0, le=1.0)
    assistant_lora_path: str | None = None
    checkpoint_filename: str | None = None
    vae_path: str | None = None
    text_encoder_path: str | None = None
    max_text_length: int = Field(512, ge=1, le=4096)
    compile: bool | None = None
    block_compile: bool = False
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "default"
    compile_fullgraph: bool = False
    compile_dynamic: bool = True
    cache_size_limit: int | None = Field(default=None, ge=1)


class AiToolkitDatasetOptions(BaseModel):
    model_config = _CAMEL_CONFIG

    resolutions: list[int] | None = None
    buckets: bool = True
    random_crop: bool = False
    random_scale: bool = False
    scale: float = Field(1.0, gt=0)
    flip_x: bool = False
    flip_y: bool = False
    shuffle_tokens: bool = False
    token_dropout_rate: float = Field(0.0, ge=0.0, le=1.0)
    keep_tokens: int = Field(0, ge=0)
    cache_latents: bool = False
    cache_text_embeddings: bool = False
    load_image_when_caching_latents: bool = False
    num_workers: int = Field(2, ge=0)
    prefetch_factor: int = Field(2, ge=1)
    default_caption: str | None = None
    trigger_word: str | None = None

    @field_validator("resolutions")
    @classmethod
    def _validate_resolutions(cls, value: list[int] | None) -> list[int] | None:
        if value is not None and (not value or any(item < 64 for item in value)):
            raise ValueError("ai_toolkit dataset resolutions must be at least 64")
        return list(dict.fromkeys(value)) if value is not None else None


class AiToolkitNetworkOptions(BaseModel):
    model_config = _CAMEL_CONFIG

    lokr_factor: int = -1
    lokr_full_rank: bool = False
    old_lokr_format: bool = False
    lorm_extract_mode: Literal["ratio", "fixed"] = "fixed"
    lorm_extract_mode_param: float | None = Field(default=None, gt=0)
    lorm_parameter_threshold: int = Field(0, ge=0)

    @field_validator("lokr_factor")
    @classmethod
    def _validate_lokr_factor(cls, value: int) -> int:
        if value != -1 and value < 2:
            raise ValueError("lokr_factor must be -1 or at least 2")
        return value


class AiToolkitTrainOptions(BaseModel):
    model_config = _CAMEL_CONFIG

    lr_scheduler: Literal[
        "constant",
        "constant_with_warmup",
        "linear",
        "cosine",
        "cosine_with_restarts",
    ] = "constant"
    content_or_style: Literal["balanced", "style", "content"] = "balanced"
    timestep_type: Literal[
        "sigmoid",
        "linear",
        "lognorm_blend",
        "next_sample",
        "weighted",
        "one_step",
        "two_step",
        "four_step",
        "eight_step",
    ] = "sigmoid"
    loss_type: Literal[
        "mse", "mae", "wavelet", "mean_flow", "pseudo_huber"
    ] = "mse"
    min_denoising_steps: int = Field(0, ge=0)
    max_denoising_steps: int = Field(999, ge=0)
    min_snr_gamma: float | None = Field(default=None, gt=0)
    noise_offset: float = Field(0.0, ge=0)
    prompt_dropout_prob: float = Field(0.0, ge=0.0, le=1.0)
    skip_first_sample: bool = False
    force_first_sample: bool = False
    unload_text_encoder: bool = False
    use_ema: bool = False
    ema_decay: float = Field(0.999, gt=0.0, lt=1.0)
    ema_use_feedback: bool = False
    ema_param_multiplier: float = Field(1.0, gt=0)
    max_loss: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _validate_denoising_range(self) -> AiToolkitTrainOptions:
        if self.min_denoising_steps > self.max_denoising_steps:
            raise ValueError("min_denoising_steps cannot exceed max_denoising_steps")
        return self


class AiToolkitSampleOptions(BaseModel):
    model_config = _CAMEL_CONFIG

    format: Literal["jpg", "png", "webp"] = "jpg"
    walk_seed: bool = False
    network_multiplier: float = 1.0


class AiToolkitSaveOptions(BaseModel):
    model_config = _CAMEL_CONFIG

    push_to_hub: bool = False
    hf_repo_id: str | None = None
    hf_private: bool = False

    @model_validator(mode="after")
    def _validate_hub_target(self) -> AiToolkitSaveOptions:
        if self.push_to_hub and not (self.hf_repo_id or "").strip():
            raise ValueError("hf_repo_id is required when push_to_hub is enabled")
        return self


class AiToolkitLoggingOptions(BaseModel):
    model_config = _CAMEL_CONFIG

    log_every: int = Field(1, ge=1)
    verbose: bool = False
    use_wandb: bool = False
    project_name: str = "lorahub"
    run_name: str | None = None


class AiToolkitOptions(BaseModel):
    model_config = _CAMEL_CONFIG

    model: AiToolkitModelOptions = Field(default_factory=AiToolkitModelOptions)
    dataset: AiToolkitDatasetOptions = Field(default_factory=AiToolkitDatasetOptions)
    network: AiToolkitNetworkOptions = Field(default_factory=AiToolkitNetworkOptions)
    train: AiToolkitTrainOptions = Field(default_factory=AiToolkitTrainOptions)
    sample: AiToolkitSampleOptions = Field(default_factory=AiToolkitSampleOptions)
    save: AiToolkitSaveOptions = Field(default_factory=AiToolkitSaveOptions)
    logging: AiToolkitLoggingOptions = Field(default_factory=AiToolkitLoggingOptions)


__all__ = [
    "AiToolkitDatasetOptions",
    "AiToolkitLoggingOptions",
    "AiToolkitModelOptions",
    "AiToolkitNetworkOptions",
    "AiToolkitOptions",
    "AiToolkitSampleOptions",
    "AiToolkitSaveOptions",
    "AiToolkitTrainOptions",
]
