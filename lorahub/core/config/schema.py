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
    arch: Literal["sd15", "sdxl", "flux", "sd3"] = "sdxl"
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


class LRConfig(BaseModel):
    unet: float = 1.0e-4
    text_encoder: float = 5.0e-5


class OptimizerConfig(BaseModel):
    type: str = "adamw8bit"
    lr: LRConfig = Field(default_factory=lambda: LRConfig())
    schedule: str = "cosine_with_restarts"
    warmup_steps: int = 100


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


class BackendConfig(BaseModel):
    type: Literal["kohya", "diffusion-pipe"] = "kohya"
    pin_version: str | None = None
    sd_scripts_path: Path | None = None
    python_executable: Path | None = None
    extra_args: dict[str, Any] = Field(default_factory=dict)


class RecipeConfig(BaseModel):
    """Top-level recipe configuration. One YAML file = one RecipeConfig."""

    schema_version: str = "1.0"
    base_model: BaseModelConfig
    dataset: DatasetConfig
    network: NetworkConfig = Field(default_factory=lambda: NetworkConfig())
    optimizer: OptimizerConfig = Field(default_factory=lambda: OptimizerConfig())
    schedule: ScheduleConfig = Field(default_factory=lambda: ScheduleConfig())
    precision: Literal["fp16", "bf16", "fp32"] = "bf16"
    gradient_checkpointing: bool = True
    cache_latents: bool = True
    sampling: SamplingConfig = Field(default_factory=lambda: SamplingConfig())
    output: OutputConfig = Field(default_factory=lambda: OutputConfig())
    backend: BackendConfig = Field(default_factory=lambda: BackendConfig())

    model_config = {"extra": "forbid"}
