"""Semantic config schema for LoRA training.

Re-exported from this package's submodules — keep schema.py as a thin alias.

Users write a single YAML file describing *what* they want to train. The schema
validates it, fills defaults (tuned for 8GB VRAM on SDXL), and later a
backend-specific compiler translates it into the backend's native arguments.

The schema deliberately mirrors the union of kohya-ss/sd-scripts argv and
tdrussell/diffusion-pipe TOML keys. Each backend's compiler emits whichever
fields its upstream understands and silently ignores the rest, so the same
config can be retargeted between backends with minimal edits.

Field consumption is split across three layers, which can be confusing when
spelunking a single field "is anyone reading this?":

  1. **Compiler-level fields** (``schedule.epochs``, ``optimizer.lr.unet``,
     ``loss.min_snr_gamma``, ...). Read by ``lorahub.core.backends.<kohya|
     diffusion_pipe>.compiler`` and emitted to argv / TOML. Most fields
     live here.

  2. **Runtime-level fields** (``sampling.enable_live_inference``,
     ``sampling.inference_steps`` / ``inference_cfg``,
     ``backend.repo_path``, ``backend.python_executable``).
     The compiler doesn't read these; they're consumed by
     ``lorahub.api.jobs_helpers`` (live preview worker).

     ``backend.pin_version`` is **schema-only today** — kept for
     YAML round-trip compatibility but the bootstrap installer
     ignores it. If you want to lock to a specific git ref, do
     it manually via ``cd <repo> && git checkout <sha>``.

  3. **UI-only fields** (``schema_version``, ``dataset.caption.strategy``).
     Frontend form gates UI controls on these; backends ignore them.
     Kept in the schema so YAML files round-trip cleanly through the UI.

When a backend declares a field unsupported, it should ``_track`` it via
the compiler's `dropped` audit list rather than failing — so a config
written for kohya can still validate (and partially compile) under
diffusion-pipe.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.alias_generators import to_camel

from ._shared import _CAMEL_CONFIG
from .backends import (
    AiToolkitOptions,
    AnimaLoraMethodChimeraConfig,
    AnimaLoraMethodEasyControlConfig,
    AnimaLoraMethodIPAdapterConfig,
    AnimaLoraMethodLoraConfig,
    AnimaLoraMethodPostfixConfig,
    AnimaLoraOptions,
    AnimaLoraTurboConfig,
    BackendConfig,
    DiffusionPipeOptions,
    GpuDispatchConfig,
)
from .base import ArchPathsConfig, BaseModelConfig
from .dataset import BucketConfig, CaptionConfig, DatasetConfig, DatasetSubsetConfig
from .loss import FlowMatchConfig, LossConfig
from .monitoring import MonitoringConfig, MultiNodeConfig
from .network import NetworkConfig, PerModuleLRConfig
from .optimizer import LRConfig, OptimizerConfig, ScheduleConfig
from .resume import ResumeConfig, ValidationConfig
from .runtime import (
    AttentionConfig,
    AugmentationConfig,
    DataLoaderConfig,
    OptimizationConfig,
    OutputConfig,
)
from .sampling import PromptSpec, SamplingConfig, SamplingOutputs


class TrainingConfig(BaseModel):
    """Top-level config configuration. One YAML file = one TrainingConfig."""

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
    monitoring: MonitoringConfig = Field(default_factory=lambda: MonitoringConfig())

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


__all__ = [
    "AiToolkitOptions",
    "AnimaLoraMethodChimeraConfig",
    "AnimaLoraMethodEasyControlConfig",
    "AnimaLoraMethodIPAdapterConfig",
    "AnimaLoraMethodLoraConfig",
    "AnimaLoraMethodPostfixConfig",
    "AnimaLoraOptions",
    "AnimaLoraTurboConfig",
    "ArchPathsConfig",
    "AttentionConfig",
    "AugmentationConfig",
    "BackendConfig",
    "GpuDispatchConfig",
    "BaseModelConfig",
    "BucketConfig",
    "CaptionConfig",
    "DataLoaderConfig",
    "DatasetConfig",
    "DatasetSubsetConfig",
    "DiffusionPipeOptions",
    "FlowMatchConfig",
    "LRConfig",
    "LossConfig",
    "MonitoringConfig",
    "MultiNodeConfig",
    "NetworkConfig",
    "OptimizationConfig",
    "OptimizerConfig",
    "OutputConfig",
    "PerModuleLRConfig",
    "PromptSpec",
    "ResumeConfig",
    "SamplingConfig",
    "SamplingOutputs",
    "ScheduleConfig",
    "TrainingConfig",
    "ValidationConfig",
    "_CAMEL_CONFIG",
]
