"""Backend-specific options + the aggregating ``BackendConfig``.

``BackendConfig`` lives in this package's ``__init__`` because it composes
``DiffusionPipeOptions`` and ``AnimaLoraOptions`` from the sibling
modules; keeping it next to the imports keeps the dependency graph
linear (root ``__init__`` imports the aggregate, sub-modules don't
re-import each other).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, Field, model_validator

from .._shared import _CAMEL_CONFIG
from .ai_toolkit import AiToolkitOptions
from .anima_lora import (
    AnimaLoraMethodChimeraConfig,
    AnimaLoraMethodEasyControlConfig,
    AnimaLoraMethodIPAdapterConfig,
    AnimaLoraMethodLoraConfig,
    AnimaLoraMethodPostfixConfig,
    AnimaLoraOptions,
    AnimaLoraTurboConfig,
)
from .diffusion_pipe import DiffusionPipeOptions


class GpuDispatchConfig(BaseModel):
    model_config = _CAMEL_CONFIG

    # one-job-per-gpu: scheduler assigns one GPU per training job.
    # distributed: one training job owns multiple GPUs and launches
    # backend-native multi-process training where supported.
    mode: Literal["one-job-per-gpu", "distributed"] = "one-job-per-gpu"
    # None means "all scheduler slots" in distributed mode; ignored by
    # one-job-per-gpu mode.
    num_gpus: int | None = Field(default=None, ge=1)


class FsdpConfig(BaseModel):
    model_config = _CAMEL_CONFIG

    sharding_strategy: Literal["full_shard", "shard_grad_op", "no_reshard"] = (
        "full_shard"
    )
    auto_wrap_policy: Literal["size_based", "transformer", "none"] = "size_based"
    min_num_params: int = Field(default=100_000_000, ge=0)
    state_dict_type: Literal[
        "full_state_dict", "sharded_state_dict", "local_state_dict"
    ] = "full_state_dict"
    cpu_offload: bool = False


class DeepSpeedZeroConfig(BaseModel):
    model_config = _CAMEL_CONFIG

    stage: Literal[2, 3] = 2
    offload_optimizer: Literal["none", "cpu"] = "none"
    offload_param: Literal["none", "cpu"] = "none"
    overlap_comm: bool = True


class DistributedTrainingConfig(BaseModel):
    model_config = _CAMEL_CONFIG

    strategy: Literal["ddp", "fsdp", "deepspeed_zero"] = "ddp"
    fsdp: FsdpConfig = Field(default_factory=FsdpConfig)
    zero: DeepSpeedZeroConfig = Field(default_factory=DeepSpeedZeroConfig)


class BackendConfig(BaseModel):
    model_config = _CAMEL_CONFIG

    type: Literal["kohya", "diffusion-pipe", "anima_lora", "ai_toolkit"] = "kohya"
    pin_version: str | None = None
    # Generic "backend repo path". Accepts every historical key for
    # backward compatibility with YAML files written before the rename:
    #   - ``sd_scripts_path`` / ``sdScriptsPath`` (legacy names from when
    #     this only meant kohya's sd-scripts checkout)
    #   - ``repo_path`` / ``repoPath`` (current names; camelCase wins
    #     on serialization via the model's _CAMEL_CONFIG alias generator)
    # All four read into the same field; ``cfg.backend.repo_path`` is
    # the canonical access in code.
    repo_path: Path | None = Field(  # type: ignore[pydantic-alias]
        default=None,
        validation_alias=AliasChoices(
            "repo_path", "repoPath", "sd_scripts_path", "sdScriptsPath",
        ),
    )
    python_executable: Path | None = None
    extra_args: dict[str, Any] = Field(default_factory=dict)
    gpu_dispatch: GpuDispatchConfig = Field(default_factory=GpuDispatchConfig)
    distributed: DistributedTrainingConfig = Field(
        default_factory=DistributedTrainingConfig
    )
    # Optional, dp-specific knobs. None means "use library defaults" so kohya
    # users never need to touch this field.
    diffusion_pipe: DiffusionPipeOptions | None = None
    # Optional, anima_lora-specific knobs. None means "use anima_lora's own
    # base.toml defaults so kohya / dp users never need to touch this.
    anima_lora: AnimaLoraOptions | None = None
    # Optional, ai-toolkit-specific Krea2 configuration.
    ai_toolkit: AiToolkitOptions | None = None

    @model_validator(mode="after")
    def _initialize_selected_backend_options(self) -> BackendConfig:
        if self.type == "diffusion-pipe" and self.diffusion_pipe is None:
            self.diffusion_pipe = DiffusionPipeOptions()
        if self.type == "anima_lora" and self.anima_lora is None:
            self.anima_lora = AnimaLoraOptions()
        if self.type == "ai_toolkit" and self.ai_toolkit is None:
            self.ai_toolkit = AiToolkitOptions()
        return self


__all__ = [
    "AnimaLoraMethodChimeraConfig",
    "AnimaLoraMethodEasyControlConfig",
    "AnimaLoraMethodIPAdapterConfig",
    "AnimaLoraMethodLoraConfig",
    "AnimaLoraMethodPostfixConfig",
    "AnimaLoraOptions",
    "AnimaLoraTurboConfig",
    "AiToolkitOptions",
    "BackendConfig",
    "DeepSpeedZeroConfig",
    "DistributedTrainingConfig",
    "FsdpConfig",
    "GpuDispatchConfig",
    "DiffusionPipeOptions",
]
