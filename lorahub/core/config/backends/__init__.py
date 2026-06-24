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

from pydantic import AliasChoices, BaseModel, Field

from .._shared import _CAMEL_CONFIG
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


class BackendConfig(BaseModel):
    model_config = _CAMEL_CONFIG

    type: Literal["kohya", "diffusion-pipe", "anima_lora"] = "kohya"
    pin_version: str | None = None
    # Generic "backend repo path". Accepts every historical key for
    # backward compatibility with YAML files written before the rename:
    #   - ``sd_scripts_path`` / ``sdScriptsPath`` (legacy names from when
    #     this only meant kohya's sd-scripts checkout)
    #   - ``repo_path`` / ``repoPath`` (current names; camelCase wins
    #     on serialization via the model's _CAMEL_CONFIG alias generator)
    # All four read into the same field; ``cfg.backend.repo_path`` is
    # the canonical access in code.
    repo_path: Path | None = Field(
        default=None,
        validation_alias=AliasChoices(
            "repo_path", "repoPath", "sd_scripts_path", "sdScriptsPath",
        ),
    )
    python_executable: Path | None = None
    extra_args: dict[str, Any] = Field(default_factory=dict)
    gpu_dispatch: GpuDispatchConfig = Field(default_factory=GpuDispatchConfig)
    # Optional, dp-specific knobs. None means "use library defaults" so kohya
    # users never need to touch this field.
    diffusion_pipe: DiffusionPipeOptions | None = None
    # Optional, anima_lora-specific knobs. None means "use anima_lora's own
    # base.toml defaults so kohya / dp users never need to touch this.
    anima_lora: AnimaLoraOptions | None = None


__all__ = [
    "AnimaLoraMethodChimeraConfig",
    "AnimaLoraMethodEasyControlConfig",
    "AnimaLoraMethodIPAdapterConfig",
    "AnimaLoraMethodLoraConfig",
    "AnimaLoraMethodPostfixConfig",
    "AnimaLoraOptions",
    "AnimaLoraTurboConfig",
    "BackendConfig",
    "GpuDispatchConfig",
    "DiffusionPipeOptions",
]
