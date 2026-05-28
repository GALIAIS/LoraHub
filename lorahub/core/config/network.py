"""Network (LoRA / LyCORIS) configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from ._shared import _CAMEL_CONFIG


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
    # default to 0 / None so existing configs keep emitting identical argv.
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
