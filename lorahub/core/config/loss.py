"""Loss-shaping and flow-matching hyperparameters."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ._shared import _CAMEL_CONFIG


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
