"""Shared training/validation context dataclasses.

Frozen bundles built once at the top of ``train()`` and threaded through
per-step / per-batch methods on the trainer plus the loop runner in
:mod:`library.training.loop`. Lives here (rather than in ``train.py``) so
``loop.py`` and any future trainer entrypoints can import them directly
instead of receiving them as injected class parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from accelerate import Accelerator


@dataclass(frozen=True)
class TrainCtx:
    """Training-wide state built once near the top of ``train()`` and passed to
    per-step / per-batch methods instead of 15-arg parameter lists. Fields here
    are fixed for the whole training run -- per-call values (epoch, global_step,
    progress_bar, logging keys, …) stay explicit at call sites."""

    args: Any
    accelerator: Accelerator
    network: Any
    unet: Any
    vae: Any
    text_encoders: list
    noise_scheduler: Any
    text_encoding_strategy: Any
    tokenize_strategy: Any
    vae_dtype: torch.dtype
    weight_dtype: torch.dtype
    train_text_encoder: bool
    train_unet: bool
    optimizer_eval_fn: Callable
    optimizer_train_fn: Callable
    is_tracking: bool


@dataclass(frozen=True)
class ValCtx:
    """Validation-wide state fixed for the entire training run. The per-call
    val_loss_recorder (step vs epoch) stays explicit since it differs per call
    site; everything else here is shared."""

    dataloader: Any
    sigmas: list
    steps: int
    total_steps: int
    train_loss_recorder: Any
    original_t_min: float
    original_t_max: float
    # The val DatasetGroup itself. Held so CMMD-style validation can enumerate
    # held-out items (absolute_path, caption, bucket_reso, text_encoder_outputs_npz)
    # for paired sample generation against the cached PE reference pool.
    dataset_group: Any = None
