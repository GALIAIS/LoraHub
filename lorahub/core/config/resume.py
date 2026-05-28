"""Resume / validation configs."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from ._shared import _CAMEL_CONFIG


class ResumeConfig(BaseModel):
    """Checkpoint state writing for resume support.

    When `save_state=True`, kohya writes optimizer + scheduler state next
    to the safetensors so a later run can pick up exactly where the
    interrupted one left off. State directories are large; use
    `save_state_every_n_epochs` to throttle writes if disk is tight.
    """

    model_config = _CAMEL_CONFIG

    save_state: bool = True
    save_state_at_end: bool = True
    save_state_every_n_epochs: int | None = Field(default=None, ge=1)
    # Local resume path (kohya: --resume).
    resume_from: Path | None = None
    # Retain only the most recent N state directories.
    save_last_n_epochs_state: int | None = Field(default=None, ge=1)
    save_last_n_steps_state: int | None = Field(default=None, ge=1)
    # Skip ahead to a specific step on resume (kohya).
    skip_until_initial_step: bool = False
    initial_epoch: int | None = Field(default=None, ge=1)
    initial_step: int | None = Field(default=None, ge=0)


class ValidationConfig(BaseModel):
    """Validation-loss cadence for overfit detection.

    Only takes effect when `dataset.val_split > 0`; otherwise the compiler
    skips emitting validation argv entirely. `max_samples` caps how many
    validation steps sd-scripts will run per evaluation pass — handy when
    the held-out split is large and you only want a quick signal.
    """

    model_config = _CAMEL_CONFIG

    every_n_epochs: int = Field(1, ge=1)
    every_n_steps: int | None = Field(default=None, ge=1)
    max_samples: int | None = Field(default=None, ge=1)
    seed: int | None = None
