"""Translate ``MonitoringConfig`` into ``WANDB_*`` environment variables.

Strict 1:1 mapping against the wandb.ai env-var docs:

  - WANDB_PROJECT     <- monitoring.project
  - WANDB_ENTITY      <- monitoring.entity
  - WANDB_NAME        <- monitoring.run_name
  - WANDB_RUN_ID      <- monitoring.run_id
  - WANDB_RUN_GROUP   <- monitoring.group
  - WANDB_JOB_TYPE    <- monitoring.job_type
  - WANDB_TAGS        <- monitoring.tags (comma-joined)
  - WANDB_NOTES       <- monitoring.notes
  - WANDB_MODE        <- monitoring.mode
  - WANDB_RESUME      <- monitoring.resume
  - WANDB_BASE_URL    <- monitoring.base_url

This is the *bottom-tier* integration point: every backend's wandb
client (whether invoked through accelerate, raw wandb.init, or
diffusion-pipe's own bootstrap) reads these env vars at startup.
Per-backend CLI/TOML transports (``--log_tracker_name``,
``wandb_tracker_name``, ``--wandb_run_name``) layer on top to keep
configs self-describing — but the env var path is the safety net that
guarantees wandb sees the same identity even when a backend forgets
to surface a particular field.

Secrets (api key) live in ``Settings.wandb_api_key`` and are injected
by ``lorahub.api.settings.env_overrides``; they are intentionally not
in ``MonitoringConfig`` so config YAML never carries the key.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lorahub.core.config.schema import MonitoringConfig, TrainingConfig


def _resolve_monitoring(cfg: TrainingConfig) -> MonitoringConfig:
    """Resolve the effective monitoring config.

    Top-level ``cfg.monitoring`` always wins. The legacy
    ``cfg.backend.diffusion_pipe.{enable_wandb,tracker_name,run_name}``
    fields are honored only when the top-level block is at its
    untouched default — this keeps pre-MonitoringConfig configs from
    silently losing wandb when they're loaded against the new schema.
    """
    from lorahub.core.config.schema import MonitoringConfig  # noqa: PLC0415

    monitoring = cfg.monitoring
    is_default_top_level = (
        not monitoring.enable_wandb
        and monitoring.project is None
        and monitoring.run_name is None
        and monitoring.entity is None
        and monitoring.run_id is None
    )
    # ``cfg.backend.diffusion_pipe`` is optional — kohya / anima_lora
    # configs leave the dp options block at None. Skip the legacy
    # fallback path entirely in that case.
    dp = getattr(cfg.backend, "diffusion_pipe", None) if cfg.backend else None
    if is_default_top_level and dp is not None and dp.enable_wandb:
        return MonitoringConfig(
            enable_wandb=True,
            project=dp.tracker_name,
            run_name=dp.run_name,
        )
    return monitoring


def wandb_env(cfg: TrainingConfig) -> dict[str, str]:
    """Build the ``WANDB_*`` env-var overrides for one training job.

    Returns an empty dict when monitoring is disabled, so the caller
    can ``env.update(wandb_env(cfg))`` unconditionally without
    polluting the subprocess env on tracker-disabled runs.
    """
    monitoring = _resolve_monitoring(cfg)
    if not monitoring.enable_wandb:
        return {}

    out: dict[str, str] = {}
    if monitoring.project:
        out["WANDB_PROJECT"] = monitoring.project
    if monitoring.entity:
        out["WANDB_ENTITY"] = monitoring.entity
    if monitoring.run_name:
        out["WANDB_NAME"] = monitoring.run_name
    if monitoring.run_id:
        out["WANDB_RUN_ID"] = monitoring.run_id
    if monitoring.group:
        out["WANDB_RUN_GROUP"] = monitoring.group
    if monitoring.job_type:
        out["WANDB_JOB_TYPE"] = monitoring.job_type
    if monitoring.tags:
        # WANDB_TAGS is documented as a comma-separated list.
        out["WANDB_TAGS"] = ",".join(monitoring.tags)
    if monitoring.notes:
        out["WANDB_NOTES"] = monitoring.notes
    if monitoring.mode:
        out["WANDB_MODE"] = monitoring.mode
    if monitoring.resume:
        out["WANDB_RESUME"] = monitoring.resume
    if monitoring.base_url:
        # Self-hosted W&B Server target.
        out["WANDB_BASE_URL"] = monitoring.base_url
    return out


__all__ = ["wandb_env"]
