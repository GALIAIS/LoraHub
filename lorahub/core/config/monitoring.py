"""Multi-node launcher and W&B monitoring configs."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ._shared import _CAMEL_CONFIG


class MultiNodeConfig(BaseModel):
    """Multi-node DeepSpeed launcher knobs (forwarded to ``deepspeed`` CLI).

    DeepSpeed itself reads the hostfile to discover workers and rsyncs
    code. LoraHub doesn't manage the rsync — the user is responsible for
    keeping the diffusion-pipe checkout + venv on every node. The
    configured ``master_addr`` must be reachable from every worker; if
    omitted DeepSpeed picks the first hostfile entry's hostname, which
    is fine for tightly-coupled clusters.
    """

    model_config = _CAMEL_CONFIG

    # Path to the DeepSpeed-format hostfile. Each line: ``host slots=N``.
    # Resolved relative to cwd if not absolute.
    hostfile: Path
    # Total node count. DeepSpeed cross-checks against the hostfile and
    # raises if they disagree, so this is mostly a safety check + a
    # sanity gate before launch.
    num_nodes: int = Field(ge=2)
    # Optional explicit master address for rendezvous. Leave None to let
    # DeepSpeed auto-discover from the hostfile's first host.
    master_addr: str | None = None
    # Optional master port. DeepSpeed default is 29500.
    master_port: int | None = Field(default=None, ge=1024, le=65535)


class MonitoringConfig(BaseModel):
    """Weights & Biases tracker configuration (shared by all backends).

    Strictly mirrors the wandb.ai docs: every field maps to either a
    ``wandb.init()`` keyword or a documented ``WANDB_*`` environment
    variable. Secrets (api key) live in user settings, not the config.

    The job runner translates these fields to ``WANDB_*`` env vars at
    subprocess launch so every backend (kohya / anima_lora /
    diffusion_pipe) sees a consistent run identity regardless of how
    its CLI surfaces tracker arguments.
    """

    model_config = _CAMEL_CONFIG

    # Master switch. False = no tracker is initialized; the backend's
    # log_with stays at its default (typically tensorboard).
    enable_wandb: bool = False

    # ``wandb.init(project=...)`` / ``WANDB_PROJECT``.
    # kohya & anima_lora surface this as ``--log_tracker_name``;
    # diffusion-pipe writes ``wandb_tracker_name`` into its TOML.
    project: str | None = None

    # ``wandb.init(entity=...)`` / ``WANDB_ENTITY``. User or team owner.
    entity: str | None = None

    # ``wandb.init(name=...)`` / ``WANDB_NAME``. UI display label.
    # kohya & anima_lora pass this via ``--wandb_run_name``;
    # diffusion-pipe writes ``wandb_run_name`` into its TOML.
    run_name: str | None = None

    # ``wandb.init(id=...)`` / ``WANDB_RUN_ID``. Project-unique run id;
    # required for resume policies other than ``never``.
    run_id: str | None = None

    # ``wandb.init(group=...)`` / ``WANDB_RUN_GROUP``.
    group: str | None = None

    # ``wandb.init(job_type=...)`` / ``WANDB_JOB_TYPE``.
    job_type: str | None = None

    # ``wandb.init(tags=[...])`` / ``WANDB_TAGS`` (comma-joined).
    tags: list[str] = Field(default_factory=list)

    # ``wandb.init(notes=...)`` / ``WANDB_NOTES``.
    notes: str | None = None

    # ``wandb.init(mode=...)`` / ``WANDB_MODE``. Spec values are
    # online / offline / disabled / shared.
    mode: Literal["online", "offline", "disabled", "shared"] | None = None

    # ``wandb.init(resume=...)`` / ``WANDB_RESUME``.
    resume: Literal["allow", "never", "must", "auto"] | None = None

    # ``WANDB_BASE_URL``. For self-hosted W&B Server. Empty = SaaS.
    base_url: str | None = None
