"""Job-management helpers shared between the jobs router and websocket layer.

Historically a single 1.4k-line module — split into five domain
submodules to keep complexity in check while preserving the
original import surface. Anything that ``from lorahub.api.jobs_helpers
import _foo`` worked against before still works through this re-export.

Submodules:

* :mod:`paths_norm` — recipe-path absolutisation
* :mod:`metrics` — workspace artifact listing + ``events.jsonl`` parsing
* :mod:`preview` — optional live-preview worker + GPU sampler thread
* :mod:`lifecycle` — job creation / re-launch / scheduler hookup
* :mod:`resume_dispatch` — per-backend resume specs + auto-resume hook
"""

from __future__ import annotations

from .lifecycle import (
    _TERMINAL_STATES,
    _archive_workspace,
    _enqueue_launch,
    _extract_ckpt_name,
    _launch_job,
    _relaunch_job_in_place,
    _select_backend,
)
from .metrics import (
    _classify_artifact,
    _compute_overfit_signal,
    _downsample,
    _empty_overfit_signal,
    _job_events,
    _list_workspace_files,
    _media_type_for,
    _read_metrics,
    _resolve_workspace_file,
    _tail_slope,
)
from .paths_norm import _absolutise, _normalize_recipe_paths
from .preview import _gpu_sampler_loop, _maybe_start_preview_worker
from .resume_dispatch import (
    ResumeNotReady,
    ResumeSpec,
    _anima_lora_resume_spec,
    _apply_cfg_overrides,
    _attempt_auto_resume,
    _dispatch_resume_spec,
    _dp_output_dir,
    _dp_resume_spec,
    _find_latest_dp_run_dir,
    _find_latest_safetensors,
    _find_latest_state_dir,
    _kohya_resume_spec,
    _migrate_snapshots_to_camel,
    _requeue_pending_jobs,
    _should_auto_resume,
)

__all__ = [
    # lifecycle
    "_TERMINAL_STATES",
    "_archive_workspace",
    "_enqueue_launch",
    "_extract_ckpt_name",
    "_launch_job",
    "_relaunch_job_in_place",
    "_select_backend",
    # metrics
    "_classify_artifact",
    "_compute_overfit_signal",
    "_downsample",
    "_empty_overfit_signal",
    "_job_events",
    "_list_workspace_files",
    "_media_type_for",
    "_read_metrics",
    "_resolve_workspace_file",
    "_tail_slope",
    # paths_norm
    "_absolutise",
    "_normalize_recipe_paths",
    # preview
    "_gpu_sampler_loop",
    "_maybe_start_preview_worker",
    # resume_dispatch
    "ResumeNotReady",
    "ResumeSpec",
    "_anima_lora_resume_spec",
    "_apply_cfg_overrides",
    "_attempt_auto_resume",
    "_dispatch_resume_spec",
    "_dp_output_dir",
    "_dp_resume_spec",
    "_find_latest_dp_run_dir",
    "_find_latest_safetensors",
    "_find_latest_state_dir",
    "_kohya_resume_spec",
    "_migrate_snapshots_to_camel",
    "_requeue_pending_jobs",
    "_should_auto_resume",
]
