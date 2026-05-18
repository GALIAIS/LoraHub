"""Job CRUD plus rerun / reveal / archive."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from lorahub.api import state
from lorahub.api.jobs_helpers import (
    _TERMINAL_STATES,
    _apply_cfg_overrides,
    _archive_workspace,
    _dispatch_resume_spec,
    _job_events,
    _launch_job,
    _list_workspace_files,
    _media_type_for,
    _read_metrics,
    _relaunch_job_in_place,
    _resolve_workspace_file,
    ResumeNotReady,
)
from lorahub.api.state import JobState
from lorahub.api.store import _pid_alive
from lorahub.core.config.schema import TrainingConfig

router = APIRouter(prefix="/api")

_RESUMABLE_STATES = (
    JobState.interrupted,
    JobState.failed,
    JobState.canceled,
)


class CreateJobRequest(BaseModel):
    config: dict[str, Any]
    workspace: str | None = None


@router.get("/jobs")
def list_jobs() -> dict[str, Any]:
    return {"jobs": [j.to_summary() for j in state.registry.list()]}


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    job = state.registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    # Detail view bundles the config snapshot so the UI can derive
    # things like expected total step count without a second round-trip.
    return {**job.to_summary(), "config_snapshot": job.config_snapshot}


@router.get("/jobs/{job_id}/events")
def get_recent_events(job_id: str, limit: int = 100) -> dict[str, Any]:
    job = state.registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    events = _job_events(job, limit)
    return {"events": [e.to_dict() for e in events]}


@router.post("/jobs", status_code=202)
def create_job(req: CreateJobRequest) -> dict[str, Any]:
    try:
        cfg = TrainingConfig.model_validate(req.config)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=str(e)) from e

    workspace = Path(req.workspace).resolve() if req.workspace else (
        Path.cwd() / "runs" / cfg.output.name
    ).resolve()
    return _launch_job(cfg, workspace)


@router.post("/jobs/{job_id}/rerun", status_code=202)
def rerun_job(job_id: str) -> dict[str, Any]:
    """Re-launch the existing job in place using its config snapshot.

    Resets state to ``queued`` and re-submits the same JobRecord+workspace
    to the scheduler. The id and event log are preserved so the user keeps
    a single timeline per logical job. Refuses to clobber a job that's
    still active (queued / running / canceling) — cancel first.
    """
    job = state.registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.state not in _TERMINAL_STATES:
        raise HTTPException(
            status_code=409,
            detail=f"job is {job.state.value}; cancel before rerunning",
        )

    try:
        cfg = TrainingConfig.model_validate(job.config_snapshot)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=422, detail=f"config snapshot is no longer valid: {exc}"
        ) from exc

    return _relaunch_job_in_place(
        job,
        cfg,
        metadata_patch={"last_rerun_at": datetime.now(UTC).isoformat()},
    )


@router.post("/jobs/{job_id}/resume", status_code=202)
def resume_job(job_id: str) -> dict[str, Any]:
    """Resume an interrupted/failed/canceled job from its last checkpoint.

    Backend-aware: kohya jobs are resumed via `--resume=<state_dir>` plus
    `--network_weights=<latest.safetensors>`; diffusion-pipe jobs via
    `--resume_from_checkpoint=<run_dir_basename>` with `output.output_dir`
    pinned to the original run's output dir so dp's checkpoint discovery
    finds the same `global_step*` folders.

    Re-launches the SAME JobRecord in place (id and workspace preserved)
    so a job's history stays as one timeline regardless of how many times
    it gets resumed.

    Errors:
      404 — original job id not found
      409 — original is not in a resumable state, or the backend reports
            no resumable artifacts on disk yet (no kohya state dir / no
            dp run_dir / no `latest` file)
      422 — config snapshot no longer matches the current schema
    """
    original = state.registry.get(job_id)
    if original is None:
        raise HTTPException(status_code=404, detail="job not found")
    if original.state not in _RESUMABLE_STATES:
        raise HTTPException(
            status_code=409,
            detail=(
                f"job state {original.state.value!r} is not resumable; "
                f"expected one of {[s.value for s in _RESUMABLE_STATES]}"
            ),
        )

    try:
        cfg = TrainingConfig.model_validate(original.config_snapshot)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        spec = _dispatch_resume_spec(cfg, original.workspace)
    except ResumeNotReady as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    cfg = _apply_cfg_overrides(cfg, spec.cfg_overrides)

    return _relaunch_job_in_place(
        original,
        cfg,
        extra_argv=spec.extra_argv,
        metadata_patch={"last_resumed_at": datetime.now(UTC).isoformat()},
    )


@router.post("/jobs/{job_id}/reveal")
def reveal_job(job_id: str) -> dict[str, Any]:
    """Open the job's workspace directory in the *host's* file browser.

    Local-first tool: the API process is on the user's machine, so we shell out
    to the platform's native file manager (`explorer`, `open`, `xdg-open`).
    Always uses an argv list — never `shell=True` — to avoid command injection
    via the workspace path.

    On a headless server (no DISPLAY / WAYLAND_DISPLAY on Linux, e.g. AutoDL,
    SSH-only VPS) opening a desktop app would either error out or pop a
    file manager on the *server* the user can't see. We detect this up
    front and return 409 with the resolved path so the frontend can offer
    a useful fallback (copy path / download zip) instead of crashing 500.
    """
    import os  # noqa: PLC0415
    import subprocess  # noqa: PLC0415
    import sys  # noqa: PLC0415

    job = state.registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")

    workspace = job.workspace
    if not workspace.exists():
        raise HTTPException(
            status_code=409, detail=f"workspace no longer exists: {workspace}"
        )

    workspace_str = str(workspace.resolve())

    # Headless Linux short-circuit. macOS / Windows always have a desktop;
    # only Linux can be in this state.
    if sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        raise HTTPException(
            status_code=409,
            detail={
                "code": "headless_host",
                "message": (
                    "API server is running headless (no DISPLAY); "
                    "cannot open a file manager on a remote host."
                ),
                "workspace": workspace_str,
            },
        )

    if sys.platform == "win32":
        argv = ["explorer", workspace_str]
    elif sys.platform == "darwin":
        argv = ["open", workspace_str]
    else:
        argv = ["xdg-open", workspace_str]

    try:
        subprocess.Popen(argv, close_fds=True)  # noqa: S603
    except FileNotFoundError as exc:
        # `xdg-open` / `explorer` not installed — same UX as headless.
        raise HTTPException(
            status_code=409,
            detail={
                "code": "file_manager_missing",
                "message": f"file manager not available: {exc}",
                "workspace": workspace_str,
            },
        ) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "open_failed",
                "message": str(exc),
                "workspace": workspace_str,
            },
        ) from exc

    return {"opened": workspace_str}


@router.get("/jobs/{job_id}/files")
def list_job_files(job_id: str) -> dict[str, Any]:
    """List training artifacts in the job's workspace, classified by kind.

    Pure read-only inspection — the helper walks the workspace, drops archive
    and scratch entries, and groups the rest into checkpoints / samples /
    logs / other so the dashboard can render distinct sections.
    """
    job = state.registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    buckets = _list_workspace_files(job.workspace)
    return {"workspace": str(job.workspace), **buckets}


@router.get("/jobs/{job_id}/files/raw")
def get_job_file(job_id: str, path: str) -> FileResponse:
    """Stream a single workspace artifact for download or inline preview.

    The caller-supplied `path` is resolved against the job workspace and the
    result is required to stay inside it — anything that resolves outside
    (`..` traversal, absolute paths, symlink escapes) is rejected with 400.
    Images are served `inline` so the frontend can drop them straight into an
    `<img>` tag; everything else is sent as an attachment download.
    """
    job = state.registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")

    try:
        target = _resolve_workspace_file(job.workspace, path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not target.is_file():
        raise HTTPException(status_code=404, detail="file not found")

    media_type, disposition = _media_type_for(target)
    return FileResponse(
        target,
        media_type=media_type,
        filename=target.name,
        content_disposition_type=disposition,
    )


@router.get("/jobs/{job_id}/metrics")
def get_job_metrics(job_id: str) -> dict[str, Any]:
    """Return chartable time-series extracted from `events.jsonl`.

    Empty arrays + null duration when the log doesn't exist yet (the job has
    not started or has not produced events). Long runs are downsampled to
    keep the response bounded.
    """
    job = state.registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return _read_metrics(job.workspace)


@router.get("/jobs/{job_id}/analysis")
def get_job_analysis(job_id: str) -> dict[str, Any]:
    """Return the cached AI analysis for this job, or null if none yet."""
    job = state.registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    meta = job.metadata or {}
    cached = meta.get("ai_analysis")
    return {"analysis": cached}


@router.post("/jobs/{job_id}/analyze")
def analyze_job(job_id: str) -> dict[str, Any]:
    """Run a Claude-style diagnosis over this job's metrics + config.

    Reads the events.jsonl + config snapshot, builds a compact prompt,
    sends it to the `global.default` AI route, and stamps the result on
    `job.metadata['ai_analysis']` so reloads pick it up. Idempotent:
    re-calling regenerates and overwrites.
    """
    from lorahub.api import app as app_mod  # noqa: PLC0415
    from lorahub.core.ai import client as ai_client  # noqa: PLC0415

    job = state.registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")

    ai_store = app_mod._ai_store
    if ai_store is None:
        raise HTTPException(503, "AI store not initialised")

    route = ai_store.get_route("training.analyze")
    if route is None or not (route.provider_id and route.model_id):
        route = ai_store.get_route("global.default")
    if route is None or not (route.provider_id and route.model_id):
        raise HTTPException(409, "no AI route configured")

    metrics = _read_metrics(job.workspace)
    cfg = job.config_snapshot or {}

    # Compact context for the LLM: trim huge arrays, only keep the bits
    # that actually inform a training diagnosis.
    loss = metrics.get("loss") or []
    val_loss = metrics.get("val_loss") or []
    epochs = metrics.get("epochs") or []
    overfit = metrics.get("overfit_signal")

    def _sample(seq: list[Any], n: int = 40) -> list[Any]:
        if len(seq) <= n:
            return seq
        # First, last, plus evenly spaced middle picks.
        step = max(1, len(seq) // n)
        return seq[::step][:n]

    summary_payload: dict[str, Any] = {
        "job": {
            "id": job.id,
            "state": job.state.value,
            "returncode": job.returncode,
            "duration_s": metrics.get("duration_s"),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "finished_at": job.finished_at.isoformat() if job.finished_at else None,
        },
        "config": {
            "arch": ((cfg.get("baseModel") or cfg.get("base_model") or {}).get("arch")),
            "rank": ((cfg.get("network") or {}).get("rank")),
            "alpha": ((cfg.get("network") or {}).get("alpha")),
            "lr": ((cfg.get("optimizer") or {}).get("lr")),
            "schedule": ((cfg.get("optimizer") or {}).get("schedule")),
            "epochs": ((cfg.get("schedule") or {}).get("epochs")),
            "batch_size": (
                (cfg.get("schedule") or {}).get("batchSize")
                or (cfg.get("schedule") or {}).get("batch_size")
            ),
            "grad_accum": (
                (cfg.get("schedule") or {}).get("gradAccum")
                or (cfg.get("schedule") or {}).get("grad_accum")
            ),
            "num_repeats": (
                (cfg.get("dataset") or {}).get("numRepeats")
                or (cfg.get("dataset") or {}).get("num_repeats")
            ),
        },
        "metrics": {
            "total_loss_points": len(loss),
            "loss_samples": _sample(loss, 30),
            "val_loss": val_loss,
            "epochs": epochs,
            "overfit_signal": overfit,
        },
    }

    import json as _json  # noqa: PLC0415
    prompt = (
        "You are a LoRA training diagnostician. The user wants a concise "
        "Chinese-language analysis of THIS job. Cover, in order:\n"
        "  1. 一句话结论：训练是否健康？是否出现明显问题？\n"
        "  2. 收敛趋势：loss 形状（下降/震荡/平台/发散）。\n"
        "  3. 过拟合判断：train vs val 距离 + overfit_signal。\n"
        "  4. 学习率与 schedule 是否合适。\n"
        "  5. 下一次实验建议（lr / rank / epochs / dropout 各 1-2 条具体动作）。\n\n"
        "Use plain Markdown, short paragraphs, no fluff. If data is too sparse "
        "to judge a section, say so.\n\n"
        "JSON context:\n```json\n"
        + _json.dumps(summary_payload, ensure_ascii=False, indent=2)
        + "\n```"
    )

    messages: list[dict[str, Any]] = []
    if route.system_prompt:
        messages.append({"role": "system", "content": route.system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        result = ai_client.invoke(
            ai_store,
            provider_id=route.provider_id,
            model_id=route.model_id,
            messages=messages,
            route=route,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"analysis failed: {exc}") from exc

    analysis = {
        "markdown": result.content,
        "model": f"{result.provider_name}/{result.model_id}",
        "generated_at": datetime.now(UTC).isoformat(),
        "summary_payload": summary_payload,
    }
    meta = dict(job.metadata or {})
    meta["ai_analysis"] = analysis
    job.metadata = meta
    state.registry.update(job)
    return {"analysis": analysis}


@router.post("/jobs/{job_id}/kill")
def kill_job(job_id: str) -> dict[str, Any]:
    """Force-kill a stuck training job by signalling its PID + process group.

    Last-resort cleanup for jobs whose registry says ``running`` but whose
    cancel button never returned (e.g. deepspeed launcher detached and the
    in-process handle is stale). Does NOT touch arbitrary OS PIDs — the
    caller must reference a JobRecord, and we only signal that record's
    stored PID. Best-effort kills the whole process group so detached
    deepspeed children go down with the launcher.

    On success the job row is flipped to ``interrupted`` so the UI stops
    showing it as live. Returns 404 if the job is unknown, 409 if it has
    no PID to signal.
    """
    import os  # noqa: PLC0415
    import signal  # noqa: PLC0415

    job = state.registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.pid is None or job.pid <= 0:
        raise HTTPException(
            status_code=409,
            detail="job has no recorded PID (never started or already reaped)",
        )

    pid = job.pid
    killed_group = False
    killed_pid = False
    error: str | None = None
    # Try the process group first (matches deepspeed's setsid layout); fall
    # back to the bare PID if the OS doesn't support process groups (Windows
    # via psutil) or if the group has already gone.
    try:
        if hasattr(os, "killpg") and hasattr(os, "getpgid"):
            try:
                pgid = os.getpgid(pid)
                os.killpg(pgid, signal.SIGKILL)
                killed_group = True
            except ProcessLookupError:
                pass
            except PermissionError as exc:
                error = f"permission denied: {exc}"
        if not killed_group:
            try:
                os.kill(pid, signal.SIGKILL if hasattr(signal, "SIGKILL") else signal.SIGTERM)
                killed_pid = True
            except ProcessLookupError:
                pass
            except PermissionError as exc:
                error = f"permission denied: {exc}"
    except Exception as exc:  # noqa: BLE001
        error = repr(exc)

    if not killed_group and not killed_pid and error is None:
        # Process was already gone; that's fine, still flip the state so the
        # UI doesn't keep showing a phantom running row.
        error = f"pid {pid} no longer alive"

    if error and not (killed_group or killed_pid):
        raise HTTPException(status_code=500, detail=error)

    # Reset the live record. We pick `interrupted` rather than `canceled`
    # because the run did not request cancellation cleanly — kill is the
    # graceless variant and `interrupted` is already the bucket for "process
    # gone, not by our handle's request".
    job.state = JobState.interrupted
    job.finished_at = datetime.now(UTC)
    if job.error is None:
        job.error = "force-killed via /api/jobs/{id}/kill"
    state.registry.update(job)

    return {
        "job_id": job_id,
        "pid": pid,
        "killed_process_group": killed_group,
        "killed_pid_only": killed_pid,
        "warning": error,
    }


@router.delete("/jobs/{job_id}")
def cancel_job(job_id: str, archive: bool = False) -> dict[str, Any]:
    job = state.registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")

    if archive:
        if job.state not in _TERMINAL_STATES:
            raise HTTPException(
                status_code=409,
                detail=f"job is {job.state.value}; cancel before archiving",
            )
        # Refuse to archive when another non-terminal job is currently
        # using this same workspace path. Multiple jobs share a workspace
        # whenever they were created with the same `output.name`; mv'ing
        # the dir out from under an active sibling crashes that sibling
        # mid-run with FileNotFoundError on its toml/dataset/checkpoints.
        target_ws = job.workspace.resolve()
        for other in state.registry.list():
            if other.id == job.id:
                continue
            if other.state in _TERMINAL_STATES:
                continue
            try:
                if other.workspace.resolve() == target_ws:
                    raise HTTPException(
                        status_code=409,
                        detail=(
                            f"workspace {target_ws} is in use by job "
                            f"{other.id} (state={other.state.value}); "
                            "cancel that job before archiving this one"
                        ),
                    )
            except OSError:
                # workspace path resolution failed for `other` — skip it
                # rather than blocking on a broken sibling record.
                continue
        # `interrupted` is the one terminal state where the OS process
        # might still be alive (e.g. uvicorn was kill -9'd while a
        # detached deepspeed launcher kept running, then mark_orphans
        # flipped this row to interrupted on restart). For other terminal
        # states the reaper has already observed `done`, so the PID is
        # irrelevant even if reused. Only probe for `interrupted`.
        if (
            job.state is JobState.interrupted
            and job.pid is not None
            and _pid_alive(job.pid)
        ):
            raise HTTPException(
                status_code=409,
                detail=(
                    f"job process pid={job.pid} is still alive even though "
                    f"state is interrupted; cancel it explicitly before "
                    "archiving (the workspace mv would crash the running run)"
                ),
            )
        moved, warnings = _archive_workspace(job.workspace, job.id)
        state.registry.delete(job.id)
        return {
            "archived": True,
            "workspace_moved_to": str(moved) if moved is not None else None,
            "warnings": warnings,
        }

    if job.state in _TERMINAL_STATES:
        return job.to_summary()
    if job.state is JobState.queued:
        # Worker hasn't claimed it yet — flip directly so the closure
        # short-circuits when its slot eventually pops the deque.
        job.state = JobState.canceled
        job.finished_at = datetime.now(UTC)
        state.registry.update(job)
        return job.to_summary()
    job.state = JobState.canceling
    state.registry.update(job)
    if job.handle is not None:
        job.handle.stop(graceful=True)
    return job.to_summary()
