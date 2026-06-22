"""Training-artifact management.

Spotlights the trainer outputs (LoRA `.safetensors`, kohya state
directories, sample images) across every job, so the user has a
single page to download / clean up / re-locate weights without
hopping between job detail tabs.

Routes:

* ``GET  /api/artifacts``                          — flat list of every
  job's artifacts grouped by job, with size / mtime / role.
* ``GET  /api/artifacts/{job_id}/zip``             — stream the job's
  workspace as a single ``.zip``. ``include`` query (csv) picks
  buckets: ``checkpoints,samples,logs,other``. Default is
  ``checkpoints`` only — the LoRA weights are what users actually
  ship, the rest blow up the archive size needlessly.
* ``DELETE /api/artifacts/{job_id}/file?path=...`` — drop a single
  workspace-relative file. Same path-traversal guards as
  ``/jobs/{id}/files/raw``.
* ``DELETE /api/artifacts/{job_id}/workspace``     — physically remove
  the entire workspace tree. Differs from
  ``DELETE /api/jobs/{id}?archive=true`` (which moves to
  ``_archive/``) — this one is unrecoverable. Refuses non-terminal
  jobs.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tarfile
import threading
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from lorahub.api import state
from lorahub.api.jobs_helpers import (
    _list_workspace_files,
    _resolve_workspace_file,
)
from lorahub.api.jobs_helpers.resume_dispatch import (
    _dp_output_dir,
    _iter_state_dirs,
)
from lorahub.api.state import JobState
from lorahub.core.config.schema import TrainingConfig

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

_TERMINAL_STATES = (
    JobState.succeeded,
    JobState.failed,
    JobState.canceled,
    JobState.interrupted,
)

# Buckets the zip endpoint understands. ``checkpoints`` is the
# default-and-typical case; ``samples`` is useful when the user
# wants the preview grid alongside the LoRA; ``logs`` and ``other``
# are the long-tail.
_VALID_BUCKETS: frozenset[str] = frozenset({"checkpoints", "samples", "logs", "other"})
_VALID_ARCHIVE_FORMATS: frozenset[str] = frozenset(
    {"zip", "tar", "tar.gz", "tar.bz2", "tar.xz"}
)
_ARCHIVE_MEDIA_TYPES: dict[str, str] = {
    "zip": "application/zip",
    "tar": "application/x-tar",
    "tar.gz": "application/gzip",
    "tar.bz2": "application/x-bzip2",
    "tar.xz": "application/x-xz",
}
_TAR_MODES: dict[str, str] = {
    "tar": "w:",
    "tar.gz": "w:gz",
    "tar.bz2": "w:bz2",
    "tar.xz": "w:xz",
}


def _is_same_or_parent(target: Path, child: Path) -> bool:
    try:
        child.relative_to(target)
    except ValueError:
        return False
    return True


def _validate_workspace_delete_target(workspace: Path) -> Path:
    target = workspace.expanduser().resolve()
    cwd = Path.cwd().resolve()
    home = Path.home().resolve()
    if (
        target.parent == target
        or _is_same_or_parent(target, cwd)
        or _is_same_or_parent(target, home)
    ):
        raise HTTPException(
            status_code=400,
            detail="refusing to delete unsafe workspace path",
        )
    return target


def _job_artifact_summary(job_id: str, workspace: Path) -> dict[str, Any]:
    """Reduce a workspace listing to the artifacts page's row shape."""
    buckets = _list_workspace_files(workspace)
    checkpoints = buckets.get("checkpoints", [])
    samples = buckets.get("samples", [])
    total_bytes = (
        sum(int(e.get("size_bytes") or 0) for e in checkpoints)
        + sum(int(e.get("size_bytes") or 0) for e in samples)
        + sum(int(e.get("size_bytes") or 0) for e in buckets.get("logs", []))
        + sum(int(e.get("size_bytes") or 0) for e in buckets.get("other", []))
    )
    return {
        "job_id": job_id,
        "workspace": str(workspace),
        "exists": workspace.is_dir(),
        "checkpoints": checkpoints,
        "samples": samples,
        "total_bytes": total_bytes,
        "checkpoint_count": len(checkpoints),
        "sample_count": len(samples),
    }


@router.get("/artifacts")
def list_artifacts() -> dict[str, Any]:
    """Aggregate every job's artifact summary in one response."""
    rows: list[dict[str, Any]] = []
    for job in state.registry.list():
        summary = _job_artifact_summary(job.id, job.workspace)
        # Stamp a few job-level fields the UI needs to render the row
        # without making a second /jobs/{id} call per artifact.
        summary["state"] = job.state.value
        summary["created_at"] = (
            job.created_at.isoformat() if job.created_at else None
        )
        summary["finished_at"] = (
            job.finished_at.isoformat() if job.finished_at else None
        )
        cfg = job.config_snapshot or {}
        output = cfg.get("output") if isinstance(cfg, dict) else None
        if isinstance(output, dict):
            summary["output_name"] = output.get("name")
        rows.append(summary)
    rows.sort(
        key=lambda r: r.get("finished_at") or r.get("created_at") or "",
        reverse=True,
    )
    return {"jobs": rows}


def _zip_cache_root() -> Path:
    """Project-local cache for resumable artifact ZIP downloads."""
    root = Path.cwd() / "runs" / "_download_cache" / "artifacts"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _artifact_zip_cache_key(
    *,
    job_id: str,
    include: set[str],
    archive_format: str,
    workspace: Path,
    relpaths: Iterable[Path],
) -> str:
    """Hash the requested artifact set and file metadata.

    Dynamic ZIP streams cannot be resumed safely because the byte stream
    is regenerated on every request. The cache key makes the archive a
    normal stable file as long as the source artifacts are unchanged.
    """
    import hashlib  # noqa: PLC0415

    h = hashlib.sha256()
    h.update(job_id.encode("utf-8", "surrogateescape"))
    h.update(b"\0")
    h.update(",".join(sorted(include)).encode("utf-8", "surrogateescape"))
    h.update(b"\0")
    h.update(archive_format.encode("ascii"))
    root = workspace.resolve()
    for rel in sorted({r.as_posix() for r in relpaths}):
        full = (root / rel).resolve()
        try:
            full.relative_to(root)
            stat = full.stat()
        except (OSError, ValueError):
            continue
        if not full.is_file():
            continue
        h.update(b"\0")
        h.update(rel.encode("utf-8", "surrogateescape"))
        h.update(b":")
        h.update(str(stat.st_size).encode("ascii"))
        h.update(b":")
        h.update(str(stat.st_mtime_ns).encode("ascii"))
    return h.hexdigest()[:32]


def _materialise_artifact_zip(
    *,
    job_id: str,
    include: set[str],
    archive_format: str,
    workspace: Path,
    relpaths: Iterable[Path],
) -> Path:
    """Build or reuse a stable archive file for a job artifact selection."""
    key = _artifact_zip_cache_key(
        job_id=job_id,
        include=include,
        archive_format=archive_format,
        workspace=workspace,
        relpaths=relpaths,
    )
    dest = _zip_cache_root() / f"{job_id}-{key}.{archive_format}"
    if dest.is_file() and dest.stat().st_size > 0:
        return dest

    root = workspace.resolve()
    tmp = dest.with_name(f"{dest.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.unlink(missing_ok=True)
    try:
        relnames = sorted({r.as_posix() for r in relpaths})
        if archive_format == "zip":
            with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
                for rel in relnames:
                    full = (root / rel).resolve()
                    try:
                        full.relative_to(root)
                    except ValueError:
                        continue
                    if not full.is_file():
                        continue
                    try:
                        zf.write(full, arcname=rel)
                    except OSError as exc:
                        log.warning("artifacts archive: skipping %s — %s", full, exc)
                        continue
        else:
            mode = _TAR_MODES[archive_format]
            with tarfile.open(tmp, mode, format=tarfile.PAX_FORMAT) as tf:
                for rel in relnames:
                    full = (root / rel).resolve()
                    try:
                        full.relative_to(root)
                    except ValueError:
                        continue
                    if not full.is_file():
                        continue
                    try:
                        tf.add(full, arcname=rel, recursive=False)
                    except OSError as exc:
                        log.warning("artifacts archive: skipping %s — %s", full, exc)
                        continue
        tmp.replace(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return dest


def _validate_archive_format(raw: str) -> str:
    fmt = raw.strip().lower()
    aliases = {
        "tgz": "tar.gz",
        "tbz": "tar.bz2",
        "tbz2": "tar.bz2",
        "txz": "tar.xz",
    }
    fmt = aliases.get(fmt, fmt)
    if fmt not in _VALID_ARCHIVE_FORMATS:
        raise HTTPException(
            status_code=422,
            detail=(
                "unknown archive format: "
                f"{raw!r}; valid={sorted(_VALID_ARCHIVE_FORMATS)}"
            ),
        )
    return fmt


def _list_state_candidates(workspace: Path, backend_type: str, cfg: TrainingConfig | None) -> list[dict[str, Any]]:
    """Enumerate every resumable state target the user could pick.

    Returns a list of ``{kind, path, basename, modified_at, ...}`` dicts.

    * **kohya / anima_lora** — every ``*-state*`` directory under the
      workspace, sorted by mtime descending so the latest is first.
      ``current_step`` is parsed from ``train_state.json`` when present.
    * **diffusion-pipe** — every timestamped run dir under the resolved
      output_dir that has ``latest`` + at least one ``global_step*``
      subfolder. ``global_step`` is read from ``latest`` (the file
      contains the most recent step name).
    """
    candidates: list[dict[str, Any]] = []
    if not workspace.is_dir():
        return candidates

    if backend_type in ("kohya", "anima_lora"):
        for p in _iter_state_dirs(workspace):
            try:
                mtime = p.stat().st_mtime
            except OSError:
                continue
            entry: dict[str, Any] = {
                "kind": "accelerate-state",
                "path": str(p),
                "basename": p.name,
                "modified_at": mtime,
            }
            ts_file = p / "train_state.json"
            if ts_file.is_file():
                try:
                    data = json.loads(ts_file.read_text(encoding="utf-8"))
                    if isinstance(data.get("current_step"), int):
                        entry["current_step"] = int(data["current_step"])
                    if isinstance(data.get("current_epoch"), int):
                        entry["current_epoch"] = int(data["current_epoch"])
                except (OSError, ValueError, TypeError):
                    pass
            candidates.append(entry)
        candidates.sort(key=lambda e: e["modified_at"], reverse=True)
        return candidates

    if backend_type == "diffusion-pipe":
        if cfg is None:
            return candidates
        out_dir = _dp_output_dir(workspace, cfg)
        if not out_dir.is_dir():
            return candidates
        for child in out_dir.iterdir():
            if not child.is_dir():
                continue
            latest = child / "latest"
            if not latest.is_file():
                continue
            global_steps = [
                p for p in child.iterdir()
                if p.is_dir() and p.name.startswith("global_step")
            ]
            if not global_steps:
                continue
            entry: dict[str, Any] = {
                "kind": "dp-run-dir",
                "path": str(child.resolve()),
                "basename": child.name,
                "modified_at": child.stat().st_mtime,
                "output_dir": str(out_dir),
                "global_step_count": len(global_steps),
            }
            try:
                tag = latest.read_text(encoding="utf-8").strip()
                if tag.startswith("global_step"):
                    entry["latest_step"] = int(tag.removeprefix("global_step"))
            except (OSError, ValueError):
                pass
            candidates.append(entry)
        candidates.sort(key=lambda e: e["basename"], reverse=True)
        return candidates

    return candidates


@router.get("/artifacts/{job_id}/states")
def list_states(job_id: str) -> dict[str, Any]:
    """List every resumable state target for a job.

    The frontend's clone-with-state picker calls this to populate the
    "pick a checkpoint" dropdown. Returned shape is backend-aware:

    * kohya / anima_lora — accelerate ``*-state*`` directories;
    * diffusion-pipe — timestamped run dirs with ``global_step*/``.

    Backend type is read from the job's config snapshot. If the
    snapshot is unparsable we still return the workspace path so the
    UI can show a neutral "no resumable artifacts" empty state.
    """
    job = state.registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")

    cfg: TrainingConfig | None = None
    backend_type: str | None = None
    snap = job.config_snapshot
    if isinstance(snap, dict):
        try:
            cfg = TrainingConfig.model_validate(snap)
            backend_type = cfg.backend.type
        except Exception:  # noqa: BLE001
            cfg = None
    candidates: list[dict[str, Any]] = []
    if backend_type is not None:
        candidates = _list_state_candidates(job.workspace, backend_type, cfg)

    return {
        "job_id": job_id,
        "workspace": str(job.workspace),
        "backend_type": backend_type,
        "states": candidates,
    }


@router.get("/artifacts/{job_id}/zip")
def download_zip(
    job_id: str,
    include: str = Query(
        default="checkpoints",
        description="Comma-separated buckets to include: checkpoints,samples,logs,other.",
    ),
    format: str = Query(  # noqa: A002
        default="zip",
        description="Archive format: zip, tar, tar.gz, tar.bz2, tar.xz.",
    ),
) -> FileResponse:
    """Stream the job workspace as an archive, filtered to requested buckets.

    Default is ``checkpoints`` only — the LoRA weights are what people
    actually ship; samples and logs balloon the archive without
    matching the typical user intent of "give me the model file".

    Note: accelerator resume-state trees (``<output>-NNNNNN-state/`` and
    ``<output>-checkpoint-state/``) are classified as "other", not
    "checkpoints", even though they contain ``.safetensors`` files —
    their ``model.safetensors`` is a full optimizer-paired model
    snapshot for resume, not a LoRA artifact. The default zip
    therefore excludes them. Pass ``include=other`` (or
    ``include=checkpoints,other``) to bundle the resume tree.
    """
    job = state.registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if not job.workspace.is_dir():
        raise HTTPException(status_code=404, detail="workspace missing on disk")

    archive_format = _validate_archive_format(format)
    requested = {b.strip() for b in include.split(",") if b.strip()}
    invalid = requested - _VALID_BUCKETS
    if invalid:
        raise HTTPException(
            status_code=422,
            detail=f"unknown buckets: {sorted(invalid)}; valid={sorted(_VALID_BUCKETS)}",
        )
    if not requested:
        requested = {"checkpoints"}

    buckets = _list_workspace_files(job.workspace)
    rels: list[Path] = []
    for bucket in requested:
        for entry in buckets.get(bucket, []):
            p = entry.get("path")
            if isinstance(p, str) and p:
                rels.append(Path(p))

    if not rels:
        raise HTTPException(
            status_code=404,
            detail=f"no files matched the requested buckets ({sorted(requested)})",
        )

    output_name = "lorahub-artifacts"
    cfg = job.config_snapshot or {}
    if isinstance(cfg, dict):
        out = cfg.get("output")
        if isinstance(out, dict) and isinstance(out.get("name"), str):
            output_name = out["name"]
    zip_path = _materialise_artifact_zip(
        job_id=job_id,
        include=requested,
        archive_format=archive_format,
        workspace=job.workspace,
        relpaths=rels,
    )
    timestamp = datetime.fromtimestamp(zip_path.stat().st_mtime, UTC).strftime(
        "%Y%m%d_%H%M%S"
    )
    filename = f"{output_name}_{job_id[-8:]}_{timestamp}.{archive_format}"

    return FileResponse(
        zip_path,
        media_type=_ARCHIVE_MEDIA_TYPES[archive_format],
        filename=filename,
        content_disposition_type="attachment",
    )


@router.delete("/artifacts/{job_id}/file")
def delete_file(job_id: str, path: str) -> dict[str, Any]:
    """Delete one workspace-relative file. Path-traversal protected.

    Returns ``{"deleted": <abs path>, "size_bytes": <freed>}``. 404
    if the file isn't there (idempotent in spirit but the explicit
    error helps the UI show "did you double-click?" feedback).
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

    try:
        size = target.stat().st_size
        target.unlink()
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"unlink failed: {exc}") from exc
    return {"deleted": str(target), "size_bytes": size}


@router.delete("/artifacts/{job_id}/workspace")
def delete_workspace(job_id: str) -> dict[str, Any]:
    """Physically delete the entire workspace tree. Unrecoverable.

    Differs from ``DELETE /api/jobs/{id}?archive=true`` (which moves
    the tree to ``_archive/`` so the user can still recover). Refuses
    non-terminal jobs because ripping the workspace out from under a
    running trainer would crash the subprocess and leak GPU memory.
    """
    job = state.registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.state not in _TERMINAL_STATES:
        raise HTTPException(
            status_code=409,
            detail=(
                f"job is {job.state.value}; cancel it before deleting the workspace"
            ),
        )
    target = _validate_workspace_delete_target(job.workspace)
    if not target.exists():
        # Drop the registry row anyway so the artifacts page doesn't
        # keep showing a phantom entry.
        state.registry.delete(job.id)
        return {"deleted": False, "reason": "workspace missing on disk"}

    try:
        shutil.rmtree(target)
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"rmtree failed: {exc}",
        ) from exc
    state.registry.delete(job.id)
    return {"deleted": True, "workspace": str(target)}


__all__ = ["router"]
