"""Pre-flight validation for training jobs.

Catches the common "training started, then died seconds in with an
opaque traceback" class of failures by checking the obvious blockers
*before* we ever spawn the trainer subprocess. Each check returns a
finding with a category, severity, the cfg field that triggered it,
and remediation text. The router converts any ``severity == "error"``
finding into a structured 422 response.

Design rules:
- One function per check; pure, no I/O side-effects.
- Always return *all* findings — never short-circuit. Users want to fix
  every blocker in one round-trip, not learn about them one at a time.
- Findings reference the cfg field by its camelCase JSON path (matching
  what the API consumer sees), so the frontend can highlight the
  offending form input directly.
- Read-only filesystem probes only. Touching the output dir to test
  writability is the single side-effect; we always clean up.

Categories used:
- ``model_missing``        — base model / VAE / arch_paths file missing
- ``dataset_missing``      — source / subset / reg_source / conditioning dir
- ``backend_repo_missing`` — backend.repoPath unset or scripts absent
- ``venv_missing``         — backend.pythonExecutable unreachable
- ``output_unwritable``    — output_dir / workspace cannot be written
- ``path_encoding``        — Windows-only mbcs encode failure
- ``disk_low``             — workspace partition tight on space
- ``dataset_empty``        — dataset directory has zero indexable files

This module is import-safe on non-Windows platforms; ``mbcs`` checks
no-op everywhere else.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

from lorahub.core.config.schema import TrainingConfig

Severity = Literal["info", "warn", "error"]

# Image extensions we consider "trainable" when probing dataset
# directories. Mirrors what kohya/dp expect — anything outside this
# set is silently ignored downstream so it shouldn't count.
_IMAGE_EXTS: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff", ".tif"}
)
_VIDEO_EXTS: frozenset[str] = frozenset(
    {".mp4", ".mov", ".avi", ".mkv", ".webm"}
)


@dataclass(slots=True)
class PreflightFinding:
    """A single blocker / warning surfaced before launch."""

    category: str
    severity: Severity
    field: str  # camelCase cfg path, e.g. "baseModel.checkpoint"
    message: str
    remediation: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "category": self.category,
            "severity": self.severity,
            "field": self.field,
            "message": self.message,
            "remediation": self.remediation,
        }
        if self.extra:
            d["extra"] = self.extra
        return d


def run_preflight(
    cfg: TrainingConfig,
    workspace: Path,
    *,
    skip: Iterable[str] = (),
) -> list[PreflightFinding]:
    """Top-level preflight runner.

    ``skip`` is a set of category names to skip; tests use it to
    silence checks that depend on real filesystem state.
    """
    skip_set = set(skip)
    findings: list[PreflightFinding] = []

    if "model_missing" not in skip_set:
        findings.extend(_check_model_files(cfg))
    if "dataset_missing" not in skip_set:
        findings.extend(_check_dataset(cfg))
    if "dataset_empty" not in skip_set:
        findings.extend(_check_dataset_has_samples(cfg))
    if "backend_repo_missing" not in skip_set:
        findings.extend(_check_backend_repo(cfg))
    if "venv_missing" not in skip_set:
        findings.extend(_check_python_executable(cfg))
    if "output_unwritable" not in skip_set:
        findings.extend(_check_output_writable(cfg, workspace))
    if "path_encoding" not in skip_set:
        findings.extend(_check_path_encoding(cfg, workspace))
    if "disk_low" not in skip_set:
        findings.extend(_check_disk_space(workspace))
    if "gpu_dispatch" not in skip_set:
        findings.extend(_check_gpu_dispatch(cfg))

    return findings


# --------------------------------------------------------------------- #
# Individual checks
# --------------------------------------------------------------------- #
def _check_model_files(cfg: TrainingConfig) -> list[PreflightFinding]:
    """Verify base model + VAE + arch_paths + dp model_paths files."""
    out: list[PreflightFinding] = []

    ckpt = cfg.base_model.checkpoint
    if ckpt is None or not _path_exists_as_file_or_dir(ckpt):
        out.append(
            PreflightFinding(
                category="model_missing",
                severity="error",
                field="baseModel.checkpoint",
                message=(
                    f"Base checkpoint not found: {ckpt!s}"
                    if ckpt is not None
                    else "baseModel.checkpoint is empty"
                ),
                remediation=(
                    "Set baseModel.checkpoint to the .safetensors / .ckpt "
                    "of the base model. Use the picker in the form, or run "
                    "`make download-models` to populate the default cache."
                ),
            )
        )

    vae = cfg.base_model.vae
    if vae is not None and not _path_exists_as_file_or_dir(vae):
        out.append(
            PreflightFinding(
                category="model_missing",
                severity="error",
                field="baseModel.vae",
                message=f"VAE file not found: {vae!s}",
                remediation=(
                    "Either set baseModel.vae to an existing path or clear "
                    "the field to use the checkpoint's bundled VAE."
                ),
            )
        )

    # arch_paths is a typed bag of optional component paths; loop fields
    # whose declared annotation is Path | None and check each individually.
    arch_paths = cfg.base_model.arch_paths
    arch_path_fields = {
        name
        for name, info in type(arch_paths).model_fields.items()
        if "Path" in str(info.annotation)
    }
    for fname, fval in arch_paths.model_dump(exclude_none=True).items():
        if fname not in arch_path_fields or not fval:
            continue
        p = Path(str(fval))
        if not _path_exists_as_file_or_dir(p):
            out.append(
                PreflightFinding(
                    category="model_missing",
                    severity="error",
                    field=f"baseModel.archPaths.{_to_camel(fname)}",
                    message=f"Component file not found: {p!s}",
                    remediation=(
                        f"baseModel.archPaths.{_to_camel(fname)} points at "
                        "a file that doesn't exist. Re-pick the path or "
                        "clear the field if this component isn't needed."
                    ),
                )
            )

    # diffusion-pipe `model_paths` is a free-form bag — anything in it
    # had better resolve.
    dp_opts = cfg.backend.diffusion_pipe
    if dp_opts is not None:
        for key, val in dp_opts.model_paths.items():
            if not val:
                continue
            p = Path(str(val))
            if not _path_exists_as_file_or_dir(p):
                out.append(
                    PreflightFinding(
                        category="model_missing",
                        severity="error",
                        field=f"backend.diffusionPipe.modelPaths.{key}",
                        message=f"diffusion-pipe model path not found: {p!s}",
                        remediation=(
                            f"backend.diffusionPipe.modelPaths.{key} points "
                            "at a file that doesn't exist. Update the path "
                            "or remove the entry if unused."
                        ),
                    )
                )

    return out


def _check_dataset(cfg: TrainingConfig) -> list[PreflightFinding]:
    """Verify dataset.source / subsets / reg_source / conditioning_dir."""
    out: list[PreflightFinding] = []
    ds = cfg.dataset

    # When subsets is non-empty it overrides .source — we still warn on
    # missing .source because the form often leaves both populated.
    if not ds.subsets:
        if ds.source is None or not Path(str(ds.source)).is_dir():
            out.append(
                PreflightFinding(
                    category="dataset_missing",
                    severity="error",
                    field="dataset.source",
                    message=(
                        f"Dataset directory not found: {ds.source!s}"
                        if ds.source is not None
                        else "dataset.source is empty and no subsets configured"
                    ),
                    remediation=(
                        "Pick an existing image directory in the dataset "
                        "field, or add at least one entry under "
                        "dataset.subsets."
                    ),
                )
            )
    else:
        for idx, subset in enumerate(ds.subsets):
            sp = subset.path
            if sp is None or not Path(str(sp)).is_dir():
                out.append(
                    PreflightFinding(
                        category="dataset_missing",
                        severity="error",
                        field=f"dataset.subsets[{idx}].path",
                        message=f"Subset directory not found: {sp!s}",
                        remediation=(
                            f"dataset.subsets[{idx}].path must point at an "
                            "existing directory. Either fix the path or "
                            "remove this subset entry."
                        ),
                    )
                )

    if ds.reg_source is not None and not Path(str(ds.reg_source)).is_dir():
        out.append(
            PreflightFinding(
                category="dataset_missing",
                severity="error",
                field="dataset.regSource",
                message=f"Regularisation directory not found: {ds.reg_source!s}",
                remediation=(
                    "DreamBooth regularisation expects an existing image "
                    "directory. Clear dataset.regSource if you don't want "
                    "to use regularisation."
                ),
            )
        )

    if ds.conditioning_dir is not None and not Path(str(ds.conditioning_dir)).is_dir():
        out.append(
            PreflightFinding(
                category="dataset_missing",
                severity="error",
                field="dataset.conditioningDir",
                message=f"Conditioning directory not found: {ds.conditioning_dir!s}",
                remediation=(
                    "ControlNet / inpainting conditioning needs a real "
                    "directory of conditioning images. Clear the field if "
                    "you're not training a conditional adapter."
                ),
            )
        )

    return out


def _check_dataset_has_samples(cfg: TrainingConfig) -> list[PreflightFinding]:
    """Warn if the dataset directory has zero images / videos."""
    out: list[PreflightFinding] = []
    candidates: list[tuple[str, Path]] = []

    if cfg.dataset.subsets:
        for idx, subset in enumerate(cfg.dataset.subsets):
            if subset.path is not None:
                candidates.append((f"dataset.subsets[{idx}].path", Path(str(subset.path))))
    elif cfg.dataset.source is not None:
        candidates.append(("dataset.source", Path(str(cfg.dataset.source))))

    valid_exts = _IMAGE_EXTS | _VIDEO_EXTS
    for fld, p in candidates:
        if not p.is_dir():
            # Already reported by _check_dataset; don't double-flag.
            continue
        sample = _count_with_extensions(p, valid_exts, limit=1)
        if sample == 0:
            out.append(
                PreflightFinding(
                    category="dataset_empty",
                    severity="error",
                    field=fld,
                    message=f"Dataset directory has no images or videos: {p!s}",
                    remediation=(
                        f"Add at least one trainable file (extensions: "
                        f"{sorted(valid_exts)!s}) under {p!s}, or point "
                        f"{fld} at a directory that has them."
                    ),
                )
            )

    return out


def _check_backend_repo(cfg: TrainingConfig) -> list[PreflightFinding]:
    """Verify the backend repo path looks like a real checkout."""
    out: list[PreflightFinding] = []
    repo = cfg.backend.repo_path
    if repo is None:
        # Some backends (default kohya) discover the repo automatically;
        # only flag when the path is set but missing.
        return out
    p = Path(str(repo))
    if not p.is_dir():
        out.append(
            PreflightFinding(
                category="backend_repo_missing",
                severity="error",
                field="backend.repoPath",
                message=f"Backend repository directory not found: {p!s}",
                remediation=(
                    "Point backend.repoPath at the local checkout of the "
                    "training repo (sd-scripts for kohya, diffusion-pipe "
                    "for dp, anima_lora for anima). Clone via "
                    "`scripts/install.sh` or clear the field to fall back "
                    "to the default discovery."
                ),
            )
        )
        return out

    # Per-backend canonical entry script. We don't enforce all of them —
    # only the train/launcher script, since that's what the compilers
    # invoke.
    canonical: dict[str, list[str]] = {
        "kohya": ["train_network.py", "sdxl_train_network.py", "flux_train_network.py"],
        "diffusion-pipe": ["train.py"],
        "anima_lora": ["train_network.py"],
    }
    expected = canonical.get(cfg.backend.type, [])
    if expected and not any((p / s).is_file() for s in expected):
        out.append(
            PreflightFinding(
                category="backend_repo_missing",
                severity="error",
                field="backend.repoPath",
                message=(
                    f"Backend repo at {p!s} is missing the expected entry "
                    f"script (looked for: {expected})."
                ),
                remediation=(
                    "Either the repoPath points at the wrong directory, or "
                    "the checkout is incomplete / on the wrong branch. "
                    f"Re-run scripts/install.sh to refresh the {cfg.backend.type} repo."
                ),
            )
        )
    return out


def _check_python_executable(cfg: TrainingConfig) -> list[PreflightFinding]:
    """If the config pins a Python executable, ensure it exists & is callable."""
    out: list[PreflightFinding] = []
    py = cfg.backend.python_executable
    if py is None:
        return out
    p = Path(str(py))
    if not p.is_file():
        out.append(
            PreflightFinding(
                category="venv_missing",
                severity="error",
                field="backend.pythonExecutable",
                message=f"Pinned Python executable not found: {p!s}",
                remediation=(
                    "Either install / fix the venv that backend.pythonExecutable "
                    "points at, or clear the field to fall back to the "
                    "system Python on PATH."
                ),
            )
        )
        return out
    # Best-effort access check — Windows reports True for is_file even
    # when ACL denies execution, so we don't promote this to error.
    try:
        if not os.access(p, os.X_OK):
            out.append(
                PreflightFinding(
                    category="venv_missing",
                    severity="warn",
                    field="backend.pythonExecutable",
                    message=f"Python executable is not marked executable: {p!s}",
                    remediation=(
                        "On POSIX, run `chmod +x` on the path. On Windows "
                        "this warning is harmless if the .exe runs from a "
                        "regular shell."
                    ),
                )
            )
    except OSError:
        pass
    return out


def _check_output_writable(
    cfg: TrainingConfig, workspace: Path
) -> list[PreflightFinding]:
    """Touch-test workspace + cfg.output.outputDir."""
    out: list[PreflightFinding] = []
    targets: list[tuple[str, Path]] = [("workspace", workspace)]
    if cfg.output.output_dir is not None:
        targets.append(("output.outputDir", Path(str(cfg.output.output_dir))))

    for fld, p in targets:
        try:
            p.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            out.append(
                PreflightFinding(
                    category="output_unwritable",
                    severity="error",
                    field=fld,
                    message=f"Cannot create directory {p!s}: {exc}",
                    remediation=(
                        f"Either change {fld} to a writable location or fix "
                        "the permissions / free space on the parent volume."
                    ),
                )
            )
            continue
        probe = p / ".lorahub-write-test"
        try:
            probe.write_bytes(b"ok")
            probe.unlink(missing_ok=True)
        except OSError as exc:
            out.append(
                PreflightFinding(
                    category="output_unwritable",
                    severity="error",
                    field=fld,
                    message=f"Directory exists but is not writable: {p!s} ({exc})",
                    remediation=(
                        f"Grant write permission on {p!s} (Properties → "
                        "Security on Windows; chmod / chown on POSIX), or "
                        f"change {fld} to a directory you own."
                    ),
                )
            )
    return out


def _check_path_encoding(
    cfg: TrainingConfig, workspace: Path
) -> list[PreflightFinding]:
    """Windows-only: verify every path the trainer touches is mbcs-encodable.

    cmd.exe and the kohya/anima sd-scripts launchers serialise paths
    via the active ANSI code page. A path with characters that the
    code page can't encode (e.g. an emoji, or CJK on a Latin-1 box)
    will reach the trainer as mojibake and the run dies with an
    obscure ``UnicodeEncodeError`` deep in subprocess.
    """
    out: list[PreflightFinding] = []
    if sys.platform != "win32":
        return out

    paths_to_check: list[tuple[str, Path]] = [("workspace", workspace)]
    if cfg.base_model.checkpoint is not None:
        paths_to_check.append(("baseModel.checkpoint", Path(str(cfg.base_model.checkpoint))))
    if cfg.base_model.vae is not None:
        paths_to_check.append(("baseModel.vae", Path(str(cfg.base_model.vae))))
    if cfg.dataset.source is not None:
        paths_to_check.append(("dataset.source", Path(str(cfg.dataset.source))))
    if cfg.output.output_dir is not None:
        paths_to_check.append(("output.outputDir", Path(str(cfg.output.output_dir))))
    if cfg.backend.repo_path is not None:
        paths_to_check.append(("backend.repoPath", Path(str(cfg.backend.repo_path))))
    if cfg.backend.python_executable is not None:
        paths_to_check.append(
            ("backend.pythonExecutable", Path(str(cfg.backend.python_executable)))
        )

    seen_paths: set[str] = set()
    for fld, p in paths_to_check:
        s = str(p)
        if s in seen_paths:
            continue
        seen_paths.add(s)
        try:
            s.encode("mbcs")
        except UnicodeEncodeError as exc:
            out.append(
                PreflightFinding(
                    category="path_encoding",
                    severity="error",
                    field=fld,
                    message=(
                        f"Path contains characters the system ANSI code page "
                        f"(mbcs) cannot encode: {p!s} (offending position "
                        f"{exc.start}-{exc.end})"
                    ),
                    remediation=(
                        "Move the project / dataset / model out of any "
                        "directory whose name contains characters not "
                        "representable in your Windows ANSI code page "
                        "(e.g. emoji, mixed-script names). cmd.exe and "
                        "sd-scripts pass paths through ANSI before "
                        "reaching Python."
                    ),
                )
            )
    return out


def _check_disk_space(workspace: Path) -> list[PreflightFinding]:
    """Warn when the workspace partition is below ~2 GiB free."""
    out: list[PreflightFinding] = []
    probe = workspace if workspace.exists() else workspace.parent
    if not probe.exists():
        # Workspace creation will be flagged by output_writable; nothing
        # to measure here.
        return out
    try:
        usage = shutil.disk_usage(probe)
    except OSError:
        return out

    threshold_bytes = 2 * 1024**3  # 2 GiB
    if usage.free < threshold_bytes:
        out.append(
            PreflightFinding(
                category="disk_low",
                severity="warn",
                field="workspace",
                message=(
                    f"Only {usage.free / 1024**3:.1f} GiB free on the "
                    f"workspace volume ({probe.anchor or probe}). "
                    "Checkpoints and state directories are large; the run "
                    "may fail mid-training with 'No space left on device'."
                ),
                remediation=(
                    "Free up space on the workspace partition, point "
                    "output.outputDir at a roomier disk, or set "
                    "resume.saveLastNEpochsState to keep only the most "
                    "recent state directory."
                ),
                extra={"free_bytes": int(usage.free), "threshold_bytes": threshold_bytes},
            )
        )
    return out


def _check_gpu_dispatch(cfg: TrainingConfig) -> list[PreflightFinding]:
    dispatch = cfg.backend.gpu_dispatch
    if dispatch.mode != "distributed":
        return []

    out: list[PreflightFinding] = []
    if cfg.backend.type not in {"anima_lora", "diffusion-pipe"}:
        out.append(
            PreflightFinding(
                category="gpu_dispatch",
                severity="error",
                field="backend.gpuDispatch.mode",
                message="当前后端不支持单任务多 GPU 分布式训练。",
                remediation=(
                    "改回 one-job-per-gpu，或切换到 anima_lora / diffusion-pipe。"
                ),
            )
        )
    if (
        cfg.backend.type == "anima_lora"
        and cfg.backend.anima_lora is not None
        and cfg.backend.anima_lora.turbo is not None
    ):
        out.append(
            PreflightFinding(
                category="gpu_dispatch",
                severity="error",
                field="backend.gpuDispatch.mode",
                message="anima_lora turbo 蒸馏当前不支持单任务多 GPU。",
                remediation="关闭 turbo，或把 GPU 调度改回 one-job-per-gpu。",
            )
        )

    try:
        from lorahub.api import scheduler as sched  # noqa: PLC0415

        available = sched.scheduler.concurrency
    except Exception:  # noqa: BLE001
        available = 1
    requested = dispatch.num_gpus or available
    if requested < 2:
        out.append(
            PreflightFinding(
                category="gpu_dispatch",
                severity="error",
                field="backend.gpuDispatch.mode",
                message=(
                    "单任务多 GPU 当前只会申请 1 个 GPU slot，训练会退化成单卡。"
                ),
                remediation=(
                    "把设置里的“并行训练槽位”设为 2 或更高并重启服务；"
                    "也可以在训练配置里明确设置 backend.gpuDispatch.numGpus=2。"
                ),
                extra={"available_slots": available, "requested_slots": requested},
            )
        )
    if requested > available:
        out.append(
            PreflightFinding(
                category="gpu_dispatch",
                severity="error",
                field="backend.gpuDispatch.numGpus",
                message=(
                    f"分布式训练请求 {requested} 张 GPU，但当前调度器只有 "
                    f"{available} 个 GPU slot。"
                ),
                remediation=(
                    "降低 backend.gpuDispatch.numGpus，或在设置里提高 "
                    "max_concurrent_jobs 后重启服务。"
                ),
            )
        )
    groups = _homogeneous_gpu_groups()
    if groups:
        largest = max(len(g) for g in groups)
        if requested > largest:
            out.append(
                PreflightFinding(
                    category="gpu_dispatch",
                    severity="error",
                    field="backend.gpuDispatch.numGpus",
                    message=(
                        f"请求 {requested} 张 GPU 做单任务分布式，但当前最大同构 GPU 组 "
                        f"只有 {largest} 张。异构 GPU 不会默认混跑。"
                    ),
                    remediation=(
                        "4080 + V100 这类异构机器请使用 one-job-per-gpu 并发；"
                        "单任务多 GPU 需要型号、显存、compute capability 一致的卡。"
                    ),
                    extra={"homogeneous_groups": groups},
                )
            )
    return out


def _homogeneous_gpu_groups() -> list[list[int]]:
    try:
        from lorahub.api.gpu_topology import homogeneous_slot_groups  # noqa: PLC0415

        return homogeneous_slot_groups()
    except Exception:  # noqa: BLE001
        return []


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #
def _path_exists_as_file_or_dir(p: Path | str | None) -> bool:
    if p is None:
        return False
    pp = Path(str(p))
    return pp.is_file() or pp.is_dir()


def _count_with_extensions(
    root: Path, exts: frozenset[str], *, limit: int = 1
) -> int:
    """Count files under ``root`` whose suffix is in ``exts``, capped at ``limit``.

    Walks recursively because dataset directories often have a single
    nested folder per concept.
    """
    found = 0
    try:
        for sub_root, _dirs, files in os.walk(root):
            for name in files:
                if Path(name).suffix.lower() in exts:
                    found += 1
                    if found >= limit:
                        return found
            # Don't recurse into hidden dirs.
            if sub_root != str(root):
                continue
    except OSError:
        return found
    return found


def _to_camel(snake: str) -> str:
    parts = snake.split("_")
    if not parts:
        return snake
    return parts[0] + "".join(p.title() for p in parts[1:])


__all__ = ["PreflightFinding", "run_preflight"]
