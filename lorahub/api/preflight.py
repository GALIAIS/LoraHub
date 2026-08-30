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
import subprocess
import sys
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Any, Iterable, Literal

from lorahub.core.config.schema import TrainingConfig

Severity = Literal["info", "warn", "error"]

_ANIMA_EXTRA_ARGS_CRITICAL: frozenset[str] = frozenset(
    {
        "mixed_precision",
        "max_train_epochs",
        "max_train_steps",
        "sample_every_n_epochs",
        "sample_every_n_steps",
        "validation_split_num",
        "learning_rate",
        "network_dim",
        "network_alpha",
        "static_token_count",
        "enable_native_flatten",
        "bucket_table",
    }
)
_SAMPLE_COST_WARN_THRESHOLD = 3000
_SMALL_DATASET_WARN_THRESHOLD = 100

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
    extra: dict[str, Any] = dataclass_field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
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
    if "anima_method" not in skip_set:
        findings.extend(_check_anima_method(cfg))
    if "anima_precision" not in skip_set:
        findings.extend(_check_anima_precision(cfg))
    if "sampling_cost" not in skip_set:
        findings.extend(_check_sampling_cost(cfg))
    if "validation_split" not in skip_set:
        findings.extend(_check_validation_split(cfg))
    if "extra_args" not in skip_set:
        findings.extend(_check_extra_args(cfg))
    if "optional_dependencies" not in skip_set:
        findings.extend(_check_optional_dependencies(cfg))

    return findings


# --------------------------------------------------------------------- #
# Individual checks
# --------------------------------------------------------------------- #
def _check_model_files(cfg: TrainingConfig) -> list[PreflightFinding]:
    """Verify base model + VAE + arch_paths + dp model_paths files."""
    out: list[PreflightFinding] = []

    ckpt = cfg.base_model.checkpoint
    if (
        ckpt is None
        or not _checkpoint_is_remote_model(cfg, ckpt)
        and not _path_exists_as_file_or_dir(ckpt)
    ):
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


def _checkpoint_is_remote_model(cfg: TrainingConfig, ckpt: Path) -> bool:
    """ai-toolkit Krea2 accepts Hugging Face repo IDs as model.name_or_path."""
    if cfg.backend.type != "ai_toolkit" or cfg.base_model.arch != "krea2":
        return False
    value = str(ckpt).strip().replace("\\", "/")
    return value in {"krea/Krea-2-Raw", "krea/Krea-2-Turbo"}


def _check_dataset(cfg: TrainingConfig) -> list[PreflightFinding]:
    """Verify dataset.source / subsets / reg_source / conditioning_dir."""
    out: list[PreflightFinding] = []
    ds = cfg.dataset

    # Kohya, diffusion-pipe, and ai-toolkit use an explicit subset list in
    # place of source. Anima reuses the same schema field only to carry its
    # conditioning reference directory, while its trainer always consumes
    # source. Treating Anima's reference row as a training subset made a
    # valid source look missing and rejected imported recipes with no
    # redundant subset.path.
    use_subset_roots = bool(ds.subsets) and cfg.backend.type != "anima_lora"
    if not use_subset_roots:
        if ds.source is None or not Path(str(ds.source)).is_dir():
            out.append(
                PreflightFinding(
                    category="dataset_missing",
                    severity="error",
                    field="dataset.source",
                    message=(
                        f"Dataset directory not found: {ds.source!s}"
                        if ds.source is not None
                        else "dataset.source is empty"
                    ),
                    remediation=(
                        "Pick an existing image directory in the dataset field."
                        if cfg.backend.type == "anima_lora"
                        else (
                            "Pick an existing image directory in the dataset "
                            "field, or add at least one entry under "
                            "dataset.subsets."
                        )
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

    if cfg.dataset.subsets and cfg.backend.type != "anima_lora":
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
        "anima_lora": ["train.py"],
        "ai_toolkit": ["run.py"],
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
    """Ensure the backend's resolved Python environment can start."""
    from lorahub.core.backends._common import bootstrap as common  # noqa: PLC0415
    from lorahub.core.backends.errors import BootstrapError  # noqa: PLC0415

    out: list[PreflightFinding] = []
    if cfg.backend.type == "kohya":
        from lorahub.core.backends.kohya import bootstrap as kohya_bootstrap  # noqa: PLC0415

        repo = (
            cfg.backend.repo_path
            or common.path_from_env(kohya_bootstrap._ENV_SD_SCRIPTS)
            or kohya_bootstrap.default_sd_scripts_path()
        )
        python_env_var = kohya_bootstrap._ENV_PYTHON
    elif cfg.backend.type == "diffusion-pipe":
        from lorahub.core.backends.diffusion_pipe import (  # noqa: PLC0415
            bootstrap as dp_bootstrap,
        )

        repo = (
            cfg.backend.repo_path
            or common.path_from_env(dp_bootstrap._ENV_REPO)
            or dp_bootstrap.default_repo_path()
        )
        python_env_var = dp_bootstrap._ENV_PYTHON
    elif cfg.backend.type == "anima_lora":
        from lorahub.core.backends.anima_lora import (  # noqa: PLC0415
            bootstrap as anima_bootstrap,
        )

        repo = (
            cfg.backend.repo_path
            or common.path_from_env(anima_bootstrap._ENV_REPO)
            or anima_bootstrap.default_repo_path()
        )
        python_env_var = anima_bootstrap._ENV_PYTHON
    else:
        from lorahub.core.backends.ai_toolkit import (  # noqa: PLC0415
            bootstrap as toolkit_bootstrap,
        )

        repo = (
            cfg.backend.repo_path
            or common.path_from_env(toolkit_bootstrap._ENV_REPO)
            or toolkit_bootstrap.default_repo_path()
        )
        python_env_var = toolkit_bootstrap._ENV_PYTHON

    try:
        python = common.resolve_python(
            repo,
            config_python=cfg.backend.python_executable,
            env_var=python_env_var,
        )
        common.check_python(python)
    except BootstrapError as exc:
        out.append(
            PreflightFinding(
                category="venv_missing",
                severity="error",
                field="backend.pythonExecutable",
                message=f"{cfg.backend.type} Python environment is not usable.",
                remediation=str(exc),
            )
        )
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
    strategy = cfg.backend.distributed.strategy
    if dispatch.mode != "distributed" and strategy == "ddp":
        return []

    out: list[PreflightFinding] = []
    if dispatch.mode != "distributed":
        out.append(
            PreflightFinding(
                category="gpu_dispatch",
                severity="error",
                field="backend.gpuDispatch.mode",
                message="FSDP / DeepSpeed ZeRO 必须启用单任务多 GPU 调度。",
                remediation=(
                    "把 backend.gpuDispatch.mode 设为 distributed，"
                    "并设置 backend.gpuDispatch.numGpus>=2；"
                    "或把 backend.distributed.strategy 改回 ddp。"
                ),
            )
        )
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
    if strategy != "ddp" and cfg.backend.type != "anima_lora":
        out.append(
            PreflightFinding(
                category="gpu_dispatch",
                severity="error",
                field="backend.distributed.strategy",
                message="FSDP / DeepSpeed ZeRO 当前只接入 anima_lora 后端。",
                remediation=(
                    "切换 backend.type=anima_lora，或把 "
                    "backend.distributed.strategy 改回 ddp。"
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
    if strategy != "ddp" and (
        cfg.backend.type == "anima_lora"
        and cfg.backend.anima_lora is not None
        and cfg.backend.anima_lora.turbo is not None
    ):
        out.append(
            PreflightFinding(
                category="gpu_dispatch",
                severity="error",
                field="backend.distributed.strategy",
                message="anima_lora turbo 蒸馏当前不支持 FSDP / DeepSpeed ZeRO。",
                remediation="关闭 turbo，或把 backend.distributed.strategy 改回 ddp。",
            )
        )
    if strategy != "ddp" and cfg.backend.type == "anima_lora":
        out.append(
            PreflightFinding(
                category="gpu_dispatch",
                severity="warn",
                field="backend.distributed.strategy",
                message="anima_lora 的 FSDP / DeepSpeed ZeRO 当前仍是实验性路径。",
                remediation=(
                    "优先使用 DDP。只有在 DDP 单卡显存不够时再测试 FSDP/ZeRO，"
                    "并先用小步数 smoke run 验证保存、恢复和采样。"
                ),
            )
        )
        opts = cfg.backend.anima_lora
        if opts is not None and (
            opts.compile_mode is not None
            or opts.blocks_to_swap > 0
            or opts.gradient_checkpointing
            or opts.unsloth_offload_checkpointing
            or opts.cpu_offload_checkpointing
            or opts.ema
        ):
            out.append(
                PreflightFinding(
                    category="gpu_dispatch",
                    severity="error",
                    field="backend.distributed.strategy",
                    message=(
                        "FSDP / ZeRO 暂不支持 anima_lora 的 torch.compile、"
                        "block swap、gradient checkpoint/offload 或 EMA 组合。"
                    ),
                    remediation=(
                        "把 distributed.strategy 改回 ddp；或关闭 compileMode、"
                        "blocksToSwap、gradientCheckpointing、offload 和 EMA 后，"
                        "再用短任务验证 FSDP/ZeRO。"
                    ),
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
    if strategy != "ddp" and requested < 2:
        out.append(
            PreflightFinding(
                category="gpu_dispatch",
                severity="error",
                field="backend.distributed.strategy",
                message="FSDP / DeepSpeed ZeRO 至少需要 2 张 GPU。",
                remediation=(
                    "把 backend.gpuDispatch.numGpus 设为 2 或更高，"
                    "或把 backend.distributed.strategy 改回 ddp。"
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


def _check_anima_method(cfg: TrainingConfig) -> list[PreflightFinding]:
    opts = cfg.backend.anima_lora
    if cfg.backend.type != "anima_lora" or opts is None:
        return []
    out: list[PreflightFinding] = []
    if opts.method == "ip_adapter":
        out.append(
            PreflightFinding(
                category="anima_method",
                severity="error",
                field="backend.animaLora.method",
                message="IP-Adapter 的 PE feature cache 尚未接入 LoraHub 自动预处理。",
                remediation=(
                    "暂时改用 method=lora / easycontrol；等 cache_pe_encoder.py "
                    "纳入自动预处理后再启用 IP-Adapter。"
                ),
            )
        )
    if opts.method == "easycontrol":
        has_cond_dir = _first_conditioning_dir(cfg) is not None
        if not has_cond_dir:
            out.append(
                PreflightFinding(
                    category="anima_method",
                    severity="error",
                    field="dataset.subsets.conditioningDataDir",
                    message="EasyControl 需要同名参考图目录。",
                    remediation=(
                        "在数据集子集里设置 conditioningDataDir，或把 "
                        "backend.animaLora.method 改回 lora。"
                    ),
                )
            )
    if opts.conditioning and _first_conditioning_dir(cfg) is None:
        out.append(
            PreflightFinding(
                category="anima_method",
                severity="error",
                field="dataset.subsets.conditioningDataDir",
                message="conditioning=true 需要同名参考图目录。",
                remediation=(
                    "在数据集子集里设置 conditioningDataDir，或关闭 "
                    "backend.animaLora.conditioning。"
                ),
            )
        )
    if opts.masked_loss and _first_conditioning_dir(cfg) is None:
        out.append(
            PreflightFinding(
                category="anima_method",
                severity="error",
                field="backend.animaLora.maskedLoss",
                message="maskedLoss=true 需要 conditioningDataDir 或 alpha mask。",
                remediation=(
                    "当前 LoraHub 没有为 anima_lora 接入 alpha mask 目录。"
                    "请设置 conditioningDataDir，或关闭 maskedLoss。"
                ),
            )
        )

    if opts.method == "easycontrol" or opts.conditioning or opts.masked_loss:
        cond_dir = _first_conditioning_dir(cfg)
        if cond_dir is not None:
            out.extend(_check_conditioning_pairs(cfg, cond_dir))
    return out


def _check_anima_precision(cfg: TrainingConfig) -> list[PreflightFinding]:
    opts = cfg.backend.anima_lora
    if cfg.backend.type != "anima_lora" or opts is None:
        return []
    precision = (opts.mixed_precision or cfg.precision).lower()
    if precision != "fp16":
        return []

    slots = _nvidia_slots()
    risky = [
        slot
        for slot in slots
        if _is_pre_ampere(slot.compute_capability) or "v100" in slot.name.lower()
    ]
    if not risky:
        return []

    names = ", ".join(f"{slot.index}:{slot.name}" for slot in risky)
    return [
        PreflightFinding(
            category="precision",
            severity="warn",
            field="backend.animaLora.mixedPrecision",
            message=f"检测到 V100/Volta 级 GPU 使用 anima_lora fp16：{names}。",
            remediation=(
                "Anima DiT 的 fp16 路径在 V100 上更容易出现 NaN 或黑图。"
                "优先改为 backend.animaLora.mixedPrecision=fp32；"
                "如必须 fp16，请降低 learningRate / networkDim 并先短步数验证。"
            ),
            extra={
                "gpus": [
                    {
                        "index": slot.index,
                        "name": slot.name,
                        "compute_capability": slot.compute_capability,
                    }
                    for slot in risky
                ]
            },
        )
    ]


def _check_sampling_cost(cfg: TrainingConfig) -> list[PreflightFinding]:
    opts = cfg.backend.anima_lora
    sampling = cfg.sampling
    if cfg.backend.type != "anima_lora" or opts is None or not sampling.enabled:
        return []

    prompt_steps = [
        int(prompt.steps or opts.validation_sample_steps or sampling.inference_steps)
        for prompt in sampling.prompts
    ]
    if not prompt_steps and sampling.prompts_file is None:
        # LoraHub materialises one safe default prompt for anima_lora.
        prompt_steps = [int(opts.validation_sample_steps or sampling.inference_steps)]
    prompt_count = len(prompt_steps) if prompt_steps else 1
    steps_per_round = sum(prompt_steps) if prompt_steps else int(
        opts.validation_sample_steps or sampling.inference_steps
    )

    rounds = 1 if sampling.at_first else 0
    epochs = int(opts.max_train_epochs or cfg.schedule.epochs)
    if sampling.every_n_epochs:
        rounds += max(1, epochs // int(sampling.every_n_epochs))
    if cfg.schedule.max_steps and sampling.every_n_steps:
        rounds += max(1, int(cfg.schedule.max_steps) // int(sampling.every_n_steps))

    total_sample_steps = rounds * steps_per_round
    if total_sample_steps < _SAMPLE_COST_WARN_THRESHOLD:
        return []
    return [
        PreflightFinding(
            category="sampling_cost",
            severity="warn",
            field="sampling.prompts",
            message=(
                f"采样配置偏重：约 {rounds} 轮 × {prompt_count} 个提示词，"
                f"合计 {total_sample_steps} 个采样 step。"
            ),
            remediation=(
                "减少提示词数量，或提高 sampling.everyNEpochs / "
                "sampling.everyNSteps。训练稳定前建议只保留 1-3 个固定提示词。"
            ),
            extra={
                "rounds": rounds,
                "prompt_count": prompt_count,
                "total_sample_steps": total_sample_steps,
            },
        )
    ]


def _check_validation_split(cfg: TrainingConfig) -> list[PreflightFinding]:
    opts = cfg.backend.anima_lora
    if (
        cfg.backend.type != "anima_lora"
        or opts is None
        or opts.validation_split_num <= 0
    ):
        return []

    total = _dataset_image_count(cfg, limit=_SMALL_DATASET_WARN_THRESHOLD + 1)
    if total is None or total >= _SMALL_DATASET_WARN_THRESHOLD:
        return []
    if opts.validation_split_num <= max(4, total // 10):
        return []
    return [
        PreflightFinding(
            category="validation_split",
            severity="warn",
            field="backend.animaLora.validationSplitNum",
            message=(
                f"数据集约 {total} 张图，但 validationSplitNum="
                f"{opts.validation_split_num}，验证集占比偏高。"
            ),
            remediation=(
                "小数据集建议 validationSplitNum=0 或 4。"
                "验证集过大时训练集变少，风格 LoRA 更容易学不实。"
            ),
            extra={"image_count": total, "validation_split_num": opts.validation_split_num},
        )
    ]


def _check_extra_args(cfg: TrainingConfig) -> list[PreflightFinding]:
    if cfg.backend.type != "anima_lora" or not cfg.backend.extra_args:
        return []

    out: list[PreflightFinding] = []
    for key in sorted(cfg.backend.extra_args):
        if key not in _ANIMA_EXTRA_ARGS_CRITICAL:
            continue
        out.append(
            PreflightFinding(
                category="extra_args",
                severity="warn",
                field=f"backend.extraArgs.{key}",
                message=f"extraArgs.{key} 会覆盖表单中对应的 anima_lora 编译结果。",
                remediation=(
                    "优先使用表单字段。只有调试上游新参数时才保留 extraArgs，"
                    "并在任务备注中记录覆盖原因。"
                ),
                extra={"key": key},
            )
        )
    return out


def _check_optional_dependencies(cfg: TrainingConfig) -> list[PreflightFinding]:
    if cfg.backend.type != "anima_lora" or not cfg.monitoring.enable_wandb:
        return []

    python = _resolve_anima_python(cfg)
    if python is None:
        return []
    probe = _probe_python_import(python, "wandb")
    if probe is True:
        return []

    detail = "未安装" if probe is False else "无法确认"
    return [
        PreflightFinding(
            category="optional_dependencies",
            severity="warn",
            field="monitoring.enableWandb",
            message=f"已启用 W&B，但 anima_lora 运行环境中 wandb {detail}。",
            remediation=(
                f"建议先执行 `{python} -m pip install wandb`，"
                "或关闭 monitoring.enableWandb。否则启动训练时会临时安装，"
                "网络慢时会卡在任务启动阶段。"
            ),
            extra={"python": str(python), "package": "wandb", "probe": probe},
        )
    ]


def _homogeneous_gpu_groups() -> list[list[int]]:
    try:
        from lorahub.api.gpu_topology import homogeneous_slot_groups  # noqa: PLC0415

        return homogeneous_slot_groups()
    except Exception:  # noqa: BLE001
        return []


def _nvidia_slots() -> list[Any]:
    try:
        from lorahub.api.gpu_topology import nvidia_slots  # noqa: PLC0415

        return nvidia_slots()
    except Exception:  # noqa: BLE001
        return []


def _first_conditioning_dir(cfg: TrainingConfig) -> Path | None:
    for subset in cfg.dataset.subsets:
        if subset.conditioning_data_dir is not None:
            return Path(str(subset.conditioning_data_dir))
    return None


def _resolve_anima_python(cfg: TrainingConfig) -> Path | None:
    try:
        from lorahub.core.backends.anima_lora import bootstrap  # noqa: PLC0415

        env = bootstrap.resolve(
            config_path=cfg.backend.repo_path,
            config_python=cfg.backend.python_executable,
        )
    except Exception:  # noqa: BLE001
        return None
    return env.python_executable


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


def _dataset_image_count(cfg: TrainingConfig, *, limit: int) -> int | None:
    roots = _active_training_roots(cfg)
    total = 0
    for raw in roots:
        if raw is None:
            continue
        root = Path(str(raw))
        if not root.is_dir():
            continue
        total += _count_with_extensions(root, _IMAGE_EXTS, limit=max(1, limit - total))
        if total >= limit:
            return total
    return total if total > 0 else None


def _dataset_image_paths(cfg: TrainingConfig, *, limit: int) -> list[Path]:
    roots = _active_training_roots(cfg)
    out: list[Path] = []
    for raw in roots:
        if raw is None:
            continue
        root = Path(str(raw))
        if not root.is_dir():
            continue
        try:
            for sub_root, _dirs, files in os.walk(root):
                for name in files:
                    path = Path(sub_root) / name
                    if path.suffix.lower() in _IMAGE_EXTS:
                        out.append(path)
                        if len(out) >= limit:
                            return out
                if sub_root != str(root):
                    continue
        except OSError:
            continue
    return out


def _check_conditioning_pairs(
    cfg: TrainingConfig,
    cond_dir: Path,
) -> list[PreflightFinding]:
    if not cond_dir.is_dir():
        return [
            PreflightFinding(
                category="anima_method",
                severity="error",
                field="dataset.subsets.conditioningDataDir",
                message=f"conditioningDataDir 不存在：{cond_dir!s}",
                remediation="选择一个存在的参考图目录，或关闭 conditioning / maskedLoss。",
            )
        ]

    images = _dataset_image_paths(cfg, limit=20)
    if not images:
        return []
    missing = [
        path.name
        for path in images
        if not _has_conditioning_pair(path, cfg, cond_dir)
    ]
    if not missing:
        return []
    severity: Severity = "error" if len(missing) == len(images) else "warn"
    return [
        PreflightFinding(
            category="anima_method",
            severity=severity,
            field="dataset.subsets.conditioningDataDir",
            message=(
                f"conditioningDataDir 中有 {len(missing)}/{len(images)} 个样本"
                "未找到同名参考图。"
            ),
            remediation=(
                "参考图需要与训练图同 stem，可使用相同子目录结构；"
                "支持 png/jpg/jpeg/webp/bmp。"
            ),
            extra={"missing_sample_names": missing[:10]},
        )
    ]


def _has_conditioning_pair(image_path: Path, cfg: TrainingConfig, cond_dir: Path) -> bool:
    rel_dir = Path()
    roots = _active_training_roots(cfg)
    for raw in roots:
        if raw is None:
            continue
        root = Path(str(raw))
        try:
            rel = image_path.parent.relative_to(root)
        except ValueError:
            continue
        rel_dir = rel
        break

    suffixes = [image_path.suffix.lower()] if image_path.suffix else []
    for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        if ext not in suffixes:
            suffixes.append(ext)
    for ext in suffixes:
        if (cond_dir / rel_dir / f"{image_path.stem}{ext}").is_file():
            return True
    return False


def _active_training_roots(cfg: TrainingConfig) -> list[Path | None]:
    """Return the directories a backend actually feeds to its trainer."""
    if cfg.dataset.subsets and cfg.backend.type != "anima_lora":
        return [subset.path for subset in cfg.dataset.subsets]
    return [cfg.dataset.source]


def _is_pre_ampere(compute_capability: str | None) -> bool:
    if not compute_capability:
        return False
    try:
        major = int(str(compute_capability).split(".", 1)[0])
    except (TypeError, ValueError):
        return False
    return major < 8


def _probe_python_import(python: Path, module: str) -> bool | None:
    try:
        proc = subprocess.run(  # noqa: S603
            [str(python), "-c", f"import {module}"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
            if sys.platform == "win32"
            else 0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.returncode == 0


def _to_camel(snake: str) -> str:
    parts = snake.split("_")
    if not parts:
        return snake
    return parts[0] + "".join(p.title() for p in parts[1:])


__all__ = ["PreflightFinding", "run_preflight"]
