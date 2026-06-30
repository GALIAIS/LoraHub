"""TrainingBackend wrapper for the vendored ostris/ai-toolkit."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import ulid

from lorahub.core.backends._common.vram import estimate_vram as _shared_estimate_vram
from lorahub.core.backends.ai_toolkit import bootstrap as _bootstrap
from lorahub.core.backends.ai_toolkit.compiler import CompilationError, compile_config
from lorahub.core.backends.ai_toolkit.runner import AIToolkitRunner
from lorahub.core.backends.base import (
    ModelArch,
    Severity,
    TrainingHandle,
    ValidationIssue,
    VRAMEstimate,
)
from lorahub.core.config.schema import TrainingConfig
from lorahub.core.events import TrainingEvent
from lorahub.core.paths import project_root

_SUPPORTED: set[ModelArch] = {ModelArch.krea2}


class AIToolkitBackend:
    @property
    def name(self) -> str:
        return "ai_toolkit"

    @property
    def supported_archs(self) -> set[ModelArch]:
        return set(_SUPPORTED)

    def validate(self, cfg: TrainingConfig) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        if cfg.base_model.arch not in {a.value for a in _SUPPORTED}:
            issues.append(
                ValidationIssue(
                    Severity.error,
                    "base_model.arch",
                    "ai_toolkit currently supports arch='krea2'.",
                )
            )
        try:
            _bootstrap.resolve(
                config_path=cfg.backend.repo_path,
                config_python=cfg.backend.python_executable,
            )
        except _bootstrap.BootstrapError as exc:
            issues.append(ValidationIssue(Severity.error, "backend.repo_path", str(exc)))
        try:
            compile_config(cfg, workspace=Path("/"))
        except CompilationError as exc:
            issues.append(ValidationIssue(Severity.error, "recipe", str(exc)))
        if not cfg.dataset.source.exists():
            issues.append(
                ValidationIssue(
                    Severity.warning,
                    "dataset.source",
                    f"dataset directory does not exist: {cfg.dataset.source}",
                )
            )
        return issues

    def estimate_vram(self, cfg: TrainingConfig) -> VRAMEstimate:
        return _shared_estimate_vram(
            cfg.base_model.arch,
            precision=cfg.precision,
            batch_size=cfg.schedule.batch_size,
            network_rank=cfg.network.rank,
            gradient_checkpointing=cfg.gradient_checkpointing,
        )

    def launch(
        self,
        cfg: TrainingConfig,
        workspace: Path,
        on_event: Callable[[TrainingEvent], None],
        *,
        extra_argv: list[str] | None = None,
        env: dict[str, str] | None = None,
        gpu_count: int = 1,
    ) -> TrainingHandle:
        bootstrap_env = _bootstrap.resolve(
            config_path=cfg.backend.repo_path,
            config_python=cfg.backend.python_executable,
        )
        workspace = workspace.resolve()
        workspace.mkdir(parents=True, exist_ok=True)

        from lorahub.core.backends._common.dataset_prep import (  # noqa: PLC0415
            apply_caption_dropouts,
        )

        apply_caption_dropouts(cfg, workspace)
        argv, files = compile_config(cfg, workspace)
        if extra_argv:
            argv = [*argv, *extra_argv]
        for path, content in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

        job_id = str(ulid.new())
        runner_env = dict(env or {})
        if (
            "HF_HOME" not in os.environ
            and "HF_HOME" not in runner_env
            and "HUGGINGFACE_HUB_CACHE" not in os.environ
            and "HUGGINGFACE_HUB_CACHE" not in runner_env
        ):
            runner_env["HF_HOME"] = str(project_root() / "models" / "huggingface")
        runner = AIToolkitRunner(
            python=bootstrap_env.python_executable,
            repo=bootstrap_env.repo_path,
            argv=argv,
            workspace=workspace,
            on_event=on_event,
            job_id=job_id,
            env=runner_env,
        )
        runner.start()
        return TrainingHandle(
            job_id=job_id,
            pid=runner.pid,
            _stop_fn=lambda graceful: runner.stop(graceful=graceful),
            _wait_fn=lambda timeout: runner.wait(timeout=timeout).returncode,
        )


__all__ = ["AIToolkitBackend"]
