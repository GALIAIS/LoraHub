"""KohyaBackend: implements `TrainingBackend` by wrapping kohya_ss/sd-scripts."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import ulid

from lorahub.core.backends.base import (
    ModelArch,
    Severity,
    TrainingHandle,
    ValidationIssue,
    VRAMEstimate,
)
from lorahub.core.backends.kohya import bootstrap as _bootstrap
from lorahub.core.backends.kohya.compiler import CompilationError, compile_recipe
from lorahub.core.backends.kohya.runner import KohyaRunner
from lorahub.core.config.schema import RecipeConfig
from lorahub.core.events import TrainingEvent

_SUPPORTED: set[ModelArch] = {ModelArch.sdxl, ModelArch.sd15, ModelArch.flux, ModelArch.sd3}


class KohyaBackend:
    """Wraps kohya_ss/sd-scripts as a TrainingBackend."""

    @property
    def name(self) -> str:
        return "kohya"

    @property
    def supported_archs(self) -> set[ModelArch]:
        return set(_SUPPORTED)

    def validate(self, cfg: RecipeConfig) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        try:
            _bootstrap.resolve(
                recipe_path=cfg.backend.sd_scripts_path,
                recipe_python=cfg.backend.python_executable,
            )
        except _bootstrap.BootstrapError as e:
            issues.append(ValidationIssue(Severity.error, "backend.sd_scripts_path", str(e)))

        try:
            compile_recipe(cfg, workspace=Path("/"))
        except CompilationError as e:
            issues.append(ValidationIssue(Severity.error, "recipe", str(e)))

        if not cfg.base_model.checkpoint.exists():
            issues.append(
                ValidationIssue(
                    Severity.warning,
                    "base_model.checkpoint",
                    f"checkpoint file does not exist: {cfg.base_model.checkpoint}",
                )
            )
        if not cfg.dataset.source.exists():
            issues.append(
                ValidationIssue(
                    Severity.warning,
                    "dataset.source",
                    f"dataset directory does not exist: {cfg.dataset.source}",
                )
            )

        return issues

    def estimate_vram(self, cfg: RecipeConfig) -> VRAMEstimate:
        """Coarse VRAM estimate. Refine in v0.2 once we have empirical data."""
        arch = cfg.base_model.arch
        bytes_per_param = 2 if cfg.precision in ("fp16", "bf16") else 4

        model_params = {"sd15": 860, "sdxl": 2600, "flux": 12000, "sd3": 2000}.get(arch, 2600)
        model_mib = model_params * bytes_per_param // 1

        optimizer_mib = cfg.network.rank * 8
        if not cfg.gradient_checkpointing:
            optimizer_mib *= 4

        activations_mib = cfg.schedule.batch_size * (1024 if arch in ("sdxl", "flux") else 512)
        if cfg.gradient_checkpointing:
            activations_mib //= 3

        return VRAMEstimate(
            model_mib=model_mib,
            optimizer_mib=optimizer_mib,
            activations_mib=activations_mib,
        )

    def launch(
        self,
        cfg: RecipeConfig,
        workspace: Path,
        on_event: Callable[[TrainingEvent], None],
    ) -> TrainingHandle:
        env = _bootstrap.resolve(
            recipe_path=cfg.backend.sd_scripts_path,
            recipe_python=cfg.backend.python_executable,
        )
        workspace = workspace.resolve()
        script_name, argv, files = compile_recipe(cfg, workspace)
        workspace.mkdir(parents=True, exist_ok=True)
        for path, content in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        script = env.script(script_name)

        job_id = str(ulid.new())
        runner = KohyaRunner(
            python=env.python_executable,
            script=script,
            argv=argv,
            workspace=workspace,
            on_event=on_event,
            job_id=job_id,
        )
        runner.start()

        return TrainingHandle(
            job_id=job_id,
            pid=runner.pid,
            _stop_fn=lambda graceful: runner.stop(graceful=graceful),
            _wait_fn=lambda timeout: runner.wait(timeout=timeout).returncode,
        )
