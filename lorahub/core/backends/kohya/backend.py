"""KohyaBackend: implements `TrainingBackend` by wrapping kohya_ss/sd-scripts."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import ulid

from lorahub.core.backends._common.vram import estimate_vram as _shared_estimate_vram
from lorahub.core.backends.base import (
    ModelArch,
    Severity,
    TrainingHandle,
    ValidationIssue,
    VRAMEstimate,
)
from lorahub.core.backends.kohya import bootstrap as _bootstrap
from lorahub.core.backends.kohya.compiler import (
    CompilationError,
    _KOHYA_SCRIPT_MAP,
    compile_recipe,
)
from lorahub.core.backends.kohya.runner import KohyaRunner
from lorahub.core.config.schema import RecipeConfig
from lorahub.core.events import TrainingEvent

# kohya sd-scripts ships dedicated entry points for these arches today
# (see compiler._KOHYA_SCRIPT_MAP). diffusion-pipe-only entries (Wan,
# HunyuanVideo, Cosmos, Chroma, ...) are intentionally excluded.
_SUPPORTED: set[ModelArch] = {ModelArch(arch) for arch in _KOHYA_SCRIPT_MAP}


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
        """Coarse VRAM estimate. Refine with empirical data later.

        The per-arch tables and the formula live in
        ``lorahub.core.backends._common.vram`` so the kohya and
        diffusion-pipe backends stay in lockstep.
        """
        return _shared_estimate_vram(
            cfg.base_model.arch,
            precision=cfg.precision,
            batch_size=cfg.schedule.batch_size,
            network_rank=cfg.network.rank,
            gradient_checkpointing=cfg.gradient_checkpointing,
        )

    def launch(
        self,
        cfg: RecipeConfig,
        workspace: Path,
        on_event: Callable[[TrainingEvent], None],
        *,
        extra_argv: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> TrainingHandle:
        bootstrap_env = _bootstrap.resolve(
            recipe_path=cfg.backend.sd_scripts_path,
            recipe_python=cfg.backend.python_executable,
        )
        workspace = workspace.resolve()
        script_name, argv, files = compile_recipe(cfg, workspace)
        if extra_argv:
            argv = [*argv, *extra_argv]
        workspace.mkdir(parents=True, exist_ok=True)
        for path, content in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        script = bootstrap_env.script(script_name)

        job_id = str(ulid.new())
        runner = KohyaRunner(
            python=bootstrap_env.python_executable,
            script=script,
            argv=argv,
            workspace=workspace,
            on_event=on_event,
            job_id=job_id,
            env=env,
        )
        runner.start()

        return TrainingHandle(
            job_id=job_id,
            pid=runner.pid,
            _stop_fn=lambda graceful: runner.stop(graceful=graceful),
            _wait_fn=lambda timeout: runner.wait(timeout=timeout).returncode,
        )
