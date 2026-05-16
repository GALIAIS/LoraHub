"""DiffusionPipeBackend: wraps tdrussell/diffusion-pipe as a TrainingBackend.

Translates a ``RecipeConfig`` into the TOML config files diffusion-pipe expects
(see `compiler.py`), writes them under the workspace, and launches
``python train.py --deepspeed --config <toml>`` through the shared
``SubprocessRunner`` (see `runner.py`).

We deliberately keep the supported arch set narrower than kohya's: upstream
diffusion-pipe doesn't ship an SD1.5 trainer, so SDXL / Flux / SD3 is the
honest list. Recipes that target sd15 fail validation with a clear pointer
back to the kohya backend.
"""

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
from lorahub.core.backends.diffusion_pipe import bootstrap as _bootstrap
from lorahub.core.backends.diffusion_pipe.compiler import (
    _DP_MODEL_TYPE_MAP,
    CompilationError,
    compile_recipe,
)
from lorahub.core.backends.diffusion_pipe.runner import DiffusionPipeRunner
from lorahub.core.config.schema import RecipeConfig
from lorahub.core.events import TrainingEvent

# diffusion-pipe ships trainers for every entry in `_DP_MODEL_TYPE_MAP`
# (SDXL, SD3, Flux/Flux2, Lumina2, Chroma, HiDream, OmniGen2, AuraFlow,
# Qwen-Image, Cosmos, HunyuanImage/Video, LTX-Video, Wan, Z-Image, ...).
# SD1.5 / SD2 are intentionally absent because upstream's
# `docs/supported_models.md` does not document a trainer for them; recipes
# targeting those arches fall through to the kohya backend.
_SUPPORTED: set[ModelArch] = {ModelArch(arch) for arch in _DP_MODEL_TYPE_MAP}


class DiffusionPipeBackend:
    """Wraps tdrussell/diffusion-pipe as a TrainingBackend."""

    @property
    def name(self) -> str:
        return "diffusion-pipe"

    @property
    def supported_archs(self) -> set[ModelArch]:
        return set(_SUPPORTED)

    def validate(self, cfg: RecipeConfig) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        if cfg.base_model.arch not in {a.value for a in _SUPPORTED}:
            issues.append(
                ValidationIssue(
                    Severity.error,
                    "base_model.arch",
                    (
                        f"diffusion-pipe does not support arch "
                        f"{cfg.base_model.arch!r}; supported: "
                        f"{sorted(a.value for a in _SUPPORTED)}. "
                        "Switch backend.type to 'kohya' for sd15/sd2."
                    ),
                )
            )

        try:
            _bootstrap.resolve(
                recipe_path=cfg.backend.sd_scripts_path,
                recipe_python=cfg.backend.python_executable,
            )
        except _bootstrap.BootstrapError as e:
            issues.append(ValidationIssue(Severity.error, "backend.repo_path", str(e)))

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
        """Coarse first-pass VRAM estimate.

        Reuses the shared ``_common.vram`` table so the kohya and
        diffusion-pipe backends agree on the numbers. ``sd15`` / ``sd2``
        recipes never reach ``launch`` here (``validate`` already errors),
        but they still get a sensible estimate so the UI can display one
        before the user switches backends.
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
        argv, files = compile_recipe(cfg, workspace)
        if extra_argv:
            argv = [*argv, *extra_argv]
        workspace.mkdir(parents=True, exist_ok=True)
        for path, content in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

        job_id = str(ulid.new())
        runner = DiffusionPipeRunner(
            python=bootstrap_env.python_executable,
            repo=bootstrap_env.repo_path,
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


__all__ = ["DiffusionPipeBackend"]
