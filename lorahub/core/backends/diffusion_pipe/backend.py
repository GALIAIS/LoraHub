"""DiffusionPipeBackend: scaffold implementation of the `TrainingBackend` Protocol.

v0.2 wires up validation, VRAM estimation, and bootstrap so the user can
install + select diffusion-pipe from the UI today. Actually compiling a
recipe into diffusion-pipe's deepspeed-driven argv and launching the job is
deferred to v0.3 -- `launch()` raises `NotImplementedError` until then.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from lorahub.core.backends.base import (
    ModelArch,
    Severity,
    TrainingHandle,
    ValidationIssue,
    VRAMEstimate,
)
from lorahub.core.backends.diffusion_pipe import bootstrap as _bootstrap
from lorahub.core.config.schema import RecipeConfig
from lorahub.core.events import TrainingEvent

# diffusion-pipe focuses on Flux/SD3-class video & image diffusion models.
# The set is intentionally narrower than kohya's; we'll widen it as the
# compiler in v0.3 adds support per-arch.
_SUPPORTED: set[ModelArch] = {ModelArch.flux, ModelArch.sd3, ModelArch.sdxl}


class DiffusionPipeBackend:
    """Wraps tdrussell/diffusion-pipe as a TrainingBackend (scaffold only)."""

    @property
    def name(self) -> str:
        return "diffusion-pipe"

    @property
    def supported_archs(self) -> set[ModelArch]:
        return set(_SUPPORTED)

    def validate(self, cfg: RecipeConfig) -> list[ValidationIssue]:
        """Best-effort preflight: surfaces missing checkout / paths up-front.

        The real recipe -> argv compiler isn't wired up until v0.3, so we
        deliberately avoid asserting backend-specific recipe shape here
        beyond what the schema already enforces.
        """
        issues: list[ValidationIssue] = []

        try:
            _bootstrap.resolve(
                recipe_path=cfg.backend.sd_scripts_path,
                recipe_python=cfg.backend.python_executable,
            )
        except _bootstrap.BootstrapError as e:
            issues.append(ValidationIssue(Severity.error, "backend.repo_path", str(e)))

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

        issues.append(
            ValidationIssue(
                Severity.info,
                "backend.type",
                "diffusion-pipe backend is scaffold-only in v0.2; "
                "training launch ships in v0.3.",
            )
        )

        return issues

    def estimate_vram(self, cfg: RecipeConfig) -> VRAMEstimate:
        """Coarse placeholder estimate. Refine once we run a real job through it."""
        arch = cfg.base_model.arch
        bytes_per_param = 2 if cfg.precision in ("fp16", "bf16") else 4

        # diffusion-pipe targets larger models; bias the numbers slightly.
        model_params = {"sd15": 860, "sdxl": 2600, "flux": 12000, "sd3": 2800}.get(
            arch, 2600
        )
        model_mib = model_params * bytes_per_param

        optimizer_mib = cfg.network.rank * 8
        if not cfg.gradient_checkpointing:
            optimizer_mib *= 4

        activations_mib = cfg.schedule.batch_size * (
            1024 if arch in ("sdxl", "flux", "sd3") else 512
        )
        if cfg.gradient_checkpointing:
            activations_mib //= 3

        return VRAMEstimate(
            model_mib=model_mib,
            optimizer_mib=optimizer_mib,
            activations_mib=activations_mib,
        )

    def launch(
        self,
        cfg: RecipeConfig,  # noqa: ARG002 - scaffold; signature must match Protocol
        workspace: Path,  # noqa: ARG002
        on_event: Callable[[TrainingEvent], None],  # noqa: ARG002
    ) -> TrainingHandle:
        msg = (
            "diffusion-pipe backend launch is not yet wired -- "
            "install the backend, then check back in v0.3"
        )
        raise NotImplementedError(msg)


__all__ = ["DiffusionPipeBackend"]
