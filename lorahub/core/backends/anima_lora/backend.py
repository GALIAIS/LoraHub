"""AnimaLoraBackend: wraps the vendored sorryhyun/anima_lora as a TrainingBackend.

Translates a ``TrainingConfig`` into the override-layer CLI argv anima_lora
expects (see ``compiler.py``), and launches ``<python> -m
accelerate.commands.accelerate_cli launch train.py <args>`` through the
shared ``SubprocessRunner`` (see ``runner.py``).

We deliberately keep the supported arch set narrow: anima_lora's reason
to exist is its Anima-specific algorithm stack (OrthoLoRA / T-LoRA /
Hydra / postfix / EasyControl / IP-Adapter). Recipes targeting
non-anima archs land in the kohya / dp backends instead.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import ulid

from lorahub.core.backends._common.vram import estimate_vram as _shared_estimate_vram
from lorahub.core.backends.anima_lora import bootstrap as _bootstrap
from lorahub.core.backends.anima_lora.compiler import (
    CompilationError,
    compile_config,
    compile_turbo_config,
)
from lorahub.core.backends.anima_lora.preprocess import (
    PreprocessError,
    ensure_cache,
)
from lorahub.core.backends.anima_lora.runner import AnimaLoraRunner
from lorahub.core.backends.anima_lora.turbo_runner import AnimaLoraTurboRunner
from lorahub.core.backends.base import (
    ModelArch,
    Severity,
    TrainingHandle,
    ValidationIssue,
    VRAMEstimate,
)
from lorahub.core.config.schema import TrainingConfig
from lorahub.core.events import EventType, TrainingEvent

# anima_lora is purpose-built for Anima DiT; everything else falls
# through to kohya / dp. Keeping this set tight means the validator
# can give a clear error when a user accidentally points the wrong
# arch at this backend.
_SUPPORTED: set[ModelArch] = {ModelArch.anima}


class AnimaLoraBackend:
    """Wraps the vendored sorryhyun/anima_lora source as a TrainingBackend."""

    @property
    def name(self) -> str:
        return "anima_lora"

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
                    (
                        f"anima_lora does not support arch {cfg.base_model.arch!r}; "
                        f"supported: {sorted(a.value for a in _SUPPORTED)}. "
                        "Switch backend.type to 'kohya' or 'diffusion-pipe' "
                        "for other arches."
                    ),
                )
            )

        # Bootstrap probe against the vendored copy. The repo_path field
        # is reused from BackendConfig; users normally leave it None and
        # we resolve to external/anima_lora/ automatically.
        try:
            _bootstrap.resolve(
                config_path=cfg.backend.sd_scripts_path,
                config_python=cfg.backend.python_executable,
            )
        except _bootstrap.BootstrapError as e:
            issues.append(
                ValidationIssue(Severity.error, "backend.python_executable", str(e))
            )

        # Pass an arbitrary workspace — compile_config doesn't write
        # files (anima_lora owns its own merge chain) so the path is
        # only used to construct --output_dir.
        try:
            compile_config(cfg, workspace=Path("/"))
        except CompilationError as e:
            issues.append(ValidationIssue(Severity.error, "backend.animaLora", str(e)))

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

    def estimate_vram(self, cfg: TrainingConfig) -> VRAMEstimate:
        """Reuses the shared `_common.vram` anima entry.

        Upstream reports 13.4 GB peak at rank=32, 1MP on a 5060 Ti, but
        the shared estimator is conservative and works off the
        precision / batch_size / rank knobs we already track. Good
        enough for the UI to flag ``estimated > available_vram`` cases
        before launch; a tighter model can land later if needed.
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
        cfg: TrainingConfig,
        workspace: Path,
        on_event: Callable[[TrainingEvent], None],
        *,
        extra_argv: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> TrainingHandle:
        bootstrap_env = _bootstrap.resolve(
            config_path=cfg.backend.sd_scripts_path,
            config_python=cfg.backend.python_executable,
        )
        workspace = workspace.resolve()
        workspace.mkdir(parents=True, exist_ok=True)

        # Auto-preprocess: ensure the LoRA cache under
        # <workspace>/post_image_dataset/lora is populated before the
        # trainer reads it. This keeps cfg.dataset.source pointing at
        # the user's raw image dir (same shape kohya / dp use) instead
        # of forcing them to ``make preprocess`` separately. Failures
        # turn into a CompilationError so the launcher returns a clear
        # error rather than a half-running training subprocess.
        try:
            ensure_cache(
                image_dir=cfg.dataset.source,
                workspace=workspace,
                base_model=cfg.base_model,
                env=bootstrap_env,
                on_event=on_event,
            )
        except PreprocessError as e:
            on_event(
                TrainingEvent(
                    type=EventType.error,
                    payload={"source": "preprocess", "error": str(e)},
                )
            )
            msg = f"anima_lora auto-preprocess failed: {e}"
            raise CompilationError(msg) from e

        # Branch: turbo distillation (scripts/distill_turbo.py) vs the
        # regular train.py path. Turbo is picked when the recipe has
        # backend.animaLora.turbo populated; both paths share workspace
        # setup but diverge on argv shape + runner choice.
        opts = cfg.backend.anima_lora
        is_turbo = opts is not None and opts.turbo is not None
        if is_turbo:
            argv, files = compile_turbo_config(cfg, workspace)
        else:
            argv, files = compile_config(cfg, workspace)
        if extra_argv:
            argv = [*argv, *extra_argv]
        # `files` is always empty for anima_lora — kept for shape parity
        # with kohya / dp launchers.
        for path, content in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

        job_id = str(ulid.new())
        runner: AnimaLoraRunner | AnimaLoraTurboRunner
        if is_turbo:
            runner = AnimaLoraTurboRunner(
                python=bootstrap_env.python_executable,
                repo=bootstrap_env.repo_path,
                argv=argv,
                workspace=workspace,
                on_event=on_event,
                job_id=job_id,
                env=env,
            )
        else:
            runner = AnimaLoraRunner(
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


__all__ = ["AnimaLoraBackend"]
