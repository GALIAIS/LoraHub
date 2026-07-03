"""TrainingBackend wrapper for the vendored ostris/ai-toolkit."""

from __future__ import annotations

import os
import threading
import time
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
from lorahub.core.events import EventType, TrainingEvent
from lorahub.core.paths import project_root

_SUPPORTED: set[ModelArch] = {ModelArch.krea2}
_SAMPLE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
_SAMPLE_POLL_INTERVAL = 3.0
_SAMPLE_GRACE_AFTER_EXIT = 8.0


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
        _start_sample_watcher(
            sample_root=workspace / "ai_toolkit_output",
            workspace=workspace,
            on_event=on_event,
            runner=runner,
            job_id=job_id,
        )
        return TrainingHandle(
            job_id=job_id,
            pid=runner.pid,
            _stop_fn=lambda graceful: runner.stop(graceful=graceful),
            _wait_fn=lambda timeout: runner.wait(timeout=timeout).returncode,
        )


def _scan_new_samples(
    sample_root: Path,
    workspace: Path,
    seen: set[str],
    *,
    job_id: str | None = None,
) -> list[TrainingEvent]:
    events: list[TrainingEvent] = []
    if not sample_root.is_dir():
        return events
    for path in sample_root.rglob("*"):
        if path.suffix.lower() not in _SAMPLE_SUFFIXES or not path.is_file():
            continue
        rel = path.relative_to(workspace).as_posix()
        if rel in seen:
            continue
        seen.add(rel)
        try:
            stat = path.stat()
        except OSError:
            continue
        events.append(
            TrainingEvent(
                type=EventType.sample_ready,
                payload={
                    "path": rel,
                    "size_bytes": stat.st_size,
                    "modified_at": stat.st_mtime,
                    "filename": path.name,
                },
                job_id=job_id,
            )
        )
    return events


def _start_sample_watcher(
    *,
    sample_root: Path,
    workspace: Path,
    on_event: Callable[[TrainingEvent], None],
    runner: AIToolkitRunner,
    job_id: str,
) -> None:
    seen: set[str] = set()

    def watch() -> None:
        from lorahub.api.store import _pid_alive  # noqa: PLC0415

        exit_seen_at: float | None = None
        while True:
            try:
                for event in _scan_new_samples(
                    sample_root, workspace, seen, job_id=job_id
                ):
                    on_event(event)
            except Exception:  # noqa: BLE001
                pass

            pid = runner.pid
            alive = pid is not None and _pid_alive(pid)
            if not alive:
                if exit_seen_at is None:
                    exit_seen_at = time.time()
                elif time.time() - exit_seen_at > _SAMPLE_GRACE_AFTER_EXIT:
                    return
            else:
                exit_seen_at = None
            time.sleep(_SAMPLE_POLL_INTERVAL)

    threading.Thread(
        target=watch,
        name=f"ai-toolkit-samples-{job_id[-6:]}",
        daemon=True,
    ).start()


__all__ = ["AIToolkitBackend"]
