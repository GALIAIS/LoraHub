"""Run a diffusion-pipe training job as a subprocess and stream events.

Composes the generic ``SubprocessRunner`` from ``_common`` with the
diffusion-pipe line parser. Built so callers don't need to think about the
deepspeed launcher shape: pass the env's python, the `train.py` path, and
the argv (typically `--deepspeed --config <toml>`), and we'll do the rest.
"""

from __future__ import annotations

from pathlib import Path

from lorahub.core.backends._common.runner import (
    EventListener,
    RunResult,
    SubprocessRunner,
)
from lorahub.core.backends.diffusion_pipe.parser import parse_line

__all__ = ["DiffusionPipeRunner", "RunResult"]


class DiffusionPipeRunner(SubprocessRunner):
    """Runs `python <repo>/train.py --deepspeed --config <toml> ...`."""

    def __init__(
        self,
        python: Path,
        repo: Path,
        argv: list[str],
        workspace: Path,
        on_event: EventListener,
        *,
        job_id: str | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        train_py = repo / "train.py"
        super().__init__(
            argv=[str(python), str(train_py), *argv],
            workspace=workspace,
            on_event=on_event,
            parse_line=parse_line,
            cwd=repo,
            job_id=job_id,
            env=env,
            thread_label="diffusion-pipe",
        )
