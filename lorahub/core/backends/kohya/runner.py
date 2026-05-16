"""Run a kohya training script as a subprocess and stream events.

Thin wrapper over `SubprocessRunner` that wires kohya's parser in and
keeps the historical `KohyaRunner(python=, script=, argv=, ...)` ctor
shape intact.

Each runner owns a fresh `KohyaLineParser` so per-job state (traceback
aggregation, cache-progress throttling) is isolated.
"""

from __future__ import annotations

from pathlib import Path

from lorahub.core.backends._common.runner import (
    EventListener,
    RunResult,
    SubprocessRunner,
)
from lorahub.core.backends.kohya.parser import KohyaLineParser

__all__ = ["KohyaRunner", "RunResult"]


class KohyaRunner(SubprocessRunner):
    """Runs a kohya script under a chosen Python and parses kohya stdout."""

    def __init__(
        self,
        python: Path,
        script: Path,
        argv: list[str],
        workspace: Path,
        on_event: EventListener,
        *,
        job_id: str | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        parser = KohyaLineParser()
        super().__init__(
            argv=[str(python), str(script), *argv],
            workspace=workspace,
            on_event=on_event,
            parse_line=parser.parse_line,
            cwd=script.parent,
            job_id=job_id,
            env=env,
            thread_label="kohya",
        )
