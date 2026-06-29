"""Run ai-toolkit through the shared subprocess runner."""

from __future__ import annotations

from pathlib import Path

from lorahub.core.backends._common.runner import (
    EventListener,
    RunResult,
    SubprocessRunner,
)
from lorahub.core.backends.ai_toolkit.parser import parse_line

__all__ = ["AIToolkitRunner", "RunResult"]


class AIToolkitRunner(SubprocessRunner):
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
        super().__init__(
            argv=[str(python), str(repo / "run.py"), *argv],
            workspace=workspace,
            on_event=on_event,
            parse_line=parse_line,
            cwd=repo,
            job_id=job_id,
            env=env,
            thread_label="ai-toolkit",
        )
