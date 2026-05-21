"""Run a turbo distillation subprocess (no accelerate launch wrapper).

distill_turbo.py is single-process: it manages its own optimizer +
checkpoint cycle on one GPU and doesn't go through HuggingFace
Accelerate. So unlike :class:`AnimaLoraRunner` which wraps train.py
in ``python -m accelerate.commands.accelerate_cli launch ...``, this
runner spawns a plain ``<python> scripts/distill_turbo.py <args>``.

Same SubprocessRunner contract — only the argv frame and parser
change.
"""

from __future__ import annotations

from pathlib import Path

from lorahub.core.backends._common.runner import (
    EventListener,
    RunResult,
    SubprocessRunner,
)
from lorahub.core.backends.anima_lora.turbo_parser import parse_line

__all__ = ["AnimaLoraTurboRunner", "RunResult"]


class AnimaLoraTurboRunner(SubprocessRunner):
    """Runs ``<python> external/anima_lora/scripts/distill_turbo.py <argv>``."""

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
        distill_py = repo / "scripts" / "distill_turbo.py"
        full_argv = [str(python), str(distill_py), *argv]
        # Inherit the anima ``SyntaxWarning`` filter (vendored
        # text_strategies.py uses unraw escapes). Same rationale as
        # AnimaLoraRunner — kept inside the backend so kohya/dp
        # subprocesses don't get the filter.
        merged_env = {**(env or {})}
        anima_filter = "ignore:invalid escape sequence:SyntaxWarning"
        existing = merged_env.get("PYTHONWARNINGS", "")
        merged_env["PYTHONWARNINGS"] = (
            f"{anima_filter},{existing}" if existing else anima_filter
        )
        # cwd=repo so the script can resolve relative paths in its own
        # `configs/methods/turbo.toml` defaults (model paths,
        # post_image_dataset/...) without the caller pre-resolving.
        super().__init__(
            argv=full_argv,
            workspace=workspace,
            on_event=on_event,
            parse_line=parse_line,
            cwd=repo,
            job_id=job_id,
            env=merged_env,
            thread_label="anima_lora_turbo",
        )
