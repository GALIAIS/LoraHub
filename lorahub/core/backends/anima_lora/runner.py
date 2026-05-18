"""Run an anima_lora training job as a subprocess and stream events.

anima_lora drives training through HuggingFace Accelerate, not DeepSpeed
(diffusion-pipe) or a bare Python process (kohya). Upstream's
``scripts/tasks/_common.accelerate_launch`` invokes:

    <python> -m accelerate.commands.accelerate_cli launch \
        --num_cpu_threads_per_process 3 \
        --mixed_precision bf16 \
        train.py <args>

We mirror that invocation to keep behaviour identical to ``make lora``
upstream. The mixed_precision flag here is the **launcher's** mixed
precision (Accelerate's HF-side AMP routing); ``train.py`` itself takes
its own ``--mixed_precision`` from the compiler-emitted argv, which is
fine — Accelerate's launcher and the trainer agree on the same value
for a default config.

We deliberately DON'T use the ``accelerate`` console-script entry point
(``<venv>/bin/accelerate``). Going through ``python -m`` keeps
``sys.executable`` propagating to Accelerate's workers, matching what
upstream does so launching from a no-console process (e.g. pythonw)
doesn't pop a terminal.
"""

from __future__ import annotations

from pathlib import Path

from lorahub.core.backends._common.runner import (
    EventListener,
    RunResult,
    SubprocessRunner,
)
from lorahub.core.backends.anima_lora.parser import parse_line

__all__ = ["AnimaLoraRunner", "RunResult"]


# Mirror upstream's accelerate launch defaults. Both flags are the same
# values shipped in ``external/anima_lora/scripts/tasks/_common.py``;
# centralising them here means a future upstream bump can change them
# in one spot.
_LAUNCH_THREADS = "3"
_LAUNCH_MIXED_PRECISION = "bf16"


class AnimaLoraRunner(SubprocessRunner):
    """Runs ``<python> -m accelerate ... launch train.py <argv>``."""

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
        full_argv = [
            str(python),
            "-m",
            "accelerate.commands.accelerate_cli",
            "launch",
            "--num_cpu_threads_per_process",
            _LAUNCH_THREADS,
            "--mixed_precision",
            _LAUNCH_MIXED_PRECISION,
            str(train_py),
            *argv,
        ]
        # cwd=repo so train.py can resolve relative paths in its own
        # `configs/base.toml` model paths (`models/...`) without the
        # caller having to pre-resolve everything.
        super().__init__(
            argv=full_argv,
            workspace=workspace,
            on_event=on_event,
            parse_line=parse_line,
            cwd=repo,
            job_id=job_id,
            env=env,
            thread_label="anima_lora",
        )
