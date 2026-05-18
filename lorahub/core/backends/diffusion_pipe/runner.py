"""Run a diffusion-pipe training job as a subprocess and stream events.

Composes the generic ``SubprocessRunner`` from ``_common`` with the
diffusion-pipe line parser. Built so callers don't need to think about the
deepspeed launcher shape: pass the env's python, the `train.py` path, and
the argv (typically `--deepspeed --config <toml>`), and we'll do the rest.

Multi-node (B8): callers can pass ``launcher_args`` to inject DeepSpeed
launcher flags (``--hostfile`` / ``--num_nodes`` / ``--master_addr`` /
``--master_port``) **before** the train.py path. DeepSpeed's launcher
parses its own argv up to ``train.py``, so the order matters.
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
    """Runs `<venv>/bin/deepspeed [<launcher_args>] train.py --deepspeed --config <toml> ...`."""

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
        launcher_args: list[str] | None = None,
    ) -> None:
        train_py = repo / "train.py"
        # Use the venv's `deepspeed` launcher rather than plain `python`.
        # Direct `python train.py --deepspeed` makes deepspeed think there
        # is no launcher, so it falls back to MPI discovery and crashes
        # with `ModuleNotFoundError: mpi4py` when MPI isn't installed.
        # The launcher sets the env vars deepspeed expects (LOCAL_RANK,
        # RANK, WORLD_SIZE, MASTER_ADDR, ...) and skips MPI entirely.
        deepspeed_bin = python.parent / "deepspeed"
        if not deepspeed_bin.is_file():
            deepspeed_bin = python.parent / "deepspeed.exe"  # Windows fallback
        # Launcher args (--hostfile / --num_nodes / --master_addr / etc.)
        # MUST come before train.py — DeepSpeed's launcher parses its own
        # argv up to the script path. `train.py` re-parses `--deepspeed`
        # itself for compat with both launch styles, so we keep the recipe
        # argv as-is.
        launcher_prefix = list(launcher_args or [])
        super().__init__(
            argv=[str(deepspeed_bin), *launcher_prefix, str(train_py), *argv],
            workspace=workspace,
            on_event=on_event,
            parse_line=parse_line,
            cwd=repo,
            job_id=job_id,
            env=env,
            thread_label="diffusion-pipe",
        )
