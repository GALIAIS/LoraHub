"""Run an anima_lora training job as a subprocess and stream events.

anima_lora drives training through HuggingFace Accelerate, not DeepSpeed
(diffusion-pipe) or a bare Python process (kohya). Upstream's
``scripts/tasks/_common.accelerate_launch`` invokes:

    <python> -m accelerate.commands.accelerate_cli launch \
        --num_cpu_threads_per_process 3 \
        --mixed_precision <cfg precision> \
        train.py <args>

We keep the same launcher shape, but the mixed_precision flag is
configuration-driven. The flag here is the **launcher's** mixed
precision (Accelerate's HF-side AMP routing); ``train.py`` itself takes
its own ``mixed_precision`` from the compiler-emitted config file. They
must agree: V100-class GPUs need fp16, while the upstream default is
bf16 for newer hardware.

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


# Mirror upstream's CPU-thread launch default. Mixed precision has a
# bf16 default for compatibility with upstream templates but is normally
# overridden from the LoraHub recipe before launch.
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
        num_processes: int = 1,
        accelerate_config: Path | None = None,
        mixed_precision: str = _LAUNCH_MIXED_PRECISION,
    ) -> None:
        train_py = repo / "train.py"
        full_argv = [
            str(python),
            "-m",
            "accelerate.commands.accelerate_cli",
            "launch",
        ]
        if accelerate_config is not None:
            full_argv += ["--config_file", str(accelerate_config)]
        full_argv += [
            "--num_cpu_threads_per_process",
            _LAUNCH_THREADS,
            "--mixed_precision",
            _accelerate_mixed_precision(mixed_precision),
            "--num_processes",
            str(max(1, num_processes)),
        ]
        if num_processes > 1 and accelerate_config is None:
            full_argv += ["--multi_gpu"]
        full_argv += [
            str(train_py),
            *argv,
        ]
        # Silence the ``SyntaxWarning: invalid escape sequence '\('``
        # that fires every time vendored ``library/anima/text_strategies.py``
        # is imported. Upstream's (string literal where a raw string was
        # meant) — we don't patch vendored code. Keep the filter scoped to
        # this backend: kohya / diffusion-pipe shouldn't inherit it.
        # Format: ``action:message:category:module:lineno``.
        merged_env = {**(env or {})}
        anima_filter = "ignore:invalid escape sequence:SyntaxWarning"
        existing = merged_env.get("PYTHONWARNINGS", "")
        merged_env["PYTHONWARNINGS"] = (
            f"{anima_filter},{existing}" if existing else anima_filter
        )

        # CUDA allocator: enable expandable segments by default. anima
        # ships ``torch_compile=True`` (LOCKED_TRUE) which captures
        # transformer blocks into CUDA Graphs — those Graphs sit in a
        # private memory pool that ``torch.cuda.empty_cache()`` can't
        # reclaim. With the default caching allocator a multi-GiB
        # private pool plus the optimizer / latent / TE caches
        # fragments the remaining headroom; the sampling-phase VAE
        # decode then fails on a ~1 GiB ``F.pad`` even though
        # ``reserved-but-unallocated`` says the bytes exist. PyTorch's
        # documented fix is ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``
        # (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf).
        # We only inject when the user hasn't already set the variable,
        # so a deliberate override (``garbage_collection_threshold=...``)
        # isn't clobbered.
        cuda_alloc = merged_env.get("PYTORCH_CUDA_ALLOC_CONF", "")
        if "expandable_segments" not in cuda_alloc:
            merged_env["PYTORCH_CUDA_ALLOC_CONF"] = (
                f"{cuda_alloc},expandable_segments:True"
                if cuda_alloc
                else "expandable_segments:True"
            )
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
            env=merged_env,
            thread_label="anima_lora",
        )


def _accelerate_mixed_precision(value: str) -> str:
    return "no" if value == "fp32" else value
