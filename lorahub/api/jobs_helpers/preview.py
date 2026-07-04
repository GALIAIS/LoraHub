"""Optional live-preview worker + low-frequency GPU sampler.

Both run as daemon threads alongside an active training job. Failures
in either are non-fatal — training itself never depends on these
threads, so anything that goes wrong here logs and moves on rather
than killing the run.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from lorahub.core.config.schema import TrainingConfig
from lorahub.core.events import EventType, TrainingEvent

log = logging.getLogger(__name__)


def _maybe_start_preview_worker(
    cfg: TrainingConfig,
    workspace: Path,
    job_id: str,
    on_event: Any,
    stop_evt: threading.Event,
) -> tuple[threading.Thread, Any] | None:
    """Start a PreviewWorker thread if the config asks for it.

    Returns ``(thread, worker)`` so callers can both join the thread on
    shutdown and call ``worker.notify_checkpoint(name)`` from the
    event sink to wake the worker the moment dp finishes a save —
    avoiding the up-to-5s polling latency. Returns ``None`` when the
    feature is disabled or prerequisites are missing. Failures here
    are non-fatal — training never depends on preview rendering.
    """
    sampling = cfg.sampling
    if not sampling.enabled or not sampling.enable_live_inference:
        return None
    prompts_file = sampling.prompts_file
    if prompts_file is None:
        log.info(
            "preview worker [%s]: enable_live_inference is on but no "
            "sampling.prompts_file is configured — skipping",
            job_id,
        )
        return None
    if not Path(str(prompts_file)).is_file():
        log.warning(
            "preview worker [%s]: prompts file %s not found", job_id, prompts_file
        )
        return None

    # Lazy import so the inference module isn't required for every job
    # (and so a future torch import there can't break job launch).
    try:
        from lorahub.core.inference import (  # noqa: PLC0415
            PreviewConfig,
            PreviewWorker,
            StubInference,
        )
        from lorahub.core.inference import (  # noqa: F401, PLC0415
            anima as _anima_mod,
        )
        from lorahub.core.inference import (  # noqa: F401, PLC0415
            anima_lora_backend as _anima_lora_mod,
        )
        from lorahub.core.inference import (  # noqa: F401, PLC0415
            diffusers_backend as _diffusers_mod,
        )
        from lorahub.core.inference.registry import (  # noqa: PLC0415
            registered_backend_names,
            resolve_backend,
        )
    except Exception:  # noqa: BLE001
        log.exception("preview worker [%s]: failed to import module", job_id)
        return None

    output_dir = (
        Path(str(cfg.output.output_dir)).resolve()
        if cfg.output.output_dir is not None
        else (workspace / "output").resolve()
    )
    samples_dir = (workspace / "samples").resolve()
    samples_dir.mkdir(parents=True, exist_ok=True)

    arch = cfg.base_model.arch
    inference_backend: Any
    real_backend = resolve_backend(arch=arch, config=cfg, workspace=workspace)
    if real_backend is not None:
        inference_backend = real_backend
        log.info(
            "preview worker [%s]: using %s backend for arch=%s",
            job_id,
            getattr(real_backend, "name", type(real_backend).__name__),
            arch,
        )
    else:
        inference_backend = StubInference()
        reason = (
            f"no inference backend in {registered_backend_names()} "
            f"supports arch={arch!r}"
        )
        log.warning("preview worker [%s]: %s — using StubInference", job_id, reason)
        try:
            on_event(
                TrainingEvent(
                    type=EventType.preview_unavailable,
                    job_id=job_id,
                    payload={
                        "arch": arch,
                        "available_backends": registered_backend_names(),
                        "reason": reason,
                    },
                )
            )
            on_event(
                TrainingEvent(
                    type=EventType.log,
                    job_id=job_id,
                    payload={
                        "level": "warning",
                        "source": "preview",
                        "message": (
                            f"preview placeholder used because no backend "
                            f"supports {arch}"
                        ),
                    },
                )
            )
        except Exception:  # noqa: BLE001
            log.exception("preview worker [%s]: failed to emit unavailable event", job_id)

    pcfg = PreviewConfig(
        enabled=True,
        prompts_file=Path(str(prompts_file)).resolve(),
        default_steps=sampling.inference_steps,
        default_cfg=sampling.inference_cfg,
        samples_dir=samples_dir,
        output_dir=output_dir,
        grid_stitching=sampling.outputs.grid_stitching,
        base_compare=sampling.outputs.base_compare,
        cross_ckpt_animation=sampling.outputs.cross_ckpt_animation,
        png_metadata=sampling.outputs.png_metadata,
    )
    worker = PreviewWorker(
        config=pcfg,
        inference=inference_backend,
        on_event=on_event,
        job_id=job_id,
        stop_evt=stop_evt,
    )
    thread = threading.Thread(
        target=worker.run,
        daemon=True,
        name=f"preview-{job_id[-6:]}",
    )
    thread.start()
    return thread, worker


def _gpu_sampler_loop(
    job_id: str,
    slot: int | list[int],
    on_event: Any,
    stop_evt: threading.Event,
) -> None:
    """Emit a `gpu_sample` event every ~5s while the job is running.

    Reads GPU data from the shared snapshot cache rather than spawning its
    own ``nvidia-smi``: the dashboard SSE/WS streams already probe GPUs on
    a 1s cadence, so reusing that cache drops a redundant subprocess
    spawn every 5s per running job. The 1s TTL is well inside the 5s
    sample interval, so values are always fresh enough for a trend chart.
    Failures are swallowed (the trend chart is a nice-to-have; we never
    want it to crash a training run).
    """
    from lorahub.api.system_stats import collect_snapshot_shared  # noqa: PLC0415

    interval = 5.0
    slots = [slot] if isinstance(slot, int) else list(slot)
    while not stop_evt.wait(interval):
        try:
            gpus = collect_snapshot_shared().gpus
        except Exception:  # noqa: BLE001
            continue
        if not gpus:
            continue
        for gpu_slot in slots:
            if gpu_slot < 0 or gpu_slot >= len(gpus):
                continue
            g = gpus[gpu_slot]
            try:
                on_event(
                    TrainingEvent(
                        type=EventType.gpu_sample,
                        job_id=job_id,
                        payload={
                            "gpu_index": gpu_slot,
                            "util_percent": g.utilization_percent,
                            "vram_used_mib": (
                                int(g.memory_used_bytes // (1024 * 1024))
                                if getattr(g, "memory_used_bytes", None) is not None
                                else None
                            ),
                            "vram_total_mib": (
                                int(g.memory_total_bytes // (1024 * 1024))
                                if getattr(g, "memory_total_bytes", None) is not None
                                else None
                            ),
                            "temperature_c": g.temperature_c,
                        },
                    )
                )
            except Exception:  # noqa: BLE001
                continue


__all__ = ["_gpu_sampler_loop", "_maybe_start_preview_worker"]
