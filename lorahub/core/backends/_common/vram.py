"""Shared VRAM estimator for every supported model architecture.

Both ``KohyaBackend`` and ``DiffusionPipeBackend`` previously shipped
nearly-identical, hand-tuned VRAM heuristics that only knew about a few
arches and silently fell back to "treat anything else like SDXL". With the
arch matrix grown to 23 entries (sd15 ... ernie_image), those copies were
both stale and out of sync, so we centralise the table + formula here.

The numbers below are a *first-pass approximation* derived from the
model.safetensors header sizes published on Hugging Face. They are
intentionally conservative rather than precise: VRAM in practice depends on
the trainer, the optimizer, the resolution / frame count, and the precision
mode. Refine them with empirical data once we have actual benchmark runs.
"""

from __future__ import annotations

from lorahub.core.backends.base import VRAMEstimate

# ---------------------------------------------------------------------------
# Per-arch model size in millions of parameters (~ million weights). These
# come from the public ``model.safetensors`` headers / model cards and are
# rounded to the nearest 100M for legibility. Entries that are not yet
# benchmarked carry an "(est)" comment.
# ---------------------------------------------------------------------------
ARCH_MODEL_PARAMS_M: dict[str, int] = {
    # static-image, kohya-supported
    "sd15": 860,
    "sd2": 860,
    "sdxl": 2600,
    "sd3": 2000,
    "flux": 12000,
    "flux2": 13000,  # est, FLUX.2 dev
    "lumina": 2500,  # Lumina-Image-2.0 ~2.5B
    "anima": 2000,  # Anima ~2B
    "hunyuan_image": 17000,  # HunyuanImage-2.1 ~17B
    # static-image, diffusion-pipe-only
    "chroma": 8000,
    "hidream": 17000,
    "omnigen2": 4000,
    "auraflow": 7000,
    "qwen_image": 20000,
    "cosmos": 7000,
    "cosmos_predict2": 2000,
    "z_image": 7000,
    "ernie_image": 4000,
    "krea2": 32000,  # est, Krea 2 large diffusion stack
    # video — sizes are the trunk only; activations dominate at runtime
    "hunyuan_video": 13000,
    "hunyuan_video_15": 13000,
    "ltx_video": 1900,  # ~2B
    "ltx2": 4000,  # est
    "wan": 14000,  # Wan2.1-14B variant
}

# Per-arch base activation footprint (MiB) at batch_size=1, before the
# checkpoint discount kicks in. Tuned by class:
#   - SD-era (sd15/sd2):              384
#   - SDXL / mid-flux family:         768-1024
#   - flagship static (flux/qwen):    1280-1792
#   - hunyuan_image (largest static): 2048
#   - video archs (frame-heavy):      2048-4096
ARCH_ACTIVATION_BASE_MIB: dict[str, int] = {
    "sd15": 384,
    "sd2": 384,
    "sdxl": 1024,
    "sd3": 768,
    "flux": 1536,
    "flux2": 1536,
    "lumina": 1024,
    "anima": 768,
    "hunyuan_image": 2048,
    "chroma": 1280,
    "hidream": 1536,
    "omnigen2": 1024,
    "auraflow": 1280,
    "qwen_image": 1792,
    "cosmos": 1280,
    "cosmos_predict2": 768,
    "hunyuan_video": 4096,
    "hunyuan_video_15": 4096,
    "ltx_video": 2048,
    "ltx2": 3072,
    "wan": 4096,
    "z_image": 1280,
    "ernie_image": 1024,
    "krea2": 2048,
}

# Conservative defaults for unknown arches (treat them as roughly SDXL-sized
# so the user still gets a non-zero estimate rather than a crash).
_DEFAULT_PARAMS_M = 2600
_DEFAULT_ACTIVATION_MIB = 1024


def _bytes_per_param(precision: str) -> int:
    """Return weight footprint per parameter for the given precision string."""
    return 2 if precision in ("fp16", "bf16") else 4


def estimate_vram(
    arch: str,
    *,
    precision: str,
    batch_size: int,
    network_rank: int,
    gradient_checkpointing: bool,
) -> VRAMEstimate:
    """Coarse first-pass VRAM estimate.

    The formula intentionally mirrors the original kohya heuristic so the
    numbers backends produced before this refactor stay broadly comparable:

        model_mib       = params_M * bytes_per_param
        optimizer_mib   = rank * 8                        (when checkpointed)
                        = rank * 8 * 4                    (otherwise)
        activations_mib = base[arch] * batch_size         (no checkpoint)
                        = base[arch] * batch_size // 3    (with checkpoint)

    ``overhead_mib`` is left at the ``VRAMEstimate`` default. Calibrate the
    base tables with empirical data later.
    """
    params_m = ARCH_MODEL_PARAMS_M.get(arch, _DEFAULT_PARAMS_M)
    base_act = ARCH_ACTIVATION_BASE_MIB.get(arch, _DEFAULT_ACTIVATION_MIB)

    model_mib = params_m * _bytes_per_param(precision)

    optimizer_mib = network_rank * 8
    if not gradient_checkpointing:
        optimizer_mib *= 4

    activations_mib = base_act * max(1, batch_size)
    if gradient_checkpointing:
        activations_mib //= 3

    return VRAMEstimate(
        model_mib=model_mib,
        optimizer_mib=optimizer_mib,
        activations_mib=activations_mib,
    )


__all__ = [
    "ARCH_ACTIVATION_BASE_MIB",
    "ARCH_MODEL_PARAMS_M",
    "estimate_vram",
]
