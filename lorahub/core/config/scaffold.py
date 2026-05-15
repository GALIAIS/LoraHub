"""Recipe scaffolder — turn known facts (GPU, dataset, base model) into a recipe.

`auto_scaffold()` picks reasonable defaults the way a human writes a fresh
recipe: rank/batch by VRAM tier, num_repeats inversely by image count,
target architecture from the checkpoint filename. The output is a fully
populated `RecipeConfig` ready to dump to YAML.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from lorahub.core.config.schema import RecipeConfig

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass(frozen=True, slots=True)
class VRAMTier:
    """Recipe parameters tuned for a particular VRAM band."""

    min_mib: int
    rank: int
    alpha: int
    batch_size: int
    grad_accum: int


# Lower-bound MiB to avoid OOM on the named tier. Validated empirically on
# SDXL LoRA training; SD1.5 fits comfortably in any tier.
_VRAM_TIERS: tuple[VRAMTier, ...] = (
    VRAMTier(min_mib=24576, rank=64, alpha=32, batch_size=4, grad_accum=1),
    VRAMTier(min_mib=16384, rank=64, alpha=32, batch_size=2, grad_accum=2),
    VRAMTier(min_mib=12288, rank=32, alpha=16, batch_size=2, grad_accum=2),
    VRAMTier(min_mib=10240, rank=32, alpha=16, batch_size=1, grad_accum=2),
    VRAMTier(min_mib=8192, rank=16, alpha=8, batch_size=1, grad_accum=2),
    VRAMTier(min_mib=6144, rank=8, alpha=4, batch_size=1, grad_accum=4),
    VRAMTier(min_mib=0, rank=4, alpha=2, batch_size=1, grad_accum=8),
)


def detect_gpu_vram_mib() -> int | None:
    """Best-effort total VRAM (in MiB) for the first NVIDIA GPU. None if unavailable."""
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        return None
    try:
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=memory.total", "--format=csv,nounits,noheader"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    if result.returncode != 0:
        return None
    first = result.stdout.strip().splitlines()
    if not first:
        return None
    try:
        return int(first[0].strip())
    except ValueError:
        return None


def pick_vram_tier(vram_mib: int) -> VRAMTier:
    for tier in _VRAM_TIERS:
        if vram_mib >= tier.min_mib:
            return tier
    return _VRAM_TIERS[-1]


def count_images(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    return sum(1 for p in directory.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTS)


def pick_num_repeats(image_count: int) -> int:
    """Smaller datasets need more repeats per epoch to converge."""
    if image_count <= 0:
        return 10
    if image_count < 20:
        return 10
    if image_count < 50:
        return 5
    if image_count < 200:
        return 2
    return 1


def detect_arch(checkpoint: Path) -> str:
    name = checkpoint.name.lower()
    if "flux" in name:
        return "flux"
    if "sd3" in name or "stable-diffusion-3" in name:
        return "sd3"
    if re.search(r"sdxl|illustrious|pony|noobai|animagine", name):
        return "sdxl"
    if "sd15" in name or "v1-5" in name or "sd1_5" in name:
        return "sd15"
    return "sdxl"


def auto_scaffold(
    name: str,
    checkpoint: Path,
    dataset: Path,
    *,
    vram_mib: int | None = None,
    epochs: int = 10,
) -> RecipeConfig:
    """Build a RecipeConfig from probed facts and tier defaults.

    `vram_mib=None` triggers `detect_gpu_vram_mib()`; if that fails too we
    assume the conservative 8GB tier so the recipe is still runnable on
    most users' machines.
    """
    if vram_mib is None:
        vram_mib = detect_gpu_vram_mib() or 8192
    tier = pick_vram_tier(vram_mib)
    arch = detect_arch(checkpoint)
    images = count_images(dataset)
    repeats = pick_num_repeats(images)
    resolution = [1024, 1024] if arch in ("sdxl", "flux", "sd3") else [768, 768]

    return RecipeConfig.model_validate(
        {
            "base_model": {"arch": arch, "checkpoint": str(checkpoint)},
            "dataset": {
                "source": str(dataset),
                "resolution": resolution,
                "num_repeats": repeats,
            },
            "network": {
                "type": "lora",
                "rank": tier.rank,
                "alpha": tier.alpha,
            },
            "schedule": {
                "epochs": epochs,
                "batch_size": tier.batch_size,
                "grad_accum": tier.grad_accum,
            },
            "sampling": {"enabled": False},
            "output": {"name": name},
        }
    )
