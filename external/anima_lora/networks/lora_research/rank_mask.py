"""Rank-mask utilities for experimental timestep-adaptive LoRA.

The current T-LoRA path builds one rank mask from the batch-mean timestep.
These helpers keep the same monotonic rank schedule, but preserve a separate
rank budget per sample. They are research-only until wired into LoRAModule.
"""

from __future__ import annotations

import torch


def rank_budget_values(
    timesteps: list[float] | tuple[float, ...],
    *,
    rank: int,
    min_rank: int = 1,
    alpha: float = 1.0,
    max_timestep: float = 1.0,
) -> list[float]:
    """Pure-Python twin of ``rank_budget`` for config/docs sanity checks."""

    if rank < 1:
        raise ValueError(f"rank must be >= 1, got {rank}")
    if not 1 <= min_rank <= rank:
        raise ValueError(f"min_rank must be in [1, rank], got {min_rank}")
    if max_timestep <= 0:
        raise ValueError(f"max_timestep must be > 0, got {max_timestep}")

    out: list[float] = []
    for raw in timesteps:
        frac = max(0.0, min(1.0, (max_timestep - float(raw)) / max_timestep))
        value = frac**float(alpha) * float(rank - min_rank) + float(min_rank)
        out.append(max(float(min_rank), min(float(rank), value)))
    return out


def rank_budget(
    timesteps: torch.Tensor,
    *,
    rank: int,
    min_rank: int = 1,
    alpha: float = 1.0,
    max_timestep: float = 1.0,
) -> torch.Tensor:
    """Return per-sample effective rank as a float tensor of shape ``(B,)``."""

    if rank < 1:
        raise ValueError(f"rank must be >= 1, got {rank}")
    if not 1 <= min_rank <= rank:
        raise ValueError(f"min_rank must be in [1, rank], got {min_rank}")
    if max_timestep <= 0:
        raise ValueError(f"max_timestep must be > 0, got {max_timestep}")

    if rank < 1:
        raise ValueError(f"rank must be >= 1, got {rank}")
    if not 1 <= min_rank <= rank:
        raise ValueError(f"min_rank must be in [1, rank], got {min_rank}")
    if max_timestep <= 0:
        raise ValueError(f"max_timestep must be > 0, got {max_timestep}")

    t = timesteps.detach().float().reshape(-1)
    frac = ((max_timestep - t) / max_timestep).clamp(0.0, 1.0)
    budget = frac.pow(float(alpha)) * float(rank - min_rank) + float(min_rank)
    return budget.clamp(float(min_rank), float(rank))


def per_sample_rank_mask(
    timesteps: torch.Tensor,
    *,
    rank: int,
    min_rank: int = 1,
    alpha: float = 1.0,
    max_timestep: float = 1.0,
    target_ndim: int = 3,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Build a broadcast-ready rank mask for Linear or Conv LoRA activations.

    ``target_ndim`` matches the LoRA bottleneck activation:
    ``2`` -> ``(B, R)``, ``3`` -> ``(B, 1, R)``, ``4`` -> ``(B, R, 1, 1)``.
    """

    budget = rank_budget(
        timesteps,
        rank=rank,
        min_rank=min_rank,
        alpha=alpha,
        max_timestep=max_timestep,
    )
    arange = torch.arange(rank, device=timesteps.device, dtype=budget.dtype)
    mask = (arange.unsqueeze(0) < budget.unsqueeze(1)).to(dtype or torch.float32)

    if target_ndim == 2:
        return mask
    if target_ndim == 3:
        return mask.unsqueeze(1)
    if target_ndim == 4:
        return mask.unsqueeze(-1).unsqueeze(-1)
    raise ValueError(f"target_ndim must be 2, 3, or 4, got {target_ndim}")


def _demo() -> None:
    t = torch.tensor([0.0, 0.5, 1.0])
    assert rank_budget_values([0.0, 0.5, 1.0], rank=8, min_rank=2) == [8.0, 5.0, 2.0]
    mask = per_sample_rank_mask(t, rank=8, min_rank=2, target_ndim=3)
    assert mask.shape == (3, 1, 8)
    assert int(mask[0].sum().item()) == 8
    assert int(mask[-1].sum().item()) == 2


if __name__ == "__main__":
    _demo()
