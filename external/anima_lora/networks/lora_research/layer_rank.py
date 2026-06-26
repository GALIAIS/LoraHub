"""Research-only layer rank budget rules.

No checkpoint format change: this just computes an effective rank budget from
an existing LoRA module name and timestep budget.
"""

from __future__ import annotations


def layer_rank_multiplier(module_name: str) -> float:
    name = module_name.lower()
    if ".mlp." in name or "_mlp" in name:
        return 0.5
    if "self_attn" in name:
        return 0.75
    if "cross_attn" in name:
        return 1.0
    return 1.0


def layer_rank_budget(
    timestep_budget: float,
    module_name: str,
    *,
    rank: int,
    min_rank: int = 1,
) -> int:
    if rank < 1:
        raise ValueError(f"rank must be >= 1, got {rank}")
    if not 1 <= min_rank <= rank:
        raise ValueError(f"min_rank must be in [1, rank], got {min_rank}")

    scaled = int(round(float(timestep_budget) * layer_rank_multiplier(module_name)))
    return max(min_rank, min(rank, scaled))
