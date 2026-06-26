"""Research-only LoRA algorithm experiments.

Nothing in this package is wired into the production registry until it has a
measured win over the current LoRA/T-LoRA path.
"""

from networks.lora_research.rank_mask import per_sample_rank_mask, rank_budget

__all__ = ["per_sample_rank_mask", "rank_budget"]
