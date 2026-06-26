"""Research-only LoRA algorithm experiments.

Nothing in this package is wired into the production registry until it has a
measured win over the current LoRA/T-LoRA path.
"""

from networks.lora_research.layer_rank import layer_rank_budget, layer_rank_multiplier
from networks.lora_research.rank_mask import per_sample_rank_mask, rank_budget
from networks.lora_research.style_fidelity import (
    style_data_regime,
    style_layer_multiplier,
    style_rank_budget,
    style_recipe,
)
from networks.lora_research.experiment_plans import (
    PLAN_PRESETS,
    build_all_experiment_configs,
    build_experiment_config,
    passes_promotion_gate,
)

__all__ = [
    "layer_rank_budget",
    "layer_rank_multiplier",
    "per_sample_rank_mask",
    "rank_budget",
    "style_data_regime",
    "style_layer_multiplier",
    "style_rank_budget",
    "style_recipe",
    "PLAN_PRESETS",
    "build_all_experiment_configs",
    "build_experiment_config",
    "passes_promotion_gate",
]
