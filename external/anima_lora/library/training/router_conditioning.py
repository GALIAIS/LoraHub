"""Per-step network conditioning hooks.

Drives the timestep / σ / FEI routers that live on the LoRA-family
networks. Most calls are no-ops unless the active network exposes the
corresponding ``set_*`` / ``step_*`` method.

Same hookpoint order is preserved so cudagraph capture sees a stable
sequence: timestep_mask → reft_timestep_mask → sigma → fei → balance.
"""

from __future__ import annotations

from typing import Any

import torch


def apply_router_conditioning(
    *,
    network: Any,
    noisy_model_input: torch.Tensor,
    timesteps: torch.Tensor,
    is_train: bool,
    warmup_step: int,
    max_train_steps: int,
) -> int:
    """Run all per-step router conditioning. Returns the next warmup_step value."""
    if hasattr(network, "set_timestep_mask"):
        network.set_timestep_mask(timesteps, max_timestep=1.0)
    if hasattr(network, "set_reft_timestep_mask"):
        network.set_reft_timestep_mask(timesteps, max_timestep=1.0)
    # σ-conditional HydraLoRA router (Track B, timestep-hydra.md). No-op
    # unless use_sigma_router is on and the variant is hydra/ortho_hydra.
    if hasattr(network, "set_sigma"):
        network.set_sigma(timesteps)
    # FEI router input — set_fei() drives both the per-Linear FEI router
    # (FEI-on-Hydra Phase 1) and the network-level GlobalRouter (FeRA /
    # stacked_experts). FEI is a function of the actual input the model
    # sees this step (``noisy_model_input``), not a leak from the target.
    # No-op when the active network has no FEI router.
    if getattr(network, "use_fei_router", False):
        from library.runtime.fei import compute_fei_2band, fei_sigma_low

        z = noisy_model_input
        if z.dim() == 5:
            z = z.squeeze(2)
        h_lat, w_lat = int(z.shape[-2]), int(z.shape[-1])
        div = float(getattr(network.cfg, "fei_sigma_low_div", 8.0))
        fei = compute_fei_2band(z, fei_sigma_low(h_lat, w_lat, div))
        network.set_fei(fei)

    if is_train and hasattr(network, "step_balance_loss_warmup"):
        network.step_balance_loss_warmup(warmup_step, max_train_steps)
        return warmup_step + 1
    return warmup_step
