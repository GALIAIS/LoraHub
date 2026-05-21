# DyLoRA — dynamic low-rank adaptation (Valipour et al. EACL'23,
# arXiv:2210.07558).
#
# Same on-disk shape as plain LoRA (``lora_down.weight`` /
# ``lora_up.weight`` / ``alpha``) — the difference is purely a training-
# time prefix-truncation trick:
#
#     forward (training): pick b ∈ [1, lora_dim] uniformly at random,
#                         use lora_down[:b, :] and lora_up[:, :b]
#     forward (eval):     full rank
#
# This forces the network to keep the *first b* rows / cols functional
# at every prefix, so a cheap rank-truncated checkpoint stays useful at
# inference.  Effective scale at prefix ``b`` follows the LyCORIS
# convention: ``alpha / b`` (uses live truncated rank, not the static
# ``lora_dim``) so a smaller prefix doesn't shrink the contribution.
#
# Saves under the standard variant — ComfyUI loads DyLoRA checkpoints
# as plain LoRA and ignores the metadata flag.

import math
import random
from typing import Optional

import torch

from networks.lora_modules.base import BaseLoRAModule


class DyLoRAModule(BaseLoRAModule):
    """Plain LoRA at inference; random rank truncation during training.

    Linear-only — DiT projections cover every site that DyLoRA cares
    about, and rank truncation on Conv2d would need a kernel split
    that's out of scope for the Phase-2 cut.
    """

    supports_conv2d = False

    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=None,
        rank_dropout=None,
        module_dropout=None,
        channel_scale=None,
    ):
        super().__init__(
            lora_name,
            org_module,
            multiplier=multiplier,
            lora_dim=lora_dim,
            alpha=alpha,
            dropout=dropout,
            rank_dropout=rank_dropout,
            module_dropout=module_dropout,
        )

        if org_module.__class__.__name__ != "Linear":
            raise ValueError(
                f"DyLoRAModule supports Linear only "
                f"(got {type(org_module).__name__})"
            )

        in_dim = org_module.in_features
        out_dim = org_module.out_features
        # Match plain LoRA's parameter shapes so the resulting save file
        # is bit-identical to a vanilla LoRA save (same key names, same
        # per-row order). Bias-free following LoRA convention.
        self.lora_down = torch.nn.Linear(in_dim, lora_dim, bias=False)
        self.lora_up = torch.nn.Linear(lora_dim, out_dim, bias=False)
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_up.weight)

        if channel_scale is not None:
            raise ValueError(
                "DyLoRAModule does not support channel_scale yet — the "
                "rebalance trick stores inv_scale assuming the full down "
                "matrix; truncating the first b rows would mismatch."
            )

        # List wrapping prevents nn.Module from registering org_module as a
        # submodule. apply_to() del's self.org_module after rerouting forward.
        self.org_module_ref = [org_module]
        self._fused = False

    def _truncated_rank(self) -> int:
        if not self.training:
            return self.lora_dim
        # Uniform over [1, lora_dim]. Sampled per-step per-Linear, which
        # is the upstream DyLoRA convention (each layer truncates
        # independently — gives the network broader rank coverage than a
        # network-wide single sample).
        return random.randint(1, self.lora_dim)

    def forward(self, x):
        if not self.enabled or self._fused:
            return self.org_forward(x)
        if self.training and self._skip_module():
            return self.org_forward(x)

        b = self._truncated_rank()
        # Slice the LoRA legs to rank b. ``narrow`` returns a view (no
        # copy) so this stays cheap at the inner loop.
        down = self.lora_down.weight[:b, :].float()  # (b, in)
        up = self.lora_up.weight[:, :b].float()  # (out, b)

        org_y = self.org_forward(x)
        lx = torch.nn.functional.linear(x.float(), down)
        if self.dropout is not None and self.training:
            lx = torch.nn.functional.dropout(lx, p=self.dropout)
        lx = torch.nn.functional.linear(lx, up)
        # alpha/b scaling — see module docstring on the live-rank choice.
        scale = float(self.alpha.item()) / b
        return org_y + (lx * self.multiplier * scale).to(org_y.dtype)

    def _delta_full_rank(self, multiplier: Optional[float] = None) -> torch.Tensor:
        """Materialise the full-rank ΔW (used by fuse / merge_to / get_weight)."""
        m = multiplier if multiplier is not None else self.multiplier
        down = self.lora_down.weight.float()
        up = self.lora_up.weight.float()
        # Full-rank inference uses the static lora_dim → scale via self.scale
        # which is alpha / lora_dim (the BaseLoRAModule init).
        return m * (up @ down) * self.scale

    def get_weight(self, multiplier=None):
        return self._delta_full_rank(multiplier)

    def merge_to(self, sd, dtype, device):
        with torch.no_grad():
            weight = self.org_module.weight
            org_dtype = weight.dtype
            if dtype is None:
                dtype = org_dtype
            if device is None:
                device = weight.device

            down = sd["lora_down.weight"].to(torch.float).to(device)
            up = sd["lora_up.weight"].to(torch.float).to(device)
            delta = self.multiplier * (up @ down) * self.scale
            weight.data.copy_((weight.data.float() + delta).to(dtype))

    def fuse_weight(self):
        if self._fused:
            return
        org_module = self.org_module_ref[0]
        self._w0_backup = org_module.weight.data.detach().clone()
        org_module.weight.data.add_(
            self._delta_full_rank().to(org_module.weight.dtype)
        )
        self._fused = True

    def unfuse_weight(self):
        if not self._fused:
            return
        org_module = self.org_module_ref[0]
        org_module.weight.data.copy_(self._w0_backup)
        del self._w0_backup
        self._fused = False
