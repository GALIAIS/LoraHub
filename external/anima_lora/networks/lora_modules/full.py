# Full — single learnable ΔW Parameter, same shape as the host weight.
#
# Not a LoRA in any meaningful sense — this is a "free Δ" wrapper that
# treats the adapter Linear like a full fine-tune of the slice while
# keeping the rest of the trainer (channel masking, T-LoRA, dropout
# plumbing) intact.  Use case: small Linear-heavy projections where the
# extra rank cost is acceptable and the user wants matched-capacity
# baseline numbers vs. LoRA / DoRA / etc.
#
# Saves under a dedicated ``full`` variant — on-disk shape is just
# ``delta`` (out, in) per Linear, distinct from the LoRA family. Fold
# is trivial: ``W' = W₀ + ΔW``.

from typing import Optional

import torch

from networks.lora_modules.base import BaseLoRAModule


class FullModule(BaseLoRAModule):
    """Free per-Linear ΔW Parameter.

    Linear-only.  Rank- / module-dropout from the base class still
    applies (rank_dropout is a no-op since there's no decomposition,
    but we keep it consumed for parameter parity with peers).
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
                f"FullModule supports Linear only "
                f"(got {type(org_module).__name__})"
            )

        out_dim = org_module.out_features
        in_dim = org_module.in_features
        # Zero-init so step 0 is identity (no shift on the Linear's
        # output relative to the unwrapped layer).
        self.delta = torch.nn.Parameter(torch.zeros(out_dim, in_dim))

        if channel_scale is not None:
            raise ValueError(
                "FullModule does not support channel_scale (input column "
                "rebalance has no semantic home in a free-Δ adapter)."
            )

        self.org_module_ref = [org_module]
        self._fused = False

    def _scaled_delta(self, multiplier: Optional[float] = None) -> torch.Tensor:
        """ΔW with multiplier + alpha-style scale applied."""
        m = multiplier if multiplier is not None else self.multiplier
        return m * self.delta.float() * self.scale

    def forward(self, x):
        if not self.enabled or self._fused:
            return self.org_forward(x)
        if self.training and self._skip_module():
            return self.org_forward(x)

        org = self.org_module_ref[0]
        eff = (org.weight + self._scaled_delta()).to(x.dtype)
        return torch.nn.functional.linear(x, eff, org.bias)

    def get_weight(self, multiplier=None):
        return self._scaled_delta(multiplier)

    def merge_to(self, sd, dtype, device):
        with torch.no_grad():
            weight = self.org_module.weight
            org_dtype = weight.dtype
            if dtype is None:
                dtype = org_dtype
            if device is None:
                device = weight.device

            delta = sd["delta"].to(torch.float).to(device)
            scaled = self.multiplier * delta * self.scale
            weight.data.copy_((weight.data.float() + scaled).to(dtype))

    def fuse_weight(self):
        if self._fused:
            return
        org_module = self.org_module_ref[0]
        self._w0_backup = org_module.weight.data.detach().clone()
        org_module.weight.data.add_(
            self._scaled_delta().to(org_module.weight.dtype)
        )
        self._fused = True

    def unfuse_weight(self):
        if not self._fused:
            return
        org_module = self.org_module_ref[0]
        org_module.weight.data.copy_(self._w0_backup)
        del self._w0_backup
        self._fused = False
