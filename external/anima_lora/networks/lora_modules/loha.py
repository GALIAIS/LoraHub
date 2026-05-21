# LoHA — Low-rank Hadamard product (FedPara, arXiv:2108.06098, LyCORIS).
#
# Two LoRA pairs whose products are element-wise multiplied:
#
#     ΔW = (W_a · W_b) ⊙ (W_c · W_d)
#
# Effective rank reaches ``r²`` while parameter count stays at twice a
# rank-``r`` LoRA. Linear-only in this Phase-1 cut.

import math
from typing import Optional

import torch

from networks.lora_modules.base import BaseLoRAModule


class LoHAModule(BaseLoRAModule):
    """Hadamard-product LoRA (Linear only).

    Layout::

        ΔW = (α / r) · (hada_w1_a · hada_w1_b) ⊙ (hada_w2_a · hada_w2_b)
        hada_w1_a, hada_w2_a ∈ R^{out × r}
        hada_w1_b, hada_w2_b ∈ R^{r × in}
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
                f"LoHAModule supports Linear only (got {type(org_module).__name__})"
            )
        out_dim = org_module.out_features
        in_dim = org_module.in_features
        rank = max(1, lora_dim)

        # First pair has random init; second pair is zero-init on the
        # ``b`` leg so the Hadamard product collapses to zero at step 0
        # (LyCORIS convention — keeps step 0 bit-equivalent).
        self.hada_w1_a = torch.nn.Parameter(torch.empty(out_dim, rank))
        self.hada_w1_b = torch.nn.Parameter(torch.empty(rank, in_dim))
        self.hada_w2_a = torch.nn.Parameter(torch.empty(out_dim, rank))
        self.hada_w2_b = torch.nn.Parameter(torch.zeros(rank, in_dim))

        torch.nn.init.kaiming_uniform_(self.hada_w1_a, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.hada_w1_b, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.hada_w2_a, a=math.sqrt(5))
        # hada_w2_b stays zero — see init comment above.

        if channel_scale is not None:
            raise ValueError(
                "LoHAModule does not support channel_scale (Hadamard "
                "product mixes input columns; rebalance doesn't apply)."
            )

        self.org_module_ref = [org_module]
        self._fused = False

    def _delta(self, multiplier: Optional[float] = None) -> torch.Tensor:
        """Materialise ΔW in W-space."""
        m = multiplier if multiplier is not None else self.multiplier
        w1 = self.hada_w1_a.float() @ self.hada_w1_b.float()
        w2 = self.hada_w2_a.float() @ self.hada_w2_b.float()
        return m * self.scale * (w1 * w2)

    def forward(self, x):
        if not self.enabled or self._fused:
            return self.org_forward(x)
        if self.training and self._skip_module():
            return self.org_forward(x)

        org = self.org_module_ref[0]
        delta = self._delta().to(x.dtype)
        eff = org.weight + delta
        return torch.nn.functional.linear(x, eff, org.bias)

    def get_weight(self, multiplier: Optional[float] = None) -> torch.Tensor:
        return self._delta(multiplier)

    def merge_to(self, sd, dtype, device):
        with torch.no_grad():
            weight = self.org_module.weight
            org_dtype = weight.dtype
            if dtype is None:
                dtype = org_dtype
            if device is None:
                device = weight.device

            w1_a = sd["hada_w1_a"].to(torch.float).to(device)
            w1_b = sd["hada_w1_b"].to(torch.float).to(device)
            w2_a = sd["hada_w2_a"].to(torch.float).to(device)
            w2_b = sd["hada_w2_b"].to(torch.float).to(device)
            delta = self.multiplier * self.scale * (
                (w1_a @ w1_b) * (w2_a @ w2_b)
            )
            weight.data.copy_((weight.data.float() + delta).to(dtype))

    def fuse_weight(self):
        if self._fused:
            return
        org_module = self.org_module_ref[0]
        self._w0_backup = org_module.weight.data.detach().clone()
        org_module.weight.data.add_(self._delta().to(org_module.weight.dtype))
        self._fused = True

    def unfuse_weight(self):
        if not self._fused:
            return
        org_module = self.org_module_ref[0]
        org_module.weight.data.copy_(self._w0_backup)
        del self._w0_backup
        self._fused = False
