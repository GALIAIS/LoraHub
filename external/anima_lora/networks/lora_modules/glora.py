# GLoRA-light — Generalized LoRA, scalar-gated rank variant
# (Chavan et al. arXiv:2306.07967, Table 2 "GLoRA-light").
#
# Standard LoRA + a per-rank diagonal gate ``d ∈ R^r`` between the
# down and up legs:
#
#     ΔW = (α / r) · W_up · diag(d) · W_down
#
# d is trainable, initialised to 1 (so step 0 is bit-equivalent to a
# vanilla LoRA at the same params). Adds r extra parameters; learns
# rank importance, encouraging sparsity that effectively prunes ranks.
#
# Linear-only.

import math
from typing import Optional

import torch

from networks.lora_modules.base import BaseLoRAModule


class GLoRAModule(BaseLoRAModule):
    """LoRA + per-rank diagonal gate (Linear only)."""

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
                f"GLoRAModule supports Linear only "
                f"(got {type(org_module).__name__})"
            )

        in_dim = org_module.in_features
        out_dim = org_module.out_features
        self.lora_down = torch.nn.Linear(in_dim, lora_dim, bias=False)
        self.lora_up = torch.nn.Linear(lora_dim, out_dim, bias=False)
        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_up.weight)
        # Per-rank gate, init 1.0.
        self.glora_gate = torch.nn.Parameter(torch.ones(lora_dim))

        if channel_scale is not None:
            raise ValueError(
                "GLoRAModule does not support channel_scale yet — interaction "
                "with the per-rank gate is unspecified upstream."
            )

        self.org_module_ref = [org_module]
        self._fused = False

    def _delta(self, multiplier: Optional[float] = None) -> torch.Tensor:
        m = multiplier if multiplier is not None else self.multiplier
        # diag(d) absorbed into the up-down product: W_up · diag(d) · W_down.
        gated_up = self.lora_up.weight.float() * self.glora_gate.float().unsqueeze(0)
        return m * (gated_up @ self.lora_down.weight.float()) * self.scale

    def forward(self, x):
        if not self.enabled or self._fused:
            return self.org_forward(x)
        if self.training and self._skip_module():
            return self.org_forward(x)

        org_y = self.org_forward(x)
        # Apply gate at the bottleneck for compute efficiency:
        # (down → gate → up) instead of materialising ΔW.
        lx = torch.nn.functional.linear(x.float(), self.lora_down.weight.float())
        if self.dropout is not None and self.training:
            lx = torch.nn.functional.dropout(lx, p=self.dropout)
        lx = lx * self.glora_gate.float()
        lx = torch.nn.functional.linear(lx, self.lora_up.weight.float())
        return org_y + (lx * self.multiplier * self.scale).to(org_y.dtype)

    def get_weight(self, multiplier=None):
        return self._delta(multiplier)

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
            gate = sd.get("glora_gate")
            if gate is None:
                raise KeyError(
                    f"GLoRA merge_to: missing glora_gate for {self.lora_name}"
                )
            gate = gate.to(torch.float).to(device)
            gated_up = up * gate.unsqueeze(0)
            delta = self.multiplier * (gated_up @ down) * self.scale
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
