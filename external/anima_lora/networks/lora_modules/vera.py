# VeRA — Vector-based Random Adaptation (Kopiczko et al. ICLR'24,
# arXiv:2310.11454).
#
# Two frozen random matrices ``A`` and ``B`` per Linear plus two
# learnable diagonal vectors:
#
#     ΔW = (α / r) · diag(λ_b) · B · diag(λ_d) · A
#     A ∈ R^{r × in},     frozen N(0, σ²)
#     B ∈ R^{out × r},    frozen N(0, σ²)
#     λ_b ∈ R^{out},      trainable, init 0
#     λ_d ∈ R^{r},        trainable, init 0.1
#
# Param count = ``out + r``, vs. ``r·(in + out)`` for plain LoRA.
# At ``out=in=1024, r=16`` that's 1040 vs. 32768 — ~32× fewer.
#
# Note on shared A / B:  the upstream paper shares one (A, B) pair
# *across the whole network*. We hold a per-Linear pair instead — the
# random projection is independent per layer, which keeps the trainer
# implementation simple and only loses a small fraction of the
# parameter savings (at 200 Linears the per-Linear-shared overhead
# is ~200 × (in + out)/2 fp32, dwarfed by the trainable scale vectors
# we already store). Wiring true network-wide sharing is doable in a
# follow-up patch by adding a buffer registry on ``LoRANetwork``.

import math
from typing import Optional

import torch

from networks.lora_modules.base import BaseLoRAModule


class VeRAModule(BaseLoRAModule):
    """Frozen random A/B + per-vector diagonal gates (Linear only).

    The frozen random matrices are stored as buffers (persistent so
    checkpoints can rehydrate the exact projection). The trainable
    parameters are the two scale vectors.
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
        vera_init_std: float = 0.02,
        vera_init_scale_b: float = 0.0,
        vera_init_scale_d: float = 0.1,
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
                f"VeRAModule supports Linear only "
                f"(got {type(org_module).__name__})"
            )

        in_dim = org_module.in_features
        out_dim = org_module.out_features

        # Frozen random matrices, persistent so checkpoints round-trip.
        # Initialised with Gaussian noise; the only randomness in the
        # adapter — fully determined by the seed pre-creation.
        self.register_buffer(
            "vera_A", torch.randn(lora_dim, in_dim) * vera_init_std,
            persistent=True,
        )
        self.register_buffer(
            "vera_B", torch.randn(out_dim, lora_dim) * vera_init_std,
            persistent=True,
        )
        # Trainable scale vectors. λ_b init zero gives ΔW = 0 at step 0
        # (exactly identity); λ_d init small positive so gradients flow
        # through both legs immediately.
        self.vera_lambda_b = torch.nn.Parameter(
            torch.full((out_dim,), vera_init_scale_b)
        )
        self.vera_lambda_d = torch.nn.Parameter(
            torch.full((lora_dim,), vera_init_scale_d)
        )

        if channel_scale is not None:
            raise ValueError(
                "VeRAModule does not support channel_scale (input rebalance "
                "would alter the frozen random projection's variance)."
            )

        self.org_module_ref = [org_module]
        self._fused = False

    def _delta(self, multiplier: Optional[float] = None) -> torch.Tensor:
        m = multiplier if multiplier is not None else self.multiplier
        # ΔW = diag(λ_b) · B · diag(λ_d) · A in fp32.
        gated_a = self.vera_A.float() * self.vera_lambda_d.float().unsqueeze(1)
        gated_b = self.vera_B.float() * self.vera_lambda_b.float().unsqueeze(1)
        return m * (gated_b @ gated_a) * self.scale

    def forward(self, x):
        if not self.enabled or self._fused:
            return self.org_forward(x)
        if self.training and self._skip_module():
            return self.org_forward(x)

        org_y = self.org_forward(x)
        # Bottleneck-route forward (cheaper than materialising ΔW for
        # large in/out): x → A → λ_d → B → λ_b.
        lx = torch.nn.functional.linear(x.float(), self.vera_A.float())
        lx = lx * self.vera_lambda_d.float()
        if self.dropout is not None and self.training:
            lx = torch.nn.functional.dropout(lx, p=self.dropout)
        lx = torch.nn.functional.linear(lx, self.vera_B.float())
        lx = lx * self.vera_lambda_b.float()
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

            A = sd["vera_A"].to(torch.float).to(device)
            B = sd["vera_B"].to(torch.float).to(device)
            lam_b = sd["vera_lambda_b"].to(torch.float).to(device)
            lam_d = sd["vera_lambda_d"].to(torch.float).to(device)
            gated_a = A * lam_d.unsqueeze(1)
            gated_b = B * lam_b.unsqueeze(1)
            delta = self.multiplier * (gated_b @ gated_a) * self.scale
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
