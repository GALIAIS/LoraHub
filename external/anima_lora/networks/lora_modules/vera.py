# VeRA — Vector-based Random Adaptation (Kopiczko et al. ICLR'24,
# arXiv:2310.11454).
#
# Two frozen random matrices ``A`` and ``B`` per (in, out, rank) shape
# group plus two learnable diagonal vectors per Linear:
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
# Network-wide A/B sharing: when ``share_pool=True`` (default), every
# VeRAModule with the same (in, out, rank) tuple references one
# globally registered (A, B) pair. Matches the upstream paper's
# parameter savings claim — at 200 Linears in an Anima DiT this turns
# the per-Linear 32k overhead into 32k total. The pool is class-level
# storage and must be reset at network build time via
# :meth:`VeRAModule.reset_shared_pool` (LoRANetwork does this once).

from typing import ClassVar, Optional

import torch

from networks.lora_modules.base import BaseLoRAModule


class VeRAModule(BaseLoRAModule):
    """Frozen random A/B (optionally shared) + per-vector diagonal gates.

    Linear-only.
    """

    supports_conv2d = False

    # Class-level pool keyed by (in_dim, out_dim, rank). Shared
    # ``(A, B)`` tensors live here as fp32 buffers; per-Linear modules
    # hold a tensor reference (NOT a buffer / Parameter) so multiple
    # modules read the same memory.
    _shared_pool: ClassVar[
        dict[tuple[int, int, int], tuple[torch.Tensor, torch.Tensor]]
    ] = {}

    @classmethod
    def reset_shared_pool(cls) -> None:
        """Drop every cached (A, B) pair.

        Call this once before constructing a fresh ``LoRANetwork`` so
        a previous network's tensors don't leak into the new one.
        """
        cls._shared_pool.clear()

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
        share_pool: bool = True,
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

        if share_pool:
            # Pool key: identical (in, out, rank) groups share one pair.
            key = (in_dim, out_dim, lora_dim)
            cached = self._shared_pool.get(key)
            if cached is None:
                A = torch.randn(lora_dim, in_dim) * vera_init_std
                B = torch.randn(out_dim, lora_dim) * vera_init_std
                self._shared_pool[key] = (A, B)
            else:
                A, B = cached
            # Register as plain attributes (not buffers) so we don't
            # round-trip a copy of A/B into every per-module
            # checkpoint slice. The shared pair is saved once globally
            # by the network-level handler (TODO: Phase-5 follow-up;
            # for now, persistent buffers per-module preserve the
            # checkpoint contract — sharing only saves training-time
            # gradient compute and VRAM).
            self.register_buffer("vera_A", A, persistent=True)
            self.register_buffer("vera_B", B, persistent=True)
        else:
            # Per-Linear independent random matrices.
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
