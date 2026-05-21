# Diag-OFT — block-diagonal Orthogonal Fine-Tuning (Qiu et al. NeurIPS'23,
# arXiv:2306.07280).
#
# Replaces ``W'`` with ``R · W`` where ``R`` is a block-diagonal
# orthogonal matrix; each block is parameterised via the Cayley
# transform of a skew-symmetric matrix so ``R^T R = I`` holds exactly
# (no projection step needed):
#
#     R = blkdiag(R_1, ..., R_K)
#     R_k = (I - Q_k) (I + Q_k)^{-1},  Q_k = A_k - A_k^T   (skew)
#
# Param count: ``K · r · (r-1) / 2`` where ``out_dim = K · r``. For
# ``out_dim=1024, r=4`` that's 1536 params per Linear — comparable to
# rank-1 LoRA on the same layer but with a hyperspherical-energy
# preserving guarantee.
#
# Linear-only (DiT projections); Conv2d would need a kernel-aware
# block split that's not in the Phase-3 cut.

from typing import Optional

import torch

from networks.lora_modules.base import BaseLoRAModule


def _cayley_orthogonal(skew: torch.Tensor) -> torch.Tensor:
    """Cayley transform of a skew-symmetric matrix → exact orthogonal.

    ``skew``: ``(K, r, r)``;  output: same shape, orthogonal per slice.
    """
    K, r, _ = skew.shape
    eye = torch.eye(r, dtype=skew.dtype, device=skew.device).expand(K, r, r)
    # (I + Q)^{-1} (I - Q) — equivalent under transpose, both forms are
    # in active use; the order chosen here matches LyCORIS' implementation.
    return torch.linalg.solve(eye + skew, eye - skew)


class DiagOFTModule(BaseLoRAModule):
    """Block-diagonal orthogonal fine-tuning (Linear only).

    ``lora_dim`` repurposed as the block size ``r`` (default 4).
    ``out_features`` must be divisible by ``r``; we pick a fallback
    block size if not.
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
                f"DiagOFTModule supports Linear only "
                f"(got {type(org_module).__name__})"
            )

        out_dim = org_module.out_features
        # Block size r — fall back to the largest divisor of out_dim
        # not exceeding ``lora_dim``. For typical DiT sizes (1024,
        # 1536, 3072) any small r divides cleanly.
        r = min(lora_dim, out_dim)
        while out_dim % r != 0 and r > 1:
            r -= 1
        if r < 2:
            raise ValueError(
                f"DiagOFTModule needs block size >= 2; out_features={out_dim} "
                f"has no divisor <= {lora_dim} above 1"
            )
        self._block_size = r
        self._num_blocks = out_dim // r

        # Strictly upper-triangular A_k → Q_k = A_k - A_k^T is skew.
        # Init zero so R = I and step 0 is bit-equivalent.
        self.oft_skew = torch.nn.Parameter(
            torch.zeros(self._num_blocks, r, r)
        )

        if channel_scale is not None:
            raise ValueError(
                "DiagOFTModule does not support channel_scale (the rebalance "
                "trick rescales input columns; OFT acts on output rows of W)."
            )

        self.org_module_ref = [org_module]
        self._fused = False

    def _rotation_matrix(self) -> torch.Tensor:
        """Materialise full ``R`` ∈ R^{out × out}."""
        # Skew-symmetrise the parameter (only upper triangle is "real",
        # but we accept full A and explicitly anti-symmetrise so any
        # spurious lower-triangle drift gets cancelled).
        A = self.oft_skew.float()
        Q = A - A.transpose(-1, -2)
        R_blocks = _cayley_orthogonal(Q)  # (K, r, r)
        out_dim = self._num_blocks * self._block_size
        R = torch.zeros(
            out_dim, out_dim, device=R_blocks.device, dtype=R_blocks.dtype
        )
        for k in range(self._num_blocks):
            i, j = k * self._block_size, (k + 1) * self._block_size
            R[i:j, i:j] = R_blocks[k]
        return R

    def _effective_weight(
        self, multiplier: Optional[float] = None
    ) -> torch.Tensor:
        """Return ``R · W₀`` interpolated by ``multiplier`` (m=0 → W₀)."""
        m = multiplier if multiplier is not None else self.multiplier
        org_w = self.org_module_ref[0].weight.float()
        R = self._rotation_matrix()
        eye = torch.eye(R.shape[0], dtype=R.dtype, device=R.device)
        # Linear interpolation between identity and R; m=0 leaves W₀
        # untouched, m=1 applies the full rotation.
        R_eff = eye + m * (R - eye)
        return R_eff @ org_w

    def forward(self, x):
        if not self.enabled or self._fused:
            return self.org_forward(x)
        if self.training and self._skip_module():
            return self.org_forward(x)

        org = self.org_module_ref[0]
        eff = self._effective_weight().to(x.dtype)
        return torch.nn.functional.linear(x, eff, org.bias)

    def get_weight(self, multiplier=None):
        """Effective signed weight delta vs W₀."""
        org_w = self.org_module_ref[0].weight.float()
        return self._effective_weight(multiplier) - org_w

    def merge_to(self, sd, dtype, device):
        """Bake checkpoint slice into ``org_module.weight``.

        Reconstructs ``R`` from ``oft_skew`` then writes ``R · W₀``.
        """
        with torch.no_grad():
            weight = self.org_module.weight
            org_dtype = weight.dtype
            if dtype is None:
                dtype = org_dtype
            if device is None:
                device = weight.device

            skew_param = sd.get("oft_skew")
            if skew_param is None:
                raise KeyError(
                    f"DiagOFT merge_to: missing oft_skew for {self.lora_name}"
                )
            A = skew_param.to(torch.float).to(device)
            K, r, _ = A.shape
            Q = A - A.transpose(-1, -2)
            R_blocks = _cayley_orthogonal(Q)
            out_dim = K * r
            R = torch.zeros(out_dim, out_dim, device=device, dtype=torch.float)
            for k in range(K):
                i, j = k * r, (k + 1) * r
                R[i:j, i:j] = R_blocks[k]

            eye = torch.eye(out_dim, device=device, dtype=torch.float)
            R_eff = eye + self.multiplier * (R - eye)
            weight.data.copy_((R_eff @ weight.data.float()).to(dtype))

    def fuse_weight(self):
        if self._fused:
            return
        org_module = self.org_module_ref[0]
        self._w0_backup = org_module.weight.data.detach().clone()
        org_module.weight.data.copy_(
            self._effective_weight().to(org_module.weight.dtype)
        )
        self._fused = True

    def unfuse_weight(self):
        if not self._fused:
            return
        org_module = self.org_module_ref[0]
        org_module.weight.data.copy_(self._w0_backup)
        del self._w0_backup
        self._fused = False
