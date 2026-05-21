# BOFT — Butterfly Orthogonal Fine-Tuning (Liu et al., arXiv:2311.06243).
#
# Replaces Diag-OFT's single block-diagonal R with a product of m
# block-diagonal orthogonal matrices interleaved by butterfly
# permutations:
#
#     R = B_m · P_{m-1} · B_{m-1} · P_{m-2} · ... · B_1
#
# where each ``B_i`` is block-diagonal with blocks of size 2 (or any
# fixed ``block_size``), and ``P_i`` is a butterfly permutation that
# cycles the row partition. Composing m ≥ log_2(out_dim) factors
# recovers an arbitrary orthogonal matrix; in practice m = 4 ~ 8 is
# enough for fine-tuning.
#
# Param count: ``m · K · r·(r-1)/2`` where ``K · r = out_dim`` and
# ``r`` is the block size. For ``out_dim=1024, r=2, m=4`` that's
# 1024 × 4 / 2 = 2048 params per Linear.
#
# Linear-only.

from typing import Optional

import torch

from networks.lora_modules.base import BaseLoRAModule
from networks.lora_modules.diag_oft import _cayley_orthogonal


def _butterfly_permutation(out_dim: int, factor: int) -> torch.Tensor:
    """Index permutation for the ``factor``-th butterfly stage.

    Mirrors LyCORIS: at stage ``i``, the permutation interleaves rows
    by stride ``2^i``.  Returns a 1-D index tensor ``perm`` such that
    ``y = x[perm]`` applies the permutation.
    """
    stride = 2 ** factor
    if stride >= out_dim:
        return torch.arange(out_dim)
    # Group rows into pairs (low, high) with stride ``stride``;
    # shuffled = [low0, high0, low1, high1, ...]
    idx = torch.arange(out_dim).reshape(-1, 2 * stride)
    lows = idx[:, :stride]  # (G, stride)
    highs = idx[:, stride:]
    interleaved = torch.empty_like(idx)
    interleaved[:, 0::2] = lows
    interleaved[:, 1::2] = highs
    return interleaved.reshape(-1)


class BOFTModule(BaseLoRAModule):
    """Butterfly composition of block-diagonal orthogonal matrices.

    ``lora_dim`` repurposed as the block size ``r`` (default 2 per
    upstream).  ``alpha`` informational (no scale on R).
    """

    supports_conv2d = False

    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=2,
        alpha=1,
        dropout=None,
        rank_dropout=None,
        module_dropout=None,
        channel_scale=None,
        boft_factors: int = 4,
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
                f"BOFTModule supports Linear only "
                f"(got {type(org_module).__name__})"
            )

        out_dim = org_module.out_features
        # Block size r — round down to a divisor of out_dim that's
        # also a power of 2 (butterfly cleanly partitions powers of 2).
        r = min(lora_dim, out_dim)
        while out_dim % r != 0 and r > 1:
            r -= 1
        if r < 2:
            raise ValueError(
                f"BOFTModule needs block size >= 2; out_features={out_dim} "
                f"has no usable divisor"
            )
        self._block_size = r
        self._num_blocks = out_dim // r
        self._num_factors = max(1, boft_factors)

        # m × K × r × r skew-symmetric parameter, zero-init so each B_i
        # starts as identity → R = I → forward bit-equivalent.
        self.boft_skew = torch.nn.Parameter(
            torch.zeros(self._num_factors, self._num_blocks, r, r)
        )

        # Pre-compute butterfly permutations (and their inverses).
        self.register_buffer(
            "_perms",
            torch.stack(
                [
                    _butterfly_permutation(out_dim, f)
                    for f in range(self._num_factors)
                ]
            ),
            persistent=False,
        )
        self.register_buffer(
            "_inv_perms",
            torch.stack(
                [torch.argsort(self._perms[f]) for f in range(self._num_factors)]
            ),
            persistent=False,
        )

        if channel_scale is not None:
            raise ValueError(
                "BOFTModule does not support channel_scale (rebalance is "
                "an input-axis trick; BOFT rotates the output axis)."
            )

        self.org_module_ref = [org_module]
        self._fused = False

    def _rotation_matrix(self) -> torch.Tensor:
        """Compose the m butterfly factors into a single ``out × out`` matrix."""
        out_dim = self._num_blocks * self._block_size
        # Identity start; multiply m factors left-to-right with the
        # appropriate butterfly permutation between each.
        R = torch.eye(
            out_dim, dtype=self.boft_skew.dtype, device=self.boft_skew.device
        )
        for f in range(self._num_factors):
            A = self.boft_skew[f].float()
            Q = A - A.transpose(-1, -2)
            B_blocks = _cayley_orthogonal(Q)  # (K, r, r)
            B = torch.zeros_like(R)
            for k in range(self._num_blocks):
                i, j = k * self._block_size, (k + 1) * self._block_size
                B[i:j, i:j] = B_blocks[k]
            # Permute rows according to butterfly stage ``f``.
            perm = self._perms[f]
            inv_perm = self._inv_perms[f]
            # Apply: R ← P^{-1} · B · P · R   (butterfly conjugation)
            R = R[perm]
            R = B @ R
            R = R[inv_perm]
        return R

    def _effective_weight(
        self, multiplier: Optional[float] = None
    ) -> torch.Tensor:
        m = multiplier if multiplier is not None else self.multiplier
        org_w = self.org_module_ref[0].weight.float()
        R = self._rotation_matrix()
        eye = torch.eye(R.shape[0], dtype=R.dtype, device=R.device)
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
        org_w = self.org_module_ref[0].weight.float()
        return self._effective_weight(multiplier) - org_w

    def merge_to(self, sd, dtype, device):
        with torch.no_grad():
            weight = self.org_module.weight
            org_dtype = weight.dtype
            if dtype is None:
                dtype = org_dtype
            if device is None:
                device = weight.device

            skew_param = sd.get("boft_skew")
            if skew_param is None:
                raise KeyError(
                    f"BOFT merge_to: missing boft_skew for {self.lora_name}"
                )
            A_all = skew_param.to(torch.float).to(device)
            m_factors, K, r, _ = A_all.shape
            out_dim = K * r
            R = torch.eye(out_dim, device=device, dtype=torch.float)
            for f in range(m_factors):
                Q = A_all[f] - A_all[f].transpose(-1, -2)
                B_blocks = _cayley_orthogonal(Q)
                B = torch.zeros_like(R)
                for k in range(K):
                    i, j = k * r, (k + 1) * r
                    B[i:j, i:j] = B_blocks[k]
                perm = self._perms[f].to(device)
                inv_perm = self._inv_perms[f].to(device)
                R = R[perm]
                R = B @ R
                R = R[inv_perm]

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
