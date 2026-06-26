# LoKr — Low-rank Kronecker decomposition (LyCORIS, arXiv:2212.10650).
#
# Decomposes ΔW into a Kronecker product of a small dense matrix and a
# low-rank LoRA pair:
#
#     ΔW = α/r · (W₁ ⊗ (B · A))                       (Linear)
#
# where the host weight is reshaped ``(out=a·c, in=b·d)``. ``W₁`` is
# ``(a, b)`` and trainable; ``A`` is ``(r, d)`` and ``B`` is ``(c, r)``.
# Parameter count is ``a·b + c·r + r·d`` — tiny vs. ``out·in`` for
# typical DiT projections.
#
# The split factor selects ``a, b, c, d`` so ``a ≤ factor`` and the
# reshape is exact. We pick ``a`` as the largest divisor of ``out`` not
# exceeding ``factor``, then ``c = out / a``; same for ``b, d``.
#
# Conv2d not supported in this Phase-1 cut — the tensor reshape is more
# involved (kernel × kernel × in_chunks × out_chunks) and DiT projection
# layers are Linear. LyCORIS-style locon support can land later.

import math
from typing import Dict, Optional, Tuple

import torch

from networks.lora_modules.base import BaseLoRAModule


def _factorise(dim: int, factor: int) -> Tuple[int, int]:
    """Pick ``(a, c)`` such that ``a · c = dim`` and ``a ≤ factor``.

    LyCORIS picks the largest valid ``a``; smaller ``a`` makes ``W₁``
    less expressive but ``c`` (the LoRA leg) bigger. We follow upstream:
    largest divisor of ``dim`` not exceeding ``factor``.
    """
    if factor <= 1:
        return 1, dim
    # Walk down from min(factor, sqrt(dim)) until we find a divisor.
    cap = min(factor, dim)
    for a in range(cap, 0, -1):
        if dim % a == 0:
            return a, dim // a
    # ``a = 1`` always divides; loop guarantees a return.
    return 1, dim  # pragma: no cover


class LoKrModule(BaseLoRAModule):
    """Kronecker decomposition LoRA (Linear only).

    Layout::

        ΔW = (α / r) · (W₁ ⊗ (B · A))
        W₁ ∈ R^{a × b}      lokr_w1
        A  ∈ R^{r × d}      lokr_w2_a   (input leg)
        B  ∈ R^{c × r}      lokr_w2_b   (output leg)

    The full ΔW is materialised at forward time — for typical DiT
    projection sizes (`out`, `in` ≤ a few thousand) this is ``out·in``
    fp32, well within VRAM. The compute saving is in *parameter count*,
    not in matmul cost.
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
        factor: int = 8,
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
                f"LoKrModule supports Linear only (got {type(org_module).__name__})"
            )
        out_dim = org_module.out_features
        in_dim = org_module.in_features
        a, c = _factorise(out_dim, factor)
        b, d = _factorise(in_dim, factor)
        self._shape = (a, b, c, d)

        rank = max(1, lora_dim)
        # Match LyCORIS init: w1 ~ N(0, 1) scaled, w2_b zero-init so
        # ΔW = 0 at step 0.
        self.lokr_w1 = torch.nn.Parameter(torch.empty(a, b))
        self.lokr_w2_a = torch.nn.Parameter(torch.empty(rank, d))
        self.lokr_w2_b = torch.nn.Parameter(torch.zeros(c, rank))

        torch.nn.init.kaiming_uniform_(self.lokr_w1, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.lokr_w2_a, a=math.sqrt(5))
        # lokr_w2_b stays zero so ΔW = w1 ⊗ 0 = 0.

        if channel_scale is not None:
            raise ValueError(
                "LoKrModule does not support channel_scale (input rebalance "
                "doesn't compose with the Kronecker reshape)."
            )

        self.org_module_ref = [org_module]
        self._fused = False

    def _delta(self, multiplier: Optional[float] = None) -> torch.Tensor:
        """Materialise ΔW in W-space (out × in)."""
        m = multiplier if multiplier is not None else self.multiplier
        # (c, d) low-rank piece.
        w2 = self.lokr_w2_b.float() @ self.lokr_w2_a.float()
        # Kronecker product. torch.kron returns (a·c, b·d) directly.
        kron = torch.kron(self.lokr_w1.float(), w2)
        return m * self.scale * kron

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
        """Effective LoKr delta in W-space."""
        return self._delta(multiplier)

    def merge_to(self, sd, dtype, device):
        """Bake ΔW into ``org_module.weight``."""
        self.normalize_state_dict_for_runtime(sd)
        with torch.no_grad():
            weight = self.org_module.weight
            org_dtype = weight.dtype
            if dtype is None:
                dtype = org_dtype
            if device is None:
                device = weight.device

            w1 = sd["lokr_w1"].to(torch.float).to(device)
            w2_a = sd["lokr_w2_a"].to(torch.float).to(device)
            w2_b = sd["lokr_w2_b"].to(torch.float).to(device)
            w2 = w2_b @ w2_a
            kron = torch.kron(w1, w2)
            delta = self.multiplier * self.scale * kron

            weight.data.copy_((weight.data.float() + delta).to(dtype))

    def normalize_state_dict_for_runtime(self, sd: Dict[str, torch.Tensor]) -> None:
        """Accept both internal legacy and LyCORIS/Comfy LoKr tensor order."""
        a = sd.get("lokr_w2_a")
        b = sd.get("lokr_w2_b")
        if a is None or b is None:
            return
        expected_a = tuple(self.lokr_w2_a.shape)
        expected_b = tuple(self.lokr_w2_b.shape)
        if tuple(a.shape) == expected_a and tuple(b.shape) == expected_b:
            return
        if tuple(a.shape) == expected_b and tuple(b.shape) == expected_a:
            sd["lokr_w2_a"], sd["lokr_w2_b"] = b, a

    def fuse_weight(self):
        if self._fused:
            return
        org_module = self.org_module_ref[0]
        self._w0_backup = org_module.weight.data.detach().clone()
        delta = self._delta().to(org_module.weight.dtype)
        org_module.weight.data.add_(delta)
        self._fused = True

    def unfuse_weight(self):
        if not self._fused:
            return
        org_module = self.org_module_ref[0]
        org_module.weight.data.copy_(self._w0_backup)
        del self._w0_backup
        self._fused = False


class FactorizedLoKrModule(LoKrModule):
    """LoKr with an equivalent factorized adapter forward.

    The checkpoint layout is identical to :class:`LoKrModule`; only training
    forward avoids materialising ``torch.kron(w1, w2)`` and ``weight + delta``.
    """

    def forward(self, x):
        if not self.enabled or self._fused:
            return self.org_forward(x)
        if self.training and self._skip_module():
            return self.org_forward(x)

        org_forwarded = self.org_forward(x)
        a, b, c, d = self._shape
        orig_shape = x.shape[:-1]
        x2 = x.reshape(-1, b, d).float()
        w1 = self.lokr_w1.float()
        w2_a = self.lokr_w2_a.float()
        w2_b = self.lokr_w2_b.float()

        # Two contractions are cheaper than a four-input einsum planner here:
        # first project the d-axis into rank, then mix b/r into a/c.
        y_rank = torch.einsum("nbd,rd->nbr", x2, w2_a)
        y = torch.einsum("nbr,ab,cr->nac", y_rank, w1, w2_b)
        y = y.reshape(*orig_shape, a * c)
        y = y * (self.multiplier * self.scale)
        return org_forwarded + y.to(org_forwarded.dtype)


def lokr_state_dict_to_lycoris(state_dict: Dict[str, torch.Tensor]) -> None:
    """Write LoKr tensors in the common LyCORIS/Comfy ``w2_a @ w2_b`` order."""
    for key in list(state_dict.keys()):
        if not key.endswith(".lokr_w2_a"):
            continue
        prefix = key[: -len(".lokr_w2_a")]
        a_key = f"{prefix}.lokr_w2_a"
        b_key = f"{prefix}.lokr_w2_b"
        a = state_dict.get(a_key)
        b = state_dict.get(b_key)
        if a is None or b is None or a.ndim != 2 or b.ndim != 2:
            continue
        state_dict[a_key], state_dict[b_key] = b, a
