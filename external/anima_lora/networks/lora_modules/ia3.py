# IA3 — Infused Adapter by Inhibiting and Amplifying Inner Activations
# (Liu et al., NeurIPS 2022, arXiv:2205.05638).
#
# Per-output-channel rescaling — the simplest possible PEFT method:
#
#     y = (W·x + b) ⊙ ℓ
#
# ``ℓ ∈ R^{out}`` is the only learnable tensor (initialised to ones so
# step 0 is bit-equivalent to the unwrapped Linear).  Foldable into
# the host weight at save time:  ``W' = ℓ ⊙ W,  b' = ℓ ⊙ b``.
#
# Conv2d is supported the same way — broadcast ``ℓ`` along the
# channel-out axis.  Channel-scale (SmoothQuant) absorption is *not*
# wired here because IA3 doesn't multiply along the input dim — the
# rebalance trick targets activation magnitudes per input channel,
# which doesn't match the per-output rescale.

from typing import Optional

import torch

from networks.lora_modules.base import BaseLoRAModule


class IA3Module(BaseLoRAModule):
    """Per-output-channel learnable scale on a Linear / Conv2d.

    Inherits :class:`BaseLoRAModule` for ``apply_to`` plumbing but does
    *not* use the LoRA ``lora_down`` / ``lora_up`` legs; ``lora_dim``
    in this context is ignored at the math level (still tracked so the
    adapter scaffolding stays consistent with the rest of the family).
    """

    supports_conv2d = True

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

        out_dim = (
            org_module.out_channels
            if org_module.__class__.__name__ == "Conv2d"
            else org_module.out_features
        )
        # IA3 weight — initialised to ones so step 0 is identity. Stored
        # as a 1-D Parameter; broadcast at the multiply site to match
        # both Linear (B, ..., out) and Conv2d (B, out, H, W) layouts.
        self.ia3_weight = torch.nn.Parameter(torch.ones(out_dim))

        if channel_scale is not None:
            # IA3 doesn't act on the input axis, so SmoothQuant-style
            # input absorption has no semantically equivalent home
            # here. Reject loudly so users don't ship it expecting a
            # silent fallback.
            raise ValueError(
                "IA3Module does not support channel_scale — the rebalance "
                "trick rescales input columns, but IA3 only multiplies the "
                "output. Drop channel_scale or pick a different variant."
            )

        # List wrapping prevents nn.Module from registering org_module as a
        # submodule (would double-count params). apply_to() deletes
        # self.org_module after rerouting forward, leaving this as the only
        # handle for fuse/unfuse.
        self.org_module_ref = [org_module]
        self._fused = False

    def forward(self, x):
        if not self.enabled or self._fused:
            return self.org_forward(x)

        if self.training and self._skip_module():
            return self.org_forward(x)

        y = self.org_forward(x)
        scale = self.ia3_weight.to(y.dtype)
        if y.dim() == 4:
            # Conv2d output: (B, out, H, W). Broadcast on channel.
            return y * scale.view(1, -1, 1, 1) * self.multiplier
        # Linear output: (B, ..., out). Broadcast on last dim.
        return y * scale * self.multiplier

    def get_weight(self, multiplier: Optional[float] = None) -> torch.Tensor:
        """Return the additive weight delta vs W₀.

        IA3 multiplies, so the "delta" is signed:  ``ΔW = (ℓ - 1) · W₀``.
        ``multiplier`` scales the *adapter strength* — at multiplier=0 the
        delta must be zero. To make the math consistent we interpolate
        ``ℓ_eff = 1 + multiplier · (ℓ - 1)``.
        """
        m = multiplier if multiplier is not None else self.multiplier
        org_weight = self.org_module_ref[0].weight.float()
        scale = 1.0 + m * (self.ia3_weight.float() - 1.0)
        if org_weight.dim() == 2:
            scaled = scale.unsqueeze(1) * org_weight
        else:
            scaled = scale.view(-1, 1, 1, 1) * org_weight
        return scaled - org_weight

    def merge_to(self, sd, dtype, device):
        """Bake checkpoint slice into ``org_module.weight`` (and bias)."""
        with torch.no_grad():
            weight = self.org_module.weight
            org_dtype = weight.dtype
            if dtype is None:
                dtype = org_dtype
            if device is None:
                device = weight.device

            ia3 = sd.get("ia3_weight")
            if ia3 is None:
                raise KeyError(
                    f"IA3 merge_to: missing ia3_weight key for {self.lora_name}"
                )
            ia3 = ia3.to(torch.float).to(device)

            w = weight.data.float()
            scale = 1.0 + self.multiplier * (ia3 - 1.0)
            if w.dim() == 2:
                w = scale.unsqueeze(1) * w
            else:
                w = scale.view(-1, 1, 1, 1) * w
            weight.data.copy_(w.to(dtype))

            bias = getattr(self.org_module, "bias", None)
            if bias is not None:
                bias.data.copy_(
                    (bias.data.float() * scale).to(bias.data.dtype)
                )

    def fuse_weight(self):
        """Bake into org_module.weight + bias; subsequent forwards no-op."""
        if self._fused:
            return
        org_module = self.org_module_ref[0]
        scale = self.ia3_weight.float()
        w = org_module.weight.data.float()
        if w.dim() == 2:
            self._w0_backup = w.clone()
            org_module.weight.data.copy_(
                (scale.unsqueeze(1) * w).to(org_module.weight.dtype)
            )
        else:
            self._w0_backup = w.clone()
            org_module.weight.data.copy_(
                (scale.view(-1, 1, 1, 1) * w).to(org_module.weight.dtype)
            )

        bias = getattr(org_module, "bias", None)
        if bias is not None:
            self._bias_backup = bias.data.clone()
            bias.data.copy_((bias.data.float() * scale).to(bias.data.dtype))

        self._fused = True

    def unfuse_weight(self):
        """Restore W₀ + bias from the stash taken at fuse_weight."""
        if not self._fused:
            return
        org_module = self.org_module_ref[0]
        org_module.weight.data.copy_(self._w0_backup)
        del self._w0_backup
        bias = getattr(org_module, "bias", None)
        if bias is not None and hasattr(self, "_bias_backup"):
            bias.data.copy_(self._bias_backup)
            del self._bias_backup
        self._fused = False
