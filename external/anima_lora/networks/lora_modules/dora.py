# DoRA — Weight-decomposed LoRA (Liu et al. ICML'24, arXiv:2402.09353).
#
# Decomposes each adapted Linear's weight into a directional unit-norm
# matrix and a per-output-channel magnitude vector:
#
#     W_eff = m · (W₀ + ΔW) / ‖W₀ + ΔW‖_c
#
# Training freezes W₀, learns ΔW via a standard LoRA pair, and learns a
# fresh ``magnitude`` vector. At save time the magnitude is renamed
# ``.dora_scale`` (see ``rename_dora_keys`` in lora.py) so ComfyUI's
# stock LoRA loader consumes it without a custom node.
#
# Conv2d is supported — column-norm is taken across all non-output dims
# (matches LyCORIS's ``dora_wd`` layout).

from typing import Optional

import torch

from networks.lora_modules.lora import LoRAModule


class DoRAModule(LoRAModule):
    """LoRA + per-output-channel magnitude vector.

    Inherits the LoRA legs (``lora_down`` / ``lora_up``) and channel
    scale plumbing from :class:`LoRAModule`; replaces ``forward`` with
    DoRA's direction-magnitude split.
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
            channel_scale=channel_scale,
        )

        # Trainable magnitude — initialised to W₀'s column norm so step 0
        # leaves the layer's output bit-equivalent to the unwrapped
        # Linear (LoRA delta is also zero at init since lora_up.weight
        # is zero-init).
        org_norm = self._column_norm(org_module.weight.detach())
        self.magnitude = torch.nn.Parameter(org_norm.clone())

    @staticmethod
    def _column_norm(weight: torch.Tensor) -> torch.Tensor:
        """L2 norm along everything except the output channel.

        Linear ``(out, in)`` → ``(out, 1)``; Conv2d ``(out, in, kH, kW)``
        → ``(out, 1, 1, 1)``. Computed in fp32 for stability.
        """
        if weight.dim() == 2:
            return weight.float().norm(dim=1, keepdim=True)
        return (
            weight.float()
            .reshape(weight.shape[0], -1)
            .norm(dim=1)
            .reshape(weight.shape[0], 1, 1, 1)
        )

    def _scaled_lora_delta(self, multiplier: Optional[float] = None) -> torch.Tensor:
        """LoRA delta in W-space, channel-scale undo applied."""
        if multiplier is None:
            multiplier = self.multiplier
        up_w = self.lora_up.weight.float()
        down_w = self.lora_down.weight.float()
        if self._has_channel_scale and down_w.dim() == 2:
            down_w = down_w * self.inv_scale.to(down_w).unsqueeze(0)
        if down_w.dim() == 2:
            return multiplier * (up_w @ down_w) * self.scale
        if down_w.size()[2:4] == (1, 1):
            return (
                multiplier
                * (up_w.squeeze(3).squeeze(2) @ down_w.squeeze(3).squeeze(2))
                .unsqueeze(2)
                .unsqueeze(3)
                * self.scale
            )
        conved = torch.nn.functional.conv2d(
            down_w.permute(1, 0, 2, 3), up_w
        ).permute(1, 0, 2, 3)
        return multiplier * conved * self.scale

    def _effective_weight(self) -> torch.Tensor:
        """Return ``m · (W₀ + ΔW) / ‖W₀ + ΔW‖_c`` in fp32."""
        org = self.org_module_ref[0].weight
        merged = org.float() + self._scaled_lora_delta()
        norm = self._column_norm(merged).clamp_min(1e-12)
        return (self.magnitude.float() / norm) * merged

    def forward(self, x):
        if not self.enabled or self._fused:
            return self.org_forward(x)
        if self.training and self._skip_module():
            return self.org_forward(x)

        org = self.org_module_ref[0]
        eff = self._effective_weight().to(x.dtype)

        if eff.dim() == 2:
            return torch.nn.functional.linear(x, eff, org.bias)
        return torch.nn.functional.conv2d(
            x,
            eff,
            org.bias,
            stride=org.stride,
            padding=org.padding,
            dilation=org.dilation,
            groups=org.groups,
        )

    def get_weight(self, multiplier=None):
        """Effective DoRA weight delta vs W₀ (signed, fp32)."""
        org = self.org_module_ref[0].weight.float()
        merged = org + self._scaled_lora_delta(multiplier)
        norm = self._column_norm(merged).clamp_min(1e-12)
        # Use the requested multiplier on the magnitude scale too — at
        # multiplier=0 we should recover W₀ exactly.
        m = multiplier if multiplier is not None else self.multiplier
        scaled = (self.magnitude.float() / norm) * merged
        return m * (scaled - org)

    def merge_to(self, sd, dtype, device):
        """Bake a checkpoint slice into ``org_module.weight``.

        Reconstructs ``W_eff`` from ``lora_down`` / ``lora_up`` /
        ``dora_scale`` (ComfyUI rename). ``magnitude`` is the legacy
        in-memory key — accept both for round-trip with old runs.
        """
        with torch.no_grad():
            weight = self.org_module.weight
            org_dtype = weight.dtype
            if dtype is None:
                dtype = org_dtype
            if device is None:
                device = weight.device

            w0 = weight.data.float()
            down_w = sd["lora_down.weight"].to(torch.float).to(device)
            up_w = sd["lora_up.weight"].to(torch.float).to(device)
            mag = sd.get("dora_scale", sd.get("magnitude"))
            if mag is None:
                raise KeyError(
                    f"DoRA merge_to: missing dora_scale/magnitude for {self.lora_name}"
                )
            mag = mag.to(torch.float).to(device)

            if "inv_scale" in sd:
                inv_scale = sd["inv_scale"].to(torch.float).to(device)
                if down_w.dim() == 2:
                    down_w = down_w * inv_scale.unsqueeze(0)

            if w0.dim() == 2:
                delta = self.multiplier * (up_w @ down_w) * self.scale
                merged = w0 + delta
                norm = merged.norm(dim=1, keepdim=True).clamp_min(1e-12)
                mag = mag.reshape(merged.shape[0], 1)
            elif down_w.size()[2:4] == (1, 1):
                delta = (
                    self.multiplier
                    * (
                        up_w.squeeze(3).squeeze(2)
                        @ down_w.squeeze(3).squeeze(2)
                    )
                    .unsqueeze(2)
                    .unsqueeze(3)
                    * self.scale
                )
                merged = w0 + delta
                norm = (
                    merged.reshape(merged.shape[0], -1)
                    .norm(dim=1)
                    .reshape(merged.shape[0], 1, 1, 1)
                    .clamp_min(1e-12)
                )
                mag = mag.reshape(merged.shape[0], 1, 1, 1)
            else:
                conved = torch.nn.functional.conv2d(
                    down_w.permute(1, 0, 2, 3), up_w
                ).permute(1, 0, 2, 3)
                delta = self.multiplier * conved * self.scale
                merged = w0 + delta
                norm = (
                    merged.reshape(merged.shape[0], -1)
                    .norm(dim=1)
                    .reshape(merged.shape[0], 1, 1, 1)
                    .clamp_min(1e-12)
                )
                mag = mag.reshape(merged.shape[0], 1, 1, 1)

            weight.data.copy_(((mag / norm) * merged).to(dtype))

    def fuse_weight(self):
        """Bake into org_module.weight; subsequent forwards no-op."""
        if self._fused:
            return
        org_module = self.org_module_ref[0]
        # Stash W₀ so unfuse_weight can restore exactly.
        self._w0_backup = org_module.weight.data.detach().clone()
        org_module.weight.data.copy_(
            self._effective_weight().to(org_module.weight.dtype)
        )
        self._fused = True

    def unfuse_weight(self):
        """Restore W₀ from the stash taken at ``fuse_weight``."""
        if not self._fused:
            return
        org_module = self.org_module_ref[0]
        org_module.weight.data.copy_(self._w0_backup)
        del self._w0_backup
        self._fused = False
