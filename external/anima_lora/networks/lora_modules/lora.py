# Classic LoRA. `merge_to` bakes a checkpoint slice into the base Linear/Conv2d
# weight; `fuse_weight` bakes the live delta and turns forward into a no-op.

import math
from typing import Dict, List

import torch

from networks.attn_fuse import match_fused_spec
from networks.lora_modules.base import BaseLoRAModule
from networks.lora_modules.custom_autograd import lora_down_project


class LoRAModule(BaseLoRAModule):
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
        """if alpha == 0 or None, alpha is rank (no scaling)."""
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

        if org_module.__class__.__name__ == "Conv2d":
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            self.lora_down = torch.nn.Conv2d(
                in_dim, self.lora_dim, kernel_size, stride, padding, bias=False
            )
            self.lora_up = torch.nn.Conv2d(
                self.lora_dim, out_dim, (1, 1), (1, 1), bias=False
            )
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.lora_down = torch.nn.Linear(in_dim, self.lora_dim, bias=False)
            self.lora_up = torch.nn.Linear(self.lora_dim, out_dim, bias=False)

        torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_up.weight)

        self._register_channel_scale(self.lora_down.weight.data, channel_scale)

        # Opt-in (Linear-only): save bf16 x instead of fp32 x_lora for backward.
        # Set by the network factory.
        self.use_custom_down_autograd = False

        # List wrapping prevents nn.Module from registering org_module as a
        # submodule (would double-count params). apply_to() deletes
        # self.org_module after rerouting forward, leaving this as the only
        # handle for fuse/unfuse.
        self.org_module_ref = [org_module]
        self._fused = False

    def forward(self, x):
        if not self.enabled or self._fused:
            return self.org_forward(x)

        org_forwarded = self.org_forward(x)

        if not self.training:
            x_lora = self._rebalance(x)
            lx = self.lora_up(self.lora_down(x_lora))
            return org_forwarded + lx * self.multiplier * self.scale

        # Training: bf16 storage, fp32 bottleneck matmuls — recovers mantissa
        # precision that bf16 sheds across the large-embed_dim accumulation.
        if self._skip_module():
            return org_forwarded

        if self.use_custom_down_autograd and isinstance(
            self.lora_down, torch.nn.Linear
        ):
            inv_scale = self.inv_scale if self._has_channel_scale else None
            lx = lora_down_project(x, self.lora_down.weight, inv_scale)
        else:
            x_lora = self._rebalance(x)
            lx = torch.nn.functional.linear(
                x_lora.float(), self.lora_down.weight.float()
            )

        lx = lx * self._timestep_mask

        if self.dropout is not None:
            lx = torch.nn.functional.dropout(lx, p=self.dropout)

        lx, scale = self._apply_rank_dropout(lx)

        lx = torch.nn.functional.linear(lx, self.lora_up.weight.float())
        return org_forwarded + (lx * self.multiplier * scale).to(org_forwarded.dtype)

    def get_weight(self, multiplier=None):
        """Return the LoRA delta as a tensor matching org_module.weight shape."""
        if multiplier is None:
            multiplier = self.multiplier

        up_weight = self.lora_up.weight.to(torch.float)
        down_weight = self.lora_down.weight.to(torch.float)

        # Undo channel absorption so the merged delta applies to raw inputs.
        if self._has_channel_scale and down_weight.dim() == 2:
            down_weight = down_weight * self.inv_scale.to(down_weight).unsqueeze(0)

        if len(down_weight.size()) == 2:
            weight = multiplier * (up_weight @ down_weight) * self.scale
        elif down_weight.size()[2:4] == (1, 1):
            weight = (
                multiplier
                * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2))
                .unsqueeze(2)
                .unsqueeze(3)
                * self.scale
            )
        else:
            conved = torch.nn.functional.conv2d(
                down_weight.permute(1, 0, 2, 3), up_weight
            ).permute(1, 0, 2, 3)
            weight = multiplier * conved * self.scale

        return weight

    def merge_to(self, sd, dtype, device):
        """Merge a per-LoRA state-dict slice into org_module.weight in-place.

        Alternative to apply_to: delta lands on the base weight, no forward
        hook. `sd` is the raw checkpoint slice — the LoRA module hasn't been
        load_state_dict'd at this point.
        """
        with torch.no_grad():
            weight = self.org_module.weight
            org_dtype = weight.dtype
            if dtype is None:
                dtype = org_dtype
            if device is None:
                device = weight.device

            w = weight.data.float()

            down_weight = sd["lora_down.weight"].to(torch.float).to(device)
            up_weight = sd["lora_up.weight"].to(torch.float).to(device)

            # Merged forward has no x rebalancing — undo absorption first.
            if "inv_scale" in sd:
                inv_scale = sd["inv_scale"].to(torch.float).to(device)
                if down_weight.dim() == 2:
                    down_weight = down_weight * inv_scale.unsqueeze(0)

            if len(w.size()) == 2:
                w += self.multiplier * (up_weight @ down_weight) * self.scale
            elif down_weight.size()[2:4] == (1, 1):
                w += (
                    self.multiplier
                    * (
                        up_weight.squeeze(3).squeeze(2)
                        @ down_weight.squeeze(3).squeeze(2)
                    )
                    .unsqueeze(2)
                    .unsqueeze(3)
                    * self.scale
                )
            else:
                conved = torch.nn.functional.conv2d(
                    down_weight.permute(1, 0, 2, 3), up_weight
                ).permute(1, 0, 2, 3)
                w += self.multiplier * conved * self.scale

            weight.data.copy_(w.to(dtype))

    def fuse_weight(self):
        """Bake LoRA delta into org_module.weight; subsequent forwards no-op."""
        if self._fused:
            return
        org_module = self.org_module_ref[0]
        delta = self.get_weight().to(org_module.weight.dtype)
        org_module.weight.data += delta
        self._fused = True

    def unfuse_weight(self):
        """Subtract a previously fused LoRA delta back out of org_module.weight."""
        if not self._fused:
            return
        org_module = self.org_module_ref[0]
        delta = self.get_weight().to(org_module.weight.dtype)
        org_module.weight.data -= delta
        self._fused = False


# ---------------------------------------------------------------------------
# Save-pipeline helpers (state_dict-level, no module instance required).
#
# Co-located with LoRAModule because they operate on the layout this class
# writes (``.lora_down.weight`` / ``.lora_up.weight`` / ``.alpha`` /
# optional ``.dora_scale``). The standard variant write fires these; the
# Hydra and Chimera writers also defuse their plain-LoRA legs by calling
# :func:`defuse_standard_qkv` directly.
# ---------------------------------------------------------------------------


def rename_dora_keys(state_dict: Dict[str, torch.Tensor]) -> None:
    """Rename ``.magnitude`` → ``.dora_scale`` and drop ``._org_weight_norm``.

    DoRA training stores its learned column-norm vector under
    ``.magnitude``; ComfyUI's LoRA loader looks for ``.dora_scale``. The
    ``_org_weight_norm`` buffer is a training-only frozen reference and
    isn't consumed downstream.
    """
    for key in list(state_dict.keys()):
        if key.endswith(".magnitude"):
            new_key = key.replace(".magnitude", ".dora_scale")
            state_dict[new_key] = state_dict.pop(key)
        elif key.endswith("._org_weight_norm"):
            del state_dict[key]


def defuse_standard_qkv(state_dict: Dict[str, torch.Tensor]) -> None:
    """Split runtime-fused ``…_qkv_proj`` / ``…_kv_proj`` keys per-component.

    Operates on the plain LoRA layout (single ``.lora_down.weight`` +
    single ``.lora_up.weight`` per fused Linear). The down projection is
    cloned per component; the up projection (rows = concatenated output
    channels) is chunked along dim 0. ``.alpha`` / ``.dora_scale`` get
    cloned/chunked alongside.

    Used by:
      * the standard write path,
      * the Hydra write path's "plain-LoRA leg" (modules excluded from
        ``router_targets`` save under the plain layout),
      * the Chimera write path's plain-LoRA leg (router_targets excludes
        attention projections by default — OrthoLoRA fallback lands as
        plain LoRA after the ortho distill step).
    """
    fused_groups: List[tuple] = []
    for key in list(state_dict.keys()):
        if not key.endswith(".lora_down.weight"):
            continue
        prefix = key.removesuffix(".lora_down.weight")
        spec = match_fused_spec(prefix)
        if spec is not None:
            fused_groups.append((prefix, spec))

    for prefix, spec in fused_groups:
        suffixes = spec.component_letters
        n = len(suffixes)
        down = state_dict.pop(f"{prefix}.lora_down.weight")
        up = state_dict.pop(f"{prefix}.lora_up.weight")
        alpha = state_dict.pop(f"{prefix}.alpha", None)
        dora_scale = state_dict.pop(f"{prefix}.dora_scale", None)

        up_chunks = up.chunk(n, dim=0)
        dora_chunks = (
            dora_scale.chunk(n, dim=0) if dora_scale is not None else [None] * n
        )

        base_prefix = prefix.removesuffix(spec.fused_frag)
        for letter, up_chunk, dora_chunk in zip(suffixes, up_chunks, dora_chunks):
            new_prefix = base_prefix + spec.component_frag(letter)
            state_dict[f"{new_prefix}.lora_down.weight"] = down.clone()
            state_dict[f"{new_prefix}.lora_up.weight"] = up_chunk
            if alpha is not None:
                state_dict[f"{new_prefix}.alpha"] = alpha.clone()
            if dora_chunk is not None:
                state_dict[f"{new_prefix}.dora_scale"] = dora_chunk


def rename_dora_and_defuse_standard(
    state_dict: Dict[str, torch.Tensor],
) -> None:
    """Standard write pipeline: DoRA rename + qkv defuse, in that order."""
    rename_dora_keys(state_dict)
    defuse_standard_qkv(state_dict)
