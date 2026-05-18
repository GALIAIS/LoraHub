# Memory-saving autograd for the LoRA down projection.
#
# `F.linear(x.float(), weight.float())` saves the fp32-cast input for backward
# (~32 MiB per 2048-wide Linear at 4096 tokens, ×N adapted modules). These
# Functions save the bf16 `x` and recompute the cast in backward. Forward and
# backward matmuls run in fp32 — bitwise-identical to the existing path for
# deterministic kernels.
#
# Two Functions (scaled / unscaled) instead of one with an optional tensor:
# keeps the compile graph shape fixed.

from __future__ import annotations

import torch


class LoRADownProjectFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight):
        out = torch.nn.functional.linear(x.float(), weight.float())
        ctx.save_for_backward(x, weight)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        go = grad_out.float()
        w_f = weight.float()
        x_f = x.float()

        grad_x = go.matmul(w_f).to(x.dtype)
        grad_weight = go.reshape(-1, go.shape[-1]).transpose(0, 1).matmul(
            x_f.reshape(-1, x_f.shape[-1])
        )
        return grad_x, grad_weight.to(weight.dtype)


class ScaledLoRADownProjectFn(torch.autograd.Function):
    """Scaled variant: forward is ``F.linear((x * inv_scale.to(x.dtype)).float(), weight.float())``.

    Matches the legacy ``_rebalance`` path exactly — ``inv_scale`` is cast to
    ``x.dtype`` (bf16 under full_bf16) before the multiply, so the rebalance
    keeps the bf16 activation chain instead of getting promoted to fp32.

    ``inv_scale`` is a calibration buffer (no gradient). Saving bf16 ``x`` plus
    the 1-D ``inv_scale`` (size == in_features) avoids retaining the
    materialized fp32 ``x * inv_scale`` that the current path otherwise holds.
    """

    @staticmethod
    def forward(ctx, x, weight, inv_scale):
        inv = inv_scale.to(x.dtype)
        x_work = x * inv  # bf16 = bf16 * bf16, matches legacy _rebalance
        out = torch.nn.functional.linear(x_work.float(), weight.float())
        ctx.save_for_backward(x, weight, inv_scale)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weight, inv_scale = ctx.saved_tensors
        inv = inv_scale.to(x.dtype)
        x_work = x * inv  # bf16 recompute matches forward
        go = grad_out.float()
        w_f = weight.float()
        x_f = x_work.float()

        grad_weight = go.reshape(-1, go.shape[-1]).transpose(0, 1).matmul(
            x_f.reshape(-1, x_f.shape[-1])
        )

        # Legacy autograd for ``out = F.linear((x * inv).float(), w.float())``:
        #   grad_y_fp32 = go @ w_f          (fp32)
        #   grad_x_work = grad_y_fp32.to(x.dtype)   (bf16, from .float() backward)
        #   grad_x      = grad_x_work * inv         (bf16, from multiply backward)
        grad_x_work = go.matmul(w_f).to(x.dtype)
        grad_x = grad_x_work * inv
        return grad_x, grad_weight.to(weight.dtype), None


def lora_down_project(x, weight, inv_scale):
    """Dispatch helper: picks the scaled or unscaled Function based on inv_scale."""
    if inv_scale is None:
        return LoRADownProjectFn.apply(x, weight)
    return ScaledLoRADownProjectFn.apply(x, weight, inv_scale)
