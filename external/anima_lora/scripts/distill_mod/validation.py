"""Validation pass: teacher↔student MSE on the val set at fixed sigmas.

Noise is drawn from a fixed-seed generator so val loss is comparable across
runs. If a :class:`~scripts.distill_mod.teacher_cache.ValTeacherCache` is
provided, teacher predictions are memoized by ``(batch_idx, sigma_idx)`` —
the first pass fills the cache, every subsequent pass skips the teacher
forward.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .teacher_cache import ValTeacherCache
from .uncond import uncond_for_batch


@torch.no_grad()
def run_validation(
    model,
    val_dataloader,
    *,
    device,
    dtype,
    sigmas: list[float],
    max_steps: int | None,
    seed: int,
    uncond_te_1: torch.Tensor,
    teacher_cache: ValTeacherCache | None = None,
):
    """Compute teacher↔student MSE on the val set at fixed sigmas.

    Returns ``(per_sigma_mean, overall_mean)``.
    """
    gen = torch.Generator(device=device).manual_seed(seed)
    per_sigma: dict[float, list[float]] = {s: [] for s in sigmas}
    overall: list[float] = []

    for i, (_idxs, latents, crossattn_emb, pooled_text) in enumerate(val_dataloader):
        if max_steps is not None and i >= max_steps:
            break
        latents = latents.to(device, dtype=dtype, non_blocking=True)
        crossattn_emb = crossattn_emb.to(device, dtype=dtype, non_blocking=True)
        pooled_text = pooled_text.to(device, dtype=dtype, non_blocking=True)
        B = latents.shape[0]

        noise = torch.randn(
            latents.shape, device=device, dtype=latents.dtype, generator=gen
        )
        padding_mask = torch.zeros(
            B, 1, latents.shape[-2], latents.shape[-1], dtype=dtype, device=device
        )
        uncond = uncond_for_batch(uncond_te_1, crossattn_emb)

        for s_idx, sigma in enumerate(sigmas):
            sig_b = torch.full((B,), float(sigma), device=device, dtype=latents.dtype)
            sig_e = sig_b.view(B, 1, 1, 1)
            noisy = (1.0 - sig_e) * latents + sig_e * noise
            noisy = noisy.unsqueeze(2)

            cached = (
                teacher_cache.get(i, s_idx) if teacher_cache is not None else None
            )
            if cached is not None:
                teacher_pred = cached.to(device, dtype=dtype, non_blocking=True)
            else:
                if model.blocks_to_swap:
                    model.prepare_block_swap_before_forward()
                torch.compiler.cudagraph_mark_step_begin()
                with torch.autocast("cuda", dtype=dtype):
                    teacher_pred = model.forward_mini_train_dit(
                        noisy,
                        sig_b,
                        crossattn_emb,
                        padding_mask=padding_mask,
                        skip_pooled_text_proj=True,
                    )
                teacher_pred = teacher_pred.clone()
                if teacher_cache is not None:
                    teacher_cache.put(i, s_idx, teacher_pred)

            if model.blocks_to_swap:
                model.prepare_block_swap_before_forward()
            torch.compiler.cudagraph_mark_step_begin()
            with torch.autocast("cuda", dtype=dtype):
                student_pred = model.forward_mini_train_dit(
                    noisy,
                    sig_b,
                    uncond,
                    padding_mask=padding_mask,
                    pooled_text_override=pooled_text,
                )

            loss = nn.functional.mse_loss(
                student_pred.float(), teacher_pred.float()
            ).item()
            per_sigma[sigma].append(loss)
            overall.append(loss)

    per_sigma_mean = {
        s: (sum(v) / len(v) if v else float("nan")) for s, v in per_sigma.items()
    }
    overall_mean = sum(overall) / len(overall) if overall else float("nan")
    return per_sigma_mean, overall_mean
