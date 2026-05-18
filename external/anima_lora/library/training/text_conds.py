"""Text-encoder conditioning prep: unpack, H2D move, caption dropout.

The dataloader hands us a 4- or 5-tuple from the text-encoder cache. In
the 5-tuple case the cache includes ``crossattn_emb`` (already in T5
target space); in the 4-tuple case the trainer runs the text encoder on
the fly via ``prompt_embeds``.

Caption dropout writes in-place on the GPU tensors after the H2D copy,
which avoids cloning the dataloader's CPU tensors on the critical path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import torch


@dataclass(slots=True)
class PreparedTextConds:
    """Device-resident text conditioning for a single forward pass."""

    # T5-target embedding (cached path). None when training the text encoder.
    crossattn_emb: Optional[torch.Tensor]
    # Qwen3 conditioning (live-encoded path). None on cached crossattn path.
    prompt_embeds: Optional[torch.Tensor]
    attn_mask: Optional[torch.Tensor]
    t5_input_ids: Optional[torch.Tensor]
    t5_attn_mask: Optional[torch.Tensor]
    # CPU-precomputed max sequence length for KV trimming. None when trim is off.
    max_crossattn_seqlen: Optional[int]


def prepare_text_conds(
    *,
    text_encoder_conds: Sequence[Optional[torch.Tensor]],
    batch: Mapping[str, Any],
    text_encoding_strategy: Any,
    network: Any,
    device: torch.device,
    weight_dtype: torch.dtype,
    trim_crossattn_kv: bool,
) -> PreparedTextConds:
    """Unpack the conds tuple, move to device, and apply caption dropout in-place."""
    # Unpack
    crossattn_emb: Optional[torch.Tensor] = None
    if len(text_encoder_conds) == 5:
        prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask, crossattn_emb = (
            text_encoder_conds
        )
    else:
        prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask = text_encoder_conds

    # Pre-compute max sequence length on CPU to avoid GPU sync in KV trimming
    max_crossattn_seqlen: Optional[int] = None
    if trim_crossattn_kv and t5_attn_mask is not None:
        max_crossattn_seqlen = int(t5_attn_mask.sum(dim=-1).max())

    # H2D move
    if crossattn_emb is None:
        prompt_embeds = prompt_embeds.to(device, dtype=weight_dtype)
        attn_mask = attn_mask.to(device)
        t5_input_ids = t5_input_ids.to(device, dtype=torch.long)
        t5_attn_mask = t5_attn_mask.to(device)
    else:
        crossattn_emb = crossattn_emb.to(device, dtype=weight_dtype)
        if trim_crossattn_kv or hasattr(network, "append_postfix"):
            t5_attn_mask = t5_attn_mask.to(device)

    # On-device caption dropout. The freshly-transferred GPU tensors are
    # not aliased to the dataloader's CPU copies, so we can write in-place
    # — no clones, no main-thread CPU memcpy on the critical path.
    caption_dropout_rates = (
        batch.get("caption_dropout_rates") if isinstance(batch, dict) else None
    )
    if caption_dropout_rates is not None:
        if crossattn_emb is None:
            text_encoding_strategy.apply_caption_dropout_inplace(
                caption_dropout_rates,
                prompt_embeds=prompt_embeds,
                attn_mask=attn_mask,
                t5_input_ids=t5_input_ids,
                t5_attn_mask=t5_attn_mask,
            )
        else:
            # prompt_embeds / attn_mask / t5_input_ids stay on CPU because
            # they're unused downstream — only zero what the model actually
            # consumes (and only touch t5_attn_mask if it was moved above).
            text_encoding_strategy.apply_caption_dropout_inplace(
                caption_dropout_rates,
                crossattn_emb=crossattn_emb,
                t5_attn_mask=(
                    t5_attn_mask
                    if t5_attn_mask is not None and t5_attn_mask.is_cuda
                    else None
                ),
            )

    return PreparedTextConds(
        crossattn_emb=crossattn_emb,
        prompt_embeds=prompt_embeds if crossattn_emb is None else None,
        attn_mask=attn_mask if crossattn_emb is None else None,
        t5_input_ids=t5_input_ids if crossattn_emb is None else None,
        t5_attn_mask=t5_attn_mask,
        max_crossattn_seqlen=max_crossattn_seqlen,
    )
