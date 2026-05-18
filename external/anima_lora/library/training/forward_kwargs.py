"""Build the kwargs dict (and rewrite ``crossattn_emb``) for the DiT forward.

Two concerns sit here:

* **Postfix injection** — for networks with ``append_postfix``, splice the
  learned postfix vectors onto the cached T5 embedding and pool the *real*
  text BEFORE the splice so modulation guidance only sees real text.
* **KV trim** — when ``trim_crossattn_kv`` is on, pass per-sample
  cross-attention sequence lengths so the attention kernel can skip the
  zero-padded tail. Postfix tokens are appended after the real tokens, so
  the trim length is the real seqlen + the postfix width.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch


@dataclass(slots=True)
class ForwardKwargs:
    """Outputs of ``build_forward_kwargs``.

    ``crossattn_emb`` is the (possibly postfix-extended) embedding; ``kw``
    is the dict of additional kwargs to pass to the DiT call.
    """

    crossattn_emb: torch.Tensor
    kw: dict[str, Any]
    has_postfix: bool


def build_forward_kwargs(
    *,
    network: Any,
    crossattn_emb: torch.Tensor,
    t5_attn_mask: Optional[torch.Tensor],
    timesteps: torch.Tensor,
    max_crossattn_seqlen: Optional[int],
    trim_crossattn_kv: bool,
) -> ForwardKwargs:
    has_postfix = hasattr(network, "append_postfix")
    kw: dict[str, Any] = {}

    if has_postfix:
        # Pool text BEFORE injection so modulation guidance sees only real text.
        kw["pooled_text_override"] = crossattn_emb.max(dim=1).values
        seqlens = t5_attn_mask.sum(dim=-1).to(torch.int32)
        crossattn_emb = network.append_postfix(
            crossattn_emb, seqlens, timesteps=timesteps
        )

    if trim_crossattn_kv:
        crossattn_seqlens = t5_attn_mask.sum(dim=-1).to(torch.int32)
        max_cs = max_crossattn_seqlen
        if has_postfix:
            crossattn_seqlens = crossattn_seqlens + network.num_postfix_tokens
            if max_cs is not None:
                max_cs = max_cs + network.num_postfix_tokens
        kw["crossattn_seqlens"] = crossattn_seqlens
        kw["max_crossattn_seqlen"] = max_cs

    return ForwardKwargs(crossattn_emb=crossattn_emb, kw=kw, has_postfix=has_postfix)
