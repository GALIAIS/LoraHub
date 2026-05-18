"""T5("") unconditional cross-attention sidecar.

Both the prep CLI and the training CLI route their unconditional text input
through the same staged file — ``<cache_dir>/_anima_uncond_te.safetensors`` —
so the student's CFG-uncond branch matches Anima's own inference path
(``library/inference/text.py:99-127``). This is paper-faithful (Starodubcev
et al., ICLR 2026, arXiv:2602.09268v1 §5) and avoids the
``torch.zeros_like(crossattn_emb)`` shortcut that would be neither.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file as _load_safetensors
from safetensors.torch import save_file

from library.inference.text import MAX_CROSSATTN_TOKENS

logger = logging.getLogger(__name__)

UNCOND_TE_FILENAME = "_anima_uncond_te.safetensors"
DEFAULT_SEQ_LEN = MAX_CROSSATTN_TOKENS  # matches library/inference/text.py CFG-uncond padding


def encode_uncond_crossattn(
    qwen3_path: str,
    dit_path: str,
    *,
    t5_tokenizer_path: str | None = None,
    seq_len: int = DEFAULT_SEQ_LEN,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run ``T5("")`` through Qwen3 + LLM adapter, zero padding positions,
    pad/truncate to ``seq_len``. Returns ``(crossattn_emb, pooled)``, both bf16
    on CPU. Shape: ``(seq_len, 1024)`` and ``(1024,)``.

    Mirrors the negative-prompt path in ``library/inference/text.py:99-127``
    and the encode path in ``preprocess/cache_text_embeddings.py:71-105``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from library.anima import weights as anima_utils
    from library.anima.strategy import AnimaTextEncodingStrategy, AnimaTokenizeStrategy

    logger.info(f"Loading Qwen3 text encoder from {qwen3_path} ...")
    text_encoder, qwen3_tokenizer = anima_utils.load_qwen3_text_encoder(
        qwen3_path, dtype=torch.bfloat16, device=str(device)
    )
    t5_tokenizer = anima_utils.load_t5_tokenizer(t5_tokenizer_path)

    logger.info(f"Loading LLM adapter from {dit_path} ...")
    llm_adapter = anima_utils.load_llm_adapter(
        dit_path, dtype=torch.bfloat16, device=str(device)
    )

    tokenize_strategy = AnimaTokenizeStrategy(
        qwen3_tokenizer=qwen3_tokenizer, t5_tokenizer=t5_tokenizer
    )
    encoding_strategy = AnimaTextEncodingStrategy()

    with torch.no_grad():
        tokens_and_masks = tokenize_strategy.tokenize([""])
        prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask = (
            encoding_strategy.encode_tokens(
                tokenize_strategy, [text_encoder], tokens_and_masks
            )
        )
        crossattn_emb = llm_adapter(
            source_hidden_states=prompt_embeds,
            target_input_ids=t5_input_ids.to(device, dtype=torch.long),
            target_attention_mask=t5_attn_mask.to(device),
            source_attention_mask=attn_mask,
        )
        # Zero padding positions — attention sinks in cross-attention softmax.
        crossattn_emb[~t5_attn_mask.to(device).bool()] = 0

    # crossattn_emb: (1, S, 1024). Pad or truncate to seq_len.
    cur_seq = crossattn_emb.shape[1]
    if cur_seq < seq_len:
        crossattn_emb = F.pad(crossattn_emb, (0, 0, 0, seq_len - cur_seq))
    elif cur_seq > seq_len:
        crossattn_emb = crossattn_emb[:, :seq_len, :]

    crossattn_emb = crossattn_emb.squeeze(0).to(dtype=torch.bfloat16).cpu()
    pooled = crossattn_emb.amax(dim=0)  # matches load_cached_text_features fallback

    text_encoder.to("cpu")
    del text_encoder, llm_adapter
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return crossattn_emb, pooled


def stage_uncond_sidecar(
    cache_dir: Path,
    qwen3_path: str,
    dit_path: str,
    *,
    t5_tokenizer_path: str | None,
    seq_len: int,
    overwrite: bool,
) -> Path:
    """Phase 1 entry point. Writes ``<cache_dir>/_anima_uncond_te.safetensors``."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / UNCOND_TE_FILENAME

    if out_path.exists() and not overwrite:
        logger.info(
            f"Uncond sidecar already exists at {out_path}; pass --overwrite to regenerate."
        )
        return out_path

    crossattn_emb, pooled = encode_uncond_crossattn(
        qwen3_path,
        dit_path,
        t5_tokenizer_path=t5_tokenizer_path,
        seq_len=seq_len,
    )
    save_file({"crossattn_emb": crossattn_emb, "pooled": pooled}, str(out_path))
    logger.info(
        f"Wrote {out_path}  (crossattn_emb={tuple(crossattn_emb.shape)}, "
        f"pooled={tuple(pooled.shape)}, dtype={crossattn_emb.dtype})"
    )
    return out_path


def load_uncond_crossattn(path: str, device, dtype) -> torch.Tensor:
    """Load the ``T5("")`` sidecar staged by ``make distill-prep`` and return a
    ``(1, seq, 1024)`` tensor on ``device`` in ``dtype``. Used as the student's
    unconditional cross-attention input; replaces ``torch.zeros_like(...)``,
    which is neither paper-faithful nor what Anima uses at CFG-uncond inference.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Unconditional TE sidecar not found at {path!r}. "
            f"Run `make distill-prep` (or `python tasks.py distill-prep`) first."
        )
    sd = _load_safetensors(path)
    uncond = sd.get("crossattn_emb")
    if uncond is None:
        raise KeyError(
            f"Expected key 'crossattn_emb' in {path!r}; got {list(sd.keys())}"
        )
    if uncond.dim() != 2:
        raise ValueError(
            f"Expected (seq, dim) tensor in {path!r}; got shape {tuple(uncond.shape)}"
        )
    return uncond.to(device=device, dtype=dtype).unsqueeze(0).contiguous()


def uncond_for_batch(uncond_1: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Broadcast ``uncond_1`` (1, S_u, D) to ``(B, S_ref, D)`` matching ``ref``.
    Pads with zeros (attention sinks) if ``S_u < S_ref``; truncates if larger.
    """
    B, S_ref, _D = ref.shape
    S_u = uncond_1.shape[1]
    if S_u < S_ref:
        uncond_1 = F.pad(uncond_1, (0, 0, 0, S_ref - S_u))
    elif S_u > S_ref:
        uncond_1 = uncond_1[:, :S_ref, :]
    return uncond_1.expand(B, -1, -1)
