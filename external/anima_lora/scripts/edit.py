"""DirectEdit CLI — image editing via flow inversion + ψ_tar resampling.

Two prompts in (``--prompt_src``, ``--prompt_tar``), one edited image out.
The source prompt feeds the inversion pass; the target prompt drives the
edit forward pass anchored to per-step inversion residuals (DirectEdit,
Yang & Ye arXiv:2605.02417v1).

Usage:
    python scripts/edit.py \
        --image path/to/source.png \
        --prompt_src "1girl, smile, school_uniform" \
        --prompt_tar "1girl, smile, school_uniform, double peace" \
        --dit models/diffusion_models/anima-base-v1.0.safetensors \
        --text_encoder models/text_encoders/qwen_3_06b_base.safetensors \
        --vae models/vae/qwen_image_vae.safetensors \
        --save_path output/tests/directedit/

Wired by ``scripts/experimental_tasks/inference.py::cmd_test_directedit``
under ``make exp-test-directedit`` — that task picks a random source image,
runs the Anima Tagger to seed ``--prompt_src``, and forms ``--prompt_tar``
from ``PROMPT`` env (the user's edit instruction).

v1.1 status:
  * V-injection: WIRED. ``--t_inj N`` injects src self-attn V into the tar
    pass for the first N steps (paper Eq. 13). ``--t_inj_blocks`` selects
    the block subset (default = all but the final block, SD3.5-style).
  * Mask blending: still inactive — ``--mask`` reserved (paper Eq. 12 v3).
  * Inversion runs at ``--invert_guidance 1.0`` (no CFG); the edit pass uses
    the user's ``--guidance_scale`` (default 4.0, Anima base-v1.0 standard).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from torchvision import transforms
from typing import Optional

# Make ``anima_lora/`` importable when this script is invoked as
# ``python scripts/edit.py``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from library.anima import strategy as strategy_anima, text_strategies  # noqa: E402
from library.datasets.buckets import CONSTANT_TOKEN_BUCKETS  # noqa: E402
from library.inference import directedit, sampling as inference_utils  # noqa: E402
from library.inference.directedit_splice import splice_crossattn_emb  # noqa: E402
from library.inference.smc_cfg import SMCCFGState  # noqa: E402
from library.inference.edit_dispatcher import (  # noqa: E402
    derive_target_caption,
    encode_last_pooled_via_anima_strategy,
)
from library.inference.models import load_dit_model, load_text_encoder  # noqa: E402
from library.inference.output import save_images  # noqa: E402
from library.inference.text import MAX_CROSSATTN_TOKENS, prepare_text_inputs  # noqa: E402
from library.log import setup_logging  # noqa: E402
from library.models import qwen_vae as qwen_image_autoencoder_kl  # noqa: E402
from library.runtime.device import clean_memory_on_device  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DirectEdit image editing for Anima")

    # Model paths (mirror inference.py defaults so the make target passes the
    # same trio it always does).
    p.add_argument("--dit", required=True)
    p.add_argument("--text_encoder", required=True)
    p.add_argument("--vae", required=True)
    p.add_argument("--attn_mode", default="flash")

    # Editing inputs.
    p.add_argument("--image", required=True, help="Source image path")
    p.add_argument(
        "--prompt_src",
        default="",
        help="Source caption (for inversion). Typically Anima Tagger output "
        "for external images, or the recorded prompt for self-generated "
        "images. Ignored when --cached_embed is set.",
    )
    p.add_argument(
        "--prompt_tar",
        default="",
        help="Target caption (for the edit pass). Usually `prompt_src + edit`. "
        "Ignored when --cached_embed is set. When --edit_instruction is given "
        "and --prompt_tar is empty, the dispatcher derives this automatically.",
    )
    p.add_argument(
        "--edit_instruction",
        default="",
        help="Short tag-phrase edit (e.g. 'large breasts', '-hair ornament', "
        "'no hair ornament'). When set, the dispatcher derives --prompt_tar "
        "from --prompt_src + this instruction: explicit '-X' or 'no X' "
        "(matching an existing tag) does REMOVE; Qwen3 last-pool cosine + "
        "threshold gate fires REPLACE on confident matches; otherwise APPEND. "
        "Ignored when --prompt_tar is set explicitly or when --cached_embed "
        "is set.",
    )
    p.add_argument(
        "--replace_threshold",
        type=float,
        default=0.92,
        help="Dispatcher: top-1 cosine must exceed this to fire REPLACE. "
        "Tuned against scripts/probes/edit_nearest_tag.py.",
    )
    p.add_argument(
        "--replace_gap",
        type=float,
        default=0.04,
        help="Dispatcher: top1−top2 cosine gap must exceed this to fire "
        "REPLACE. Probe ambiguous cases (huge+large both present, medium-vs-"
        "grey hair near-tie) sit at gap < 0.01 and abstain into APPEND.",
    )
    p.add_argument(
        "--use_slot_surgery",
        action="store_true",
        help="Build embed_tar by transplanting only the T5-diff-span slots of "
        "ψ_tar's crossattn_emb into ψ_src's encoding. Off by default (uses "
        "the full ψ_tar encoding as today). Requires --prompt_src non-empty. "
        "Untouched slots come from ψ_src — see library/inference/"
        "directedit_splice.py for the invariant.",
    )
    p.add_argument(
        "--cached_embed",
        default=None,
        help="Sanity-check mode: load a preprocessed `_anima_te.safetensors` "
        "cache (the file `cache_text_embeddings.py` writes — same format the "
        "trainer consumes) and run one invert + edit pass per stored variant "
        "with ψ_tar == ψ_src. With `--caption_shuffle_variants N` caches, "
        "this sweeps v0..v{N-1} (pristine + tag-shuffled re-encodings); "
        "single-variant caches collapse to one pass. Skips the text encoder "
        "entirely. Mismatched reconstruction across variants flags numeric "
        "drift in invert/edit_forward.",
    )
    p.add_argument(
        "--cached_embed_variants",
        default="all",
        help="Which variants to run from the --cached_embed cache. "
        "'all' (default) sweeps every stored variant. Otherwise pass a "
        "comma-separated list of indices, e.g. '0' for the pristine caption "
        "only, '0,2' for v0 + v2. Out-of-range indices fail loud. "
        "Ignored unless --cached_embed is set.",
    )
    p.add_argument(
        "--negative_prompt",
        # default="",
        default="",
        help="Negative prompt for CFG on the edit pass (default empty). In "
        "--cached_embed mode, an empty value is auto-replaced with 'worst "
        "quality' so CFG can still fire (the TE is loaded briefly to encode "
        "just the neg, then dropped).",
    )
    p.add_argument(
        "--mask",
        default=None,
        help="Reserved — background-lock mask path (v2). Currently ignored.",
    )

    # Sampling knobs.
    p.add_argument("--infer_steps", type=int, default=28)
    p.add_argument("--flow_shift", type=float, default=1.0)
    p.add_argument(
        "--guidance_scale",
        type=float,
        default=4.0,
        help="CFG scale for the edit (target) pass.",
    )
    p.add_argument(
        "--invert_guidance",
        type=float,
        default=1.0,
        help="CFG scale during inversion. Default 1.0 (no CFG); raise only if "
        "you need the inverted noise to match a high-CFG generation seed.",
    )
    p.add_argument(
        "--smc_cfg",
        action="store_true",
        help="α-adaptive Sliding-Mode Control on the edit pass's CFG combine "
        "(library/inference/smc_cfg.py). Clamps small/noisy CFG-residual "
        "voxels while preserving large semantic moves; composes with t_inj "
        "V-injection (SMC operates on the post-injection v_cond_tar / v_neg "
        "residual). No-op on the inversion pass.",
    )
    p.add_argument(
        "--smc_cfg_lambda",
        type=float,
        default=5.0,
        help="SMC sliding-manifold slope λ. Defaults match inference.py.",
    )
    p.add_argument(
        "--smc_cfg_alpha",
        type=float,
        default=0.1,
        help="SMC adaptive gain α ∈ (0, 1]. Defaults match inference.py.",
    )
    p.add_argument(
        "--t_inj",
        type=int,
        default=2,
        help="Number of early editing steps to inject src self-attn V into "
        "the tar pass (paper Eq. 13). Default 0 = pure ΔZ-anchored edit. "
        "Typical paper setting: t_inj ≈ T/10..T/3 (e.g. 3..9 at T=28). "
        "Higher = stronger source-feature preservation.",
    )
    p.add_argument(
        "--t_inj_blocks",
        default="all_but_last",
        help="Which DiT blocks V-injection targets. Accepts 'all', "
        "'all_but_last' (default, SD3.5-style), or a comma/range string like "
        "'8-22' or '8,9,12,14-18'. Ignored when --t_inj 0.",
    )
    p.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        default=None,
        help="Override image size (H W). Default: snap to closest "
        "CONSTANT_TOKEN_BUCKETS bucket for the source aspect ratio.",
    )
    p.add_argument("--seed", type=int, default=42)

    # I/O.
    p.add_argument("--save_path", required=True)

    # Plumbing flags inference.py exposes that downstream code reads — keep
    # passthroughs so generation-side accessors don't trip.
    p.add_argument("--vae_chunk_size", type=int, default=64)
    p.add_argument("--vae_disable_cache", action="store_true", default=True)
    p.add_argument("--text_encoder_cpu", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--no_metadata", action="store_true")
    p.add_argument("--lora_weight", nargs="*", default=None)
    p.add_argument("--lora_multiplier", nargs="*", type=float, default=1.0)
    p.add_argument(
        "--compile_blocks",
        action="store_true",
        default=True,
        help="torch.compile each transformer block's _forward individually "
        "(per-block compile, not full-model). Speeds up the inversion + edit "
        "loops; first call per shape pays a compile cost.",
    )
    p.add_argument(
        "--compile_inductor_mode",
        default=None,
        help="Inductor preset passed through to torch.compile(mode=...). "
        "e.g. 'reduce-overhead' for per-block CUDAGraphs.",
    )

    args = p.parse_args()
    args.compile = False
    return args


def _pick_bucket(img: Image.Image) -> tuple[int, int]:
    """Return (H, W) from CONSTANT_TOKEN_BUCKETS closest to the source aspect."""
    rw, rh = img.size
    target = rw / rh
    best = min(CONSTANT_TOKEN_BUCKETS, key=lambda wh: abs(wh[0] / wh[1] - target))
    return best[1], best[0]  # bucket is (W, H); we return (H, W)


def _parse_t_inj_blocks(spec: str, n_blocks: int) -> list[int] | None:
    """Parse `--t_inj_blocks` into a list of block indices.

    'all' → every block (0..n-1). 'all_but_last' → 0..n-2 (default; matches
    paper's SD3.5 placement). Otherwise parses comma-separated entries that
    are either a single int or a closed range 'A-B'. Returns None for the
    'all_but_last' default so the directedit module's own default applies
    (and the log message stays consistent across callers).
    """
    spec = spec.strip().lower()
    if spec in ("", "all_but_last"):
        return None  # → directedit default
    if spec == "all":
        return list(range(n_blocks))
    out: list[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo_s, hi_s = chunk.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if lo > hi:
                raise ValueError(f"--t_inj_blocks range {chunk!r}: lo > hi")
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(chunk))
    if not out:
        raise ValueError(f"--t_inj_blocks={spec!r} parsed to empty set")
    bad = [i for i in out if i < 0 or i >= n_blocks]
    if bad:
        raise ValueError(
            f"--t_inj_blocks={spec!r}: indices {sorted(set(bad))} out of "
            f"range (model has {n_blocks} blocks; valid 0..{n_blocks - 1})"
        )
    return sorted(set(out))


def _parse_variant_selector(selector: str, n_available: int) -> list[int]:
    """Parse `--cached_embed_variants` into a list of variant indices.

    'all' yields [0..n_available-1]; comma-separated indices yield those.
    Out-of-range indices fail loud so a typo doesn't silently fall back to
    the full sweep.
    """
    if selector == "all":
        return list(range(n_available))
    try:
        wanted = [int(s.strip()) for s in selector.split(",") if s.strip()]
    except ValueError as e:
        raise ValueError(
            f"--cached_embed_variants={selector!r}: expected 'all' or a "
            "comma-separated list of integers"
        ) from e
    if not wanted:
        raise ValueError("--cached_embed_variants is empty")
    bad = [i for i in wanted if i < 0 or i >= n_available]
    if bad:
        raise ValueError(
            f"--cached_embed_variants={selector!r}: indices {bad} out of "
            f"range — cache has {n_available} variant(s) (0..{n_available - 1})"
        )
    return wanted


def _load_cached_embed_variants(
    cache_path: str,
    anima,
    device: torch.device,
    selector: str = "all",
) -> list[tuple[str, torch.Tensor]]:
    """Load preprocessed crossattn embeds from a `_anima_te.safetensors` cache.

    Returns a list of `(variant_label, crossattn_emb)` ready to feed
    DirectEdit. Mirrors `AnimaTextEncoderOutputsCachingStrategy.load_outputs_npz`
    but emits the variants requested by `selector` (default 'all') instead of
    stochastically sampling one — this is a sweep, not training.

    Behavior:
      * Multi-variant caches (`num_variants` key present): yields v_i for every
        i selected by `selector`.  v0 is the pristine caption; v1..v{N-1} are
        tag-shuffled re-encodings.
      * Single-variant caches: yields one pass.  `selector` must be 'all' or
        '0'.
      * Pre-baked `crossattn_emb*` (cached when training was preprocessed
        with `cache_llm_adapter_outputs=True`) is used directly. Otherwise
        we run `anima._preprocess_text_embeds` ourselves so the cache stays
        usable regardless of how it was preprocessed.

    Fails loud if the file is missing, shape-mismatched, or `selector` names a
    missing variant.
    """
    from safetensors import safe_open

    if not os.path.isfile(cache_path):
        raise FileNotFoundError(
            f"--cached_embed file not found: {cache_path}\n"
            "Run `make preprocess-te` (with --caption_shuffle_variants N to "
            "get a multi-variant cache) before running the dry test."
        )

    out: list[tuple[str, torch.Tensor]] = []
    with safe_open(cache_path, framework="pt") as f:
        keys = set(f.keys())
        has_variants = "num_variants" in keys
        if has_variants:
            n = int(f.get_tensor("num_variants"))
            wanted = _parse_variant_selector(selector, n)
            indices = [(f"v{i}", f"_v{i}") for i in wanted]
        else:
            # Single-variant cache: only v0 exists; reject anything else.
            _parse_variant_selector(selector, 1)
            indices = [("v0", "")]

        for label, suf in indices:
            crossattn_key = f"crossattn_emb{suf}"
            if crossattn_key in keys:
                crossattn_emb = f.get_tensor(crossattn_key).to(
                    device, dtype=torch.bfloat16
                )
                # Cache stores per-sample tensors unbatched, e.g. (512, 1024).
                # The DiT expects (B, N, D) — add the batch dim.
                if crossattn_emb.dim() == 2:
                    crossattn_emb = crossattn_emb.unsqueeze(0)
                # Pre-baked from training preprocess — already adapter-projected.
            else:
                # Run llm_adapter ourselves on the raw Qwen3 prompt_embeds.
                prompt_embeds = f.get_tensor(f"prompt_embeds{suf}").to(device)
                attn_mask = f.get_tensor(f"attn_mask{suf}").to(device)
                t5_input_ids = f.get_tensor(f"t5_input_ids{suf}").to(device)
                t5_attn_mask = f.get_tensor(f"t5_attn_mask{suf}").to(device)
                # Cached tensors are unbatched (shape [N, D] etc.); the
                # adapter expects a batch dim — add it for everything.
                if prompt_embeds.dim() == 2:
                    prompt_embeds = prompt_embeds.unsqueeze(0)
                if attn_mask.dim() == 1:
                    attn_mask = attn_mask.unsqueeze(0)
                if t5_input_ids.dim() == 1:
                    t5_input_ids = t5_input_ids.unsqueeze(0)
                if t5_attn_mask.dim() == 1:
                    t5_attn_mask = t5_attn_mask.unsqueeze(0)
                crossattn_emb, _ = anima._preprocess_text_embeds(
                    source_hidden_states=prompt_embeds,
                    target_input_ids=t5_input_ids,
                    target_attention_mask=t5_attn_mask,
                    source_attention_mask=attn_mask,
                )
                crossattn_emb[~t5_attn_mask.bool()] = 0
                crossattn_emb = crossattn_emb.to(torch.bfloat16)
            out.append((label, crossattn_emb))
    return out


def main() -> None:
    args = parse_args()

    if args.mask:
        logger.warning("--mask ignored: background-lock blending is v3.")
    if args.t_inj > 0 and args.compile_blocks:
        # The V-injection scope monkey-patches Attention.forward at runtime,
        # which would invalidate dynamo's cached graph for every block on the
        # first call. Recompile cost > the speedup compile would give us, so
        # turn it off for editing. (Inversion would still benefit, but the
        # compile state is per-process — flipping mid-run isn't worth the
        # complexity.)
        logger.info(
            "--t_inj %d > 0: disabling --compile_blocks for V-injection "
            "(monkey-patch breaks dynamo graph cache).",
            args.t_inj,
        )
        args.compile_blocks = False

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    args.device = device

    # 1. Load source image, pick bucket if --image_size unset.
    src_pil = Image.open(args.image).convert("RGB")
    if args.image_size is None:
        h_pix, w_pix = _pick_bucket(src_pil)
        args.image_size = [h_pix, w_pix]
        logger.info(
            "Image size auto-picked from source aspect %.3f -> %dx%d (HxW)",
            src_pil.size[0] / src_pil.size[1],
            h_pix,
            w_pix,
        )
    h_pix, w_pix = args.image_size
    src_pil = src_pil.resize((w_pix, h_pix), Image.LANCZOS)

    # 2. Tokenize strategies (matches inference.py main()).
    tokenize_strategy = strategy_anima.AnimaTokenizeStrategy(
        qwen3_path=args.text_encoder,
        t5_tokenizer_path=None,
        qwen3_max_length=MAX_CROSSATTN_TOKENS,
        t5_max_length=MAX_CROSSATTN_TOKENS,
    )
    text_strategies.TokenizeStrategy.set_strategy(tokenize_strategy)
    text_strategies.TextEncodingStrategy.set_strategy(
        strategy_anima.AnimaTextEncodingStrategy()
    )

    # 3. Load DiT first (needed by prepare_text_inputs's _preprocess_text_embeds).
    logger.info("Loading DiT model...")
    anima = load_dit_model(args, device, dit_weight_dtype=torch.bfloat16)
    if args.compile_blocks:
        # split_attn=True (the inference default) slices q/k/v per batch
        # element with `attn_params.seqlens[i]` as the slice end — a 0-dim
        # CUDA tensor whose use as a Python slice forces a `.item()` sync,
        # which breaks the dynamo graph inside every block._forward.
        # Batch=1 DirectEdit at a single bucket has no real per-element
        # trimming to do, so we disable split_attn for the compile path.
        # Self-attn now sees the ~0–1.6% image-token padding (zero keys
        # /values) instead of trimming it; cross-attn is unchanged from the
        # trained zero-padded-text invariant (see CLAUDE.md "Text encoder
        # padding").
        anima.split_attn = False
        anima.compile_blocks(mode=args.compile_inductor_mode)

    # 4. Encode source + target text — or, in --cached_embed mode, load
    #    preprocessed crossattn variants from the TE cache file (one pass per
    #    stored vN; ψ_tar == ψ_src so each pass reconstructs the source).
    cached_variants: list[tuple[str, torch.Tensor]] | None = None
    if args.cached_embed is not None:
        cached_variants = _load_cached_embed_variants(
            args.cached_embed, anima, device, args.cached_embed_variants
        )
        embed_src = embed_tar = None  # filled per-variant below

        # Cache file has no neg slot — encode one on the fly so CFG can fire.
        # Default to 'worst quality' when --negative_prompt is empty.
        neg_prompt = args.negative_prompt or ""
        if not args.negative_prompt:
            logger.info(
                "DirectEdit dry: --negative_prompt empty; defaulting to "
                "'' for CFG."
            )

        # Reuse prepare_text_inputs: set prompt == negative_prompt so the
        # positive forward hits the conds_cache and only one TE pass actually
        # runs. We discard the positive ctx and keep ctx_neg.
        args_neg = SimpleNamespace(**vars(args))
        args_neg.prompt = neg_prompt
        args_neg.negative_prompt = neg_prompt

        te_dtype = torch.bfloat16
        te_device = torch.device("cpu") if args.text_encoder_cpu else device
        text_encoder = load_text_encoder(args, dtype=te_dtype, device=te_device)
        shared = {"text_encoder": text_encoder, "conds_cache": {}}
        _, ctx_neg = prepare_text_inputs(args_neg, device, anima, shared)
        text_encoder.to("cpu")
        del text_encoder, shared
        clean_memory_on_device(device)

        embed_neg = ctx_neg["embed"][0].to(device, dtype=torch.bfloat16)
        logger.info(
            "DirectEdit dry: loaded %d variant(s) from %s; CFG enabled "
            "(neg=%r).",
            len(cached_variants),
            args.cached_embed,
            neg_prompt,
        )
    else:
        # Load TE first — the dispatcher (when --edit_instruction is set) needs
        # Qwen3 hidden states before we can build the args for prepare_text_inputs.
        logger.info("Loading text encoder...")
        te_dtype = torch.bfloat16
        te_device = torch.device("cpu") if args.text_encoder_cpu else device
        text_encoder = load_text_encoder(args, dtype=te_dtype, device=te_device)
        text_encoder.eval()

        # Dispatcher: derive ψ_tar from (ψ_src + edit_instruction) if requested
        # and --prompt_tar wasn't given. Explicit --prompt_tar always wins so
        # users can override the dispatcher's choice without removing the flag.
        if args.edit_instruction and not args.prompt_tar:
            tokenize_strategy = text_strategies.TokenizeStrategy.get_strategy()
            encoding_strategy = text_strategies.TextEncodingStrategy.get_strategy()
            # Dispatcher needs TE on-device; move there if --text_encoder_cpu
            # parked it on CPU, then restore.
            te_was_on = text_encoder.device
            text_encoder.to(device)
            encode_fn = lambda phrases: encode_last_pooled_via_anima_strategy(  # noqa: E731
                phrases, text_encoder, tokenize_strategy, encoding_strategy, device,
            )
            plan = derive_target_caption(
                args.prompt_src,
                args.edit_instruction,
                encode_last_pooled=encode_fn,
                replace_threshold=args.replace_threshold,
                replace_gap=args.replace_gap,
            )
            text_encoder.to(te_was_on)
            args.prompt_tar = plan.tar_caption
            logger.info(plan.log_line())
            logger.info("DirectEdit dispatcher: ψ_tar=%r", plan.tar_caption)
        elif args.use_slot_surgery and not args.prompt_tar:
            raise SystemExit(
                "--use_slot_surgery requires a ψ_tar source: pass --prompt_tar "
                "explicitly or --edit_instruction to derive it."
            )

        if args.use_slot_surgery and not args.prompt_src:
            raise SystemExit(
                "--use_slot_surgery requires a non-empty --prompt_src "
                "(surgery transplants from ψ_src's encoding)."
            )

        args_src = SimpleNamespace(**vars(args))
        args_src.prompt = args.prompt_src
        args_src.negative_prompt = args.negative_prompt

        args_tar = SimpleNamespace(**vars(args))
        args_tar.prompt = args.prompt_tar
        args_tar.negative_prompt = args.negative_prompt

        logger.info("Encoding prompts...")
        # Share the text-encoder instance across both prompt encodings.
        shared = {"text_encoder": text_encoder, "conds_cache": {}}

        ctx_src, ctx_neg = prepare_text_inputs(args_src, device, anima, shared)
        ctx_tar, _ = prepare_text_inputs(args_tar, device, anima, shared)

        embed_src = ctx_src["embed"][0].to(device, dtype=torch.bfloat16)
        embed_tar = ctx_tar["embed"][0].to(device, dtype=torch.bfloat16)
        embed_neg = ctx_neg["embed"][0].to(device, dtype=torch.bfloat16)

        if args.use_slot_surgery:
            # ctx["embed"] = [crossattn_emb_cpu, qwen3_attn_mask, t5_ids, t5_attn_mask].
            # T5 IDs were never moved off CPU (encode_tokens only moves qwen3
            # tensors), so a fresh `.tolist()` in splice_crossattn_emb is fine.
            t5_ids_src = ctx_src["embed"][2]
            t5_ids_tar = ctx_tar["embed"][2]
            tokenize_strategy = text_strategies.TokenizeStrategy.get_strategy()
            pad_id = tokenize_strategy.t5_tokenizer.pad_token_id
            embed_tar_full = embed_tar
            embed_tar, span = splice_crossattn_emb(
                crossattn_emb_src=embed_src,
                crossattn_emb_tar=embed_tar_full,
                t5_ids_src=t5_ids_src.to(device),
                t5_ids_tar=t5_ids_tar.to(device),
                pad_id=pad_id,
            )
            logger.info(
                "DirectEdit slot surgery: diff span src[%d:%d] -> tar[%d:%d] "
                "(src_len=%d tar_len=%d suffix_len=%d)",
                span.start, span.src_end, span.start, span.tar_end,
                span.src_len, span.tar_len, span.suffix_len,
            )

        # Drop TE; conds_cache hands us bare tensors and surgery is done.
        text_encoder.to("cpu")
        del text_encoder, shared
        clean_memory_on_device(device)

        with torch.no_grad():
            d_st = (embed_src.float() - embed_tar.float()).abs().mean().item()
            d_sn = (embed_src.float() - embed_neg.float()).abs().mean().item()
            d_tn = (embed_tar.float() - embed_neg.float()).abs().mean().item()
        logger.info(
            "DirectEdit embed diffs (abs mean): "
            "|src-tar|=%.6f  |src-neg|=%.6f  |tar-neg|=%.6f  "
            "(src.norm=%.3f tar.norm=%.3f shape=%s)",
            d_st, d_sn, d_tn,
            embed_src.float().norm().item(),
            embed_tar.float().norm().item(),
            tuple(embed_src.shape),
        )

    # 5. VAE-encode the source image -> clean latent (5D, frame=1).
    logger.info("Loading VAE for source encode...")
    vae = qwen_image_autoencoder_kl.load_vae(
        args.vae,
        device="cpu",
        disable_mmap=True,
        spatial_chunk_size=args.vae_chunk_size,
        disable_cache=args.vae_disable_cache,
    )
    vae.to(torch.bfloat16).eval().to(device)

    tfm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    )
    # 5D [B, C, T=1, H, W] — qwen_vae preserves input rank, and the DiT
    # expects 5D latents (it concats a per-frame padding mask along dim=1).
    img_t = (
        tfm(src_pil).unsqueeze(0).unsqueeze(2).to(device, dtype=torch.bfloat16)
    )  # [1, 3, 1, H, W] in [-1,1]

    with torch.no_grad():
        z_clean = vae.encode_pixels_to_latents(img_t)  # [1, C, 1, H/8, W/8]
    logger.info("Encoded source latent: %s", tuple(z_clean.shape))

    # Move VAE off-device for the DiT loop, bring it back for decode.
    vae.to("cpu")
    clean_memory_on_device(device)

    # 6. Sigma schedule. timesteps in generate_body are sigmas/1000-shifted,
    #    but invert/edit_forward consume sigmas directly.
    timesteps, sigmas = inference_utils.get_timesteps_sigmas(
        args.infer_steps, args.flow_shift, device
    )
    sigmas = sigmas.to(device)

    # 7. Build variant pass list. Real-text mode: one pass with the encoded
    #    src/tar pair. --cached_embed mode: one pass per stored variant in the
    #    cache file, each using ψ_tar == ψ_src for the reconstruction check.
    if cached_variants is not None:
        variant_passes = [(label, e, e) for label, e in cached_variants]
    else:
        variant_passes = [(None, embed_src, embed_tar)]

    # 8. Inversion -> editing per variant. Hold all z_edits before re-mounting
    #    the VAE so we only do one DiT-off / VAE-on swap.
    z_edits: list[tuple[Optional[str], torch.Tensor]] = []
    for variant, e_src, e_tar in variant_passes:
        tag = f"variant={variant}, " if variant else ""
        # Fresh SMC state per variant so e_prev resets cleanly between passes.
        # SMC is no-op on the inversion path (single-forward, no residual).
        smc_state = (
            SMCCFGState(lam=args.smc_cfg_lambda, alpha=args.smc_cfg_alpha)
            if args.smc_cfg
            else None
        )
        logger.info(
            "DirectEdit: %sinversion (T=%d, src_guidance=%.2f) -> edit "
            "(tar_guidance=%.2f, t_inj=%d, smc_cfg=%s)",
            tag,
            args.infer_steps,
            args.invert_guidance,
            args.guidance_scale,
            args.t_inj,
            (
                f"λ={args.smc_cfg_lambda},α={args.smc_cfg_alpha}"
                if args.smc_cfg
                else "off"
            ),
        )
        z_inv, delta_z = directedit.invert(
            anima=anima,
            z_clean=z_clean,
            embed_src=e_src,
            embed_neg=embed_neg if args.invert_guidance != 1.0 else None,
            sigmas=sigmas,
            guidance_scale=args.invert_guidance,
        )
        t_inj_blocks = (
            _parse_t_inj_blocks(args.t_inj_blocks, len(anima.blocks))
            if args.t_inj > 0
            else None
        )
        z_edit = directedit.edit_forward(
            anima=anima,
            z_init=z_inv[0],
            delta_z=delta_z,
            embed_tar=e_tar,
            embed_neg=embed_neg,
            sigmas=sigmas,
            guidance_scale=args.guidance_scale,
            embed_src=e_src if args.t_inj > 0 else None,
            t_inj=args.t_inj,
            t_inj_blocks=t_inj_blocks,
            z_inv=z_inv if args.t_inj > 0 else None,
            smc_cfg_state=smc_state,
        )
        z_edits.append((variant, z_edit))

    # 9. Decode + save (one VAE re-mount for all variants).
    del anima
    clean_memory_on_device(device)
    vae.to(device)
    os.makedirs(args.save_path, exist_ok=True)
    src_stem = Path(args.image).stem
    for variant, z_edit in z_edits:
        with torch.no_grad():
            pixels = vae.decode_to_pixels(z_edit.to(device, dtype=vae.dtype))
        if pixels.ndim == 5:
            pixels = pixels.squeeze(2)
        pixels = pixels[0].to("cpu", dtype=torch.float32)
        base = f"{src_stem}_{variant}" if variant else src_stem
        # save_images reads args.seed + args.save_path + args.no_metadata.
        saved = save_images(pixels, args, original_base_name=base)
        logger.info("DirectEdit done -> %s.png", saved)


if __name__ == "__main__":
    main()
