"""Phase 2 — teacher-driven synthetic clean latents.

Walks each existing ``*_anima_te.safetensors`` in ``--cache_dir``, picks the
sibling latent NPZ's resolution, runs the frozen teacher (base DiT,
``skip_pooled_text_proj=True``) from fresh noise through full CFG denoising
(positive = cached crossattn_emb v0, negative = T5("") from the Phase 1
sidecar), saves the resulting clean latent under ``--synth_dir`` using the
same NPZ layout as ``preprocess/cache_latents.py``. The trainer can then point
at ``--synth_dir`` instead of (or alongside) the real-image cache to fit on
the teacher's own manifold, removing the real-vs-teacher distribution gap
that inflates the irreducible MSE floor.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file as _load_safetensors
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _load_uncond_for_synth(
    uncond_path: Path, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Load the Phase 1 sidecar as a ``(1, seq, 1024)`` tensor for CFG-negative."""
    sd = _load_safetensors(str(uncond_path))
    uncond = sd["crossattn_emb"]
    return uncond.to(device=device, dtype=dtype).unsqueeze(0).contiguous()


def denoise_one(
    model,
    crossattn_pos: torch.Tensor,
    crossattn_neg: torch.Tensor,
    *,
    H_lat: int,
    W_lat: int,
    num_steps: int,
    cfg_scale: float,
    flow_shift: float,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Run the frozen teacher through CFG denoising from fresh noise.

    Returns the clean latent ``(1, 16, H_lat, W_lat)`` in float32 on CPU.
    Mirrors the dense-path branch of ``library/inference/generation.py:529-706``
    minus all the extras (spectrum / dcw / mod-guidance / postfix / hydra) —
    the teacher here is the bare base DiT, so none of those apply.
    """
    from library.inference import sampling as inference_utils

    timesteps, sigmas = inference_utils.get_timesteps_sigmas(
        num_steps, flow_shift, device
    )
    timesteps = (timesteps / 1000.0).to(device, dtype=dtype)

    sampler = inference_utils.ERSDESampler(sigmas, seed=seed, device=device)

    gen = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(
        (1, 16, 1, H_lat, W_lat),
        dtype=dtype,
        device=device,
        generator=gen,
    )
    padding_mask = torch.zeros(1, 1, H_lat, W_lat, dtype=dtype, device=device)

    do_cfg = abs(cfg_scale - 1.0) > 1e-6

    for i, t in enumerate(timesteps):
        t_expand = t.expand(latents.shape[0])
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype):
            noise_pred = model.forward_mini_train_dit(
                latents,
                t_expand,
                crossattn_pos,
                padding_mask=padding_mask,
                skip_pooled_text_proj=True,
            )
            if do_cfg:
                uncond_noise_pred = model.forward_mini_train_dit(
                    latents,
                    t_expand,
                    crossattn_neg,
                    padding_mask=padding_mask,
                    skip_pooled_text_proj=True,
                )
                noise_pred = uncond_noise_pred + cfg_scale * (
                    noise_pred - uncond_noise_pred
                )

        denoised = latents.float() - sigmas[i] * noise_pred.float()
        latents = sampler.step(latents, denoised, i).to(latents.dtype)

    # latents: (1, 16, 1, H_lat, W_lat) → drop temporal dim
    return latents.float().squeeze(2).cpu()


def _filter_and_cap_pairs(
    pairs: list,
    buckets: list[tuple[int, int]] | None,
    n_per_bucket: int | None,
    shuffle_seed: int | None,
    get_latent_resolution,
) -> list:
    """Restrict pairs to a (H_pix, W_pix) allowlist and cap N per bucket.

    Mirrors DCW's ``pick_cached_samples`` stratification: shuffle-seeded
    selection across the bucket's full candidate pool so incremental re-runs
    grow coverage instead of resampling the same prompts.
    """
    import numpy as np

    if buckets is None and n_per_bucket is None:
        return pairs

    allowed_latent: set[tuple[int, int]] | None = (
        {(h // 8, w // 8) for h, w in buckets} if buckets is not None else None
    )
    by_bucket: dict[tuple[int, int], list] = {}
    for p in pairs:
        try:
            res = get_latent_resolution(p.npz_path)  # "HxW"
            H_lat, W_lat = (int(x) for x in res.split("x"))
        except Exception:
            continue
        key = (H_lat, W_lat)
        if allowed_latent is not None and key not in allowed_latent:
            continue
        by_bucket.setdefault(key, []).append(p)

    if shuffle_seed is not None:
        rng = np.random.default_rng(int(shuffle_seed))
        for items in by_bucket.values():
            rng.shuffle(items)

    out: list = []
    for key in sorted(by_bucket):
        items = by_bucket[key]
        if n_per_bucket is not None:
            items = items[: int(n_per_bucket)]
        out.extend(items)
        H_lat, W_lat = key
        logger.info(
            f"  bucket {H_lat * 8}x{W_lat * 8} (latent {H_lat}x{W_lat}): "
            f"{len(items)} pair(s)"
        )
    return out


def generate_synthetic_latents(
    cache_dir: Path,
    synth_dir: Path,
    *,
    dit_path: str,
    uncond_path: Path,
    attn_mode: str,
    num_steps: int,
    cfg_scale: float,
    flow_shift: float,
    seed: int,
    variant: int,
    max_samples: int | None,
    blocks_to_swap: int,
    overwrite: bool,
    buckets: list[tuple[int, int]] | None = None,
    n_per_bucket: int | None = None,
    shuffle_seed: int | None = None,
    compile_core: bool = True,
) -> None:
    """Phase 2 entry point. Iterates TE caches, runs teacher denoising, dumps NPZs."""
    from library.anima import weights as anima_utils
    from library.anima.models import Anima
    from library.io.cache import (
        discover_cached_pairs,
        get_latent_resolution,
        load_cached_text_features,
    )

    pairs = discover_cached_pairs(str(cache_dir))
    if not pairs:
        logger.warning(
            f"No (latent.npz, TE) pairs discovered in {cache_dir}. Run preprocess first."
        )
        return

    synth_dir.mkdir(parents=True, exist_ok=True)

    if buckets is not None or n_per_bucket is not None:
        bucket_str = (
            ", ".join(f"{h}x{w}" for h, w in buckets) if buckets else "(all)"
        )
        logger.info(
            f"Phase 2 bucket filter: buckets=[{bucket_str}], "
            f"n_per_bucket={n_per_bucket}, shuffle_seed={shuffle_seed}"
        )
        pairs = _filter_and_cap_pairs(
            pairs, buckets, n_per_bucket, shuffle_seed, get_latent_resolution
        )

    if max_samples is not None:
        pairs = pairs[: int(max_samples)]
    logger.info(
        f"Phase 2: synthesizing {len(pairs)} clean latents from teacher "
        f"(steps={num_steps}, cfg={cfg_scale}, flow_shift={flow_shift}, seed={seed})"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    logger.info(f"Loading base DiT (teacher) from {dit_path} ...")
    model: Anima = anima_utils.load_anima_model(
        device,
        dit_path,
        attn_mode=attn_mode,
        split_attn=False,
        loading_device="cpu" if blocks_to_swap > 0 else device,
        dit_weight_dtype=dtype,
    )
    if blocks_to_swap > 0:
        model.enable_block_swap(blocks_to_swap, device)
        model.move_to_device_except_swap_blocks(device)
    else:
        model.to(device)
    model.set_static_token_count(4096)
    model.eval()

    # Compile the constant-shape block stack — set_static_token_count(4096)
    # above is the precondition. Traces once; one CUDAGraph serves every
    # bucket in pairs, and both CFG branches reuse it.
    if compile_core and blocks_to_swap == 0:
        try:
            model.compile_core(mode="default")
        except Exception as e:
            logger.warning(f"compile_core failed ({e}); falling back to eager.")
    elif compile_core and blocks_to_swap > 0:
        logger.info("compile_core skipped: incompatible with blocks_to_swap>0.")

    crossattn_neg = _load_uncond_for_synth(uncond_path, device, dtype)

    pbar = tqdm(pairs, desc="synth latents")
    n_written = 0
    n_skipped = 0
    for sample_idx, pair in enumerate(pbar):
        # Resolution from sibling real-image latent NPZ — keeps synthetic
        # aspect distribution matched to the real dataset and guarantees
        # the constant-token-bucketing invariant is satisfied.
        try:
            res_str = get_latent_resolution(pair.npz_path)  # e.g. "64x64"
            H_lat, W_lat = (int(x) for x in res_str.split("x"))
        except Exception as e:
            logger.warning(f"  skip {pair.stem}: bad latent NPZ ({e})")
            continue

        out_path = synth_dir / f"{pair.stem}_{H_lat}x{W_lat}_anima.npz"
        if out_path.exists() and not overwrite:
            n_skipped += 1
            pbar.set_postfix_str(f"skip {pair.stem}")
            continue

        crossattn_pos, _pooled = load_cached_text_features(
            pair.te_path, variant=variant
        )
        if crossattn_pos is None:
            logger.warning(f"  skip {pair.stem}: no crossattn_emb in TE cache")
            continue
        crossattn_pos = crossattn_pos.to(device=device, dtype=dtype).unsqueeze(0)

        # Per-sample seed deterministic-but-varied across samples — same noise
        # for the same stem on re-runs makes the pool stable for ablations.
        per_seed = (int(seed) * 1_000_003 + sample_idx) & 0x7FFFFFFF

        clean = denoise_one(
            model,
            crossattn_pos,
            crossattn_neg,
            H_lat=H_lat,
            W_lat=W_lat,
            num_steps=num_steps,
            cfg_scale=cfg_scale,
            flow_shift=flow_shift,
            seed=per_seed,
            device=device,
            dtype=dtype,
        )  # (1, 16, H_lat, W_lat) float32 CPU

        # Original pixel size: H_lat*8, W_lat*8 (matches the VAE downsample).
        H_pix, W_pix = H_lat * 8, W_lat * 8
        key_suffix = f"_{H_lat}x{W_lat}"
        np.savez(
            out_path,
            **{
                f"latents{key_suffix}": clean.squeeze(0).numpy(),  # (16, H_lat, W_lat)
                f"original_size{key_suffix}": np.array([W_pix, H_pix]),
                f"crop_ltrb{key_suffix}": np.array([0, 0, W_pix, H_pix]),
            },
        )
        n_written += 1
        pbar.set_postfix_str(f"{pair.stem} {H_lat}x{W_lat}")

    pbar.close()
    logger.info(
        f"Phase 2 done: wrote {n_written}, skipped {n_skipped} (already cached). "
        f"Output → {synth_dir}"
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
