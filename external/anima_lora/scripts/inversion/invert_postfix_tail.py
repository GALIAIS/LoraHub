"""Per-image inversion of the orthogonal postfix tail (probe entry point).

See ``docs/proposal/postfix_residual_per_image_inversion.md``. This is a
measurement instrument, not a deployable adapter: it optimizes the K-dim
scale vector ``s`` per image so that

    ψ = concat(T5(tags), Q @ diag(s))

minimizes flow-matching loss against the frozen DiT. The output is the
K-vector ``s`` plus an optimization log. Downstream analysis (ceiling, PCA
spectrum, content clustering, lane-discipline, multi-seed functional
equivalence) consumes ``s`` directly.

Two modes:

* ``--image_dir`` (primary, batched): reads cached latents + cached T5
  prefixes from ``post_image_dataset/lora`` (or any dir with
  ``{stem}_*_anima.npz`` + ``{stem}_anima_te.safetensors`` pairs) and inverts
  ``--num_images`` of them.
* ``--image_stem``: single named image inside ``--image_dir``. Useful for
  one-off / debug.

Outputs per image, under ``--output_dir``:

* ``s/{stem}_s.safetensors`` — the (K,) trained vector, fp32.
* ``loss/{stem}.csv`` — per-step optimization log.

The SVD-of-cached-TE basis is computed once for ``(K, kind, seed)`` and
cached at ``--basis_path`` so repeat runs (different seeds / images) skip the
expensive SVD over the corpus.

Usage::

    uv run python scripts/inversion/invert_postfix_tail.py \\
        --dit models/diffusion_models/anima-base-v1.0.safetensors \\
        --image_dir post_image_dataset/lora \\
        --num_images 30 --shuffle --seed 0 \\
        --K 48 --basis svd_te \\
        --basis_path output/probes/postfix_tail/svd_basis_K48.pt \\
        --steps 100 --lr 0.01 --grad_accum 4 \\
        --lambda_zero 0.0 --sigma_min 0.0 \\
        --output_dir output/probes/postfix_tail/v0_first_run
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

ANIMA_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ANIMA_ROOT))

import torch  # noqa: E402

from library.anima import weights as anima_utils  # noqa: E402
from library.inference.postfix_inversion import (  # noqa: E402
    TailInversionConfig,
    invert_tail,
    load_cached_prefix,
    load_or_build_basis,
    save_tail_s,
)
from library.io.cache import (  # noqa: E402
    discover_cached_images,
    discover_cached_pairs,
    load_cached_latents,
)
from library.log import setup_logging  # noqa: E402

setup_logging()
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-image inversion of the orthogonal postfix tail "
        "(probe — see docs/proposal/postfix_residual_per_image_inversion.md)"
    )

    # Model
    p.add_argument("--dit", type=str, required=True, help="DiT checkpoint path")
    p.add_argument("--attn_mode", type=str, default="flash", help="Attention backend")
    p.add_argument(
        "--blocks_to_swap",
        type=int,
        default=16,
        help="Number of transformer blocks to swap to CPU (0 = none, "
        "<0 = gradient checkpointing instead)",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (default: cuda if available)",
    )
    p.add_argument(
        "--compile_blocks",
        action="store_true",
        default=True,
        help="torch.compile each transformer block's _forward (default on). "
        "Incompatible with block swap — silently skipped when "
        "--blocks_to_swap > 0.",
    )
    p.add_argument(
        "--no_compile_blocks",
        dest="compile_blocks",
        action="store_false",
        help="Disable torch.compile (run eager). Use for debugging or when "
        "compile time outweighs runtime gain.",
    )
    p.add_argument(
        "--compile_inductor_mode",
        type=str,
        default=None,
        help="Inductor preset passed through to torch.compile(mode=...). "
        "e.g. 'reduce-overhead' for per-block CUDAGraphs. None = inductor default.",
    )

    # Data
    p.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory with cached latents + cached T5 outputs "
        "(typically post_image_dataset/lora)",
    )
    p.add_argument(
        "--image_stem",
        type=str,
        default=None,
        help="Process only one image by stem (overrides --num_images/--shuffle)",
    )
    p.add_argument(
        "--num_images",
        type=int,
        default=30,
        help="Number of images to invert from --image_dir (default: 30, "
        "the proposal's N≈30–50 lower bound)",
    )
    p.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle image order before slicing the first --num_images",
    )
    p.add_argument(
        "--shuffle_seed",
        type=int,
        default=0,
        help="Seed for image-order shuffling (separate from --seed which "
        "drives per-image optimization)",
    )
    p.add_argument(
        "--basis_te_dir",
        type=str,
        default=None,
        help="Directory of cached *_anima_te.safetensors for SVD basis "
        "construction. Defaults to --image_dir.",
    )

    # Basis
    p.add_argument(
        "--K",
        type=int,
        default=48,
        help="Tail length (number of orthonormal slots). Default 48 matches "
        "the companion encoder proposal.",
    )
    p.add_argument(
        "--basis",
        type=str,
        default="svd_te",
        choices=["svd_te", "random"],
        help="Basis kind. 'svd_te' = top-K right singular vectors of cached "
        "T5 corpus; 'random' = QR of a Gaussian.",
    )
    p.add_argument(
        "--basis_path",
        type=str,
        default=None,
        help="Path to cache/load the basis. If file exists and shape matches, "
        "reuses it; otherwise computes and saves.",
    )
    p.add_argument(
        "--basis_seed",
        type=int,
        default=0,
        help="Seed for basis construction (row-shuffle for svd_te, RNG for random)",
    )
    p.add_argument(
        "--svd_num_files",
        type=int,
        default=256,
        help="Number of cached TE files sampled for SVD basis computation",
    )
    p.add_argument(
        "--embed_dim",
        type=int,
        default=1024,
        help="T5-compatible embedding dim (Qwen3 hidden size = 1024)",
    )

    # Optimization
    p.add_argument(
        "--steps", type=int, default=50, help="Optimization steps per image"
    )
    p.add_argument("--lr", type=float, default=0.01, help="Learning rate (AdamW)")
    p.add_argument(
        "--lr_schedule",
        type=str,
        default="cosine",
        choices=["cosine", "constant"],
    )
    p.add_argument(
        "--grad_accum", type=int, default=2, help="Gradient accumulation steps"
    )
    p.add_argument(
        "--timesteps_per_step",
        type=int,
        default=1,
        help="Extra σ samples per optimizer update — multiplies into grad_accum "
        "(total micro-iterations = grad_accum × timesteps_per_step). Each "
        "micro-iteration is a separate forward at batch=1, so raising this "
        "trades wall-time for variance reduction without growing VRAM.",
    )
    p.add_argument(
        "--sigma_sampling",
        type=str,
        default="sigmoid",
        choices=["uniform", "sigmoid"],
    )
    p.add_argument("--sigmoid_scale", type=float, default=1.0)
    p.add_argument(
        "--sigma_min",
        type=float,
        default=0.0,
        help="Lower bound for sampled sigmas (P-GRAFT-style low-σ skip — proposal "
        "calls out sigma_min ∈ {0, 0.1, 0.2} as a relevant sweep here)",
    )
    p.add_argument(
        "--sigma_max",
        type=float,
        default=0.25,
        help="Upper bound for sampled sigmas. Set < 1.0 to restrict supervision "
        "to low-σ steps (e.g. 0.25), where the FM target carries more per-image "
        "identity. Must be > --sigma_min.",
    )
    p.add_argument(
        "--lambda_zero",
        type=float,
        default=0.0,
        help="‖s‖² regularization weight. Proposal: primary run uses 0.0; "
        "auxiliary sweep at {0.001, 0.01, 0.1} measures lane-discipline cost.",
    )
    p.add_argument(
        "--init_std",
        type=float,
        default=0.0,
        help="Gaussian std for s init. 0.0 = zero-init (baseline). The "
        "archive script's 0.149 default is here as a documented ablation.",
    )
    p.add_argument("--seed", type=int, default=0, help="Per-image RNG seed")
    p.add_argument("--log_every", type=int, default=5)

    # Variance-reduced FM (AsymFlow §5.2 control variate, adapted per-image)
    p.add_argument(
        "--vr",
        dest="vr_enabled",
        action="store_true",
        help="Use VR-FM loss: per microstep, blend in a no-grad reference forward "
        "at s=0 on FEI-low-passed latents and supervise (y + λ·z)² with λ "
        "estimated online via EMA. (σ, noise, z) tuples are pre-sampled to a "
        "pool of --vr_pool_size to amortize the extra reference forwards.",
    )
    p.add_argument(
        "--vr_pool_size",
        type=int,
        default=32,
        help="VR pool size — # of (σ, noise, z) tuples precomputed per image",
    )
    p.add_argument(
        "--vr_lambda_beta",
        type=float,
        default=0.2,
        help="EMA β for λ. Training default is 0.01; per-image inversion bumps "
        "this to ~0.2 because the horizon is only ~50 microsteps per image.",
    )
    p.add_argument(
        "--vr_fei_sigma_low_div",
        type=float,
        default=4.0,
        help="FEI low-pass divisor (σ_low = min(H_lat, W_lat) / div). Matches "
        "FeRA's bench-validated 8.0 default — aspect-invariant on Anima.",
    )

    # Output
    p.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Probe run directory. Outputs: s/{stem}_s.safetensors, "
        "loss/{stem}.csv, manifest.json",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Reinvert images even if their s file already exists "
        "(default: skip cached)",
    )

    args = p.parse_args()
    if args.K < 1:
        p.error("--K must be >= 1")
    if args.K > args.embed_dim:
        p.error(f"--K ({args.K}) must be <= --embed_dim ({args.embed_dim})")
    return args


def _resolve_device(args) -> torch.device:
    if args.device:
        return torch.device(args.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _pick_images(args) -> list:
    """Discover cached image pairs in --image_dir; optionally filter / shuffle / slice."""
    images = discover_cached_images(args.image_dir)
    if not images:
        # Fall back to cache-only discovery (the post_image_dataset/lora layout
        # where the cache dir is decoupled from the source images).
        images = discover_cached_pairs(args.image_dir)
    if not images:
        raise FileNotFoundError(
            f"No cached latent+TE pairs found under {args.image_dir!r}. "
            "Run `make preprocess` first."
        )

    images = [im for im in images if im.te_path is not None]
    if not images:
        raise FileNotFoundError(
            f"No images under {args.image_dir!r} have cached "
            "_anima_te.safetensors prefixes (run `make preprocess-te`)"
        )

    if args.image_stem is not None:
        matches = [im for im in images if im.stem == args.image_stem]
        if not matches:
            raise ValueError(
                f"--image_stem {args.image_stem!r} not found in {args.image_dir}"
            )
        return matches

    if args.shuffle:
        rng = random.Random(args.shuffle_seed)
        images = list(images)
        rng.shuffle(images)

    if args.num_images is not None and args.num_images > 0:
        images = images[: args.num_images]
    return images


def _load_anima(args, device: torch.device):
    """Load DiT frozen on device, with the same swap/grad-ckpt switches as
    archive/inversion/invert_reference.py."""
    is_swapping = args.blocks_to_swap > 0
    grad_ckpt = args.blocks_to_swap < 0
    logger.info(f"Loading DiT: {args.dit}")
    anima = anima_utils.load_anima_model(
        device="cpu" if is_swapping else device,
        dit_path=args.dit,
        attn_mode=args.attn_mode,
        split_attn=True,
        loading_device="cpu" if is_swapping else device,
        dit_weight_dtype=torch.bfloat16,
    )
    anima.to(torch.bfloat16)
    anima.requires_grad_(False)
    # All steps share the same image, so no per-sample variance to exploit —
    # turn off split_attn to avoid data-dependent graph breaks.
    anima.split_attn = False

    if is_swapping:
        logger.info(f"Enabling block swap: {args.blocks_to_swap} blocks to CPU")
        anima.enable_block_swap(args.blocks_to_swap, device)
        anima.move_to_device_except_swap_blocks(device)
        anima.prepare_block_swap_before_forward()
        # block_swap moves weights CPU↔GPU mid-forward; incompatible with any
        # torch.compile mode — leave eager.
    else:
        anima.to(device)
        if grad_ckpt:
            logger.info("Enabling gradient checkpointing")
            anima.enable_gradient_checkpointing()
            for block in anima.blocks:  # type: ignore[union-attr]
                block.train()
        if args.compile_blocks:
            # Per-block compile against the 4096-token bucket. Pinning the token
            # count once means every aspect bucket in the run traces through the
            # same shape — no Dynamo recompile when we move from a 720×1440 image
            # to a 1024×1024 one. Compiles block._forward (not .forward) so the
            # unsloth_checkpoint @torch._disable_dynamo decorator doesn't blow
            # the trace under grad_ckpt — same contract as train.py's
            # compile_mode='blocks' path.
            anima.set_static_token_count(4096)
            anima.compile_blocks(
                backend="inductor", mode=args.compile_inductor_mode
            )
        else:
            logger.info("torch.compile disabled (--no_compile_blocks)")
    return anima


def main() -> None:
    args = parse_args()
    device = _resolve_device(args)
    logger.info(f"Device: {device}")

    images = _pick_images(args)
    logger.info(f"Inverting {len(images)} images from {args.image_dir}")

    out_root = Path(args.output_dir)
    s_dir = out_root / "s"
    loss_dir = out_root / "loss"
    s_dir.mkdir(parents=True, exist_ok=True)
    loss_dir.mkdir(parents=True, exist_ok=True)

    # Pre-build / cache the basis BEFORE the DiT load so its VRAM cost is
    # peaked first (SVD over a 256-file corpus is CPU-bound but allocates).
    basis_te_dir = args.basis_te_dir or args.image_dir
    Q = load_or_build_basis(
        K=args.K,
        D=args.embed_dim,
        kind=args.basis,
        te_cache_dir=basis_te_dir,
        basis_path=args.basis_path,
        svd_num_files=args.svd_num_files,
        seed=args.basis_seed,
    )

    anima = _load_anima(args, device)

    cfg = TailInversionConfig(
        K=args.K,
        steps=args.steps,
        lr=args.lr,
        lr_schedule=args.lr_schedule,
        grad_accum=args.grad_accum,
        timesteps_per_step=args.timesteps_per_step,
        sigma_sampling=args.sigma_sampling,
        sigmoid_scale=args.sigmoid_scale,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        lambda_zero=args.lambda_zero,
        init_std=args.init_std,
        log_every=args.log_every,
        vr_enabled=args.vr_enabled,
        vr_pool_size=args.vr_pool_size,
        vr_lambda_beta=args.vr_lambda_beta,
        vr_fei_sigma_low_div=args.vr_fei_sigma_low_div,
    )

    manifest = {
        "K": args.K,
        "embed_dim": args.embed_dim,
        "basis": args.basis,
        "basis_seed": args.basis_seed,
        "basis_path": args.basis_path,
        "svd_num_files": args.svd_num_files,
        "image_dir": args.image_dir,
        "num_images_requested": args.num_images,
        "shuffle": args.shuffle,
        "shuffle_seed": args.shuffle_seed,
        "seed": args.seed,
        "steps": args.steps,
        "lr": args.lr,
        "lr_schedule": args.lr_schedule,
        "grad_accum": args.grad_accum,
        "timesteps_per_step": args.timesteps_per_step,
        "sigma_sampling": args.sigma_sampling,
        "sigmoid_scale": args.sigmoid_scale,
        "sigma_min": args.sigma_min,
        "sigma_max": args.sigma_max,
        "lambda_zero": args.lambda_zero,
        "init_std": args.init_std,
        "vr_enabled": args.vr_enabled,
        "vr_pool_size": args.vr_pool_size,
        "vr_lambda_beta": args.vr_lambda_beta,
        "vr_fei_sigma_low_div": args.vr_fei_sigma_low_div,
        "dit": args.dit,
        "attn_mode": args.attn_mode,
        "compile_blocks": args.compile_blocks,
        "compile_inductor_mode": args.compile_inductor_mode,
        "results": [],
    }

    for i, img in enumerate(images):
        stem = img.stem
        s_path = s_dir / f"{stem}_s.safetensors"
        loss_path = loss_dir / f"{stem}.csv"

        if s_path.exists() and not args.overwrite:
            logger.info(f"[{i + 1}/{len(images)}] {stem} — already inverted, skipping")
            continue

        logger.info(f"[{i + 1}/{len(images)}] {stem}")

        lat, _, orig_h, orig_w = load_cached_latents(img.npz_path)
        latents = lat.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
        prefix_emb = load_cached_prefix(img.te_path, device=device)

        result = invert_tail(
            anima,
            prefix_emb=prefix_emb,
            latents=latents,
            basis_Q=Q,
            config=cfg,
            device=device,
            seed=args.seed,
            log_path=str(loss_path),
        )

        save_tail_s(
            str(s_path),
            result.s,
            K=args.K,
            D=args.embed_dim,
            basis_kind=args.basis,
            metadata={
                "ss_source_stem": stem,
                "ss_image_hw": f"{orig_h}x{orig_w}",
                "ss_best_loss": f"{result.best_loss:.6f}",
                "ss_best_fm_loss": f"{result.best_fm_loss:.6f}",
                "ss_best_step": str(result.best_step),
                "ss_final_s_l2": f"{result.final_s_l2:.6f}",
                "ss_steps": str(args.steps),
                "ss_lr": str(args.lr),
                "ss_lambda_zero": str(args.lambda_zero),
                "ss_init_std": str(args.init_std),
                "ss_sigma_min": str(args.sigma_min),
                "ss_sigma_max": str(args.sigma_max),
                "ss_seed": str(args.seed),
                "ss_basis_kind": args.basis,
            },
        )

        manifest["results"].append(
            {
                "stem": stem,
                "image_hw": [orig_h, orig_w],
                "best_loss": result.best_loss,
                "best_fm_loss": result.best_fm_loss,
                "best_step": result.best_step,
                "final_s_l2": result.final_s_l2,
                "final_lambda_ema": result.final_lambda_ema,
                "s_path": str(s_path.relative_to(out_root)),
                "loss_path": str(loss_path.relative_to(out_root)),
            }
        )
        logger.info(
            f"  saved {s_path.name} (best_loss={result.best_loss:.6f} "
            f"@ step {result.best_step}, fm={result.best_fm_loss:.6f}, "
            f"s‖₂={result.final_s_l2:.3f})"
        )

    manifest_path = out_root / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Manifest: {manifest_path}")
    logger.info("Done")


if __name__ == "__main__":
    main()
