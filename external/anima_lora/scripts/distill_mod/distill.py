"""Modulation guidance distillation (Phase 1).

Trains ``pooled_text_proj`` to inject pooled text embedding into the AdaLN
modulation path. The entire DiT backbone is frozen; only the small projection
MLP (~8M params) receives gradients.

Distillation setup (Starodubcev et al., ICLR 2026, Section 5):
  - Teacher: normal forward with real crossattn_emb, pooled_text_proj disabled.
  - Student: forward with T5("") crossattn_emb (unconditional T5, matches Anima's
    own CFG-uncond path at inference time — ``library/inference/text.py:99-127``),
    but pooled_text_proj receives the real pooled text vector.
  - Loss: MSE(student_pred, teacher_pred).

The unconditional sidecar is staged by ``make distill-prep`` (writes
``<data_dir>/_anima_uncond_te.safetensors``).

This forces pooled_text_proj to encode text information through modulation,
complementing the cross-attention path.

Usage:
    python -m scripts.distill_mod.distill [--iterations 4000] [--lr 1e-4] [--batch_size 1]
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402
from tqdm import tqdm  # noqa: E402

from library.anima import weights as anima_utils  # noqa: E402
from library.anima.models import Anima  # noqa: E402
from library.datasets.distill import CachedDataset  # noqa: E402
from scripts.distill_mod.teacher_cache import (  # noqa: E402
    TeacherCache,
    ValTeacherCache,
    prefill_teacher_cache,
)
from scripts.distill_mod.uncond import (  # noqa: E402
    UNCOND_TE_FILENAME,
    load_uncond_crossattn,
    uncond_for_batch,
)
from scripts.distill_mod.validation import run_validation  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main():
    parser = argparse.ArgumentParser(description="Modulation guidance distillation")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="post_image_dataset/lora",
        help="Directory with cached latents and text encoder outputs",
    )
    parser.add_argument(
        "--uncond_te_path",
        type=str,
        default=None,
        help=(
            "Path to the T5(\"\") sidecar used as the student's unconditional "
            "cross-attention input. Defaults to "
            "``<data_dir>/_anima_uncond_te.safetensors`` (staged by "
            "``make distill-prep``)."
        ),
    )
    parser.add_argument(
        "--synth_data_dir",
        type=str,
        default=None,
        help=(
            "Optional dir of teacher-generated synthetic clean latents from "
            "``make distill-prep`` (Phase 2). When set, training reads latents "
            "from here (matched by stem + resolution) and TE caches from "
            "``--data_dir`` — paper-faithful setup that removes the real-vs-"
            "teacher distribution gap. Default: real-image latents only."
        ),
    )
    parser.add_argument(
        "--dit_path",
        type=str,
        default="models/diffusion_models/anima-base-v1.0.safetensors",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="output/ckpt/pooled_text_proj.safetensors",
        help="Where to save the trained projection weights",
    )
    parser.add_argument("--iterations", type=int, default=7500)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--blocks_to_swap",
        type=int,
        default=0,
        help="Number of transformer blocks to offload to CPU",
    )
    parser.add_argument(
        "--save_every", type=int, default=750, help="Save checkpoint every N iterations"
    )
    parser.add_argument(
        "--attn_mode",
        type=str,
        default="flash",
        help="Attention mode (torch, flash). flash4 not supported yet.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sigmoid_scale",
        type=float,
        default=1.0,
        help="Scale for sigmoid timestep sampling",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from a saved pooled_text_proj checkpoint",
    )
    parser.add_argument(
        "--grad_accum", type=int, default=1, help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--torch_compile",
        action="store_true",
        default=True,
        help="Compile block._forward with torch.compile",
    )
    parser.add_argument(
        "--no_compile",
        dest="torch_compile",
        action="store_false",
        help="Disable torch.compile",
    )
    parser.add_argument(
        "--compile_mode",
        type=str,
        choices=["blocks", "full"],
        default="blocks",
        help="'blocks': compile each block._forward (default). "
        "'full': compile the constant-shape _run_blocks stack (one CUDAGraph "
        "across buckets — requires --no_grad_ckpt and --blocks_to_swap 0).",
    )
    parser.add_argument(
        "--compile_inductor_mode",
        type=str,
        default="",
        help="Inductor preset, e.g. 'reduce-overhead' for CUDAGraphs",
    )
    parser.add_argument(
        "--grad_ckpt",
        action="store_true",
        default=True,
        help="Enable gradient checkpointing with CPU offload (default on)",
    )
    parser.add_argument(
        "--no_grad_ckpt",
        dest="grad_ckpt",
        action="store_false",
        help="Disable gradient checkpointing (faster, more VRAM)",
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=0.02,
        help="Warmup steps: int >= 1 for absolute steps, float < 1 for ratio of iterations",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Iterate entire DataLoader without training to test collation",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="output/logs/distill_mod",
        help="TensorBoard log directory. A timestamped subdir is created per run.",
    )
    parser.add_argument(
        "--no_log",
        action="store_true",
        help="Disable TensorBoard logging",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="Log scalars to TensorBoard every N optimizer steps",
    )
    parser.add_argument(
        "--sample_ratio",
        type=float,
        default=1.0,
        help="Fraction of (post-split) samples to keep per bucket. Mirrors the "
        "LoRA per-subset sample_ratio; useful with PRESET=debug/half/quarter/tenth "
        "for fast iteration on a small slice of the dataset.",
    )
    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.01,
        help="Fraction of dataset held out for validation (e.g. 0.05 for 5 percent)",
    )
    parser.add_argument(
        "--validation_seed",
        type=int,
        default=42,
        help="Seed for deterministic train/val split + validation noise",
    )
    parser.add_argument(
        "--validate_every_n_steps",
        type=int,
        default=750,
        help="Run validation every N optimizer steps (only if validation_split>0)",
    )
    parser.add_argument(
        "--validation_sigmas",
        type=float,
        nargs="+",
        default=[0.1, 0.4, 0.7],
        help="Fixed sigma values for validation loss (mirrors train.py default)",
    )
    parser.add_argument(
        "--max_validation_steps",
        type=int,
        default=None,
        help="Cap on validation batches per pass. None = use the entire val set.",
    )
    parser.add_argument(
        "--teacher_cache_K",
        type=int,
        default=6,
        help="Number of pre-sampled sigma bins for the teacher prediction cache. "
        "Each sample sees K distinct (sigma, noise) pairs over the run. "
        "Higher K = more diversity but slower cache fill / larger RAM.",
    )
    parser.add_argument(
        "--teacher_cache_seed",
        type=int,
        default=1234,
        help="Seed for the K-sigma grid and per-(sample, sigma) deterministic noise. "
        "Independent of --seed so cache contents are reproducible across training runs.",
    )
    parser.add_argument(
        "--no_teacher_cache",
        action="store_true",
        help="Disable teacher prediction caching (re-runs the teacher forward every step). "
        "Use to A/B against the cached path or to recover the original continuous-sigma sampler.",
    )
    parser.add_argument(
        "--prefill_teacher_cache",
        action="store_true",
        help="Eagerly run teacher predictions for every (sample, sigma_idx) before training. "
        "Adds ~K * N * t_teacher up front but eliminates teacher forwards during training.",
    )
    parser.add_argument(
        "--no_val_teacher_cache",
        action="store_true",
        help="Disable validation-time teacher prediction caching (re-runs the teacher "
        "forward on every val pass). Default is enabled — val is deterministic across "
        "calls, so the first pass fills a (batch_idx, sigma_idx) cache and every "
        "subsequent pass skips teacher forwards entirely.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # --- Dry run: test DataLoader collation without loading the model ---
    if args.dry_run:
        dataset = CachedDataset(
            args.data_dir,
            batch_size=args.batch_size,
            sample_ratio=args.sample_ratio,
            synth_data_dir=args.synth_data_dir,
        )

        def _collate_dry(batch):
            return (
                [b[0] for b in batch],
                torch.stack([b[1] for b in batch]),
                torch.stack([b[2] for b in batch]),
                torch.stack([b[3] for b in batch]),
            )

        dl = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
            collate_fn=_collate_dry,
        )
        total = len(dl)
        for i, (_idxs, lat, te, pooled) in enumerate(tqdm(dl, desc="dry-run")):
            if (i + 1) % 200 == 0:
                logger.info(
                    f"  batch {i + 1}/{total}  latents={lat.shape}  te={te.shape}  pooled={pooled.shape}"
                )
        logger.info(f"Dry run OK: {total} batches, no collation errors.")
        return

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # --- Load unconditional T5("") sidecar (staged by `make distill-prep`) ---
    uncond_te_path = args.uncond_te_path or os.path.join(
        args.data_dir, UNCOND_TE_FILENAME
    )
    uncond_te_1 = load_uncond_crossattn(uncond_te_path, device, dtype)
    logger.info(
        f"Loaded uncond crossattn from {uncond_te_path} "
        f"(shape={tuple(uncond_te_1.shape)})"
    )

    # --- Load model ---
    logger.info("Loading DiT model...")
    model: Anima = anima_utils.load_anima_model(
        device,
        args.dit_path,
        attn_mode=args.attn_mode,
        split_attn=False,
        loading_device="cpu" if args.blocks_to_swap > 0 else device,
        dit_weight_dtype=dtype,
    )

    # pooled_text_proj isn't in the pretrained checkpoint, so its params are
    # still meta tensors after load_state_dict(assign=True). Materialize on CPU
    # before any .to(device) calls.
    model.pooled_text_proj.to_empty(device="cpu")
    nn.init.kaiming_uniform_(model.pooled_text_proj[0].weight, a=math.sqrt(5))
    nn.init.zeros_(model.pooled_text_proj[0].bias)
    nn.init.zeros_(model.pooled_text_proj[-1].weight)
    nn.init.zeros_(model.pooled_text_proj[-1].bias)

    # Resume from checkpoint if provided
    if args.resume:
        logger.info(f"Resuming from {args.resume}")
        from safetensors.torch import load_file

        state = load_file(args.resume)
        model.pooled_text_proj.load_state_dict(state)

    # Enable block swap for VRAM efficiency (two forwards per step)
    if args.blocks_to_swap > 0:
        model.enable_block_swap(args.blocks_to_swap, device)
        model.move_to_device_except_swap_blocks(device)
        model.switch_block_swap_for_training()  # forward+backward block movement
    else:
        model.to(device)

    # Static token count: pad all spatial sequences to 4096 tokens so
    # torch.compile sees a single shape across all bucket resolutions.
    model.set_static_token_count(4096)

    # Compile individual block._forward for speedup.
    # unsloth_checkpoint wraps Block.forward with @torch._disable_dynamo,
    # so we compile _forward (the inner computation) not forward.
    # compile_mode='full' instead compiles the constant-shape _run_blocks stack
    # — single trace across all buckets, but incompatible with grad ckpt / block swap.
    if args.torch_compile:
        if args.compile_mode == "full":
            assert not args.grad_ckpt, (
                "compile_mode='full' is incompatible with gradient checkpointing — "
                "pass --no_grad_ckpt"
            )
            assert args.blocks_to_swap == 0, (
                "compile_mode='full' is incompatible with block swap — "
                "pass --blocks_to_swap 0"
            )
            model.compile_core(mode=args.compile_inductor_mode)
        else:
            model.compile_blocks(mode=args.compile_inductor_mode)

    # Gradient checkpointing with CPU offload: recompute block activations
    # during backward, offloading saved tensors to CPU between forward/backward.
    # Teacher runs under no_grad so only the student pass holds activations;
    # peak is ~12 GB without checkpointing, flat otherwise. Disable with
    # --no_grad_ckpt for speed when you have the VRAM headroom.
    # Note: must keep model in train() mode because Block.forward gates
    # checkpointing behind self.training.
    if args.grad_ckpt:
        model.enable_gradient_checkpointing(unsloth_offload=True)
        logger.info("Gradient checkpointing: enabled (unsloth CPU offload)")
    else:
        logger.info("Gradient checkpointing: disabled")
    model.train()

    # Freeze everything, then unfreeze pooled_text_proj
    for param in model.parameters():
        param.requires_grad_(False)
    for param in model.pooled_text_proj.parameters():
        param.requires_grad_(True)

    # Train pooled_text_proj in float32 for precision
    model.pooled_text_proj.to(dtype=torch.float32)

    trainable_params = sum(p.numel() for p in model.pooled_text_proj.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Trainable: {trainable_params:,} / {total_params:,} params "
        f"({trainable_params / total_params * 100:.4f}%)"
    )

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(
        model.pooled_text_proj.parameters(),
        lr=args.lr,
        fused=torch.cuda.is_available(),
    )

    # Warmup + cosine annealing
    warmup_steps = (
        int(args.warmup) if args.warmup >= 1 else int(args.warmup * args.iterations)
    )
    if warmup_steps > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-6 / args.lr, total_iters=warmup_steps
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.iterations - warmup_steps, eta_min=args.lr * 0.1
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.iterations, eta_min=args.lr * 0.1
        )

    # --- Dataset (train + optional val split) ---
    dataset = CachedDataset(
        args.data_dir,
        batch_size=args.batch_size,
        split="train",
        validation_split=args.validation_split,
        validation_seed=args.validation_seed,
        sample_ratio=args.sample_ratio,
        synth_data_dir=args.synth_data_dir,
    )

    val_dataset = None
    val_dataloader = None
    if args.validation_split > 0.0:
        val_dataset = CachedDataset(
            args.data_dir,
            batch_size=args.batch_size,
            split="val",
            validation_split=args.validation_split,
            validation_seed=args.validation_seed,
            sample_ratio=args.sample_ratio,
            synth_data_dir=args.synth_data_dir,
        )

    # Custom collate to bypass collate_tensor_fn's _new_shared_filename_cpu
    # which creates non-resizable storage on some PyTorch/Python 3.13 builds.
    def _collate(batch):
        return (
            [b[0] for b in batch],
            torch.stack([b[1] for b in batch]),
            torch.stack([b[2] for b in batch]),
            torch.stack([b[3] for b in batch]),
        )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,  # dataset is pre-bucketed; shuffling would mix resolutions
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        collate_fn=_collate,
    )

    if val_dataset is not None and len(val_dataset) > 0:
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
            drop_last=True,
            collate_fn=_collate,
        )
    elif args.validation_split > 0.0:
        logger.warning(
            "validation_split>0 but val set is empty after bucket-remainder drop; "
            "skipping validation. Lower batch_size or raise validation_split."
        )

    # --- Teacher prediction cache (item 1: caches teacher forward results
    # keyed by (sample_idx, sigma_idx) so subsequent visits skip the teacher
    # forward entirely; sigmas are pre-sampled from the same sigmoid(scale * N(0,1))
    # distribution as the original sampler, noise is deterministic per pair) ---
    teacher_cache = None
    if not args.no_teacher_cache:
        teacher_cache = TeacherCache(
            K=args.teacher_cache_K,
            sigmoid_scale=args.sigmoid_scale,
            base_seed=args.teacher_cache_seed,
        )
        # Per-entry size from the first sample's latent shape (16 ch * H * W * bf16).
        _peek = dataset[0][1]
        bytes_per_entry = _peek.numel() * 2
        approx_gb = len(dataset) * args.teacher_cache_K * bytes_per_entry / 1e9
        logger.info(
            f"Teacher cache enabled: K={args.teacher_cache_K} sigmas, "
            f"{len(dataset)} samples → up to {len(dataset) * args.teacher_cache_K} entries, "
            f"~{approx_gb:.2f} GB RAM at full fill (bf16)."
        )
        if args.prefill_teacher_cache:
            prefill_teacher_cache(teacher_cache, dataset, model, device, dtype)

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)

    # --- TensorBoard ---
    writer = None
    if not args.no_log:
        from datetime import datetime

        run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_log_dir = os.path.join(args.log_dir, run_name)
        os.makedirs(run_log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=run_log_dir)
        writer.add_text("config", "  \n".join(f"{k}: {v}" for k, v in vars(args).items()))
        logger.info(f"TensorBoard logs -> {run_log_dir}")

    # --- Training loop ---
    grad_accum = args.grad_accum
    logger.info(
        f"Starting distillation: {args.iterations} iterations, "
        f"grad_accum={grad_accum} (effective batch={args.batch_size * grad_accum})"
    )

    data_iter = iter(dataloader)
    running_loss = 0.0
    log_interval = 50

    val_enabled = val_dataloader is not None and args.validate_every_n_steps > 0
    best_val_loss = float("inf")
    val_teacher_cache = (
        ValTeacherCache() if val_enabled and not args.no_val_teacher_cache else None
    )

    progress = tqdm(range(args.iterations), desc="distill")
    accum_loss_t = torch.zeros((), device=device)
    for step in progress:
        accum_loss_t.zero_()

        for accum_step in range(grad_accum):
            # Get batch (infinite cycling)
            try:
                idx_list, latents, crossattn_emb, pooled_text = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                idx_list, latents, crossattn_emb, pooled_text = next(data_iter)

            # latents: (B, 16, H, W), crossattn_emb: (B, seq, 1024), pooled_text: (B, 1024)
            latents = latents.to(device, dtype=dtype, non_blocking=True)
            crossattn_emb = crossattn_emb.to(device, dtype=dtype, non_blocking=True)
            pooled_text = pooled_text.to(device, dtype=dtype, non_blocking=True)

            B = latents.shape[0]

            # Sigma + noise: with teacher cache, draw from the K-grid and use
            # deterministic noise per (sample_idx, sigma_idx) so cache hits and
            # misses produce identical (latents, noise, sigma) inputs to the
            # student. Without the cache, fall back to the original
            # continuous-sigmoid sampler + fresh noise per step.
            if teacher_cache is not None:
                sigma_idx_list = teacher_cache.sample_sigma_idx(B)
                sigmas = torch.tensor(
                    [teacher_cache.get_sigma(si) for si in sigma_idx_list],
                    device=device,
                    dtype=latents.dtype,
                )
                noise_parts = [
                    teacher_cache.make_noise(
                        idx_list[i],
                        sigma_idx_list[i],
                        (1,) + tuple(latents.shape[1:]),
                        device,
                        latents.dtype,
                    )
                    for i in range(B)
                ]
                noise = torch.cat(noise_parts, dim=0)
            else:
                sigma_idx_list = None
                noise = torch.randn_like(latents)
                sigmas = torch.sigmoid(
                    args.sigmoid_scale * torch.randn(B, device=device)
                )

            timesteps = sigmas  # [0, 1] range (model expects this)

            # Noisy input: (1-σ) * latents + σ * noise
            sigmas_expand = sigmas.view(B, 1, 1, 1)
            noisy_input = (1.0 - sigmas_expand) * latents + sigmas_expand * noise

            # Add temporal dim: (B, 16, H, W) -> (B, 16, 1, H, W)
            noisy_input = noisy_input.unsqueeze(2)

            # Padding mask (all zeros = no padding)
            padding_mask = torch.zeros(
                B, 1, latents.shape[-2], latents.shape[-1], dtype=dtype, device=device
            )

            # --- Teacher forward: real crossattn, pooled_text_proj skipped ---
            # (skipped entirely on a full-batch cache hit).
            cached_list = None
            if teacher_cache is not None:
                cached_list = [
                    teacher_cache.get(idx_list[i], sigma_idx_list[i])
                    for i in range(B)
                ]
                all_hit = all(c is not None for c in cached_list)
            else:
                all_hit = False

            if all_hit:
                teacher_pred = torch.cat(
                    [c.to(device, dtype=dtype) for c in cached_list], dim=0
                )
            else:
                if model.blocks_to_swap:
                    model.prepare_block_swap_before_forward()
                # Fresh CUDA-graph epoch so teacher_pred outlives the student call.
                torch.compiler.cudagraph_mark_step_begin()
                with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
                    teacher_pred = model.forward_mini_train_dit(
                        noisy_input,
                        timesteps,
                        crossattn_emb,
                        padding_mask=padding_mask,
                        skip_pooled_text_proj=True,
                    )
                # Detach from the static buffer in case a future caller adds
                # a third compiled-fn invocation in the same step.
                teacher_pred = teacher_pred.clone()
                if teacher_cache is not None:
                    for i in range(B):
                        if cached_list[i] is None:
                            teacher_cache.put(
                                idx_list[i], sigma_idx_list[i], teacher_pred[i : i + 1]
                            )

            # --- Student forward: T5("") crossattn, real pooled text through proj ---
            # requires_grad_ needed for gradient checkpointing
            noisy_input = noisy_input.requires_grad_()
            if model.blocks_to_swap:
                model.prepare_block_swap_before_forward()
            uncond_crossattn = uncond_for_batch(uncond_te_1, crossattn_emb)
            torch.compiler.cudagraph_mark_step_begin()
            with torch.autocast("cuda", dtype=dtype):
                student_pred = model.forward_mini_train_dit(
                    noisy_input,
                    timesteps,
                    uncond_crossattn,
                    padding_mask=padding_mask,
                    pooled_text_override=pooled_text,
                )

            # --- MSE loss (scaled for accumulation) ---
            loss = nn.functional.mse_loss(student_pred.float(), teacher_pred.float())
            loss = loss / grad_accum
            loss.backward()
            accum_loss_t += loss.detach()

        # Grad-norm snapshot before stepping (cheap; ~8M params)
        grad_norm = None
        if writer is not None and (step + 1) % args.log_interval == 0:
            sq = 0.0
            for p in model.pooled_text_proj.parameters():
                if p.grad is not None:
                    sq += p.grad.detach().float().pow(2).sum().item()
            grad_norm = sq**0.5

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        accum_loss = accum_loss_t.item()
        running_loss += accum_loss
        lr = scheduler.get_last_lr()[0]

        if writer is not None and (step + 1) % args.log_interval == 0:
            writer.add_scalar("train/loss", accum_loss, step + 1)
            writer.add_scalar("train/lr", lr, step + 1)
            if grad_norm is not None:
                writer.add_scalar("train/grad_norm", grad_norm, step + 1)
            if teacher_cache is not None:
                tc_total = teacher_cache.hits + teacher_cache.misses
                hit_rate = teacher_cache.hits / tc_total if tc_total else 0.0
                writer.add_scalar("teacher_cache/hit_rate", hit_rate, step + 1)
                writer.add_scalar("teacher_cache/size", len(teacher_cache), step + 1)

        if (step + 1) % log_interval == 0:
            avg = running_loss / log_interval
            progress.set_postfix(loss=f"{avg:.6f}", lr=f"{lr:.2e}")
            if writer is not None:
                writer.add_scalar("train/loss_avg50", avg, step + 1)
            running_loss = 0.0
        else:
            progress.set_postfix(loss=f"{accum_loss:.6f}", lr=f"{lr:.2e}")

        # --- Validation pass ---
        do_validate = (
            val_dataloader is not None
            and args.validate_every_n_steps > 0
            and (
                (step + 1) % args.validate_every_n_steps == 0
                or (step + 1) == args.iterations
            )
        )
        improved = False
        overall_mean = None
        if do_validate:
            per_sigma_mean, overall_mean = run_validation(
                model,
                val_dataloader,
                device=device,
                dtype=dtype,
                sigmas=args.validation_sigmas,
                max_steps=args.max_validation_steps,
                seed=args.validation_seed,
                uncond_te_1=uncond_te_1,
                teacher_cache=val_teacher_cache,
            )
            sigma_str = ", ".join(
                f"σ={s:.2f}:{v:.4e}" for s, v in per_sigma_mean.items()
            )
            logger.info(
                f"[val @ step {step + 1}] mean={overall_mean:.6f}  {sigma_str}"
            )
            if writer is not None:
                writer.add_scalar("val/loss", overall_mean, step + 1)
                for s, v in per_sigma_mean.items():
                    writer.add_scalar(f"val/loss_sigma_{s:.2f}", v, step + 1)
                if val_teacher_cache is not None:
                    vc_total = val_teacher_cache.hits + val_teacher_cache.misses
                    vc_hit_rate = (
                        val_teacher_cache.hits / vc_total if vc_total else 0.0
                    )
                    writer.add_scalar(
                        "val_teacher_cache/hit_rate", vc_hit_rate, step + 1
                    )
                    writer.add_scalar(
                        "val_teacher_cache/size", len(val_teacher_cache), step + 1
                    )
            if overall_mean < best_val_loss:
                best_val_loss = overall_mean
                improved = True

        # Save checkpoint: when validation is enabled, only overwrite on
        # val-loss improvement. Otherwise fall back to step-cadence saves.
        if val_enabled:
            should_save = improved
        else:
            should_save = (
                (step + 1) % args.save_every == 0 or (step + 1) == args.iterations
            )
        if should_save:
            save_path = args.output_path
            state = {
                k: v.to(torch.bfloat16)
                for k, v in model.pooled_text_proj.state_dict().items()
            }
            save_file(state, save_path)
            if val_enabled:
                logger.info(
                    f"Saved checkpoint at step {step + 1} "
                    f"(val={overall_mean:.6f}, new best) -> {save_path}"
                )
            else:
                logger.info(f"Saved checkpoint at step {step + 1} -> {save_path}")
        elif do_validate:
            logger.info(
                f"Skipped save at step {step + 1}: "
                f"val={overall_mean:.6f} >= best={best_val_loss:.6f}"
            )

    if teacher_cache is not None:
        tc_total = teacher_cache.hits + teacher_cache.misses
        hit_rate = (teacher_cache.hits / tc_total * 100) if tc_total else 0.0
        logger.info(
            f"Teacher cache final: {len(teacher_cache)} entries, "
            f"{teacher_cache.hits} hits / {teacher_cache.misses} misses "
            f"({hit_rate:.1f}% hit rate)"
        )

    if val_teacher_cache is not None:
        vc_total = val_teacher_cache.hits + val_teacher_cache.misses
        vc_hit_rate = (val_teacher_cache.hits / vc_total * 100) if vc_total else 0.0
        logger.info(
            f"Val teacher cache final: {len(val_teacher_cache)} entries, "
            f"{val_teacher_cache.hits} hits / {val_teacher_cache.misses} misses "
            f"({vc_hit_rate:.1f}% hit rate)"
        )

    if writer is not None:
        writer.close()
    logger.info("Distillation complete.")


if __name__ == "__main__":
    main()
