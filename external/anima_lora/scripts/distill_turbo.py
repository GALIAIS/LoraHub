"""Turbo Anima — Decoupled DMD2 distillation.

Trains a 4-step LoRA student against the 28-step CFG=4 Anima teacher, using
Liu et al.'s Decoupled-Hybrid schedule (arXiv:2511.22677, Table 1 row 4) on
top of a co-LoRA fake score model.

Proposal: ``docs/proposal/turbo_anima_dmd_lora.md``.
Config:   ``configs/methods/turbo.toml`` (CLI flags override TOML values).

One frozen DiT serves three roles via per-network ``set_enabled`` toggling:

    teacher view  — both LoRA stacks off (base velocity)
    student view  — student on, fake off (v_student for x_pred)
    fake view     — student off, fake on (s_fake_cond_dm)

Per training step (single-call DMD2 — no inference sampler unroll at train
time, gradient is one ODE step from the sampled generator-t):

    1.  v_student = student(x_t, t, c)        # grad to student params
        x_pred    = x_t - t · v_student       # endpoint estimate

    2.  CA branch (τ_CA > t)                  # paper's CFG-bake engine
        v_real_cond_ca   = teacher(x_τ_ca, τ_CA, c)        # no_grad
        v_real_uncond_ca = teacher(x_τ_ca, τ_CA, c_null)   # no_grad
        Δ_cfg = v_real_cond_ca - v_real_uncond_ca

    3.  DM branch (τ_DM ∈ [0, 1])             # regularizer
        v_real_cond_dm = teacher(x_τ_dm, τ_DM, c)          # no_grad
        v_fake_cond_dm = fake   (x_τ_dm, τ_DM, c)          # no_grad
        Δ_dm = v_real_cond_dm - v_fake_cond_dm

    4.  α_eff ramps 1.0 → α over alpha_warmup_steps         # CA warmup
        grad_signal = Δ_dm + (α_eff - 1) · Δ_cfg
        loss_student = (-grad_signal · x_pred).mean()
        loss_student.backward()  → student.step()

    5.  Fake update — flow-matching loss on student's x_pred distribution:
        τ_fake ~ U[0,1]
        x_t_fake = (1-τ_fake)·x_pred.detach() + τ_fake·ε_fake
        v_fake   = fake(x_t_fake, τ_fake, c)                # grad to fake params
        target   = ε_fake - x_pred.detach()                 # flow-matching target
        fake_loss = MSE(v_fake, target)  → fake.step()

Output: ``output/ckpt/anima_turbo.safetensors`` — a normal plain-LoRA file
loadable by the standard inference path at ``--infer_steps 4 --cfg 1.0``.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from library.anima import weights as anima_utils
from library.anima.models import Anima
from networks.methods.turbo_dmd import TurboDMDNetwork
from library.datasets.distill import CachedDataset

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Python 3.11+; fall back to `tomli` if needed.
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


def load_turbo_config(path: str) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def _flatten(cfg: dict, key_path: str, default):
    """Look up ``a.b.c`` in a nested TOML dict, falling back to ``default``."""
    node = cfg
    for part in key_path.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


# ---------------------------------------------------------------------------
# Re-noising primitive
# ---------------------------------------------------------------------------


def renoise(x_pred: torch.Tensor, tau: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
    """``x_τ = (1 - τ)·x_pred + τ·ε`` — flow-matching forward path at level τ.

    ``tau`` is per-batch; broadcast to ``x_pred``'s shape.
    """
    tau_e = tau.view(-1, *([1] * (x_pred.dim() - 1)))
    return (1.0 - tau_e) * x_pred + tau_e * eps


def sample_t_above(t: torch.Tensor, min_gap: float = 0.05) -> torch.Tensor:
    """Sample τ ~ U(t + min_gap, 1.0) per batch element.

    Clamps the lower bound so very-late steps (t ≈ 1) don't collapse to a
    near-empty interval (proposal R5).
    """
    lower = (t + min_gap).clamp(max=1.0 - 1e-4)
    u = torch.rand_like(t)
    return lower + u * (1.0 - lower)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Turbo Anima — Decoupled DMD2 distillation")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/methods/turbo.toml",
        help="Path to the turbo TOML config (CLI flags override TOML values).",
    )
    # CLI overrides — every TOML key has a matching flag. Default sentinels
    # (None / -1.0) mean "use the TOML value".
    parser.add_argument("--dit_path", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--output_name", type=str, default=None)
    parser.add_argument("--iterations", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--student_rank", type=int, default=-1)
    parser.add_argument("--fake_rank", type=int, default=-1)
    parser.add_argument(
        "--use_custom_down_autograd",
        action="store_true",
        default=None,
        help="Memory-saving down-projection autograd (skips fp32 input save). "
        "Default: read from TOML (top-level scalar), else off.",
    )
    parser.add_argument(
        "--no_use_custom_down_autograd",
        dest="use_custom_down_autograd",
        action="store_false",
    )
    parser.add_argument("--student_lr", type=float, default=-1.0)
    parser.add_argument("--fake_lr", type=float, default=-1.0)
    parser.add_argument(
        "--fake_steps_per_student_step",
        type=int,
        default=-1,
        help="Number of fake (DM regularizer) updates per student step. "
        "Standard DMD2 practice keeps the fake ahead of the moving x_pred "
        "distribution; >1 gives the fake extra SGD iterations on resampled "
        "(τ, ε) noise against the same x_pred.detach(). Default: TOML "
        "(optim.fake_steps_per_student_step, default 1).",
    )
    parser.add_argument("--alpha", type=float, default=-1.0, help="DMD CFG-bake α (overrides dmd.teacher_cfg)")
    parser.add_argument("--alpha_warmup_steps", type=int, default=-1)
    parser.add_argument("--student_steps", type=int, default=-1, help="Sampler step count baked into the student")
    parser.add_argument("--blocks_to_swap", type=int, default=0)
    parser.add_argument("--attn_mode", type=str, default="flash")
    parser.add_argument("--grad_ckpt", action="store_true", default=False)
    parser.add_argument("--no_grad_ckpt", dest="grad_ckpt", action="store_false")
    parser.add_argument(
        "--torch_compile",
        action="store_true",
        default=True,
        help="Compile block._forward. Off by default — multiple forwards per step "
        "are not yet validated under cudagraphs; turn on once Phase 0 is green.",
    )
    parser.add_argument("--save_every", type=int, default=-1)
    parser.add_argument("--log_interval", type=int, default=-1)
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--no_log", action="store_true")
    parser.add_argument(
        "--single_prompt_idx",
        type=int,
        default=None,
        help="Phase 0 overfit mode — pin the dataloader to a single (latent, text) pair.",
    )
    parser.add_argument("--sample_ratio", type=float, default=1.0)
    args = parser.parse_args()

    cfg = load_turbo_config(args.config)

    # Resolve every knob: CLI override (non-sentinel) wins, else TOML, else default.
    def pick(cli_val, toml_key, default):
        if cli_val is not None and cli_val != -1 and cli_val != -1.0:
            return cli_val
        return _flatten(cfg, toml_key, default)

    dit_path = pick(args.dit_path, "dit_path", "models/diffusion_models/anima-base-v1.0.safetensors")
    data_dir = pick(args.data_dir, "data_dir", "post_image_dataset/lora")
    output_dir = pick(args.output_dir, "output_dir", "output/ckpt")
    output_name = pick(args.output_name, "output_name", "anima_turbo")
    iterations = int(pick(args.iterations, "iterations", 20000))
    batch_size = int(pick(args.batch_size, "batch_size", 1))
    seed = int(pick(args.seed, "seed", 42))

    student_rank = int(pick(args.student_rank, "network.student_rank", 48))
    fake_rank = int(pick(args.fake_rank, "network.fake_rank", 48))
    student_alpha = float(_flatten(cfg, "network.student_alpha", student_rank))
    fake_alpha = float(_flatten(cfg, "network.fake_alpha", fake_rank))
    attn_mode = pick(args.attn_mode, "network.attn_mode", "flash")
    # use_custom_down_autograd lives at TOML top level (matches the LoRA family's
    # config layout in methods/lora.toml). CLI flag wins when set explicitly.
    if args.use_custom_down_autograd is None:
        use_custom_down_autograd = bool(_flatten(cfg, "use_custom_down_autograd", False))
    else:
        use_custom_down_autograd = bool(args.use_custom_down_autograd)

    student_steps = int(pick(args.student_steps, "dmd.student_steps", 4))
    teacher_cfg = float(pick(args.alpha, "dmd.teacher_cfg", 4.0))
    tau_ca_strategy = _flatten(cfg, "dmd.tau_ca_strategy", "above_t")
    tau_dm_strategy = _flatten(cfg, "dmd.tau_dm_strategy", "uniform")
    tau_ca_min_gap = float(_flatten(cfg, "dmd.tau_ca_min_gap", 0.05))
    tau_ca_skip_above_t = float(_flatten(cfg, "dmd.tau_ca_skip_above_t", 0.95))

    student_lr = float(pick(args.student_lr, "optim.student_lr", 1e-5))
    fake_lr = float(pick(args.fake_lr, "optim.fake_lr", 1e-5))
    fake_steps_per_student_step = int(
        pick(args.fake_steps_per_student_step, "optim.fake_steps_per_student_step", 1)
    )
    if fake_steps_per_student_step < 1:
        raise ValueError(
            f"optim.fake_steps_per_student_step={fake_steps_per_student_step}: must be ≥ 1"
        )
    alpha_warmup_steps = int(pick(args.alpha_warmup_steps, "optim.alpha_warmup_steps", 1000))
    weight_decay = float(_flatten(cfg, "optim.weight_decay", 0.0))
    grad_clip = float(_flatten(cfg, "optim.grad_clip", 1.0))

    t_distribution = _flatten(cfg, "sampling.t_distribution", "uniform")
    sigmoid_scale = float(_flatten(cfg, "sampling.sigmoid_scale", 1.0))

    save_every = int(pick(args.save_every, "io.save_every", 1000))
    log_interval = int(pick(args.log_interval, "io.log_interval", 2))
    log_dir = pick(args.log_dir, "io.log_dir", "output/logs/turbo")

    torch.manual_seed(seed)

    # Sanity checks (cheap, catch config typos early).
    if tau_ca_strategy not in ("above_t",):
        raise ValueError(f"dmd.tau_ca_strategy={tau_ca_strategy!r}: only 'above_t' supported in v1")
    if tau_dm_strategy not in ("uniform",):
        raise ValueError(f"dmd.tau_dm_strategy={tau_dm_strategy!r}: only 'uniform' supported in v1")
    if t_distribution not in ("uniform", "sigmoid"):
        raise ValueError(f"sampling.t_distribution={t_distribution!r}: expected 'uniform' or 'sigmoid'")
    if fake_rank < student_rank:
        logger.warning(
            f"fake_rank={fake_rank} < student_rank={student_rank}: DM regularizer "
            "has less capacity than the student — proposal R1 risk amplified. "
            "Consider bumping fake_rank to 2 x student_rank."
        )

    device = torch.device("cuda")
    dtype = torch.bfloat16

    # ---------------- Model ----------------
    logger.info(f"loading DiT: {dit_path}")
    model: Anima = anima_utils.load_anima_model(
        device,
        dit_path,
        attn_mode=attn_mode,
        split_attn=False,
        loading_device="cpu" if args.blocks_to_swap > 0 else device,
        dit_weight_dtype=dtype,
    )

    # Block swap setup (per-forward prepare hook done at each forward call below).
    if args.blocks_to_swap > 0:
        model.enable_block_swap(args.blocks_to_swap, device)
        model.move_to_device_except_swap_blocks(device)
        model.switch_block_swap_for_training()
    else:
        model.to(device)

    # Static 4096 tokens so torch.compile sees a single shape across buckets.
    model.set_static_token_count(4096)

    if args.torch_compile:
        model.compile_blocks(mode="default")

    if args.grad_ckpt:
        model.enable_gradient_checkpointing(unsloth_offload=True)
        logger.info("gradient checkpointing: on (unsloth CPU offload)")
    else:
        logger.info("gradient checkpointing: off")

    # ---------------- LoRA stacks ----------------
    turbo = TurboDMDNetwork(
        unet=model,
        student_rank=student_rank,
        fake_rank=fake_rank,
        student_alpha=student_alpha,
        fake_alpha=fake_alpha,
        use_custom_down_autograd=use_custom_down_autograd,
    )
    turbo.freeze_dit()
    turbo.student.to(device=device, dtype=dtype)
    turbo.fake.to(device=device, dtype=dtype)
    model.train()  # block.forward gates ckpt on self.training

    n_student = sum(p.numel() for p in turbo.student_params())
    n_fake = sum(p.numel() for p in turbo.fake_params())
    logger.info(f"trainable: student={n_student:,}  fake={n_fake:,}")

    # ---------------- Optimizers ----------------
    student_opt = torch.optim.AdamW(
        turbo.student_params(),
        lr=student_lr,
        weight_decay=weight_decay,
        fused=torch.cuda.is_available(),
    )
    fake_opt = torch.optim.AdamW(
        turbo.fake_params(),
        lr=fake_lr,
        weight_decay=weight_decay,
        fused=torch.cuda.is_available(),
    )

    # Warmup + cosine.
    def _make_scheduler(opt, total_steps, lr):
        warmup_steps = max(1, int(0.02 * total_steps))
        warmup = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=1e-6 / lr, total_iters=warmup_steps
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=total_steps - warmup_steps, eta_min=lr * 0.1
        )
        return torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup, cosine], milestones=[warmup_steps]
        )

    student_sched = _make_scheduler(student_opt, iterations, student_lr)
    # Fake gets ``fake_steps_per_student_step`` updates per outer iteration.
    # Cosine should anneal across the actual update count, not the student's.
    fake_sched = _make_scheduler(
        fake_opt, iterations * fake_steps_per_student_step, fake_lr
    )

    # ---------------- Dataset ----------------
    dataset = CachedDataset(
        data_dir,
        batch_size=batch_size,
        sample_ratio=args.sample_ratio,
    )
    if args.single_prompt_idx is not None:
        # Phase 0 overfit — wrap as a 1-sample list so the dataloader cycles it.
        # The "N samples from ..." line above is CachedDataset.__init__'s own
        # log, fired BEFORE this slice; we re-log post-slice so the live
        # dataset state is unambiguous in the run log.
        pinned_idx = args.single_prompt_idx % len(dataset.samples)
        only = dataset.samples[pinned_idx]
        dataset.samples = [only]
        latent_stem = os.path.basename(only[0])
        logger.info(
            f"single-prompt overfit mode: pinned to idx={args.single_prompt_idx} "
            f"(post-slice len(dataset)={len(dataset)}, latent={latent_stem})"
        )

    def _collate(batch):
        return (
            [b[0] for b in batch],
            torch.stack([b[1] for b in batch]),
            torch.stack([b[2] for b in batch]),
            torch.stack([b[3] for b in batch]),  # pooled — unused, but CachedDataset returns it
        )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # bucket-grouped
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        collate_fn=_collate,
    )

    # ---------------- Logging ----------------
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    writer = None
    if not args.no_log:
        from datetime import datetime

        run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_log = Path(log_dir) / run_name
        run_log.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(run_log))
        writer.add_text(
            "config",
            "  \n".join(
                f"{k}: {v}"
                for k, v in {
                    "student_rank": student_rank, "fake_rank": fake_rank,
                    "student_steps": student_steps, "teacher_cfg": teacher_cfg,
                    "alpha_warmup_steps": alpha_warmup_steps,
                    "student_lr": student_lr, "fake_lr": fake_lr,
                    "fake_steps_per_student_step": fake_steps_per_student_step,
                    "iterations": iterations, "batch_size": batch_size,
                    "tau_ca_strategy": tau_ca_strategy,
                    "tau_dm_strategy": tau_dm_strategy,
                    "tau_ca_min_gap": tau_ca_min_gap,
                    "tau_ca_skip_above_t": tau_ca_skip_above_t,
                    "t_distribution": t_distribution,
                    "data_dir": data_dir, "dit_path": dit_path,
                }.items()
            ),
        )
        logger.info(f"TB logs -> {run_log}")

    # ---------------- Training loop ----------------
    # Per-step caches keyed by tensor shape. ``pad`` is a zero tensor in the
    # spatial shape of ``latents``; with constant-token bucketing the shape is
    # stable within a step (and constant in single-prompt mode), so we recycle
    # it instead of re-allocating per forward. Same idea for ``c_null`` — the
    # CFG uncond branch reads ``zeros_like(crossattn_emb)`` 1×/step at most.
    _pad_cache: dict[tuple[int, int, int], torch.Tensor] = {}
    _c_null_cache: dict[tuple, torch.Tensor] = {}

    def _get_pad(x: torch.Tensor) -> torch.Tensor:
        key = (x.shape[0], x.shape[-2], x.shape[-1])
        pad = _pad_cache.get(key)
        if pad is None or pad.dtype != dtype or pad.device != x.device:
            pad = torch.zeros(
                x.shape[0], 1, x.shape[-2], x.shape[-1], dtype=dtype, device=x.device
            )
            _pad_cache[key] = pad
        return pad

    def _forward(view: str, x: torch.Tensor, t_b: torch.Tensor, c: torch.Tensor, *, no_grad: bool):
        """Helper: switch view, prepare block swap, run forward.

        ``x`` is (B, 16, H, W); we unsqueeze to (B, 16, 1, H, W) inside.

        Per-forward CPU prep is the GPU-idle window between launches —
        ``set_view`` short-circuits when already in ``view`` (see
        ``TurboDMDNetwork.set_view``), and the cudagraph step-begin marker
        is hoisted to once per outer step in the loop below.
        """
        turbo.set_view(view)
        if model.blocks_to_swap:
            # free_cache=False: base DiT is frozen, LoRA shapes are constant,
            # block swap moves params at identical shape, and static 4096
            # tokens pins activation sizes — the allocator reaches a steady
            # state within a few steps and per-forward empty_cache() is pure
            # sync + refragmentation overhead.
            model.prepare_block_swap_before_forward(free_cache=False)
        pad = _get_pad(x)
        x_in = x.unsqueeze(2)  # add temporal dim
        ctx = torch.no_grad() if no_grad else torch.enable_grad()
        with ctx, torch.autocast("cuda", dtype=dtype):
            return model.forward_mini_train_dit(
                x_in, t_b, c, padding_mask=pad, skip_pooled_text_proj=True
            )

    logger.info(f"starting DMD2 training: {iterations} iterations")
    data_iter = iter(dataloader)
    progress = tqdm(range(iterations), desc="turbo")
    running_student = 0.0
    running_fake = 0.0
    running_alpha = 0.0
    running_grad = 0.0
    running_dm = 0.0
    running_cfg = 0.0
    running_xpred = 0.0
    running_v_student = 0.0
    running_v_real_dm = 0.0
    running_v_fake_dm = 0.0
    # Per-τ_DM-bucket δ_dm tracking (proposal R1 mitigation b): three bins
    # over [0, 1/3), [1/3, 2/3), [2/3, 1]. Lets us see whether the fake is
    # under-tracking globally or only at a specific noise band.
    bucket_labels = ("lo", "mid", "hi")
    running_dm_buckets = [0.0, 0.0, 0.0]
    running_dm_bucket_counts = [0, 0, 0]

    for step in progress:
        try:
            _idx, latents, crossattn_emb, _pooled = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            _idx, latents, crossattn_emb, _pooled = next(data_iter)

        latents = latents.to(device, dtype=dtype, non_blocking=True)
        crossattn_emb = crossattn_emb.to(device, dtype=dtype, non_blocking=True)
        B = latents.shape[0]

        # One step-begin marker per training step (not per forward).
        # ``compile_blocks(mode="default")`` doesn't enable cudagraphs, so this
        # is semantically a no-op today, but it's the right cadence if/when
        # the script switches to ``mode="reduce-overhead"``.
        torch.compiler.cudagraph_mark_step_begin()

        # --- Sample generator-t ---
        if t_distribution == "uniform":
            t = torch.rand(B, device=device, dtype=dtype)
        else:  # sigmoid
            t = torch.sigmoid(sigmoid_scale * torch.randn(B, device=device, dtype=dtype))

        # --- Build x_t = (1-t)·x_0 + t·ε ---
        eps = torch.randn_like(latents)
        t_e = t.view(B, 1, 1, 1)
        x_t = ((1.0 - t_e) * latents + t_e * eps).requires_grad_()  # requires_grad for grad-ckpt

        # --- 1. STUDENT FORWARD (grad to student) ---
        v_student = _forward("student", x_t, t, crossattn_emb, no_grad=False)
        # v_student: (B, 16, 1, H, W). Drop temporal dim for arithmetic.
        v_student = v_student.squeeze(2)
        x_pred = x_t.squeeze(2) - t_e * v_student   # (B, 16, H, W), grad-bearing

        # --- 2. CA BRANCH (no grad, teacher × 2) ---
        # Skip CA when t is very late (proposal R5): collapsed interval → noisy grad.
        do_ca = bool((t < tau_ca_skip_above_t).any().item())
        if do_ca:
            tau_ca = sample_t_above(t.float(), min_gap=tau_ca_min_gap).to(dtype)
            eps_ca = torch.randn_like(x_pred)
            x_renoised_ca = renoise(x_pred.detach(), tau_ca, eps_ca)
            v_real_cond_ca = _forward(
                "teacher", x_renoised_ca, tau_ca, crossattn_emb, no_grad=True
            ).squeeze(2)
            # zeros_like is cheap, but reusing one buffer also lets dynamo
            # avoid retracing on a fresh tensor id every step.
            key = tuple(crossattn_emb.shape)
            c_null = _c_null_cache.get(key)
            if c_null is None or c_null.dtype != crossattn_emb.dtype or c_null.device != crossattn_emb.device:
                c_null = torch.zeros_like(crossattn_emb)
                _c_null_cache[key] = c_null
            v_real_uncond_ca = _forward(
                "teacher", x_renoised_ca, tau_ca, c_null, no_grad=True
            ).squeeze(2)
            delta_cfg = v_real_cond_ca - v_real_uncond_ca
        else:
            delta_cfg = torch.zeros_like(x_pred)

        # --- 3. DM BRANCH (no grad teacher + no grad fake) ---
        tau_dm = torch.rand(B, device=device, dtype=dtype)
        eps_dm = torch.randn_like(x_pred)
        x_renoised_dm = renoise(x_pred.detach(), tau_dm, eps_dm)
        v_real_cond_dm = _forward(
            "teacher", x_renoised_dm, tau_dm, crossattn_emb, no_grad=True
        ).squeeze(2)
        v_fake_cond_dm = _forward(
            "fake", x_renoised_dm, tau_dm, crossattn_emb, no_grad=True
        ).squeeze(2)
        delta_dm = v_real_cond_dm - v_fake_cond_dm

        # --- 4. ASSEMBLE + BACKWARD into student ---
        warmup_frac = min(1.0, (step + 1) / max(1, alpha_warmup_steps))
        alpha_eff = teacher_cfg * warmup_frac + 1.0 * (1.0 - warmup_frac)
        grad_signal = (delta_dm + (alpha_eff - 1.0) * delta_cfg).detach()

        # DMD2 grad trick: sneak the detached gradient into autograd via a
        # dummy scalar whose ∂/∂x_pred equals weight. Backward then walks
        # x_pred -> v_student -> student params.
        loss_student = (-grad_signal * x_pred).mean()
        loss_student.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(turbo.student_params(), max_norm=grad_clip)
        student_opt.step()
        student_opt.zero_grad(set_to_none=True)
        student_sched.step()

        # --- 5. FAKE UPDATE ---
        # Run ``fake_steps_per_student_step`` inner updates against the same
        # x_pred.detach(), resampling (τ_fake, ε_fake) each iteration so each
        # inner step sees a different rung of the flow-matching forward path.
        # Standard DMD2 practice: keep the fake's regression target ahead of
        # the student's moving x_pred distribution.
        x_pred_d = x_pred.detach()
        fake_loss_sum = 0.0
        for _ in range(fake_steps_per_student_step):
            tau_fake = (
                torch.rand(B, device=device, dtype=dtype)
                if t_distribution == "uniform"
                else torch.sigmoid(sigmoid_scale * torch.randn(B, device=device, dtype=dtype))
            )
            eps_fake = torch.randn_like(x_pred_d)
            x_t_fake = renoise(x_pred_d, tau_fake, eps_fake).requires_grad_()
            v_fake = _forward(
                "fake", x_t_fake, tau_fake, crossattn_emb, no_grad=False
            ).squeeze(2)
            target_v_fake = eps_fake - x_pred_d  # flow-matching target
            fake_loss = nn.functional.mse_loss(v_fake.float(), target_v_fake.float())
            fake_loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(turbo.fake_params(), max_norm=grad_clip)
            fake_opt.step()
            fake_opt.zero_grad(set_to_none=True)
            fake_sched.step()
            fake_loss_sum += fake_loss.detach().item()
        fake_loss_mean = fake_loss_sum / fake_steps_per_student_step

        # --- logging ---
        # DMD2 health scalars. loss_student is a sign-random gradient vehicle
        # (not a real loss); the RMS norms below are what actually track
        # whether the student is getting a usable signal:
        #   grad_rms — overall DMD2 gradient magnitude into x_pred
        #   dm_rms   — DM regularizer strength (v_real - v_fake)
        #   cfg_rms  — CA branch strength (CFG bake direction)
        #   xpred_rms — x_pred dispersion: → 0 means collapse to mean,
        #               drifting upward means student is exploding.
        s_now = loss_student.detach().item()
        f_now = fake_loss_mean
        with torch.no_grad():
            grad_rms = grad_signal.float().pow(2).mean().sqrt().item()
            dm_rms = delta_dm.float().pow(2).mean().sqrt().item()
            cfg_rms = delta_cfg.float().pow(2).mean().sqrt().item()
            xpred_rms = x_pred.detach().float().std().item()
            # Direct student velocity magnitude — runaway student manifests
            # here before x_pred_std catches up (x_pred = x_t − t·v_student).
            v_student_rms = v_student.detach().float().pow(2).mean().sqrt().item()
            # Teacher vs fake magnitudes at the DM-branch evaluation point.
            # If v_real_dm stays bounded while v_fake_dm explodes (or stays
            # tiny while real grows), the fake-tracking gap is asymmetric in
            # a diagnostic way the aggregate δ_dm hides.
            v_real_dm_rms = v_real_cond_dm.float().pow(2).mean().sqrt().item()
            v_fake_dm_rms = v_fake_cond_dm.float().pow(2).mean().sqrt().item()
            # Per-sample δ_dm, bucketed by τ_DM. Tells us whether the fake's
            # tracking gap is uniform across noise levels or concentrated at
            # one band (the proposal's R1 mitigation b).
            per_sample_dm = delta_dm.float().pow(2).mean(dim=(1, 2, 3)).sqrt()  # (B,)
            tau_dm_bucket = (tau_dm.float() * 3.0).clamp(max=2.999).floor().long()  # (B,)
            for b in range(B):
                bi = int(tau_dm_bucket[b].item())
                running_dm_buckets[bi] += float(per_sample_dm[b].item())
                running_dm_bucket_counts[bi] += 1
        running_student += s_now
        running_fake += f_now
        running_alpha += alpha_eff
        running_grad += grad_rms
        running_dm += dm_rms
        running_cfg += cfg_rms
        running_xpred += xpred_rms
        running_v_student += v_student_rms
        running_v_real_dm += v_real_dm_rms
        running_v_fake_dm += v_fake_dm_rms

        # Update tqdm every step so the bar always shows live signal (gating
        # this behind ``log_interval`` left the first N steps with a blank
        # postfix). TB scalars stay on the log_interval cadence below.
        progress.set_postfix(
            g=f"{grad_rms:.3e}",
            dca=f"{cfg_rms:.3e}",
            ddm=f"{dm_rms:.3e}",
            xp=f"{xpred_rms:.3f}",
            vs=f"{v_student_rms:.3f}",
            fake=f"{f_now:.3e}",
        )

        if (step + 1) % log_interval == 0:
            avg_s = running_student / log_interval
            avg_f = running_fake / log_interval
            avg_a = running_alpha / log_interval
            avg_g = running_grad / log_interval
            avg_dm = running_dm / log_interval
            avg_cfg = running_cfg / log_interval
            avg_xp = running_xpred / log_interval
            avg_vs = running_v_student / log_interval
            avg_vrdm = running_v_real_dm / log_interval
            avg_vfdm = running_v_fake_dm / log_interval
            if writer is not None:
                writer.add_scalar("train/student_loss", avg_s, step + 1)
                writer.add_scalar("train/fake_loss", avg_f, step + 1)
                writer.add_scalar("train/alpha_eff", avg_a, step + 1)
                writer.add_scalar("train/grad_signal_rms", avg_g, step + 1)
                writer.add_scalar("train/delta_dm_rms", avg_dm, step + 1)
                writer.add_scalar("train/delta_cfg_rms", avg_cfg, step + 1)
                writer.add_scalar("train/x_pred_std", avg_xp, step + 1)
                writer.add_scalar("train/v_student_rms", avg_vs, step + 1)
                writer.add_scalar("train/v_real_dm_rms", avg_vrdm, step + 1)
                writer.add_scalar("train/v_fake_dm_rms", avg_vfdm, step + 1)
                for bi, label in enumerate(bucket_labels):
                    n = running_dm_bucket_counts[bi]
                    if n > 0:
                        writer.add_scalar(
                            f"train/delta_dm_rms_tau_{label}",
                            running_dm_buckets[bi] / n,
                            step + 1,
                        )
                writer.add_scalar(
                    "train/student_lr", student_sched.get_last_lr()[0], step + 1
                )
                writer.add_scalar("train/fake_lr", fake_sched.get_last_lr()[0], step + 1)
                writer.add_scalar("train/t_mean", t.float().mean().item(), step + 1)
            running_student = running_fake = running_alpha = 0.0
            running_grad = running_dm = running_cfg = running_xpred = 0.0
            running_v_student = running_v_real_dm = running_v_fake_dm = 0.0
            running_dm_buckets = [0.0, 0.0, 0.0]
            running_dm_bucket_counts = [0, 0, 0]

        # --- save ---
        if (step + 1) % save_every == 0 or (step + 1) == iterations:
            save_path = str(Path(output_dir) / f"{output_name}.safetensors")
            turbo.save_student(
                save_path,
                dtype=torch.bfloat16,
                metadata={
                    "ss_turbo_student_rank": str(student_rank),
                    "ss_turbo_student_steps": str(student_steps),
                    "ss_turbo_teacher_cfg": str(teacher_cfg),
                    "ss_turbo_step": str(step + 1),
                },
            )

    if writer is not None:
        writer.close()
    logger.info("turbo distillation complete.")


if __name__ == "__main__":
    main()
