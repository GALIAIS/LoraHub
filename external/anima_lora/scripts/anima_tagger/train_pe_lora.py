"""End-to-end PE-LoRA training path.

PE-Core's trailing N blocks are unfrozen via :func:`inject_pe_lora`; the
trainer reads pre-resized images from ``out_dir/.cache/resized-<encoder>/``
(built via ``--mode build_resized``) and runs encoder + mean-pool + head
per step. Two param groups so the head trains at ``--lr`` and the LoRA
delta at ``--pe_lora_lr``.

Bucket-grouped batches keep encoder forwards shape-homogeneous —
``BucketBatchSampler`` is shared with the cached-image dataset class.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from .train_common import (
    GroupRouter,
    build_warmup_cosine_scheduler,
    compute_grouped_loss,
    people_class_weights,
    rating_class_weights,
    save_history_plot,
)

logger = logging.getLogger(__name__)


def _u8_to_minus1to1(u8_batch: torch.Tensor) -> torch.Tensor:
    """``uint8 [0..255]`` → ``float32 [-1, 1]`` (matches IMAGE_TRANSFORMS)."""
    return u8_batch.to(torch.float32) / 127.5 - 1.0


def cmd_train_pe_lora(args: argparse.Namespace) -> None:
    """End-to-end PE-LoRA path: PE encoder is unfrozen on its trailing N
    blocks via ``inject_pe_lora``; trainer reads pre-resized images and
    runs encoder + mean-pool + head per step."""
    from safetensors.torch import load_file as st_load_file
    from safetensors.torch import save_file as st_save

    from library.captioning.anima_tagger_data import (
        BucketBatchSampler,
        CachedImageDataset,
        TaggerManifest,
        collate_image_batch,
    )
    from library.captioning.anima_tagger_model import (
        AnimaTaggerConfig,
        AnimaTaggerHead,
    )
    from library.vision.encoder import load_pe_encoder
    from library.vision.encoders import get_encoder_info
    from networks.methods.ip_adapter_pe_lora import inject_pe_lora

    out_dir = Path(args.out_dir)
    manifest_path = out_dir / "dataset.json"
    vocab_path = out_dir / "vocab.json"
    image_cache_dir = out_dir / ".cache" / f"resized-{args.encoder}"
    if not manifest_path.exists():
        raise SystemExit(f"missing {manifest_path} — run --mode build_vocab first.")
    if not vocab_path.exists():
        raise SystemExit(f"missing {vocab_path} — run --mode build_vocab first.")
    if not image_cache_dir.exists():
        raise SystemExit(
            f"missing {image_cache_dir} — run --mode build_resized first "
            f"(required for --pe_lora_rank > 0)."
        )
    manifest = TaggerManifest.from_path(manifest_path)
    with open(vocab_path) as f:
        vocab_dict = json.load(f)
    spec = get_encoder_info(args.encoder).bucket_spec
    d_enc = get_encoder_info(args.encoder).d_enc

    train_ds = CachedImageDataset(
        manifest, image_cache_dir, spec, stems_subset=manifest.train_stems
    )
    val_ds = CachedImageDataset(
        manifest, image_cache_dir, spec, stems_subset=manifest.val_stems
    )
    logger.info(
        "train (PE-LoRA r=%d, last %d blocks): N=%d  val: N=%d  d_enc=%d  "
        "n_tags=%d  n_ratings=%d  n_people=%d",
        args.pe_lora_rank,
        args.pe_lora_layers,
        len(train_ds),
        len(val_ds),
        d_enc,
        train_ds.n_tags,
        train_ds.n_ratings,
        train_ds.n_people_counts,
    )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Head — same arch, ``d_in`` now comes from the encoder's d_enc.
    cfg = AnimaTaggerConfig(
        d_in=d_enc,
        n_tags=train_ds.n_tags,
        n_ratings=train_ds.n_ratings,
        n_people_counts=train_ds.n_people_counts,
        d_hidden=args.d_hidden,
        dropout=args.dropout,
        pool_kind=args.pool_kind,
        pool_n_queries=args.pool_n_queries,
        pool_n_heads=args.pool_n_heads,
        pool_use_cls=args.pool_use_cls,
        pool_use_mean=args.pool_use_mean,
    )
    model = AnimaTaggerHead(cfg).to(device)
    logger.info(
        "head: pool_kind=%s n_queries=%d n_heads=%d use_cls=%s use_mean=%s "
        "trunk_in=%d d_hidden=%d",
        cfg.pool_kind, cfg.pool_n_queries, cfg.pool_n_heads,
        cfg.pool_use_cls, cfg.pool_use_mean, cfg.trunk_in_dim, cfg.d_hidden,
    )

    # Optional warm-start from a Stage-1 cached-feature run. Loads strict —
    # any key mismatch (different vocab / hidden dim) errors out instead of
    # silently dropping params. Optimizer state is intentionally NOT loaded;
    # Stage 2 re-builds Adam from scratch since the param groups (head + LoRA)
    # and the LR schedule both differ from Stage 1.
    if getattr(args, "init_head_from", None):
        init_path = Path(args.init_head_from)
        if not init_path.exists():
            raise SystemExit(f"--init_head_from: {init_path} does not exist")
        head_state = st_load_file(str(init_path))
        missing, unexpected = model.load_state_dict(head_state, strict=False)
        if missing or unexpected:
            raise SystemExit(
                f"--init_head_from: state_dict mismatch against current "
                f"AnimaTaggerHead config "
                f"(d_in={cfg.d_in}, n_tags={cfg.n_tags}, n_ratings={cfg.n_ratings}, "
                f"n_people_counts={cfg.n_people_counts}, d_hidden={cfg.d_hidden}). "
                f"missing={list(missing)[:5]}{'...' if len(missing) > 5 else ''}  "
                f"unexpected={list(unexpected)[:5]}{'...' if len(unexpected) > 5 else ''}"
            )
        logger.info("warm-started head from %s", init_path)

    # Frozen encoder + LoRA on the trailing blocks.
    bundle = load_pe_encoder(device, name=args.encoder, dtype=torch.bfloat16)
    pe_inner = bundle.encoder.inner       # PEVisionTransformer
    pe_inner.requires_grad_(False)
    pe_lora = inject_pe_lora(
        pe_inner,
        rank=args.pe_lora_rank,
        alpha=args.pe_lora_alpha,
        target_qkv=args.pe_lora_qkv,
        target_attn_out=args.pe_lora_attn_out,
        target_mlp=args.pe_lora_mlp,
        layer_from=args.pe_lora_layers,
    )
    pe_lora.to(device=device, dtype=torch.float32)

    # Loss weights — the multi-hot / rating / people tensors live on the
    # dataset already in fp32 / int64; aggregate over the train split for
    # pos / class weights once.
    train_mh_full = train_ds.multi_hot.to(device)
    train_rate_full = train_ds.rating_idx.to(device)
    train_people_full = train_ds.people_idx.to(device)
    router = GroupRouter.from_vocab(vocab_dict, train_mh_full, device=device)
    rating_w = rating_class_weights(train_rate_full, train_ds.n_ratings).to(device)
    ce = torch.nn.CrossEntropyLoss(weight=rating_w)
    if train_ds.n_people_counts > 0:
        people_w = people_class_weights(train_people_full, train_ds.n_people_counts).to(device)
        ce_people = torch.nn.CrossEntropyLoss(weight=people_w)
        logger.info(
            "people-count head: %d classes, sqrt-inverse weights=%s",
            train_ds.n_people_counts,
            [round(float(w), 3) for w in people_w.cpu().tolist()],
        )
    else:
        ce_people = None
        logger.info("no people-count labels in manifest — skipping people head")
    if router.is_active():
        n_softmax_tags = (
            int(router.softmax_member_indices.numel())
            if router.softmax_member_indices is not None else 0
        )
        logger.info(
            "groups active: %d softmax groups (%d softmax-member tags / %d total)",
            len(router.softmax_groups), n_softmax_tags, train_ds.n_tags,
        )
        for g in router.softmax_groups:
            logger.info(
                "  %-14s mode=%-18s K=%d  escape=%d",
                g.name, g.mode, int(g.tag_indices.numel()),
                int(g.escape_indices.numel()),
            )
    else:
        logger.info("no typed groups — pure BCE on every tag")

    train_sampler = BucketBatchSampler(
        train_ds.buckets, batch_size=args.batch_size, seed=args.seed, shuffle=True
    )
    val_sampler = BucketBatchSampler(
        val_ds.buckets, batch_size=args.batch_size, seed=args.seed, shuffle=False
    )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        num_workers=args.feature_cache_workers,
        collate_fn=collate_image_batch,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_sampler=val_sampler,
        num_workers=args.feature_cache_workers,
        collate_fn=collate_image_batch,
        pin_memory=True,
    )

    # Two param groups so the head trains at --lr and the LoRA at --pe_lora_lr.
    opt = torch.optim.AdamW(
        [
            {"params": list(model.parameters()), "lr": args.lr,
             "weight_decay": args.weight_decay},
            {"params": list(pe_lora.parameters()), "lr": args.pe_lora_lr,
             "weight_decay": 0.0},
        ],
        fused=torch.cuda.is_available(),
    )
    sched = build_warmup_cosine_scheduler(
        opt,
        warmup_steps=int(getattr(args, "warmup_steps", 0)),
        total_steps=max(args.epochs * len(train_loader), 1),
        eta_min=args.lr * 0.05,
    )

    def _forward_pool(images_u8: torch.Tensor) -> torch.Tensor:
        """images_u8 [B, C, H, W] uint8 (CPU) → encoder features (device, fp32).

        Output shape depends on ``pool_kind``:
          * ``mean`` → ``[B, d_enc]`` (mean-pooled, matches FeatureCacheBuilder)
          * ``map``  → ``[B, T, d_enc]`` (full token sequence; the head's
            MAPHead pools internally)
        """
        x = _u8_to_minus1to1(images_u8.to(device, non_blocking=True))
        x = x.to(bundle.dtype)
        # pe_inner.encode returns (last_hidden_state[B,T,D], pooled[B,D_pool]).
        feats, _pooled = pe_inner.encode(x)                    # [B, T, D_enc]
        feats = feats.to(torch.float32)
        if cfg.pool_kind == "mean":
            return feats.mean(dim=1)                           # [B, D_enc]
        return feats                                           # [B, T, D_enc]

    best_f1 = -1.0
    best_head_state: Dict[str, torch.Tensor] = {}
    best_lora_state: Dict[str, torch.Tensor] = {}
    history: List[Dict[str, float]] = []

    from tqdm import tqdm as _tqdm

    # Sync-defer cadence for the tqdm postfix. Every per-step .item() forces
    # a host-device sync; accumulating losses as GPU tensors and reading them
    # only every N steps cuts the sync count ~Nx without losing precision in
    # the epoch-end average (tensors are summed on-device, .item() once).
    postfix_every = max(1, int(getattr(args, "postfix_every", 10)))

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        pe_lora.train()
        ep_loss = torch.zeros((), device=device)
        ep_tag_loss = torch.zeros((), device=device)
        ep_rate_loss = torch.zeros((), device=device)
        ep_people_loss = torch.zeros((), device=device)
        n_batches = 0
        bar = _tqdm(
            train_loader,
            desc=f"ep {epoch + 1}/{args.epochs}",
            leave=False,
            unit="step",
        )
        for step, (images_u8, mh_cpu, rate_cpu, people_cpu, _bucket) in enumerate(bar):
            mh = mh_cpu.to(device, non_blocking=True)
            rate = rate_cpu.to(device, non_blocking=True)
            people = people_cpu.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                feat = _forward_pool(images_u8)
                tag_logits, rating_logits, people_logits = model(feat)
                l_tag, _per_group = compute_grouped_loss(tag_logits, mh, router)
                l_rate = ce(rating_logits, rate)
                loss = l_tag + args.lambda_rating * l_rate
                if ce_people is not None and people_logits is not None:
                    l_people = ce_people(people_logits, people)
                    loss = loss + args.lambda_people * l_people
                else:
                    l_people = loss.new_zeros(())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            ep_loss += loss.detach()
            ep_tag_loss += l_tag.detach()
            ep_rate_loss += l_rate.detach()
            ep_people_loss += l_people.detach()
            n_batches += 1
            if step % postfix_every == 0:
                postfix = {
                    "loss": f"{loss.item():.4f}",
                    "tag": f"{l_tag.item():.4f}",
                    "rate": f"{l_rate.item():.4f}",
                }
                if ce_people is not None and people_logits is not None:
                    postfix["ppl"] = f"{l_people.item():.4f}"
                bar.set_postfix(**postfix)
        denom = max(n_batches, 1)
        avg_loss = (ep_loss / denom).item()
        avg_tag = (ep_tag_loss / denom).item()
        avg_rate = (ep_rate_loss / denom).item()
        avg_people = (ep_people_loss / denom).item()

        # Eval — collect logits over val in mini-batches, then reuse the
        # existing macro-F1 helper. Threshold sweep happens at calibrate.
        model.eval()
        pe_lora.eval()
        val_tag_logits: List[torch.Tensor] = []
        val_rating_logits: List[torch.Tensor] = []
        val_people_logits: List[torch.Tensor] = []
        val_mh_chunks: List[torch.Tensor] = []
        val_rate_chunks: List[torch.Tensor] = []
        val_people_chunks: List[torch.Tensor] = []
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for images_u8, mh_cpu, rate_cpu, people_cpu, _bucket in val_loader:
                feat = _forward_pool(images_u8)
                tl, rl, pl = model(feat)
                val_tag_logits.append(tl.float())
                val_rating_logits.append(rl.float())
                if pl is not None:
                    val_people_logits.append(pl.float())
                val_mh_chunks.append(mh_cpu.to(device, non_blocking=True))
                val_rate_chunks.append(rate_cpu.to(device, non_blocking=True))
                val_people_chunks.append(people_cpu.to(device, non_blocking=True))
        tag_logits_all = torch.cat(val_tag_logits, dim=0)
        rating_logits_all = torch.cat(val_rating_logits, dim=0)
        people_logits_all = (
            torch.cat(val_people_logits, dim=0) if val_people_logits else None
        )
        val_mh = torch.cat(val_mh_chunks, dim=0)
        val_rate = torch.cat(val_rate_chunks, dim=0)
        val_people = torch.cat(val_people_chunks, dim=0)
        # F1 excludes softmax-group tags when the router is active — those
        # are argmax-only at inference, so their sigmoid threshold isn't
        # the right metric. Per-group accuracy is reported separately.
        if router.is_active() and router.softmax_member_indices is not None:
            keep_mask = torch.ones(
                tag_logits_all.shape[1], dtype=torch.bool, device=tag_logits_all.device
            )
            keep_mask[router.softmax_member_indices] = False
            kept_idx = keep_mask.nonzero(as_tuple=False).squeeze(1)
            f1_logits = tag_logits_all.index_select(1, kept_idx)
            f1_target = val_mh.index_select(1, kept_idx)
        else:
            f1_logits = tag_logits_all
            f1_target = val_mh
        pred = (f1_logits.sigmoid() > 0.5).float()
        tp = (pred * f1_target).sum(dim=0)
        fp = (pred * (1 - f1_target)).sum(dim=0)
        fn = ((1 - pred) * f1_target).sum(dim=0)
        prec = tp / (tp + fp).clamp_min(1.0)
        rec = tp / (tp + fn).clamp_min(1.0)
        f1 = 2 * prec * rec / (prec + rec).clamp_min(1e-8)
        rating_acc = (rating_logits_all.argmax(dim=-1) == val_rate).float().mean().item()
        with torch.no_grad():
            val_l_tag, _ = compute_grouped_loss(tag_logits_all, val_mh, router)
            val_l_rate = ce(rating_logits_all, val_rate)
            val_l_total = val_l_tag + args.lambda_rating * val_l_rate
            if ce_people is not None and people_logits_all is not None:
                val_l_people = ce_people(people_logits_all, val_people)
                val_l_total = val_l_total + args.lambda_people * val_l_people
                people_acc = (
                    people_logits_all.argmax(dim=-1) == val_people
                ).float().mean().item()
            else:
                val_l_people = None
                people_acc = float("nan")
        val_metrics = {
            "macro_f1": f1.mean().item(),
            "macro_precision": prec.mean().item(),
            "macro_recall": rec.mean().item(),
            "rating_acc": rating_acc,
            "val_tag_loss": val_l_tag.item(),
            "val_rate_loss": val_l_rate.item(),
            "val_loss": val_l_total.item(),
        }
        if val_l_people is not None:
            val_metrics["val_people_loss"] = val_l_people.item()
            val_metrics["people_acc"] = people_acc
        # Per-softmax-group argmax accuracy.
        if router.is_active():
            solo_mask = router.solo_mask(val_mh)
            for g in router.softmax_groups:
                if g.escape_indices.numel() > 0:
                    has_escape = val_mh.index_select(1, g.escape_indices).any(dim=1)
                else:
                    has_escape = torch.zeros_like(solo_mask)
                applicable = (solo_mask & ~has_escape) if g.mode == "softmax_when_solo" else ~has_escape
                gl = tag_logits_all.index_select(1, g.tag_indices)
                gt = val_mh.index_select(1, g.tag_indices)
                has_label = gt.sum(dim=1) > 0
                keep = applicable & has_label
                n_keep = int(keep.sum().item())
                if n_keep == 0:
                    val_metrics[f"acc_{g.name}"] = 0.0
                    val_metrics[f"n_{g.name}"] = 0
                    continue
                pred_idx = gl[keep].argmax(dim=1)
                true_idx = gt[keep].argmax(dim=1)
                val_metrics[f"acc_{g.name}"] = (pred_idx == true_idx).float().mean().item()
                val_metrics[f"n_{g.name}"] = n_keep
        logger.info(
            "epoch %2d/%d  loss=%.4f (tag=%.4f rate=%.4f people=%.4f)  "
            "val_loss=%.4f (tag=%.4f rate=%.4f people=%.4f)  "
            "val_f1=%.4f  val_p=%.4f  val_r=%.4f  rate_acc=%.4f  people_acc=%.4f  lr=%.2e",
            epoch + 1,
            args.epochs,
            avg_loss,
            avg_tag,
            avg_rate,
            avg_people,
            val_metrics["val_loss"],
            val_metrics["val_tag_loss"],
            val_metrics["val_rate_loss"],
            val_metrics.get("val_people_loss", float("nan")),
            val_metrics["macro_f1"],
            val_metrics["macro_precision"],
            val_metrics["macro_recall"],
            val_metrics["rating_acc"],
            val_metrics.get("people_acc", float("nan")),
            sched.get_last_lr()[0],
        )
        history.append({
            "epoch": epoch + 1,
            "loss": avg_loss,
            "tag_loss": avg_tag,
            "rate_loss": avg_rate,
            "people_loss": avg_people,
            **val_metrics,
        })
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_head_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            best_lora_state = {
                k: v.detach().cpu().clone() for k, v in pe_lora.state_dict().items()
            }

    if not best_head_state:
        raise SystemExit("no epochs ran — empty training set?")

    ckpt_path = out_dir / "model.safetensors"
    pe_lora_path = out_dir / "pe_lora.safetensors"
    cfg_path = out_dir / "config.json"
    history_path = out_dir / "train_history.json"
    st_save(best_head_state, str(ckpt_path))
    st_save(best_lora_state, str(pe_lora_path))
    with open(cfg_path, "w") as f:
        json.dump(
            {
                "model": cfg.to_dict(),
                "encoder": args.encoder,
                "d_in": d_enc,
                "best_val_macro_f1": best_f1,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "lambda_rating": args.lambda_rating,
                "lambda_people": args.lambda_people,
                "seed": args.seed,
                "pe_lora": True,
                "pe_lora_rank": args.pe_lora_rank,
                "pe_lora_alpha": args.pe_lora_alpha,
                "pe_lora_layers": args.pe_lora_layers,
                "pe_lora_lr": args.pe_lora_lr,
                "pe_lora_qkv": args.pe_lora_qkv,
                "pe_lora_attn_out": args.pe_lora_attn_out,
                "pe_lora_mlp": args.pe_lora_mlp,
                "init_head_from": (
                    str(args.init_head_from) if args.init_head_from else None
                ),
                "pool_kind": args.pool_kind,
            },
            f,
            indent=2,
        )
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    plot_path = out_dir / "train_history.png"
    save_history_plot(history, plot_path)
    logger.info(
        "wrote %s / %s / %s / %s / %s",
        ckpt_path, pe_lora_path, cfg_path, history_path, plot_path,
    )
    print(f"  best val macro_f1: {best_f1:.4f}")
