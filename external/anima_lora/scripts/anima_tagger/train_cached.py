"""Frozen-encoder training path.

Two cache layouts are supported, selected by ``--pool_kind``:

* ``mean`` (legacy) — pre-pooled ``[d_enc]`` features under
  ``out_dir/.cache/pooled-<encoder>/``. Whole train/val sets fit in
  VRAM at ~50 MB each, so we push them once and slice batches by index;
  no DataLoader.
* ``map`` (default for new runs) — full token sequences ``[T, d_enc]``
  under ``out_dir/.cache/tokens-<encoder>/``. Loaded lazily through a
  bucket-grouped DataLoader so within-batch T is constant; the
  :class:`AnimaTaggerHead` MAP pool runs inside the model forward.

Both build via ``--mode build_features`` (which dispatches the matching
cache builder based on the same ``--pool_kind``).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from .caches import cache_dir_for
from .train_common import (
    GroupRouter,
    build_warmup_cosine_scheduler,
    compute_grouped_loss,
    eval_split,
    people_class_weights,
    rating_class_weights,
    save_history_plot,
)

logger = logging.getLogger(__name__)


def _routing_indices_from_vocab(
    vocab_dict: Dict, n_tags: int,
) -> Tuple[List[int], List[int]]:
    """Split vocab indices into (main-lean, aux-lean) buckets by category.

    Main-lean (PE-Core-aligned): character, copyright, artist, count — these
    are global semantic / identity-class signals that match what CLIP-style
    PE-Core was trained to recognize. Aux-lean (PE-Spatial-aligned):
    everything else (general, metadata, deprecated) — patch-level detail
    where the spatial encoder's per-token features carry the signal.

    Returns a deterministic partition of ``[0, n_tags)`` in vocab order.
    """
    main_cats = {"character", "copyright", "artist", "count"}
    main: List[int] = []
    aux: List[int] = []
    for t in vocab_dict.get("tags", []):
        idx = int(t["index"])
        cat = str(t.get("category", "general"))
        if cat in main_cats:
            main.append(idx)
        else:
            aux.append(idx)
    main.sort()
    aux.sort()
    if sorted(main + aux) != list(range(n_tags)):
        raise SystemExit(
            f"routing partition is malformed: {len(main)} main + {len(aux)} aux "
            f"!= {n_tags} expected. vocab.json may carry duplicate or missing "
            f"tag indices."
        )
    return main, aux


def _make_cfg_from_args(
    args,
    d_in,
    n_tags,
    n_ratings,
    n_people_counts,
    *,
    d_in_aux: Optional[int] = None,
    routing: Optional[Tuple[List[int], List[int]]] = None,
):
    from library.captioning.anima_tagger_model import AnimaTaggerConfig

    # pool_kind_aux=None inherits pool_kind (matches the dual-MAP path's
    # original behavior); pass through whatever the CLI set explicitly so
    # mixed configs (mean main + map aux) round-trip.
    use_routing = bool(getattr(args, "use_per_head_routing", False))
    if use_routing and routing is None:
        raise SystemExit(
            "use_per_head_routing=True but no (main, aux) index partition was "
            "supplied — call _routing_indices_from_vocab(vocab_dict, n_tags) first."
        )
    tag_main, tag_aux = routing if (use_routing and routing) else ([], [])
    return AnimaTaggerConfig(
        d_in=d_in,
        n_tags=n_tags,
        n_ratings=n_ratings,
        n_people_counts=n_people_counts,
        d_hidden=args.d_hidden,
        dropout=args.dropout,
        pool_kind=args.pool_kind,
        pool_n_queries=args.pool_n_queries,
        pool_n_heads=args.pool_n_heads,
        pool_use_cls=args.pool_use_cls,
        pool_use_mean=args.pool_use_mean,
        d_in_aux=d_in_aux,
        pool_kind_aux=args.pool_kind_aux,
        pool_n_queries_aux=args.pool_n_queries_aux,
        pool_n_heads_aux=args.pool_n_heads_aux,
        pool_use_cls_aux=args.pool_use_cls_aux,
        pool_use_mean_aux=args.pool_use_mean_aux,
        use_per_head_routing=use_routing,
        tag_indices_main=tag_main,
        tag_indices_aux=tag_aux,
    )


def _save_cfg_dict(args, cfg, d_in, best_f1):
    d = {
        "model": cfg.to_dict(),
        "encoder": args.encoder,
        "d_in": d_in,
        "best_val_macro_f1": best_f1,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "lambda_rating": args.lambda_rating,
        "lambda_people": args.lambda_people,
        "seed": args.seed,
        "pe_lora": False,
        "pool_kind": args.pool_kind,
    }
    aux_encoder = getattr(args, "aux_encoder", None)
    if aux_encoder:
        d["aux_encoder"] = aux_encoder
        d["d_in_aux"] = cfg.d_in_aux
    if cfg.use_per_head_routing:
        d["use_per_head_routing"] = True
        d["n_tag_indices_main"] = len(cfg.tag_indices_main)
        d["n_tag_indices_aux"] = len(cfg.tag_indices_aux)
    return d


# ── pool_kind == "mean" : legacy in-VRAM-tensor path ─────────────────────


def _train_cached_mean(args: argparse.Namespace) -> None:
    from safetensors.torch import save_file as st_save

    from library.captioning.anima_tagger_data import (
        CachedFeatureDataset,
        TaggerManifest,
    )
    from library.captioning.anima_tagger_model import AnimaTaggerHead

    out_dir = Path(args.out_dir)
    manifest_path = out_dir / "dataset.json"
    vocab_path = out_dir / "vocab.json"
    cache_dir = cache_dir_for(out_dir, args.pool_kind, args.encoder)
    if not manifest_path.exists():
        raise SystemExit(f"missing {manifest_path} — run --mode build_vocab first.")
    if not vocab_path.exists():
        raise SystemExit(f"missing {vocab_path} — run --mode build_vocab first.")
    if not cache_dir.exists():
        raise SystemExit(
            f"missing {cache_dir} — run --mode build_features --pool_kind=mean first."
        )
    manifest = TaggerManifest.from_path(manifest_path)
    with open(vocab_path) as f:
        vocab_dict = json.load(f)
    train_ds = CachedFeatureDataset(manifest, cache_dir, stems_subset=manifest.train_stems)
    val_ds = CachedFeatureDataset(manifest, cache_dir, stems_subset=manifest.val_stems)
    logger.info(
        "train (cached mean): N=%d  val: N=%d  d_in=%d  n_tags=%d  n_ratings=%d  n_people=%d",
        len(train_ds), len(val_ds), train_ds.d_in,
        train_ds.n_tags, train_ds.n_ratings, train_ds.n_people_counts,
    )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if getattr(args, "use_per_head_routing", False):
        # Mean-pool path is single-encoder; routing requires aux. Fail fast.
        raise SystemExit(
            "--use_per_head_routing requires --aux_encoder, which is "
            "incompatible with the single-encoder mean-pool path. Re-run "
            "with --aux_encoder pe_spatial --pool_kind map."
        )
    cfg = _make_cfg_from_args(
        args,
        d_in=train_ds.d_in,
        n_tags=train_ds.n_tags,
        n_ratings=train_ds.n_ratings,
        n_people_counts=train_ds.n_people_counts,
    )
    model = AnimaTaggerHead(cfg).to(device)

    train_feats = train_ds.features.to(device)
    train_mh = train_ds.multi_hot.to(device)
    train_rate = train_ds.rating_idx.to(device)
    train_people = train_ds.people_idx.to(device)
    val_feats = val_ds.features.to(device)
    val_mh = val_ds.multi_hot.to(device)
    val_rate = val_ds.rating_idx.to(device)
    val_people = val_ds.people_idx.to(device)

    router = GroupRouter.from_vocab(vocab_dict, train_mh, device=device)
    rating_w = rating_class_weights(train_rate, train_ds.n_ratings).to(device)
    ce = torch.nn.CrossEntropyLoss(weight=rating_w)
    if train_ds.n_people_counts > 0:
        people_w = people_class_weights(train_people, train_ds.n_people_counts).to(device)
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

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        fused=torch.cuda.is_available(),
    )
    n_train = len(train_ds)
    steps_per_epoch = (n_train + args.batch_size - 1) // args.batch_size
    sched = build_warmup_cosine_scheduler(
        opt,
        warmup_steps=int(getattr(args, "warmup_steps", 0)),
        total_steps=max(args.epochs * steps_per_epoch, 1),
        eta_min=args.lr * 0.05,
    )

    rng = torch.Generator(device="cpu").manual_seed(args.seed)
    best_f1 = -1.0
    best_state: Dict[str, torch.Tensor] = {}
    history: List[Dict[str, float]] = []

    from tqdm import tqdm as _tqdm

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n_train, generator=rng)
        ep_loss = 0.0
        ep_tag_loss = 0.0
        ep_rate_loss = 0.0
        ep_people_loss = 0.0
        n_batches = 0
        n_steps = (n_train + args.batch_size - 1) // args.batch_size
        bar = _tqdm(
            range(0, n_train, args.batch_size),
            total=n_steps,
            desc=f"ep {epoch + 1}/{args.epochs}",
            leave=False,
            unit="step",
        )
        for start in bar:
            idx = perm[start : start + args.batch_size]
            feat = train_feats[idx]
            mh = train_mh[idx]
            rate = train_rate[idx]
            people = train_people[idx]
            tag_logits, rating_logits, people_logits = model(feat)
            l_tag, _per_group = compute_grouped_loss(tag_logits, mh, router)
            l_rate = ce(rating_logits, rate)
            loss = l_tag + args.lambda_rating * l_rate
            if ce_people is not None and people_logits is not None:
                l_people = ce_people(people_logits, people)
                loss = loss + args.lambda_people * l_people
                ep_people_loss += l_people.item()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            ep_loss += loss.item()
            ep_tag_loss += l_tag.item()
            ep_rate_loss += l_rate.item()
            n_batches += 1
            postfix = {
                "loss": f"{loss.item():.4f}",
                "tag": f"{l_tag.item():.4f}",
                "rate": f"{l_rate.item():.4f}",
            }
            if ce_people is not None and people_logits is not None:
                postfix["ppl"] = f"{l_people.item():.4f}"
            bar.set_postfix(**postfix)
        denom = max(n_batches, 1)
        avg_loss = ep_loss / denom
        avg_tag = ep_tag_loss / denom
        avg_rate = ep_rate_loss / denom
        avg_people = ep_people_loss / denom
        val_metrics = eval_split(
            model, val_feats, val_mh, val_rate,
            ce=ce, lambda_rating=args.lambda_rating, router=router,
            people_idx=val_people if ce_people is not None else None,
            ce_people=ce_people,
            lambda_people=args.lambda_people,
        )
        people_acc = val_metrics.get("people_acc", float("nan"))
        people_loss = val_metrics.get("val_people_loss", float("nan"))
        logger.info(
            "epoch %2d/%d  loss=%.4f (tag=%.4f rate=%.4f people=%.4f)  "
            "val_loss=%.4f (tag=%.4f rate=%.4f people=%.4f)  "
            "val_f1=%.4f  val_p=%.4f  val_r=%.4f  rate_acc=%.4f  people_acc=%.4f  lr=%.2e",
            epoch + 1, args.epochs,
            avg_loss, avg_tag, avg_rate, avg_people,
            val_metrics["val_loss"], val_metrics["val_tag_loss"],
            val_metrics["val_rate_loss"], people_loss,
            val_metrics["macro_f1"], val_metrics["macro_precision"],
            val_metrics["macro_recall"], val_metrics["rating_acc"],
            people_acc, sched.get_last_lr()[0],
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
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if not best_state:
        raise SystemExit("no epochs ran — empty training set?")

    ckpt_path = out_dir / "model.safetensors"
    cfg_path = out_dir / "config.json"
    history_path = out_dir / "train_history.json"
    st_save(best_state, str(ckpt_path))
    with open(cfg_path, "w") as f:
        json.dump(_save_cfg_dict(args, cfg, train_ds.d_in, best_f1), f, indent=2)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    plot_path = out_dir / "train_history.png"
    save_history_plot(history, plot_path)
    logger.info(
        "wrote %s / %s / %s / %s", ckpt_path, cfg_path, history_path, plot_path
    )
    print(f"  best val macro_f1: {best_f1:.4f}")


# ── pool_kind == "map" : bucket-grouped DataLoader path ──────────────────


@torch.no_grad()
def _eval_via_token_loader(
    model,
    loader,
    *,
    device,
    router: GroupRouter,
    ce: torch.nn.Module,
    ce_people: Optional[torch.nn.Module],
    lambda_rating: float,
    lambda_people: float,
    dual: bool = False,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Run val through a bucket-grouped token DataLoader and compute the
    same metrics ``eval_split`` produces. Logits are concatenated across
    buckets before metric reduction so macro-F1 / per-group accuracy are
    over the full val set.

    ``dual=True`` switches to the paired-token loader format
    ``(tokens, tokens_aux, mh, rate, people, bucket_pair)``.
    """
    model.eval()
    tag_chunks: List[torch.Tensor] = []
    rate_chunks: List[torch.Tensor] = []
    people_chunks: List[torch.Tensor] = []
    mh_chunks: List[torch.Tensor] = []
    rate_target_chunks: List[torch.Tensor] = []
    people_target_chunks: List[torch.Tensor] = []
    has_people_head = False
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for batch in loader:
            if dual:
                tokens, tokens_aux, mh, rate, people, _bucket = batch
                tokens_aux = tokens_aux.to(device, non_blocking=True)
            else:
                tokens, mh, rate, people, _bucket = batch
                tokens_aux = None
            tokens = tokens.to(device, non_blocking=True)
            mh_dev = mh.to(device, non_blocking=True)
            rate_dev = rate.to(device, non_blocking=True)
            people_dev = people.to(device, non_blocking=True)
            tl, rl, pl = model(tokens, tokens_aux) if dual else model(tokens)
            tag_chunks.append(tl.float())
            rate_chunks.append(rl.float())
            mh_chunks.append(mh_dev)
            rate_target_chunks.append(rate_dev)
            people_target_chunks.append(people_dev)
            if pl is not None:
                has_people_head = True
                people_chunks.append(pl.float())
    tag_logits = torch.cat(tag_chunks, dim=0)
    rating_logits = torch.cat(rate_chunks, dim=0)
    multi_hot = torch.cat(mh_chunks, dim=0)
    rating_idx = torch.cat(rate_target_chunks, dim=0)
    people_idx = torch.cat(people_target_chunks, dim=0)
    people_logits = torch.cat(people_chunks, dim=0) if has_people_head else None

    # Macro-F1 (residual tags only when softmax groups are active).
    if router.is_active() and router.softmax_member_indices is not None:
        keep_mask = torch.ones(
            tag_logits.shape[1], dtype=torch.bool, device=tag_logits.device
        )
        keep_mask[router.softmax_member_indices] = False
        kept_idx = keep_mask.nonzero(as_tuple=False).squeeze(1)
        f1_logits = tag_logits.index_select(1, kept_idx)
        f1_target = multi_hot.index_select(1, kept_idx)
    else:
        f1_logits = tag_logits
        f1_target = multi_hot
    pred = (f1_logits.sigmoid() > threshold).float()
    tp = (pred * f1_target).sum(dim=0)
    fp = (pred * (1 - f1_target)).sum(dim=0)
    fn = ((1 - pred) * f1_target).sum(dim=0)
    prec = tp / (tp + fp).clamp_min(1.0)
    rec = tp / (tp + fn).clamp_min(1.0)
    f1 = 2 * prec * rec / (prec + rec).clamp_min(1e-8)
    rating_acc = (rating_logits.argmax(dim=-1) == rating_idx).float().mean().item()

    val_l_tag, _ = compute_grouped_loss(tag_logits, multi_hot, router)
    val_l_rate = ce(rating_logits, rating_idx)
    val_l_total = val_l_tag + lambda_rating * val_l_rate
    out = {
        "macro_f1": f1.mean().item(),
        "macro_precision": prec.mean().item(),
        "macro_recall": rec.mean().item(),
        "rating_acc": rating_acc,
        "val_tag_loss": val_l_tag.item(),
        "val_rate_loss": val_l_rate.item(),
    }
    if ce_people is not None and people_logits is not None:
        val_l_people = ce_people(people_logits, people_idx)
        val_l_total = val_l_total + lambda_people * val_l_people
        out["val_people_loss"] = val_l_people.item()
        out["people_acc"] = (
            (people_logits.argmax(dim=-1) == people_idx).float().mean().item()
        )
    out["val_loss"] = val_l_total.item()

    # Per-softmax-group argmax accuracy.
    if router.is_active():
        solo_mask = router.solo_mask(multi_hot)
        for g in router.softmax_groups:
            if g.escape_indices.numel() > 0:
                has_escape = multi_hot.index_select(1, g.escape_indices).any(dim=1)
            else:
                has_escape = torch.zeros_like(solo_mask)
            applicable = (
                (solo_mask & ~has_escape) if g.mode == "softmax_when_solo" else ~has_escape
            )
            gl = tag_logits.index_select(1, g.tag_indices)
            gt = multi_hot.index_select(1, g.tag_indices)
            has_label = gt.sum(dim=1) > 0
            keep = applicable & has_label
            n_keep = int(keep.sum().item())
            if n_keep == 0:
                out[f"acc_{g.name}"] = 0.0
                out[f"n_{g.name}"] = 0
                continue
            pred_idx = gl[keep].argmax(dim=1)
            true_idx = gt[keep].argmax(dim=1)
            out[f"acc_{g.name}"] = (pred_idx == true_idx).float().mean().item()
            out[f"n_{g.name}"] = n_keep
    return out


def _train_cached_map(args: argparse.Namespace) -> None:
    """Single-encoder MAP-pool training (no aux). The dual-encoder path (any
    pool_kind combination) is handled by :func:`_train_cached_dual`.
    """
    from safetensors.torch import save_file as st_save
    from torch.utils.data import DataLoader

    from library.captioning.anima_tagger_data import (
        BucketBatchSampler,
        CachedDualDataset,
        CachedTokenDataset,
        TaggerManifest,
        collate_dual_token_batch,
        collate_token_batch,
    )
    from library.captioning.anima_tagger_model import AnimaTaggerHead
    from library.vision.encoders import get_encoder_info

    out_dir = Path(args.out_dir)
    manifest_path = out_dir / "dataset.json"
    vocab_path = out_dir / "vocab.json"
    cache_dir = cache_dir_for(out_dir, args.pool_kind, args.encoder)
    if not manifest_path.exists():
        raise SystemExit(f"missing {manifest_path} — run --mode build_vocab first.")
    if not vocab_path.exists():
        raise SystemExit(f"missing {vocab_path} — run --mode build_vocab first.")
    if not cache_dir.exists():
        raise SystemExit(
            f"missing {cache_dir} — run --mode build_features "
            f"--pool_kind={args.pool_kind} --encoder {args.encoder} first."
        )
    manifest = TaggerManifest.from_path(manifest_path)
    with open(vocab_path) as f:
        vocab_dict = json.load(f)
    spec = get_encoder_info(args.encoder).bucket_spec

    aux_encoder = getattr(args, "aux_encoder", None)
    dual = bool(aux_encoder)
    if dual:
        # Per-side pool_kind. None inherits main → preserves the dual-MAP
        # default. Aux cache_dir is keyed on its own pool_kind so mixed
        # configs (mean main + map aux) read from .cache/tokens-pe_spatial/
        # while the main side reads from .cache/tokens-pe/.
        pool_kind_aux = args.pool_kind_aux or args.pool_kind
        spec_aux = (
            get_encoder_info(aux_encoder).bucket_spec
            if pool_kind_aux == "map" else None
        )
        cache_dir_aux = cache_dir_for(out_dir, pool_kind_aux, aux_encoder)
        if not cache_dir_aux.exists():
            raise SystemExit(
                f"missing aux cache {cache_dir_aux} — run "
                f"`--mode build_features --pool_kind={args.pool_kind} "
                f"--encoder {args.encoder} --aux_encoder {aux_encoder} "
                f"--pool_kind_aux {pool_kind_aux}` first."
            )
        train_ds = CachedDualDataset(
            manifest, cache_dir, args.pool_kind, spec,
            cache_dir_aux, pool_kind_aux, spec_aux,
            stems_subset=manifest.train_stems,
        )
        val_ds = CachedDualDataset(
            manifest, cache_dir, args.pool_kind, spec,
            cache_dir_aux, pool_kind_aux, spec_aux,
            stems_subset=manifest.val_stems,
        )
        d_in_aux = train_ds.d_in_aux
    else:
        train_ds = CachedTokenDataset(
            manifest, cache_dir, spec, stems_subset=manifest.train_stems
        )
        val_ds = CachedTokenDataset(
            manifest, cache_dir, spec, stems_subset=manifest.val_stems
        )
        d_in_aux = None
    logger.info(
        "train (cached map): N=%d  val: N=%d  d_in=%d%s  n_tags=%d  n_ratings=%d  n_people=%d",
        len(train_ds), len(val_ds), train_ds.d_in,
        f"  d_in_aux={d_in_aux}" if dual else "",
        train_ds.n_tags, train_ds.n_ratings, train_ds.n_people_counts,
    )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    routing: Optional[Tuple[List[int], List[int]]] = None
    if getattr(args, "use_per_head_routing", False):
        if not dual:
            raise SystemExit(
                "--use_per_head_routing requires --aux_encoder (the per-head "
                "soft gate needs an aux trunk to mix in)."
            )
        routing = _routing_indices_from_vocab(vocab_dict, train_ds.n_tags)
        logger.info(
            "per-head routing: %d main-lean tags (character/copyright/artist/count) + "
            "%d aux-lean tags (general/metadata/deprecated)",
            len(routing[0]), len(routing[1]),
        )
    cfg = _make_cfg_from_args(
        args,
        d_in=train_ds.d_in,
        n_tags=train_ds.n_tags,
        n_ratings=train_ds.n_ratings,
        n_people_counts=train_ds.n_people_counts,
        d_in_aux=d_in_aux,
        routing=routing,
    )
    model = AnimaTaggerHead(cfg).to(device)
    logger.info(
        "head: pool_kind=%s n_queries=%d n_heads=%d use_cls=%s use_mean=%s "
        "trunk_in=%s d_hidden=%d%s",
        cfg.pool_kind, cfg.pool_n_queries, cfg.pool_n_heads,
        cfg.pool_use_cls, cfg.pool_use_mean,
        (
            f"main={cfg.main_trunk_in_dim}/aux={cfg.aux_trunk_in_dim} (routed)"
            if cfg.use_per_head_routing else str(cfg.trunk_in_dim)
        ),
        cfg.d_hidden,
        (
            f"  aux: n_q={cfg.pool_n_queries_aux} n_h={cfg.pool_n_heads_aux} "
            f"use_cls={cfg.pool_use_cls_aux} use_mean={cfg.pool_use_mean_aux}"
            if dual else ""
        ),
    )

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
    collate_fn = collate_dual_token_batch if dual else collate_token_batch
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        num_workers=args.feature_cache_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_sampler=val_sampler,
        num_workers=args.feature_cache_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        fused=torch.cuda.is_available(),
    )
    sched = build_warmup_cosine_scheduler(
        opt,
        warmup_steps=int(getattr(args, "warmup_steps", 0)),
        total_steps=max(args.epochs * len(train_loader), 1),
        eta_min=args.lr * 0.05,
    )

    best_f1 = -1.0
    best_state: Dict[str, torch.Tensor] = {}
    history: List[Dict[str, float]] = []

    from tqdm import tqdm as _tqdm

    postfix_every = max(1, int(getattr(args, "postfix_every", 10)))

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
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
        for step, batch in enumerate(bar):
            if dual:
                tokens, tokens_aux, mh_cpu, rate_cpu, people_cpu, _bucket = batch
                tokens_aux = tokens_aux.to(device, non_blocking=True)
            else:
                tokens, mh_cpu, rate_cpu, people_cpu, _bucket = batch
                tokens_aux = None
            tokens = tokens.to(device, non_blocking=True)
            mh = mh_cpu.to(device, non_blocking=True)
            rate = rate_cpu.to(device, non_blocking=True)
            people = people_cpu.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                tag_logits, rating_logits, people_logits = (
                    model(tokens, tokens_aux) if dual else model(tokens)
                )
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
        val_metrics = _eval_via_token_loader(
            model, val_loader,
            device=device, router=router,
            ce=ce, ce_people=ce_people,
            lambda_rating=args.lambda_rating,
            lambda_people=args.lambda_people,
            dual=dual,
        )
        people_acc = val_metrics.get("people_acc", float("nan"))
        people_loss = val_metrics.get("val_people_loss", float("nan"))
        logger.info(
            "epoch %2d/%d  loss=%.4f (tag=%.4f rate=%.4f people=%.4f)  "
            "val_loss=%.4f (tag=%.4f rate=%.4f people=%.4f)  "
            "val_f1=%.4f  val_p=%.4f  val_r=%.4f  rate_acc=%.4f  people_acc=%.4f  lr=%.2e",
            epoch + 1, args.epochs,
            avg_loss, avg_tag, avg_rate, avg_people,
            val_metrics["val_loss"], val_metrics["val_tag_loss"],
            val_metrics["val_rate_loss"], people_loss,
            val_metrics["macro_f1"], val_metrics["macro_precision"],
            val_metrics["macro_recall"], val_metrics["rating_acc"],
            people_acc, sched.get_last_lr()[0],
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
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if not best_state:
        raise SystemExit("no epochs ran — empty training set?")

    ckpt_path = out_dir / "model.safetensors"
    cfg_path = out_dir / "config.json"
    history_path = out_dir / "train_history.json"
    st_save(best_state, str(ckpt_path))
    with open(cfg_path, "w") as f:
        json.dump(_save_cfg_dict(args, cfg, train_ds.d_in, best_f1), f, indent=2)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    plot_path = out_dir / "train_history.png"
    save_history_plot(history, plot_path)
    logger.info(
        "wrote %s / %s / %s / %s", ckpt_path, cfg_path, history_path, plot_path
    )
    print(f"  best val macro_f1: {best_f1:.4f}")


def cmd_train_cached(args: argparse.Namespace) -> None:
    """Frozen-encoder path. Dispatches on ``--pool_kind`` and ``--aux_encoder``.

    * single encoder, mean pool → :func:`_train_cached_mean` (in-VRAM tensors)
    * single encoder, map pool → :func:`_train_cached_map` (bucket DataLoader)
    * dual encoder (any per-side pool kind) → also :func:`_train_cached_map`,
      which now consumes :class:`CachedDualDataset` and dispatches per-side
      cache layout. The "_map" name is historical; the function is the
      bucket-DataLoader path generally.
    """
    aux_encoder = getattr(args, "aux_encoder", None)
    if aux_encoder:
        _train_cached_map(args)
    elif args.pool_kind == "mean":
        _train_cached_mean(args)
    elif args.pool_kind == "map":
        _train_cached_map(args)
    else:
        raise SystemExit(f"unknown --pool_kind={args.pool_kind!r}")
