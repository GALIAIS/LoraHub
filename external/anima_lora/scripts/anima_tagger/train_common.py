"""Shared training helpers — loss weights, eval, history plot, group router.

Used by both ``train_cached.py`` (frozen-encoder fast path) and
``train_pe_lora.py`` (end-to-end PE-LoRA path). The two paths differ in
how they get features — one reads pre-pooled tensors, one runs PE-Core
each step — but they share loss formulation, per-epoch eval, and the
group-routing logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .constants import _COUNT_RE


def pos_weight_sqrt(multi_hot: torch.Tensor) -> torch.Tensor:
    """``sqrt(n_neg / n_pos)`` per tag — softens BCE long-tail without overshoot."""
    n_pos = multi_hot.sum(dim=0).clamp_min(1.0)
    n_neg = multi_hot.shape[0] - n_pos
    return torch.sqrt(n_neg / n_pos)


def build_warmup_cosine_scheduler(
    opt: torch.optim.Optimizer,
    *,
    warmup_steps: int,
    total_steps: int,
    eta_min: float,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Linear warmup (1e-3 → 1.0) then cosine decay to ``eta_min``.

    Stepped per batch — callers must call ``sched.step()`` after every
    ``opt.step()``, not per epoch. When ``warmup_steps == 0`` returns a
    plain ``CosineAnnealingLR`` (drop-in for the legacy schedule, just on
    a per-step cadence).
    """
    if total_steps <= 0:
        raise ValueError(f"total_steps must be > 0, got {total_steps}")
    if not 0 <= warmup_steps < total_steps:
        raise ValueError(
            f"warmup_steps must be in [0, total_steps); got "
            f"warmup_steps={warmup_steps} total_steps={total_steps}"
        )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=total_steps - warmup_steps, eta_min=eta_min
    )
    if warmup_steps == 0:
        return cosine
    warmup = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps
    )
    return torch.optim.lr_scheduler.SequentialLR(
        opt, schedulers=[warmup, cosine], milestones=[warmup_steps]
    )


def rating_class_weights(rating_idx: torch.Tensor, n_ratings: int) -> torch.Tensor:
    """Inverse-frequency weights, normalized so mean weight = 1."""
    counts = torch.bincount(rating_idx, minlength=n_ratings).float().clamp_min(1.0)
    inv = 1.0 / counts
    inv = inv * n_ratings / inv.sum()
    return inv


def people_class_weights(people_idx: torch.Tensor, n_people: int) -> torch.Tensor:
    """Sqrt-inverse-frequency weights for the people-count head.

    Sqrt-inverse (vs the rating head's straight inverse) is the right choice
    here because the people-count distribution is much heavier-tailed than
    rating: ``1girl`` typically dominates 60–80 % of an anime corpus while
    ``1girl_1boy`` / ``2girls_1boy`` may sit at <2 %. Inverse-frequency would
    weight rare classes ~30× the dominant one and make CE volatile. Sqrt
    softens that to ~5×, comparable to ``pos_weight_sqrt`` for the BCE head.
    Normalized so mean weight = 1 (same convention as ``rating_class_weights``).
    """
    counts = torch.bincount(people_idx, minlength=n_people).float().clamp_min(1.0)
    inv = 1.0 / counts.sqrt()
    inv = inv * n_people / inv.sum()
    return inv


# ── Group routing ─────────────────────────────────────────────────────────


@dataclass
class _SoftmaxGroup:
    """One softmax group projected onto trainer-side tensor indices."""

    name: str
    mode: str                    # "softmax_when_solo" | "softmax"
    tag_indices: torch.Tensor    # LongTensor [K_g]
    escape_indices: torch.Tensor # LongTensor [E_g]


@dataclass
class GroupRouter:
    """Per-batch loss routing for typed tag groups.

    Built once at trainer init from ``vocab.json[groups]``. Maintains:

    * ``bce_pos_weight`` — full ``[n_tags]`` pos-weight, same as the
      pre-grouping trainer. BCE applies to all tags by default; the
      :func:`compute_grouped_loss` helper masks out (sample, tag)
      positions where CE fires for that sample-group pair.
    * ``softmax_groups`` — per-group ``(mode, tag_indices, escape_indices)``;
      CE applies on these, gated by solo/escape for ``softmax_when_solo``.
    * ``softmax_member_indices`` — union of all softmax-group tag indices.
      Used by :func:`eval_split` and the calibrator to skip those tags
      from sigmoid-threshold F1 / threshold sweep (they're argmax-only at
      inference, so per-tag thresholds don't apply).
    * ``solo_indices`` / ``multi_indices`` — vocab indices used to detect
      single-subject samples at runtime from ``multi_hot``.

    A vocab built without ``--groups`` produces an empty router: BCE
    applies everywhere and behavior matches the pre-grouping trainer
    exactly.
    """

    n_tags: int
    bce_pos_weight: torch.Tensor           # FloatTensor [n_tags]
    softmax_groups: List[_SoftmaxGroup] = field(default_factory=list)
    softmax_member_indices: Optional[torch.Tensor] = None  # LongTensor [Σ K_g]
    solo_indices: Optional[torch.Tensor] = None    # LongTensor [s]
    multi_indices: Optional[torch.Tensor] = None   # LongTensor [m]

    @classmethod
    def from_vocab(
        cls,
        vocab_dict: Dict,
        train_multi_hot: torch.Tensor,
        device: torch.device,
    ) -> "GroupRouter":
        """Build the router from a vocab dict + the train split's multi_hot."""
        n_tags = int(train_multi_hot.shape[1])
        groups_raw: List[Dict] = list(vocab_dict.get("groups") or [])

        # Collect the union of softmax-group tag indices.
        softmax_member: List[int] = []
        softmax_groups: List[_SoftmaxGroup] = []
        for g in groups_raw:
            mode = g["mode"]
            if mode in ("softmax_when_solo", "softmax"):
                idxs = list(g.get("tag_indices") or [])
                esc = list(g.get("escape_indices") or [])
                if not idxs:
                    continue
                softmax_member.extend(idxs)
                softmax_groups.append(
                    _SoftmaxGroup(
                        name=str(g["name"]),
                        mode=mode,
                        tag_indices=torch.tensor(idxs, dtype=torch.long, device=device),
                        escape_indices=torch.tensor(
                            esc, dtype=torch.long, device=device
                        ),
                    )
                )
            # multilabel groups are documentation-only — they stay in BCE.

        softmax_member_indices = (
            torch.tensor(sorted(set(softmax_member)), dtype=torch.long, device=device)
            if softmax_member else None
        )

        # Full-vocab pos-weight (matches pre-grouping trainer). BCE
        # applies to every tag by default; per-batch masking knocks out
        # the (sample, group_tag) positions that CE supervises instead.
        bce_pos_weight = pos_weight_sqrt(train_multi_hot).to(device)

        # Solo/multi vocab-index sets, derived from the tag names. Vocab
        # tags use canonical space form; the regex in ``constants`` matches
        # both. ``solo`` is a non-count membership tag — gelcrawl writes
        # it alongside ``1girl``/``1boy`` when there's exactly one figure.
        single_count_names = {"solo", "1girl", "1boy", "1other"}
        solo_idx_list: List[int] = []
        multi_idx_list: List[int] = []
        for t in vocab_dict.get("tags", []):
            name = t["name"]
            idx = int(t["index"])
            if name in single_count_names:
                solo_idx_list.append(idx)
            elif _COUNT_RE.match(name):
                # Any count-tag name that isn't in the single-count set
                # (e.g. 2girls, 3boys, multiple_girls).
                multi_idx_list.append(idx)
        solo_indices = (
            torch.tensor(solo_idx_list, dtype=torch.long, device=device)
            if solo_idx_list else None
        )
        multi_indices = (
            torch.tensor(multi_idx_list, dtype=torch.long, device=device)
            if multi_idx_list else None
        )

        return cls(
            n_tags=n_tags,
            bce_pos_weight=bce_pos_weight,
            softmax_groups=softmax_groups,
            softmax_member_indices=softmax_member_indices,
            solo_indices=solo_indices,
            multi_indices=multi_indices,
        )

    def is_active(self) -> bool:
        return bool(self.softmax_groups)

    def solo_mask(self, multi_hot: torch.Tensor) -> torch.Tensor:
        """``[B] bool`` — True when sample is single-subject.

        Single-subject = at least one of ``solo``/``1girl``/``1boy``/``1other``
        fires AND no other count tag (``2+girls``, ``multiple_*``, …)
        fires. When the vocab carries no count tags at all (degenerate),
        every sample is treated as solo so the trainer doesn't silently
        skip every CE update.
        """
        B = multi_hot.shape[0]
        if self.solo_indices is None:
            # No solo signal in the vocab. Be permissive — assume solo.
            return torch.ones(B, dtype=torch.bool, device=multi_hot.device)
        has_single = multi_hot[:, self.solo_indices].any(dim=1)
        if self.multi_indices is None:
            return has_single
        has_multi = multi_hot[:, self.multi_indices].any(dim=1)
        return has_single & ~has_multi


def compute_grouped_loss(
    tag_logits: torch.Tensor,    # [B, n_tags]
    multi_hot: torch.Tensor,     # [B, n_tags]
    router: GroupRouter,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Return ``(total_tag_loss, per_group_metrics_for_logging)``.

    BCE applies element-wise across all (sample, tag) positions. For
    each softmax group, samples where (gating allows AND a group label
    fires) get K-way CE on the group's logits — and BCE for those
    (sample, group_tag) positions is masked out so we don't double-count
    supervision. Samples without that gate (multi-subject, escape, no
    in-group label) keep BCE on the group's tags as a fallback.

    Returned metrics: ``"bce"`` (mean of unmasked BCE entries) plus
    ``f"ce_{group_name}"`` for each softmax group; loss curves stay
    separable in TensorBoard.
    """
    B, n_tags = tag_logits.shape
    metrics: Dict[str, float] = {}

    # Element-wise BCE-with-logits — we'll mask and reduce manually.
    bce_per_elem = F.binary_cross_entropy_with_logits(
        tag_logits,
        multi_hot,
        pos_weight=router.bce_pos_weight,
        reduction="none",
    )

    # Default: BCE applies to every position. CE-supervised positions
    # get masked off below.
    bce_mask = torch.ones(
        B, n_tags, dtype=torch.bool, device=tag_logits.device
    )

    ce_total = tag_logits.new_zeros(())
    if router.softmax_groups:
        solo_mask = router.solo_mask(multi_hot)
        for g in router.softmax_groups:
            if g.escape_indices.numel() > 0:
                has_escape = multi_hot.index_select(1, g.escape_indices).any(dim=1)
            else:
                has_escape = torch.zeros_like(solo_mask)
            if g.mode == "softmax_when_solo":
                applicable = solo_mask & ~has_escape
            else:                                                # "softmax"
                applicable = ~has_escape

            group_logits = tag_logits.index_select(1, g.tag_indices)     # [B, K_g]
            group_target = multi_hot.index_select(1, g.tag_indices)      # [B, K_g]
            has_label = group_target.sum(dim=1) > 0
            ce_samples = applicable & has_label
            n_keep = int(ce_samples.sum().item())
            if n_keep == 0:
                metrics[f"ce_{g.name}"] = 0.0
                continue
            sel_logits = group_logits[ce_samples]                        # [n_keep, K_g]
            sel_target = group_target[ce_samples].argmax(dim=1)          # [n_keep]
            l_ce = F.cross_entropy(sel_logits, sel_target)
            ce_total = ce_total + l_ce
            metrics[f"ce_{g.name}"] = float(l_ce.detach().item())

            # Mask BCE for the supervised (sample, group_tag) cells.
            # Broadcasted indexing: ``mask[ce_idx[:, None], tag_idx[None, :]]``
            # touches the cartesian product.
            ce_idx = ce_samples.nonzero(as_tuple=False).squeeze(1)
            bce_mask[ce_idx[:, None], g.tag_indices[None, :]] = False

    bce_count = bce_mask.sum().clamp_min(1.0)
    l_bce = (bce_per_elem * bce_mask.float()).sum() / bce_count
    metrics["bce"] = float(l_bce.detach().item())

    return l_bce + ce_total, metrics


@torch.no_grad()
def eval_split(
    model: torch.nn.Module,
    feats: torch.Tensor,
    multi_hot: torch.Tensor,
    rating_idx: torch.Tensor,
    threshold: float = 0.5,
    ce: Optional[torch.nn.Module] = None,
    lambda_rating: float = 0.0,
    router: Optional[GroupRouter] = None,
    people_idx: Optional[torch.Tensor] = None,
    ce_people: Optional[torch.nn.Module] = None,
    lambda_people: float = 0.0,
) -> Dict[str, float]:
    """Macro-F1 over tags + rating accuracy + people-count accuracy.

    When ``router`` carries softmax groups, F1 is computed over residual
    (BCE-supervised) tags only — softmax-group tags are evaluated by
    per-group argmax accuracy instead, since their sigmoid scores are
    untrained noise. The rating-CE / people-CE losses are reported when
    their CE modules are given. The combined val tag-loss matches the
    training objective: ``residual BCE + Σ ce_per_softmax_group``.

    People-head metrics (``people_acc`` / ``val_people_loss``) are
    reported only when the model has a people head (i.e. ``forward``
    returned a third tensor) AND ``people_idx`` is supplied. Backwards-
    compatible: omitting these arguments matches the pre-people behavior
    exactly.
    """
    model.eval()
    tag_logits, rating_logits, people_logits = model(feats)

    if router is not None and router.is_active() and router.softmax_member_indices is not None:
        # Macro-F1 excludes softmax-group tags — those are argmax-only at
        # inference, so per-tag thresholds (and the F1 they induce) don't
        # apply. Per-group accuracy is reported separately below.
        keep_mask = torch.ones(tag_logits.shape[1], dtype=torch.bool, device=tag_logits.device)
        keep_mask[router.softmax_member_indices] = False
        kept_idx = keep_mask.nonzero(as_tuple=False).squeeze(1)
        f1_logits = tag_logits.index_select(1, kept_idx)
        f1_target = multi_hot.index_select(1, kept_idx)
        pred = (f1_logits.sigmoid() > threshold).float()
        tp = (pred * f1_target).sum(dim=0)
        fp = (pred * (1 - f1_target)).sum(dim=0)
        fn = ((1 - pred) * f1_target).sum(dim=0)
    else:
        pred = (tag_logits.sigmoid() > threshold).float()
        tp = (pred * multi_hot).sum(dim=0)
        fp = (pred * (1 - multi_hot)).sum(dim=0)
        fn = ((1 - pred) * multi_hot).sum(dim=0)
    prec = tp / (tp + fp).clamp_min(1.0)
    rec = tp / (tp + fn).clamp_min(1.0)
    f1 = 2 * prec * rec / (prec + rec).clamp_min(1e-8)
    rating_pred = rating_logits.argmax(dim=-1)
    rating_acc = (rating_pred == rating_idx).float().mean().item()
    out = {
        "macro_f1": f1.mean().item(),
        "macro_precision": prec.mean().item(),
        "macro_recall": rec.mean().item(),
        "rating_acc": rating_acc,
    }

    # Per-group argmax accuracy (only counts samples where the group's
    # gating applies AND the label is present).
    if router is not None and router.is_active():
        solo_mask = router.solo_mask(multi_hot)
        for g in router.softmax_groups:
            if g.escape_indices.numel() > 0:
                has_escape = multi_hot.index_select(1, g.escape_indices).any(dim=1)
            else:
                has_escape = torch.zeros_like(solo_mask)
            applicable = (solo_mask & ~has_escape) if g.mode == "softmax_when_solo" else ~has_escape
            group_logits = tag_logits.index_select(1, g.tag_indices)
            group_target = multi_hot.index_select(1, g.tag_indices)
            has_label = group_target.sum(dim=1) > 0
            keep = applicable & has_label
            n_keep = int(keep.sum().item())
            if n_keep == 0:
                out[f"acc_{g.name}"] = 0.0
                out[f"n_{g.name}"] = 0
                continue
            pred_idx = group_logits[keep].argmax(dim=1)
            true_idx = group_target[keep].argmax(dim=1)
            acc = (pred_idx == true_idx).float().mean().item()
            out[f"acc_{g.name}"] = acc
            out[f"n_{g.name}"] = n_keep

    # People-head accuracy (independent of the CE-loss reporting branch).
    if people_logits is not None and people_idx is not None:
        people_pred = people_logits.argmax(dim=-1)
        out["people_acc"] = (people_pred == people_idx).float().mean().item()

    if ce is not None:
        l_rate = ce(rating_logits, rating_idx)
        if router is not None:
            l_tag, _per_group = compute_grouped_loss(tag_logits, multi_hot, router)
        else:
            # Backwards-compat path: caller didn't pass a router. Skip the
            # tag-loss reporting since BCE alone wouldn't match training.
            l_tag = tag_logits.new_zeros(())
        out["val_tag_loss"] = l_tag.item()
        out["val_rate_loss"] = l_rate.item()
        l_total = l_tag + lambda_rating * l_rate
        if (
            ce_people is not None
            and people_logits is not None
            and people_idx is not None
        ):
            l_people = ce_people(people_logits, people_idx)
            out["val_people_loss"] = l_people.item()
            l_total = l_total + lambda_people * l_people
        out["val_loss"] = l_total.item()
    return out


def save_history_plot(history: List[Dict[str, float]], path: Path) -> None:
    """Two-panel matplotlib figure: loss curves on top, val F1 / rating-acc below.

    Tolerates missing keys so it works for both the cached and PE-LoRA paths
    and any partial future variants.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, (ax_loss, ax_acc) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

    ax_loss.plot(epochs, [h["loss"] for h in history], label="train total", color="C0")
    if all("tag_loss" in h for h in history):
        ax_loss.plot(
            epochs, [h["tag_loss"] for h in history],
            label="train tag (BCE)", color="C0", alpha=0.4, linestyle=":",
        )
        ax_loss.plot(
            epochs, [h["rate_loss"] for h in history],
            label="train rate (CE)", color="C0", alpha=0.4, linestyle="--",
        )
    if all("val_loss" in h for h in history):
        ax_loss.plot(
            epochs, [h["val_loss"] for h in history], label="val total", color="C1"
        )
        if all("val_tag_loss" in h for h in history):
            ax_loss.plot(
                epochs, [h["val_tag_loss"] for h in history],
                label="val tag (BCE)", color="C1", alpha=0.4, linestyle=":",
            )
            ax_loss.plot(
                epochs, [h["val_rate_loss"] for h in history],
                label="val rate (CE)", color="C1", alpha=0.4, linestyle="--",
            )
        if all("val_people_loss" in h for h in history):
            ax_loss.plot(
                epochs, [h["val_people_loss"] for h in history],
                label="val people (CE)", color="C4", alpha=0.4, linestyle="--",
            )
    ax_loss.set_ylabel("loss")
    ax_loss.legend(loc="best", fontsize=8)
    ax_loss.grid(alpha=0.3)

    ax_acc.plot(
        epochs, [h["macro_f1"] for h in history], label="val macro F1", color="C2"
    )
    ax_acc.plot(
        epochs, [h["rating_acc"] for h in history], label="val rating acc", color="C3"
    )
    if all("people_acc" in h for h in history):
        ax_acc.plot(
            epochs, [h["people_acc"] for h in history],
            label="val people acc", color="C4",
        )
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("metric")
    ax_acc.set_ylim(0, 1)
    ax_acc.legend(loc="best", fontsize=8)
    ax_acc.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
