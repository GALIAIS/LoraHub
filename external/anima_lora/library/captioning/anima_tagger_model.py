"""Anima tagger head — multi-label tags + 3-class rating + 8-class people-count, off frozen PE.

Each encoder side independently picks ``"mean"`` (consume a pre-pooled
``[B, D]`` feature) or ``"map"`` (consume ``[B, T, D]`` tokens via a learned
``MAPHead`` + optional CLS / mean concat). The two sides are configured
separately via ``pool_kind`` (main) and ``pool_kind_aux`` (aux), so e.g.
PE-Core can ride a cheap mean pool while PE-Spatial gets the full MAP
treatment for spatial detail.

When ``use_per_head_routing=True`` (dual-encoder only), the head replaces
the single concat-trunk with two parallel projection trunks (main-only
and aux-only, both → ``[B, d_hidden]``) plus a learnable scalar gate per
output head. Each output head fuses ``α · h_main + (1 − α) · h_aux`` with
``α = sigmoid(gate)``; the gate is initialized to bias each head toward
its "natural" encoder (PE-Core for identity-class tags / rating / people-
count, PE-Spatial for general / metadata / deprecated). The multi-label
``tag_head`` is split into two sub-heads (main-lean over ``tag_indices_main``
and aux-lean over ``tag_indices_aux``) and the two outputs are scattered
back into a full ``[B, n_tags]`` tensor in vocab order so the downstream
loss / threshold paths see the same shape as the legacy architecture.

Architecture (dual-encoder, ``pool_kind="map"`` + ``pool_kind_aux="map"``):

::

    main tokens [T_m, d_in]                         # PE-Core patch tokens, CLS at [0]
        ├─ MAPHead(K queries, H heads)  → [K, d_in]
        ├─ CLS  = tokens[:, 0]          → [1, d_in]
        └─ mean = tokens.mean(dim=1)    → [1, d_in]
              concat → [(K+use_cls+use_mean) * d_in]

    aux tokens  [T_a, d_in_aux]   (optional)        # PE-Spatial patch tokens
        ├─ MAPHead(K_a queries, H_a heads) → [K_a, d_in_aux]
        ├─ CLS  = tokens[:, 0]          → [1, d_in_aux]
        └─ mean = tokens.mean(dim=1)    → [1, d_in_aux]
              concat → [(K_a+use_cls_aux+use_mean_aux) * d_in_aux]

    [main_pool ‖ aux_pool] → LayerNorm + Linear(trunk_in_dim, d_hidden) + GELU + Dropout
    trunk_h [d_hidden]
        ├─→ Linear(d_hidden, n_tags)          → tag_logits
        ├─→ Linear(d_hidden, n_ratings)       → rating_logits
        └─→ Linear(d_hidden, n_people_counts) → people_logits  (omitted when n_people_counts == 0)

Mixed example (``pool_kind="mean"`` + ``pool_kind_aux="map"``): main side
contributes a single ``[d_in]`` channel (no MAPHead, no CLS / mean concat —
the cached feature *is* the mean pool), aux side gets the full MAP +
CLS / mean concat. ``trunk_in_dim`` becomes ``d_in + d_in_aux *
(K_a + use_cls_aux + use_mean_aux)``.

The aux encoder is **opt-in** via ``d_in_aux`` — when None the head is
single-encoder (PE-Core only) and the second forward arg must be omitted.
This preserves backward-compat with anima-tagger-v1 checkpoints whose
``config.json`` lacks the aux fields entirely. ``pool_kind_aux`` defaults
to ``None`` which inherits ``pool_kind`` — the dual-MAP path therefore
loads with no extra config keys.

The trunk is shared between heads so the auxiliary signals (rating /
people-count) nudge the same representation that's predicting tags.
``n_tags``/``n_ratings``/``n_people_counts``/``d_in``/``d_in_aux`` all come
from ``vocab.json`` + the cached PE token dimension (always ``d_enc``, not
the pooled feature dim — pool-output dim is derived from ``d_in`` × the
active pool channels).

Inference receives all heads in one forward; training computes per-head
losses and combines with ``λ_rating`` / ``λ_people``. ``n_people_counts=0``
in the config means "no people head was trained" — used to load legacy
checkpoints; ``forward`` returns ``None`` in that slot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class AnimaTaggerConfig:
    d_in: int                        # PE token dim (d_enc), not the pool-output dim.
    n_tags: int
    n_ratings: int = 3
    # 0 = no people head (legacy checkpoint). Trainer always sets this from
    # the manifest (currently len(library.captioning.anima_tagger.PEOPLE_COUNT_LABELS) == 8) when in use.
    n_people_counts: int = 0
    d_hidden: int = 1024
    dropout: float = 0.1
    # Main encoder pool. ``"mean"`` (legacy) consumes a pre-pooled [B, d_in]
    # feature; ``"map"`` consumes a [B, T, d_in] token sequence and runs
    # MAPHead + (optional) CLS + (optional) mean inside the head. Default
    # is "mean" so legacy config.json files load unchanged.
    pool_kind: str = "mean"
    pool_n_queries: int = 4
    pool_n_heads: int = 8
    pool_use_cls: bool = True
    pool_use_mean: bool = True
    # Optional auxiliary encoder (e.g. PE-Spatial-B16-512). When d_in_aux
    # is None the head is single-encoder.
    d_in_aux: Optional[int] = None
    # Aux pool kind. ``None`` inherits ``pool_kind`` (so dual-MAP and dual-mean
    # configs don't need an extra key). Setting it explicitly lets the two
    # sides differ — e.g. main="mean" + aux="map" pays the MAP cost only on
    # the encoder where spatial detail matters.
    pool_kind_aux: Optional[str] = None
    pool_n_queries_aux: int = 4
    pool_n_heads_aux: int = 8
    pool_use_cls_aux: bool = True
    pool_use_mean_aux: bool = True
    # Per-head routing — split tag_head by category, gate each output head
    # between main / aux trunks. Requires dual encoder. When False, the head
    # collapses to the legacy concat-trunk + single tag_head path.
    use_per_head_routing: bool = False
    # Vocab indices routed to the main-leaning tag sub-head (typically
    # character / copyright / artist / count). The aux-leaning sub-head
    # receives the complement. Together they MUST partition [0, n_tags).
    tag_indices_main: List[int] = field(default_factory=list)
    tag_indices_aux: List[int] = field(default_factory=list)
    # Sigmoid-of-scalar gate init biases. +2.0 → α ≈ 0.88 (main-lean),
    # −2.0 → α ≈ 0.12 (aux-lean). The trainer can override but the model
    # ignores these after init (they're a head-by-head prior, not a runtime
    # knob).
    gate_init_bias_main: float = 2.0
    gate_init_bias_aux: float = -2.0

    @property
    def effective_pool_kind_aux(self) -> str:
        """Resolved aux pool kind. Mirrors ``pool_kind`` when unset."""
        return self.pool_kind_aux or self.pool_kind

    def _validate_routing(self) -> None:
        """Sanity-check routing fields when ``use_per_head_routing`` is on."""
        if not self.use_per_head_routing:
            return
        if not self.has_aux:
            raise ValueError(
                "use_per_head_routing=True requires d_in_aux (dual encoder) — "
                "the per-head gate has no aux trunk to mix in otherwise."
            )
        main = list(self.tag_indices_main)
        aux = list(self.tag_indices_aux)
        if not main and not aux:
            raise ValueError(
                "use_per_head_routing=True requires tag_indices_main / "
                "tag_indices_aux to be set (built from vocab categories at "
                "trainer init)."
            )
        combined = sorted(main + aux)
        if combined != list(range(self.n_tags)):
            raise ValueError(
                f"tag_indices_main ∪ tag_indices_aux must partition "
                f"[0, n_tags={self.n_tags}); got {len(main)} main + "
                f"{len(aux)} aux = {len(combined)} total (with duplicates "
                f"or gaps)."
            )

    def __post_init__(self) -> None:
        self._validate_routing()

    def _trunk_chans(
        self, d_in: int, kind: str, n_q: int, use_cls: bool, use_mean: bool,
    ) -> int:
        """One side's contribution to ``trunk_in_dim``."""
        if kind == "mean":
            return d_in
        if kind == "map":
            return d_in * (n_q + int(use_cls) + int(use_mean))
        raise ValueError(f"unknown pool_kind={kind!r}")

    @property
    def trunk_in_dim(self) -> int:
        """Width of the trunk's first Linear — main + (optional) aux contributions."""
        # Single-encoder mean: legacy single-vector trunk. (Stays at d_in
        # so anima-tagger-v1 checkpoints keep loading bit-identically.)
        if not self.has_aux and self.pool_kind == "mean":
            return self.d_in
        total = self._trunk_chans(
            self.d_in, self.pool_kind,
            self.pool_n_queries, self.pool_use_cls, self.pool_use_mean,
        )
        if self.has_aux:
            total += self._trunk_chans(
                self.d_in_aux, self.effective_pool_kind_aux,
                self.pool_n_queries_aux, self.pool_use_cls_aux, self.pool_use_mean_aux,
            )
        return total

    @property
    def main_trunk_in_dim(self) -> int:
        """Width of the *main-only* trunk Linear (per-head routing path)."""
        return self._trunk_chans(
            self.d_in, self.pool_kind,
            self.pool_n_queries, self.pool_use_cls, self.pool_use_mean,
        )

    @property
    def aux_trunk_in_dim(self) -> int:
        """Width of the *aux-only* trunk Linear (per-head routing path)."""
        if not self.has_aux:
            raise ValueError("aux_trunk_in_dim is only defined when d_in_aux is set")
        return self._trunk_chans(
            self.d_in_aux, self.effective_pool_kind_aux,
            self.pool_n_queries_aux, self.pool_use_cls_aux, self.pool_use_mean_aux,
        )

    @property
    def has_aux(self) -> bool:
        return self.d_in_aux is not None

    def to_dict(self) -> dict:
        d = {
            "d_in": self.d_in,
            "n_tags": self.n_tags,
            "n_ratings": self.n_ratings,
            "n_people_counts": self.n_people_counts,
            "d_hidden": self.d_hidden,
            "dropout": self.dropout,
            "pool_kind": self.pool_kind,
            "pool_n_queries": self.pool_n_queries,
            "pool_n_heads": self.pool_n_heads,
            "pool_use_cls": self.pool_use_cls,
            "pool_use_mean": self.pool_use_mean,
        }
        # Only emit the aux block when configured. Keeps single-encoder
        # config.json files visually identical to the v1 layout.
        if self.d_in_aux is not None:
            d.update({
                "d_in_aux": self.d_in_aux,
                "pool_n_queries_aux": self.pool_n_queries_aux,
                "pool_n_heads_aux": self.pool_n_heads_aux,
                "pool_use_cls_aux": self.pool_use_cls_aux,
                "pool_use_mean_aux": self.pool_use_mean_aux,
            })
            # pool_kind_aux is only emitted when it differs from main —
            # absent value means "inherit", which keeps the dual-MAP-from-
            # default-pool_kind config.json identical to the prior version.
            if self.pool_kind_aux is not None and self.pool_kind_aux != self.pool_kind:
                d["pool_kind_aux"] = self.pool_kind_aux
        # Routing block — only emitted when active. Keeps non-routed configs
        # byte-identical to the prior shape so legacy checkpoints round-trip.
        if self.use_per_head_routing:
            d["use_per_head_routing"] = True
            d["tag_indices_main"] = list(self.tag_indices_main)
            d["tag_indices_aux"] = list(self.tag_indices_aux)
            d["gate_init_bias_main"] = self.gate_init_bias_main
            d["gate_init_bias_aux"] = self.gate_init_bias_aux
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "AnimaTaggerConfig":
        d_in_aux_raw = d.get("d_in_aux")
        pool_kind_aux_raw = d.get("pool_kind_aux")
        return cls(
            d_in=int(d["d_in"]),
            n_tags=int(d["n_tags"]),
            n_ratings=int(d.get("n_ratings", 3)),
            n_people_counts=int(d.get("n_people_counts", 0)),
            d_hidden=int(d.get("d_hidden", 1024)),
            dropout=float(d.get("dropout", 0.1)),
            pool_kind=str(d.get("pool_kind", "mean")),
            pool_n_queries=int(d.get("pool_n_queries", 4)),
            pool_n_heads=int(d.get("pool_n_heads", 8)),
            pool_use_cls=bool(d.get("pool_use_cls", True)),
            pool_use_mean=bool(d.get("pool_use_mean", True)),
            d_in_aux=int(d_in_aux_raw) if d_in_aux_raw is not None else None,
            pool_kind_aux=str(pool_kind_aux_raw) if pool_kind_aux_raw is not None else None,
            pool_n_queries_aux=int(d.get("pool_n_queries_aux", 4)),
            pool_n_heads_aux=int(d.get("pool_n_heads_aux", 8)),
            pool_use_cls_aux=bool(d.get("pool_use_cls_aux", True)),
            pool_use_mean_aux=bool(d.get("pool_use_mean_aux", True)),
            use_per_head_routing=bool(d.get("use_per_head_routing", False)),
            tag_indices_main=list(d.get("tag_indices_main") or []),
            tag_indices_aux=list(d.get("tag_indices_aux") or []),
            gate_init_bias_main=float(d.get("gate_init_bias_main", 2.0)),
            gate_init_bias_aux=float(d.get("gate_init_bias_aux", -2.0)),
        )


class MAPHead(nn.Module):
    """Multi-query attention pool — K learnable queries attend over the token grid.

    Shape: ``[B, T, D] → [B, K, D]``. Pre-norm on K/V (the queries are
    learnable parameters and don't need it). Uses :class:`nn.MultiheadAttention`
    with ``batch_first=True``; PyTorch routes through SDPA so this is a
    single fused kernel on CUDA.

    Initialization: queries drawn from N(0, 1/√D) so the dot-product scale
    matches the post-LayerNorm key/value scale and the initial attention
    map is roughly uniform (no early collapse onto a single token).
    """

    def __init__(self, d: int, n_queries: int = 4, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        if d % n_heads != 0:
            raise ValueError(f"MAPHead: d={d} must be divisible by n_heads={n_heads}")
        self.q = nn.Parameter(torch.randn(n_queries, d) * (d ** -0.5))
        self.norm_kv = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, T, D]
        B = tokens.shape[0]
        q = self.q.unsqueeze(0).expand(B, -1, -1)        # [B, K, D]
        kv = self.norm_kv(tokens)                        # [B, T, D]
        out, _ = self.attn(q, kv, kv, need_weights=False)
        return out                                       # [B, K, D]


class AnimaTaggerHead(nn.Module):
    def __init__(self, cfg: AnimaTaggerConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.pool_kind not in ("mean", "map"):
            raise ValueError(f"unknown pool_kind={cfg.pool_kind!r}")
        if cfg.has_aux and cfg.effective_pool_kind_aux not in ("mean", "map"):
            raise ValueError(
                f"unknown pool_kind_aux={cfg.effective_pool_kind_aux!r}"
            )
        # Per-side MAPHead — only instantiated when that side uses MAP pool.
        # Kept as None on mean-pool sides so the state_dict stays minimal
        # (no phantom buffers on legacy mean-pool checkpoints).
        self.pool: Optional[MAPHead] = (
            MAPHead(
                d=cfg.d_in,
                n_queries=cfg.pool_n_queries,
                n_heads=cfg.pool_n_heads,
                dropout=0.0,
            )
            if cfg.pool_kind == "map" else None
        )
        self.pool_aux: Optional[MAPHead] = (
            MAPHead(
                d=cfg.d_in_aux,
                n_queries=cfg.pool_n_queries_aux,
                n_heads=cfg.pool_n_heads_aux,
                dropout=0.1,
            )
            if cfg.has_aux and cfg.effective_pool_kind_aux == "map" else None
        )

        if cfg.use_per_head_routing:
            # Two parallel projection trunks — each pool projects to d_hidden
            # independently, then per-head soft gates fuse them. No shared
            # concat trunk on this path.
            self.trunk = None
            self.trunk_main = nn.Sequential(
                nn.LayerNorm(cfg.main_trunk_in_dim),
                nn.Linear(cfg.main_trunk_in_dim, cfg.d_hidden),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
            )
            self.trunk_aux = nn.Sequential(
                nn.LayerNorm(cfg.aux_trunk_in_dim),
                nn.Linear(cfg.aux_trunk_in_dim, cfg.d_hidden),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
            )
            # Per-head scalar gates. α = sigmoid(g); main-lean inits to +2.0
            # (α ≈ 0.88), aux-lean to −2.0 (α ≈ 0.12). The split tag heads
            # supervise disjoint vocab slices — scattered back into [n_tags]
            # in forward() so callers see the legacy output shape.
            self.gate_tag_main = nn.Parameter(
                torch.full((), cfg.gate_init_bias_main)
            )
            self.gate_tag_aux = nn.Parameter(
                torch.full((), cfg.gate_init_bias_aux)
            )
            self.gate_rating = nn.Parameter(
                torch.full((), cfg.gate_init_bias_main)
            )
            n_main = len(cfg.tag_indices_main)
            n_aux = len(cfg.tag_indices_aux)
            self.tag_head_main = nn.Linear(cfg.d_hidden, n_main) if n_main > 0 else None
            self.tag_head_aux = nn.Linear(cfg.d_hidden, n_aux) if n_aux > 0 else None
            # Cache the vocab-index tensors as non-trainable buffers so they
            # ride device / dtype moves with the module and round-trip in
            # state_dict (useful for sanity-checking loaded checkpoints).
            self.register_buffer(
                "tag_idx_main",
                torch.tensor(cfg.tag_indices_main, dtype=torch.long),
                persistent=True,
            )
            self.register_buffer(
                "tag_idx_aux",
                torch.tensor(cfg.tag_indices_aux, dtype=torch.long),
                persistent=True,
            )
            self.tag_head = None
        else:
            self.trunk = nn.Sequential(
                nn.LayerNorm(cfg.trunk_in_dim),
                nn.Linear(cfg.trunk_in_dim, cfg.d_hidden),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
            )
            self.tag_head = nn.Linear(cfg.d_hidden, cfg.n_tags)
            self.trunk_main = None
            self.trunk_aux = None
            self.tag_head_main = None
            self.tag_head_aux = None

        self.rating_head = nn.Linear(cfg.d_hidden, cfg.n_ratings)
        # Optional — older checkpoints have n_people_counts=0 and no people
        # head in the state_dict. Keeping the attribute as None lets `forward`
        # return a stable 3-tuple shape in both cases.
        self.people_head: Optional[nn.Linear] = (
            nn.Linear(cfg.d_hidden, cfg.n_people_counts)
            if cfg.n_people_counts > 0 else None
        )
        # Gate for the people head — main-lean. Only created on the routing
        # path AND when the people head exists.
        if cfg.use_per_head_routing and self.people_head is not None:
            self.gate_people: Optional[nn.Parameter] = nn.Parameter(
                torch.full((), cfg.gate_init_bias_main)
            )
        else:
            self.gate_people = None

    @staticmethod
    def _pool_one(
        tokens: torch.Tensor,
        pool: MAPHead,
        use_cls: bool,
        use_mean: bool,
    ) -> torch.Tensor:
        """[B, T, D] → [B, (K + use_cls + use_mean) * D] via MAP + (optional) CLS / mean concat."""
        chans = [pool(tokens).flatten(1)]                       # [B, K*D]
        if use_cls:
            chans.append(tokens[:, 0])                          # [B, D]
        if use_mean:
            chans.append(tokens.mean(dim=1))                    # [B, D]
        return torch.cat(chans, dim=-1)

    def _pool_side(
        self,
        feat: torch.Tensor,
        kind: str,
        pool: Optional[MAPHead],
        use_cls: bool,
        use_mean: bool,
        side_name: str,
    ) -> torch.Tensor:
        """Apply the right pooling for one side, returning [B, channels].

        Validates the input tensor rank against ``kind``: ``mean`` expects
        ``[B, D]`` (the cached feature is already the pool); ``map`` expects
        ``[B, T, D]`` (head's MAPHead pools internally).
        """
        if kind == "mean":
            if feat.dim() != 2:
                raise ValueError(
                    f"{side_name} side: pool_kind='mean' expects pre-pooled "
                    f"[B, D] but got rank {feat.dim()}"
                )
            return feat
        if kind == "map":
            if feat.dim() != 3:
                raise ValueError(
                    f"{side_name} side: pool_kind='map' expects [B, T, D] "
                    f"tokens but got rank {feat.dim()}"
                )
            assert pool is not None, f"{side_name} MAP path called without configured pool"
            return self._pool_one(feat, pool, use_cls, use_mean)
        raise ValueError(f"{side_name} side: unknown pool_kind={kind!r}")

    def forward(
        self,
        feat: torch.Tensor,
        feat_aux: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        cfg = self.cfg
        # Aux presence must match config.
        if cfg.has_aux and feat_aux is None:
            raise ValueError(
                "config has aux encoder (d_in_aux is set) but feat_aux was not "
                "provided to forward()"
            )
        if feat_aux is not None and not cfg.has_aux:
            raise ValueError(
                "feat_aux provided but config has no aux encoder (d_in_aux is None)"
            )

        main = self._pool_side(
            feat, cfg.pool_kind, self.pool,
            cfg.pool_use_cls, cfg.pool_use_mean, "main",
        )
        aux: Optional[torch.Tensor] = None
        if cfg.has_aux:
            assert feat_aux is not None
            aux = self._pool_side(
                feat_aux, cfg.effective_pool_kind_aux, self.pool_aux,
                cfg.pool_use_cls_aux, cfg.pool_use_mean_aux, "aux",
            )

        if cfg.use_per_head_routing:
            # Dual-trunk + per-head soft gate. Each output head fuses
            # `α · h_main + (1 − α) · h_aux` with its own learnable α.
            assert aux is not None  # use_per_head_routing forces has_aux
            assert self.trunk_main is not None and self.trunk_aux is not None
            h_main = self.trunk_main(main)
            h_aux = self.trunk_aux(aux)

            def _fuse(gate: torch.Tensor) -> torch.Tensor:
                alpha = torch.sigmoid(gate)
                return alpha * h_main + (1.0 - alpha) * h_aux

            B = h_main.shape[0]
            tag_logits = h_main.new_zeros((B, cfg.n_tags))
            if self.tag_head_main is not None:
                h_tg_main = _fuse(self.gate_tag_main)
                logits_main = self.tag_head_main(h_tg_main)
                tag_logits.index_copy_(1, self.tag_idx_main, logits_main)
            if self.tag_head_aux is not None:
                h_tg_aux = _fuse(self.gate_tag_aux)
                logits_aux = self.tag_head_aux(h_tg_aux)
                tag_logits.index_copy_(1, self.tag_idx_aux, logits_aux)

            h_rate = _fuse(self.gate_rating)
            rating_logits = self.rating_head(h_rate)

            if self.people_head is not None and self.gate_people is not None:
                h_people = _fuse(self.gate_people)
                people_logits = self.people_head(h_people)
            else:
                people_logits = None
            return tag_logits, rating_logits, people_logits

        # Legacy concat-trunk path.
        x = torch.cat([main, aux], dim=-1) if aux is not None else main
        assert self.trunk is not None and self.tag_head is not None
        h = self.trunk(x)
        people_logits = self.people_head(h) if self.people_head is not None else None
        return self.tag_head(h), self.rating_head(h), people_logits
