"""CLI entry — argparse + mode dispatcher.

External-corpus paths are resolved via the ``CAPTION_CORPUS_DIR`` env var
(typically set in ``anima_lora/.env``). The corpus directory is expected to
contain ``retrieved/`` (raw caption pool), ``selected/`` (curated subset),
``tag_rules.yaml`` (caption normalization rules), and ``.tag_cache.json``
(per-tag Booru-style category cache, indexed under ``retrieved/``). All of
these can be overridden individually by CLI flags.

Modes (selected by ``--mode``):

* ``build_vocab``    — scan caption sources, intersect with the tag-taxonomy
                       cache, snapshot ``tag_rules.yaml``, emit
                       ``vocab.json`` plus a fixed train/val split and a
                       per-stem ``dataset.json`` manifest.
* ``build_features`` — encode every manifest image through frozen PE-Core
                       and write per-stem cache. ``--pool_kind=map`` (default)
                       writes the full token sequence (consumed by the MAP
                       attention-pool head); ``--pool_kind=mean`` writes the
                       legacy mean-pooled vector.
* ``build_resized``  — LANCZOS-resize every manifest image to its PE bucket,
                       cache as ``uint8 [C, H, W]`` for end-to-end PE-LoRA.
* ``train``          — train the multi-label head + 3-class rating head + 8-class people-count head.
                       Dispatches to the cached path or PE-LoRA path based
                       on ``--pe_lora_rank``.
* ``calibrate``      — sweep per-tag F1-optimal thresholds on the val split.
* ``predict``        — single-image debug entry.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Make ``anima_lora/`` importable when invoked as ``python -m
# scripts.anima_tagger.cli`` from outside the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from library.env import load_dotenv  # noqa: E402
from library.log import setup_logging  # noqa: E402

# Pull CAPTION_CORPUS_DIR (and any other overrides) from anima_lora/.env
# before argparse builds defaults. CLI flags still win over env values.
load_dotenv()

setup_logging()
logger = logging.getLogger(__name__)


def _corpus_default(rel: str):
    """Resolve ``$CAPTION_CORPUS_DIR/<rel>`` for argparse defaults.

    Returns ``None`` when the env var is unset so argparse renders an
    explicit '(unset)' marker in --help instead of a misleading empty path.
    """
    root = os.environ.get("CAPTION_CORPUS_DIR")
    if not root:
        return None
    return str(Path(root) / rel)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Anima tagger trainer")
    p.add_argument(
        "--mode",
        choices=[
            "build_vocab",
            "build_features",
            "build_resized",
            "train",
            "calibrate",
            "predict",
            "scan_role_markers",
        ],
        default="build_vocab",
    )
    p.add_argument(
        "--encoder",
        default="pe",
        help="Vision encoder registry name (passed to load_pe_encoder). "
        "Default: pe (PE-Core-L14-336).",
    )
    p.add_argument(
        "--aux_encoder",
        default="pe_spatial",
        help="Auxiliary vision encoder for dual-encoder training (default: "
        "'pe_spatial' for PE-Spatial-B16-512). build_features builds a "
        "parallel cache; train builds a dual-MAPHead model that concatenates "
        "pool outputs from both encoders. Pass --aux_encoder '' (empty) for "
        "the single-encoder path.",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Torch device for build_features / train (default: cuda if available).",
    )
    p.add_argument(
        "--feature_cache_workers",
        type=int,
        default=6,
        help="DataLoader workers for build_features CPU-side decode + LANCZOS "
        "resize (default: 4). Set to 0 to run inline on the main process.",
    )

    # Vocab-build inputs. All three default to subpaths of
    # ``$CAPTION_CORPUS_DIR``; pass --caption_roots / --tag_cache / --rules
    # explicitly to override.
    raw_default = _corpus_default("retrieved")
    curated_default = _corpus_default("selected")
    p.add_argument(
        "--caption_roots",
        nargs="+",
        default=[d for d in (raw_default, curated_default, "image_dataset") if d],
        help="Directories to scan recursively for *.txt caption files. "
        "First-match-wins by stem when a duplicate appears across roots. "
        "Defaults: $CAPTION_CORPUS_DIR/retrieved + "
        "$CAPTION_CORPUS_DIR/selected + image_dataset/.",
    )
    p.add_argument(
        "--tag_cache",
        default=_corpus_default("retrieved/.tag_cache.json"),
        help="Tag-taxonomy JSON (tag → integer type ID). "
        "Default: $CAPTION_CORPUS_DIR/retrieved/.tag_cache.json.",
    )
    p.add_argument(
        "--rules",
        default=_corpus_default("tag_rules.yaml"),
        help="Caption-normalization rules (snapshotted into out_dir at "
        "build time). Default: $CAPTION_CORPUS_DIR/tag_rules.yaml.",
    )
    p.add_argument(
        "--groups",
        default=_corpus_default("tag_groups.yaml"),
        help="Tag-groups YAML (typed groupings — eye_color, hair_color, "
        "rating, …). Resolved against the kept vocab and embedded into "
        "vocab.json[groups]; the YAML is snapshotted to out_dir/groups.yaml. "
        "Optional — pass empty / unset to build a flat-vocab checkpoint. "
        "Default: $CAPTION_CORPUS_DIR/tag_groups.yaml.",
    )
    p.add_argument("--min_freq", type=int, default=20)
    p.add_argument("--val_frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)

    # Train-mode knobs.
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch_size", type=int, default=96)
    p.add_argument(
        "--postfix_every",
        type=int,
        default=8,
        help="PE-LoRA training: refresh the tqdm postfix (and force a "
        "host-device sync) every N steps. Higher = fewer syncs / faster "
        "training; lower = more responsive progress bar (default: 10).",
    )
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument(
        "--warmup_steps",
        type=int,
        default=250,
        help="Linear lr warmup over the first N optimizer steps before cosine "
        "decay takes over. 0 (default) disables warmup and runs pure cosine "
        "on a per-step schedule. Typical values: 200-1000 for fresh-head "
        "training on this scale.",
    )
    p.add_argument("--d_hidden", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)

    # Pool architecture. ``map`` = K learnable queries attend over PE patch
    # tokens (CLS + mean concatenated as auxiliary channels). ``mean`` =
    # legacy mean-pool path (head consumes a pre-pooled [B, d_enc] feature).
    # build_features / train / calibrate all read --pool_kind to pick the
    # cache subdir and the head shape.
    p.add_argument(
        "--pool_kind",
        choices=["map", "mean"],
        default="mean",
        help="Pool head over the main encoder's tokens. 'map' (default): "
        "K-query attention pool + CLS + mean concat → trunk. 'mean': "
        "single-vector mean-pool. Selects cache subdir "
        "(.cache/tokens-<encoder>/ vs pooled-<encoder>/) and head arch.",
    )
    p.add_argument(
        "--pool_kind_aux",
        choices=["map", "mean"],
        default="map",
        help="Pool kind for the auxiliary encoder. Default 'map' pairs with "
        "PE-Spatial's full attention pool. Set 'mean' to swap for a cheap "
        "mean-pool on the aux side (rare — defeats the point of PE-Spatial).",
    )
    p.add_argument(
        "--pool_n_queries",
        type=int,
        default=0,
        help="MAP pool: number of learnable queries (default 4). Each query "
        "produces one [d_enc] vector; trunk input is "
        "(K + use_cls + use_mean) * d_enc.",
    )
    p.add_argument(
        "--pool_n_heads",
        type=int,
        default=0,
        help="MAP pool: number of attention heads (default 8). Must divide "
        "the encoder dim (d_enc=1024 for PE-Core).",
    )
    p.add_argument(
        "--pool_use_cls",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="MAP pool: concat the encoder's CLS token as an aux channel "
        "(default on).",
    )
    p.add_argument(
        "--pool_use_mean",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="MAP pool: concat the patch-token mean as an aux channel "
        "(default on — gives the legacy baseline as a residual).",
    )

    # Aux encoder MAP-pool knobs. Only consulted when --aux_encoder is set;
    # otherwise inert. Defaults mirror the main pool — change per-encoder
    # only when there's a reason (e.g. PE-Spatial's d=768 admits more head
    # divisors so a bigger n_heads_aux is fine if it pays off in F1).
    p.add_argument(
        "--pool_n_queries_aux",
        type=int,
        default=4,
        help="Aux MAP pool: number of learnable queries (default 4). Each "
        "query produces one [d_in_aux] vector.",
    )
    p.add_argument(
        "--pool_n_heads_aux",
        type=int,
        default=8,
        help="Aux MAP pool: attention heads (default 8). Must divide d_in_aux "
        "(768 for PE-Spatial-B16-512 — divisors include 8, 12, 16, 24).",
    )
    p.add_argument(
        "--pool_use_cls_aux",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Aux MAP pool: concat the encoder's CLS token (default on).",
    )
    p.add_argument(
        "--pool_use_mean_aux",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Aux MAP pool: concat the patch-token mean (default on).",
    )
    p.add_argument(
        "--use_per_head_routing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Replace the concat-trunk + single tag_head with two parallel "
        "trunks (one per encoder) and per-head soft gates. tag_head is split "
        "by vocab category into a main-lean sub-head "
        "(character/copyright/artist/count) biased toward PE-Core and an "
        "aux-lean sub-head (general/metadata/deprecated) biased toward "
        "PE-Spatial; rating + people heads start main-leaning. Requires "
        "--aux_encoder. Default off (legacy concat-trunk path).",
    )
    p.add_argument(
        "--lambda_rating",
        type=float,
        default=0.1,
        help="Weight on the rating CE loss relative to multi-label BCE.",
    )
    p.add_argument(
        "--lambda_people",
        type=float,
        default=0.1,
        help="Weight on the people-count CE loss relative to multi-label BCE. "
        "0 disables the head's gradient contribution (still runs forward "
        "if the manifest carries labels).",
    )

    # PE-LoRA knobs (end-to-end PE-Core fine-tuning on the trailing N blocks).
    # When --pe_lora_rank > 0, the trainer ignores the pre-pooled feature
    # cache and reads pre-resized images from .cache/resized-<encoder>/
    # (build via `--mode build_resized`). The frozen PE encoder runs each
    # step with LoRA active on the last `--pe_lora_layers` resblocks.
    p.add_argument(
        "--pe_lora_rank",
        type=int,
        default=0,
        help="LoRA rank on PE-Core's trailing blocks. 0 (default) → encoder "
        "stays frozen and trainer reads pre-pooled features from cache. "
        ">0 → end-to-end PE-LoRA training; reads pre-resized images from "
        ".cache/resized-<encoder>/ (build via --mode build_resized).",
    )
    p.add_argument(
        "--pe_lora_alpha",
        type=float,
        default=16.0,
        help="LoRA scale = alpha / rank.",
    )
    p.add_argument(
        "--pe_lora_layers",
        type=int,
        default=2,
        help="Number of trailing PE resblocks to adapt with LoRA. Mapped to "
        "inject_pe_lora's layer_from arg.",
    )
    p.add_argument(
        "--pe_lora_lr",
        type=float,
        default=1e-4,
        help="Learning rate for PE-LoRA params (head/trunk keeps --lr).",
    )
    p.add_argument(
        "--pe_lora_qkv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Adapt the QKV in_proj path (default: on).",
    )
    p.add_argument(
        "--pe_lora_attn_out",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Adapt attn.out_proj (default: on).",
    )
    p.add_argument(
        "--pe_lora_mlp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Adapt MLP c_fc / c_proj (default: on).",
    )
    p.add_argument(
        "--init_head_from",
        default=None,
        help="PE-LoRA training: warm-start the head from a Stage-1 "
        "checkpoint (path to model.safetensors). The head state_dict layout "
        "must match the new run's AnimaTaggerHead config (same n_tags, "
        "n_ratings, n_people_counts, d_hidden, d_in). Optimizer state is NOT loaded — "
        "Stage 2 re-builds Adam from scratch.",
    )

    # Predict mode: single-image debug entry.
    p.add_argument(
        "--image",
        default=None,
        help="Image path for --mode predict.",
    )
    p.add_argument(
        "--show_scores",
        action="store_true",
        help="Predict mode: also print rating distribution + top-K kept tags.",
    )
    p.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Predict mode: number of top kept tags to show with --show_scores.",
    )

    # scan_role_markers mode: rank character-typed tags by solo co-occurrence
    # (high ratio → likely a class/affiliation marker mis-typed as character).
    p.add_argument(
        "--min_solo",
        type=int,
        default=5,
        help="scan_role_markers: drop tags with fewer than this many solo "
        "occurrences (default: 5).",
    )
    p.add_argument(
        "--min_ratio",
        type=float,
        default=0.5,
        help="scan_role_markers: drop tags whose conditional co-occurrence "
        "ratio with another character on solo images is below this (default: 0.5).",
    )
    p.add_argument(
        "--top_partners",
        type=int,
        default=3,
        help="scan_role_markers: how many top co-occurring partners to print "
        "per row (default: 3).",
    )
    p.add_argument(
        "--min_role_partners",
        type=int,
        default=5,
        help="scan_role_markers: a candidate with at least this many distinct "
        "co-occurrence partners is classified D_role (broad pool → "
        "affiliation marker). Default: 5.",
    )
    p.add_argument(
        "--pair_dominance",
        type=float,
        default=0.6,
        help="scan_role_markers: a candidate whose top-1 partner accounts for "
        "at least this fraction of co-occurrences is classified C_pair "
        "(narrow pool → genuine couple/sibling). Default: 0.6.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=200,
        help="scan_role_markers: cap rows printed in the table (default: 200).",
    )
    p.add_argument(
        "--out_yaml",
        default=None,
        help="scan_role_markers: optional path for a YAML stub of candidates, "
        "ready to paste into tag_rules.yaml.",
    )

    # Output.
    p.add_argument(
        "--out_dir",
        default="models/captioners/anima-tagger-v1",
    )

    args = p.parse_args()

    if args.mode == "build_vocab":
        missing = [
            name
            for name, val in (
                ("--tag_cache", args.tag_cache),
                ("--rules", args.rules),
            )
            if not val
        ]
        if missing or not args.caption_roots:
            raise SystemExit(
                "build_vocab needs CAPTION_CORPUS_DIR set in anima_lora/.env "
                f"(or {', '.join(missing) or '--caption_roots'} passed "
                "explicitly). Add a line like\n"
                "    CAPTION_CORPUS_DIR=/path/to/corpus\n"
                "to anima_lora/.env, or pass the paths via CLI flags."
            )

    # Empty-string opt-out: --aux_encoder "" disables the dual-encoder path.
    # Both --aux_encoder and --pool_kind_aux now default to truthy values
    # (pe_spatial / map), so opting out via empty string is the user-facing
    # off switch. Normalize to None so downstream truthiness checks work.
    if args.aux_encoder == "":
        args.aux_encoder = None

    if args.aux_encoder:
        if args.aux_encoder == args.encoder:
            raise SystemExit(
                f"--aux_encoder={args.aux_encoder!r} matches --encoder; aux must "
                f"be a different encoder (e.g. --encoder pe --aux_encoder pe_spatial)."
            )
        if args.mode == "train" and args.pe_lora_rank > 0:
            raise SystemExit(
                "--aux_encoder + --pe_lora_rank>0 is not supported. PE-LoRA "
                "training reads pre-resized images and runs the encoder live; "
                "the dual-encoder path consumes pre-encoded caches. "
                "Use --pe_lora_rank=0 for v1 (or pass --aux_encoder '' "
                "to disable the default aux encoder)."
            )
    # calibrate / predict load encoder + aux config from out_dir/config.json,
    # so cross-arg validation only applies to modes that consume CLI flags
    # as the source of truth.
    if args.mode in ("train", "build_features", "build_resized"):
        if args.use_per_head_routing and not args.aux_encoder:
            raise SystemExit(
                "--use_per_head_routing needs --aux_encoder (the per-head soft "
                "gate mixes a main and an aux trunk — there's no aux trunk to "
                "mix in single-encoder mode). Drop --aux_encoder '' or pass "
                "an explicit aux encoder name."
            )

    return args


def main() -> None:
    args = parse_args()
    if args.mode == "build_vocab":
        from .vocab import cmd_build_vocab

        cmd_build_vocab(args)
    elif args.mode == "build_features":
        from .caches import cmd_build_features

        cmd_build_features(args)
    elif args.mode == "build_resized":
        from .caches import cmd_build_resized

        cmd_build_resized(args)
    elif args.mode == "train":
        if args.pe_lora_rank > 0:
            from .train_pe_lora import cmd_train_pe_lora

            cmd_train_pe_lora(args)
        else:
            from .train_cached import cmd_train_cached

            cmd_train_cached(args)
    elif args.mode == "calibrate":
        from .calibrate import cmd_calibrate

        cmd_calibrate(args)
    elif args.mode == "predict":
        from .predict import cmd_predict

        cmd_predict(args)
    elif args.mode == "scan_role_markers":
        from .role_markers import cmd_scan_role_markers

        cmd_scan_role_markers(args)
    else:
        raise SystemExit(f"unknown --mode={args.mode!r}")


if __name__ == "__main__":
    main()
