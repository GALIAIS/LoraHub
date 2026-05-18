"""Cache builders for the training paths.

* ``cmd_build_features`` — encode each manifest image through the frozen
  PE-Core encoder. Dispatches on ``--pool_kind``:

  * ``map`` (default) → :class:`TokenCacheBuilder` writes per-stem
    ``[T, d_enc] bf16`` to ``.cache/tokens-<encoder>/``. Consumed by
    the MAP-pool head (single fixed cache → swappable pool design).
  * ``mean`` → :class:`FeatureCacheBuilder` writes per-stem
    ``[d_enc] fp32`` mean-pooled vectors to ``.cache/pooled-<encoder>/``.
    Legacy path; kept for backward compat with old checkpoints.

* ``cmd_build_resized`` — LANCZOS-resize each manifest image to its PE
  bucket and write per-stem ``uint8 [C, H, W]`` safetensors. Consumed by
  the end-to-end PE-LoRA path (where the encoder is unfrozen and
  pre-pooled features can't track it).

All modes are idempotent — re-runs only fill in missing entries.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def cache_dir_for(out_dir: Path, pool_kind: str, encoder: str) -> Path:
    """Resolve the per-encoder cache subdir for the given pool kind.

    Shared with the trainer / calibrator so they all agree on the file
    layout — change here, propagate everywhere.
    """
    sub = "tokens" if pool_kind == "map" else "pooled"
    return out_dir / ".cache" / f"{sub}-{encoder}"


def _build_one_encoder(
    args: argparse.Namespace,
    manifest,
    encoder_name: str,
    pool_kind: str,
    device: torch.device,
) -> None:
    """Build a single token / feature cache for one encoder at the given pool_kind."""
    from library.captioning.anima_tagger_data import (
        FeatureCacheBuilder,
        TokenCacheBuilder,
    )

    out_dir = Path(args.out_dir)
    cache_dir = cache_dir_for(out_dir, pool_kind, encoder_name)
    # Auto-shortcut PIL decode + LANCZOS when the resized cache for this
    # encoder is already on disk (output bit-equivalent up to uint8
    # quantization — well below the encoder's bf16 noise floor).
    resized_cache_dir = out_dir / ".cache" / f"resized-{encoder_name}"
    resized_present = resized_cache_dir.exists() and any(resized_cache_dir.iterdir())
    logger.info(
        "build_features: pool_kind=%s  %d manifest entries → %s (device=%s, encoder=%s, "
        "resized_shortcut=%s)",
        pool_kind,
        len(manifest.stems),
        cache_dir,
        device,
        encoder_name,
        resized_present,
    )
    builder_kwargs = dict(
        manifest=manifest,
        cache_dir=cache_dir,
        device=device,
        encoder_name=encoder_name,
        num_workers=args.feature_cache_workers,
        resized_cache_dir=resized_cache_dir if resized_present else None,
    )
    if pool_kind == "map":
        builder = TokenCacheBuilder(**builder_kwargs)
    else:
        builder = FeatureCacheBuilder(**builder_kwargs)
    n_new = builder.build()
    n_total = len(manifest.stems) - len(builder.missing_stems())
    print(f"  [{encoder_name}/{pool_kind}] cache dir:        {cache_dir}")
    print(f"  [{encoder_name}/{pool_kind}] newly encoded:    {n_new}")
    print(f"  [{encoder_name}/{pool_kind}] cached / total:   {n_total} / {len(manifest.stems)}")


def cmd_build_features(args: argparse.Namespace) -> None:
    """Build per-encoder token / feature caches.

    With ``--aux_encoder`` set, also builds a parallel cache for the
    auxiliary encoder. The two sides may use different pool kinds —
    ``--pool_kind`` controls main, ``--pool_kind_aux`` controls aux (defaults
    to inheriting ``--pool_kind``). Each ``(encoder, pool_kind)`` pair gets
    its own ``.cache/{tokens,pooled}-<name>/`` subdir so they can be built
    / refreshed independently.
    """
    from library.captioning.anima_tagger_data import TaggerManifest

    out_dir = Path(args.out_dir)
    manifest_path = out_dir / "dataset.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"missing {manifest_path} — run --mode build_vocab first."
        )
    manifest = TaggerManifest.from_path(manifest_path)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    _build_one_encoder(args, manifest, args.encoder, args.pool_kind, device)
    aux_encoder = getattr(args, "aux_encoder", None)
    if aux_encoder:
        pool_kind_aux = args.pool_kind_aux or args.pool_kind
        _build_one_encoder(args, manifest, aux_encoder, pool_kind_aux, device)


def cmd_build_resized(args: argparse.Namespace) -> None:
    from library.captioning.anima_tagger_data import (
        ImageCacheBuilder,
        TaggerManifest,
    )
    from library.vision.encoders import get_encoder_info

    out_dir = Path(args.out_dir)
    manifest_path = out_dir / "dataset.json"
    if not manifest_path.exists():
        raise SystemExit(f"missing {manifest_path} — run --mode build_vocab first.")
    manifest = TaggerManifest.from_path(manifest_path)
    cache_dir = out_dir / ".cache" / f"resized-{args.encoder}"
    spec = get_encoder_info(args.encoder).bucket_spec
    logger.info(
        "build_resized: %d manifest entries → %s (encoder=%s, patch=%d)",
        len(manifest.stems),
        cache_dir,
        args.encoder,
        spec.patch,
    )
    builder = ImageCacheBuilder(
        manifest=manifest,
        cache_dir=cache_dir,
        spec=spec,
        num_workers=args.feature_cache_workers,
    )
    n_new = builder.build()
    n_total = len(manifest.stems) - len(builder.missing_stems())
    print(f"  cache dir:        {cache_dir}")
    print(f"  newly resized:    {n_new}")
    print(f"  cached / total:   {n_total} / {len(manifest.stems)}")
