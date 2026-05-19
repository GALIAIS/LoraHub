"""Cached-pair dataset for distillation scripts.

Loads pre-cached VAE latents + text encoder outputs from disk, grouped by
latent resolution so that each batch has uniform spatial dimensions (matching
the bucket-based batching used in LoRA training). Shared by
``scripts/distill_mod/distill.py`` and ``scripts/distill_turbo.py``.
"""

from __future__ import annotations

import glob
import logging
import os
import random

import torch

from library.io.cache import (
    LATENT_CACHE_SUFFIX,
    discover_cached_pairs,
    get_latent_resolution,
    load_cached_latents,
    load_cached_text_features,
)

logger = logging.getLogger(__name__)


class CachedDataset(torch.utils.data.Dataset):
    """Loads pre-cached latents and text encoder outputs for distillation.

    Samples are grouped by latent resolution so that each batch has uniform
    spatial dimensions (matching the bucket-based batching used in training).
    A deterministic per-bucket split (seeded by ``validation_seed``) carves off
    the last ``validation_split`` fraction for the val set, mirroring the
    LoRA training convention.
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 1,
        *,
        split: str = "train",
        validation_split: float = 0.0,
        validation_seed: int = 42,
        sample_ratio: float = 1.0,
        synth_data_dir: str | None = None,
    ):
        assert split in ("train", "val")
        self.data_dir = data_dir
        self.synth_data_dir = synth_data_dir
        cached = discover_cached_pairs(data_dir)

        # When --synth_data_dir is set, rewrite each sample's latent path to the
        # synthetic NPZ for the same stem. Samples without a synthetic
        # counterpart are dropped — the teacher pool is bounded by what was
        # generated via `make distill-prep`. TE paths remain in data_dir.
        # Lookup is stem-keyed (not basename-keyed) because the lora cache uses
        # WxH pixel dims zero-padded to 4 (e.g. `0896x1152`) while the synth
        # writer uses HxW latent dims (e.g. `144x112`) — the same logical pair
        # has different basenames in the two dirs.
        n_dropped_no_synth = 0
        if synth_data_dir is not None:
            synth_by_stem: dict[str, str] = {}
            for path in glob.glob(
                os.path.join(synth_data_dir, f"*{LATENT_CACHE_SUFFIX}")
            ):
                # `{stem}_{HxW}_anima.npz` → strip suffix, drop trailing `_HxW`
                without_suffix = os.path.basename(path).removesuffix(LATENT_CACHE_SUFFIX)
                stem = without_suffix.rsplit("_", 1)[0]
                synth_by_stem.setdefault(stem, path)
            remapped: list = []
            for img in cached:
                if img.te_path is None:
                    continue
                synth_path = synth_by_stem.get(img.stem)
                if synth_path is None:
                    n_dropped_no_synth += 1
                    continue
                # Reuse CachedImage shape so the downstream code is unchanged.
                remapped.append(img._replace(npz_path=synth_path))
            cached = remapped
            if n_dropped_no_synth:
                logger.warning(
                    f"[{split}] {n_dropped_no_synth} samples have no synthetic "
                    f"latent under {synth_data_dir}; dropped."
                )

        # Group samples by latent resolution
        buckets: dict[str, list[tuple[str, str]]] = {}
        for img in cached:
            if img.te_path is None:
                continue
            res = get_latent_resolution(img.npz_path)
            buckets.setdefault(res, []).append((img.npz_path, img.te_path))

        # Per-bucket deterministic shuffle, then carve last `validation_split`
        # off as val so train/val never overlap and remain bucket-grouped.
        # Apply sample_ratio per-bucket (mirrors the LoRA pipeline's per-subset
        # subsampling), keeping at least one sample per non-empty bucket so
        # debug/half presets don't silently drop entire resolutions.
        # Drop per-bucket remainders for whichever side we're emitting.
        rng = random.Random(validation_seed)
        self.samples: list[tuple[str, str]] = []
        n_train = n_val = 0
        for _res, items in buckets.items():
            items = list(items)
            rng.shuffle(items)
            n = len(items)
            n_v = int(round(n * validation_split)) if validation_split > 0.0 else 0
            n_t = n - n_v
            train_items = items[:n_t]
            val_items = items[n_t:]
            n_train += n_t
            n_val += n_v
            picked = train_items if split == "train" else val_items
            if sample_ratio < 1.0 and picked:
                n_keep = max(1, int(round(len(picked) * sample_ratio)))
                picked = picked[:n_keep]
            full = (len(picked) // batch_size) * batch_size
            self.samples.extend(picked[:full])

        sr_note = f", sample_ratio={sample_ratio}" if sample_ratio < 1.0 else ""
        source = (
            f"latents={synth_data_dir} (synth), te={data_dir}"
            if synth_data_dir is not None
            else data_dir
        )
        logger.info(
            f"[{split}] {len(self.samples)} samples from {source} "
            f"({len(buckets)} buckets; pre-drop train={n_train}, val={n_val}{sr_note})"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        latent_path, te_path = self.samples[idx]
        latents, _res, _h, _w = load_cached_latents(latent_path)  # (16, H, W)
        # Fixed variant=0: distill-mod targets a deterministic teacher mapping,
        # and the teacher cache keys on (sample_idx, sigma_idx) only — drawing
        # a random variant per visit would let cache hits return a teacher pred
        # computed under a different caption than the student is conditioned on.
        crossattn_emb, pooled_text = load_cached_text_features(te_path, variant=0)
        return idx, latents, crossattn_emb, pooled_text
