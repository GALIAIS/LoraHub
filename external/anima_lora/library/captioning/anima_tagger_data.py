"""Dataset plumbing for the Anima tagger.

Six pieces:

* :class:`TaggerManifest` — loads ``dataset.json`` (the per-stem
  image-path + multi-hot tag indices + rating-class index + people-count
  class index emitted by ``python -m scripts.anima_tagger.cli --mode
  build_vocab``).
* :class:`FeatureCacheBuilder` — legacy mean-pool cache. Encodes each
  manifest image through a frozen PE-Core trunk, mean-pools over patch
  tokens, writes a per-stem ``[d_enc] fp32`` safetensors. Idempotent.
  Used by ``pool_kind="mean"`` checkpoints.
* :class:`CachedFeatureDataset` — reads the legacy mean-pool cache into
  one in-memory tensor and exposes
  ``(feature, multi_hot, rating_idx, people_idx)`` tuples.
* :class:`TokenCacheBuilder` — full token cache for ``pool_kind="map"``.
  Encodes each manifest image through frozen PE-Core, writes a per-stem
  ``[T, d_enc] bf16`` safetensors (CLS at row 0). Storage is ~1.2 MB /
  stem at PE-Core-L14-336 (~14 GB for a 12K-stem dataset); the win is
  that swapping pool architectures no longer requires re-encoding.
* :class:`CachedTokenDataset` + :func:`collate_token_batch` — lazy
  bucket-grouped token dataset for the MAP-pool trainer (mirrors the
  PE-LoRA :class:`CachedImageDataset` pattern; T is constant within a
  bucket so the collate just stacks).
* :class:`ImageCacheBuilder` — LANCZOS-resizes each manifest image to
  its PE bucket and writes per-stem ``uint8 [C,H,W]`` safetensors. Pairs
  with PE-LoRA training where the encoder is unfrozen and pre-pooled
  features can't track it.
* :class:`CachedImageDataset` + :class:`BucketBatchSampler` — bucket-grouped
  image dataset for end-to-end PE-LoRA training. Each yielded batch is
  shape-homogeneous so the encoder can be called once per batch.

Cache layout is locked into each builder's file format — swap pooling /
storage layout → invalidate the cache dir → rebuild. The image cache
file format is plain ``uint8`` HWC post-LANCZOS-resize, no
normalization — the trainer applies ``(x/127.5) - 1`` at load time to
recover ``[-1, 1]``.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file as st_load
from safetensors.torch import save_file as st_save
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

from library.datasets.image_utils import IMAGE_TRANSFORMS
from library.vision.buckets import (
    BucketSpec,
    bucket_pixel_size,
    pick_bucket,
)
from library.vision.encoder import (
    VisionEncoderBundle,
    encode_pe_from_imageminus1to1,
    load_pe_encoder,
)

logger = logging.getLogger(__name__)


def pil_resize_to_bucket(img: Image.Image, spec: BucketSpec) -> Image.Image:
    """LANCZOS-resize a PIL image to its closest bucket size for ``spec``.

    Pre-resizing on the PIL side (high quality LANCZOS) avoids decoding
    multi-megapixel source images into a tensor only to bilinear-resize
    them down inside the encoder. Speeds up cache builds 5–10× on
    high-resolution corpora and removes a quality penalty (LANCZOS >
    bilinear for severe downscales).
    """
    w, h = img.size
    h_p, w_p = pick_bucket(h, w, spec)
    target_h, target_w = bucket_pixel_size((h_p, w_p), spec)
    if (h, w) != (target_h, target_w):
        img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    return img


class _ResizeDataset(Dataset):
    """CPU-side decode + bucket-resize + IMAGE_TRANSFORMS for one stem.

    When ``resized_cache_dir`` is given AND the per-stem uint8 sidecar
    exists, skips the PIL decode + LANCZOS path and loads the cached
    tensor directly, applying the same ``(x/127.5) - 1`` recovery the
    PE-LoRA trainer uses. This lets ``build_features`` ride on a
    pre-existing ``.cache/resized-<encoder>/`` (built by
    ``build_resized``) without any cost — output is bit-equivalent up to
    uint8 quantization (≤1/127.5 per channel), which is well below the
    encoder's bf16 noise floor.

    Returns ``(stem, tensor[C,H,W] | None, err)``. Errors are surfaced as
    a non-empty string so the consumer can log and continue.
    """

    def __init__(
        self,
        stems: Sequence[str],
        image_paths: Sequence[Path],
        spec: BucketSpec,
        resized_cache_dir: Optional[Path] = None,
    ):
        self._stems = list(stems)
        self._paths = list(image_paths)
        self._spec = spec
        self._resized_cache_dir = resized_cache_dir

    def __len__(self) -> int:
        return len(self._stems)

    def __getitem__(self, k: int):
        stem = self._stems[k]
        path = self._paths[k]
        try:
            if self._resized_cache_dir is not None:
                cache_file = _image_cache_path(self._resized_cache_dir, stem)
                if cache_file.exists():
                    u8 = st_load(str(cache_file))["image"]    # [C, H, W] uint8
                    tensor = u8.to(torch.float32) / 127.5 - 1.0
                    return stem, tensor, ""
            with Image.open(path) as im:
                im = pil_resize_to_bucket(im.convert("RGB"), self._spec)
                arr = np.array(im)
            tensor = IMAGE_TRANSFORMS(arr)  # [C, H, W]
            return stem, tensor, ""
        except Exception as e:
            return stem, None, f"{type(e).__name__}: {e}"


def _identity_collate(batch):
    # batch_size=1 — buckets vary per image so we can't stack.
    return batch[0]


@dataclass
class TaggerManifest:
    """Trainable-sample manifest emitted by ``--mode build_vocab``."""

    stems: List[str]
    image_paths: List[Path]
    tag_indices: List[List[int]]
    rating_indices: List[int]
    people_count_indices: List[int]
    train_stems: List[str]
    val_stems: List[str]
    n_tags: int
    n_ratings: int
    n_people_counts: int

    @classmethod
    def from_path(cls, path: Path) -> "TaggerManifest":
        with open(path) as f:
            d = json.load(f)
        # ``people_count_indices`` / ``n_people_counts`` were added late; old
        # manifests rebuild on next ``build_vocab``. Until then, default to a
        # zero-length head so the trainer can detect "no people supervision"
        # cleanly (rather than crashing with a KeyError).
        people_idx = list(d.get("people_count_indices") or [])
        n_people = int(d.get("n_people_counts", 0))
        if people_idx and not n_people:
            n_people = max(people_idx) + 1
        return cls(
            stems=list(d["stems"]),
            image_paths=[Path(p) for p in d["image_paths"]],
            tag_indices=[list(idxs) for idxs in d["tag_indices"]],
            rating_indices=list(d["rating_indices"]),
            people_count_indices=people_idx,
            train_stems=list(d["split"]["train"]),
            val_stems=list(d["split"]["val"]),
            n_tags=int(d["n_tags"]),
            n_ratings=int(d["n_ratings"]),
            n_people_counts=n_people,
        )

    def stem_index(self) -> Dict[str, int]:
        return {s: i for i, s in enumerate(self.stems)}


# ── Feature cache ─────────────────────────────────────────────────────────


def _cache_path(cache_dir: Path, stem: str) -> Path:
    return cache_dir / f"{stem}.safetensors"


class FeatureCacheBuilder:
    """Build per-stem mean-pooled PE-Core features into ``cache_dir``.

    Uses a single-image-per-forward path for simplicity (PE-Core supports
    dynamic resolution; we don't need to bucket-batch). One forward per
    image is fast enough that 12K stems finish in ~10–20 minutes on a
    single GPU; a bucketed-batch path can be added later if it shows up
    in profiling.

    ``resized_cache_dir`` (optional) — when supplied and the per-stem
    uint8 sidecar exists, skips PIL decode + LANCZOS for that stem and
    loads the cached tensor instead. Lets the cache build ride on the
    PE-LoRA path's ``.cache/resized-<encoder>/`` for free.
    """

    def __init__(
        self,
        manifest: TaggerManifest,
        cache_dir: Path,
        device: torch.device,
        encoder_name: str = "pe",
        dtype: torch.dtype = torch.bfloat16,
        num_workers: int = 4,
        resized_cache_dir: Optional[Path] = None,
    ):
        self.manifest = manifest
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.encoder_name = encoder_name
        self.dtype = dtype
        self.num_workers = num_workers
        self.resized_cache_dir = resized_cache_dir
        self._bundle: Optional[VisionEncoderBundle] = None

    def _bundle_lazy(self) -> VisionEncoderBundle:
        if self._bundle is None:
            self._bundle = load_pe_encoder(
                self.device, name=self.encoder_name, dtype=self.dtype
            )
        return self._bundle

    def missing_stems(self) -> List[int]:
        return [
            i
            for i, stem in enumerate(self.manifest.stems)
            if not _cache_path(self.cache_dir, stem).exists()
        ]

    @torch.no_grad()
    def build(self) -> int:
        """Encode + cache every stem missing from ``cache_dir``.

        Returns the count of newly cached entries (0 if everything was
        already cached). Errors on individual images are logged and the
        loop continues — a single corrupt image shouldn't tank the run.
        """
        missing = self.missing_stems()
        if not missing:
            logger.info("feature cache: all %d entries present", len(self.manifest.stems))
            return 0

        logger.info(
            "feature cache: encoding %d missing entries (out of %d total)",
            len(missing),
            len(self.manifest.stems),
        )
        bundle = self._bundle_lazy()
        spec = bundle.bucket_spec
        d_enc = bundle.d_enc

        ds = _ResizeDataset(
            stems=[self.manifest.stems[i] for i in missing],
            image_paths=[self.manifest.image_paths[i] for i in missing],
            spec=spec,
            resized_cache_dir=self.resized_cache_dir,
        )
        loader = DataLoader(
            ds,
            batch_size=1,
            num_workers=self.num_workers,
            prefetch_factor=2 if self.num_workers > 0 else None,
            collate_fn=_identity_collate,
            pin_memory=False,
        )

        n_done = 0
        for stem, tensor, err in tqdm(loader, desc="pooled-pe", unit="img", total=len(ds)):
            if tensor is None:
                logger.warning("failed to decode %s: %s", stem, err)
                continue
            try:
                tensor = tensor.unsqueeze(0)
                feats_list = encode_pe_from_imageminus1to1(
                    bundle, tensor, same_bucket=True
                )
                feats = feats_list[0]                      # [T, d_enc]
                pooled = feats.mean(dim=0).to(torch.float32).cpu()  # [d_enc]
                assert pooled.shape == (d_enc,), pooled.shape
                st_save({"feature": pooled}, str(_cache_path(self.cache_dir, stem)))
                n_done += 1
            except Exception as e:
                logger.warning("failed to encode %s: %s", stem, e)
        logger.info("feature cache: wrote %d new entries", n_done)
        return n_done


# ── Dataset for the trainer ───────────────────────────────────────────────


class CachedFeatureDataset(Dataset):
    """In-memory ``(feature, multi_hot, rating_idx, people_idx)`` tuples.

    Loads every cached feature for the requested stems into one tensor at
    init. The full training feature tensor at 12K × 1024 × float32 is ~48
    MB — small enough that we don't bother with lazy disk reads.
    """

    def __init__(
        self,
        manifest: TaggerManifest,
        cache_dir: Path,
        stems_subset: Optional[Sequence[str]] = None,
    ):
        idx_of = manifest.stem_index()
        if stems_subset is None:
            stems_subset = manifest.stems
        kept_stems: List[str] = []
        kept_features: List[torch.Tensor] = []
        kept_tag_idx: List[List[int]] = []
        kept_rating_idx: List[int] = []
        kept_people_idx: List[int] = []
        # Old manifests built before the people-count head: per-stem labels
        # not present. Populate with zeros so positional unpacking stays sound;
        # the trainer detects n_people_counts==0 and skips the people loss.
        has_people = bool(manifest.people_count_indices)
        n_missing = 0
        for stem in stems_subset:
            i = idx_of.get(stem)
            if i is None:
                n_missing += 1
                continue
            cache_file = _cache_path(cache_dir, stem)
            if not cache_file.exists():
                n_missing += 1
                continue
            t = st_load(str(cache_file))["feature"]
            kept_stems.append(stem)
            kept_features.append(t)
            kept_tag_idx.append(manifest.tag_indices[i])
            kept_rating_idx.append(manifest.rating_indices[i])
            kept_people_idx.append(manifest.people_count_indices[i] if has_people else 0)
        if not kept_stems:
            raise RuntimeError(
                f"no cached features found in {cache_dir} for the requested "
                f"stems (n_requested={len(stems_subset)}, n_missing={n_missing})"
            )
        if n_missing:
            logger.warning(
                "CachedFeatureDataset: %d stems missing from cache (out of %d "
                "requested) - they will not contribute to training",
                n_missing,
                len(stems_subset),
            )
        self.stems = kept_stems
        self.features = torch.stack(kept_features, dim=0)               # [N, d]
        self.multi_hot = torch.zeros(len(kept_stems), manifest.n_tags)  # [N, T]
        for row, idxs in enumerate(kept_tag_idx):
            self.multi_hot[row, idxs] = 1.0
        self.rating_idx = torch.tensor(kept_rating_idx, dtype=torch.long)
        self.people_idx = torch.tensor(kept_people_idx, dtype=torch.long)
        self.n_tags = manifest.n_tags
        self.n_ratings = manifest.n_ratings
        self.n_people_counts = manifest.n_people_counts
        self.d_in = self.features.shape[-1]

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int):
        return (
            self.features[idx],
            self.multi_hot[idx],
            self.rating_idx[idx],
            self.people_idx[idx],
        )


# ── Token cache (for MAP-pool / pool_kind="map" training) ────────────────


def _token_cache_path(cache_dir: Path, stem: str) -> Path:
    return cache_dir / f"{stem}.safetensors"


class TokenCacheBuilder:
    """Build per-stem full-token PE-Core caches into ``cache_dir``.

    Writes each stem as ``{"tokens": bf16 [T, d_enc]}`` with the encoder's
    native CLS token at row 0 (use_cls=True). T varies per aspect-bucket
    (~576–588 for PE-Core-L14-336).

    Storage per stem ≈ ``T * d_enc * 2`` bytes; ~1.2 MB at PE-Core defaults.
    At 12K stems that's ~14 GB total — pay once, iterate on pool design
    freely. Per-image encoder forwards are fast enough that 12K stems
    finishes in ~10–20 minutes on a single GPU.

    ``resized_cache_dir`` (optional) — when supplied and the per-stem
    uint8 sidecar exists, skips PIL decode + LANCZOS for that stem and
    loads the cached tensor instead. Lets the cache build ride on the
    PE-LoRA path's ``.cache/resized-<encoder>/`` for free.
    """

    def __init__(
        self,
        manifest: TaggerManifest,
        cache_dir: Path,
        device: torch.device,
        encoder_name: str = "pe",
        dtype: torch.dtype = torch.bfloat16,
        num_workers: int = 4,
        resized_cache_dir: Optional[Path] = None,
    ):
        self.manifest = manifest
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.encoder_name = encoder_name
        self.dtype = dtype
        self.num_workers = num_workers
        self.resized_cache_dir = resized_cache_dir
        self._bundle: Optional[VisionEncoderBundle] = None

    def _bundle_lazy(self) -> VisionEncoderBundle:
        if self._bundle is None:
            self._bundle = load_pe_encoder(
                self.device, name=self.encoder_name, dtype=self.dtype
            )
        return self._bundle

    def missing_stems(self) -> List[int]:
        return [
            i
            for i, stem in enumerate(self.manifest.stems)
            if not _token_cache_path(self.cache_dir, stem).exists()
        ]

    @torch.no_grad()
    def build(self) -> int:
        """Encode + cache every stem missing from ``cache_dir``.

        Returns the count of newly cached entries (0 if everything was
        already cached). Errors on individual images are logged and the
        loop continues — a single corrupt image shouldn't tank the run.
        """
        missing = self.missing_stems()
        if not missing:
            logger.info("token cache: all %d entries present", len(self.manifest.stems))
            return 0

        logger.info(
            "token cache: encoding %d missing entries (out of %d total)",
            len(missing),
            len(self.manifest.stems),
        )
        bundle = self._bundle_lazy()
        spec = bundle.bucket_spec
        d_enc = bundle.d_enc

        ds = _ResizeDataset(
            stems=[self.manifest.stems[i] for i in missing],
            image_paths=[self.manifest.image_paths[i] for i in missing],
            spec=spec,
            resized_cache_dir=self.resized_cache_dir,
        )
        loader = DataLoader(
            ds,
            batch_size=1,
            num_workers=self.num_workers,
            prefetch_factor=2 if self.num_workers > 0 else None,
            collate_fn=_identity_collate,
            pin_memory=False,
        )

        n_done = 0
        for stem, tensor, err in tqdm(loader, desc="tokens-pe", unit="img", total=len(ds)):
            if tensor is None:
                logger.warning("failed to decode %s: %s", stem, err)
                continue
            try:
                tensor = tensor.unsqueeze(0)
                feats_list = encode_pe_from_imageminus1to1(
                    bundle, tensor, same_bucket=True
                )
                feats = feats_list[0]                              # [T, d_enc]
                assert feats.shape[-1] == d_enc, feats.shape
                # Stash as bf16 — encoder's native dtype, halves cache size
                # vs fp32 with no quality loss on the downstream pool (the
                # MAP head's LayerNorm + Linear projects back into fp32 /
                # autocast-bf16 anyway).
                tokens = feats.to(torch.bfloat16).cpu().contiguous()
                st_save(
                    {"tokens": tokens},
                    str(_token_cache_path(self.cache_dir, stem)),
                )
                n_done += 1
            except Exception as e:
                logger.warning("failed to encode %s: %s", stem, e)
        logger.info("token cache: wrote %d new entries", n_done)
        return n_done


class CachedTokenDataset(Dataset):
    """Lazy per-stem ``(tokens [T,D], multi_hot, rating_idx, people_idx, bucket_key)``.

    Loads each token sidecar on demand (the full corpus is ~14 GB at
    PE-Core-L14-336, too big for RAM-resident eager loading on most
    cards). Pairs with :class:`BucketBatchSampler` so each yielded batch
    is bucket-homogeneous — within-bucket T is constant, so the collate
    just stacks without padding masks.

    ``bucket_key`` is the ``(h_patches, w_patches)`` tuple derived from
    each cache file's stored token count (read from the safetensors
    header, no full tensor read at init time).
    """

    def __init__(
        self,
        manifest: TaggerManifest,
        cache_dir: Path,
        spec: BucketSpec,
        stems_subset: Optional[Sequence[str]] = None,
    ):
        idx_of = manifest.stem_index()
        if stems_subset is None:
            stems_subset = manifest.stems
        kept_stems: List[str] = []
        kept_paths: List[Path] = []
        kept_tag_idx: List[List[int]] = []
        kept_rating_idx: List[int] = []
        kept_people_idx: List[int] = []
        kept_buckets: List[tuple[int, int]] = []
        # Map T (token count) → bucket. Only buckets that match the
        # encoder's spec are kept; unexpected token counts are flagged.
        token_to_bucket: Dict[int, tuple[int, int]] = {
            h * w + (1 if spec.use_cls else 0): (h, w) for h, w in spec.buckets
        }
        has_people = bool(manifest.people_count_indices)
        n_missing = 0
        n_unknown_bucket = 0
        for stem in stems_subset:
            i = idx_of.get(stem)
            if i is None:
                n_missing += 1
                continue
            cache_file = _token_cache_path(cache_dir, stem)
            if not cache_file.exists():
                n_missing += 1
                continue
            with safe_open(str(cache_file), framework="pt") as f:
                shape = f.get_slice("tokens").get_shape()
            T = int(shape[0])
            bucket = token_to_bucket.get(T)
            if bucket is None:
                n_unknown_bucket += 1
                continue
            kept_stems.append(stem)
            kept_paths.append(cache_file)
            kept_tag_idx.append(manifest.tag_indices[i])
            kept_rating_idx.append(manifest.rating_indices[i])
            kept_people_idx.append(manifest.people_count_indices[i] if has_people else 0)
            kept_buckets.append(bucket)
        if not kept_stems:
            raise RuntimeError(
                f"no cached token sidecars found in {cache_dir} for the requested "
                f"stems (n_requested={len(stems_subset)}, n_missing={n_missing}, "
                f"n_unknown_bucket={n_unknown_bucket})"
            )
        if n_missing:
            logger.warning(
                "CachedTokenDataset: %d stems missing from cache (out of %d "
                "requested) - they will not contribute to training",
                n_missing,
                len(stems_subset),
            )
        if n_unknown_bucket:
            logger.warning(
                "CachedTokenDataset: %d stems had unexpected token counts "
                "(not in spec.buckets) and were skipped",
                n_unknown_bucket,
            )
        self.stems = kept_stems
        self.paths = kept_paths
        self.buckets = kept_buckets
        self.multi_hot = torch.zeros(len(kept_stems), manifest.n_tags)
        for row, idxs in enumerate(kept_tag_idx):
            self.multi_hot[row, idxs] = 1.0
        self.rating_idx = torch.tensor(kept_rating_idx, dtype=torch.long)
        self.people_idx = torch.tensor(kept_people_idx, dtype=torch.long)
        self.n_tags = manifest.n_tags
        self.n_ratings = manifest.n_ratings
        self.n_people_counts = manifest.n_people_counts
        self.spec = spec
        # d_in is the token D — peek the first file to read it.
        with safe_open(str(kept_paths[0]), framework="pt") as f:
            self.d_in = int(f.get_slice("tokens").get_shape()[-1])

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int):
        tokens = st_load(str(self.paths[idx]))["tokens"]    # [T, D] bf16
        return (
            tokens,
            self.multi_hot[idx],
            self.rating_idx[idx],
            self.people_idx[idx],
            self.buckets[idx],
        )


def collate_token_batch(batch):
    """Stack a same-bucket batch into ``(tokens[B,T,D], multi_hot, rating, people, bucket)``.

    Asserts shape homogeneity (BucketBatchSampler guarantees same T) and
    leaves the tokens as bf16; trainer casts as needed under autocast.
    """
    tokens = torch.stack([b[0] for b in batch], dim=0)        # [B, T, D] bf16
    multi_hot = torch.stack([b[1] for b in batch], dim=0)     # [B, n_tags]
    rating_idx = torch.stack([b[2] for b in batch], dim=0)    # [B]
    people_idx = torch.stack([b[3] for b in batch], dim=0)    # [B]
    bucket = batch[0][4]
    return tokens, multi_hot, rating_idx, people_idx, bucket


# ── Dual-encoder cache (PE-Core + PE-Spatial; per-side pool kind) ────────


class CachedDualDataset(Dataset):
    """Lazy per-stem ``(feat_main, feat_aux, multi_hot, rating, people, bucket_pair)``.

    Each side independently picks ``"mean"`` or ``"map"``:
      * ``"mean"`` → load pooled cache (``{stem}.safetensors`` with key
        ``"feature"``), shape ``[D]``. Bucket key is ``None`` for that side
        (no shape variation → no batch-bucketing constraint).
      * ``"map"`` → load token cache (``{stem}.safetensors`` with key
        ``"tokens"``), shape ``[T, D]``. Bucket key is ``(h_p, w_p)`` derived
        from T via the encoder's :class:`BucketSpec`.

    The compound bucket key is ``(main_bucket | None, aux_bucket | None)``;
    :class:`BucketBatchSampler` groups by it, so within-batch shape
    homogeneity holds for whichever side(s) need it. When PE-Core uses
    mean and PE-Spatial uses map (the most common asymmetric setup),
    batches are grouped by aux bucket only.

    ``cache_dir`` / ``cache_dir_aux`` should be the per-side outputs of
    :func:`scripts.anima_tagger.caches.cache_dir_for` (i.e.
    ``.cache/pooled-<encoder>/`` for mean, ``.cache/tokens-<encoder>/`` for
    map). Spec is only used when the side is map (to map T → bucket); pass
    ``None`` for the spec on a mean side.

    Stems present in only one cache are skipped (logged); typical cause is
    an asymmetric incremental cache build.
    """

    def __init__(
        self,
        manifest: TaggerManifest,
        cache_dir: Path,
        pool_kind: str,
        spec: Optional[BucketSpec],
        cache_dir_aux: Path,
        pool_kind_aux: str,
        spec_aux: Optional[BucketSpec],
        stems_subset: Optional[Sequence[str]] = None,
    ):
        if pool_kind not in ("mean", "map"):
            raise ValueError(f"pool_kind must be 'mean' or 'map', got {pool_kind!r}")
        if pool_kind_aux not in ("mean", "map"):
            raise ValueError(f"pool_kind_aux must be 'mean' or 'map', got {pool_kind_aux!r}")
        if pool_kind == "map" and spec is None:
            raise ValueError("pool_kind='map' requires a BucketSpec for the main side")
        if pool_kind_aux == "map" and spec_aux is None:
            raise ValueError("pool_kind_aux='map' requires a BucketSpec for the aux side")

        idx_of = manifest.stem_index()
        if stems_subset is None:
            stems_subset = manifest.stems

        # Per-side T → bucket map (only used on map sides). For mean sides
        # the dict is empty and the bucket key is fixed at None.
        def _bucket_map(spec: Optional[BucketSpec]) -> Dict[int, tuple[int, int]]:
            if spec is None:
                return {}
            return {
                h * w + (1 if spec.use_cls else 0): (h, w) for h, w in spec.buckets
            }

        token_to_bucket = _bucket_map(spec)
        token_to_bucket_aux = _bucket_map(spec_aux)

        kept_stems: List[str] = []
        kept_paths: List[Path] = []
        kept_paths_aux: List[Path] = []
        kept_tag_idx: List[List[int]] = []
        kept_rating_idx: List[int] = []
        kept_people_idx: List[int] = []
        kept_buckets: List[tuple[Optional[tuple[int, int]], Optional[tuple[int, int]]]] = []
        has_people = bool(manifest.people_count_indices)
        n_missing_main = 0
        n_missing_aux = 0
        n_unknown_bucket = 0
        for stem in stems_subset:
            i = idx_of.get(stem)
            if i is None:
                n_missing_main += 1
                continue
            # Both pooled and token caches use the same per-stem filename;
            # the differentiator is the cache_dir (.cache/pooled-X/ vs
            # .cache/tokens-X/) and the safetensors key inside.
            cache_file = cache_dir / f"{stem}.safetensors"
            cache_file_aux = cache_dir_aux / f"{stem}.safetensors"
            if not cache_file.exists():
                n_missing_main += 1
                continue
            if not cache_file_aux.exists():
                n_missing_aux += 1
                continue
            bucket_main: Optional[tuple[int, int]] = None
            if pool_kind == "map":
                with safe_open(str(cache_file), framework="pt") as f:
                    T_main = int(f.get_slice("tokens").get_shape()[0])
                bucket_main = token_to_bucket.get(T_main)
                if bucket_main is None:
                    n_unknown_bucket += 1
                    continue
            bucket_aux: Optional[tuple[int, int]] = None
            if pool_kind_aux == "map":
                with safe_open(str(cache_file_aux), framework="pt") as f:
                    T_aux = int(f.get_slice("tokens").get_shape()[0])
                bucket_aux = token_to_bucket_aux.get(T_aux)
                if bucket_aux is None:
                    n_unknown_bucket += 1
                    continue
            kept_stems.append(stem)
            kept_paths.append(cache_file)
            kept_paths_aux.append(cache_file_aux)
            kept_tag_idx.append(manifest.tag_indices[i])
            kept_rating_idx.append(manifest.rating_indices[i])
            kept_people_idx.append(manifest.people_count_indices[i] if has_people else 0)
            kept_buckets.append((bucket_main, bucket_aux))
        if not kept_stems:
            raise RuntimeError(
                f"no paired sidecars in {cache_dir} + {cache_dir_aux} for the "
                f"requested stems (n_requested={len(stems_subset)}, "
                f"n_missing_main={n_missing_main}, n_missing_aux={n_missing_aux}, "
                f"n_unknown_bucket={n_unknown_bucket})"
            )
        if n_missing_main or n_missing_aux:
            logger.warning(
                "CachedDualDataset: missing main=%d aux=%d (out of %d "
                "requested) - those stems are skipped",
                n_missing_main, n_missing_aux, len(stems_subset),
            )
        if n_unknown_bucket:
            logger.warning(
                "CachedDualDataset: %d stems had unexpected token counts "
                "(not in spec.buckets) and were skipped",
                n_unknown_bucket,
            )
        self.stems = kept_stems
        self.paths = kept_paths
        self.paths_aux = kept_paths_aux
        self.buckets = kept_buckets
        self.pool_kind = pool_kind
        self.pool_kind_aux = pool_kind_aux
        self.multi_hot = torch.zeros(len(kept_stems), manifest.n_tags)
        for row, idxs in enumerate(kept_tag_idx):
            self.multi_hot[row, idxs] = 1.0
        self.rating_idx = torch.tensor(kept_rating_idx, dtype=torch.long)
        self.people_idx = torch.tensor(kept_people_idx, dtype=torch.long)
        self.n_tags = manifest.n_tags
        self.n_ratings = manifest.n_ratings
        self.n_people_counts = manifest.n_people_counts
        self.spec = spec
        self.spec_aux = spec_aux
        # Peek the first sidecar of each side to record d_in / d_in_aux.
        # Key differs by pool_kind ("feature" for mean, "tokens" for map).
        self.d_in = self._peek_d(kept_paths[0], pool_kind)
        self.d_in_aux = self._peek_d(kept_paths_aux[0], pool_kind_aux)

    @staticmethod
    def _peek_d(path: Path, kind: str) -> int:
        key = "feature" if kind == "mean" else "tokens"
        with safe_open(str(path), framework="pt") as f:
            return int(f.get_slice(key).get_shape()[-1])

    def __len__(self) -> int:
        return len(self.stems)

    @staticmethod
    def _load_one(path: Path, kind: str) -> torch.Tensor:
        key = "feature" if kind == "mean" else "tokens"
        return st_load(str(path))[key]

    def __getitem__(self, idx: int):
        feat = self._load_one(self.paths[idx], self.pool_kind)
        feat_aux = self._load_one(self.paths_aux[idx], self.pool_kind_aux)
        return (
            feat,
            feat_aux,
            self.multi_hot[idx],
            self.rating_idx[idx],
            self.people_idx[idx],
            self.buckets[idx],
        )


# Back-compat alias — earlier code (and the smoke tests) refer to the
# original name. The new class generalizes the original; callers that
# imported the old name keep working.
CachedDualTokenDataset = CachedDualDataset


def collate_dual_token_batch(batch):
    """Stack a same-bucket-pair batch into
    ``(feat_main, feat_aux, multi_hot, rating, people, bucket_pair)``.

    BucketBatchSampler guarantees both shapes are constant within a batch
    (sampler groups by the compound ``(main_bucket | None, aux_bucket | None)``
    key). torch.stack works whether each side is rank-2 (mean-pool) or
    rank-3 (token sequence).
    """
    feat = torch.stack([b[0] for b in batch], dim=0)          # [B, ...] (rank depends on side)
    feat_aux = torch.stack([b[1] for b in batch], dim=0)      # [B, ...]
    multi_hot = torch.stack([b[2] for b in batch], dim=0)     # [B, n_tags]
    rating_idx = torch.stack([b[3] for b in batch], dim=0)    # [B]
    people_idx = torch.stack([b[4] for b in batch], dim=0)    # [B]
    bucket_pair = batch[0][5]
    return feat, feat_aux, multi_hot, rating_idx, people_idx, bucket_pair


# ── Pre-resized image cache (for end-to-end PE-LoRA training) ─────────────


def _image_cache_path(cache_dir: Path, stem: str) -> Path:
    return cache_dir / f"{stem}.safetensors"


class ImageCacheBuilder:
    """LANCZOS-resize each manifest image to its PE bucket and cache as uint8.

    Pairs with end-to-end PE-LoRA training where the encoder is unfrozen
    and pre-pooled features can't track it (mirrors the IP-Adapter
    "cached features can't track a moving encoder" pattern). Doing the
    LANCZOS-resize once up front keeps the train-time dataloader to a
    cheap ``st_load`` + integer→float cast.

    Storage: one safetensors per stem, ``{"image": uint8 [C, H, W]}``.
    Pixel range is 0..255; the trainer recovers ``[-1, 1]`` via
    ``(x/127.5) - 1`` (equivalent to ``ToTensor + Normalize(0.5, 0.5)``).
    Bucket can be derived from H/W at load time, so it's not stored.
    """

    def __init__(
        self,
        manifest: TaggerManifest,
        cache_dir: Path,
        spec: BucketSpec,
        num_workers: int = 6,
    ):
        self.manifest = manifest
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.spec = spec
        self.num_workers = num_workers

    def missing_stems(self) -> List[int]:
        return [
            i
            for i, stem in enumerate(self.manifest.stems)
            if not _image_cache_path(self.cache_dir, stem).exists()
        ]

    def build(self) -> int:
        """Resize + cache every stem missing from ``cache_dir``."""
        missing = self.missing_stems()
        if not missing:
            logger.info(
                "image cache: all %d entries present", len(self.manifest.stems)
            )
            return 0
        logger.info(
            "image cache: resizing %d missing entries (out of %d total)",
            len(missing),
            len(self.manifest.stems),
        )

        ds = _ResizeDataset(
            stems=[self.manifest.stems[i] for i in missing],
            image_paths=[self.manifest.image_paths[i] for i in missing],
            spec=self.spec,
        )
        loader = DataLoader(
            ds,
            batch_size=1,
            num_workers=self.num_workers,
            prefetch_factor=2 if self.num_workers > 0 else None,
            collate_fn=_identity_collate,
            pin_memory=False,
        )

        n_done = 0
        for stem, tensor, err in tqdm(
            loader, desc="resized-pe", unit="img", total=len(ds)
        ):
            if tensor is None:
                logger.warning("failed to decode %s: %s", stem, err)
                continue
            try:
                # _ResizeDataset returns the tensor through IMAGE_TRANSFORMS
                # already (i.e. [-1, 1] float). Recover uint8 [0..255] for
                # disk-cheap storage.
                u8 = ((tensor.clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(torch.uint8)
                st_save({"image": u8}, str(_image_cache_path(self.cache_dir, stem)))
                n_done += 1
            except Exception as e:
                logger.warning("failed to cache %s: %s", stem, e)
        logger.info("image cache: wrote %d new entries", n_done)
        return n_done


class CachedImageDataset(Dataset):
    """Per-stem ``(image_uint8, multi_hot, rating_idx, people_idx, bucket_key)``.

    Loads the cached uint8 tensor lazily per ``__getitem__`` (faster than
    holding ~4 GB of images in RAM for 12k stems at PE-Core-L14-336).
    Returned image is uint8 ``[C, H, W]`` — the trainer converts to float
    [-1, 1] and stacks per-batch.

    ``bucket_key`` is the ``(h_patches, w_patches)`` tuple that the
    sampler uses to keep batches shape-homogeneous.
    """

    def __init__(
        self,
        manifest: TaggerManifest,
        cache_dir: Path,
        spec: BucketSpec,
        stems_subset: Optional[Sequence[str]] = None,
    ):
        idx_of = manifest.stem_index()
        if stems_subset is None:
            stems_subset = manifest.stems
        kept_stems: List[str] = []
        kept_paths: List[Path] = []
        kept_tag_idx: List[List[int]] = []
        kept_rating_idx: List[int] = []
        kept_people_idx: List[int] = []
        kept_buckets: List[tuple[int, int]] = []
        has_people = bool(manifest.people_count_indices)
        n_missing = 0
        for stem in stems_subset:
            i = idx_of.get(stem)
            if i is None:
                n_missing += 1
                continue
            cache_file = _image_cache_path(cache_dir, stem)
            if not cache_file.exists():
                n_missing += 1
                continue
            # Read the tensor *shape* only via the safetensors header —
            # avoids ~4 GB of one-shot reads for 12k stems at init time.
            with safe_open(str(cache_file), framework="pt") as f:
                shape = f.get_slice("image").get_shape()
            _, h_pix, w_pix = shape
            h_p, w_p = h_pix // spec.patch, w_pix // spec.patch
            kept_stems.append(stem)
            kept_paths.append(cache_file)
            kept_tag_idx.append(manifest.tag_indices[i])
            kept_rating_idx.append(manifest.rating_indices[i])
            kept_people_idx.append(manifest.people_count_indices[i] if has_people else 0)
            kept_buckets.append((h_p, w_p))
        if not kept_stems:
            raise RuntimeError(
                f"no cached images found in {cache_dir} for the requested "
                f"stems (n_requested={len(stems_subset)}, n_missing={n_missing})"
            )
        if n_missing:
            logger.warning(
                "CachedImageDataset: %d stems missing from cache (out of %d "
                "requested) - they will not contribute to training",
                n_missing,
                len(stems_subset),
            )
        self.stems = kept_stems
        self.paths = kept_paths
        self.buckets = kept_buckets
        self.multi_hot = torch.zeros(len(kept_stems), manifest.n_tags)
        for row, idxs in enumerate(kept_tag_idx):
            self.multi_hot[row, idxs] = 1.0
        self.rating_idx = torch.tensor(kept_rating_idx, dtype=torch.long)
        self.people_idx = torch.tensor(kept_people_idx, dtype=torch.long)
        self.n_tags = manifest.n_tags
        self.n_ratings = manifest.n_ratings
        self.n_people_counts = manifest.n_people_counts
        self.spec = spec

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int):
        u8 = st_load(str(self.paths[idx]))["image"]
        return (
            u8,
            self.multi_hot[idx],
            self.rating_idx[idx],
            self.people_idx[idx],
            self.buckets[idx],
        )


class BucketBatchSampler(Sampler[List[int]]):
    """Yields batches of indices that share a single bucket.

    Within each epoch: shuffle the per-bucket index pools, chunk into
    batches of ``batch_size`` (drop_last=False — partial trailing batches
    are kept since dataset sizes don't divide evenly), then shuffle the
    batch order across buckets so the encoder doesn't see all of one
    aspect ratio in a row.
    """

    def __init__(
        self,
        buckets: Sequence[tuple[int, int]],
        batch_size: int,
        seed: int = 42,
        shuffle: bool = True,
    ):
        self.buckets = list(buckets)
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self._epoch = 0
        # Group sample indices by bucket key.
        self._by_bucket: Dict[tuple[int, int], List[int]] = defaultdict(list)
        for i, b in enumerate(self.buckets):
            self._by_bucket[b].append(i)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __iter__(self) -> Iterator[List[int]]:
        rng = torch.Generator().manual_seed(self.seed + self._epoch)
        all_batches: List[List[int]] = []
        for _, idxs in sorted(self._by_bucket.items()):
            order = idxs[:]
            if self.shuffle:
                perm = torch.randperm(len(order), generator=rng).tolist()
                order = [order[k] for k in perm]
            for s in range(0, len(order), self.batch_size):
                all_batches.append(order[s : s + self.batch_size])
        if self.shuffle:
            perm = torch.randperm(len(all_batches), generator=rng).tolist()
            all_batches = [all_batches[k] for k in perm]
        yield from all_batches

    def __len__(self) -> int:
        n = 0
        for idxs in self._by_bucket.values():
            n += (len(idxs) + self.batch_size - 1) // self.batch_size
        return n


def collate_image_batch(batch):
    """Stack a same-bucket batch into ``(image[B,C,H,W], multi_hot, rating, people, bucket)``.

    Asserts shape homogeneity — the bucket sampler is expected to produce
    only same-bucket batches. The image tensor is left as ``uint8``; the
    trainer does the [-1, 1] cast on-device.
    """
    images = torch.stack([b[0] for b in batch], dim=0)            # [B, C, H, W] uint8
    multi_hot = torch.stack([b[1] for b in batch], dim=0)         # [B, T]
    rating_idx = torch.stack([b[2] for b in batch], dim=0)        # [B]
    people_idx = torch.stack([b[3] for b in batch], dim=0)        # [B]
    bucket = batch[0][4]
    return images, multi_hot, rating_idx, people_idx, bucket
