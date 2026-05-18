"""CMMD (CLIP-MMD) for validation-phase quality scoring.

Port of google-research/cmmd/distance.py to torch, with PE-Core swapped in
for CLIP ViT-L/14. PE is a CLIP-family L/14 vision tower that Anima already
conditions on, so distances measured here live in the same feature space
the model targets at training time.

Pipeline at each val pass:

1. Load cached reference PE features from ``{stem}_anima_pe.safetensors``
   sidecars (produced by ``preprocess/cache_pe_encoder.py``) for every
   image in the validation split.
2. For each val item, sample an image using the live model + cached prompt
   embedding (paired by stem with the reference).
3. Run the generated pixel tensor through the live PE encoder.
4. Mean-pool over patch tokens (drop CLS), L2-normalize, compute MMD².

The Gaussian-kernel MMD² formula matches the reference repo exactly:

    MMD²(X, Y) = E_{x,x'}[k(x, x')] + E_{y,y'}[k(y, y')] - 2 E_{x,y}[k(x, y)]
    k(a, b) = exp(-‖a - b‖² / (2σ²))

with ``σ=10`` and an overall ``×1000`` scale (so values land in a readable
range — these are the original CMMD paper constants).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

import torch
from safetensors.torch import load_file

logger = logging.getLogger(__name__)

# Original CMMD paper / repo constants (CLIP-L/14 features, L2-normalized).
_SIGMA = 10.0
_SCALE = 1000.0


def _pool_pe(feats: torch.Tensor, *, drop_cls: bool = True) -> torch.Tensor:
    """Mean over patch tokens. ``feats`` is ``[T, D]``; returns ``[D]``.

    Matches ``preprocess/cache_pe_encoder.py:_pool_pe`` so the CMMD reference
    pool is comparable to the IP-Adapter centroid.
    """
    if drop_cls and feats.shape[0] > 1:
        feats = feats[1:]
    return feats.mean(dim=0)


def _l2_normalize(x: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True).clamp_min(eps))


def pool_and_normalize(feats: torch.Tensor) -> torch.Tensor:
    """Pool a single ``[T, D]`` PE feature map down to a unit ``[D]`` vector."""
    return _l2_normalize(_pool_pe(feats.to(torch.float32)))


def resolve_pe_sidecar(
    image_path: Path | str,
    *,
    encoder: str = "pe",
    cache_dir: Path | str | None = None,
) -> Path:
    """Match ``preprocess/cache_pe_encoder.cache_path_for`` so the val pass
    looks for sidecars in the same place the cache step wrote them."""
    image_path = Path(image_path)
    name = f"{image_path.stem}_anima_{encoder}.safetensors"
    if cache_dir is not None:
        return Path(cache_dir) / name
    return image_path.with_name(name)


def load_reference_features(
    sidecar_paths: Iterable[Path | str],
) -> torch.Tensor:
    """Pool + L2-normalize cached PE features.

    ``sidecar_paths`` are absolute paths to ``{stem}_anima_{encoder}.safetensors``
    files (caller resolves via :func:`resolve_pe_sidecar` or from each
    ``ImageInfo``'s ``text_encoder_outputs_npz`` sibling). Returns ``[N, D]``
    of unit-norm float32 vectors. Skips entries with no sidecar and logs
    a warning; raises if nothing loads.
    """
    sidecar_paths = list(sidecar_paths)
    pooled: list[torch.Tensor] = []
    missing = 0
    for sidecar in sidecar_paths:
        sidecar = Path(sidecar)
        if not sidecar.exists():
            missing += 1
            continue
        sd = load_file(str(sidecar))
        feats = sd.get("image_features")
        if feats is None:
            missing += 1
            continue
        pooled.append(pool_and_normalize(feats))
    if not pooled:
        raise RuntimeError(
            f"CMMD: no PE feature sidecars found for {len(sidecar_paths)} val "
            "images. Run `make preprocess-pe` first."
        )
    if missing:
        logger.warning(
            f"CMMD: {missing}/{len(sidecar_paths)} val items had no PE "
            "sidecar; skipped. Run `make preprocess-pe` to fill the cache."
        )
    return torch.stack(pooled, dim=0)


def mmd_gaussian(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    sigma: float = _SIGMA,
    scale: float = _SCALE,
) -> torch.Tensor:
    """Gaussian-kernel MMD² between two empirical feature sets.

    ``x``: ``[N, D]``, ``y``: ``[M, D]`` (both already L2-normalized, both
    in float32 on the same device). Returns a scalar tensor.
    """
    x = x.to(torch.float32)
    y = y.to(torch.float32)
    gamma = 1.0 / (2.0 * sigma * sigma)

    x_sq = (x * x).sum(dim=1)
    y_sq = (y * y).sum(dim=1)

    # Pairwise squared distances via ‖a-b‖² = ‖a‖² + ‖b‖² − 2 a·b.
    d_xx = x_sq.unsqueeze(1) + x_sq.unsqueeze(0) - 2.0 * (x @ x.t())
    d_yy = y_sq.unsqueeze(1) + y_sq.unsqueeze(0) - 2.0 * (y @ y.t())
    d_xy = x_sq.unsqueeze(1) + y_sq.unsqueeze(0) - 2.0 * (x @ y.t())

    # Negative values can creep in via float rounding on tiny distances.
    d_xx = d_xx.clamp_min(0.0)
    d_yy = d_yy.clamp_min(0.0)
    d_xy = d_xy.clamp_min(0.0)

    k_xx = torch.exp(-gamma * d_xx).mean()
    k_yy = torch.exp(-gamma * d_yy).mean()
    k_xy = torch.exp(-gamma * d_xy).mean()

    return scale * (k_xx + k_yy - 2.0 * k_xy)


def cmmd_from_pools(
    ref_pool: torch.Tensor,
    gen_pool: torch.Tensor,
    *,
    sigma: float = _SIGMA,
    scale: float = _SCALE,
) -> float:
    """Convenience wrapper returning a python float."""
    return float(mmd_gaussian(ref_pool, gen_pool, sigma=sigma, scale=scale).item())


def encode_pixel_batch(
    bundle,
    images_minus1to1: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Pool + L2-normalize PE features for a list of ``[3, H, W]`` tensors.

    Each tensor may have its own (H, W) — we call the encoder one-at-a-time
    so PE-Core's per-image bucket selection runs normally. Returns ``[K, D]``
    float32 on CPU; the caller stacks with the reference pool on the right
    device for ``mmd_gaussian``.
    """
    from library.vision.encoder import encode_pe_from_imageminus1to1

    pooled: list[torch.Tensor] = []
    for img in images_minus1to1:
        if img.dim() == 3:
            img = img.unsqueeze(0)
        feats_list = encode_pe_from_imageminus1to1(bundle, img, same_bucket=True)
        pooled.append(pool_and_normalize(feats_list[0]).cpu())
    return torch.stack(pooled, dim=0)
