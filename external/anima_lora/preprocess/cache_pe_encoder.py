#!/usr/bin/env python3
"""Cache PE-Core (or other registered vision-encoder) features.

Mirrors the live PE encoding done at training time so callers can read
patch-token features off disk instead of running the encoder every step.
Loads each pre-resized image from ``--dir`` in [-1, 1], picks the
encoder's nearest-aspect bucket, runs a single forward, and saves
``{stem}_anima_{encoder}.safetensors`` into ``--cache_dir`` (or alongside
the image when omitted). Skips already-cached entries (idempotent).

Wrapped by ``make preprocess-pe`` (reads ``post_image_dataset/resized/``,
writes ``post_image_dataset/lora/``). The same sidecars are consumed by the
LoRA / REPA pipeline and by IP-Adapter -- they share the cache directory.

The cache key matches what the encoder produces at training time:
``encode_pe_from_imageminus1to1(bundle, x, same_bucket=True)`` -> ``[T_pe, d_enc]``.
Variable T per encoder bucket; per-image stored as a single tensor (no padding).

Centroid sidecar
----------------

Pass ``--centroid`` to also emit ``anima_pe_centroid_{encoder}.safetensors``
(dataset-mean of mean-over-patch-tokens pooled features, ``[D]`` fp32) after
the cache pass. Pass ``--centroid_only`` to skip encoding entirely and just
pool existing caches under ``--cache_dir``. Consumed by IP-Adapter
(``ip_centroid_path``) and DCW v4 (``cos(c_pool, μ_centroid)`` channel) --
targets the participation-ratio-6 manifold collapse on this dataset (see
``bench/ip_adapter/analysis.md``).
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from library.datasets.image_utils import IMAGE_EXTENSIONS, IMAGE_TRANSFORMS
from library.vision.encoder import encode_pe_from_imageminus1to1, load_pe_encoder

ROOT = Path(__file__).resolve().parents[1]


def _pool_pe(feats: torch.Tensor, *, drop_cls: bool = True) -> torch.Tensor:
    """Mean over patch tokens. ``feats`` is ``[T, D]``; returns ``[D]``."""
    if drop_cls and feats.shape[0] > 1:
        feats = feats[1:]
    return feats.mean(dim=0)


def _write_centroid_sidecar(
    cache_dir: Path,
    out_path: Path,
    *,
    encoder: str,
    limit: int = 0,
) -> None:
    """Stream-pool cached PE features in ``cache_dir`` -> centroid sidecar.

    Walks ``cache_dir`` recursively so nested caches (mirroring the source
    subfolder structure) are included in the pool.
    """
    from safetensors.torch import load_file, save_file

    suffix = f"_anima_{encoder}.safetensors"
    files = sorted(p for p in cache_dir.rglob(f"*{suffix}") if p.is_file())
    files = [p for p in files if not p.name.startswith("anima_pe_centroid")]
    if not files:
        print(f"No '{suffix}' caches under {cache_dir}", file=sys.stderr)
        sys.exit(1)
    if limit > 0:
        files = files[:limit]

    print(f"\nCentroid pass: {len(files)} files under {cache_dir}")
    centroid: torch.Tensor | None = None
    n = 0
    for p in tqdm(files, desc="pooling"):
        sd = load_file(str(p))
        feats = sd.get("image_features")
        if feats is None:
            print(f"  skip {p.name}: no 'image_features' key", file=sys.stderr)
            continue
        pool = _pool_pe(feats.to(torch.float32))
        if centroid is None:
            centroid = torch.zeros_like(pool)
        centroid += pool
        n += 1

    if n == 0 or centroid is None:
        print("No usable PE features found.", file=sys.stderr)
        sys.exit(1)
    centroid = centroid / n

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"centroid": centroid.contiguous()},
        str(out_path),
        metadata={
            "encoder": encoder,
            "n_images": str(n),
            "d_enc": str(centroid.shape[0]),
            "pool": "mean_over_patch_tokens",
        },
    )
    print(
        f"centroid shape: {tuple(centroid.shape)}  "
        f"‖centroid‖={float(centroid.norm()):.3f}  "
        f"mean={float(centroid.mean()):.4f}  std={float(centroid.std()):.4f}"
    )
    print(f"wrote {out_path}")


def cache_path_for(
    image_path: Path,
    encoder: str,
    cache_dir: Path | None = None,
    image_dir: Path | None = None,
) -> Path:
    suffix = f"_anima_{encoder}.safetensors"
    if cache_dir is None:
        return image_path.with_name(image_path.stem + suffix)
    from library.io.cache import resolve_cache_path

    return Path(
        resolve_cache_path(
            str(image_path),
            suffix,
            cache_dir=str(cache_dir),
            image_dir=str(image_dir) if image_dir is not None else None,
        )
    )


class _PEImageGroup(Dataset):
    """Reads images from one ``(W, H)`` resolution group.

    Each ``__getitem__`` returns ``(str_path, str_out_path, [3, H, W] tensor in
    [-1, 1])`` so the main thread can write safetensors in batch order without
    holding the PIL.Image object across the worker boundary. We pass paths as
    strings (instead of ``Path``) because ``Path`` is picklable but heavier;
    safetensors' ``save_file`` takes a string anyway.
    """

    def __init__(self, paths: list[Path], out_paths: list[Path]):
        self._paths = [str(p) for p in paths]
        self._out_paths = [str(p) for p in out_paths]

    def __len__(self) -> int:
        return len(self._paths)

    def __getitem__(self, idx: int):
        p = self._paths[idx]
        with Image.open(p) as img:
            tensor = IMAGE_TRANSFORMS(np.array(img.convert("RGB")))
        return p, self._out_paths[idx], tensor


def _collate(batch):
    """Stack tensors into ``[B, 3, H, W]``; group already guarantees same shape."""
    paths, out_paths, tensors = zip(*batch)
    return list(paths), list(out_paths), torch.stack(tensors, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Dataset directory. Required unless --centroid_only is set.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help=(
            "Optional directory to write PE caches into (created if needed). "
            "Defaults to writing alongside each source image."
        ),
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default="pe",
        help="Vision encoder registry name (default: pe). See library/vision/encoders.py.",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="Override the encoder's default model id / checkpoint path.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Forward batch size within each (H, W) group (default: 8).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help=(
            "DataLoader workers for parallel PIL decode + transform. "
            "0 = single-threaded (decode on the main thread, GPU sits idle "
            "during decode + safetensors write). Default 4."
        ),
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Storage dtype for cached features (default: bfloat16, matches train-time).",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help=(
            "Walk subfolders under --dir. Caches mirror the source subdir "
            "structure under --cache_dir; stems must be unique within each "
            "subfolder but the same stem can repeat across folders."
        ),
    )
    parser.add_argument(
        "--centroid",
        action="store_true",
        help=(
            "After the cache pass, stream-pool all '_anima_{encoder}.safetensors' "
            "files under --cache_dir and emit a dataset-mean centroid sidecar "
            "consumed by IP-Adapter and DCW v4. Requires --cache_dir."
        ),
    )
    parser.add_argument(
        "--centroid_only",
        action="store_true",
        help=(
            "Skip encoding; just pool existing PE caches under --cache_dir and "
            "write the centroid sidecar. --cache_dir defaults to "
            "'post_image_dataset/lora' in this mode."
        ),
    )
    parser.add_argument(
        "--centroid_out",
        type=str,
        default=None,
        help=(
            "Output path for the centroid sidecar. Defaults to "
            "post_image_dataset/ip_adapter/anima_pe_centroid_{encoder}.safetensors "
            "(separate from the shared PE cache dir so LoRA stays untouched)."
        ),
    )
    parser.add_argument(
        "--centroid_limit",
        type=int,
        default=0,
        help="Cap the number of cache files pooled into the centroid (0 = all).",
    )
    args = parser.parse_args()

    if not args.centroid_only and args.dir is None:
        parser.error("--dir is required unless --centroid_only is set")
    if args.centroid and not args.cache_dir:
        parser.error(
            "--centroid needs --cache_dir (centroid pools files in a directory; "
            "alongside-image layout has no single dir to walk)"
        )

    from safetensors.torch import save_file as _save_safetensors

    cache_dir = Path(args.cache_dir) if args.cache_dir else None

    if args.centroid_only:
        centroid_cache_dir = cache_dir or (ROOT / "post_image_dataset" / "lora")
        if not centroid_cache_dir.is_absolute():
            centroid_cache_dir = (ROOT / centroid_cache_dir).resolve()
        if not centroid_cache_dir.is_dir():
            print(f"--cache_dir not found: {centroid_cache_dir}", file=sys.stderr)
            sys.exit(1)
        out_path = (
            Path(args.centroid_out)
            if args.centroid_out
            else ROOT
            / "post_image_dataset"
            / "ip_adapter"
            / f"anima_pe_centroid_{args.encoder}.safetensors"
        )
        _write_centroid_sidecar(
            centroid_cache_dir,
            out_path,
            encoder=args.encoder,
            limit=args.centroid_limit,
        )
        return

    data_dir = Path(args.dir)
    if not data_dir.is_dir():
        print(f"--dir not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    print(f"Loading vision encoder '{args.encoder}' on {device} ...")
    bundle = load_pe_encoder(device, name=args.encoder, model_id=args.model_id)
    print(
        f"  encoder={bundle.name} d_enc={bundle.d_enc} "
        f"patch={bundle.bucket_spec.patch} cls={bundle.bucket_spec.use_cls}"
    )

    # Group images by their post-resize pixel dimensions so a single forward
    # serves the whole group (same encoder bucket -> same T_pe -> same shape).
    if args.recursive:
        image_files = sorted(
            p
            for p in data_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        # Per-subdir uniqueness — see cache_latents.py for the rationale.
        stems: dict[tuple[Path, str], Path] = {}
        collisions: list[tuple[str, Path, Path]] = []
        for p in image_files:
            key = (p.parent, p.stem)
            if key in stems:
                collisions.append((p.stem, stems[key], p))
            else:
                stems[key] = p
        if collisions:
            print(
                "Duplicate image stems within a single folder of --dir "
                "(caches collide on identical stems in the same subdir):"
            )
            for stem, a, b in collisions:
                print(f"  '{stem}': {a} <-> {b}")
            sys.exit(1)
    else:
        image_files = sorted(
            p for p in data_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        )
    if not image_files:
        print(f"No images found in {data_dir}/", file=sys.stderr)
        sys.exit(1)

    # Pre-skip cached files so workers never decode them. The header read in
    # PIL.Image.open is cheap but adds up at 100k+ images; we still do it
    # below for grouping, but only on uncached entries.
    pending: list[Path] = []
    skipped = 0
    for p in image_files:
        if cache_path_for(
            p, bundle.name, cache_dir=cache_dir, image_dir=data_dir
        ).exists():
            skipped += 1
        else:
            pending.append(p)

    reso_groups: dict[tuple[int, int], list[Path]] = {}
    for p in pending:
        with Image.open(p) as img:
            size = img.size  # (W, H)
        reso_groups.setdefault(size, []).append(p)

    cached = 0

    metadata = {
        "encoder": bundle.name,
        "d_enc": str(bundle.d_enc),
        "patch": str(bundle.bucket_spec.patch),
    }

    pbar = tqdm(
        total=len(pending),
        desc=f"Caching {bundle.name} features",
    )
    for (w, h), paths in reso_groups.items():
        out_paths = [
            cache_path_for(p, bundle.name, cache_dir=cache_dir, image_dir=data_dir)
            for p in paths
        ]
        ds = _PEImageGroup(paths, out_paths)
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=_collate,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(args.num_workers > 0 and len(paths) > args.batch_size),
        )
        for batch_paths, batch_out_paths, img_batch in loader:
            with torch.no_grad():
                feats_list = encode_pe_from_imageminus1to1(
                    bundle, img_batch, same_bucket=True
                )
            for src, dst, feats in zip(batch_paths, batch_out_paths, feats_list):
                save_dict = {
                    "image_features": feats.detach().to(save_dtype).cpu().contiguous()
                }
                _save_safetensors(save_dict, dst, metadata=metadata)
                cached += 1
                pbar.update(1)
                pbar.set_postfix_str(f"{Path(src).name} → T={feats.shape[0]}")

    pbar.close()
    print(
        f"\n{bundle.name} feature caching complete: {cached} cached, {skipped} skipped"
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.centroid:
        out_path = (
            Path(args.centroid_out)
            if args.centroid_out
            else ROOT
            / "post_image_dataset"
            / "ip_adapter"
            / f"anima_pe_centroid_{bundle.name}.safetensors"
        )
        _write_centroid_sidecar(
            cache_dir,
            out_path,
            encoder=bundle.name,
            limit=args.centroid_limit,
        )


if __name__ == "__main__":
    main()
