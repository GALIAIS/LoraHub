#!/usr/bin/env python3
"""Cache pooled text embeddings (max over the sequence dim) from existing TE caches.

Reads each ``{stem}_anima_te.safetensors`` in a cache directory and writes a
matching ``{stem}_anima_pooled.safetensors`` sidecar holding
``pooled_v{i} = crossattn_emb_v{i}.amax(dim=0)`` for every variant present.

Consumed by ``scripts/distill_mod/distill.py`` (modulation guidance distillation):
``pooled_text_proj`` ingests this tensor at every training microstep and val
sigma; pre-caching it eliminates a redundant ``.max(dim=1)`` per step.

No GPU / text encoder needed -- pure tensor reduction on the cached crossattn.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from library.io.cache import POOLED_CACHE_SUFFIX, TE_CACHE_SUFFIX  # noqa: E402


def _emit_pooled(te_path: Path, pooled_path: Path) -> bool:
    sd = load_file(str(te_path))
    out: dict = {}

    if "num_variants" in sd:
        n = int(sd["num_variants"])
        out["num_variants"] = sd["num_variants"]
        for vi in range(n):
            key = f"crossattn_emb_v{vi}"
            if key in sd:
                out[f"pooled_v{vi}"] = sd[key].amax(dim=0).contiguous()
        if not any(k.startswith("pooled_v") for k in out):
            return False
    elif "crossattn_emb_v0" in sd:
        out["pooled_v0"] = sd["crossattn_emb_v0"].amax(dim=0).contiguous()
    elif "crossattn_emb" in sd:
        out["pooled"] = sd["crossattn_emb"].amax(dim=0).contiguous()
    else:
        return False

    save_file(out, str(pooled_path))
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Cache directory containing *_anima_te.safetensors files. Pooled "
        "sidecars are written into the same directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-emit pooled sidecars even when they already exist.",
    )
    args = parser.parse_args()

    cache_dir = Path(args.dir)
    # rglob so nested caches (mirroring subfoldered source layouts) are
    # picked up. Pooled sidecars are written next to each TE file, so the
    # same nested structure is preserved automatically.
    te_files = sorted(cache_dir.rglob(f"*{TE_CACHE_SUFFIX}"))
    if not te_files:
        print(f"No {TE_CACHE_SUFFIX} files found in {cache_dir}")
        return

    written = 0
    skipped = 0
    failed = 0

    for te_path in tqdm(te_files, desc="Caching pooled"):
        stem = te_path.name.removesuffix(TE_CACHE_SUFFIX)
        pooled_path = te_path.parent / (stem + POOLED_CACHE_SUFFIX)
        if pooled_path.exists() and not args.overwrite:
            skipped += 1
            continue
        if _emit_pooled(te_path, pooled_path):
            written += 1
        else:
            failed += 1

    print(
        f"Pooled cache: {written} written, {skipped} skipped (already existed), "
        f"{failed} failed (no crossattn key)"
    )


if __name__ == "__main__":
    main()
