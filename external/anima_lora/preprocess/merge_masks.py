#!/usr/bin/env python3
"""Merge masks from multiple sources by taking the pixel-wise minimum (union of masked regions).

Walks each input mask directory recursively and keys merges by
``(rel_dir, name)`` so masks at the same relative path across inputs collide.
The output preserves the same nested layout under ``--output-dir``. Flat
layouts (no subdirs) collapse to flat output as before.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mask_dirs", nargs="+", help="Input mask directories to merge")
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for merged masks",
    )
    args = parser.parse_args()

    mask_dirs = [Path(d) for d in args.mask_dirs]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect (rel_dir, name) → list[abs_path] across all sources
    by_rel: dict[tuple[str, str], list[Path]] = {}
    for d in mask_dirs:
        if not d.exists():
            continue
        for p in d.rglob("*_mask.png"):
            rel = p.parent.relative_to(d)
            rel_str = "" if str(rel) in ("", ".") else str(rel)
            by_rel.setdefault((rel_str, p.name), []).append(p)

    if not by_rel:
        print("No masks found.")
        return

    merged = 0
    for (rel_str, name), sources in tqdm(sorted(by_rel.items()), desc="Merging masks"):
        if len(sources) == 1:
            arr = np.array(Image.open(sources[0]))
        else:
            # Pixel-wise minimum: lower alpha = more masking
            arr = np.array(Image.open(sources[0]))
            for src in sources[1:]:
                other = np.array(
                    Image.open(src).resize((arr.shape[1], arr.shape[0]), Image.NEAREST)
                )
                arr = np.minimum(arr, other)

        target_dir = output_dir / rel_str if rel_str else output_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr, mode="L").save(target_dir / name)
        merged += 1

    print(f"Merged {merged} masks into {output_dir}/")


if __name__ == "__main__":
    main()
