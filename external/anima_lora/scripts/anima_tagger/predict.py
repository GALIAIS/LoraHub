"""Single-image debug entry — runs the trained head on an image and prints
the canonical caption + (optionally) rating distribution + top-K kept tags.

If ``--image`` is omitted, samples a random stem from the val split
(falling back to the full manifest) so you can spot-check predictions
against ground-truth tags from ``dataset.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)


def cmd_predict(args: argparse.Namespace) -> None:
    """Run a single image through the trained tagger and print the caption.

    With ``--show_scores`` also prints rating distribution + top-K kept tags
    sorted by probability — useful for sanity-checking thresholds.
    """
    from PIL import Image

    from library.captioning.anima_tagger import AnimaTagger
    from library.captioning.anima_tagger_data import TaggerManifest

    out_dir = Path(args.out_dir)
    if not (out_dir / "model.safetensors").exists():
        raise SystemExit(
            f"missing {out_dir / 'model.safetensors'} — run --mode train first."
        )

    gt_tags: Optional[List[str]] = None
    gt_rating: Optional[str] = None
    gt_people: Optional[str] = None
    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            raise SystemExit(f"image not found: {image_path}")
    else:
        manifest_path = out_dir / "dataset.json"
        if not manifest_path.exists():
            raise SystemExit(
                f"--image not given and no manifest at {manifest_path} to sample "
                f"from. Pass --image <path> or run --mode build_vocab first."
            )
        manifest = TaggerManifest.from_path(manifest_path)
        stem_to_idx = manifest.stem_index()
        pool = manifest.val_stems or manifest.stems
        stem = random.choice(pool)
        i = stem_to_idx[stem]
        image_path = manifest.image_paths[i]
        # Resolve ground-truth labels for the side-by-side comparison.
        with open(out_dir / "vocab.json") as f:
            vocab = json.load(f)
        idx_to_name = {t["index"]: t["name"] for t in vocab["tags"]}
        gt_tags = [idx_to_name[k] for k in manifest.tag_indices[i] if k in idx_to_name]
        gt_rating = vocab["ratings"][manifest.rating_indices[i]]
        if manifest.people_count_indices and "people_count_labels" in vocab:
            gt_people = vocab["people_count_labels"][manifest.people_count_indices[i]]
        print(f"sampled stem: {stem}")
        print(f"image:        {image_path}")
        print(f"split:        {'val' if stem in set(manifest.val_stems) else 'train'}")
        print()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tagger = AnimaTagger(out_dir, device=device)
    img = Image.open(image_path)
    out = tagger.predict(img)
    caption = tagger.predict_caption(img)

    print(caption)
    if gt_tags is not None:
        gt_caption = ", ".join([gt_rating] + gt_tags).replace("_", " ")
        print()
        print(f"ground truth: {gt_caption}")
        if gt_people is not None:
            print(f"gt people:    {gt_people}")
    if args.show_scores:
        rating_scores = out["rating_scores"]
        print()
        print("rating:")
        for r, p in sorted(rating_scores.items(), key=lambda kv: -kv[1]):
            print(f"  {r:<10} {p:.3f}")
        people_scores = out.get("people_count_scores")
        if people_scores:
            print("people_count:")
            for r, p in sorted(people_scores.items(), key=lambda kv: -kv[1]):
                print(f"  {r:<14} {p:.3f}")
        kept = out["kept"]
        top = sorted(kept.items(), key=lambda kv: -kv[1])[: args.top_k]
        print(f"top {len(top)} kept tags (of {len(kept)} above threshold):")
        for name, p in top:
            print(f"  {p:.3f}  {name}")
