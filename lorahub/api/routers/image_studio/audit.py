"""Dataset audit / health-check endpoints.

Aggregates per-image diagnostics into a single JSON report a UI can
turn into histograms + actionable issue lists. Designed to run without
GPU models — every metric here is fast (~few ms per image on CPU) so
a 1000-image dataset finishes in well under a minute.

Heavy GPU-backed metrics (aesthetic predictor, NSFW classifier) live
elsewhere and feed in via the same report shape; this router is just
the cheap-pass version.

Output shape:

    {
      "dataset_path": "...",
      "scanned_at": "...iso...",
      "image_count": 111,
      "captioned_count": 95,
      "trigger_word_hits": 89,
      "trigger_word": "@thornsdance" | null,

      "resolution_histogram": [{"bucket": "768-1023",  "count": 18}, ...],
      "ar_histogram":         [{"bucket": "0.5-0.75", "count": 12}, ...],
      "filesize_histogram":   [{"bucket": "0-256k",    "count": 4},  ...],
      "caption_length_histogram": [{"bucket": "0-20", "count": 16}, ...],

      "tag_vocab": [{"tag": "1girl", "count": 89}, ...],   # top 50

      "issues": [
        {"kind": "corrupt",       "path": "...", "msg": "..."},
        {"kind": "tiny",          "path": "...", "width": 512, "height": 384},
        {"kind": "exif_rotation", "path": "...", "orientation": 6},
        {"kind": "no_caption",    "path": "..."},
        {"kind": "missing_trigger","path": "...", "trigger": "@thornsdance"},
        {"kind": "blurry",        "path": "...", "score": 18.4},
      ]
    }

The cache lives at ``<dataset>/.workbench/audit.json``. ``GET /report``
returns the cache as-is; ``POST /scan`` re-runs and overwrites it.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from PIL import ExifTags, Image, UnidentifiedImageError
from pydantic import BaseModel

from lorahub.api.dataset_files import (
    IMAGE_SUFFIXES,
    iter_safe_files,
    resolve_dataset_directory,
)

from ._shared import _atomic_write_text

router = APIRouter(prefix="/api/image-studio", tags=["image-studio"])


def _scan_images(directory: Path, recursive: bool) -> list[Path]:
    """Uncached scan so each audit reflects the current filesystem."""
    return [
        path
        for path in iter_safe_files(directory, recursive=recursive)
        if path.suffix.lower() in IMAGE_SUFFIXES
    ]

# Anti-DDoS: avoid auditing absurdly-large image globs in one go. The
# UI can re-trigger if it really needs a 100k-image pass.
_MAX_IMAGES = 5000

# Bucket edges chosen to match anima_lora's training resolution
# spectrum (256-2048, step 64-ish). Histograms use the long edge so a
# portrait at 1024x1536 lands in 1280-1535 just like a landscape would.
_RES_BUCKETS = [
    (0,    383,  "0-383"),
    (384,  511,  "384-511"),
    (512,  767,  "512-767"),
    (768,  1023, "768-1023"),
    (1024, 1279, "1024-1279"),
    (1280, 1535, "1280-1535"),
    (1536, 2047, "1536-2047"),
    (2048, 999_999, "2048+"),
]

# AR buckets — wide enough that anima's bucket generator doesn't scatter
# images across 30+ slots; narrow enough that "portrait-ish" stands
# distinct from "square".
_AR_BUCKETS = [
    (0.0,  0.4999, "<0.5"),
    (0.5,  0.7499, "0.5-0.75"),
    (0.75, 0.9999, "0.75-1.0"),
    (1.0,  1.0,    "1.0"),
    (1.0001, 1.5,  "1.0-1.5"),
    (1.5001, 2.0,  "1.5-2.0"),
    (2.0001, 999, ">2.0"),
]

_FILESIZE_BUCKETS = [
    (0,         256 * 1024,   "0-256K"),
    (256 * 1024,    1024 * 1024, "256K-1M"),
    (1024 * 1024,    4 * 1024 * 1024, "1-4M"),
    (4 * 1024 * 1024,    16 * 1024 * 1024, "4-16M"),
    (16 * 1024 * 1024,    64 * 1024 * 1024, "16-64M"),
    (64 * 1024 * 1024,    1 << 40,  "64M+"),
]

_CAPTION_LEN_BUCKETS = [
    (0,   1,   "empty"),
    (1,   20,  "1-20"),
    (20,  60,  "20-60"),
    (60,  120, "60-120"),
    (120, 240, "120-240"),
    (240, 99_999, "240+"),
]

# Below this Laplacian variance, the image is effectively blurry. The
# threshold is dataset-dependent (high-detail anime art floors at
# ~150; smartphone photos at ~80) but 50 is a safe lower bound that
# mostly catches genuinely out-of-focus / motion-blur shots without
# false-flagging stylised art.
_BLUR_THRESHOLD = 50.0
_TINY_LONG_EDGE = 512


class ScanRequest(BaseModel):
    dataset_path: str
    recursive: bool = True
    trigger_word: str | None = None
    # Optional cap so a typo on dataset_path doesn't trigger a 100k
    # image scan. None means use the global ceiling.
    max_images: int | None = None
    # Skip the Laplacian blur check — saves ~30% of scan time on big
    # datasets when the user only wants the resolution + caption
    # report. Default on; UI exposes it as a toggle.
    blur_check: bool = True


@dataclass
class _Bucket:
    label: str
    count: int = 0


@dataclass
class _Report:
    dataset_path: str
    scanned_at: str
    image_count: int = 0
    captioned_count: int = 0
    trigger_word: str | None = None
    trigger_word_hits: int = 0
    resolution_histogram: list[_Bucket] = field(default_factory=list)
    ar_histogram: list[_Bucket] = field(default_factory=list)
    filesize_histogram: list[_Bucket] = field(default_factory=list)
    caption_length_histogram: list[_Bucket] = field(default_factory=list)
    tag_vocab: list[dict[str, Any]] = field(default_factory=list)
    issues: list[dict[str, Any]] = field(default_factory=list)
    duration_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_path": self.dataset_path,
            "scanned_at": self.scanned_at,
            "image_count": self.image_count,
            "captioned_count": self.captioned_count,
            "trigger_word": self.trigger_word,
            "trigger_word_hits": self.trigger_word_hits,
            "resolution_histogram": [
                {"bucket": b.label, "count": b.count} for b in self.resolution_histogram
            ],
            "ar_histogram": [
                {"bucket": b.label, "count": b.count} for b in self.ar_histogram
            ],
            "filesize_histogram": [
                {"bucket": b.label, "count": b.count} for b in self.filesize_histogram
            ],
            "caption_length_histogram": [
                {"bucket": b.label, "count": b.count}
                for b in self.caption_length_histogram
            ],
            "tag_vocab": self.tag_vocab,
            "issues": self.issues,
            "duration_s": round(self.duration_s, 3),
        }


def _bucket_for(value: int | float, edges: list[tuple]) -> str:
    for low, high, label in edges:
        if low <= value <= high:
            return label
    return edges[-1][2]


def _laplacian_variance(img: Image.Image) -> float:
    """Compute Laplacian variance via numpy (no opencv dep).

    Higher = more high-frequency content = sharper. Below ~50 is
    typically blurry. Resizes to a fixed 256-long-edge thumb so the
    score is comparable across resolutions.
    """
    long_edge = max(img.size)
    if long_edge > 256:
        scale = 256 / long_edge
        new_size = (max(1, int(img.size[0] * scale)), max(1, int(img.size[1] * scale)))
        img = img.resize(new_size, Image.Resampling.BILINEAR)
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    # Discrete 3x3 Laplacian kernel.
    k = np.array(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
        dtype=np.float32,
    )
    # Convolution via FFT-free numpy: pad, slide, sum.
    pad = np.pad(gray, 1, mode="edge")
    out = (
        k[0, 0] * pad[:-2, :-2]
        + k[0, 1] * pad[:-2, 1:-1]
        + k[0, 2] * pad[:-2, 2:]
        + k[1, 0] * pad[1:-1, :-2]
        + k[1, 1] * pad[1:-1, 1:-1]
        + k[1, 2] * pad[1:-1, 2:]
        + k[2, 0] * pad[2:, :-2]
        + k[2, 1] * pad[2:, 1:-1]
        + k[2, 2] * pad[2:, 2:]
    )
    return float(out.var())


_ORIENTATION_TAG = next(
    (k for k, v in ExifTags.TAGS.items() if v == "Orientation"), 0x0112,
)


def _exif_orientation(img: Image.Image) -> int | None:
    """Return the EXIF orientation tag value, or None if absent.

    Orientation 1 = normal. 2-8 mean the image displays rotated /
    flipped relative to its pixel grid; training reads pixels directly
    so the user really wants these baked in before training.
    """
    try:
        exif = img.getexif()
    except Exception:  # noqa: BLE001
        return None
    if not exif:
        return None
    return exif.get(_ORIENTATION_TAG)


def _split_caption_tags(caption: str) -> list[str]:
    """Split a tag-style caption into normalised lowercase tags.

    LoRA captions are typically comma-separated. Whitespace gets
    trimmed; empty tags filtered. Case-folded so 'Long Hair' and
    'long hair' merge in the vocab histogram.
    """
    parts = [t.strip().lower() for t in caption.split(",")]
    return [t for t in parts if t]


def _audit_cache_path(dataset_path: str) -> Path:
    try:
        root = resolve_dataset_directory(dataset_path)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return root / ".workbench" / "audit.json"


@router.post("/audit/scan")
def audit_scan(req: ScanRequest) -> dict[str, Any]:
    """Run a full audit and write the cache. Returns the report inline.

    Synchronous; on a 1000-image dataset this takes 30-90s depending
    on whether the blur check is enabled. The UI shows a spinner.
    """
    import time  # noqa: PLC0415

    started = time.time()
    try:
        root = resolve_dataset_directory(req.dataset_path)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    paths = _scan_images(root, req.recursive)
    cap = req.max_images or _MAX_IMAGES
    if len(paths) > cap:
        raise HTTPException(
            413,
            f"dataset has {len(paths)} images; cap is {cap}. Pass max_images "
            "to override or run on a sub-folder.",
        )

    report = _Report(
        dataset_path=str(root),
        scanned_at=datetime.now(UTC).replace(microsecond=0).isoformat(),
        trigger_word=req.trigger_word,
    )
    res_buckets: dict[str, _Bucket] = {
        b[2]: _Bucket(label=b[2]) for b in _RES_BUCKETS
    }
    ar_buckets: dict[str, _Bucket] = {
        b[2]: _Bucket(label=b[2]) for b in _AR_BUCKETS
    }
    fs_buckets: dict[str, _Bucket] = {
        b[2]: _Bucket(label=b[2]) for b in _FILESIZE_BUCKETS
    }
    cl_buckets: dict[str, _Bucket] = {
        b[2]: _Bucket(label=b[2]) for b in _CAPTION_LEN_BUCKETS
    }
    tag_count: dict[str, int] = {}

    trigger_lower = req.trigger_word.lower() if req.trigger_word else None

    for p in paths:
        report.image_count += 1

        # File size first — cheapest, also catches 0-byte placeholders.
        try:
            size = p.stat().st_size
        except OSError as exc:
            report.issues.append({"kind": "corrupt", "path": str(p), "msg": str(exc)})
            continue
        fs_label = _bucket_for(size, [(b[0], b[1] - 1, b[2]) for b in _FILESIZE_BUCKETS])
        fs_buckets[fs_label].count += 1

        # Image probe — verify, then re-open for actual reads (verify
        # leaves the file pointer in an unusable state).
        try:
            with Image.open(p) as img:
                img.verify()
        except (UnidentifiedImageError, OSError) as exc:
            report.issues.append({"kind": "corrupt", "path": str(p), "msg": str(exc)})
            continue

        try:
            with Image.open(p) as img:
                w, h = img.size
                orientation = _exif_orientation(img)
                # Blur check — load greyscale via the same Image
                # context to avoid opening twice.
                blur_score: float | None = None
                if req.blur_check:
                    try:
                        blur_score = _laplacian_variance(img)
                    except Exception:  # noqa: BLE001
                        blur_score = None
        except (UnidentifiedImageError, OSError) as exc:
            report.issues.append({"kind": "corrupt", "path": str(p), "msg": str(exc)})
            continue

        # Histograms.
        long_edge = max(w, h)
        ar = w / h if h else 0.0
        res_label = _bucket_for(long_edge, _RES_BUCKETS)
        ar_label = _bucket_for(ar, _AR_BUCKETS)
        res_buckets[res_label].count += 1
        ar_buckets[ar_label].count += 1

        # Issue diagnostics.
        if long_edge < _TINY_LONG_EDGE:
            report.issues.append(
                {"kind": "tiny", "path": str(p), "width": w, "height": h},
            )
        if orientation and orientation != 1:
            report.issues.append(
                {"kind": "exif_rotation", "path": str(p), "orientation": orientation},
            )
        if blur_score is not None and blur_score < _BLUR_THRESHOLD:
            report.issues.append(
                {"kind": "blurry", "path": str(p), "score": round(blur_score, 1)},
            )

        # Caption + tag vocab.
        cap_path = p.with_suffix(".txt")
        caption_text = ""
        if cap_path.is_file():
            try:
                caption_text = cap_path.read_text(encoding="utf-8").strip()
            except OSError:
                caption_text = ""
        if caption_text:
            report.captioned_count += 1
            for tag in _split_caption_tags(caption_text):
                tag_count[tag] = tag_count.get(tag, 0) + 1
        else:
            report.issues.append({"kind": "no_caption", "path": str(p)})

        cl_label = _bucket_for(len(caption_text), _CAPTION_LEN_BUCKETS)
        cl_buckets[cl_label].count += 1

        # Trigger word check (case-insensitive).
        if trigger_lower:
            if trigger_lower in caption_text.lower():
                report.trigger_word_hits += 1
            else:
                report.issues.append(
                    {
                        "kind": "missing_trigger",
                        "path": str(p),
                        "trigger": req.trigger_word,
                    },
                )

    report.resolution_histogram = list(res_buckets.values())
    report.ar_histogram = list(ar_buckets.values())
    report.filesize_histogram = list(fs_buckets.values())
    report.caption_length_histogram = list(cl_buckets.values())
    report.tag_vocab = sorted(
        ({"tag": t, "count": c} for t, c in tag_count.items()),
        key=lambda r: r["count"],
        reverse=True,
    )[:50]
    report.duration_s = time.time() - started

    # Cache.
    cache_path = _audit_cache_path(req.dataset_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(
        cache_path,
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
    )
    return report.to_dict()


@router.get("/audit/report")
def audit_report(dataset_path: str) -> dict[str, Any]:
    """Return the cached audit report; 404 when the dataset hasn't been scanned."""
    cache_path = _audit_cache_path(dataset_path)
    if not cache_path.is_file():
        raise HTTPException(
            404,
            "no audit report cached for this dataset; run POST /audit/scan first",
        )
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(500, f"failed to read audit cache: {exc}") from None


# Suppress unused-import warning.
_ = IMAGE_SUFFIXES
_ = math
