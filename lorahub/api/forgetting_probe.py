"""Catastrophic-forgetting probe.

Compares per-checkpoint sample images for "neutral" prompts against
the same prompt's earliest sample (the closest thing we have to a
pristine-base reference, sampled at step 0 / first checkpoint). A
small drift means the adapter has preserved the base model's
behaviour on prompts it was never trained on; a large drift means
the adapter has overwritten general-purpose capability — the canonical
catastrophic-forgetting failure mode.

Comparison metric: dHash + mean perceptual luminance distance. We
deliberately avoid LPIPS / CLIP-image to keep the API host
GPU-free — preservation drift is qualitative, the point is to flag
"this run forgot a lot" / "this run preserved well" rather than
quantify image similarity to four decimal places. dHash + Hamming
distance gives us a stable, scale-invariant signal in < 5 ms per
image with zero new dependencies.

Triggering: a `sample_ready` event lands in lifecycle.on_event. If the
sample's filename matches a "forget:" / "_forget_" tag (the user
flags certain prompts in their prompts file as neutral monitors), the
probe is scheduled. The first qualifying sample for each
``(prompt_id, seed)`` pair becomes the baseline; subsequent samples
are compared against it.

The probe writes a `forgetting_probe` event aggregating the latest
similarity per prompt. The UI plots ``preserved`` over training
steps so the user sees forgetting trajectories at a glance.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lorahub.core.events import EventType, TrainingEvent

_log = logging.getLogger(__name__)


# Per-job cache of (prompt_key -> baseline_dhash). We never bound this
# because runs typically have ~5-10 forget prompts; if a user goes wild
# the cap is implicit (no run has more than thousands of prompts).
_baselines: dict[tuple[str, str], "_Baseline"] = {}
_lock = threading.Lock()


@dataclass(frozen=True, slots=True)
class _Baseline:
    """First-seen sample for one (job_id, prompt_key) pair."""

    job_id: str
    prompt_key: str
    dhash_hex: str
    image_path: Path
    step: int | None


# Filename markers a user can place in their prompts file to mark a
# row as a "forgetting monitor": neutral prompts whose output should
# stay close to the base model. Matched case-insensitively against
# the sample filename — different backends format the prompt index
# differently, so we look for any of the markers in the basename.
_FORGET_MARKERS = ("forget", "neutral", "preserve")
_MARKER_RX = re.compile(
    r"(?:^|[._-])(" + "|".join(_FORGET_MARKERS) + r")(?:[._-]|$)", re.IGNORECASE
)


def schedule_forgetting_probe(
    sample: Path,
    *,
    prompt_key: str,
    step: int | None,
    on_event: Callable[[TrainingEvent], None],
    job_id: str,
) -> threading.Thread | None:
    """Run the probe in a daemon thread; returns None if skipped.

    Skips when the sample doesn't exist (anima sometimes emits the
    `sample_ready` event before the file flush settles — we'll see
    it on the next checkpoint anyway).
    """
    if not sample.is_file():
        return None

    def _work() -> None:
        try:
            similarity = _probe(sample, job_id=job_id, prompt_key=prompt_key)
        except Exception as exc:  # noqa: BLE001
            _log.warning(
                "forgetting-probe failed for %s: %s", sample, exc
            )
            return
        if similarity is None:
            return
        on_event(
            TrainingEvent(
                type=EventType.forgetting_probe,
                payload={
                    "step": step,
                    "checkpoint": str(sample),
                    "preserved": similarity.preserved,
                    "samples": 1,
                    "image_path": str(sample),
                    "prompt_key": prompt_key,
                    "baseline_step": similarity.baseline_step,
                },
                job_id=job_id,
            )
        )

    t = threading.Thread(
        target=_work,
        name=f"forget-probe-{step or '?'}",
        daemon=True,
    )
    t.start()
    return t


def is_neutral_prompt(sample_path: Path) -> bool:
    """True if the sample's filename matches one of the forget markers."""
    return bool(_MARKER_RX.search(sample_path.name))


def derive_prompt_key(sample_path: Path) -> str:
    """Stable key for grouping samples by ``(prompt, seed)``.

    We use the full filename minus any ``e000123`` / ``s000050`` /
    ``step_456`` step indicator, since those vary across checkpoints
    while the prompt+seed components don't.
    """
    base = sample_path.stem.lower()
    # Strip per-checkpoint counters. Match either at word boundaries
    # or surrounded by underscores / dashes (the kohya / anima naming
    # convention often uses ``_e12_s500`` or ``-step50-``).
    base = re.sub(
        r"(?:^|[\s_-])(?:e|ep|epoch|s|step)[\s_-]*\d+(?=$|[\s_-])",
        "",
        base,
    )
    base = re.sub(r"_+", "_", base).strip("_-.")
    return base or sample_path.stem


@dataclass(frozen=True, slots=True)
class _Similarity:
    preserved: float
    baseline_step: int | None


def _probe(
    sample: Path, *, job_id: str, prompt_key: str
) -> _Similarity | None:
    cur_dhash = _dhash(sample)
    if cur_dhash is None:
        return None
    key = (job_id, prompt_key)
    with _lock:
        baseline = _baselines.get(key)
        if baseline is None:
            _baselines[key] = _Baseline(
                job_id=job_id,
                prompt_key=prompt_key,
                dhash_hex=cur_dhash,
                image_path=sample,
                step=None,
            )
            # The first-seen sample is the baseline — preservation == 1.0
            # by definition. The probe still emits so the UI has a
            # starting point on the chart.
            return _Similarity(preserved=1.0, baseline_step=None)
    distance = _hamming_hex(cur_dhash, baseline.dhash_hex)
    # 64-bit dHash → max distance 64. Map (distance/64) to "preserved"
    # in [0..1] so the bar fills up when similarity is high.
    preserved = max(0.0, 1.0 - distance / 64.0)
    return _Similarity(preserved=preserved, baseline_step=baseline.step)


def _dhash(path: Path) -> str | None:
    """Compute an 8x8 difference hash over a luminance-downsampled image.

    Tries Pillow first (already a transitive dep via tagging). Returns
    None if Pillow isn't available so callers can degrade gracefully.
    """
    try:
        from PIL import Image  # noqa: PLC0415
    except ImportError:
        _log.debug("forgetting-probe: Pillow not installed, skipping")
        return None
    with Image.open(path) as im:
        im = im.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        pixels = list(im.getdata())
    bits = 0
    for row in range(8):
        row_off = row * 9
        for col in range(8):
            left = pixels[row_off + col]
            right = pixels[row_off + col + 1]
            bits = (bits << 1) | (1 if left > right else 0)
    return f"{bits:016x}"


def _hamming_hex(a_hex: str, b_hex: str) -> int:
    """Bit-level Hamming distance between two equal-length hex digests."""
    a = int(a_hex, 16)
    b = int(b_hex, 16)
    diff = a ^ b
    # popcount via bin().count("1") — Python 3.10+ has int.bit_count.
    if hasattr(diff, "bit_count"):
        return diff.bit_count()
    return bin(diff).count("1")


def reset_baselines(job_id: str) -> None:
    """Drop cached baselines for a job (call on relaunch / cancel)."""
    with _lock:
        for key in [k for k in _baselines if k[0] == job_id]:
            _baselines.pop(key, None)


def fingerprint(path: Path) -> str:
    """Stable cache key for a sample file (path + size + mtime).

    Useful in tests and for the "did the file actually change" check.
    """
    stat = path.stat()
    h = hashlib.sha1()
    h.update(str(path).encode("utf-8"))
    h.update(str(stat.st_size).encode("utf-8"))
    h.update(str(int(stat.st_mtime)).encode("utf-8"))
    return h.hexdigest()[:16]


def _now() -> float:
    return time.time()
