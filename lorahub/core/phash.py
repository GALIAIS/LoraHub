"""Perceptual hash computation for image deduplication.

Implements phash64 (DCT-based) and dhash64 (difference hash) using
only Pillow — no external imagehash dependency needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image

FloatArray = NDArray[np.float64]


def phash64(path: Path | str) -> str:
    """Compute a 64-bit perceptual hash (DCT-based) as hex string."""
    img = Image.open(path).convert("L").resize((32, 32), Image.Resampling.LANCZOS)
    pixels = np.array(img, dtype=np.float64)
    dct = _dct2(pixels)
    low_freq = dct[:8, :8]
    median = np.median(low_freq)
    bits = (low_freq > median).flatten()
    return _bits_to_hex(bits)


def dhash64(path: Path | str) -> str:
    """Compute a 64-bit difference hash as hex string."""
    img = Image.open(path).convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = np.array(img, dtype=np.float64)
    bits = (pixels[:, 1:] > pixels[:, :-1]).flatten()
    return _bits_to_hex(bits)


def hamming_distance(h1: str, h2: str) -> int:
    """Compute hamming distance between two hex hash strings."""
    n1 = int(h1, 16)
    n2 = int(h2, 16)
    return bin(n1 ^ n2).count("1")


def _dct2(block: FloatArray) -> FloatArray:
    """2D DCT via separable 1D DCT-II (no scipy dependency)."""
    return _dct1((_dct1(block.T)).T)


def _dct1(vec: FloatArray) -> FloatArray:
    """1D DCT-II along axis 0 using the matrix definition."""
    n = vec.shape[0]
    k = np.arange(n).reshape(-1, 1)
    cos_table = np.cos(np.pi * (2 * np.arange(n) + 1) * k / (2 * n))
    return cast(FloatArray, cos_table @ vec)


def _bits_to_hex(bits: NDArray[np.bool_]) -> str:
    """Convert a boolean array of 64 bits to a 16-char hex string."""
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return f"{value:016x}"
