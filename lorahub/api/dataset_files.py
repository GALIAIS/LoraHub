"""Dataset file helpers: path allow-list, thumbnail cache, caption I/O.

The HTTP API is localhost-only, but we still need to keep dataset endpoints
from turning into a remote-readable file proxy. Every caller-supplied path
is resolved against a small allow-list before we touch the filesystem:

    * the current working directory (where the user launched `lorahub serve`,
      so anything under the project tree is reachable),
    * `LORAHUB_DATASETS_ROOT` -- one or more `os.pathsep`-separated extra
      roots for users who keep their datasets elsewhere,
    * every registered job workspace (training samples are valid dataset
      sources too -- you can browse them like any other folder).

Anything that resolves outside that union is rejected with `ValueError`,
which the router maps to a 400. Symlink escapes are blocked because we
fully resolve before checking ``relative_to``.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import threading
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from lorahub.api import state

# Lift Pillow's "decompression bomb" guard. A modern phone shoots ~8000×6000
# (~48MP) which is already above Pillow's default 89MP cap once you stitch a
# panorama. Training datasets routinely include 8K / 12K renders. We trust
# our local-only filesystem source, so cap at 1Gpx (more than enough for
# anything realistic, still far below memory blow-up territory once the
# JPEG draft below kicks in).
Image.MAX_IMAGE_PIXELS = 1_000_000_000


# Cap concurrent thumbnail *generation* at a small number. FastAPI runs
# sync route handlers on a 40-worker threadpool; without this guard a
# burst of 30 first-access big-image requests (typical right after a
# bulk upload) decodes 30 multi-MP files at once, saturating the
# Python GIL via libjpeg/libpng C calls and starving the rest of the
# API. Cached lookups still hit the fast path; only first-time builds
# wait on the semaphore.
_THUMB_BUILD_SEM = threading.BoundedSemaphore(value=4)

# Suffixes we are willing to thumbnail / treat as dataset images. Mirrors the
# set used by `_scan_dataset_path` so the UI sees a consistent surface.
IMAGE_SUFFIXES = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp"}

# Bounds for the `size` query param on /datasets/thumb. Lower bound keeps
# tiny degenerate requests from spamming the cache; upper bound prevents
# someone from asking us to render a 10k-by-10k preview.
_MIN_THUMB_SIZE = 32
_MAX_THUMB_SIZE = 1024

# Cache lives under the workspace's `runs/` tree so it gets the same
# git-ignore + cleanup treatment as everything else the server writes.
_THUMB_SUBDIR = Path("runs") / ".thumbs"


def _allowed_roots() -> list[Path]:
    """Return the de-duplicated set of directories dataset paths may live in.

    Order is preserved so the most-specific root (cwd) is checked first.
    """
    roots: list[Path] = []
    with contextlib.suppress(OSError, RuntimeError):
        roots.append(Path.cwd().resolve())

    extra = os.environ.get("LORAHUB_DATASETS_ROOT")
    if extra:
        for piece in extra.split(os.pathsep):
            piece = piece.strip()
            if not piece:
                continue
            with contextlib.suppress(OSError, RuntimeError):
                roots.append(Path(piece).expanduser().resolve())

    # Each registered job workspace is also a legitimate root: training
    # samples that landed there should be browsable from the dataset page.
    with contextlib.suppress(Exception):
        for job in state.registry.list():
            with contextlib.suppress(OSError, RuntimeError):
                roots.append(job.workspace.resolve())

    seen: set[Path] = set()
    unique: list[Path] = []
    for r in roots:
        if r in seen:
            continue
        seen.add(r)
        unique.append(r)
    return unique


def _resolve_under_roots(raw: str) -> Path:
    """Resolve `raw` strictly and confirm it lives under an allowed root.

    Raises `ValueError` for empty input, unresolvable paths, or paths that
    escape the allow-list. We always resolve symlinks before the containment
    check so a symlink inside an allowed dir cannot be used to escape it.
    """
    if not raw or not raw.strip():
        raise ValueError("path is required")
    try:
        target = Path(raw).expanduser().resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise ValueError("invalid path") from exc
    for root in _allowed_roots():
        try:
            target.relative_to(root)
        except ValueError:
            continue
        return target
    raise ValueError("path is outside allowed dataset roots")


def resolve_image_path(raw: str) -> Path:
    """Resolve a caller-supplied image path. Suffix must be in IMAGE_SUFFIXES."""
    target = _resolve_under_roots(raw)
    if target.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError("unsupported image type")
    return target


def resolve_caption_path(raw_image_path: str) -> Path:
    """Resolve the `.txt` companion for the given image path.

    `raw_image_path` must end in one of `IMAGE_SUFFIXES`; that's what
    forces the writeable target to land on a `.txt` we synthesise from
    the image stem. Letting callers hand us an arbitrary path -- even
    one inside the allow-list -- would let them rewrite recipe files,
    settings, or training logs by guessing names.

    The image path itself is allow-list checked; the `.txt` file inherits
    that containment because it always sits in the same directory.
    """
    image_target = resolve_image_path(raw_image_path)
    caption = image_target.with_suffix(".txt")
    if caption.suffix.lower() != ".txt":
        raise ValueError("caption file must end in .txt")
    return caption


def _thumbnail_cache_dir() -> Path:
    """Return `<cwd>/runs/.thumbs/`, creating it lazily."""
    target = (Path.cwd() / _THUMB_SUBDIR).resolve()
    target.mkdir(parents=True, exist_ok=True)
    return target


def _thumb_cache_path(image: Path, size: int) -> Path:
    digest = hashlib.sha256(
        f"{image.as_posix()}|{size}".encode("utf-8")
    ).hexdigest()
    return _thumbnail_cache_dir() / f"{digest}.webp"


def get_or_build_thumbnail(image: Path, size: int) -> Path:
    """Return a cached webp thumbnail for `image`, generating one on miss.

    The output is square-bounded (Pillow's `thumbnail` keeps the aspect
    ratio so portrait/landscape images stay un-stretched) and re-encoded
    as WEBP for compactness. Cache entries are invalidated when the source
    image's mtime is newer than the cached thumb.

    Raises `ValueError` for out-of-range sizes and `OSError` if Pillow
    cannot decode or re-encode the image.
    """
    if size < _MIN_THUMB_SIZE or size > _MAX_THUMB_SIZE:
        raise ValueError(
            f"size must be between {_MIN_THUMB_SIZE} and {_MAX_THUMB_SIZE}"
        )
    cache = _thumb_cache_path(image, size)
    if cache.is_file():
        with contextlib.suppress(OSError):
            if cache.stat().st_mtime >= image.stat().st_mtime:
                return cache

    # Throttle first-time builds so a bulk-upload doesn't spawn dozens of
    # parallel Pillow decodes. Re-check the cache once we hold the slot —
    # another worker may have produced the file while we were queued.
    with _THUMB_BUILD_SEM:
        if cache.is_file():
            with contextlib.suppress(OSError):
                if cache.stat().st_mtime >= image.stat().st_mtime:
                    return cache
        try:
            with Image.open(image) as im:
                # Fast path for JPEG: ``draft`` instructs libjpeg to return a
                # pre-scaled DCT-aware downsample (1/2, 1/4, or 1/8 of the
                # original). Skipping full-resolution decode is the difference
                # between thumbing a 12K landscape in 60ms vs 4s. Safe no-op
                # on formats that don't support it.
                with contextlib.suppress(Exception):
                    im.draft("RGB", (size * 2, size * 2))
                # Convert away from palette / grayscale modes so WEBP encoding
                # stays predictable; we deliberately drop alpha to keep thumbs
                # compact and consistent across sources.
                if im.mode not in ("RGB",):
                    im = im.convert("RGB")
                im.thumbnail((size, size))
                cache.parent.mkdir(parents=True, exist_ok=True)
                im.save(cache, format="WEBP", quality=82, method=4)
        except (OSError, UnidentifiedImageError, ValueError, Image.DecompressionBombError) as exc:
            raise OSError(f"could not generate thumbnail: {exc}") from exc
    return cache


__all__ = [
    "IMAGE_SUFFIXES",
    "get_or_build_thumbnail",
    "resolve_caption_path",
    "resolve_image_path",
]
