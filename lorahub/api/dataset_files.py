"""Dataset file helpers: path allow-list, thumbnail cache, caption I/O.

The HTTP API can be exposed with token authentication, so dataset endpoints
must not become an arbitrary file proxy. Every caller-supplied path
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

import base64
import contextlib
import hashlib
import os
import tempfile
import threading
from pathlib import Path
from typing import Iterator

from PIL import Image, UnidentifiedImageError

from lorahub.api import paths as api_paths
from lorahub.api import state

# Keep large training renders usable without allowing an uploaded image to
# request an effectively unbounded decode. Pillow raises at twice this value;
# deployments with genuinely larger sources can opt in explicitly.
try:
    _max_image_pixels = int(os.environ.get("LORAHUB_MAX_IMAGE_PIXELS", "200000000"))
except ValueError:
    _max_image_pixels = 200_000_000
Image.MAX_IMAGE_PIXELS = max(1, _max_image_pixels)


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
DATASET_META_FILENAME = "dataset.json"
_DEFAULT_MAX_AI_IMAGE_BYTES = 25 * 1024**2


class ImageInputTooLarge(ValueError):
    """Raised when a vision request exceeds the configured byte limit."""

# Bounds for the `size` query param on /datasets/thumb. Lower bound keeps
# tiny degenerate requests from spamming the cache; upper bound prevents
# someone from asking us to render a 10k-by-10k preview.
_MIN_THUMB_SIZE = 32
_MAX_THUMB_SIZE = 4096

def _allowed_roots() -> list[Path]:
    """Return the de-duplicated set of directories dataset paths may live in.

    Order is preserved so the most-specific root (cwd) is checked first.
    """
    roots: list[Path] = []
    with contextlib.suppress(OSError, RuntimeError):
        roots.append(api_paths.project_root().resolve())

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


def _writable_dataset_roots() -> list[Path]:
    roots: list[Path] = []
    with contextlib.suppress(OSError, RuntimeError):
        roots.append((api_paths.project_root() / "datasets").resolve())
    extra = os.environ.get("LORAHUB_DATASETS_ROOT")
    if extra:
        for piece in extra.split(os.pathsep):
            if piece.strip():
                with contextlib.suppress(OSError, RuntimeError):
                    roots.append(Path(piece).expanduser().resolve())
    return list(dict.fromkeys(roots))


def resolve_dataset_directory(raw: str) -> Path:
    """Resolve an existing dataset that API routes are allowed to mutate."""
    if not raw or not raw.strip():
        raise ValueError("dataset path is required")
    lexical = Path(raw).expanduser()
    if is_link_like(lexical):
        raise ValueError("dataset path cannot be a link")
    try:
        target = lexical.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise ValueError("invalid dataset path") from exc
    if not target.is_dir():
        raise ValueError(f"dataset not found: {target}")
    for root in _writable_dataset_roots():
        try:
            target.relative_to(root)
        except ValueError:
            continue
        return target
    raise ValueError(
        "dataset is outside writable roots; move it under datasets/ or set "
        "LORAHUB_DATASETS_ROOT"
    )


def resolve_dataset_file(raw: str) -> Path:
    """Resolve an existing file that dataset workflows are allowed to mutate."""
    if not raw or not raw.strip():
        raise ValueError("dataset file path is required")
    lexical = Path(raw).expanduser()
    if is_link_like(lexical):
        raise ValueError("dataset file path cannot be a link")
    try:
        target = lexical.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise ValueError("invalid dataset file path") from exc
    if not target.is_file():
        raise ValueError(f"dataset file not found: {target}")
    for root in _writable_dataset_roots():
        try:
            target.relative_to(root)
        except ValueError:
            continue
        return target
    raise ValueError(
        "dataset file is outside writable roots; move it under datasets/ or set "
        "LORAHUB_DATASETS_ROOT"
    )


def resolve_dataset_source_directory(raw: str) -> Path:
    """Resolve an existing dataset for read-only inspection or export."""
    target = _resolve_under_roots(raw)
    if not target.is_dir():
        raise ValueError(f"dataset not found: {target}")
    return target


def is_link_like(path: Path) -> bool:
    """Return true for symlinks and Windows junction/reparse directories."""
    try:
        if path.is_symlink():
            return True
        is_junction = getattr(path, "is_junction", None)
        return bool(is_junction is not None and is_junction())
    except OSError:
        return True


def resolve_file_under(root: Path, candidate: Path) -> Path | None:
    """Resolve a regular non-link file while keeping it below ``root``."""
    if is_link_like(candidate):
        return None
    try:
        resolved_root = root.resolve()
        resolved = candidate.resolve()
        resolved.relative_to(resolved_root)
    except (OSError, RuntimeError, ValueError):
        return None
    return resolved if resolved.is_file() else None


def iter_safe_files(
    root: Path,
    *,
    recursive: bool,
    skip_dirs: frozenset[str] = frozenset(),
) -> Iterator[Path]:
    """Yield files without following links or leaving the resolved root."""
    resolved_root = root.resolve()
    if recursive:
        for current, dirs, files in os.walk(resolved_root, followlinks=False):
            current_path = Path(current)
            safe_dirs: list[str] = []
            for name in dirs:
                candidate = current_path / name
                if name in skip_dirs or is_link_like(candidate):
                    continue
                try:
                    candidate.resolve().relative_to(resolved_root)
                except (OSError, RuntimeError, ValueError):
                    continue
                safe_dirs.append(name)
            dirs[:] = safe_dirs
            for name in files:
                resolved = resolve_file_under(resolved_root, current_path / name)
                if resolved is not None:
                    yield resolved
        return

    for candidate in resolved_root.iterdir():
        resolved = resolve_file_under(resolved_root, candidate)
        if resolved is not None:
            yield resolved


def resolve_image_path(raw: str) -> Path:
    """Resolve a caller-supplied image path. Suffix must be in IMAGE_SUFFIXES."""
    target = _resolve_under_roots(raw)
    if target.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError("unsupported image type")
    return target


def max_ai_image_bytes() -> int:
    try:
        configured = int(
            os.environ.get(
                "LORAHUB_MAX_AI_IMAGE_BYTES",
                str(_DEFAULT_MAX_AI_IMAGE_BYTES),
            )
        )
    except ValueError:
        configured = _DEFAULT_MAX_AI_IMAGE_BYTES
    return max(1, configured)


def encode_image_data_url(path: Path) -> str:
    """Read an allowed image into a bounded data URL for vision providers."""
    target = resolve_image_path(str(path))
    if not target.is_file():
        raise ValueError("image file not found")
    limit = max_ai_image_bytes()
    try:
        size = target.stat().st_size
    except OSError as exc:
        raise ValueError("cannot inspect image file") from exc
    if size > limit:
        raise ImageInputTooLarge(f"image exceeds AI input limit of {limit} bytes")
    try:
        with target.open("rb") as handle:
            data = handle.read(limit + 1)
    except OSError as exc:
        raise ValueError("cannot read image file") from exc
    if len(data) > limit:
        raise ImageInputTooLarge(f"image exceeds AI input limit of {limit} bytes")
    mime = {
        ".bmp": "image/bmp",
        ".gif": "image/gif",
        ".jpeg": "image/jpeg",
        ".jpg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }[target.suffix.lower()]
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def resolve_caption_path(raw_image_path: str, *, writable: bool = False) -> Path:
    """Resolve the `.txt` companion for the given image path.

    `raw_image_path` must end in one of `IMAGE_SUFFIXES`; that's what
    forces the writeable target to land on a `.txt` we synthesise from
    the image stem. Letting callers hand us an arbitrary path -- even
    one inside the allow-list -- would let them rewrite config files,
    settings, or training logs by guessing names.

    The image path itself is allow-list checked; the `.txt` file inherits
    that containment because it always sits in the same directory.
    """
    image_target = (
        resolve_dataset_file(raw_image_path)
        if writable
        else resolve_image_path(raw_image_path)
    )
    if image_target.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError("unsupported image type")
    caption = image_target.with_suffix(".txt")
    if caption.suffix.lower() != ".txt":
        raise ValueError("caption file must end in .txt")
    return caption


def _thumbnail_cache_dir() -> Path:
    """Return the resolved project run cache, creating it lazily."""
    runs = api_paths.runs_dir().resolve()
    lexical = runs / ".thumbs"
    if is_link_like(lexical):
        raise OSError(f"thumbnail cache cannot be a link: {lexical}")
    lexical.mkdir(parents=True, exist_ok=True)
    target = lexical.resolve()
    try:
        target.relative_to(runs)
    except ValueError as exc:
        raise OSError("thumbnail cache escapes the runs directory") from exc
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
    if is_link_like(cache):
        raise OSError(f"thumbnail cache file cannot be a link: {cache}")
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
                converted = im if im.mode == "RGB" else im.convert("RGB")
                converted.thumbnail((size, size))
                cache.parent.mkdir(parents=True, exist_ok=True)
                fd, raw_temp = tempfile.mkstemp(
                    dir=cache.parent,
                    prefix=f".{cache.name}.",
                    suffix=".tmp",
                )
                os.close(fd)
                temp_path = Path(raw_temp)
                try:
                    converted.save(temp_path, format="WEBP", quality=82, method=4)
                    temp_path.replace(cache)
                finally:
                    temp_path.unlink(missing_ok=True)
        except (OSError, UnidentifiedImageError, ValueError, Image.DecompressionBombError) as exc:
            raise OSError(f"could not generate thumbnail: {exc}") from exc
    return cache


__all__ = [
    "DATASET_META_FILENAME",
    "IMAGE_SUFFIXES",
    "get_or_build_thumbnail",
    "is_link_like",
    "iter_safe_files",
    "resolve_caption_path",
    "resolve_dataset_directory",
    "resolve_dataset_file",
    "resolve_dataset_source_directory",
    "resolve_file_under",
    "resolve_image_path",
]
