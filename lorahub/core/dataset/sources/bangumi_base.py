"""Download a single character subset from a BangumiBase Hugging Face dataset.

BangumiBase repositories follow a uniform layout: every detected character lives
under a numeric directory containing `dataset.zip` (all images for that
character) plus 8 `preview_*.png` thumbnails. We use the `dataset.zip` only —
unpack it into the user's target directory and seed empty caption files so they
can be tagged later (manually or by a future auto-tagger).

License-wise BangumiBase repos are MIT (per their dataset cards as of 2026-05).
We still let the caller verify the specific repo's license at fetch time.
"""

from __future__ import annotations

import os
import shutil
import stat
import tempfile
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from lorahub.core.net import hf_api as _make_hf_api, hf_download

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_MAX_ARCHIVE_ENTRIES = 100_000
_MAX_EXPANDED_BYTES = 20 * 1024**3
_MAX_IMAGE_BYTES = 512 * 1024**2
_COPY_CHUNK_BYTES = 1024**2


@dataclass(frozen=True, slots=True)
class FetchResult:
    repo_id: str
    character_id: str
    output_dir: Path
    image_count: int
    license: str | None


class BangumiBaseError(RuntimeError):
    """Raised when a BangumiBase fetch cannot be completed."""


def list_characters(repo_id: str) -> list[str]:
    """Return every character directory id (e.g. ['0', '1', ..., '106'])."""
    full = repo_id if "/" in repo_id else f"BangumiBase/{repo_id}"
    api = _make_hf_api()
    files = api.list_repo_files(full, repo_type="dataset")
    ids: set[str] = set()
    for f in files:
        head, _, rest = f.partition("/")
        if not rest or not head.isdigit():
            continue
        ids.add(head)
    return sorted(ids, key=int)


def download_preview(
    repo_id: str,
    character_id: str,
    output_dir: Path,
    *,
    index: int = 1,
) -> Path:
    """Download one preview thumbnail (1-8) so the user can identify the character."""
    _validate_character_id(character_id)
    if isinstance(index, bool) or not 1 <= index <= 8:
        raise BangumiBaseError("preview index must be an integer from 1 to 8")
    _prepare_output_dir(output_dir)
    full = repo_id if "/" in repo_id else f"BangumiBase/{repo_id}"
    cached = hf_download(
        repo_id=full,
        filename=f"{character_id}/preview_{index}.png",
        repo_type="dataset",
    )
    dst = output_dir / f"preview_{index}.png"
    if _is_link_like(dst):
        raise BangumiBaseError(f"refusing to replace linked preview: {dst}")
    if not dst.exists():
        _publish_file(Path(cached), dst)
    return dst


def fetch_character(
    repo_id: str,
    character_id: str,
    output_dir: Path,
    *,
    limit: int | None = None,
    seed_captions: bool = True,
    on_progress: Callable[[str], None] | None = None,
) -> FetchResult:
    """Download one character's `dataset.zip`, unpack images, optionally seed captions.

    Parameters mirror the CLI: caller can cap how many images to keep with
    `limit`, and ask us to write empty `.txt` next to each image (kohya tag-file
    convention) so the dataset is immediately usable as training input.
    """
    _validate_character_id(character_id)
    if limit is not None and (
        isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
    ):
        raise BangumiBaseError("limit must be a positive integer")
    _prepare_output_dir(output_dir)
    full = repo_id if "/" in repo_id else f"BangumiBase/{repo_id}"

    if on_progress:
        on_progress(f"resolving {full}/{character_id}/dataset.zip")
    try:
        cached_zip = hf_download(
            repo_id=full,
            filename=f"{character_id}/dataset.zip",
            repo_type="dataset",
        )
    except Exception as exc:
        msg = f"failed to download {full}/{character_id}/dataset.zip: {exc}"
        raise BangumiBaseError(msg) from exc

    if on_progress:
        on_progress(f"unpacking into {output_dir}")
    images = _extract_images(Path(cached_zip), output_dir, limit=limit)

    if seed_captions:
        for img in images:
            caption = img.with_suffix(".txt")
            if not caption.exists():
                caption.write_text("", encoding="utf-8")

    license_str = _read_dataset_license(full)
    return FetchResult(
        repo_id=full,
        character_id=character_id,
        output_dir=output_dir,
        image_count=len(images),
        license=license_str,
    )


def _extract_images(zip_path: Path, output_dir: Path, *, limit: int | None) -> list[Path]:
    """Extract selected images without overwriting existing dataset files.

    Archive members are staged first, bounded by declared and streamed byte
    limits, then published with exclusive-create semantics. Nested archive
    paths are flattened for the existing CLI contract; duplicate basenames
    receive deterministic numeric suffixes instead of replacing one another.
    """
    max_entries = _env_positive_int(
        "LORAHUB_BANGUMI_MAX_ARCHIVE_ENTRIES", _MAX_ARCHIVE_ENTRIES
    )
    max_bytes = _env_positive_int(
        "LORAHUB_BANGUMI_MAX_EXPANDED_BYTES", _MAX_EXPANDED_BYTES
    )
    max_image_bytes = _env_positive_int(
        "LORAHUB_BANGUMI_MAX_IMAGE_BYTES", _MAX_IMAGE_BYTES
    )
    try:
        with zipfile.ZipFile(zip_path) as zf:
            entries = zf.infolist()
            if len(entries) > max_entries:
                raise BangumiBaseError(
                    f"archive has {len(entries)} entries; limit is {max_entries}"
                )
            members = sorted(
                (
                    entry
                    for entry in entries
                    if not entry.is_dir()
                    and Path(entry.filename).suffix.lower() in _IMAGE_EXTS
                ),
                key=lambda entry: entry.filename,
            )
            if limit is not None:
                members = members[:limit]
            _validate_members(
                members,
                max_total_bytes=max_bytes,
                max_image_bytes=max_image_bytes,
            )

            total_written = 0
            with tempfile.TemporaryDirectory(
                dir=output_dir,
                prefix=".bangumi-extract-",
            ) as temp_dir:
                staging = Path(temp_dir)
                staged_members: list[tuple[Path, str]] = []
                for index, member in enumerate(members):
                    staged = staging / f"{index:08d}{Path(member.filename).suffix.lower()}"
                    written = _copy_member(
                        zf,
                        member,
                        staged,
                        remaining_bytes=max_bytes - total_written,
                        max_image_bytes=max_image_bytes,
                    )
                    total_written += written
                    staged_members.append((staged, Path(member.filename).name))

                extracted: list[Path] = []
                try:
                    for staged, source_name in staged_members:
                        target = _next_available_path(output_dir, source_name)
                        _publish_file(staged, target)
                        extracted.append(target)
                except Exception:
                    for published in extracted:
                        published.unlink(missing_ok=True)
                    raise
                return extracted
    except BangumiBaseError:
        raise
    except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise BangumiBaseError(f"invalid or unreadable dataset archive: {exc}") from exc


def _validate_character_id(character_id: str) -> None:
    if (
        not isinstance(character_id, str)
        or not character_id.isascii()
        or not character_id.isdigit()
    ):
        raise BangumiBaseError("character id must contain ASCII digits only")


def _is_link_like(path: Path) -> bool:
    try:
        if path.is_symlink():
            return True
        is_junction = getattr(os.path, "isjunction", None)
        return bool(is_junction and is_junction(path))
    except OSError:
        return True


def _prepare_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if _is_link_like(output_dir) or not output_dir.is_dir():
        raise BangumiBaseError(f"output directory must be a real directory: {output_dir}")


def _env_positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _validate_members(
    members: list[zipfile.ZipInfo],
    *,
    max_total_bytes: int,
    max_image_bytes: int,
) -> None:
    declared_total = 0
    for member in members:
        unix_mode = (member.external_attr >> 16) & 0xFFFF
        if unix_mode and stat.S_ISLNK(unix_mode):
            raise BangumiBaseError(f"archive contains a symbolic link: {member.filename}")
        if member.flag_bits & 0x1:
            raise BangumiBaseError(f"archive contains an encrypted file: {member.filename}")
        if member.file_size < 0 or member.file_size > max_image_bytes:
            raise BangumiBaseError(
                f"archive image exceeds {max_image_bytes} bytes: {member.filename}"
            )
        declared_total += member.file_size
        if declared_total > max_total_bytes:
            raise BangumiBaseError(
                f"archive expands beyond the {max_total_bytes} byte limit"
            )


def _copy_member(
    zf: zipfile.ZipFile,
    member: zipfile.ZipInfo,
    target: Path,
    *,
    remaining_bytes: int,
    max_image_bytes: int,
) -> int:
    written = 0
    try:
        with zf.open(member) as src, target.open("xb") as dst:
            while chunk := src.read(_COPY_CHUNK_BYTES):
                written += len(chunk)
                if written > max_image_bytes or written > remaining_bytes:
                    raise BangumiBaseError(
                        f"archive extraction limit exceeded by {member.filename}"
                    )
                dst.write(chunk)
            dst.flush()
            os.fsync(dst.fileno())
    except Exception:
        target.unlink(missing_ok=True)
        raise
    return written


def _next_available_path(output_dir: Path, source_name: str) -> Path:
    clean_name = Path(source_name).name
    stem = Path(clean_name).stem or "image"
    suffix = Path(clean_name).suffix.lower()
    candidate = output_dir / f"{stem}{suffix}"
    index = 1
    while candidate.exists() or _is_link_like(candidate):
        candidate = output_dir / f"{stem}_{index}{suffix}"
        index += 1
    return candidate


def _publish_file(source: Path, target: Path) -> None:
    """Publish a staged file without ever replacing an existing path."""
    if target.exists() or _is_link_like(target):
        raise BangumiBaseError(f"refusing to overwrite existing file: {target}")
    try:
        os.link(source, target)
    except FileExistsError as exc:
        raise BangumiBaseError(f"refusing to overwrite existing file: {target}") from exc
    except OSError:
        created = False
        try:
            with source.open("rb") as src, target.open("xb") as dst:
                created = True
                shutil.copyfileobj(src, dst, length=_COPY_CHUNK_BYTES)
                dst.flush()
                os.fsync(dst.fileno())
        except Exception:
            if created:
                target.unlink(missing_ok=True)
            raise


def _read_dataset_license(repo_id: str) -> str | None:
    try:
        info = _make_hf_api().dataset_info(repo_id)
    except Exception:  # noqa: BLE001
        return None
    card_data = getattr(info, "card_data", None)
    if card_data is None:
        return None
    return getattr(card_data, "license", None)
