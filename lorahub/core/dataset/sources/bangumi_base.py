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

import shutil
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


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
    api = HfApi()
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
    full = repo_id if "/" in repo_id else f"BangumiBase/{repo_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    cached = hf_hub_download(
        repo_id=full,
        filename=f"{character_id}/preview_{index}.png",
        repo_type="dataset",
    )
    dst = output_dir / f"preview_{index}.png"
    shutil.copy2(cached, dst)
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
    full = repo_id if "/" in repo_id else f"BangumiBase/{repo_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if on_progress:
        on_progress(f"resolving {full}/{character_id}/dataset.zip")
    try:
        cached_zip = hf_hub_download(
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
    extracted: list[Path] = []
    with zipfile.ZipFile(zip_path) as zf:
        members = sorted(
            (m for m in zf.namelist() if Path(m).suffix.lower() in _IMAGE_EXTS),
        )
        if limit is not None:
            members = members[:limit]
        for name in members:
            target_name = Path(name).name
            target = output_dir / target_name
            with zf.open(name) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(target)
    return extracted


def _read_dataset_license(repo_id: str) -> str | None:
    try:
        info = HfApi().dataset_info(repo_id)
    except Exception:  # noqa: BLE001
        return None
    card_data = getattr(info, "card_data", None)
    if card_data is None:
        return None
    return getattr(card_data, "license", None)
