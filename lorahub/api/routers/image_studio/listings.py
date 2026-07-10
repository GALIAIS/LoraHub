"""Image Studio listings + per-image inspect endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from lorahub.api.dataset_files import _resolve_under_roots
from lorahub.api.image_studio_store import ImageStudioStore

from ._shared import _image_item, _scan_images, _store

router = APIRouter(prefix="/api/image-studio", tags=["image-studio"])


class ListQuery(BaseModel):
    path: str
    recursive: bool = False
    page: int = 1
    limit: int = 48
    sort: Literal["name", "mtime", "size"] = "name"
    filter_caption: Literal["with", "missing"] | None = Field(
        None,
        alias="filter.caption",
    )
    filter_quality: str | None = Field(None, alias="filter.quality")
    filter_aspect: Literal["landscape", "portrait", "square"] | None = Field(
        None,
        alias="filter.aspect",
    )

    model_config = {"populate_by_name": True}


def _sort_images(images: list[Path], sort: str) -> list[Path]:
    def stat_value(path: Path, attr: str) -> float:
        try:
            return float(getattr(path.stat(), attr))
        except OSError:
            return -1.0

    if sort == "mtime":
        return sorted(
            images,
            key=lambda path: stat_value(path, "st_mtime"),
            reverse=True,
        )
    if sort == "size":
        return sorted(
            images,
            key=lambda path: stat_value(path, "st_size"),
            reverse=True,
        )
    return sorted(images, key=lambda p: p.name.lower())


@router.get("/list")
def list_images(
    path: str,
    recursive: bool = False,
    page: int = 1,
    limit: int = 48,
    sort: Literal["name", "mtime", "size"] = "name",
    filter_caption: Literal["with", "missing"] | None = None,
    filter_quality: str | None = None,
    filter_aspect: Literal["landscape", "portrait", "square"] | None = None,
) -> dict[str, Any]:
    directory = _resolve_under_roots(path)
    if not directory.is_dir():
        raise HTTPException(400, f"not a directory: {path}")
    images = [image for image in _scan_images(directory, recursive) if image.is_file()]
    images = _sort_images(images, sort)

    store = _store()

    # Apply filters
    if filter_caption or filter_quality or filter_aspect:
        images = _apply_filters(
            images, store,
            caption=filter_caption,
            quality=filter_quality,
            aspect=filter_aspect,
        )

    page = max(int(page), 1)
    limit = min(max(int(limit), 1), 500)
    total = len(images)
    start = (page - 1) * limit
    page_items = images[start : start + limit]
    items: list[dict[str, Any]] = []
    for image in page_items:
        try:
            items.append(_image_item(image, directory, store))
        except OSError:
            # A concurrent curation operation can move a cached entry after
            # the page was sliced. Return the rest of the page, not a 500.
            continue
    return {"path": str(directory), "total": total, "page": page, "limit": limit, "items": items}


def _apply_filters(
    images: list[Path],
    store: ImageStudioStore,
    *,
    caption: str | None = None,
    quality: str | None = None,
    aspect: str | None = None,
) -> list[Path]:
    """Filter image list by caption existence, quality label, and aspect ratio."""
    filtered: list[Path] = []
    for p in images:
        # Caption filter
        if caption:
            has_caption = p.with_suffix(".txt").is_file()
            if caption == "with" and not has_caption:
                continue
            if caption == "missing" and has_caption:
                continue

        # Fetch annotation once if needed by quality or aspect filters
        ann = None
        if quality or aspect:
            ann = store.get_annotation(str(p))

        # Quality filter (checks annotation ai_quality_label or favorite)
        if quality:
            if quality in ("favourite", "favorite"):
                if not ann or not ann.favorite:
                    continue
            else:
                if not ann or ann.ai_quality_label != quality:
                    continue

        # Aspect ratio filter
        if aspect:
            w: int | None = ann.width if ann else None
            h: int | None = ann.height if ann else None
            if w is None or h is None:
                try:
                    from PIL import Image  # noqa: PLC0415
                    with Image.open(p) as img:
                        w, h = img.size
                except Exception:  # noqa: BLE001
                    continue
            if aspect == "landscape" and not (w > h):
                continue
            if aspect == "portrait" and not (h > w):
                continue
            if aspect == "square" and w != h:
                continue

        filtered.append(p)
    return filtered


@router.get("/image")
def get_image(path: str) -> dict[str, Any]:
    file_path = _resolve_under_roots(path)
    if not file_path.is_file():
        raise HTTPException(404, "image not found")
    store = _store()
    directory = file_path.parent
    item = _image_item(file_path, directory, store)
    phashes = store.get_phashes(str(file_path))
    pending = store.list_pending_ops(str(file_path))
    item["phash"] = {ph.algo: ph.hash for ph in phashes}
    item["pendingOps"] = [
        {"id": op.id, "op": op.op, "payload": op.payload, "createdAt": op.created_at}
        for op in pending
    ]
    return item


# Re-export the helper used by `_apply_filters`-aware tests / future use.
__all__ = [
    "router",
    "ListQuery",
    "list_images",
    "get_image",
    "_apply_filters",
    "_sort_images",
]
