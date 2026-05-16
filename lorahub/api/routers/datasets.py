"""Dataset scanning, caption I/O, and on-the-fly thumbnails."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from lorahub.api.dataset_files import (
    get_or_build_thumbnail,
    resolve_caption_path,
    resolve_image_path,
)
from lorahub.api.helpers import _scan_dataset_path

router = APIRouter(prefix="/api")


@router.get("/datasets/scan")
def scan_dataset(path: str, recursive: bool = False, limit: int = 40) -> dict[str, Any]:
    return _scan_dataset_path(Path(path), recursive=recursive, limit=limit)


@router.get("/datasets/thumb")
def dataset_thumbnail(path: str, size: int = 256) -> FileResponse:
    """Return a cached square-bounded WEBP thumbnail for an image.

    `path` must resolve to an image inside an allowed dataset root (the
    cwd, `LORAHUB_DATASETS_ROOT`, or any registered job workspace). Cache
    files live under `runs/.thumbs/<sha256>.webp` and are reused across
    requests; the browser also gets a 24h `Cache-Control` header so the
    grid stays responsive on revisit.
    """
    try:
        image = resolve_image_path(path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not image.is_file():
        raise HTTPException(status_code=404, detail="image not found")
    try:
        thumb = get_or_build_thumbnail(image, size)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError:
        # Generation failed (corrupt image, unsupported codec, etc.). 404
        # keeps the frontend's broken-image fallback simple.
        raise HTTPException(status_code=404, detail="thumbnail unavailable") from None
    return FileResponse(
        thumb,
        media_type="image/webp",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@router.get("/datasets/caption")
def get_caption(path: str) -> dict[str, Any]:
    """Read the `.txt` caption sibling for `path`.

    Returns `caption=null` when the companion file does not exist yet --
    that's a normal "needs tagging" state, not a 404.
    """
    try:
        caption_path = resolve_caption_path(path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    caption: str | None = None
    if caption_path.is_file():
        try:
            caption = caption_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"path": str(caption_path), "caption": caption}


class CaptionUpdate(BaseModel):
    path: str = Field(..., description="Path to the image whose .txt to write")
    caption: str = Field(default="", description="Caption text to persist (UTF-8)")


@router.put("/datasets/caption")
def put_caption(body: CaptionUpdate) -> dict[str, Any]:
    """Write `caption` to the `.txt` sibling of `path`.

    The file is created if missing. We always write UTF-8 without a BOM
    and normalise line endings to LF so kohya's tooling sees the same text
    regardless of the platform that produced it.
    """
    try:
        caption_path = resolve_caption_path(body.path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    text = body.caption.replace("\r\n", "\n").replace("\r", "\n")
    try:
        caption_path.parent.mkdir(parents=True, exist_ok=True)
        caption_path.write_text(text, encoding="utf-8", newline="\n")
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "path": str(caption_path),
        "caption": text,
        "bytes": len(text.encode("utf-8")),
    }
