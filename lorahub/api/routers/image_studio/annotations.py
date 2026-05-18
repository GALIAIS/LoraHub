"""Image Studio annotation CRUD endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from lorahub.api.dataset_files import _resolve_under_roots
from lorahub.api.image_studio_store import ImageAnnotation

from ._shared import _ann_to_dict, _file_sha256, _store

router = APIRouter(prefix="/api/image-studio", tags=["image-studio"])


class AnnotationInput(BaseModel):
    path: str
    userQualityLabel: str | None = None
    userNotes: str | None = None
    favorite: bool | None = None
    softDeleted: bool | None = None


@router.put("/annotations")
def save_annotation(body: AnnotationInput) -> dict[str, Any]:
    file_path = _resolve_under_roots(body.path)
    if not file_path.is_file():
        raise HTTPException(404, "image not found")
    store = _store()
    existing = store.get_annotation(str(file_path))
    sha = existing.sha256 if existing else _file_sha256(file_path)
    ann = ImageAnnotation(
        image_path=str(file_path),
        sha256=sha,
        width=existing.width if existing else None,
        height=existing.height if existing else None,
        bytes=existing.bytes if existing else int(file_path.stat().st_size),
        ai_caption=existing.ai_caption if existing else None,
        ai_caption_provider=existing.ai_caption_provider if existing else None,
        ai_caption_at=existing.ai_caption_at if existing else None,
        ai_quality_score=existing.ai_quality_score if existing else None,
        ai_quality_label=existing.ai_quality_label if existing else None,
        ai_quality_reason=existing.ai_quality_reason if existing else None,
        ai_quality_at=existing.ai_quality_at if existing else None,
        ai_composition=existing.ai_composition if existing else None,
        ai_composition_at=existing.ai_composition_at if existing else None,
        ai_trigger_words=existing.ai_trigger_words if existing else None,
        ai_trigger_words_at=existing.ai_trigger_words_at if existing else None,
        user_quality_label=(
            body.userQualityLabel if body.userQualityLabel is not None
            else (existing.user_quality_label if existing else None)
        ),
        user_notes=(
            body.userNotes if body.userNotes is not None
            else (existing.user_notes if existing else None)
        ),
        soft_deleted=(
            body.softDeleted if body.softDeleted is not None
            else (existing.soft_deleted if existing else False)
        ),
        favorite=(
            body.favorite if body.favorite is not None
            else (existing.favorite if existing else False)
        ),
    )
    store.upsert_annotation(ann)
    return {"ok": True, "annotation": _ann_to_dict(ann)}


@router.delete("/annotations")
def delete_annotation(path: str) -> dict[str, bool]:
    _resolve_under_roots(path)
    store = _store()
    store.delete_annotation(path)
    return {"ok": True}
