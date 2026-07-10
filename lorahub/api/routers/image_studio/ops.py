"""Image Studio pending-ops endpoints (queue + apply mutations)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from lorahub.api.dataset_files import _resolve_under_roots
from lorahub.api.image_studio_store import PendingOp

from ._shared import (
    _atomic_save_image,
    _atomic_write_text,
    _clear_dataset_view_caches,
    _file_mutation,
    _soft_delete,
    _store,
    _writable_dataset_file,
)

router = APIRouter(prefix="/api/image-studio", tags=["image-studio"])


class PendingOpInput(BaseModel):
    path: str
    op: Literal[
        "rotate",
        "flip",
        "replace_caption",
        "merge_caption",
        "favorite",
        "delete",
    ]
    payload: dict[str, Any] = Field(default_factory=dict)


@router.post("/ops")
def add_op(body: PendingOpInput) -> dict[str, Any]:
    file_path = _writable_dataset_file(body.path)
    store = _store()
    op = store.add_pending_op(
        PendingOp(id="", image_path=str(file_path), op=body.op, payload=body.payload)
    )
    return {"id": op.id, "op": op.op, "payload": op.payload, "createdAt": op.created_at}


@router.get("/ops")
def list_ops(path: str | None = None) -> dict[str, Any]:
    if path:
        _resolve_under_roots(path)
    store = _store()
    ops = store.list_pending_ops(path)
    return {
        "ops": [
            {"id": o.id, "imagePath": o.image_path, "op": o.op,
             "payload": o.payload, "createdAt": o.created_at}
            for o in ops
        ]
    }


@router.delete("/ops/{op_id}")
def delete_op(op_id: str) -> dict[str, bool]:
    store = _store()
    deleted = store.delete_pending_op(op_id)
    if not deleted:
        raise HTTPException(404, "op not found")
    return {"ok": True}


class ApplyOpsInput(BaseModel):
    path: str


@router.post("/ops/apply")
def apply_ops(body: ApplyOpsInput) -> dict[str, Any]:
    file_path = _writable_dataset_file(body.path)
    store = _store()
    ops = store.list_pending_ops(str(file_path))
    applied: list[str] = []
    errors: list[dict[str, str]] = []
    for op in ops:
        try:
            _execute_op(file_path, op)
            applied.append(op.id)
            store.delete_pending_op(op.id)
        except Exception as exc:  # noqa: BLE001
            errors.append({"id": op.id, "error": str(exc)})
    if applied:
        _clear_dataset_view_caches(file_path.parent)
    return {"applied": applied, "errors": errors}


def _execute_op(file_path: Path, op: PendingOp) -> None:
    """Execute a single pending op on disk. Raises on failure."""
    from PIL import Image  # noqa: PLC0415

    with _file_mutation(file_path):
        if op.op == "rotate":
            raw_degrees = op.payload.get("degrees", 90)
            if not isinstance(raw_degrees, (int, float)) or isinstance(
                raw_degrees,
                bool,
            ):
                raise ValueError("rotate degrees must be numeric")
            degrees = float(raw_degrees) % 360
            with Image.open(file_path) as img:
                image_format = img.format
                rotated = img.rotate(-degrees, expand=True)
                rotated.load()
            _atomic_save_image(rotated, file_path, image_format=image_format)
        elif op.op == "flip":
            direction = op.payload.get("direction", "horizontal")
            if direction not in {"horizontal", "vertical"}:
                raise ValueError("flip direction must be horizontal or vertical")
            with Image.open(file_path) as img:
                from PIL import ImageOps  # noqa: PLC0415

                image_format = img.format
                flipped = (
                    ImageOps.mirror(img)
                    if direction == "horizontal"
                    else ImageOps.flip(img)
                )
                flipped.load()
            _atomic_save_image(flipped, file_path, image_format=image_format)
        elif op.op == "replace_caption":
            caption = op.payload.get("caption", "")
            if not isinstance(caption, str):
                raise ValueError("caption must be text")
            _atomic_write_text(file_path.with_suffix(".txt"), caption)
        elif op.op == "merge_caption":
            caption = op.payload.get("caption", "")
            if not isinstance(caption, str):
                raise ValueError("caption must be text")
            caption_path = file_path.with_suffix(".txt")
            existing = (
                caption_path.read_text(encoding="utf-8")
                if caption_path.is_file()
                else ""
            )
            merged = f"{existing.strip()}, {caption}".strip(", ")
            _atomic_write_text(caption_path, merged)
        elif op.op == "favorite":
            pass  # handled at annotation level, no file mutation
        elif op.op == "delete":
            _soft_delete(file_path)
        else:
            raise ValueError(f"unknown op: {op.op}")
