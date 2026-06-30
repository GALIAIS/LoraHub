"""Image Studio pending-ops endpoints (queue + apply mutations)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from lorahub.api.dataset_files import _resolve_under_roots
from lorahub.api.image_studio_store import PendingOp

from ._shared import _clear_dataset_view_caches, _soft_delete, _store

router = APIRouter(prefix="/api/image-studio", tags=["image-studio"])


class PendingOpInput(BaseModel):
    path: str
    op: str
    payload: dict[str, Any] = Field(default_factory=dict)


@router.post("/ops")
def add_op(body: PendingOpInput) -> dict[str, Any]:
    _resolve_under_roots(body.path)
    store = _store()
    op = store.add_pending_op(
        PendingOp(id="", image_path=body.path, op=body.op, payload=body.payload)
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
    file_path = _resolve_under_roots(body.path)
    if not file_path.is_file():
        raise HTTPException(404, "image not found")
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

    if op.op == "rotate":
        degrees = op.payload.get("degrees", 90)
        with Image.open(file_path) as img:
            rotated = img.rotate(-degrees, expand=True)
            rotated.save(file_path)
    elif op.op == "flip":
        direction = op.payload.get("direction", "horizontal")
        with Image.open(file_path) as img:
            from PIL import ImageOps  # noqa: PLC0415
            flipped = ImageOps.mirror(img) if direction == "horizontal" else ImageOps.flip(img)
            flipped.save(file_path)
    elif op.op == "replace_caption":
        caption_path = file_path.with_suffix(".txt")
        caption_path.write_text(op.payload.get("caption", ""), encoding="utf-8")
    elif op.op == "merge_caption":
        caption_path = file_path.with_suffix(".txt")
        existing = caption_path.read_text(encoding="utf-8") if caption_path.is_file() else ""
        merged = existing.strip() + ", " + op.payload.get("caption", "")
        caption_path.write_text(merged.strip(", "), encoding="utf-8")
    elif op.op == "favorite":
        pass  # handled at annotation level, no file mutation
    elif op.op == "delete":
        _soft_delete(file_path)
    else:
        raise ValueError(f"unknown op: {op.op}")
