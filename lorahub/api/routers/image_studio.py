"""Image Studio API router — IS-0 foundation endpoints.

Provides listing, per-image inspect, annotations CRUD, and pending ops
management. All paths are validated against the dataset allow-list.
"""

from __future__ import annotations

import hashlib
import json as json_stdlib
import os
import shutil
import tempfile
import threading
import time as _time
import uuid as _uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from lorahub.api import app as app_module
from lorahub.api.dataset_files import IMAGE_SUFFIXES, _resolve_under_roots
from lorahub.api.image_studio_store import (
    ImageAnnotation,
    ImageEmbedding,
    ImageStudioStore,
    PendingOp,
)

router = APIRouter(prefix="/api/image-studio", tags=["image-studio"])


def _store() -> ImageStudioStore:
    s = app_module._image_studio_store
    if s is None:
        raise HTTPException(503, "image studio store not initialised")
    return s


# --------------------------------------------------------------------------- #
# Listing
# --------------------------------------------------------------------------- #


class ListQuery(BaseModel):
    path: str
    recursive: bool = False
    page: int = 1
    limit: int = 48
    sort: str = "name"
    filter_caption: str | None = Field(None, alias="filter.caption")
    filter_quality: str | None = Field(None, alias="filter.quality")
    filter_aspect: str | None = Field(None, alias="filter.aspect")

    model_config = {"populate_by_name": True}


def _scan_images(directory: Path, recursive: bool) -> list[Path]:
    """Collect image files from directory, respecting IMAGE_SUFFIXES."""
    results: list[Path] = []
    if recursive:
        for root, _dirs, files in os.walk(directory):
            for f in files:
                p = Path(root) / f
                if p.suffix.lower() in IMAGE_SUFFIXES:
                    results.append(p)
    else:
        for p in directory.iterdir():
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
                results.append(p)
    return results


def _image_item(p: Path, directory: Path, store: ImageStudioStore) -> dict[str, Any]:
    """Build a single item dict for the listing response."""
    stat = p.stat()
    rel = str(p.relative_to(directory)).replace("\\", "/")
    ann = store.get_annotation(str(p))
    caption_path = p.with_suffix(".txt")
    caption_exists = caption_path.is_file()
    caption = caption_path.read_text(encoding="utf-8").strip() if caption_exists else None
    return {
        "path": str(p),
        "relativePath": rel,
        "name": p.name,
        "width": ann.width if ann else None,
        "height": ann.height if ann else None,
        "bytes": stat.st_size,
        "mtime": stat.st_mtime,
        "caption": caption,
        "captionExists": caption_exists,
        "annotation": _ann_to_dict(ann) if ann else None,
        "thumbUrl": f"/api/datasets/thumb?path={str(p)}&size=256",
    }


def _ann_to_dict(ann: ImageAnnotation) -> dict[str, Any]:
    return {
        "aiCaption": ann.ai_caption,
        "aiQualityScore": ann.ai_quality_score,
        "aiQualityLabel": ann.ai_quality_label,
        "aiQualityReason": ann.ai_quality_reason,
        "aiComposition": ann.ai_composition,
        "aiTriggerWords": ann.ai_trigger_words,
        "userQualityLabel": ann.user_quality_label,
        "userNotes": ann.user_notes,
        "softDeleted": ann.soft_deleted,
        "favorite": ann.favorite,
    }


def _sort_images(images: list[Path], sort: str) -> list[Path]:
    if sort == "mtime":
        return sorted(images, key=lambda p: p.stat().st_mtime, reverse=True)
    if sort == "size":
        return sorted(images, key=lambda p: p.stat().st_size, reverse=True)
    return sorted(images, key=lambda p: p.name.lower())


@router.get("/list")
def list_images(
    path: str,
    recursive: bool = False,
    page: int = 1,
    limit: int = 48,
    sort: str = "name",
    filter_caption: str | None = None,
    filter_quality: str | None = None,
    filter_aspect: str | None = None,
) -> dict[str, Any]:
    directory = _resolve_under_roots(path)
    if not directory.is_dir():
        raise HTTPException(400, f"not a directory: {path}")
    images = _scan_images(directory, recursive)
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

    total = len(images)
    start = (page - 1) * limit
    page_items = images[start : start + limit]
    items = [_image_item(p, directory, store) for p in page_items]
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


# --------------------------------------------------------------------------- #
# Per-image inspect
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# Annotations CRUD
# --------------------------------------------------------------------------- #


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


# --------------------------------------------------------------------------- #
# Pending ops
# --------------------------------------------------------------------------- #


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


def _soft_delete(file_path: Path) -> None:
    """Move file + sidecar to trash directory."""
    from datetime import UTC, datetime  # noqa: PLC0415

    trash_dir = Path("runs") / "_image_studio_trash" / datetime.now(UTC).strftime("%Y-%m-%d")
    trash_dir.mkdir(parents=True, exist_ok=True)
    dest = trash_dir / file_path.name
    file_path.rename(dest)
    caption = file_path.with_suffix(".txt")
    if caption.is_file():
        caption.rename(trash_dir / caption.name)


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# AI batch endpoints
# --------------------------------------------------------------------------- #


class AIBatchCaptionInput(BaseModel):
    path: str
    recursive: bool = False
    task: str = "tagging.assist"
    mergeStrategy: str = "replace"


@router.post("/ai/caption")
def ai_batch_caption(body: AIBatchCaptionInput) -> dict[str, Any]:
    """Queue AI captioning for all images in a directory.

    For IS-3 this is synchronous (processes sequentially). A future
    iteration will use the session pattern for async progress.
    """
    from lorahub.api import app as app_mod  # noqa: PLC0415
    from lorahub.core.ai import client as ai_client  # noqa: PLC0415

    directory = _resolve_under_roots(body.path)
    if not directory.is_dir():
        raise HTTPException(400, "not a directory")

    ai_store = app_mod._ai_store
    if ai_store is None:
        raise HTTPException(503, "AI store not initialised")

    route = ai_store.get_route(body.task)
    if route is None or not (route.provider_id and route.model_id):
        route = ai_store.get_route("global.default")
    if route is None or not (route.provider_id and route.model_id):
        raise HTTPException(409, f"no AI route for task {body.task!r}")

    images = _scan_images(directory, body.recursive)
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    store = _store()
    for img_path in images:
        try:
            import base64  # noqa: PLC0415
            import mimetypes  # noqa: PLC0415

            mime = mimetypes.guess_type(str(img_path))[0] or "image/png"
            data = img_path.read_bytes()
            b64 = base64.b64encode(data).decode("ascii")
            data_url = f"data:{mime};base64,{b64}"

            messages: list[dict[str, Any]] = []
            if route.system_prompt:
                messages.append({"role": "system", "content": route.system_prompt})
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            })

            result = ai_client.invoke(
                ai_store,
                provider_id=route.provider_id,
                model_id=route.model_id,
                messages=messages,
                route=route,
            )

            caption_path = img_path.with_suffix(".txt")
            existing = caption_path.read_text(encoding="utf-8") if caption_path.is_file() else ""

            if body.mergeStrategy == "append":
                new_caption = (existing.strip() + ", " + result.content).strip(", ")
            elif body.mergeStrategy == "rewrite":
                new_caption = result.content
            else:
                new_caption = result.content

            caption_path.write_text(new_caption, encoding="utf-8")

            ann = store.get_annotation(str(img_path))
            if ann is None:
                from lorahub.api.image_studio_store import ImageAnnotation  # noqa: PLC0415
                ann = ImageAnnotation(
                    image_path=str(img_path),
                    sha256=_file_sha256(img_path),
                )
            ann.ai_caption = result.content
            ann.ai_caption_provider = f"{result.provider_name}/{result.model_id}"
            from datetime import UTC, datetime  # noqa: PLC0415
            ann.ai_caption_at = datetime.now(UTC).isoformat()
            store.upsert_annotation(ann)

            results.append({"path": str(img_path), "caption": new_caption})
        except Exception as exc:  # noqa: BLE001
            errors.append({"path": str(img_path), "error": str(exc)})

    return {"processed": len(results), "results": results, "errors": errors}


class AIBatchQualityInput(BaseModel):
    path: str
    recursive: bool = False
    task: str = "quality.score"


@router.post("/ai/quality")
def ai_batch_quality(body: AIBatchQualityInput) -> dict[str, Any]:
    """Score image quality via VLM for all images in a directory."""
    from lorahub.api import app as app_mod  # noqa: PLC0415
    from lorahub.core.ai import client as ai_client  # noqa: PLC0415

    directory = _resolve_under_roots(body.path)
    if not directory.is_dir():
        raise HTTPException(400, "not a directory")

    ai_store = app_mod._ai_store
    if ai_store is None:
        raise HTTPException(503, "AI store not initialised")

    route = ai_store.get_route(body.task)
    if route is None or not (route.provider_id and route.model_id):
        route = ai_store.get_route("global.default")
    if route is None or not (route.provider_id and route.model_id):
        raise HTTPException(409, f"no AI route for task {body.task!r}")

    images = _scan_images(directory, body.recursive)
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    store = _store()
    for img_path in images:
        try:
            import base64  # noqa: PLC0415
            import json as json_mod  # noqa: PLC0415
            import mimetypes  # noqa: PLC0415

            mime = mimetypes.guess_type(str(img_path))[0] or "image/png"
            data = img_path.read_bytes()
            b64 = base64.b64encode(data).decode("ascii")
            data_url = f"data:{mime};base64,{b64}"

            system_prompt = route.system_prompt or (
                'Rate this training image on a 0-100 scale. '
                'Return JSON: {"score": 0-100, "label": "good"|"medium"|"bad", "reason": "..."}'
            )

            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]},
            ]

            result = ai_client.invoke(
                ai_store,
                provider_id=route.provider_id,
                model_id=route.model_id,
                messages=messages,
                route=route,
            )

            score: float | None = None
            label: str | None = None
            reason: str | None = None
            try:
                parsed = json_mod.loads(result.content)
                score = float(parsed.get("score", 0)) / 100.0
                label = parsed.get("label")
                reason = parsed.get("reason")
            except (json_mod.JSONDecodeError, ValueError, TypeError):
                reason = result.content

            ann = store.get_annotation(str(img_path))
            if ann is None:
                from lorahub.api.image_studio_store import ImageAnnotation  # noqa: PLC0415
                ann = ImageAnnotation(
                    image_path=str(img_path),
                    sha256=_file_sha256(img_path),
                )
            ann.ai_quality_score = score
            ann.ai_quality_label = label
            ann.ai_quality_reason = reason
            from datetime import UTC, datetime  # noqa: PLC0415
            ann.ai_quality_at = datetime.now(UTC).isoformat()
            store.upsert_annotation(ann)

            results.append({
                "path": str(img_path),
                "score": score,
                "label": label,
                "reason": reason,
            })
        except Exception as exc:  # noqa: BLE001
            errors.append({"path": str(img_path), "error": str(exc)})

    return {"processed": len(results), "results": results, "errors": errors}


# --------------------------------------------------------------------------- #
# Dedupe / similarity endpoints
# --------------------------------------------------------------------------- #


class DedupeScanInput(BaseModel):
    path: str
    recursive: bool = False
    algo: str = "phash64"
    threshold: int = 10


@router.post("/dedupe/scan")
def dedupe_scan(body: DedupeScanInput) -> dict[str, Any]:
    """Compute perceptual hashes for all images and store them."""
    from lorahub.core.phash import dhash64, phash64  # noqa: PLC0415

    directory = _resolve_under_roots(body.path)
    if not directory.is_dir():
        raise HTTPException(400, "not a directory")

    images = _scan_images(directory, body.recursive)
    store = _store()
    computed = 0
    errors: list[dict[str, str]] = []

    hash_fn = phash64 if body.algo == "phash64" else dhash64

    for img_path in images:
        try:
            h = hash_fn(img_path)
            from lorahub.api.image_studio_store import ImagePhash  # noqa: PLC0415
            store.upsert_phash(ImagePhash(
                image_path=str(img_path), algo=body.algo, hash=h
            ))
            computed += 1
        except Exception as exc:  # noqa: BLE001
            errors.append({"path": str(img_path), "error": str(exc)})

    return {"computed": computed, "total": len(images), "errors": errors}


@router.get("/dedupe/clusters")
def dedupe_clusters(
    path: str,
    kind: str = "phash",
    threshold: int = 10,
) -> dict[str, Any]:
    """Return clusters of near-duplicate images based on stored hashes."""
    from lorahub.core.phash import hamming_distance  # noqa: PLC0415

    directory = _resolve_under_roots(path)
    if not directory.is_dir():
        raise HTTPException(400, "not a directory")

    store = _store()
    algo = "phash64" if kind == "phash" else "dhash64"
    all_hashes = store.list_phashes(algo)

    dir_str = str(directory)
    hashes = [h for h in all_hashes if h.image_path.startswith(dir_str)]

    clusters = _compute_clusters(hashes, threshold)

    result_clusters: list[dict[str, Any]] = []
    for i, cluster in enumerate(clusters):
        members = []
        for ph in cluster:
            members.append({
                "path": ph.image_path,
                "hash": ph.hash,
            })
        if len(members) < 2:
            continue
        suggested_keep = _pick_best_keep(members)
        result_clusters.append({
            "id": f"phash-{i}",
            "kind": "phash",
            "members": members,
            "suggestedKeep": suggested_keep,
        })

    return {"clusters": result_clusters}


def _compute_clusters(
    hashes: list["ImagePhash"], threshold: int
) -> list[list["ImagePhash"]]:
    """Simple greedy clustering by hamming distance."""
    from lorahub.core.phash import hamming_distance  # noqa: PLC0415

    assigned: set[int] = set()
    clusters: list[list["ImagePhash"]] = []

    for i, h1 in enumerate(hashes):
        if i in assigned:
            continue
        cluster = [h1]
        assigned.add(i)
        for j in range(i + 1, len(hashes)):
            if j in assigned:
                continue
            if hamming_distance(h1.hash, hashes[j].hash) <= threshold:
                cluster.append(hashes[j])
                assigned.add(j)
        clusters.append(cluster)

    return clusters


def _pick_best_keep(members: list[dict[str, Any]]) -> str:
    """Pick the best image to keep: largest file size wins."""
    best = members[0]
    best_size = 0
    for m in members:
        try:
            size = Path(m["path"]).stat().st_size
            if size > best_size:
                best_size = size
                best = m
        except OSError:
            pass
    return best["path"]


class BatchDeleteSelectInput(BaseModel):
    paths: list[str]
    forceFavorites: bool = False


@router.post("/dedupe/batch-delete")
def batch_delete(body: BatchDeleteSelectInput) -> dict[str, Any]:
    """Soft-delete a batch of images (move to trash)."""
    deleted: list[str] = []
    errors: list[dict[str, str]] = []
    bytes_freed = 0

    store = _store()
    for p in body.paths:
        try:
            file_path = _resolve_under_roots(p)
            if not file_path.is_file():
                errors.append({"path": p, "error": "file not found"})
                continue
            if not body.forceFavorites:
                ann = store.get_annotation(str(file_path))
                if ann and ann.favorite:
                    errors.append({"path": p, "error": "is a favourite; set forceFavorites=true"})
                    continue
            size = file_path.stat().st_size
            _soft_delete(file_path)
            bytes_freed += size
            deleted.append(p)
            store.delete_annotation(str(file_path))
            store.delete_phashes(str(file_path))
        except Exception as exc:  # noqa: BLE001
            errors.append({"path": p, "error": str(exc)})

    return {
        "deletedCount": len(deleted),
        "deleted": deleted,
        "bytesFreed": bytes_freed,
        "errors": errors,
    }


# --------------------------------------------------------------------------- #
# Smart caption (WD14 + Vision LLM)
# --------------------------------------------------------------------------- #

_SMART_CAPTION_PROMPT = (
    "Based on the following WD14 tags and the image, write an optimized training caption.\n"
    "WD14 tags: {tags}\n"
    "Write a natural language caption suitable for LoRA training. Include all relevant details "
    "about the subject, pose, clothing, background, and composition. Be concise but complete."
)


class SmartCaptionBatchInput(BaseModel):
    path: str
    recursive: bool = False
    taggerModel: str = "SmilingWolf/wd-v1-4-vit-tagger-v2"
    visionTask: str = "tagging.assist"
    mergeStrategy: str = "replace"
    device: str = "auto"
    generalThreshold: float = 0.35
    characterThreshold: float = 0.85


def _smart_caption_single_image(
    img_path: Path,
    tagger: "WD14Tagger",
    ai_store: Any,
    route: Any,
    merge_strategy: str,
    store: ImageStudioStore,
) -> dict[str, Any]:
    """Run WD14 tagging then vision LLM on a single image. Returns result dict."""
    import base64  # noqa: PLC0415
    import mimetypes  # noqa: PLC0415
    from datetime import UTC, datetime  # noqa: PLC0415

    from lorahub.core.ai import client as ai_client  # noqa: PLC0415

    # Step 1: WD14 tagging
    tag_result = tagger.tag_image(img_path)
    tags_str = tag_result.caption(underscores=False, include_character=True)

    # Step 2: Prepare image for vision LLM
    mime = mimetypes.guess_type(str(img_path))[0] or "image/png"
    data = img_path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    data_url = f"data:{mime};base64,{b64}"

    prompt_text = _SMART_CAPTION_PROMPT.format(tags=tags_str)

    messages: list[dict[str, Any]] = []
    if route.system_prompt:
        messages.append({"role": "system", "content": route.system_prompt})
    messages.append({
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": data_url}},
            {"type": "text", "text": prompt_text},
        ],
    })

    # Step 3: Invoke vision LLM
    result = ai_client.invoke(
        ai_store,
        provider_id=route.provider_id,
        model_id=route.model_id,
        messages=messages,
        route=route,
    )

    # Step 4: Save caption
    caption_path = img_path.with_suffix(".txt")
    existing = caption_path.read_text(encoding="utf-8") if caption_path.is_file() else ""

    if merge_strategy == "append":
        new_caption = (existing.strip() + ", " + result.content).strip(", ")
    elif merge_strategy == "prepend":
        new_caption = (result.content + ", " + existing.strip()).strip(", ")
    else:  # replace
        new_caption = result.content

    caption_path.write_text(new_caption, encoding="utf-8")

    # Step 5: Update annotation
    ann = store.get_annotation(str(img_path))
    if ann is None:
        ann = ImageAnnotation(
            image_path=str(img_path),
            sha256=_file_sha256(img_path),
        )
    ann.ai_caption = result.content
    ann.ai_caption_provider = f"{result.provider_name}/{result.model_id}"
    ann.ai_caption_at = datetime.now(UTC).isoformat()
    store.upsert_annotation(ann)

    return {
        "path": str(img_path),
        "wd14Tags": tags_str,
        "caption": new_caption,
    }


@router.post("/ai/smart-caption")
def ai_smart_caption_batch(body: SmartCaptionBatchInput) -> dict[str, Any]:
    """Run WD14 tagging + vision LLM captioning for all images in a directory."""
    from lorahub.api import app as app_mod  # noqa: PLC0415
    from lorahub.core.tagging.wd14 import WD14Tagger  # noqa: PLC0415

    directory = _resolve_under_roots(body.path)
    if not directory.is_dir():
        raise HTTPException(400, "not a directory")

    ai_store = app_mod._ai_store
    if ai_store is None:
        raise HTTPException(503, "AI store not initialised")

    route = ai_store.get_route(body.visionTask)
    if route is None or not (route.provider_id and route.model_id):
        route = ai_store.get_route("global.default")
    if route is None or not (route.provider_id and route.model_id):
        raise HTTPException(409, f"no AI route for task {body.visionTask!r}")

    # Initialize WD14 tagger
    tagger = WD14Tagger(
        model_id=body.taggerModel,
        general_threshold=body.generalThreshold,
        character_threshold=body.characterThreshold,
        device=body.device,
    )
    tagger.load()

    images = _scan_images(directory, body.recursive)
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    store = _store()
    for img_path in images:
        try:
            item = _smart_caption_single_image(
                img_path, tagger, ai_store, route, body.mergeStrategy, store
            )
            results.append(item)
        except Exception as exc:  # noqa: BLE001
            errors.append({"path": str(img_path), "error": str(exc)})

    return {"processed": len(results), "results": results, "errors": errors}


class SmartCaptionSingleInput(BaseModel):
    path: str
    taggerModel: str = "SmilingWolf/wd-v1-4-vit-tagger-v2"
    visionTask: str = "tagging.assist"
    mergeStrategy: str = "replace"
    device: str = "auto"
    generalThreshold: float = 0.35
    characterThreshold: float = 0.85


@router.post("/ai/smart-caption/single")
def ai_smart_caption_single(body: SmartCaptionSingleInput) -> dict[str, Any]:
    """Run WD14 tagging + vision LLM captioning for a single image."""
    from lorahub.api import app as app_mod  # noqa: PLC0415
    from lorahub.core.tagging.wd14 import WD14Tagger  # noqa: PLC0415

    file_path = _resolve_under_roots(body.path)
    if not file_path.is_file():
        raise HTTPException(404, "image not found")

    ai_store = app_mod._ai_store
    if ai_store is None:
        raise HTTPException(503, "AI store not initialised")

    route = ai_store.get_route(body.visionTask)
    if route is None or not (route.provider_id and route.model_id):
        route = ai_store.get_route("global.default")
    if route is None or not (route.provider_id and route.model_id):
        raise HTTPException(409, f"no AI route for task {body.visionTask!r}")

    tagger = WD14Tagger(
        model_id=body.taggerModel,
        general_threshold=body.generalThreshold,
        character_threshold=body.characterThreshold,
        device=body.device,
    )
    tagger.load()

    store = _store()
    try:
        item = _smart_caption_single_image(
            file_path, tagger, ai_store, route, body.mergeStrategy, store
        )
        return {"ok": True, **item}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"smart caption failed: {exc}") from exc


# --------------------------------------------------------------------------- #
# Dataset management
# --------------------------------------------------------------------------- #

_DATASET_META_FILE = "dataset.json"
_ARCHIVE_SUFFIXES = {".zip", ".tar", ".gz", ".tgz", ".7z", ".rar"}
_CAPTION_SUFFIXES = {".txt", ".caption"}


def _datasets_root() -> Path:
    """Return the configured datasets root directory."""
    extra = os.environ.get("LORAHUB_DATASETS_ROOT")
    if extra:
        root = Path(extra.split(os.pathsep)[0].strip())
    else:
        root = Path.cwd() / "datasets"
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _read_dataset_meta(ds_path: Path) -> dict[str, Any]:
    meta_file = ds_path / _DATASET_META_FILE
    if meta_file.is_file():
        return json_stdlib.loads(meta_file.read_text(encoding="utf-8"))
    return {}


def _write_dataset_meta(ds_path: Path, meta: dict[str, Any]) -> None:
    meta_file = ds_path / _DATASET_META_FILE
    meta_file.write_text(
        json_stdlib.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _count_images(ds_path: Path) -> int:
    count = 0
    for p in ds_path.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            count += 1
    return count


def _dataset_cover(ds_path: Path) -> str | None:
    """Return the first image path as cover thumbnail."""
    for p in sorted(ds_path.iterdir(), key=lambda x: x.name.lower()):
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
            return str(p)
    return None


@router.get("/datasets")
def list_datasets() -> dict[str, Any]:
    """List all datasets under the datasets root."""
    root = _datasets_root()
    datasets: list[dict[str, Any]] = []
    for entry in sorted(root.iterdir(), key=lambda x: x.name.lower()):
        if not entry.is_dir():
            continue
        if entry.name.startswith("."):
            continue
        meta = _read_dataset_meta(entry)
        image_count = _count_images(entry)
        cover = _dataset_cover(entry)
        datasets.append({
            "name": entry.name,
            "path": str(entry),
            "imageCount": image_count,
            "coverPath": cover,
            "coverUrl": (
                f"/api/datasets/thumb?path={cover}&size=256" if cover else None
            ),
            "meta": meta,
        })
    return {"root": str(root), "datasets": datasets}


class CreateDatasetInput(BaseModel):
    name: str
    description: str = ""
    targetResolution: str = ""
    triggerWord: str = ""


@router.post("/datasets")
def create_dataset(body: CreateDatasetInput) -> dict[str, Any]:
    """Create a new dataset directory with metadata."""
    root = _datasets_root()
    name = body.name.strip()
    if not name:
        raise HTTPException(400, "dataset name is required")
    if "/" in name or "\\" in name or ".." in name:
        raise HTTPException(400, "invalid dataset name")
    ds_path = root / name
    if ds_path.exists():
        raise HTTPException(409, f"dataset '{name}' already exists")
    ds_path.mkdir(parents=True)
    meta = {
        "name": name,
        "description": body.description,
        "targetResolution": body.targetResolution,
        "triggerWord": body.triggerWord,
    }
    _write_dataset_meta(ds_path, meta)
    return {"ok": True, "path": str(ds_path), "meta": meta}


@router.get("/datasets/{name}/meta")
def get_dataset_meta(name: str) -> dict[str, Any]:
    root = _datasets_root()
    ds_path = root / name
    if not ds_path.is_dir():
        raise HTTPException(404, "dataset not found")
    return _read_dataset_meta(ds_path)


class UpdateDatasetMetaInput(BaseModel):
    description: str | None = None
    targetResolution: str | None = None
    triggerWord: str | None = None


@router.put("/datasets/{name}/meta")
def update_dataset_meta(name: str, body: UpdateDatasetMetaInput) -> dict[str, Any]:
    root = _datasets_root()
    ds_path = root / name
    if not ds_path.is_dir():
        raise HTTPException(404, "dataset not found")
    meta = _read_dataset_meta(ds_path)
    if body.description is not None:
        meta["description"] = body.description
    if body.targetResolution is not None:
        meta["targetResolution"] = body.targetResolution
    if body.triggerWord is not None:
        meta["triggerWord"] = body.triggerWord
    _write_dataset_meta(ds_path, meta)
    return {"ok": True, "meta": meta}


@router.delete("/datasets/{name}")
def delete_dataset(name: str) -> dict[str, Any]:
    """Move dataset to trash (not permanent delete)."""
    root = _datasets_root()
    ds_path = root / name
    if not ds_path.is_dir():
        raise HTTPException(404, "dataset not found")
    from datetime import UTC, datetime  # noqa: PLC0415
    trash = Path("runs") / "_dataset_trash" / datetime.now(UTC).strftime("%Y-%m-%d")
    trash.mkdir(parents=True, exist_ok=True)
    shutil.move(str(ds_path), str(trash / name))
    return {"ok": True}


# --------------------------------------------------------------------------- #
# Upload + extract
# --------------------------------------------------------------------------- #


def _is_archive(filename: str) -> bool:
    lower = filename.lower()
    return any(lower.endswith(s) for s in _ARCHIVE_SUFFIXES)


def _is_image_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in IMAGE_SUFFIXES


def _is_caption_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in _CAPTION_SUFFIXES


def _extract_archive(
    archive_path: Path,
    dest: Path,
    keep_captions: bool,
    on_conflict: str,
) -> tuple[int, list[str]]:
    """Extract images from archive. Returns (count, errors)."""
    import zipfile  # noqa: PLC0415
    import tarfile  # noqa: PLC0415

    extracted = 0
    errors: list[str] = []
    suffix = archive_path.suffix.lower()
    name_lower = archive_path.name.lower()

    if suffix == ".zip":
        try:
            with zipfile.ZipFile(archive_path, "r") as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    fname = Path(info.filename).name
                    if not fname:
                        continue
                    if _is_image_file(fname) or (keep_captions and _is_caption_file(fname)):
                        target = dest / fname
                        target = _resolve_conflict(target, on_conflict)
                        if target is None:
                            continue
                        with zf.open(info) as src, open(target, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        extracted += 1
        except zipfile.BadZipFile as e:
            errors.append(f"bad zip: {e}")
    elif suffix in (".tar", ".gz", ".tgz") or name_lower.endswith(".tar.gz"):
        try:
            mode = "r:gz" if suffix in (".gz", ".tgz") or name_lower.endswith(".tar.gz") else "r"
            with tarfile.open(archive_path, mode) as tf:
                for member in tf.getmembers():
                    if not member.isfile():
                        continue
                    fname = Path(member.name).name
                    if not fname:
                        continue
                    if _is_image_file(fname) or (keep_captions and _is_caption_file(fname)):
                        target = dest / fname
                        target = _resolve_conflict(target, on_conflict)
                        if target is None:
                            continue
                        f = tf.extractfile(member)
                        if f is None:
                            continue
                        with open(target, "wb") as dst:
                            shutil.copyfileobj(f, dst)
                        extracted += 1
        except (tarfile.TarError, OSError) as e:
            errors.append(f"tar error: {e}")
    elif suffix == ".7z":
        try:
            import py7zr  # noqa: PLC0415
            with py7zr.SevenZipFile(archive_path, "r") as sz:
                for fname, bio in sz.read().items():
                    base = Path(fname).name
                    if not base:
                        continue
                    if _is_image_file(base) or (keep_captions and _is_caption_file(base)):
                        target = dest / base
                        target = _resolve_conflict(target, on_conflict)
                        if target is None:
                            continue
                        with open(target, "wb") as dst:
                            dst.write(bio.read())
                        extracted += 1
        except Exception as e:  # noqa: BLE001
            errors.append(f"7z error: {e}")
    else:
        errors.append(f"unsupported archive format: {suffix}")

    return extracted, errors


def _resolve_conflict(target: Path, strategy: str) -> Path | None:
    """Handle file name conflicts. Returns final path or None to skip."""
    if not target.exists():
        return target
    if strategy == "skip":
        return None
    if strategy == "overwrite":
        return target
    # rename: append _1, _2, etc.
    stem = target.stem
    suffix = target.suffix
    parent = target.parent
    i = 1
    while True:
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


@router.post("/datasets/{name}/upload")
async def upload_to_dataset(
    name: str,
    files: list[UploadFile] = File(...),
    keepCaptions: bool = Form(True),
    onConflict: str = Form("rename"),
) -> StreamingResponse:
    """Upload files to a dataset. Supports images and archives.

    Returns SSE stream with progress events.
    """
    root = _datasets_root()
    ds_path = root / name
    if not ds_path.is_dir():
        raise HTTPException(404, "dataset not found")

    import asyncio  # noqa: PLC0415

    async def _generate():
        total_files = len(files)
        total_extracted = 0
        all_errors: list[str] = []

        for idx, upload in enumerate(files):
            filename = upload.filename or "unknown"
            yield _sse_event("progress", {
                "file": filename,
                "index": idx,
                "total": total_files,
                "status": "processing",
            })

            if _is_archive(filename):
                # Save to temp, then extract
                tmp = tempfile.NamedTemporaryFile(
                    delete=False, suffix=Path(filename).suffix
                )
                try:
                    content = await upload.read()
                    tmp.write(content)
                    tmp.close()
                    count, errs = _extract_archive(
                        Path(tmp.name), ds_path, keepCaptions, onConflict
                    )
                    total_extracted += count
                    all_errors.extend(errs)
                    yield _sse_event("extracted", {
                        "file": filename,
                        "count": count,
                        "errors": errs,
                    })
                finally:
                    os.unlink(tmp.name)
            elif _is_image_file(filename) or (keepCaptions and _is_caption_file(filename)):
                target = ds_path / filename
                target = _resolve_conflict(target, onConflict)
                if target is not None:
                    content = await upload.read()
                    target.write_bytes(content)
                    total_extracted += 1
            else:
                all_errors.append(f"skipped non-image: {filename}")

            yield _sse_event("progress", {
                "file": filename,
                "index": idx + 1,
                "total": total_files,
                "status": "done",
            })
            await asyncio.sleep(0)

        yield _sse_event("complete", {
            "totalExtracted": total_extracted,
            "errors": all_errors,
        })

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse_event(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json_stdlib.dumps(data, ensure_ascii=False)}\n\n"


# --------------------------------------------------------------------------- #
# L2 AI Semantic Similarity
# --------------------------------------------------------------------------- #


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SimilarityScanInput(BaseModel):
    path: str
    recursive: bool = False
    mode: str = "embedding"  # "embedding" | "pairwise"
    threshold: float = Field(default=0.92, ge=0.0, le=1.0)
    task: str = "similarity.embedding"




@router.post("/similarity/scan")
def similarity_scan(body: SimilarityScanInput) -> dict[str, Any]:
    """Compute AI embeddings or pairwise VLM verdicts for duplicate detection."""
    from lorahub.api import app as app_mod  # noqa: PLC0415
    from lorahub.api.image_studio_store import ImageEmbedding  # noqa: PLC0415
    from lorahub.core.ai import client as ai_client  # noqa: PLC0415

    directory = _resolve_under_roots(body.path)
    if not directory.is_dir():
        raise HTTPException(400, "not a directory")

    ai_store = app_mod._ai_store
    if ai_store is None:
        raise HTTPException(503, "AI store not initialised")

    route = ai_store.get_route(body.task)
    if route is None or not (route.provider_id and route.model_id):
        route = ai_store.get_route("global.default")
    if route is None or not (route.provider_id and route.model_id):
        raise HTTPException(409, f"no AI route for task {body.task!r}")

    images = _scan_images(directory, body.recursive)
    store = _store()

    if body.mode == "embedding":
        return _similarity_embedding_scan(
            images, store, ai_store, ai_client, route, body.threshold
        )
    elif body.mode == "pairwise":
        return _similarity_pairwise_scan(
            images, store, ai_store, ai_client, route, directory
        )
    else:
        raise HTTPException(400, f"unknown mode: {body.mode!r}")




def _similarity_embedding_scan(
    images: list[Path],
    store: "ImageStudioStore",
    ai_store: Any,
    ai_client: Any,
    route: Any,
    threshold: float,
) -> dict[str, Any]:
    """Generate text embeddings from AI captions and cluster by cosine similarity."""
    from lorahub.api.image_studio_store import ImageEmbedding  # noqa: PLC0415

    computed = 0
    errors: list[dict[str, str]] = []

    for img_path in images:
        try:
            # Get or generate caption text for embedding
            ann = store.get_annotation(str(img_path))
            caption = ann.ai_caption if ann else None
            if not caption:
                # Read caption from .txt sidecar
                caption_path = img_path.with_suffix(".txt")
                if caption_path.is_file():
                    caption = caption_path.read_text(encoding="utf-8").strip()
            if not caption:
                errors.append({"path": str(img_path), "error": "no caption available"})
                continue

            # Request embedding via chat completion (ask model to return vector)
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": (
                    "You are an embedding encoder. Given the text below, "
                    "return ONLY a JSON array of 64 float numbers representing "
                    "a normalized semantic embedding vector. No explanation."
                )},
                {"role": "user", "content": caption},
            ]

            result = ai_client.invoke(
                ai_store,
                provider_id=route.provider_id,
                model_id=route.model_id,
                messages=messages,
                route=route,
            )


            import json as json_mod  # noqa: PLC0415

            # Parse the embedding vector from response
            raw = result.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            vector = json_mod.loads(raw)
            if not isinstance(vector, list):
                raise ValueError("response is not a list")

            emb = ImageEmbedding(
                image_path=str(img_path),
                model_id=route.model_id,
                dim=len(vector),
                vector=[float(v) for v in vector],
            )
            store.upsert_embedding(emb)
            computed += 1
        except Exception as exc:  # noqa: BLE001
            errors.append({"path": str(img_path), "error": str(exc)})

    # Cluster by cosine similarity
    all_embeddings = store.list_embeddings(route.model_id)
    dir_str = str(images[0].parent) if images else ""
    dir_embs = [e for e in all_embeddings if e.image_path.startswith(dir_str)]
    clusters = _cluster_embeddings(dir_embs, threshold)

    return {
        "computed": computed,
        "total": len(images),
        "clusters": len(clusters),
        "errors": errors,
    }




def _cluster_embeddings(
    embeddings: list["ImageEmbedding"], threshold: float
) -> list[list["ImageEmbedding"]]:
    """Greedy clustering by cosine similarity."""
    assigned: set[int] = set()
    clusters: list[list["ImageEmbedding"]] = []

    for i, e1 in enumerate(embeddings):
        if i in assigned:
            continue
        cluster = [e1]
        assigned.add(i)
        for j in range(i + 1, len(embeddings)):
            if j in assigned:
                continue
            sim = _cosine_similarity(e1.vector, embeddings[j].vector)
            if sim >= threshold:
                cluster.append(embeddings[j])
                assigned.add(j)
        clusters.append(cluster)

    return clusters




def _similarity_pairwise_scan(
    images: list[Path],
    store: "ImageStudioStore",
    ai_store: Any,
    ai_client: Any,
    route: Any,
    directory: Path,
) -> dict[str, Any]:
    """Use VLM to compare pairs from existing phash clusters."""
    from lorahub.core.phash import hamming_distance  # noqa: PLC0415

    algo = "phash64"
    all_hashes = store.list_phashes(algo)
    dir_str = str(directory)
    hashes = [h for h in all_hashes if h.image_path.startswith(dir_str)]

    # Get phash clusters (threshold=10 for seed candidates)
    phash_clusters = _compute_clusters(hashes, threshold=10)

    verdicts: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for cluster in phash_clusters:
        if len(cluster) < 2:
            continue
        # Compare first pair in each cluster via VLM
        pair = cluster[:2]
        try:
            import base64  # noqa: PLC0415
            import mimetypes  # noqa: PLC0415

            contents: list[dict[str, Any]] = []
            for ph in pair:
                p = Path(ph.image_path)
                if not p.is_file():
                    continue
                mime = mimetypes.guess_type(str(p))[0] or "image/png"
                data = p.read_bytes()
                b64 = base64.b64encode(data).decode("ascii")
                data_url = f"data:{mime};base64,{b64}"
                contents.append({"type": "image_url", "image_url": {"url": data_url}})


            if len(contents) < 2:
                continue

            contents.append({
                "type": "text",
                "text": (
                    "Are these two images duplicates or near-duplicates? "
                    'Return JSON: {"duplicate": true/false, "confidence": 0.0-1.0, '
                    '"reason": "brief explanation"}'
                ),
            })

            messages: list[dict[str, Any]] = [
                {"role": "user", "content": contents},
            ]

            result = ai_client.invoke(
                ai_store,
                provider_id=route.provider_id,
                model_id=route.model_id,
                messages=messages,
                route=route,
            )

            import json as json_mod  # noqa: PLC0415

            raw = result.content.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            parsed = json_mod.loads(raw)
            verdicts.append({
                "pair": [pair[0].image_path, pair[1].image_path],
                "duplicate": parsed.get("duplicate", False),
                "confidence": parsed.get("confidence", 0.0),
                "reason": parsed.get("reason", ""),
            })
        except Exception as exc:  # noqa: BLE001
            errors.append({
                "path": pair[0].image_path if pair else "unknown",
                "error": str(exc),
            })

    return {
        "mode": "pairwise",
        "verdicts": verdicts,
        "errors": errors,
    }




@router.get("/similarity/clusters")
def similarity_clusters(
    path: str,
    kind: str = "ai",
    threshold: float = 0.92,
) -> dict[str, Any]:
    """Return clusters of semantically similar images based on stored embeddings."""
    directory = _resolve_under_roots(path)
    if not directory.is_dir():
        raise HTTPException(400, "not a directory")

    store = _store()

    if kind == "ai":
        # Use the first available model_id from embeddings in this dir
        from lorahub.api import app as app_mod  # noqa: PLC0415

        ai_store = app_mod._ai_store
        model_id = "unknown"
        if ai_store:
            route = ai_store.get_route("similarity.embedding")
            if route is None or not (route.provider_id and route.model_id):
                route = ai_store.get_route("global.default")
            if route and route.model_id:
                model_id = route.model_id

        all_embeddings = store.list_embeddings(model_id)
        dir_str = str(directory)
        dir_embs = [e for e in all_embeddings if e.image_path.startswith(dir_str)]
        clusters = _cluster_embeddings(dir_embs, threshold)

        result_clusters: list[dict[str, Any]] = []
        for i, cluster in enumerate(clusters):
            if len(cluster) < 2:
                continue
            members = []
            for emb in cluster:
                members.append({"path": emb.image_path, "model": emb.model_id})

            # Compute average pairwise similarity for confidence
            sims: list[float] = []
            for a_idx in range(len(cluster)):
                for b_idx in range(a_idx + 1, len(cluster)):
                    sims.append(_cosine_similarity(
                        cluster[a_idx].vector, cluster[b_idx].vector
                    ))
            avg_sim = sum(sims) / len(sims) if sims else 0.0

            suggested_keep = _pick_best_keep(
                [{"path": e.image_path} for e in cluster]
            )
            result_clusters.append({
                "id": f"ai-{i}",
                "kind": "ai",
                "members": members,
                "confidence": round(avg_sim, 4),
                "suggestedKeep": suggested_keep,
            })

        return {"clusters": result_clusters}

    # Fallback to phash clusters
    return dedupe_clusters(path=path, kind="phash", threshold=int(threshold * 100))




class SimilarityBatchDeleteInput(BaseModel):
    paths: list[str]
    forceFavorites: bool = False


@router.post("/similarity/batch-delete")
def similarity_batch_delete(body: SimilarityBatchDeleteInput) -> dict[str, Any]:
    """Soft-delete a batch of images from AI similarity clusters."""
    deleted: list[str] = []
    errors: list[dict[str, str]] = []
    bytes_freed = 0

    store = _store()
    for p in body.paths:
        try:
            file_path = _resolve_under_roots(p)
            if not file_path.is_file():
                errors.append({"path": p, "error": "file not found"})
                continue
            if not body.forceFavorites:
                ann = store.get_annotation(str(file_path))
                if ann and ann.favorite:
                    errors.append({
                        "path": p,
                        "error": "is a favourite; set forceFavorites=true",
                    })
                    continue
            size = file_path.stat().st_size
            _soft_delete(file_path)
            bytes_freed += size
            deleted.append(p)
            store.delete_annotation(str(file_path))
            store.delete_phashes(str(file_path))
            store.delete_embeddings(str(file_path))
        except Exception as exc:  # noqa: BLE001
            errors.append({"path": p, "error": str(exc)})

    return {
        "deletedCount": len(deleted),
        "deleted": deleted,
        "bytesFreed": bytes_freed,
        "errors": errors,
    }




# --------------------------------------------------------------------------- #
# Tagging from Image Studio
# --------------------------------------------------------------------------- #


class ISTaggingStartInput(BaseModel):
    path: str
    tagger: str = "wd14"  # "wd14" | "joytag"
    model_id: str | None = None
    general: float = Field(default=0.35, ge=0.0, le=1.0)
    character: float = Field(default=0.85, ge=0.0, le=1.0)
    joytag_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    device: str = "auto"
    overwrite: bool = False
    recursive: bool = False
    include_character: bool = True
    underscores: bool = False




@dataclass(slots=True)
class _ISTaggingSession:
    session_id: str
    path: str
    tagger: str
    model_id: str
    device: str
    general: float
    character: float
    joytag_threshold: float
    overwrite: bool
    recursive: bool
    include_character: bool
    underscores: bool
    status: Literal["running", "succeeded", "failed"] = "running"
    percent: float = 0.0
    events: list[dict[str, Any]] = field(default_factory=list)
    written: int = 0
    total: int | None = None
    active_provider: str = ""
    error: str | None = None
    started_at: float = field(default_factory=_time.time)
    finished_at: float | None = None
    _stop_flag: bool = field(default=False, repr=False)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def push(
        self, message: str, *, percent: float | None = None, image: str | None = None
    ) -> None:
        with self.lock:
            if percent is not None:
                self.percent = max(self.percent, min(100.0, float(percent)))
            self.events.append(
                {"ts": _time.time(), "message": message, "percent": self.percent, "image": image}
            )
            self.events = self.events[-200:]


    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "session_id": self.session_id,
                "path": self.path,
                "tagger": self.tagger,
                "model_id": self.model_id,
                "device": self.device,
                "general": self.general,
                "character": self.character,
                "joytag_threshold": self.joytag_threshold,
                "overwrite": self.overwrite,
                "recursive": self.recursive,
                "include_character": self.include_character,
                "underscores": self.underscores,
                "status": self.status,
                "percent": self.percent,
                "events": list(self.events),
                "written": self.written,
                "total": self.total,
                "active_provider": self.active_provider,
                "error": self.error,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
            }

    @property
    def should_stop(self) -> bool:
        return self._stop_flag

    def request_stop(self) -> None:
        self._stop_flag = True


_is_tagging_sessions: dict[str, _ISTaggingSession] = {}
_is_tagging_lock = threading.Lock()




def _build_is_tagger(req: ISTaggingStartInput) -> "BaseTagger":
    """Build a tagger instance from the request params."""
    from lorahub.core.tagging.joytag import JoyTagger  # noqa: PLC0415
    from lorahub.core.tagging.wd14 import DEFAULT_MODEL, WD14Tagger  # noqa: PLC0415

    if req.tagger == "joytag":
        return JoyTagger(
            predict_threshold=req.joytag_threshold,
            device=req.device,
        )
    return WD14Tagger(
        model_id=req.model_id or DEFAULT_MODEL,
        general_threshold=req.general,
        character_threshold=req.character,
        device=req.device,
    )


@router.post("/tagging/start", status_code=202)
def is_tagging_start(req: ISTaggingStartInput) -> dict[str, Any]:
    """Start a background tagging session from Image Studio."""
    from lorahub.core.tagging.wd14 import DEFAULT_MODEL  # noqa: PLC0415

    target = _resolve_under_roots(req.path)
    if not target.is_dir():
        raise HTTPException(400, f"not a directory: {target}")

    session = _ISTaggingSession(
        session_id=_uuid.uuid4().hex,
        path=str(target),
        tagger=req.tagger,
        model_id=req.model_id or DEFAULT_MODEL,
        device=req.device,
        general=req.general,
        character=req.character,
        joytag_threshold=req.joytag_threshold,
        overwrite=req.overwrite,
        recursive=req.recursive,
        include_character=req.include_character,
        underscores=req.underscores,
    )
    session.push("tagging queued", percent=0)
    with _is_tagging_lock:
        _is_tagging_sessions[session.session_id] = session


    def run() -> None:
        try:
            from lorahub.core.tagging.wd14 import (  # noqa: PLC0415
                CudaUnavailableError,
                _iter_images,
            )

            tagger = _build_is_tagger(req)
            session.push(f"loading {req.tagger}")
            tagger.load()
            with session.lock:
                session.active_provider = tagger.active_provider
            session.push(f"running on {tagger.active_provider}", percent=2)

            all_images = list(_iter_images(target, recursive=req.recursive))
            with session.lock:
                session.total = len(all_images)
            if not all_images:
                session.push("no images found", percent=100)
                with session.lock:
                    session.status = "succeeded"
                    session.percent = 100
                    session.finished_at = _time.time()
                return

            total = len(all_images)

            def on_progress(image: Path, _result: object) -> None:
                if session.should_stop:
                    raise InterruptedError("stopped by user")
                with session.lock:
                    session.written += 1
                    pct = min(100.0, 2 + 98 * session.written / max(total, 1))
                session.push(f"tagged {image.name}", percent=pct, image=str(image))

            tagger.tag_directory(
                target,
                recursive=req.recursive,
                write_caption=True,
                skip_existing=not req.overwrite,
                underscores=req.underscores,
                include_character=req.include_character,
                on_progress=on_progress,
            )
            with session.lock:
                session.status = "succeeded"
                session.percent = 100
                session.finished_at = _time.time()
            session.push(f"done - wrote {session.written} caption(s)", percent=100)
        except CudaUnavailableError as exc:
            with session.lock:
                session.status = "failed"
                session.error = str(exc)
                session.finished_at = _time.time()
            session.push(f"cuda unavailable: {exc}")
        except InterruptedError:
            with session.lock:
                session.status = "failed"
                session.error = "stopped by user"
                session.finished_at = _time.time()
            session.push("stopped by user")
        except Exception as exc:  # noqa: BLE001
            with session.lock:
                session.status = "failed"
                session.error = str(exc)
                session.finished_at = _time.time()
            session.push(f"tagging failed: {exc}")

    threading.Thread(
        target=run,
        name=f"is-tag-{req.tagger}-{session.session_id[:8]}",
        daemon=True,
    ).start()
    return {"session_id": session.session_id}




@router.get("/tagging/{session_id}")
def is_tagging_status(session_id: str) -> dict[str, Any]:
    """Return the current snapshot of a tagging session."""
    with _is_tagging_lock:
        session = _is_tagging_sessions.get(session_id)
    if session is None:
        raise HTTPException(404, "tagging session not found")
    return session.snapshot()


@router.post("/tagging/stop/{session_id}")
def is_tagging_stop(session_id: str) -> dict[str, Any]:
    """Request a running tagging session to stop."""
    with _is_tagging_lock:
        session = _is_tagging_sessions.get(session_id)
    if session is None:
        raise HTTPException(404, "tagging session not found")
    session.request_stop()
    return {"session_id": session_id, "status": "stop_requested"}
