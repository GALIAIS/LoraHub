"""Image Studio dedupe endpoints (perceptual-hash clustering + batch delete)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from lorahub.api.dataset_files import _resolve_under_roots

from ._shared import (
    _scan_images,
    _soft_delete,
    _store,
    _stored_path_is_within,
    _writable_dataset_file,
)

if TYPE_CHECKING:
    from lorahub.api.image_studio_store import ImagePhash

router = APIRouter(prefix="/api/image-studio", tags=["image-studio"])


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
    directory = _resolve_under_roots(path)
    if not directory.is_dir():
        raise HTTPException(400, "not a directory")

    store = _store()
    algo = "phash64" if kind == "phash" else "dhash64"
    all_hashes = store.list_phashes(algo)

    hashes = [
        h for h in all_hashes if _stored_path_is_within(h.image_path, directory)
    ]

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
    hashes: list[ImagePhash], threshold: int
) -> list[list[ImagePhash]]:
    """Simple greedy clustering by hamming distance."""
    from lorahub.core.phash import hamming_distance  # noqa: PLC0415

    assigned: set[int] = set()
    clusters: list[list[ImagePhash]] = []

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
            file_path = _writable_dataset_file(p)
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
