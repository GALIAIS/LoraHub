"""Image Studio L2 AI semantic similarity endpoints."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from lorahub.api.dataset_files import _resolve_under_roots, encode_image_data_url

from ._shared import (
    _scan_images,
    _soft_delete,
    _store,
    _stored_path_is_within,
    _writable_dataset_file,
)
from .dedupe import _compute_clusters, _pick_best_keep, dedupe_clusters

if TYPE_CHECKING:
    from lorahub.api.image_studio_store import ImageEmbedding, ImageStudioStore

router = APIRouter(prefix="/api/image-studio", tags=["image-studio"])


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = math.fsum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(math.fsum(x * x for x in a))
    norm_b = math.sqrt(math.fsum(x * x for x in b))
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
            images, store, ai_store, ai_client, route, directory, body.threshold
        )
    if body.mode == "pairwise":
        return _similarity_pairwise_scan(
            images, store, ai_store, ai_client, route, directory
        )
    raise HTTPException(400, f"unknown mode: {body.mode!r}")


def _similarity_embedding_scan(
    images: list[Path],
    store: ImageStudioStore,
    ai_store: Any,
    ai_client: Any,
    route: Any,
    directory: Path,
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
    dir_embs = [
        e
        for e in all_embeddings
        if _stored_path_is_within(e.image_path, directory)
    ]
    clusters = _cluster_embeddings(dir_embs, threshold)

    return {
        "computed": computed,
        "total": len(images),
        "clusters": len(clusters),
        "errors": errors,
    }


def _cluster_embeddings(
    embeddings: list[ImageEmbedding], threshold: float
) -> list[list[ImageEmbedding]]:
    """Greedy clustering by cosine similarity."""
    assigned: set[int] = set()
    clusters: list[list[ImageEmbedding]] = []

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
    store: ImageStudioStore,
    ai_store: Any,
    ai_client: Any,
    route: Any,
    directory: Path,
) -> dict[str, Any]:
    """Use VLM to compare pairs from existing phash clusters."""
    algo = "phash64"
    all_hashes = store.list_phashes(algo)
    hashes = [
        h for h in all_hashes if _stored_path_is_within(h.image_path, directory)
    ]

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
            contents: list[dict[str, Any]] = []
            for ph in pair:
                p = Path(ph.image_path)
                if not p.is_file():
                    continue
                data_url = encode_image_data_url(p)
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
        dir_embs = [
            e
            for e in all_embeddings
            if _stored_path_is_within(e.image_path, directory)
        ]
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
            file_path = _writable_dataset_file(p)
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
