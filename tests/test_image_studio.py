"""Tests for ImageStudioStore + router endpoints."""

from __future__ import annotations

import struct
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lorahub.api import app as app_module
from lorahub.api import scheduler as sched_module
from lorahub.api import state as state_module
from lorahub.api.image_studio_store import (
    ImageAnnotation,
    ImageEmbedding,
    ImagePhash,
    ImageStudioStore,
    PendingOp,
)


# --------------------------------------------------------------------------- #
# Store unit tests
# --------------------------------------------------------------------------- #


def test_annotation_round_trip(tmp_path: Path) -> None:
    s = ImageStudioStore(tmp_path / "is.sqlite")
    ann = ImageAnnotation(
        image_path="/data/img001.png",
        sha256="abc123",
        width=1024,
        height=768,
        bytes=500_000,
        ai_quality_label="good",
        ai_trigger_words=["1girl", "blue_hair"],
        favorite=True,
    )
    s.upsert_annotation(ann)
    fetched = s.get_annotation("/data/img001.png")
    assert fetched is not None
    assert fetched.sha256 == "abc123"
    assert fetched.width == 1024
    assert fetched.ai_trigger_words == ["1girl", "blue_hair"]
    assert fetched.favorite is True
    assert fetched.updated_at != ""


def test_annotation_upsert_overwrites(tmp_path: Path) -> None:
    s = ImageStudioStore(tmp_path / "is.sqlite")
    s.upsert_annotation(ImageAnnotation(
        image_path="/x.png", sha256="aaa", user_notes="first"
    ))
    s.upsert_annotation(ImageAnnotation(
        image_path="/x.png", sha256="aaa", user_notes="second"
    ))
    fetched = s.get_annotation("/x.png")
    assert fetched is not None
    assert fetched.user_notes == "second"


def test_annotation_list_filters(tmp_path: Path) -> None:
    s = ImageStudioStore(tmp_path / "is.sqlite")
    s.upsert_annotation(ImageAnnotation(
        image_path="/a.png", sha256="a", ai_quality_label="good", favorite=True
    ))
    s.upsert_annotation(ImageAnnotation(
        image_path="/b.png", sha256="b", ai_quality_label="bad"
    ))
    s.upsert_annotation(ImageAnnotation(
        image_path="/c.png", sha256="c", ai_quality_label="good", soft_deleted=True
    ))
    assert len(s.list_annotations(quality_label="good")) == 2
    assert len(s.list_annotations(favorite=True)) == 1
    assert len(s.list_annotations(soft_deleted=True)) == 1


def test_annotation_delete(tmp_path: Path) -> None:
    s = ImageStudioStore(tmp_path / "is.sqlite")
    s.upsert_annotation(ImageAnnotation(image_path="/x.png", sha256="x"))
    assert s.delete_annotation("/x.png") is True
    assert s.get_annotation("/x.png") is None
    assert s.delete_annotation("/x.png") is False


def test_phash_round_trip(tmp_path: Path) -> None:
    s = ImageStudioStore(tmp_path / "is.sqlite")
    s.upsert_phash(ImagePhash("/img.png", "phash64", "abcdef01"))
    s.upsert_phash(ImagePhash("/img.png", "dhash64", "12345678"))
    hashes = s.get_phashes("/img.png")
    assert len(hashes) == 2
    algos = {h.algo for h in hashes}
    assert algos == {"phash64", "dhash64"}


def test_phash_upsert_overwrites(tmp_path: Path) -> None:
    s = ImageStudioStore(tmp_path / "is.sqlite")
    s.upsert_phash(ImagePhash("/img.png", "phash64", "old"))
    s.upsert_phash(ImagePhash("/img.png", "phash64", "new"))
    hashes = s.get_phashes("/img.png")
    assert len(hashes) == 1
    assert hashes[0].hash == "new"


def test_phash_list_by_algo(tmp_path: Path) -> None:
    s = ImageStudioStore(tmp_path / "is.sqlite")
    s.upsert_phash(ImagePhash("/a.png", "phash64", "aaa"))
    s.upsert_phash(ImagePhash("/b.png", "phash64", "bbb"))
    s.upsert_phash(ImagePhash("/c.png", "dhash64", "ccc"))
    assert len(s.list_phashes("phash64")) == 2
    assert len(s.list_phashes("dhash64")) == 1


def test_pending_ops_lifecycle(tmp_path: Path) -> None:
    s = ImageStudioStore(tmp_path / "is.sqlite")
    op = s.add_pending_op(PendingOp(
        id="", image_path="/img.png", op="rotate", payload={"degrees": 90}
    ))
    assert op.id != ""
    assert op.created_at != ""
    ops = s.list_pending_ops("/img.png")
    assert len(ops) == 1
    assert ops[0].payload == {"degrees": 90}
    assert s.delete_pending_op(op.id) is True
    assert s.list_pending_ops("/img.png") == []


def test_pending_ops_clear(tmp_path: Path) -> None:
    s = ImageStudioStore(tmp_path / "is.sqlite")
    s.add_pending_op(PendingOp(id="", image_path="/a.png", op="rotate", payload={}))
    s.add_pending_op(PendingOp(id="", image_path="/a.png", op="flip", payload={}))
    s.add_pending_op(PendingOp(id="", image_path="/b.png", op="rotate", payload={}))
    assert s.clear_pending_ops("/a.png") == 2
    assert s.list_pending_ops("/a.png") == []
    assert len(s.list_pending_ops("/b.png")) == 1


def test_embedding_round_trip(tmp_path: Path) -> None:
    s = ImageStudioStore(tmp_path / "is.sqlite")
    vec = [0.1, 0.2, 0.3, 0.4]
    emb = ImageEmbedding(
        image_path="/img.png", model_id="clip-v1", dim=4, vector=vec
    )
    s.upsert_embedding(emb)
    fetched = s.get_embedding("/img.png", "clip-v1")
    assert fetched is not None
    assert fetched.dim == 4
    assert len(fetched.vector) == 4
    assert abs(fetched.vector[0] - 0.1) < 1e-6
    assert abs(fetched.vector[3] - 0.4) < 1e-6


def test_embedding_list_by_model(tmp_path: Path) -> None:
    s = ImageStudioStore(tmp_path / "is.sqlite")
    s.upsert_embedding(ImageEmbedding("/a.png", "m1", 2, [1.0, 2.0]))
    s.upsert_embedding(ImageEmbedding("/b.png", "m1", 2, [3.0, 4.0]))
    s.upsert_embedding(ImageEmbedding("/c.png", "m2", 2, [5.0, 6.0]))
    assert len(s.list_embeddings("m1")) == 2
    assert len(s.list_embeddings("m2")) == 1


def test_embedding_delete(tmp_path: Path) -> None:
    s = ImageStudioStore(tmp_path / "is.sqlite")
    s.upsert_embedding(ImageEmbedding("/x.png", "m1", 2, [1.0, 2.0]))
    s.delete_embeddings("/x.png")
    assert s.get_embedding("/x.png", "m1") is None


# --------------------------------------------------------------------------- #
# Router HTTP tests
# --------------------------------------------------------------------------- #


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.chdir(tmp_path)
    registry = state_module.JobRegistry()
    monkeypatch.setattr(state_module, "registry", registry)
    fresh_sched = sched_module.JobScheduler(concurrency=1)
    monkeypatch.setattr(sched_module, "scheduler", fresh_sched)
    store = ImageStudioStore(tmp_path / "is.sqlite")
    monkeypatch.setattr(app_module, "_image_studio_store", store)
    with TestClient(app_module.app) as c:
        yield c


@pytest.fixture
def sample_dir(tmp_path: Path) -> Path:
    """Create a sample dataset directory with images."""
    d = tmp_path / "dataset"
    d.mkdir()
    from PIL import Image  # noqa: PLC0415
    for name in ("a.png", "b.png", "c.png"):
        img = Image.new("RGB", (64, 64), color="red")
        img.save(d / name)
    (d / "a.txt").write_text("caption for a", encoding="utf-8")
    return d


def test_list_images(client: TestClient, sample_dir: Path) -> None:
    r = client.get("/api/image-studio/list", params={"path": str(sample_dir)})
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 3
    assert len(body["items"]) == 3
    names = {item["name"] for item in body["items"]}
    assert names == {"a.png", "b.png", "c.png"}


def test_list_pagination(client: TestClient, sample_dir: Path) -> None:
    r = client.get("/api/image-studio/list", params={
        "path": str(sample_dir), "limit": 2, "page": 1
    })
    assert r.status_code == 200
    assert len(r.json()["items"]) == 2
    r2 = client.get("/api/image-studio/list", params={
        "path": str(sample_dir), "limit": 2, "page": 2
    })
    assert len(r2.json()["items"]) == 1


def test_get_image(client: TestClient, sample_dir: Path) -> None:
    img_path = str(sample_dir / "a.png")
    r = client.get("/api/image-studio/image", params={"path": img_path})
    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "a.png"
    assert body["captionExists"] is True
    assert body["caption"] == "caption for a"
    assert body["phash"] == {}
    assert body["pendingOps"] == []


def test_annotation_crud_via_api(client: TestClient, sample_dir: Path) -> None:
    img_path = str(sample_dir / "a.png")
    r = client.put("/api/image-studio/annotations", json={
        "path": img_path, "userQualityLabel": "good", "favorite": True
    })
    assert r.status_code == 200
    assert r.json()["annotation"]["favorite"] is True
    assert r.json()["annotation"]["userQualityLabel"] == "good"

    r2 = client.delete("/api/image-studio/annotations", params={"path": img_path})
    assert r2.status_code == 200


def test_pending_ops_via_api(client: TestClient, sample_dir: Path) -> None:
    img_path = str(sample_dir / "b.png")
    r = client.post("/api/image-studio/ops", json={
        "path": img_path, "op": "rotate", "payload": {"degrees": 90}
    })
    assert r.status_code == 200
    op_id = r.json()["id"]
    assert op_id

    r2 = client.get("/api/image-studio/ops", params={"path": img_path})
    assert len(r2.json()["ops"]) == 1

    r3 = client.delete(f"/api/image-studio/ops/{op_id}")
    assert r3.status_code == 200


def test_apply_ops_rotate(client: TestClient, sample_dir: Path) -> None:
    img_path = str(sample_dir / "b.png")
    client.post("/api/image-studio/ops", json={
        "path": img_path, "op": "rotate", "payload": {"degrees": 90}
    })
    r = client.post("/api/image-studio/ops/apply", json={"path": img_path})
    assert r.status_code == 200
    assert len(r.json()["applied"]) == 1
    assert r.json()["errors"] == []


# --------------------------------------------------------------------------- #
# Phash unit tests
# --------------------------------------------------------------------------- #


def test_phash64_deterministic(tmp_path: Path) -> None:
    from lorahub.core.phash import phash64

    from PIL import Image
    img = Image.new("RGB", (64, 64), color="blue")
    p = tmp_path / "blue.png"
    img.save(p)
    h1 = phash64(p)
    h2 = phash64(p)
    assert h1 == h2
    assert len(h1) == 16  # 64 bits = 16 hex chars


def test_dhash64_deterministic(tmp_path: Path) -> None:
    from lorahub.core.phash import dhash64

    from PIL import Image
    img = Image.new("RGB", (64, 64), color="green")
    p = tmp_path / "green.png"
    img.save(p)
    h1 = dhash64(p)
    h2 = dhash64(p)
    assert h1 == h2
    assert len(h1) == 16


def test_hamming_distance() -> None:
    from lorahub.core.phash import hamming_distance
    assert hamming_distance("0000000000000000", "0000000000000000") == 0
    assert hamming_distance("0000000000000000", "0000000000000001") == 1
    assert hamming_distance("ffffffffffffffff", "0000000000000000") == 64


def test_similar_images_low_distance(tmp_path: Path) -> None:
    from lorahub.core.phash import hamming_distance, phash64

    from PIL import Image
    img1 = Image.new("RGB", (128, 128), color=(100, 150, 200))
    img2 = img1.copy()
    # Slightly modify one pixel
    img2.putpixel((0, 0), (101, 150, 200))
    p1 = tmp_path / "img1.png"
    p2 = tmp_path / "img2.png"
    img1.save(p1)
    img2.save(p2)
    h1 = phash64(p1)
    h2 = phash64(p2)
    assert hamming_distance(h1, h2) <= 5


# --------------------------------------------------------------------------- #
# Dedupe router tests
# --------------------------------------------------------------------------- #


def test_dedupe_scan_and_clusters(client: TestClient, sample_dir: Path) -> None:
    # Create two identical images (should cluster together)
    from PIL import Image
    img = Image.new("RGB", (64, 64), color="red")
    img.save(sample_dir / "dup1.png")
    img.save(sample_dir / "dup2.png")

    r = client.post("/api/image-studio/dedupe/scan", json={
        "path": str(sample_dir), "algo": "phash64", "threshold": 10
    })
    assert r.status_code == 200
    assert r.json()["computed"] >= 2

    r2 = client.get("/api/image-studio/dedupe/clusters", params={
        "path": str(sample_dir), "kind": "phash", "threshold": 10
    })
    assert r2.status_code == 200
    clusters = r2.json()["clusters"]
    # At least one cluster with the two identical images
    assert len(clusters) >= 1
    dup_cluster = next(
        (c for c in clusters if any("dup1" in m["path"] for m in c["members"])),
        None,
    )
    assert dup_cluster is not None
    assert len(dup_cluster["members"]) >= 2


def test_batch_delete(client: TestClient, sample_dir: Path) -> None:
    img_path = str(sample_dir / "c.png")
    r = client.post("/api/image-studio/dedupe/batch-delete", json={
        "paths": [img_path]
    })
    assert r.status_code == 200
    assert r.json()["deletedCount"] == 1
    assert not (sample_dir / "c.png").exists()


def test_batch_delete_blocks_favorites(client: TestClient, sample_dir: Path) -> None:
    img_path = str(sample_dir / "a.png")
    # Mark as favorite
    client.put("/api/image-studio/annotations", json={
        "path": img_path, "favorite": True
    })
    r = client.post("/api/image-studio/dedupe/batch-delete", json={
        "paths": [img_path]
    })
    assert r.status_code == 200
    assert r.json()["deletedCount"] == 0
    assert len(r.json()["errors"]) == 1
    assert "favourite" in r.json()["errors"][0]["error"]
