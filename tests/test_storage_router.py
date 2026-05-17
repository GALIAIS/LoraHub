"""Smoke tests for the storage maintenance endpoints."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lorahub.api import app as app_module
from lorahub.api import scheduler as sched_module
from lorahub.api import state as state_module


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Run with cwd pinned to tmp_path so storage endpoints look at a clean tree."""
    monkeypatch.chdir(tmp_path)
    registry = state_module.JobRegistry()
    monkeypatch.setattr(state_module, "registry", registry)
    fresh_sched = sched_module.JobScheduler(concurrency=1)
    monkeypatch.setattr(sched_module, "scheduler", fresh_sched)
    with TestClient(app_module.app) as c:
        yield c


def test_storage_usage_returns_filesystem_and_dirs(client: TestClient) -> None:
    r = client.get("/api/storage/usage")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["filesystem"]["total_bytes"] > 0
    assert "runs" in body["directories"]
    assert "runs_archive" in body["directories"]
    assert "models" in body["directories"]


def test_storage_usage_counts_bytes_under_runs(
    client: TestClient, tmp_path: Path
) -> None:
    runs = tmp_path / "runs"
    runs.mkdir(exist_ok=True)
    (runs / "foo.txt").write_bytes(b"x" * 1024)
    r = client.get("/api/storage/usage")
    body = r.json()
    runs_dir = body["directories"]["runs"]
    assert runs_dir["exists"] is True
    assert runs_dir["bytes"] >= 1024
    assert runs_dir["files"] >= 1


def test_archive_list_empty_when_dir_missing(client: TestClient) -> None:
    r = client.get("/api/storage/archive")
    assert r.status_code == 200
    assert r.json()["entries"] == []


def test_archive_list_and_delete_entry(client: TestClient, tmp_path: Path) -> None:
    archive = tmp_path / "runs" / "_archive"
    entry = archive / "ws-old"
    entry.mkdir(parents=True)
    (entry / "events.jsonl").write_bytes(b"y" * 256)

    r = client.get("/api/storage/archive")
    assert r.status_code == 200
    entries = r.json()["entries"]
    assert any(e["name"] == "ws-old" for e in entries)

    r = client.delete("/api/storage/archive/ws-old")
    assert r.status_code == 200
    assert r.json()["bytes_freed"] >= 256
    assert not entry.exists()


def test_archive_delete_rejects_path_traversal(
    client: TestClient, tmp_path: Path
) -> None:
    (tmp_path / "outside").mkdir()
    # The router rejects names containing ".." even when URL-decoded.
    r = client.delete("/api/storage/archive/..something")
    # 400 (validator) or 404 (resolve miss) are both acceptable; what we
    # really care about is that no entry outside _archive can be deleted.
    assert r.status_code in (400, 404, 405)
    assert (tmp_path / "outside").exists()


def test_archive_clear_all(client: TestClient, tmp_path: Path) -> None:
    archive = tmp_path / "runs" / "_archive"
    for name in ["a", "b", "c"]:
        d = archive / name
        d.mkdir(parents=True)
        (d / "f").write_bytes(b"z" * 64)

    r = client.delete("/api/storage/archive")
    assert r.status_code == 200
    body = r.json()
    assert sorted(body["deleted"]) == ["a", "b", "c"]
    assert body["bytes_freed"] >= 64 * 3
    assert not (archive / "a").exists()


def test_hf_cache_clear_returns_404_when_missing(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Force the HF lookup helpers to a non-existent path.
    monkeypatch.setenv("HF_HOME", str(tmp_path / "no-such-hf"))
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(tmp_path / "no-such-hub"))
    # The fallback default points at the user home; on most CI this will
    # also miss. If it doesn't, that's a genuine cache and the test would
    # delete it — guard by returning early.
    from lorahub.api.routers.storage import _hf_cache_root  # noqa: PLC0415

    if _hf_cache_root() is not None:
        pytest.skip("real HF cache present on host; refusing to test deletion")

    r = client.delete("/api/storage/hf-cache")
    assert r.status_code == 404


def test_hf_cache_clear_succeeds_when_path_exists(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_cache = tmp_path / "hub-cache"
    fake_cache.mkdir()
    (fake_cache / "blob.bin").write_bytes(b"q" * 4096)
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(fake_cache))

    r = client.delete("/api/storage/hf-cache")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["bytes_freed"] >= 4096
    assert not fake_cache.exists()
