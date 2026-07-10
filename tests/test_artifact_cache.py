from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from fastapi import HTTPException

from lorahub.api import app as _app  # noqa: F401
from lorahub.api.jobs_helpers.metrics import (
    _list_workspace_files,
    _resolve_workspace_file,
)
from lorahub.api.routers import artifacts
from lorahub.api.routers.artifacts import _prune_artifact_archive_cache


def test_artifact_cache_uses_resolved_runs_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runs = tmp_path / "state" / "runs"
    monkeypatch.setattr(artifacts.api_paths, "runs_dir", lambda: runs)

    root = artifacts._zip_cache_root()

    assert root == (runs / "_download_cache" / "artifacts").resolve()
    assert root.is_dir()


def test_artifact_cache_refuses_linked_cache_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runs = tmp_path / "runs"
    outside = tmp_path / "outside"
    runs.mkdir()
    outside.mkdir()
    cache_parent = runs / "_download_cache"
    try:
        cache_parent.symlink_to(outside, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    monkeypatch.setattr(artifacts.api_paths, "runs_dir", lambda: runs)

    with pytest.raises(RuntimeError, match="cannot be a link"):
        artifacts._zip_cache_root()


def test_artifact_cache_prunes_oldest_but_keeps_active_file(tmp_path: Path) -> None:
    old = tmp_path / "old.zip"
    middle = tmp_path / "middle.zip"
    active = tmp_path / "active.zip"
    for index, path in enumerate((old, middle, active), start=1):
        path.write_bytes(b"x" * 10)
        os.utime(path, (index, index))

    _prune_artifact_archive_cache(
        tmp_path,
        keep={active},
        max_bytes=15,
        max_files=2,
        max_age_s=int(time.time()) + 100,
    )

    assert active.exists()
    assert not old.exists()
    assert not middle.exists()


def test_artifact_cache_removes_expired_archive(tmp_path: Path) -> None:
    expired = tmp_path / "expired.tar.gz"
    expired.write_bytes(b"archive")
    os.utime(expired, (1, 1))

    _prune_artifact_archive_cache(
        tmp_path,
        max_bytes=1024,
        max_files=10,
        max_age_s=1,
    )

    assert not expired.exists()


def test_artifact_cache_never_prunes_active_download(tmp_path: Path) -> None:
    active = tmp_path / "active.zip"
    other = tmp_path / "other.zip"
    active.write_bytes(b"active")
    other.write_bytes(b"other")
    os.utime(active, (1, 1))
    os.utime(other, (2, 2))

    artifacts._acquire_archive(active)
    try:
        _prune_artifact_archive_cache(
            tmp_path,
            max_bytes=1,
            max_files=1,
            max_age_s=int(time.time()) + 100,
        )
        assert active.exists()
        assert not other.exists()
    finally:
        artifacts._release_archive(active)


def test_workspace_artifacts_skip_link_aliases(tmp_path: Path) -> None:
    workspace = tmp_path / "run"
    workspace.mkdir()
    real = workspace / "real.safetensors"
    real.write_bytes(b"weights")
    alias = workspace / "alias.safetensors"
    try:
        alias.symlink_to(real)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    listed = _list_workspace_files(workspace)

    names = {entry["path"] for entry in listed["checkpoints"]}
    assert names == {"real.safetensors"}
    with pytest.raises(ValueError, match="cannot be a link"):
        _resolve_workspace_file(workspace, "alias.safetensors")


def test_workspace_delete_rejects_link_alias(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    marker = target / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(target, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    with pytest.raises(HTTPException) as captured:
        artifacts._validate_workspace_delete_target(alias)

    assert captured.value.status_code == 400
    assert marker.is_file()
