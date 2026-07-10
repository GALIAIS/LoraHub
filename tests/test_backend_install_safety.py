from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from lorahub.api.bootstrap_session import (
    BootstrapRequest,
    _BootstrapSession,
    _prepare_cloned_backend_target,
)
from lorahub.core.backends._common import installer as common

REPO_URL = "https://example.test/backend.git"


def test_bootstrap_session_keeps_bounded_event_backlog() -> None:
    session = _BootstrapSession("session")
    for index in range(250):
        session._emit("info", "step", message=str(index))

    queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    offset, events = session.attach(queue)  # type: ignore[arg-type]

    assert offset == 50
    assert len(events) == 200
    assert events[0]["message"] == "50"


def test_bootstrap_session_drops_oldest_event_for_slow_listener() -> None:
    queue: asyncio.Queue[dict[str, object]] = asyncio.Queue(maxsize=1)
    queue.put_nowait({"message": "old"})

    _BootstrapSession._put_latest(queue, {"message": "new"})  # type: ignore[arg-type]

    assert queue.get_nowait()["message"] == "new"


def test_cleanup_partial_refuses_project_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    keep = tmp_path / "keep.txt"
    keep.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(common, "project_root", lambda: tmp_path)

    with pytest.raises(ValueError, match="protected"):
        common.cleanup_partial(tmp_path, REPO_URL)

    assert keep.read_text(encoding="utf-8") == "keep"


def test_cleanup_partial_refuses_user_data_subtree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models = tmp_path / "models"
    target = models / "mistaken-backend"
    target.mkdir(parents=True)
    keep = target / "weights.safetensors"
    keep.write_bytes(b"weights")
    monkeypatch.setattr(common, "project_root", lambda: tmp_path)

    with pytest.raises(ValueError, match="protected"):
        common.cleanup_partial(target, REPO_URL)

    assert keep.read_bytes() == b"weights"


def test_cleanup_partial_removes_only_marked_external_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "external" / "backend"
    target.mkdir(parents=True)
    (target / "partial.pack").write_bytes(b"partial")
    monkeypatch.setattr(common, "project_root", lambda: tmp_path)
    common._write_install_marker(target, REPO_URL)

    common.cleanup_partial(target, REPO_URL)

    assert not target.exists()
    assert not common._install_marker_path(target).exists()


def test_cleanup_partial_preserves_backend_with_wrong_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "external" / "backend"
    target.mkdir(parents=True)
    keep = target / "user-file.txt"
    keep.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(common, "project_root", lambda: tmp_path)
    common._write_install_marker(target, "https://example.test/other.git")

    with pytest.raises(ValueError, match="matching install marker"):
        common.cleanup_partial(target, REPO_URL)

    assert keep.read_text(encoding="utf-8") == "keep"


def test_reinstall_complete_backend_removes_only_managed_venv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "backend"
    venv = target / ".venv"
    venv.mkdir(parents=True)
    source = target / "train.py"
    source.write_text("print('train')", encoding="utf-8")
    plan = SimpleNamespace(target=target)

    class Installer:
        @staticmethod
        def cleanup_environment(_plan: object) -> None:
            common.cleanup_managed_venvs(target)

    monkeypatch.setattr(common, "is_complete_git_repo", lambda _target: True)
    monkeypatch.setattr(common, "project_root", lambda: tmp_path / "lorahub-data")

    _prepare_cloned_backend_target(
        plan,
        BootstrapRequest(force=True),
        Installer,
        repo_url="https://example.test/backend.git",
    )

    assert not venv.exists()
    assert source.read_text(encoding="utf-8") == "print('train')"


def test_force_retry_preserves_unknown_non_git_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "backend"
    target.mkdir()
    keep = target / "user-file.txt"
    keep.write_text("keep", encoding="utf-8")
    plan = SimpleNamespace(target=target)

    class Installer:
        @staticmethod
        def cleanup_partial(_plan: object) -> None:
            pytest.fail("unknown directories must not be deleted")

    monkeypatch.setattr(common, "is_complete_git_repo", lambda _target: False)

    with pytest.raises(HTTPException, match="files were preserved"):
        _prepare_cloned_backend_target(
            plan,
            BootstrapRequest(force=True),
            Installer,
            repo_url="https://example.test/backend.git",
        )

    assert keep.read_text(encoding="utf-8") == "keep"


def test_backend_target_rejects_linked_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "linked" / "backend"
    monkeypatch.setattr(
        common,
        "_is_link_path",
        lambda path: path == target.parent,
    )

    with pytest.raises(ValueError, match="cannot use links"):
        common.validate_backend_source_target(target)
