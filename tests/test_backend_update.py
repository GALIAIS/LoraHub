from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from lorahub.api import backend_update


def _result(
    args: list[str],
    *,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args, returncode, stdout, stderr)


def test_backend_update_refuses_dirty_worktree_without_pull(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    calls: list[tuple[str, ...]] = []

    def fake_git(_repo: Path, *args: str, timeout: float = 30):
        calls.append(args)
        if args[:3] == ("rev-parse", "--abbrev-ref", "HEAD"):
            return _result(list(args), stdout="main\n")
        if args[:2] == ("status", "--porcelain"):
            return _result(list(args), stdout=" M train.py\n")
        raise AssertionError(f"unexpected git command: {args}, timeout={timeout}")

    monkeypatch.setattr(backend_update, "_git", fake_git)

    result = backend_update.apply_update(repo)

    assert "local changes" in (result.error or "")
    assert not any(call and call[0] in {"pull", "reset"} for call in calls)


def test_backend_update_never_resets_after_fast_forward_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    calls: list[tuple[str, ...]] = []

    def fake_git(_repo: Path, *args: str, timeout: float = 30):
        calls.append(args)
        if args[:3] == ("rev-parse", "--abbrev-ref", "HEAD"):
            return _result(list(args), stdout="main\n")
        if args[:2] == ("status", "--porcelain"):
            return _result(list(args))
        if args and args[0] == "pull":
            return _result(list(args), returncode=1, stderr="Not possible to fast-forward")
        raise AssertionError(f"unexpected git command: {args}, timeout={timeout}")

    monkeypatch.setattr(backend_update, "_git", fake_git)

    result = backend_update.apply_update(repo)

    assert "could not fast-forward" in (result.error or "")
    assert "no files were reset" in (result.error or "")
    assert not any(call and call[0] == "reset" for call in calls)


def test_backend_update_refuses_detached_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    monkeypatch.setattr(
        backend_update,
        "_git",
        lambda _repo, *args, **_kwargs: _result(list(args), stdout="HEAD\n"),
    )

    result = backend_update.apply_update(repo)

    assert result.branch == "HEAD"
    assert "detached HEAD" in (result.error or "")


def test_backend_update_check_refuses_detached_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    calls: list[tuple[str, ...]] = []

    def fake_git(_repo: Path, *args: str, **_kwargs):
        calls.append(args)
        return _result(list(args), stdout="HEAD\n")

    monkeypatch.setattr(backend_update, "_git", fake_git)

    result = backend_update.check_update(repo)

    assert result.branch == "HEAD"
    assert "detached HEAD" in (result.error or "")
    assert not any(call and call[0] == "fetch" for call in calls)
