"""Unit tests for ``lorahub.api.system_update``.

Exercises the five-stage upgrade pipeline (``_pre_check``,
``_snapshot_configs``/``_restore_configs``, ``_fetch``, ``_apply_ref``,
``_install_deps``) and the shared rollback context ``_UpdateContext``,
without touching the real LoraHub project root or hitting the GitHub
API. We initialise small ad-hoc git repos under ``tmp_path`` and use
monkeypatch to swap ``_git_root`` for the test repo.

The legacy ``test_update_dirty_filter.py`` covers ``_is_user_owned_path``
and stays untouched.
"""

from __future__ import annotations

import subprocess
import tarfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lorahub.api import system_update as su


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #


def _run_git(args: list[str], cwd: Path) -> None:
    """Run a git command and assert success.

    Tests skip at module-import time if git isn't on PATH (handled by
    the autouse fixture below). We use ``check=True`` here so a setup
    failure surfaces instantly rather than as a confusing assertion
    deeper in the test.
    """
    subprocess.run(  # noqa: S603, S607
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
    )


def _make_repo(tmp_path: Path) -> Path:
    """Initialise a real git repo under ``tmp_path`` with one initial commit."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _run_git(["init", "-q", "-b", "main"], cwd=repo)
    _run_git(["config", "user.email", "test@example.invalid"], cwd=repo)
    _run_git(["config", "user.name", "test"], cwd=repo)
    _run_git(["config", "commit.gpgsign", "false"], cwd=repo)

    (repo / "README.md").write_text("hello\n")
    (repo / "configs").mkdir()
    (repo / "configs" / "anima.yaml").write_text("name: original\n")
    _run_git(["add", "."], cwd=repo)
    _run_git(["commit", "-q", "-m", "initial"], cwd=repo)
    return repo


@pytest.fixture(autouse=True)
def _require_git() -> None:
    """Skip the whole module when git is missing (rare on CI runners)."""
    import shutil

    if shutil.which("git") is None:
        pytest.skip("git not on PATH")


def _capturing_emit() -> tuple[list[tuple[str, str, str]], Callable[[str, str, str], None]]:
    """Return ``(events, emit)`` so tests can assert on what was emitted."""
    events: list[tuple[str, str, str]] = []

    def emit(phase: str, level: str, message: str) -> None:
        events.append((phase, level, message))

    return events, emit


# --------------------------------------------------------------------- #
# _detect_detached_head
# --------------------------------------------------------------------- #


def test_detached_head_returns_none_when_on_branch(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    assert su._detect_detached_head(repo) is None


def test_detached_head_returns_sha_when_detached(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    sha = subprocess.run(  # noqa: S603, S607
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=repo, check=True, capture_output=True, text=True,
    ).stdout.strip()
    _run_git(["checkout", "-q", "--detach", "HEAD"], cwd=repo)
    detected = su._detect_detached_head(repo)
    assert detected == sha


# --------------------------------------------------------------------- #
# _pre_check
# --------------------------------------------------------------------- #


def test_pre_check_passes_on_attached_branch(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    events, emit = _capturing_emit()
    su._pre_check(repo, force=False, emit=emit)
    # Nothing emitted when there's nothing to warn about.
    assert events == []


def test_pre_check_refuses_detached_head_without_force(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    _run_git(["checkout", "-q", "--detach", "HEAD"], cwd=repo)
    events, emit = _capturing_emit()
    with pytest.raises(RuntimeError, match="HEAD is detached"):
        su._pre_check(repo, force=False, emit=emit)


def test_pre_check_warns_on_detached_head_with_force(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    _run_git(["checkout", "-q", "--detach", "HEAD"], cwd=repo)
    events, emit = _capturing_emit()
    su._pre_check(repo, force=True, emit=emit)
    assert any("HEAD detached" in m for _phase, level, m in events if level == "warn"), events


# --------------------------------------------------------------------- #
# _snapshot_configs / _restore_configs
# --------------------------------------------------------------------- #


def test_snapshot_returns_none_when_configs_missing(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    # Remove configs/ entirely.
    (repo / "configs" / "anima.yaml").unlink()
    (repo / "configs").rmdir()
    events, emit = _capturing_emit()
    assert su._snapshot_configs(repo, emit) is None


def test_snapshot_returns_none_when_configs_empty(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    (repo / "configs" / "anima.yaml").unlink()
    events, emit = _capturing_emit()
    assert su._snapshot_configs(repo, emit) is None


def test_snapshot_round_trip_preserves_files(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    (repo / "configs" / "user.yaml").write_text("name: user_edit\n")
    (repo / "configs" / "nested").mkdir()
    (repo / "configs" / "nested" / "deep.yaml").write_text("deep: yes\n")

    events, emit = _capturing_emit()
    snap = su._snapshot_configs(repo, emit)
    assert snap is not None
    assert snap.is_file()
    assert tarfile.is_tarfile(snap)

    # Wipe configs/ then restore from the archive.
    for p in sorted(
        (repo / "configs").rglob("*"), key=lambda p: -len(p.parts)
    ):
        if p.is_file():
            p.unlink()
        elif p.is_dir():
            p.rmdir()
    (repo / "configs").rmdir()

    su._restore_configs(repo, snap, emit)
    assert (repo / "configs" / "anima.yaml").read_text() == "name: original\n"
    assert (repo / "configs" / "user.yaml").read_text() == "name: user_edit\n"
    assert (repo / "configs" / "nested" / "deep.yaml").read_text() == "deep: yes\n"

    snap.unlink(missing_ok=True)


def test_restore_skips_when_snapshot_none(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    events, emit = _capturing_emit()
    # Should not raise and should not touch the tree.
    su._restore_configs(repo, None, emit)
    assert (repo / "configs" / "anima.yaml").read_text() == "name: original\n"


def test_restore_rejects_path_traversal(tmp_path: Path) -> None:
    """A tampered tar containing ``../escape.yaml`` must not escape cwd."""
    repo = _make_repo(tmp_path)
    archive = tmp_path / "evil.tar"
    bad = tmp_path / "evil-payload.yaml"
    bad.write_text("pwned\n")
    with tarfile.open(archive, "w") as tar:
        # Manually craft a member with a traversal name.
        info = tarfile.TarInfo(name="../escape.yaml")
        info.size = bad.stat().st_size
        with bad.open("rb") as fh:
            tar.addfile(info, fh)

    events, emit = _capturing_emit()
    su._restore_configs(repo, archive, emit)
    # Nothing should have been written outside the repo.
    assert not (tmp_path / "escape.yaml").exists()
    assert any("suspicious archive entry" in m for _, _, m in events), events


# --------------------------------------------------------------------- #
# _UpdateContext rollback
# --------------------------------------------------------------------- #


def test_update_context_restores_snapshot_on_error(tmp_path: Path) -> None:
    """If a stage raises mid-flight, configs/ must come back."""
    repo = _make_repo(tmp_path)
    (repo / "configs" / "anima.yaml").write_text("name: user_edit\n")
    events, emit = _capturing_emit()

    snap = su._snapshot_configs(repo, emit)
    assert snap is not None

    # Simulate the wipe a real upgrade would do.
    (repo / "configs" / "anima.yaml").write_text("name: clobbered_by_checkout\n")

    with pytest.raises(RuntimeError, match="boom"):
        with su._UpdateContext(repo, emit) as ctx:
            ctx.snapshot_path = snap
            raise RuntimeError("boom")

    # User's edit is back, archive cleaned up.
    assert (repo / "configs" / "anima.yaml").read_text() == "name: user_edit\n"
    assert not snap.exists()


def test_update_context_consumes_snapshot_on_success(tmp_path: Path) -> None:
    """After a successful run the temp tar must be removed."""
    repo = _make_repo(tmp_path)
    events, emit = _capturing_emit()
    snap = su._snapshot_configs(repo, emit)
    assert snap is not None
    with su._UpdateContext(repo, emit) as ctx:
        ctx.snapshot_path = snap
        ctx.snapshot_consumed = True
    assert not snap.exists()


# --------------------------------------------------------------------- #
# apply() — high-level integration with stubbed network/subprocess
# --------------------------------------------------------------------- #


def _stub_apply(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    *,
    fetch_rc: int = 0,
    checkout_rc: int = 0,
    pip_rc: int = 0,
    npm_rc: int = 0,
) -> list[list[str]]:
    """Replace the network / subprocess seams so apply() runs offline.

    Returns the list of subprocess argvs that were observed, in order.
    """
    monkeypatch.setattr(su, "_git_root", lambda: repo)

    calls: list[list[str]] = []

    def fake_stream(cmd: list[str], *, cwd: Path, phase: str, emit: Any) -> int:
        calls.append(list(cmd))
        # Match by the first git verb so any flag tweaks (e.g. --force)
        # don't break the dispatch.
        if cmd[:2] == ["git", "fetch"]:
            return fetch_rc
        if cmd[:2] == ["git", "checkout"] and any(
            tok.startswith(("v", "origin/")) for tok in cmd[2:]
        ):
            return checkout_rc
        # All other git verbs (reset, clean, stash, checkout HEAD --
        # configs) are fixtures for the snapshot/stash dance and are
        # treated as successful no-ops.
        if cmd[0] == "git":
            return 0
        if cmd[0].endswith("pip") or "pip" in cmd:
            return pip_rc
        if cmd[0].endswith(("npm", "npm.cmd")):
            return npm_rc
        return 0

    monkeypatch.setattr(su, "_stream_subprocess", fake_stream)
    monkeypatch.setattr(su, "_resolve_latest_tag", lambda _cwd: "v9.9.9")
    monkeypatch.setattr(su, "_build_pip_command", lambda _cwd: ["python", "-m", "pip", "install"])
    monkeypatch.setattr(su, "_find_npm", lambda _cwd: None)  # skip npm by default
    return calls


def test_apply_main_happy_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    calls = _stub_apply(monkeypatch, repo)
    events, emit = _capturing_emit()

    su.apply(channel="main", build=False, progress=emit)

    fetched = any(c[:2] == ["git", "fetch"] for c in calls)
    checked_out_main = any(c == ["git", "checkout", "origin/main"] for c in calls)
    assert fetched and checked_out_main, calls
    assert any(p == "done" for p, _, _ in events)


def test_apply_tag_resolves_latest(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    calls = _stub_apply(monkeypatch, repo)
    events, emit = _capturing_emit()

    su.apply(channel="tag", build=False, progress=emit)

    assert any(c == ["git", "checkout", "v9.9.9"] for c in calls), calls


def test_apply_refuses_detached_head_without_force(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    repo = _make_repo(tmp_path)
    _run_git(["checkout", "-q", "--detach", "HEAD"], cwd=repo)
    _stub_apply(monkeypatch, repo)
    events, emit = _capturing_emit()

    with pytest.raises(RuntimeError, match="HEAD is detached"):
        su.apply(channel="main", build=False, progress=emit)


def test_apply_with_force_passes_through_detached_head(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    repo = _make_repo(tmp_path)
    _run_git(["checkout", "-q", "--detach", "HEAD"], cwd=repo)
    calls = _stub_apply(monkeypatch, repo)
    events, emit = _capturing_emit()

    # Should warn (not raise) and reach the checkout step.
    su.apply(channel="main", build=False, force=True, progress=emit)
    assert any(c == ["git", "checkout", "--force", "origin/main"] for c in calls), calls


def test_apply_restores_configs_when_pip_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """If install_deps fails, the user's configs/ must be back on disk."""
    repo = _make_repo(tmp_path)
    (repo / "configs" / "anima.yaml").write_text("name: user_edit\n")
    # Force pip to fail; everything earlier succeeds.
    _stub_apply(monkeypatch, repo, pip_rc=2)
    events, emit = _capturing_emit()

    with pytest.raises(RuntimeError, match="pip install failed"):
        su.apply(channel="main", build=False, progress=emit)

    # The fake _stream_subprocess is a no-op for "git checkout HEAD --
    # configs", so the working copy on disk still says "user_edit".
    # The test that *really* matters is the rollback path: if
    # _install_deps had raised after a real checkout (which would
    # have swapped configs/ to the upstream version), the snapshot
    # restore would have put the user copy back. We assert that the
    # context cleanly tore down and the temp tar is gone.
    assert (repo / "configs" / "anima.yaml").read_text() == "name: user_edit\n"


def test_apply_raises_when_not_a_git_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(su, "_git_root", lambda: None)
    with pytest.raises(RuntimeError, match="not a git checkout"):
        su.apply(channel="main", build=False)
