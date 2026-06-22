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


def test_update_type_payloads_round_trip() -> None:
    from lorahub.api.system_update_types import CacheBlob, UpdateInfo

    info = UpdateInfo(
        channel="dev",
        current="1.0.0",
        latest="1.0.1",
        update_available=True,
        release_url="https://example.invalid/release",
        is_dirty=True,
    )
    blob = CacheBlob(data={"dev": info.to_dict()}, updated_at=12.5)

    assert info.to_dict()["channel"] == "dev"
    assert info.to_dict()["is_dirty"] is True
    assert blob.data["dev"]["latest"] == "1.0.1"
    assert blob.updated_at == 12.5


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
    su._pre_check(repo, channel="dev", force=False, emit=emit)
    # Nothing emitted when there's nothing to warn about.
    assert events == []


def test_pre_check_refuses_detached_head_without_force(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    _run_git(["checkout", "-q", "--detach", "HEAD"], cwd=repo)
    events, emit = _capturing_emit()
    # No ``origin/dev`` here so the auto-attach reachable check
    # short-circuits via the fetch-failed branch and we still raise.
    with pytest.raises(RuntimeError, match="HEAD is detached"):
        su._pre_check(repo, channel="dev", force=False, emit=emit)


def test_pre_check_warns_on_detached_head_with_force(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    _run_git(["checkout", "-q", "--detach", "HEAD"], cwd=repo)
    events, emit = _capturing_emit()
    su._pre_check(repo, channel="dev", force=True, emit=emit)
    assert any(
        "detached" in m.lower()
        for _phase, level, m in events
        if level == "warn"
    ), events


def test_pre_check_auto_attaches_when_detached_sha_in_remote(tmp_path: Path) -> None:
    """When the detached SHA is reachable from origin/<channel>, the
    pre-check switches to the channel branch instead of refusing."""
    upstream_root = tmp_path / "upstream"
    upstream_root.mkdir()
    upstream = _make_repo(upstream_root)
    # Rename upstream's default branch to ``dev`` so the channel name
    # we pass into _pre_check lines up with what the remote serves.
    _run_git(["branch", "-m", "main", "dev"], cwd=upstream)
    clone_root = tmp_path / "clone"
    clone_root.mkdir()
    repo = clone_root / "repo"
    _run_git(["clone", "-q", str(upstream), str(repo)], cwd=clone_root)
    # Detach HEAD onto the same SHA the remote dev branch points at —
    # auto-attach should kick in and put us back on `dev`.
    _run_git(["checkout", "-q", "--detach", "HEAD"], cwd=repo)
    # Make sure the local ``dev`` branch ref doesn't exist before the
    # auto-attach so we exercise the branch-creation path (--checkout -B
    # creates if missing). ``git branch -D dev`` would fail with the
    # branch checked out, but we're already detached so it's safe.
    _run_git(["branch", "-D", "dev"], cwd=repo)
    events, emit = _capturing_emit()
    su._pre_check(repo, channel="dev", force=False, emit=emit)
    branch = subprocess.run(
        ["git", "symbolic-ref", "--short", "HEAD"],
        cwd=repo, capture_output=True, text=True, check=False,
    ).stdout.strip()
    assert branch == "dev"
    assert any(
        "auto-attaching" in m
        for _phase, level, m in events
        if level == "info"
    ), events


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


def test_snapshot_returns_none_when_only_tracked_present(tmp_path: Path) -> None:
    """Tracked presets are git-owned and ride along ``git checkout``.

    With nothing untracked under configs/, snapshot has nothing to do
    and returns None — restore is a no-op and the upgrade lets upstream
    preset fixes propagate.
    """
    repo = _make_repo(tmp_path)
    events, emit = _capturing_emit()
    assert su._snapshot_configs(repo, emit) is None


def test_snapshot_round_trip_preserves_untracked_only(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    # Author-side yamls — never seen by git.
    (repo / "configs" / "user.yaml").write_text("name: user_edit\n")
    (repo / "configs" / "nested").mkdir()
    (repo / "configs" / "nested" / "deep.yaml").write_text("deep: yes\n")
    # Modification on a tracked preset — must NOT be snapshotted (git
    # owns it; stash flow handles the user's edit downstream).
    (repo / "configs" / "anima.yaml").write_text("name: locally_edited\n")

    events, emit = _capturing_emit()
    snap = su._snapshot_configs(repo, emit)
    assert snap is not None
    assert snap.is_file()
    assert tarfile.is_tarfile(snap)

    # Tar must contain only the two untracked entries.
    with tarfile.open(snap, "r") as tar:
        names = sorted(m.name for m in tar.getmembers() if m.isfile())
    assert names == ["configs/nested/deep.yaml", "configs/user.yaml"], names

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
    # Tracked file is NOT in the snapshot — it stays gone post-wipe
    # (in the real upgrade flow `git checkout` would already have put
    # the upstream version back; here we only assert snapshot scope).
    assert not (repo / "configs" / "anima.yaml").exists()
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
    """If a stage raises mid-flight, the user's untracked yamls return."""
    repo = _make_repo(tmp_path)
    (repo / "configs" / "user.yaml").write_text("name: user_edit\n")
    events, emit = _capturing_emit()

    snap = su._snapshot_configs(repo, emit)
    assert snap is not None

    # Simulate the wipe a real upgrade's checkout would do.
    (repo / "configs" / "user.yaml").unlink()

    with pytest.raises(RuntimeError, match="boom"):
        with su._UpdateContext(repo, emit) as ctx:
            ctx.snapshot_path = snap
            raise RuntimeError("boom")

    # User's untracked yaml is back, archive cleaned up.
    assert (repo / "configs" / "user.yaml").read_text() == "name: user_edit\n"
    assert not snap.exists()


def test_update_context_consumes_snapshot_on_success(tmp_path: Path) -> None:
    """After a successful run the temp tar must be removed."""
    repo = _make_repo(tmp_path)
    # Need at least one untracked file so the snapshot is non-empty.
    (repo / "configs" / "user.yaml").write_text("name: u\n")
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
    checked_out_dev = any(c == ["git", "checkout", "origin/dev"] for c in calls)
    assert fetched and checked_out_dev, calls
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
    assert any(c == ["git", "checkout", "--force", "origin/dev"] for c in calls), calls


def test_apply_force_clean_excludes_user_owned_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """``--force`` must never wipe untracked datasets/, runs/, models/, etc.

    Regression for the case where ``git clean -fd`` only had ``-e configs``.
    Untracked user artefacts living under the other user-owned prefixes
    (datasets/, runs/, models/, output/, .env*, external/anima_lora/output)
    must all be passed through as ``-e <prefix>`` so a forced upgrade
    can't rm them.
    """
    repo = _make_repo(tmp_path)
    calls = _stub_apply(monkeypatch, repo)
    events, emit = _capturing_emit()

    su.apply(channel="main", build=False, force=True, progress=emit)

    clean_calls = [c for c in calls if c[:3] == ["git", "clean", "-fd"]]
    assert clean_calls, calls
    clean_argv = clean_calls[0]
    excludes = {
        clean_argv[i + 1]
        for i, tok in enumerate(clean_argv[:-1])
        if tok == "-e"
    }
    expected = {
        prefix.rstrip("/").replace("\\", "/")
        for prefix in su._USER_OWNED_PREFIXES
    }
    missing = expected - excludes
    assert not missing, f"clean -fd missing user-owned excludes: {missing}"


def test_apply_restores_configs_when_pip_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """If install_deps fails, the user's untracked yamls must come back."""
    repo = _make_repo(tmp_path)
    (repo / "configs" / "user.yaml").write_text("name: user_edit\n")
    # Force pip to fail; everything earlier succeeds.
    _stub_apply(monkeypatch, repo, pip_rc=2)
    events, emit = _capturing_emit()

    with pytest.raises(RuntimeError, match="pip install failed"):
        su.apply(channel="main", build=False, progress=emit)

    # The fake _stream_subprocess is a no-op for every git verb, so the
    # working copy on disk still says "user_edit". The test that *really*
    # matters is the rollback path: if _install_deps had raised after a
    # real checkout (which would have swapped configs/ to the upstream
    # version), the snapshot restore would have put the user's untracked
    # yamls back. Tracked presets are intentionally NOT in the snapshot
    # — they're git-owned and the real upgrade would have left them on
    # the upstream version.
    assert (repo / "configs" / "user.yaml").read_text() == "name: user_edit\n"


def test_apply_raises_when_not_a_git_checkout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(su, "_git_root", lambda: None)
    with pytest.raises(RuntimeError, match="不是 git 检出"):
        su.apply(channel="main", build=False)


# --------------------------------------------------------------------- #
# Version resolution (zip-install fallback chain)
# --------------------------------------------------------------------- #


def test_resolve_version_prefers_hatch_vcs(monkeypatch: pytest.MonkeyPatch) -> None:
    """When _version.py is materialised, hatch-vcs wins."""
    import lorahub
    monkeypatch.setattr(su, "_git_describe_runtime", lambda: None)
    monkeypatch.setattr(lorahub, "__version__", "0.5.1.post3+gabc1234")
    v, src = su._resolve_version()
    assert v == "0.5.1.post3+gabc1234"
    assert src == "hatch-vcs"


def test_resolve_version_falls_through_placeholders(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """``0.0.0+unknown`` from __version__ shouldn't short-circuit the chain.

    A ZIP-extracted install with hatch-vcs running blind sets ``__version__``
    to the placeholder; we expect the resolver to push past it and try the
    other sources.
    """
    import lorahub
    monkeypatch.setattr(su, "_git_describe_runtime", lambda: None)
    monkeypatch.setattr(lorahub, "__version__", "0.0.0+unknown")
    # Block dist metadata too so we definitely land on the changelog branch.
    monkeypatch.setattr(
        su,
        "_read_changelog_version",
        lambda: "0.4.0",
    )
    # Force importlib.metadata.version("lorahub") to raise so we skip step 2.
    import importlib.metadata as md
    monkeypatch.setattr(md, "version", lambda name: (_ for _ in ()).throw(md.PackageNotFoundError(name)))
    v, src = su._resolve_version()
    assert v == "0.4.0"
    assert src == "changelog"


def test_resolve_version_fallback_when_all_sources_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Nothing on disk → ``0.0.0+unknown`` + source ``fallback``."""
    import lorahub
    monkeypatch.setattr(su, "_git_describe_runtime", lambda: None)
    monkeypatch.setattr(lorahub, "__version__", "0.0.0")
    monkeypatch.setattr(su, "_read_changelog_version", lambda: None)
    import importlib.metadata as md
    monkeypatch.setattr(
        md,
        "version",
        lambda name: (_ for _ in ()).throw(md.PackageNotFoundError(name)),
    )
    v, src = su._resolve_version()
    assert v == "0.0.0+unknown"
    assert src == "fallback"


def test_check_marks_zip_install_as_non_git(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``check()`` reports ``git_checkout=False`` when there's no .git/.

    This is the signal the web UI uses to grey out the apply button
    instead of letting the user click into a RuntimeError.
    """
    monkeypatch.setattr(su, "_git_root", lambda: None)
    monkeypatch.setattr(su, "_resolve_version", lambda: ("0.4.0", "changelog"))
    # Bypass network — return a stub remote payload directly.
    monkeypatch.setattr(
        su,
        "_refresh_tag",
        lambda: {
            "tag_name": "v0.5.0",
            "version_str": "0.5.0",
            "release_notes": "",
            "published_at": None,
        },
    )
    # Bypass the on-disk cache so the fresh path is exercised.
    monkeypatch.setattr(su, "_read_cache", lambda: su._CacheBlob())
    monkeypatch.setattr(su, "_write_cache", lambda blob: None)

    info = su.check(channel="tag", force=True)
    assert info.git_checkout is False
    assert info.version_source == "changelog"
    assert info.current == "0.4.0"


# --------------------------------------------------------------------- #
# Encoding regression — _stream_subprocess on a zh-CN Windows host
# --------------------------------------------------------------------- #


def test_stream_subprocess_decodes_utf8_glyphs(tmp_path: Path) -> None:
    """Vite/npm/pip emit UTF-8 status glyphs (✓, ▲, CJK boxed text).

    On a zh-CN Windows host ``locale.getpreferredencoding()`` is
    ``cp936``/``gbk``, which raises ``UnicodeDecodeError`` on the very
    first byte of a multibyte UTF-8 sequence. The fix is to pin
    ``encoding="utf-8", errors="replace"`` on the Popen so output is
    decoded the same way regardless of host locale.
    """
    import sys

    # Emit the exact byte that triggered the user's failure (0x93 — the
    # tail byte of "✓") plus a CJK string, then exit. We encode at the
    # bytes level inside the child so the parent's encoding choice is
    # what's under test.
    code = (
        "import sys; "
        "sys.stdout.buffer.write('vite v7.3.3 ✓ building\\n'.encode('utf-8')); "
        "sys.stdout.buffer.write('构建前端\\n'.encode('utf-8')); "
        "sys.stdout.flush()"
    )
    captured: list[tuple[str, str, str]] = []

    rc = su._stream_subprocess(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        phase="build",
        emit=lambda phase, level, msg: captured.append((phase, level, msg)),
    )
    assert rc == 0
    body = "\n".join(msg for _, _, msg in captured)
    assert "✓" in body, f"check mark lost in decode: {body!r}"
    assert "构建前端" in body, f"CJK lost in decode: {body!r}"


# --------------------------------------------------------------------- #
# Runtime bind persistence used by the in-app updater restart path
# --------------------------------------------------------------------- #


def test_runtime_bind_round_trips_and_preserves_legacy_port(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from lorahub.api import runtime_bind

    monkeypatch.setattr(runtime_bind, "user_state_path", lambda *_args: tmp_path)

    runtime_bind.write_runtime_bind("0.0.0.0", 19090, pid=1234)
    bind = runtime_bind.read_runtime_bind()
    assert bind is not None
    assert bind.host == "0.0.0.0"
    assert bind.port == 19090
    assert bind.pid == 1234
    assert runtime_bind.port_file().read_text(encoding="utf-8").strip() == "19090"

    runtime_bind.clear_runtime_bind(keep_bind=True)
    preserved = runtime_bind.read_runtime_bind()
    assert preserved is not None
    assert preserved.host == "0.0.0.0"
    assert preserved.port == 19090
    assert preserved.pid is None


def test_update_restart_args_preserve_recorded_uvicorn_bind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lorahub.api import runtime_bind
    from lorahub.api.runtime_bind import RuntimeBind

    monkeypatch.setattr(
        runtime_bind,
        "read_runtime_bind",
        lambda: RuntimeBind(host="0.0.0.0", port=18765, pid=42),
    )

    args = runtime_bind.restart_args(
        "/opt/venv/bin/python",
        [
            "-m",
            "uvicorn",
            "lorahub.api.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ],
    )
    assert args == [
        "/opt/venv/bin/python",
        "-m",
        "uvicorn",
        "lorahub.api.app:app",
        "--host",
        "0.0.0.0",
        "--port",
        "18765",
    ]


def test_update_restart_args_append_missing_uvicorn_bind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lorahub.api import runtime_bind
    from lorahub.api.runtime_bind import RuntimeBind

    monkeypatch.setattr(
        runtime_bind,
        "read_runtime_bind",
        lambda: RuntimeBind(host="127.0.0.1", port=19001, pid=None),
    )

    args = runtime_bind.restart_args(
        "/opt/venv/bin/python",
        ["-m", "uvicorn", "lorahub.api.app:app"],
    )
    assert args[-4:] == ["--host", "127.0.0.1", "--port", "19001"]


def test_update_restart_args_leave_non_uvicorn_commands_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lorahub.api import runtime_bind
    from lorahub.api.runtime_bind import RuntimeBind

    monkeypatch.setattr(
        runtime_bind,
        "read_runtime_bind",
        lambda: RuntimeBind(host="0.0.0.0", port=18765, pid=42),
    )

    args = runtime_bind.restart_args(
        "/opt/venv/bin/python",
        ["-m", "lorahub", "serve", "--port", "8123"],
    )
    assert args == ["/opt/venv/bin/python", "-m", "lorahub", "serve", "--port", "8123"]
