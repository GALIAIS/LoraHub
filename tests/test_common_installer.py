"""Tests for the shared backend installer helpers.

Focused on the streaming-progress contract of ``run_step`` (every stderr
line gets forwarded live, last 12 lines are kept as a tail on failure)
and on the ``--progress`` flag passed to ``git clone``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from lorahub.core.backends._common import installer as common
from lorahub.core.backends.errors import BootstrapError


class _FakeProc:
    """Minimal Popen stand-in: yields stderr lines then returns ``rc``."""

    def __init__(self, lines: list[str], rc: int = 0) -> None:
        # Newline-suffix every line so the iterator behaves like a real
        # text-mode file object; run_step rstrips them anyway.
        self.stderr = iter(line if line.endswith("\n") else line + "\n" for line in lines)
        self._rc = rc

    def wait(self) -> int:
        return self._rc


def _patch_popen(
    monkeypatch: pytest.MonkeyPatch,
    proc: _FakeProc,
    recorder: MagicMock | None = None,
) -> None:
    def fake_popen(cmd: list[str], **kw: Any) -> _FakeProc:
        if recorder is not None:
            recorder(cmd, **kw)
        return proc

    monkeypatch.setattr(common.subprocess, "Popen", fake_popen)


def test_run_step_streams_each_line_to_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every non-empty stderr line should be forwarded to the progress callback."""
    lines = [
        "Cloning into 'sd-scripts'...",
        "remote: Counting objects: 100% (10/10), done.",
        "Receiving objects:  50% (50/100)",
        "Receiving objects: 100% (100/100), 5.00 MiB | 2.50 MiB/s, done.",
        "Resolving deltas: 100% (5/5), done.",
    ]
    _patch_popen(monkeypatch, _FakeProc(lines, rc=0))

    seen: list[str] = []
    common.run_step(["git", "clone", "x"], step="clone foo", progress=seen.append)

    # First entry is the step name itself; the rest are streamed git lines,
    # each indented with two spaces so a combined log stays readable.
    assert seen[0] == "clone foo"
    assert seen[1:] == [f"  {line}" for line in lines]


def test_run_step_skips_blank_stderr_lines(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pure whitespace lines from git progress carriage returns shouldn't spam progress."""
    _patch_popen(monkeypatch, _FakeProc(["", "  ", "real line", ""], rc=0))

    seen: list[str] = []
    common.run_step(["git", "clone", "x"], step="clone foo", progress=seen.append)

    # step name + one real line.
    assert seen == ["clone foo", "  real line"]


def test_run_step_raises_with_tail_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """On non-zero exit, the last 12 stderr lines must be appended to progress."""
    lines = [f"line {i}" for i in range(20)]  # 20 lines, only last 12 should survive
    _patch_popen(monkeypatch, _FakeProc(lines, rc=1))

    seen: list[str] = []
    with pytest.raises(BootstrapError) as info:
        common.run_step(["git", "clone", "x"], step="clone foo", progress=seen.append)

    assert info.value.step == "clone foo"
    assert info.value.returncode == 1

    # Last entry is the failure summary carrying the tail (max 12 lines).
    failure_msg = seen[-1]
    assert failure_msg.startswith("clone foo failed (exit 1):")
    tail_lines = failure_msg.split("\n")[1:]
    assert len(tail_lines) == 12
    assert tail_lines == [f"line {i}" for i in range(8, 20)]


def test_run_step_no_progress_callback_still_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Passing progress=None should not crash, but should still raise on failure."""
    _patch_popen(monkeypatch, _FakeProc(["boom"], rc=2))
    with pytest.raises(BootstrapError) as info:
        common.run_step(["git", "clone", "x"], step="clone foo", progress=None)
    assert info.value.returncode == 2


def test_run_step_hides_windows_console(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = MagicMock()
    monkeypatch.setattr(common.sys, "platform", "win32")
    monkeypatch.setattr(common.subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)
    _patch_popen(monkeypatch, _FakeProc([], rc=0), recorder=recorder)

    common.run_step(["git", "clone", "x"], step="clone foo", progress=None)

    assert recorder.call_args.kwargs["creationflags"] == 0x08000000


def test_clone_repo_passes_progress_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """clone_repo must pass --progress so git emits status under a non-TTY pipe."""
    captured: list[list[str]] = []

    def fake_run_step(
        cmd: list[str],
        step: str,
        progress: common.ProgressCallback | None,
    ) -> None:
        captured.append(cmd)

    monkeypatch.setattr(common, "run_step", fake_run_step)

    plan = MagicMock()
    plan.target = tmp_path / "repo"
    plan.git_depth = 1
    plan.github_proxy = None

    common.clone_repo(plan, repo_url="https://example.invalid/repo.git", label="repo")

    assert len(captured) == 1
    cmd = captured[0]
    assert cmd[:2] == ["git", "clone"]
    # The streaming reader needs git to keep printing under a pipe.
    assert "--progress" in cmd
    assert "--depth" in cmd
    assert cmd[cmd.index("--depth") + 1] == "1"
    assert cmd[-2] == "https://example.invalid/repo.git"
    assert cmd[-1] == str(plan.target)


def test_clone_repo_refuses_non_empty_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-empty target must short-circuit before any git invocation."""
    target = tmp_path / "repo"
    target.mkdir()
    (target / "stale.txt").write_text("existing", encoding="utf-8")

    called = MagicMock()
    monkeypatch.setattr(common, "run_step", called)

    plan = MagicMock()
    plan.target = target
    plan.git_depth = 1
    plan.github_proxy = None

    with pytest.raises(BootstrapError):
        common.clone_repo(plan, repo_url="https://example.invalid/r.git", label="r")
    called.assert_not_called()


def test_clone_repo_skips_when_complete_git_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If target is already a complete git checkout, clone should be skipped."""
    target = tmp_path / "repo"
    target.mkdir()
    (target / ".git").mkdir()
    (target / "file.txt").write_text("x", encoding="utf-8")

    # Make _is_complete_git_repo return True
    monkeypatch.setattr(common, "_is_complete_git_repo", lambda t: True)

    called = MagicMock()
    monkeypatch.setattr(common, "run_step", called)

    plan = MagicMock()
    plan.target = target
    plan.git_depth = 1
    plan.github_proxy = None

    seen: list[str] = []
    common.clone_repo(
        plan, repo_url="https://example.invalid/r.git", label="r", progress=seen.append
    )
    called.assert_not_called()
    assert len(seen) == 1
    assert "skipped" in seen[0]


def test_clone_repo_raises_when_non_empty_not_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-empty dir that is NOT a git repo should still raise BootstrapError."""
    target = tmp_path / "repo"
    target.mkdir()
    (target / "random.txt").write_text("x", encoding="utf-8")

    monkeypatch.setattr(common, "_is_complete_git_repo", lambda t: False)

    called = MagicMock()
    monkeypatch.setattr(common, "run_step", called)

    plan = MagicMock()
    plan.target = target
    plan.git_depth = 1
    plan.github_proxy = None

    with pytest.raises(BootstrapError):
        common.clone_repo(plan, repo_url="https://example.invalid/r.git", label="r")
    called.assert_not_called()


def test_torch_index_candidates_put_user_source_first() -> None:
    indexes = common.torch_index_candidates(
        "https://example.invalid/pytorch/whl",
        "cu121",
    )

    assert indexes[0] == "https://example.invalid/pytorch/whl/cu121"
    assert "https://download.pytorch.org/whl/cu121" in indexes


def test_torch_index_from_base_replaces_existing_cuda_suffix() -> None:
    assert (
        common.torch_index_from_base(
            "https://example.invalid/pytorch/whl/cu128",
            "cu124",
        )
        == "https://example.invalid/pytorch/whl/cu124"
    )


def test_pip_install_with_torch_index_fallback_tries_next_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def fake_pip_install(
        _venv_py: Path,
        args: list[str],
        *,
        step: str,
        progress=None,
    ) -> None:
        calls.append(args)
        if len(calls) == 1:
            raise RuntimeError("missing wheel")

    monkeypatch.setattr(common._uv, "pip_install", fake_pip_install)

    plan = MagicMock()
    plan.venv_python = tmp_path / "venv" / "bin" / "python"
    plan.cuda_version = "cu124"
    plan.torch_index_base = "https://bad.invalid/pytorch/whl"

    seen: list[str] = []
    common.pip_install_with_torch_index_fallback(
        plan,
        ["torch==2.6.0", "--index-url", "https://bad.invalid/pytorch/whl/cu124"],
        step="install torch",
        progress=seen.append,
    )

    assert len(calls) == 2
    first_index = calls[0][calls[0].index("--index-url") + 1]
    second_index = calls[1][calls[1].index("--index-url") + 1]
    assert first_index == "https://bad.invalid/pytorch/whl/cu124"
    assert second_index != first_index
    assert second_index.endswith("/cu124")
    assert any("trying" in item for item in seen)
