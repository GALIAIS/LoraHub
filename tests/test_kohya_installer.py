"""Tests for kohya installer (subprocess-mocked)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from lorahub.core.backends.kohya import installer
from lorahub.core.toolchain import uv as _uv


@pytest.fixture
def fake_run(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch every subprocess we shell out to (installer git Popen + uv run).

    Returns a MagicMock recording every cmd. Tweak ``fake_run.popen_rc`` to
    simulate a non-zero git exit, ``fake_run.popen_stderr_lines`` to feed
    fake stderr lines into the streaming reader, and ``fake_run.git_clone_hook``
    (a callable taking the cmd) to mutate the filesystem when the simulated
    `git clone` "runs" -- e.g. mkdir the target + drop a requirements.txt
    so later steps in `bootstrap()` find what they need.
    """
    mock = MagicMock(return_value=MagicMock(returncode=0, stderr=""))
    mock.popen_rc = 0
    mock.popen_stderr_lines = []
    mock.git_clone_hook = None

    def fake_popen(cmd: list[str], **kw: Any) -> MagicMock:
        # Funnel Popen calls through the same mock so call_args_list captures them.
        mock(cmd, **kw)
        if mock.git_clone_hook is not None:
            mock.git_clone_hook(cmd)
        proc = MagicMock()
        proc.stderr = iter(list(mock.popen_stderr_lines))
        proc.wait.return_value = mock.popen_rc
        return proc

    monkeypatch.setattr(installer.subprocess, "run", mock)
    monkeypatch.setattr(installer.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(_uv.subprocess, "run", mock)
    # Pretend uv is already on PATH so ensure_uv() doesn't try to bootstrap it.
    monkeypatch.setattr(_uv, "_UV_CACHED", "/fake/uv", raising=False)
    return mock


def _commands(fake_run: MagicMock) -> list[list[str]]:
    return [call.args[0] for call in fake_run.call_args_list]


def test_plan_picks_platform_appropriate_python(tmp_path: Path) -> None:
    plan = installer.BootstrapPlan(target=tmp_path / "sd")
    if sys.platform == "win32":
        assert plan.venv_python.parts[-2:] == ("Scripts", "python.exe")
    else:
        assert plan.venv_python.parts[-2:] == ("bin", "python")


def test_plan_torch_index_uses_cuda_version() -> None:
    plan = installer.BootstrapPlan(target=Path("/tmp/sd"), cuda_version="cu121")
    assert plan.torch_index.endswith("/cu121")


def test_clone_runs_git_clone(tmp_path: Path, fake_run: MagicMock) -> None:
    plan = installer.BootstrapPlan(target=tmp_path / "sd-scripts")
    installer.clone(plan)
    cmd = _commands(fake_run)[0]
    assert cmd[:2] == ["git", "clone"]
    # `--progress` is required so git emits its "Receiving objects..." lines
    # in non-TTY mode; the streaming reader in run_step relays them live.
    assert "--progress" in cmd
    assert "--depth" in cmd
    assert cmd[-2] == installer.KOHYA_REPO_URL
    assert cmd[-1] == str(plan.target)


def test_clone_refuses_non_empty_target(tmp_path: Path, fake_run: MagicMock) -> None:
    target = tmp_path / "sd-scripts"
    target.mkdir()
    (target / "junk.txt").write_text("not empty", encoding="utf-8")

    with pytest.raises(installer.BootstrapError):
        installer.clone(installer.BootstrapPlan(target=target))
    fake_run.assert_not_called()


def test_install_torch_uses_cuda_index(tmp_path: Path, fake_run: MagicMock) -> None:
    plan = installer.BootstrapPlan(target=tmp_path / "sd", cuda_version="cu121")
    installer.install_torch(plan)
    cmd = _commands(fake_run)[0]
    assert "--index-url" in cmd
    assert cmd[cmd.index("--index-url") + 1].endswith("/cu121")
    assert any(a.startswith("torch==") for a in cmd)
    assert any(a.startswith("torchvision==") for a in cmd)


def test_install_requirements_needs_file(tmp_path: Path, fake_run: MagicMock) -> None:
    plan = installer.BootstrapPlan(target=tmp_path / "sd")
    plan.target.mkdir()
    with pytest.raises(installer.BootstrapError):
        installer.install_requirements(plan)
    fake_run.assert_not_called()


def test_install_requirements_runs_pip(tmp_path: Path, fake_run: MagicMock) -> None:
    target = tmp_path / "sd"
    target.mkdir()
    (target / "requirements.txt").write_text("accelerate\n", encoding="utf-8")
    plan = installer.BootstrapPlan(target=target)

    installer.install_requirements(plan)

    cmd = _commands(fake_run)[0]
    assert "-r" in cmd
    assert str(target / "requirements.txt") in cmd


def test_xformers_skipped_when_disabled(tmp_path: Path, fake_run: MagicMock) -> None:
    plan = installer.BootstrapPlan(target=tmp_path / "sd", install_xformers=False)
    installer.install_xformers(plan)
    fake_run.assert_not_called()


def test_xformers_uses_cuda_index(tmp_path: Path, fake_run: MagicMock) -> None:
    plan = installer.BootstrapPlan(target=tmp_path / "sd", cuda_version="cu121")
    installer.install_xformers(plan)
    cmd = _commands(fake_run)[0]
    assert "xformers" in cmd
    assert cmd[cmd.index("--index-url") + 1].endswith("/cu121")


def test_bootstrap_orchestrates_every_step(tmp_path: Path, fake_run: MagicMock) -> None:
    target = tmp_path / "sd"
    plan = installer.BootstrapPlan(target=target)

    def on_clone(cmd: list[str]) -> None:
        # Mimic git clone landing on disk so install_requirements finds files.
        if cmd[:2] == ["git", "clone"]:
            target.mkdir(parents=True, exist_ok=True)
            (target / "requirements.txt").write_text("accelerate\n", encoding="utf-8")

    fake_run.git_clone_hook = on_clone

    seen_steps: list[str] = []
    installer.bootstrap(plan, progress=seen_steps.append)

    # Six logical steps; "upgrade pip" is a no-op under uv but still emits a
    # status line. We may also emit indented streaming lines from git stderr,
    # so filter to top-level step lines (no leading whitespace) for the count.
    main_steps = [s for s in seen_steps if not s.startswith("  ")]
    assert len(main_steps) == 6
    assert "clone" in main_steps[0]
    assert "venv" in main_steps[1]
    assert "torch" in main_steps[3]
    assert "requirements" in main_steps[4]
    assert "xformers" in main_steps[5]


def test_failure_raises_with_step_name(tmp_path: Path, fake_run: MagicMock) -> None:
    fake_run.popen_rc = 128
    with pytest.raises(installer.BootstrapError) as info:
        installer.clone(installer.BootstrapPlan(target=tmp_path / "sd"))
    assert "clone" in info.value.step
    assert info.value.returncode == 128


def test_cleanup_removes_partial_target(tmp_path: Path) -> None:
    target = tmp_path / "sd"
    target.mkdir()
    (target / "x").write_text("y", encoding="utf-8")
    installer.cleanup_partial(installer.BootstrapPlan(target=target))
    assert not target.exists()
