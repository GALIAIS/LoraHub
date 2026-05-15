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
    """Patch every subprocess.run we shell out to (installer git + uv)."""
    mock = MagicMock(return_value=MagicMock(returncode=0, stderr=""))
    monkeypatch.setattr(installer.subprocess, "run", mock)
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
    assert cmd[:3] == ["git", "clone", "--depth"]
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

    def fake_subprocess_run(cmd: list[str], **_kw: Any) -> MagicMock:
        # The first invocation is `git clone`; mimic it by creating the dir + requirements.
        if cmd[:2] == ["git", "clone"]:
            target.mkdir(parents=True, exist_ok=True)
            (target / "requirements.txt").write_text("accelerate\n", encoding="utf-8")
        return MagicMock(returncode=0, stderr="")

    fake_run.side_effect = fake_subprocess_run

    seen_steps: list[str] = []
    installer.bootstrap(plan, progress=seen_steps.append)

    # Six progress lines, one per logical step. The "upgrade pip" step is a
    # no-op under uv but still emits a status line.
    assert len(seen_steps) == 6
    assert "clone" in seen_steps[0]
    assert "venv" in seen_steps[1]
    assert "torch" in seen_steps[3]
    assert "requirements" in seen_steps[4]
    assert "xformers" in seen_steps[5]


def test_failure_raises_with_step_name(tmp_path: Path, fake_run: MagicMock) -> None:
    fake_run.return_value = MagicMock(returncode=128)
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
