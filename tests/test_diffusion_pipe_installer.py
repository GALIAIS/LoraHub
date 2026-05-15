"""Tests for diffusion-pipe installer (subprocess-mocked)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from lorahub.core.backends.diffusion_pipe import installer
from lorahub.core.toolchain import uv as _uv


@pytest.fixture
def fake_run(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock = MagicMock(return_value=MagicMock(returncode=0, stderr=""))
    monkeypatch.setattr(installer.subprocess, "run", mock)
    monkeypatch.setattr(_uv.subprocess, "run", mock)
    monkeypatch.setattr(_uv, "_UV_CACHED", "/fake/uv", raising=False)
    return mock


def _commands(fake_run: MagicMock) -> list[list[str]]:
    return [call.args[0] for call in fake_run.call_args_list]


def test_install_requirements_strips_deepspeed_line(
    tmp_path: Path, fake_run: MagicMock
) -> None:
    target = tmp_path / "dp"
    target.mkdir()
    (target / "requirements.txt").write_text(
        "deepspeed==0.18.4\ntransformers\nnumpy\n", encoding="utf-8"
    )
    plan = installer.BootstrapPlan(target=target)

    seen: list[str] = []
    installer.install_requirements(plan, progress=seen.append)

    # The filtered file got written without deepspeed.
    filtered = (target / "requirements.lorahub.txt").read_text(encoding="utf-8")
    assert "deepspeed" not in filtered
    assert "transformers" in filtered
    assert "numpy" in filtered

    # Progress events must mention skipping deepspeed.
    assert any("skip" in s.lower() and "deepspeed" in s.lower() for s in seen)

    # uv pip install -r requirements.lorahub.txt was the actual subprocess.
    cmd = _commands(fake_run)[0]
    assert "pip" in cmd and "install" in cmd
    assert str(filtered_path := target / "requirements.lorahub.txt") in cmd
    assert filtered_path.is_file()


def test_install_deepspeed_skips_on_windows(
    tmp_path: Path, fake_run: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    plan = installer.BootstrapPlan(target=tmp_path / "dp")
    seen: list[str] = []
    installer.install_deepspeed(plan, progress=seen.append)
    fake_run.assert_not_called()
    assert any("skip deepspeed" in s.lower() for s in seen)


def test_install_deepspeed_runs_on_linux(
    tmp_path: Path, fake_run: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    plan = installer.BootstrapPlan(target=tmp_path / "dp")
    installer.install_deepspeed(plan)
    cmd = _commands(fake_run)[0]
    assert "deepspeed" in cmd
    assert "pip" in cmd and "install" in cmd


def test_bootstrap_runs_six_logical_steps(tmp_path: Path, fake_run: MagicMock) -> None:
    target = tmp_path / "dp"
    plan = installer.BootstrapPlan(target=target)

    def fake_subprocess_run(cmd: list[str], **_kw: Any) -> MagicMock:
        if cmd[:2] == ["git", "clone"]:
            target.mkdir(parents=True, exist_ok=True)
            (target / "requirements.txt").write_text("transformers\n", encoding="utf-8")
        return MagicMock(returncode=0, stderr="")

    fake_run.side_effect = fake_subprocess_run

    seen: list[str] = []
    installer.bootstrap(plan, progress=seen.append)

    # clone, venv, pip-noop, torch, requirements, deepspeed.
    # The "skipping from requirements" line is informational and is *also*
    # emitted when deepspeed is in requirements; we filter it for the count.
    main_steps = [s for s in seen if not s.startswith("skipping from")]
    assert len(main_steps) >= 6
    assert "clone" in main_steps[0]
    assert "venv" in main_steps[1]
    assert "torch" in main_steps[3]
    assert "requirements" in main_steps[4]
