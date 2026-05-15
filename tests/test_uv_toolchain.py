"""Unit tests for the uv toolchain helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lorahub.core.toolchain import uv as _uv


@pytest.fixture(autouse=True)
def _reset_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_uv, "_UV_CACHED", None, raising=False)


def test_find_uv_returns_path_when_on_system(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_uv.shutil, "which", lambda _: "/usr/local/bin/uv")
    assert _uv.find_uv() == "/usr/local/bin/uv"


def test_find_uv_returns_none_when_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(_uv.shutil, "which", lambda _: None)
    monkeypatch.setattr(_uv, "_local_uv_path", lambda: tmp_path / "uv")
    assert _uv.find_uv() is None


def test_ensure_uv_uses_existing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_uv.shutil, "which", lambda _: "/usr/local/bin/uv")
    assert _uv.ensure_uv() == "/usr/local/bin/uv"


def test_ensure_uv_bootstraps_when_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(_uv.shutil, "which", lambda _: None)
    fake_target = tmp_path / "uv-fake"
    monkeypatch.setattr(_uv, "_local_uv_path", lambda: fake_target)

    bootstrap_calls: list[tuple] = []

    def fake_bootstrap(progress: object) -> str:
        bootstrap_calls.append((progress,))
        fake_target.parent.mkdir(parents=True, exist_ok=True)
        fake_target.write_text("#!/bin/sh", encoding="utf-8")
        return str(fake_target)

    monkeypatch.setattr(_uv, "_bootstrap_uv", fake_bootstrap)
    out = _uv.ensure_uv()
    assert out == str(fake_target)
    assert bootstrap_calls  # bootstrap_uv was actually invoked


def test_pip_install_invokes_uv_with_python(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(_uv, "_UV_CACHED", "/fake/uv", raising=False)
    captured: list[list[str]] = []

    def fake_run(cmd: list[str], **_kw: object) -> MagicMock:
        captured.append(cmd)
        return MagicMock(returncode=0, stderr="")

    monkeypatch.setattr(_uv.subprocess, "run", fake_run)
    venv_py = tmp_path / "venv" / "bin" / "python"
    _uv.pip_install(venv_py, ["torch", "--index-url", "https://x"], step="install torch")
    assert captured, "uv pip install was not called"
    cmd = captured[0]
    assert cmd[0] == "/fake/uv"
    assert cmd[1:3] == ["pip", "install"]
    assert "--python" in cmd
    assert cmd[cmd.index("--python") + 1] == str(venv_py)
    assert "torch" in cmd
    assert "--index-url" in cmd


def test_create_venv_returns_python_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(_uv, "_UV_CACHED", "/fake/uv", raising=False)
    monkeypatch.setattr(_uv.subprocess, "run", MagicMock(return_value=MagicMock(returncode=0, stderr="")))
    target = tmp_path / "proj"
    py = _uv.create_venv(target)
    expected_dir = target / "venv" / ("Scripts" if sys.platform == "win32" else "bin")
    assert py.parent == expected_dir
    assert py.name in {"python.exe", "python"}
