"""Unit tests for the uv toolchain helpers."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lorahub.core.toolchain import uv as _uv


class _FakePopen:
    def __init__(
        self,
        cmd: list[str],
        *args: object,
        returncode: int = 0,
        output: list[str] | None = None,
        **kwargs: object,
    ) -> None:
        self.cmd = cmd
        self.args = args
        self.kwargs = kwargs
        self.returncode = returncode
        self.stdout = iter(output or [])

    def wait(self) -> int:
        return self.returncode


@pytest.fixture(autouse=True)
def _reset_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(_uv, "_UV_CACHED", None, raising=False)
    monkeypatch.setattr(_uv, "_local_uv_path", lambda: tmp_path / "uv")


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

    def fake_popen(cmd: list[str], **_kw: object) -> _FakePopen:
        captured.append(cmd)
        return _FakePopen(cmd)

    monkeypatch.setattr(_uv.subprocess, "Popen", fake_popen)
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
    monkeypatch.setattr(_uv.subprocess, "Popen", lambda cmd, **kw: _FakePopen(cmd, **kw))
    target = tmp_path / "proj"
    py = _uv.create_venv(target)
    expected_dir = target / "venv" / ("Scripts" if sys.platform == "win32" else "bin")
    assert py.parent == expected_dir
    assert py.name in {"python.exe", "python"}


def test_run_uv_merges_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_uv, "_UV_CACHED", "/fake/uv", raising=False)
    captured: list[tuple[list[str], dict[str, object]]] = []

    def fake_popen(cmd: list[str], **kwargs: object) -> _FakePopen:
        captured.append((cmd, kwargs))
        return _FakePopen(cmd, **kwargs)

    monkeypatch.setattr(_uv.subprocess, "Popen", fake_popen)

    _uv.run_uv(["sync"], step="sync", env={"UV_CACHE_DIR": "/work/.cache/uv"})

    cmd, kwargs = captured[0]
    assert cmd == ["/fake/uv", "sync"]
    env = kwargs["env"]
    assert isinstance(env, dict)
    assert env["UV_CACHE_DIR"] == "/work/.cache/uv"


def test_run_uv_hides_windows_console(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_uv, "_UV_CACHED", "/fake/uv", raising=False)
    monkeypatch.setattr(_uv.sys, "platform", "win32")
    monkeypatch.setattr(_uv.subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)
    captured: list[dict[str, object]] = []

    def fake_popen(cmd: list[str], **kwargs: object) -> _FakePopen:
        captured.append(kwargs)
        return _FakePopen(cmd, **kwargs)

    monkeypatch.setattr(_uv.subprocess, "Popen", fake_popen)

    _uv.run_uv(["sync"], step="sync")

    assert captured[0]["creationflags"] == 0x08000000


def test_venv_python_picks_right_layout_per_platform(tmp_path: Path) -> None:
    py = _uv.venv_python(tmp_path / "proj")
    if sys.platform == "win32":
        assert py.parts[-2:] == ("Scripts", "python.exe")
    else:
        assert py.parts[-2:] == ("bin", "python")


def test_bootstrap_uv_translates_pep668_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Linux distros following PEP 668 should get an actionable hint."""
    monkeypatch.setattr(_uv, "_local_uv_path", lambda: tmp_path / "bin" / "uv")
    monkeypatch.setattr(_uv, "_bin_dir", lambda: tmp_path / "bin")

    pep668_stderr = (
        "error: externally-managed-environment\n"
        "× This environment is externally managed\n"
    )
    monkeypatch.setattr(
        _uv.subprocess,
        "run",
        MagicMock(return_value=MagicMock(returncode=1, stderr=pep668_stderr)),
    )
    with pytest.raises(RuntimeError, match="externally-managed"):
        _uv._bootstrap_uv(progress=None)
