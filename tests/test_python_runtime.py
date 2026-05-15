"""Tests for the portable Python runtime helpers and endpoints."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from lorahub.core.toolchain import python_runtime
from lorahub.core.toolchain import uv as _uv


@pytest.fixture(autouse=True)
def _stub_uv_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_uv, "_UV_CACHED", "/fake/uv", raising=False)


def _ok(stdout: str = "") -> MagicMock:
    return MagicMock(returncode=0, stdout=stdout, stderr="")


def test_installed_runtimes_parses_uv_json_array(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = json.dumps([
        {"version": "3.11.10", "path": "/r/3.11.10/bin/python", "implementation": "cpython"},
        {"version": "3.12.5", "path": "/r/3.12.5/bin/python"},
    ])
    monkeypatch.setattr(python_runtime.subprocess, "run", lambda *a, **k: _ok(payload))
    out = python_runtime.installed_runtimes()
    assert len(out) == 2
    assert out[0]["version"] == "3.11.10"
    assert out[0]["path"].endswith("python")


def test_installed_runtimes_parses_jsonl_legacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = (
        '{"version":"3.11.10","path":"/r/a/python"}\n'
        '{"version":"3.12.5","path":"/r/b/python"}\n'
    )
    monkeypatch.setattr(python_runtime.subprocess, "run", lambda *a, **k: _ok(payload))
    out = python_runtime.installed_runtimes()
    assert [e["version"] for e in out] == ["3.11.10", "3.12.5"]


def test_installed_runtimes_handles_uv_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        python_runtime.subprocess,
        "run",
        lambda *a, **k: MagicMock(returncode=1, stdout="", stderr="boom"),
    )
    assert python_runtime.installed_runtimes() == []


def test_runtime_python_returns_path_for_matching_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fake_py = tmp_path / "python"
    fake_py.write_text("#!/bin/sh", encoding="utf-8")
    monkeypatch.setattr(
        python_runtime,
        "installed_runtimes",
        lambda: [{"version": "3.11.10", "path": str(fake_py)}],
    )
    out = python_runtime.runtime_python("3.11")
    assert out == fake_py


def test_runtime_python_none_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(python_runtime, "installed_runtimes", list)
    assert python_runtime.runtime_python("3.11") is None


def test_install_runtime_calls_uv_python_install(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: list[list[str]] = []

    def fake_run(cmd: list[str], **_kw: object) -> MagicMock:
        captured.append(cmd)
        return MagicMock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(python_runtime.subprocess, "run", fake_run)
    monkeypatch.setattr(python_runtime, "PYTHON_ROOT", tmp_path / "py")
    monkeypatch.setattr(
        python_runtime,
        "runtime_info",
        lambda v: {"version": v, "path": str(tmp_path / "py" / "python")},
    )

    out = python_runtime.install_runtime("3.11")
    cmd = captured[0]
    assert cmd[0] == "/fake/uv"
    assert cmd[1:3] == ["python", "install"]
    assert "--install-dir" in cmd
    assert cmd[-1] == "3.11"
    assert out["version"] == "3.11"


def test_install_runtime_translates_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        python_runtime.subprocess,
        "run",
        lambda *a, **k: MagicMock(returncode=1, stderr="network unreachable"),
    )
    with pytest.raises(RuntimeError, match="uv python install failed"):
        python_runtime.install_runtime("3.11")


def test_status_payload_has_required_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(python_runtime, "installed_runtimes", list)
    out = python_runtime.status()
    for key in ("default_version", "recommended_versions", "install_dir", "platform", "installed", "active"):
        assert key in out
    assert out["active"] is None
    assert "system" in out["platform"]


# --------------------------------------------------------------------------- #
# HTTP endpoints                                                              #
# --------------------------------------------------------------------------- #


@pytest.fixture
def runtime_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from fastapi.testclient import TestClient
    from lorahub.api import app as app_mod
    from lorahub.api.settings import SettingsStore

    monkeypatch.setattr(
        app_mod, "_settings_store", SettingsStore(tmp_path / "settings.json")
    )
    return TestClient(app_mod.app)


def test_get_runtime_status_endpoint(
    runtime_client: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(python_runtime, "installed_runtimes", list)
    r = runtime_client.get("/api/runtime/python")  # type: ignore[attr-defined]
    assert r.status_code == 200
    body = r.json()
    assert body["default_version"] == python_runtime.DEFAULT_VERSION
    assert "recommended_versions" in body


def test_install_runtime_endpoint(
    runtime_client: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: list[str] = []

    def fake_install(version: str = python_runtime.DEFAULT_VERSION) -> dict[str, object]:
        captured.append(version)
        return {"version": version, "path": "/fake/python"}

    monkeypatch.setattr(python_runtime, "install_runtime", fake_install)
    monkeypatch.setattr(python_runtime, "installed_runtimes", list)
    r = runtime_client.post(  # type: ignore[attr-defined]
        "/api/runtime/python/install", json={"version": "3.11"}
    )
    assert r.status_code == 200
    assert captured == ["3.11"]


def test_install_runtime_endpoint_surfaces_failure(
    runtime_client: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_install(version: str = python_runtime.DEFAULT_VERSION) -> dict[str, object]:
        msg = f"could not download {version}"
        raise RuntimeError(msg)

    monkeypatch.setattr(python_runtime, "install_runtime", fake_install)
    r = runtime_client.post(  # type: ignore[attr-defined]
        "/api/runtime/python/install", json={"version": "3.11"}
    )
    assert r.status_code == 500
    assert "could not download" in r.json()["detail"]
