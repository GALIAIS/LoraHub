from __future__ import annotations

import sys
import threading
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from lorahub.core import net


def _empty_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    from lorahub.api import app as app_module

    monkeypatch.setattr(
        app_module._settings_store,
        "load",
        lambda: SimpleNamespace(huggingface_endpoint=None, download_proxy=None),
    )


def test_hf_endpoint_prefers_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    _empty_settings(monkeypatch)
    monkeypatch.setenv("HF_ENDPOINT", "https://env.example")

    assert net.hf_endpoint("https://explicit.example/") == "https://explicit.example"


def test_hf_endpoint_falls_back_to_legacy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _empty_settings(monkeypatch)
    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.setenv("HUGGINGFACE_HUB_ENDPOINT", "https://legacy.example/")

    assert net.hf_endpoint() == "https://legacy.example"


def test_hf_endpoint_uses_settings_after_env(monkeypatch: pytest.MonkeyPatch) -> None:
    from lorahub.api import app as app_module

    monkeypatch.delenv("HF_ENDPOINT", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_ENDPOINT", raising=False)
    monkeypatch.setattr(
        app_module._settings_store,
        "load",
        lambda: SimpleNamespace(
            huggingface_endpoint="https://settings.example/",
            download_proxy=None,
        ),
    )

    assert net.hf_endpoint() == "https://settings.example"


def test_hf_download_uses_project_cache_when_hf_home_is_bad(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bad_home = tmp_path / "not-a-dir"
    bad_home.write_text("x", encoding="utf-8")
    monkeypatch.setenv("HF_HOME", str(bad_home))
    monkeypatch.setattr(net, "project_root", lambda: tmp_path / "lorahub")
    seen: dict[str, object] = {}

    def fake_download(**kw: object) -> str:
        seen.update(kw)
        return "ok"

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(hf_hub_download=fake_download),
    )

    assert net.hf_download("repo/model", "model.onnx") == "ok"
    assert seen["cache_dir"] == str(tmp_path / "lorahub" / "models" / "huggingface" / "hub")


def test_hf_download_keeps_explicit_cache_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    explicit = tmp_path / "explicit-cache"
    seen: dict[str, object] = {}

    def fake_download(**kw: object) -> str:
        seen.update(kw)
        return "ok"

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(hf_hub_download=fake_download),
    )

    assert net.hf_download("repo/model", "model.onnx", cache_dir=str(explicit)) == "ok"
    assert seen["cache_dir"] == str(explicit)


def test_download_preferences_resolve_modelscope_settings_and_env_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lorahub.api import app as app_module

    monkeypatch.setenv("MODELSCOPE_API_TOKEN", "env-token")
    monkeypatch.setattr(
        app_module._settings_store,
        "load",
        lambda: SimpleNamespace(
            modelscope_enabled=True,
            modelscope_token="saved-token",
            download_proxy="http://proxy.example",
        ),
    )

    preferences = net.download_preferences()

    assert preferences.prefer_modelscope is True
    assert preferences.modelscope_token == "env-token"
    assert preferences.proxy == "http://proxy.example"


def test_proxy_env_restores_previous_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://old")
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.delenv("ALL_PROXY", raising=False)

    with net.proxy_env("http://new"):
        assert net.os.environ["HTTPS_PROXY"] == "http://new"
        assert net.os.environ["HTTP_PROXY"] == "http://new"
        assert net.os.environ["ALL_PROXY"] == "http://new"

    assert net.os.environ["HTTPS_PROXY"] == "http://old"
    assert "HTTP_PROXY" not in net.os.environ
    assert "ALL_PROXY" not in net.os.environ


def test_proxy_env_serializes_process_environment_mutations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://original")
    first_entered = threading.Event()
    release_first = threading.Event()
    second_entered = threading.Event()
    observed: list[str] = []

    def first() -> None:
        with net.proxy_env("http://first"):
            observed.append(net.os.environ["HTTPS_PROXY"])
            first_entered.set()
            assert release_first.wait(2)

    def second() -> None:
        assert first_entered.wait(2)
        with net.proxy_env("http://second"):
            observed.append(net.os.environ["HTTPS_PROXY"])
            second_entered.set()

    first_thread = threading.Thread(target=first)
    second_thread = threading.Thread(target=second)
    first_thread.start()
    assert first_entered.wait(2)
    second_thread.start()
    assert not second_entered.wait(0.1)
    release_first.set()
    first_thread.join(2)
    second_thread.join(2)

    assert observed == ["http://first", "http://second"]
    assert net.os.environ["HTTPS_PROXY"] == "http://original"


def test_subprocess_env_does_not_wait_for_or_inherit_scoped_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://original")
    monkeypatch.setenv("HTTP_PROXY", "http://original")
    monkeypatch.delenv("ALL_PROXY", raising=False)
    entered = threading.Event()
    release = threading.Event()

    def download_scope() -> None:
        with net.proxy_env("http://download-secret:password@proxy.invalid"):
            entered.set()
            assert release.wait(2)

    worker = threading.Thread(target=download_scope)
    worker.start()
    assert entered.wait(2)

    started = time.monotonic()
    env = net.subprocess_env()
    elapsed = time.monotonic() - started
    release.set()
    worker.join(2)

    assert elapsed < 0.5
    assert env["HTTPS_PROXY"] == "http://original"
    assert env["HTTP_PROXY"] == "http://original"
    assert "ALL_PROXY" not in env


def test_subprocess_env_overrides_bad_hf_cache_vars(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("HF_HOME", "F:\\missing")
    monkeypatch.setenv("HF_HUB_CACHE", "F:\\missing\\hub")
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", "F:\\missing\\hub")
    monkeypatch.setattr(net, "project_root", lambda: tmp_path / "lorahub")

    env = net.subprocess_env()

    assert env["HF_HOME"] == str(tmp_path / "lorahub" / "models" / "huggingface")
    expected_hub = str(tmp_path / "lorahub" / "models" / "huggingface" / "hub")
    assert env["HF_HUB_CACHE"] == expected_hub
    assert env["HUGGINGFACE_HUB_CACHE"] == expected_hub
