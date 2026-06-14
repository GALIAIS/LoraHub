from __future__ import annotations

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
