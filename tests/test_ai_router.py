"""Tests for the AI provider catalogue + credential CRUD + chat proxy."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lorahub.api import app as app_module
from lorahub.api import scheduler as sched_module
from lorahub.api import state as state_module
from lorahub.api.ai_credentials_store import AICredential, AICredentialStore


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Per-test SQLite store + clean registry + scheduler."""
    monkeypatch.chdir(tmp_path)
    registry = state_module.JobRegistry()
    monkeypatch.setattr(state_module, "registry", registry)
    fresh_sched = sched_module.JobScheduler(concurrency=1)
    monkeypatch.setattr(sched_module, "scheduler", fresh_sched)

    store = AICredentialStore(tmp_path / "ai_creds.sqlite")
    monkeypatch.setattr(app_module, "_ai_credentials_store", store)

    with TestClient(app_module.app) as c:
        yield c


def test_get_providers_returns_catalogue(client: TestClient) -> None:
    r = client.get("/api/ai/providers")
    assert r.status_code == 200
    body = r.json()
    ids = {p["id"] for p in body["providers"]}
    assert {
        "openai",
        "anthropic",
        "google",
        "deepseek",
        "qwen",
        "kimi",
        "glm",
        "doubao",
        "openai_compat",
    } <= ids
    # Every entry is unconfigured until the user saves a credential.
    assert all(p["configured"] is False for p in body["providers"])


def test_get_providers_marks_configured_after_upsert(client: TestClient) -> None:
    r = client.put(
        "/api/ai/credentials",
        json={"provider": "deepseek", "api_key": "sk-test-1234567890"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    cred = body["credential"]
    assert cred["provider"] == "deepseek"
    assert cred["api_key_set"] is True
    # api_key on read is masked.
    assert cred["api_key"] != "sk-test-1234567890"
    assert "..." in (cred["api_key"] or "")

    listing = client.get("/api/ai/providers").json()["providers"]
    deepseek = next(p for p in listing if p["id"] == "deepseek")
    assert deepseek["configured"] is True
    assert deepseek["enabled"] is True


def test_upsert_unknown_provider_returns_400(client: TestClient) -> None:
    r = client.put(
        "/api/ai/credentials",
        json={"provider": "no-such-provider", "api_key": "x"},
    )
    assert r.status_code == 400
    assert "unknown provider" in r.json()["detail"]


def test_credentials_listing_round_trips(client: TestClient) -> None:
    client.put(
        "/api/ai/credentials",
        json={"provider": "openai", "api_key": "sk-abcdef-9876543210"},
    )
    client.put(
        "/api/ai/credentials",
        json={
            "provider": "qwen",
            "api_key": "sk-qwen-key",
            "default_model": "qwen-plus",
            "enabled": False,
        },
    )
    r = client.get("/api/ai/credentials")
    assert r.status_code == 200
    creds = {c["provider"]: c for c in r.json()["credentials"]}
    assert creds["openai"]["api_key_set"] is True
    assert creds["openai"]["api_key"] != "sk-abcdef-9876543210"
    assert creds["qwen"]["enabled"] is False
    assert creds["qwen"]["default_model"] == "qwen-plus"


def test_delete_credential(client: TestClient) -> None:
    client.put(
        "/api/ai/credentials",
        json={"provider": "openai", "api_key": "sk-x"},
    )
    r = client.delete("/api/ai/credentials/openai")
    assert r.status_code == 200
    assert r.json()["deleted"] is True
    listing = client.get("/api/ai/credentials").json()["credentials"]
    assert all(c["provider"] != "openai" for c in listing)


def test_chat_with_unconfigured_provider_returns_404(client: TestClient) -> None:
    r = client.post(
        "/api/ai/chat",
        json={
            "provider": "openai",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code == 404
    assert "not configured" in r.json()["detail"]


def test_chat_with_disabled_provider_returns_409(client: TestClient) -> None:
    client.put(
        "/api/ai/credentials",
        json={"provider": "openai", "api_key": "sk-x", "enabled": False},
    )
    r = client.post(
        "/api/ai/chat",
        json={
            "provider": "openai",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code == 409
    assert "disabled" in r.json()["detail"]


def test_chat_invalid_role_returns_400(client: TestClient) -> None:
    client.put(
        "/api/ai/credentials",
        json={"provider": "openai", "api_key": "sk-x"},
    )
    r = client.post(
        "/api/ai/chat",
        json={
            "provider": "openai",
            "messages": [{"role": "robot", "content": "hi"}],
        },
    )
    assert r.status_code == 400
    assert "invalid message role" in r.json()["detail"]


def test_chat_proxies_to_provider(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: a chat call routes through the dispatcher and returns
    the provider's text. The provider is stubbed so no network IO happens."""
    from lorahub.core.ai import provider_base
    from lorahub.core.ai.providers import deepseek as deepseek_module

    captured: dict[str, object] = {}

    def fake_chat(self, messages, options):  # type: ignore[no-untyped-def]
        captured["messages"] = messages
        captured["options"] = options
        return provider_base.ChatResult(
            text="stubbed reply",
            model="deepseek-chat",
            finish_reason="stop",
            usage_input_tokens=4,
            usage_output_tokens=2,
        )

    monkeypatch.setattr(deepseek_module.DeepSeekProvider, "chat", fake_chat)

    client.put(
        "/api/ai/credentials",
        json={"provider": "deepseek", "api_key": "sk-x"},
    )
    r = client.post(
        "/api/ai/chat",
        json={
            "provider": "deepseek",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["text"] == "stubbed reply"
    assert body["model"] == "deepseek-chat"
    assert body["usage_input_tokens"] == 4
    assert captured["messages"][0].role == "user"


def test_test_endpoint_returns_ok_on_success(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lorahub.core.ai import provider_base
    from lorahub.core.ai.providers import openai as openai_module

    def fake_chat(self, messages, options):  # type: ignore[no-untyped-def]
        return provider_base.ChatResult(
            text="ok",
            model="gpt-4o-mini",
            finish_reason="stop",
            usage_input_tokens=1,
            usage_output_tokens=1,
        )

    monkeypatch.setattr(openai_module.OpenAIProvider, "chat", fake_chat)
    r = client.post(
        "/api/ai/test",
        json={"provider": "openai", "api_key": "sk-x"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["model"] == "gpt-4o-mini"


def test_test_endpoint_surfaces_provider_error(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from lorahub.core.ai import provider_base
    from lorahub.core.ai.providers import openai as openai_module

    def boom(self, messages, options):  # type: ignore[no-untyped-def]
        raise provider_base.ProviderError(
            "401 invalid key",
            provider="openai",
            status_code=401,
        )

    monkeypatch.setattr(openai_module.OpenAIProvider, "chat", boom)
    r = client.post(
        "/api/ai/test",
        json={"provider": "openai", "api_key": "sk-bad"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert body["status_code"] == 401
    assert "401" in body["error"]
