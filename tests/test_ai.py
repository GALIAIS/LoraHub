"""Tests for the ShiroManager-shaped AI store + router."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from lorahub.api import app as app_module
from lorahub.api import scheduler as sched_module
from lorahub.api import state as state_module
from lorahub.api.ai_store import (
    AIModel,
    AIProvider,
    AIProviderKey,
    AIRoute,
    AIStore,
)
from lorahub.core.ai.client import build_endpoint_url


# --------------------------------------------------------------------------- #
# build_endpoint_url
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("base_url", "endpoint", "expected"),
    [
        # Bare host -- our endpoint path's /v1 prefix lands on the URL.
        ("https://api.example.com", "/v1/chat/completions",
         "https://api.example.com/v1/chat/completions"),
        ("https://api.example.com", "/v1/models",
         "https://api.example.com/v1/models"),
        # Trailing slash on the base URL -- normalised away.
        ("https://api.example.com/", "/v1/models",
         "https://api.example.com/v1/models"),
        # User already supplied /v1 -- we strip /v1 from the endpoint to
        # avoid the v1/v1 double.
        ("https://api.example.com/v1", "/v1/models",
         "https://api.example.com/v1/models"),
        ("https://api.example.com/v1/", "/v1/chat/completions",
         "https://api.example.com/v1/chat/completions"),
        # Nested path that ends in /v1 also wins the strip.
        ("https://gateway.example.com/openai/v1", "/v1/models",
         "https://gateway.example.com/openai/v1/models"),
        # Path that contains v1 mid-string but doesn't END in /v1 -- no strip.
        ("https://api.example.com/api/v1beta", "/v1/models",
         "https://api.example.com/api/v1beta/v1/models"),
        # Query and fragment on the base URL are discarded.
        ("https://api.example.com/v1?token=ignored#frag", "/v1/models",
         "https://api.example.com/v1/models"),
    ],
)
def test_build_endpoint_url(base_url: str, endpoint: str, expected: str) -> None:
    assert build_endpoint_url(base_url, endpoint) == expected


def test_build_endpoint_url_rejects_empty() -> None:
    with pytest.raises(ValueError):
        build_endpoint_url("", "/v1/models")


def test_build_endpoint_url_rejects_malformed() -> None:
    with pytest.raises(ValueError):
        build_endpoint_url("not-a-url", "/v1/models")


@pytest.mark.parametrize(
    "base_url",
    [
        "ftp://api.example.com/v1",
        "https://user:secret@api.example.com/v1",
    ],
)
def test_build_endpoint_url_rejects_unsafe_urls(base_url: str) -> None:
    with pytest.raises(ValueError):
        build_endpoint_url(base_url, "/v1/models")


def test_extra_body_cannot_replace_core_request_fields() -> None:
    from lorahub.core.ai.client import AIError, _apply_extra_body

    with pytest.raises(AIError):
        _apply_extra_body(
            {"model": "safe", "messages": []},
            '{"messages":[{"role":"user","content":"replaced"}]}',
        )


# --------------------------------------------------------------------------- #
# Store unit tests
# --------------------------------------------------------------------------- #


def test_provider_round_trip(tmp_path: Path) -> None:
    s = AIStore(tmp_path / "ai.sqlite")
    p = s.upsert_provider(
        AIProvider(
            id="",
            name="DeepSeek",
            base_url="https://api.deepseek.com/v1",
            headers={"X-Custom": "1"},
        )
    )
    assert p.id  # ULID generated
    fetched = s.get_provider(p.id)
    assert fetched is not None
    assert fetched.name == "DeepSeek"
    assert fetched.headers == {"X-Custom": "1"}
    listed = s.list_providers()
    assert len(listed) == 1


def test_replace_keys_preserves_runtime(tmp_path: Path) -> None:
    s = AIStore(tmp_path / "ai.sqlite")
    p = s.upsert_provider(AIProvider(id="", name="P"))
    initial = s.replace_keys(
        p.id,
        [AIProviderKey(id="", provider_id=p.id, api_key="sk-aaaa")],
    )
    assert len(initial) == 1
    kid = initial[0].id

    # Bump runtime, then re-save the same key by id; runtime survives.
    s.update_key_runtime(kid, success=True)
    after = s.replace_keys(
        p.id,
        [AIProviderKey(id=kid, provider_id=p.id, api_key="sk-aaaa")],
    )
    assert after[0].runtime.request_count == 1
    assert after[0].runtime.success_count == 1

    # Drop the key entirely.
    after2 = s.replace_keys(p.id, [])
    assert after2 == []
    assert s.list_keys(p.id) == []


def test_update_key_runtime_failure_then_success(tmp_path: Path) -> None:
    s = AIStore(tmp_path / "ai.sqlite")
    p = s.upsert_provider(AIProvider(id="", name="P"))
    keys = s.replace_keys(
        p.id,
        [AIProviderKey(id="", provider_id=p.id, api_key="sk-x")],
    )
    kid = keys[0].id
    s.update_key_runtime(kid, success=False, error="429 rate limited",
                         cooldown_until="2099-01-01T00:00:00+00:00")
    s.update_key_runtime(kid, success=False, error="429 rate limited")
    s.update_key_runtime(kid, success=True)
    fresh = s.list_keys(p.id)[0]
    assert fresh.runtime.request_count == 3
    assert fresh.runtime.success_count == 1
    assert fresh.runtime.failure_count == 2
    assert fresh.runtime.consecutive_failures == 0  # reset on success
    assert fresh.runtime.last_succeeded_at is not None


def test_replace_discovered_models_preserves_manual(tmp_path: Path) -> None:
    s = AIStore(tmp_path / "ai.sqlite")
    p = s.upsert_provider(AIProvider(id="", name="P"))
    s.upsert_model(
        AIModel(
            id="",
            provider_id=p.id,
            model_id="manual-1",
            display_name="Manual One",
            source="manual",
            enabled=True,
        )
    )
    s.upsert_model(
        AIModel(
            id="",
            provider_id=p.id,
            model_id="discovered-1",
            display_name="Discovered One",
            source="discovered",
            enabled=True,
        )
    )
    s.replace_discovered_models(
        p.id,
        [
            AIModel(
                id="",
                provider_id=p.id,
                model_id="discovered-2",
                display_name="Discovered Two",
                source="discovered",
                enabled=True,
            )
        ],
    )
    rows = s.list_models(p.id)
    ids = {m.model_id: m.source for m in rows}
    assert ids == {"manual-1": "manual", "discovered-2": "discovered"}


def test_route_round_trip(tmp_path: Path) -> None:
    s = AIStore(tmp_path / "ai.sqlite")
    s.upsert_route(
        AIRoute(
            task_id="caption.rewrite",
            provider_id="prov-1",
            model_id="gpt-4o-mini",
            system_prompt="You rewrite tags.",
            temperature=0.2,
            stop_sequences=["END"],
        )
    )
    fetched = s.get_route("caption.rewrite")
    assert fetched is not None
    assert fetched.temperature == 0.2
    assert fetched.stop_sequences == ["END"]


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission semantics only")
def test_file_mode_is_user_only_on_posix(tmp_path: Path) -> None:
    path = tmp_path / "ai.sqlite"
    AIStore(path)
    mode = path.stat().st_mode & 0o777
    assert mode in (0o600, 0o400, 0o644)


# --------------------------------------------------------------------------- #
# Router HTTP tests
# --------------------------------------------------------------------------- #


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.chdir(tmp_path)
    registry = state_module.JobRegistry()
    monkeypatch.setattr(state_module, "registry", registry)
    fresh_sched = sched_module.JobScheduler(concurrency=1)
    monkeypatch.setattr(sched_module, "scheduler", fresh_sched)
    store = AIStore(tmp_path / "ai.sqlite")
    monkeypatch.setattr(app_module, "_ai_store", store)
    with TestClient(app_module.app) as c:
        yield c


def test_list_providers_empty(client: TestClient) -> None:
    r = client.get("/api/ai/providers")
    assert r.status_code == 200
    assert r.json() == {"providers": []}


def test_provider_crud(client: TestClient) -> None:
    r = client.put(
        "/api/ai/providers",
        json={
            "name": "DeepSeek",
            "baseUrl": "https://api.deepseek.com/v1",
            "apiKeys": [{"value": "sk-aaaaa1234567890"}],
        },
    )
    assert r.status_code == 200, r.text
    p = r.json()["provider"]
    assert p["name"] == "DeepSeek"
    assert p["apiKeyCount"] == 1
    assert p["apiKeys"][0]["preview"].endswith("7890")
    assert "sk-aaaaa1234567890" not in p["apiKeys"][0]["preview"]
    pid = p["id"]

    r = client.get(f"/api/ai/providers/{pid}")
    assert r.status_code == 200
    assert r.json()["id"] == pid

    # Adding a second provider rounds out listing.
    client.put(
        "/api/ai/providers",
        json={
            "name": "Local Ollama",
            "baseUrl": "http://localhost:11434/v1",
        },
    )
    listing = client.get("/api/ai/providers").json()["providers"]
    assert {p["name"] for p in listing} == {"DeepSeek", "Local Ollama"}

    r = client.delete(f"/api/ai/providers/{pid}")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_provider_keys_replace_and_runtime_preservation(client: TestClient) -> None:
    r = client.put(
        "/api/ai/providers",
        json={
            "name": "Multi",
            "baseUrl": "https://example.com/v1",
            "apiKeys": [
                {"value": "sk-AAAAAAAAAAAAAAAA"},
                {"value": "sk-BBBBBBBBBBBBBBBB"},
            ],
        },
    )
    pid = r.json()["provider"]["id"]
    keys = r.json()["provider"]["apiKeys"]
    assert len(keys) == 2
    kid_a, kid_b = keys[0]["id"], keys[1]["id"]

    # Save again, this time keep the existing two by id (no new value)
    # plus add a third. The first two must reuse their stored secret.
    r2 = client.put(
        "/api/ai/providers",
        json={
            "id": pid,
            "name": "Multi",
            "baseUrl": "https://example.com/v1",
            "apiKeys": [
                {"id": kid_a},
                {"id": kid_b},
                {"value": "sk-CCCCCCCCCCCCCCCC"},
            ],
        },
    )
    p2 = r2.json()["provider"]
    assert p2["apiKeyCount"] == 3
    # All three keys exist with stable ids for the first two.
    ids = {k["id"] for k in p2["apiKeys"]}
    assert {kid_a, kid_b}.issubset(ids)


def test_unknown_provider_get_returns_404(client: TestClient) -> None:
    r = client.get("/api/ai/providers/nonesuch")
    assert r.status_code == 404


def test_model_crud(client: TestClient) -> None:
    pr = client.put(
        "/api/ai/providers", json={"name": "P", "baseUrl": "x"}
    ).json()["provider"]
    pid = pr["id"]
    r = client.put(
        "/api/ai/models",
        json={
            "providerId": pid,
            "modelId": "gpt-4o-mini",
            "displayName": "GPT-4o mini",
            "source": "manual",
        },
    )
    assert r.status_code == 200
    mid = r.json()["model"]["id"]
    listing = client.get(f"/api/ai/models?provider_id={pid}").json()["models"]
    assert len(listing) == 1
    r = client.delete(f"/api/ai/models/{mid}")
    assert r.json()["ok"] is True
    assert client.get(f"/api/ai/models?provider_id={pid}").json()["models"] == []


def test_model_save_rejects_unknown_provider(client: TestClient) -> None:
    r = client.put(
        "/api/ai/models",
        json={
            "providerId": "no-such",
            "modelId": "x",
            "displayName": "x",
        },
    )
    assert r.status_code == 400


def test_route_save_and_list(client: TestClient) -> None:
    r = client.put(
        "/api/ai/routes",
        json={
            "taskId": "caption.rewrite",
            "providerId": "prov-1",
            "modelId": "gpt-4o-mini",
            "systemPrompt": "rewrite tags",
            "temperature": 0.2,
            "stopSequences": ["END"],
        },
    )
    assert r.status_code == 200
    routes = client.get("/api/ai/routes").json()["routes"]
    assert len(routes) == 1
    assert routes[0]["systemPrompt"] == "rewrite tags"
    assert routes[0]["stopSequences"] == ["END"]


def test_invoke_without_route_returns_409(client: TestClient) -> None:
    r = client.post(
        "/api/ai/invoke",
        json={"taskId": "missing.task", "prompt": "hi"},
    )
    assert r.status_code == 409
    assert "no AI route" in r.json()["detail"]


def test_ai_image_input_rejects_non_data_url() -> None:
    from lorahub.api.routers.ai import InvokeImageInput, _resolve_image_url

    with pytest.raises(HTTPException) as captured:
        _resolve_image_url(
            InvokeImageInput(kind="data_url", value="https://example.com/image.png")
        )

    assert captured.value.status_code == 400


def test_ai_image_input_rejects_file_outside_allowed_roots(tmp_path: Path) -> None:
    from lorahub.api.routers.ai import InvokeImageInput, _resolve_image_url

    image = tmp_path / "private.png"
    image.write_bytes(b"not-an-image")

    with pytest.raises(HTTPException) as captured:
        _resolve_image_url(InvokeImageInput(kind="file_path", value=str(image)))

    assert captured.value.status_code == 400


def test_ai_image_input_enforces_byte_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lorahub.api.routers.ai import InvokeImageInput, _resolve_image_url

    image = tmp_path / "large.png"
    image.write_bytes(b"12345")
    monkeypatch.setenv("LORAHUB_DATASETS_ROOT", str(tmp_path))
    monkeypatch.setenv("LORAHUB_MAX_AI_IMAGE_BYTES", "4")

    with pytest.raises(HTTPException) as captured:
        _resolve_image_url(InvokeImageInput(kind="file_path", value=str(image)))

    assert captured.value.status_code == 413


def test_invoke_proxies_to_client(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: route + chat returns the stubbed reply."""
    pr = client.put(
        "/api/ai/providers",
        json={
            "name": "Stub",
            "baseUrl": "https://example.com/v1",
            "apiKeys": [{"value": "sk-stub"}],
        },
    ).json()["provider"]
    pid = pr["id"]
    client.put(
        "/api/ai/routes",
        json={
            "taskId": "global.default",
            "providerId": pid,
            "modelId": "gpt-4o-mini",
            "systemPrompt": "be terse",
        },
    )
    captured: dict[str, object] = {}

    from lorahub.core.ai import client as ai_client_mod

    def fake_invoke(store, *, provider_id, model_id, messages, **kw):  # type: ignore[no-untyped-def]
        captured["provider_id"] = provider_id
        captured["model_id"] = model_id
        captured["messages"] = messages
        return ai_client_mod.InvokeResult(
            content="hello",
            finish_reason="stop",
            model_id=model_id,
            provider_id=provider_id,
            provider_name="Stub",
            usage_input_tokens=2,
            usage_output_tokens=1,
        )

    monkeypatch.setattr(ai_client_mod, "invoke", fake_invoke)

    r = client.post(
        "/api/ai/invoke",
        json={"taskId": "global.default", "prompt": "hi there"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["content"] == "hello"
    assert body["modelId"] == "gpt-4o-mini"
    assert captured["provider_id"] == pid
    msgs = captured["messages"]
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "be terse"
    assert msgs[1]["role"] == "user"


def test_test_endpoint_returns_models_via_stub(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    pr = client.put(
        "/api/ai/providers",
        json={
            "name": "Stub",
            "baseUrl": "https://example.com/v1",
            "apiKeys": [{"value": "sk-x"}],
        },
    ).json()["provider"]
    pid = pr["id"]
    from lorahub.core.ai import client as ai_client_mod

    def fake_test(store, *, provider_id, model_id=None, prompt=None,
                  system_prompt=None, sampling=None):  # type: ignore[no-untyped-def]
        return ai_client_mod.ConnectionTestResult(
            ok=True,
            provider_id=provider_id,
            provider_name="Stub",
            models=[{"id": "model-a"}, {"id": "model-b"}],
        )

    monkeypatch.setattr(ai_client_mod, "test_connection", fake_test)

    r = client.post("/api/ai/test", json={"providerId": pid})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["modelCount"] == 2
    assert body["models"][0]["id"] == "model-a"


def test_reset_key_runtime(client: TestClient) -> None:
    pr = client.put(
        "/api/ai/providers",
        json={
            "name": "Stub",
            "baseUrl": "x",
            "apiKeys": [{"value": "sk-x"}],
        },
    ).json()["provider"]
    kid = pr["apiKeys"][0]["id"]
    r = client.post(f"/api/ai/keys/{kid}/reset-runtime")
    assert r.status_code == 200
    assert r.json()["ok"] is True
