from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lorahub.api import auth
from lorahub.api.auth import RemoteAccessMiddleware, router


def _app(*, allowed_origins: list[str] | None = None) -> FastAPI:
    app = FastAPI()
    app.add_middleware(
        RemoteAccessMiddleware,
        allowed_origins=allowed_origins or [],
    )
    app.include_router(router)

    @app.get("/api/private")
    def private() -> dict[str, bool]:
        return {"ok": True}

    return app


def test_remote_api_requires_configured_token(monkeypatch) -> None:
    monkeypatch.delenv("LORAHUB_API_TOKEN", raising=False)
    with TestClient(
        _app(),
        base_url="http://server.example",
        client=("203.0.113.10", 50000),
    ) as client:
        response = client.get("/api/private")
    assert response.status_code == 403


def test_remote_session_cookie_authenticates_http(monkeypatch) -> None:
    monkeypatch.setenv("LORAHUB_API_TOKEN", "test-secret")
    with TestClient(
        _app(),
        base_url="http://server.example",
        client=("203.0.113.10", 50000),
    ) as client:
        assert client.get("/api/private").status_code == 401
        login = client.post(
            "/api/auth/session",
            data={"token": "test-secret", "next": "/"},
            follow_redirects=False,
        )
        assert login.status_code == 303
        assert client.get("/api/private").json() == {"ok": True}


def test_remote_login_page_matches_app_theme_and_labels_token(monkeypatch) -> None:
    monkeypatch.setenv("LORAHUB_API_TOKEN", "test-secret")
    with TestClient(
        _app(),
        base_url="http://server.example",
        client=("203.0.113.10", 50000),
    ) as client:
        response = client.get("/api/auth/session")

    assert response.status_code == 200
    assert "lorahub.theme.mode" in response.text
    assert "lorahub.ui.style.v2" in response.text
    assert "class=auth-panel" in response.text
    assert "<label for=access-token>访问令牌</label>" in response.text
    assert "aria-describedby=token-help" in response.text


def test_remote_page_login_preserves_original_path_and_query(monkeypatch) -> None:
    monkeypatch.setenv("LORAHUB_API_TOKEN", "test-secret")
    with TestClient(
        _app(),
        base_url="http://server.example",
        client=("203.0.113.10", 50000),
    ) as client:
        response = client.get("/jobs/active?tab=logs&tail=200", follow_redirects=False)

    assert response.status_code == 303
    assert response.headers["location"] == (
        "/api/auth/session?next=%2Fjobs%2Factive%3Ftab%3Dlogs%26tail%3D200"
    )


def test_session_login_rejects_backslash_redirect(monkeypatch) -> None:
    monkeypatch.setenv("LORAHUB_API_TOKEN", "test-secret")
    with TestClient(
        _app(),
        base_url="http://server.example",
        client=("203.0.113.10", 50000),
    ) as client:
        response = client.post(
            "/api/auth/session",
            data={"token": "test-secret", "next": r"/\evil.example"},
            follow_redirects=False,
        )

    assert response.status_code == 303
    assert response.headers["location"] == "/"


def test_remote_bearer_authenticates_api(monkeypatch) -> None:
    monkeypatch.setenv("LORAHUB_API_TOKEN", "test-secret")
    with TestClient(
        _app(),
        base_url="http://server.example",
        client=("203.0.113.10", 50000),
    ) as client:
        response = client.get(
            "/api/private",
            headers={"Authorization": "Bearer test-secret"},
        )
    assert response.status_code == 200


def test_local_api_remains_usable_without_token(monkeypatch) -> None:
    monkeypatch.delenv("LORAHUB_API_TOKEN", raising=False)
    with TestClient(
        _app(),
        base_url="http://127.0.0.1:18765",
        client=("127.0.0.1", 50000),
    ) as client:
        response = client.get("/api/private")
    assert response.status_code == 200


def test_loopback_browser_origin_cannot_bypass_auth(monkeypatch) -> None:
    monkeypatch.delenv("LORAHUB_API_TOKEN", raising=False)
    with TestClient(
        _app(),
        base_url="http://127.0.0.1:18765",
        client=("127.0.0.1", 50000),
    ) as client:
        response = client.post(
            "/api/private",
            headers={"Origin": "https://attacker.example"},
        )

    assert response.status_code == 403


def test_loopback_same_origin_browser_remains_usable(monkeypatch) -> None:
    monkeypatch.delenv("LORAHUB_API_TOKEN", raising=False)
    with TestClient(
        _app(),
        base_url="http://127.0.0.1:18765",
        client=("127.0.0.1", 50000),
    ) as client:
        response = client.get(
            "/api/private",
            headers={"Origin": "http://127.0.0.1:18765"},
        )

    assert response.status_code == 200


def test_explicit_local_dev_origin_remains_usable(monkeypatch) -> None:
    monkeypatch.delenv("LORAHUB_API_TOKEN", raising=False)
    with TestClient(
        _app(allowed_origins=["http://localhost:6006"]),
        base_url="http://127.0.0.1:18765",
        client=("127.0.0.1", 50000),
    ) as client:
        response = client.get(
            "/api/private",
            headers={"Origin": "http://localhost:6006"},
        )

    assert response.status_code == 200


def test_reverse_proxy_headers_disable_loopback_bypass(monkeypatch) -> None:
    monkeypatch.delenv("LORAHUB_API_TOKEN", raising=False)
    with TestClient(
        _app(),
        base_url="http://127.0.0.1:18765",
        client=("127.0.0.1", 50000),
    ) as client:
        response = client.get(
            "/api/private",
            headers={"X-Forwarded-For": "203.0.113.10"},
        )

    assert response.status_code == 403


def test_generated_api_token_is_persistent(
    tmp_path: Path, monkeypatch,
) -> None:
    token_path = tmp_path / "api-token"
    monkeypatch.delenv("LORAHUB_API_TOKEN", raising=False)
    monkeypatch.setattr(auth, "_token_path", lambda: token_path)

    first = auth.ensure_api_token()
    os.environ.pop("LORAHUB_API_TOKEN", None)
    second = auth.ensure_api_token()

    assert first == second
    assert token_path.read_text(encoding="utf-8") == first
    os.environ.pop("LORAHUB_API_TOKEN", None)


def test_empty_api_token_file_is_recovered(
    tmp_path: Path, monkeypatch,
) -> None:
    token_path = tmp_path / "api-token"
    token_path.write_text("", encoding="utf-8")
    monkeypatch.delenv("LORAHUB_API_TOKEN", raising=False)
    monkeypatch.setattr(auth, "_token_path", lambda: token_path)
    monkeypatch.setattr(auth.time, "sleep", lambda _seconds: None)

    token = auth.ensure_api_token()

    assert token
    assert token_path.read_text(encoding="utf-8") == token
    os.environ.pop("LORAHUB_API_TOKEN", None)


def test_api_token_file_rejects_symlink(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "outside-token"
    target.write_text("outside-secret", encoding="utf-8")
    token_path = tmp_path / "api-token"
    try:
        token_path.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is unavailable")
    monkeypatch.delenv("LORAHUB_API_TOKEN", raising=False)
    monkeypatch.setattr(auth, "_token_path", lambda: token_path)

    with pytest.raises(RuntimeError, match="linked API token"):
        auth.ensure_api_token()

    assert target.read_text(encoding="utf-8") == "outside-secret"
    assert auth.api_auth_headers() == {}
