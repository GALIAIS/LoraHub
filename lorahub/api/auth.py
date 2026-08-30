"""Remote-access authentication for the HTTP and WebSocket surface."""

from __future__ import annotations

import hmac
import html
import ipaddress
import os
import secrets
import stat
import time
from contextlib import suppress
from http.cookies import CookieError, SimpleCookie
from pathlib import Path
from typing import Annotated
from urllib.parse import urlencode

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.datastructures import Headers
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Receive, Scope, Send

from lorahub.api.runtime_bind import state_dir

API_TOKEN_ENV = "LORAHUB_API_TOKEN"
COOKIE_NAME = "lorahub_access"
router = APIRouter(prefix="/api")


def _token_path() -> Path:
    return state_dir() / "api-token"


def api_token_path() -> Path:
    """Return the persistent token path without reading its secret value."""
    return _token_path()


def configured_api_token() -> str | None:
    return os.environ.get(API_TOKEN_ENV, "").strip() or None


def _is_link_like(path: Path) -> bool:
    """Return true for a symlink or Windows junction/reparse directory."""
    try:
        if path.is_symlink():
            return True
        is_junction = getattr(path, "is_junction", None)
        return bool(is_junction is not None and is_junction())
    except OSError:
        return True


def _read_token_file(path: Path) -> str:
    """Read a small regular token file without following Unix symlinks."""
    if _is_link_like(path):
        raise RuntimeError(f"refusing to read linked API token file: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise RuntimeError(f"API token path is not a regular file: {path}")
        with os.fdopen(fd, "r", encoding="utf-8") as handle:
            fd = -1
            value = handle.read(4097)
    finally:
        if fd >= 0:
            os.close(fd)
    if len(value) > 4096:
        raise RuntimeError(f"API token file is unexpectedly large: {path}")
    return value.strip()


def ensure_api_token() -> str:
    """Return a stable per-user token and expose it to the API process."""
    token = configured_api_token()
    if token:
        return token

    path = _token_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with suppress(OSError):
        path.parent.chmod(0o700)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    for attempt in range(21):
        try:
            token = _read_token_file(path)
        except FileNotFoundError:
            token = ""
        except RuntimeError:
            raise
        except OSError as exc:
            if attempt < 20:
                time.sleep(0.05)
                continue
            raise RuntimeError(f"could not read API token file at {path}: {exc}") from exc
        if token:
            with suppress(OSError):
                path.chmod(0o600)
            os.environ[API_TOKEN_ENV] = token
            return token

        if path.exists():
            if _is_link_like(path):
                raise RuntimeError(f"refusing to replace linked API token file: {path}")
            if attempt < 20:
                time.sleep(0.05)
                continue
            with suppress(OSError):
                path.unlink()

        token = secrets.token_urlsafe(32)
        try:
            fd = os.open(path, flags, 0o600)
        except FileExistsError:
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(token)
        with suppress(OSError):
            path.chmod(0o600)
        os.environ[API_TOKEN_ENV] = token
        return token

    raise RuntimeError(f"could not create a non-empty API token at {path}")


def api_auth_headers() -> dict[str, str]:
    """Headers for local CLI clients talking to a protected daemon."""
    token = configured_api_token()
    if token is None:
        try:
            token = _read_token_file(_token_path()) or None
        except (OSError, RuntimeError):
            token = None
    return {"Authorization": f"Bearer {token}"} if token else {}


def is_loopback_host(host: str) -> bool:
    value = host.strip().lower().strip("[]")
    if value == "localhost":
        return True
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def _host_name(headers: Headers) -> str:
    value = headers.get("host", "")
    if value.startswith("["):
        end = value.find("]")
        return value[1:end] if end > 0 else value
    return value.rsplit(":", 1)[0] if value.count(":") == 1 else value


def _origin_allows_local_bypass(
    scope: Scope,
    headers: Headers,
    allowed_origins: frozenset[str],
) -> bool:
    origin = headers.get("origin", "").strip().rstrip("/")
    if not origin:
        # Native clients such as the CLI/curl do not send Origin.
        return True
    if origin in allowed_origins:
        return True
    scheme = str(scope.get("scheme") or "http").lower()
    authority = headers.get("host", "").strip().lower().rstrip("/")
    return bool(authority and origin.lower() == f"{scheme}://{authority}")


def _is_local_scope(
    scope: Scope,
    headers: Headers,
    allowed_origins: frozenset[str] = frozenset(),
) -> bool:
    if any(
        headers.get(name)
        for name in ("forwarded", "x-forwarded-for", "x-forwarded-host", "x-real-ip")
    ):
        return False
    client = scope.get("client")
    client_host = str(client[0]) if client else ""
    if client_host == "testclient":
        return _origin_allows_local_bypass(scope, headers, allowed_origins)
    return (
        is_loopback_host(client_host)
        and is_loopback_host(_host_name(headers))
        and _origin_allows_local_bypass(scope, headers, allowed_origins)
    )


def _provided_token(headers: Headers) -> str | None:
    authorization = headers.get("authorization", "")
    scheme, _, value = authorization.partition(" ")
    if scheme.lower() == "bearer" and value.strip():
        return value.strip()
    direct = headers.get("x-lorahub-token", "").strip()
    if direct:
        return direct
    raw_cookie = headers.get("cookie", "")
    if raw_cookie:
        cookie = SimpleCookie()
        try:
            cookie.load(raw_cookie)
        except CookieError:
            return None
        morsel = cookie.get(COOKIE_NAME)
        if morsel is not None:
            return morsel.value
    return None


def _valid_token(provided: str | None, expected: str | None) -> bool:
    return bool(provided and expected and hmac.compare_digest(provided, expected))


class RemoteAccessMiddleware:
    """Require authentication whenever the request is not strictly local."""

    def __init__(
        self,
        app: ASGIApp,
        allowed_origins: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        self.app = app
        self.allowed_origins = frozenset(
            origin.strip().rstrip("/")
            for origin in (allowed_origins or ())
            if origin.strip()
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        path = str(scope.get("path", ""))
        token = configured_api_token()
        local = _is_local_scope(scope, headers, self.allowed_origins)
        if path == "/api/auth/session" or (
            scope["type"] == "http" and scope.get("method") == "OPTIONS"
        ):
            await self.app(scope, receive, send)
            return
        if token is None and local:
            await self.app(scope, receive, send)
            return
        if path == "/api/health" and local:
            await self.app(scope, receive, send)
            return
        if _valid_token(_provided_token(headers), token):
            await self.app(scope, receive, send)
            return

        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 4401 if token else 4403})
            return
        if path.startswith("/api/"):
            detail = (
                "remote access requires authentication"
                if token
                else "remote access is disabled; set LORAHUB_API_TOKEN or use an SSH tunnel"
            )
            response: Response = JSONResponse(
                {"detail": detail}, status_code=401 if token else 403
            )
        elif token:
            query = bytes(scope.get("query_string", b"")).decode("latin-1")
            next_path = f"{path}?{query}" if query else path
            response = RedirectResponse(
                f"/api/auth/session?{urlencode({'next': next_path})}",
                status_code=303,
            )
        else:
            response = HTMLResponse(
                "Remote access is disabled. Set LORAHUB_API_TOKEN or use an SSH tunnel.",
                status_code=403,
            )
        await response(scope, receive, send)


def _safe_next(value: str) -> str:
    if (
        value.startswith("/")
        and not value.startswith("//")
        and "\\" not in value
        and "\r" not in value
        and "\n" not in value
    ):
        return value
    return "/"


_LOGIN_PAGE_STYLE = """
:root {
  color-scheme: light;
  --background: oklch(0.982 0.002 95);
  --foreground: oklch(0.195 0.008 255);
  --card: oklch(0.995 0.002 95);
  --muted-foreground: oklch(0.505 0.01 255);
  --primary: oklch(0.165 0.012 255);
  --primary-foreground: oklch(0.985 0.001 95);
  --border: oklch(0.865 0.005 255);
  --control-fill: oklch(0.985 0.002 95);
  --ring: oklch(0.575 0.016 250);
  --danger: oklch(0.52 0.2 27);
  --danger-fill: oklch(0.965 0.025 27);
  --radius: 6px;
  --panel-shadow: 0 18px 40px -30px rgb(15 23 42 / 0.22);
}

:root.dark {
  color-scheme: dark;
  --background: oklch(0.145 0.008 255);
  --foreground: oklch(0.965 0.004 95);
  --card: oklch(0.182 0.01 255);
  --muted-foreground: oklch(0.72 0.008 255);
  --primary: oklch(0.955 0.004 95);
  --primary-foreground: oklch(0.16 0.008 255);
  --border: oklch(0.32 0.01 255);
  --control-fill: oklch(0.205 0.01 255);
  --ring: oklch(0.72 0.02 246);
  --danger: oklch(0.76 0.16 22);
  --danger-fill: oklch(0.24 0.06 22);
  --panel-shadow: 0 22px 46px -30px rgb(0 0 0 / 0.66);
}

:root[data-ui-style="linear"] {
  --background: oklch(0.972 0.003 285);
  --foreground: oklch(0.205 0.008 285);
  --card: oklch(0.995 0.002 285);
  --muted-foreground: oklch(0.49 0.01 285);
  --primary: oklch(0.57 0.18 278);
  --primary-foreground: oklch(0.985 0.003 285);
  --border: oklch(0.885 0.005 285);
  --control-fill: oklch(0.952 0.004 285);
  --ring: oklch(0.61 0.17 278);
  --radius: 8px;
  --panel-shadow: 0 18px 48px -24px rgb(20 20 24 / 0.28), 0 1px 3px rgb(20 20 24 / 0.08);
}

:root.dark[data-ui-style="linear"] {
  --background: oklch(0.135 0.004 285);
  --foreground: oklch(0.94 0.004 285);
  --card: oklch(0.162 0.004 285);
  --muted-foreground: oklch(0.68 0.008 285);
  --primary: oklch(0.72 0.14 278);
  --primary-foreground: oklch(0.145 0.01 285);
  --border: oklch(0.265 0.006 285);
  --control-fill: oklch(0.205 0.005 285);
  --ring: oklch(0.72 0.14 278);
  --panel-shadow: 0 20px 52px -24px rgb(0 0 0 / 0.72), 0 1px 3px rgb(0 0 0 / 0.48);
}

* { box-sizing: border-box; }

html, body { min-height: 100%; }

body {
  margin: 0;
  min-height: 100dvh;
  display: grid;
  place-items: center;
  padding: 24px;
  background: var(--background);
  color: var(--foreground);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  font-size: 14px;
  -webkit-font-smoothing: antialiased;
  text-rendering: optimizeLegibility;
}

.auth-shell { width: min(100%, 400px); }

.brand {
  margin-bottom: 14px;
  font-size: 15px;
  font-weight: 650;
}

.auth-panel {
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--card);
  padding: 26px;
  box-shadow: var(--panel-shadow);
}

h1 {
  margin: 0;
  font-size: 21px;
  line-height: 1.3;
  font-weight: 650;
}

.description {
  margin: 8px 0 22px;
  color: var(--muted-foreground);
  line-height: 1.65;
}

.field { display: grid; gap: 7px; }

label {
  font-size: 13px;
  font-weight: 550;
}

input {
  width: 100%;
  height: 40px;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  outline: 0;
  background: var(--control-fill);
  color: var(--foreground);
  padding: 0 12px;
  font: inherit;
  transition: border-color 150ms ease, box-shadow 150ms ease, background-color 150ms ease;
}

input:hover { border-color: color-mix(in oklab, var(--foreground) 20%, var(--border)); }

input:focus-visible {
  border-color: var(--ring);
  box-shadow: 0 0 0 3px color-mix(in oklab, var(--ring) 28%, transparent);
}

button {
  width: 100%;
  height: 40px;
  margin-top: 14px;
  border: 1px solid var(--primary);
  border-radius: var(--radius);
  background: var(--primary);
  color: var(--primary-foreground);
  font: inherit;
  font-weight: 600;
  cursor: pointer;
  transition: opacity 150ms ease, transform 150ms ease, box-shadow 150ms ease;
}

button:hover { opacity: 0.92; }
button:active { opacity: 0.84; }
button:focus-visible {
  outline: 0;
  box-shadow: 0 0 0 3px color-mix(in oklab, var(--ring) 34%, transparent);
}

.auth-error {
  margin: 0 0 16px;
  border: 1px solid color-mix(in oklab, var(--danger) 32%, transparent);
  border-radius: var(--radius);
  background: var(--danger-fill);
  color: var(--danger);
  padding: 9px 11px;
  line-height: 1.5;
}

@media (max-width: 480px) {
  body { padding: 16px; }
  .auth-panel { padding: 22px 18px; }
}

@media (prefers-reduced-motion: reduce) {
  input, button { transition: none; }
}
"""

_LOGIN_THEME_SCRIPT = """
(() => {
  try {
    const mode = localStorage.getItem("lorahub.theme.mode") || "system";
    const dark = mode === "dark" ||
      (mode === "system" && matchMedia("(prefers-color-scheme: dark)").matches);
    document.documentElement.classList.toggle("dark", dark);
    const style = localStorage.getItem("lorahub.ui.style.v3");
    const previous = localStorage.getItem("lorahub.ui.style.v2");
    document.documentElement.dataset.uiStyle = style === "linear" || style === "shiro"
      ? style
      : previous && previous !== "shiro" ? "linear" : "shiro";
  } catch {}
})();
"""


def _login_page(next_path: str, *, error: str = "", status_code: int = 200) -> HTMLResponse:
    safe_next = html.escape(_safe_next(next_path), quote=True)
    message = (
        f'<div class="auth-error" role="alert">{html.escape(error)}</div>'
        if error
        else ""
    )
    return HTMLResponse(
        "<!doctype html><html lang=zh-CN><head><meta charset=utf-8>"
        "<meta name=viewport content='width=device-width,initial-scale=1'>"
        "<title>登录 · LoRaHub</title>"
        f"<script>{_LOGIN_THEME_SCRIPT}</script>"
        f"<style>{_LOGIN_PAGE_STYLE}</style></head><body>"
        "<main class=auth-shell>"
        "<div class=brand>LoRaHub</div>"
        "<section class=auth-panel aria-labelledby=login-title>"
        "<h1 id=login-title>远程访问</h1>"
        "<p class=description id=token-help>输入启动日志中显示的访问令牌。</p>"
        f"{message}<form method=post>"
        f"<input type=hidden name=next value='{safe_next}'>"
        "<div class=field><label for=access-token>访问令牌</label>"
        "<input id=access-token name=token type=password required autofocus "
        "autocomplete=current-password aria-describedby=token-help></div>"
        "<button type=submit>登录</button>"
        "</form></section></main></body></html>",
        status_code=status_code,
    )


@router.get("/auth/session", response_class=HTMLResponse, include_in_schema=False)
def login_page(next: str = "/") -> HTMLResponse:
    if configured_api_token() is None:
        return HTMLResponse("Remote access is not configured.", status_code=403)
    return _login_page(next)


@router.post("/auth/session", include_in_schema=False)
def create_session(
    request: Request,
    token: Annotated[str, Form()],
    next: Annotated[str, Form()] = "/",
) -> Response:
    expected = configured_api_token()
    if not _valid_token(token, expected):
        return _login_page(next, error="访问令牌无效。", status_code=401)
    assert expected is not None
    response = RedirectResponse(_safe_next(next), status_code=303)
    response.set_cookie(
        COOKIE_NAME,
        expected,
        httponly=True,
        secure=request.url.scheme == "https",
        samesite="strict",
        max_age=30 * 24 * 60 * 60,
    )
    return response


__all__ = [
    "RemoteAccessMiddleware",
    "api_auth_headers",
    "api_token_path",
    "configured_api_token",
    "ensure_api_token",
    "is_loopback_host",
    "router",
]
