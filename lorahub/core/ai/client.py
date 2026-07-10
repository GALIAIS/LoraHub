"""OpenAI-compatible HTTP client + key-rotation dispatcher.

ShiroManager treats every provider as an OpenAI-compatible chat-completions
endpoint with a custom base_url + headers + optional org/project. We do
the same, plus:

  * key rotation across multiple keys per provider (round-robin or random,
    skipping keys that are still in cooldown)
  * cooldown windows on rate-limit / auth failures
  * runtime stats persisted back to ai.sqlite via AIStore.update_key_runtime
  * /v1/models discovery used by `discoverAiModels`
  * connection test that lists models AND runs an optional 1-token chat
  * task invocation that resolves the route's provider+model+sampling
    overrides, then issues the chat completion with prompt + system prompt

Base-URL handling mirrors ShiroManager's `buildEndpointUrl`:

  * If the user-supplied base_url already ends in `/v1` (or any segment
    ending in `/v1`), the `/v1` prefix is *stripped* from the endpoint
    path before joining — so `https://api.x/v1` + `/v1/models` becomes
    `https://api.x/v1/models`, not `…/v1/v1/models`.
  * Otherwise the endpoint path is joined verbatim — so `https://api.x`
    + `/v1/models` becomes `https://api.x/v1/models`.
  * Trailing slashes, fragments, and query strings on the base_url are
    discarded on every call.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from lorahub.api.ai_store import (
    AIModel,
    AIProvider,
    AIProviderKey,
    AIRoute,
    AIStore,
)
from lorahub.core.redaction import redact_command_text

_log = logging.getLogger(__name__)


class AIError(RuntimeError):
    """Wrapper for upstream errors with HTTP status awareness."""

    def __init__(
        self,
        message: str,
        *,
        provider_id: str | None = None,
        status_code: int | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.provider_id = provider_id
        self.status_code = status_code
        self.retryable = retryable


@dataclass(slots=True)
class InvokeResult:
    content: str
    reasoning: str | None = None
    finish_reason: str | None = None
    model_id: str = ""
    provider_id: str = ""
    provider_name: str = ""
    usage_input_tokens: int | None = None
    usage_output_tokens: int | None = None
    usage_total_tokens: int | None = None


@dataclass(slots=True)
class ConnectionTestResult:
    ok: bool
    provider_id: str
    provider_name: str
    models: list[dict[str, Any]] = field(default_factory=list)
    completion: InvokeResult | None = None
    error: str | None = None


# --------------------------------------------------------------------------- #
# URL helpers
# --------------------------------------------------------------------------- #


_TRAILING_SLASHES = re.compile(r"/+$")
_MAX_CHAT_RESPONSE_BYTES = 16 * 1024**2
_MAX_MODELS_RESPONSE_BYTES = 8 * 1024**2


def build_endpoint_url(base_url: str, endpoint_path: str) -> str:
    """Join an OpenAI-style endpoint path onto a user-supplied base URL.

    Endpoint paths are written with their conventional `/v1/...` prefix
    (e.g. `/v1/chat/completions`). When the base URL already ends in a
    `/v1` segment we strip the prefix to avoid the `/v1/v1/...` double
    that comes from naive concatenation.

    Raises ``ValueError`` if base_url is empty or unparseable.
    """
    if not base_url or not base_url.strip():
        raise ValueError("AI provider base URL is required.")
    parts = urlsplit(base_url.strip())
    if parts.scheme.lower() not in {"http", "https"} or not parts.hostname:
        raise ValueError("AI provider base URL must use HTTP or HTTPS")
    if parts.username is not None or parts.password is not None:
        raise ValueError("AI provider base URL cannot contain credentials")
    base_path = _TRAILING_SLASHES.sub("", parts.path or "")
    suffix = endpoint_path if endpoint_path.startswith("/") else f"/{endpoint_path}"
    if base_path == "/v1" or base_path.endswith("/v1"):
        joined_path = base_path + re.sub(r"^/v1", "", suffix)
        if not joined_path:
            joined_path = "/v1"
    else:
        joined_path = (base_path + suffix) if base_path else suffix
    return urlunsplit((parts.scheme, parts.netloc, joined_path, "", ""))


# --------------------------------------------------------------------------- #
# HTTP helpers
# --------------------------------------------------------------------------- #


def _provider_headers(provider: AIProvider, key: AIProviderKey) -> dict[str, str]:
    h: dict[str, str] = {
        "Authorization": f"Bearer {key.api_key}",
        "Content-Type": "application/json",
    }
    if provider.organization:
        h["OpenAI-Organization"] = provider.organization
    if provider.project:
        h["OpenAI-Project"] = provider.project
    h.update(provider.headers)
    return h


def _post_chat(
    base_url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    import httpx  # noqa: PLC0415

    url = build_endpoint_url(base_url, "/v1/chat/completions")
    try:
        with httpx.Client(timeout=timeout) as client:
            with client.stream("POST", url, headers=headers, json=body) as response:
                status_code = response.status_code
                raw = _read_http_body(response, _MAX_CHAT_RESPONSE_BYTES)
    except httpx.HTTPError as exc:
        raise AIError(
            "network error reaching AI provider: "
            + redact_command_text(str(exc))[:500],
            status_code=None,
            retryable=True,
        ) from exc
    if status_code != 200:
        detail = redact_command_text(raw.decode("utf-8", errors="replace"))[:500]
        raise AIError(
            f"upstream returned {status_code}: {detail}",
            status_code=status_code,
            retryable=status_code in {408, 429, 500, 502, 503, 504, 529},
        )
    try:
        decoded = json.loads(raw)
    except (UnicodeDecodeError, ValueError) as exc:
        detail = redact_command_text(raw.decode("utf-8", errors="replace"))[:500]
        raise AIError(f"upstream returned non-JSON: {detail}") from exc
    if not isinstance(decoded, dict):
        raise AIError("upstream returned a non-object JSON response")
    return decoded


def _get_models(
    base_url: str,
    headers: dict[str, str],
    timeout: float,
) -> list[dict[str, Any]]:
    import httpx  # noqa: PLC0415

    url = build_endpoint_url(base_url, "/v1/models")
    try:
        with httpx.Client(timeout=timeout) as client:
            with client.stream("GET", url, headers=headers) as response:
                status_code = response.status_code
                raw = _read_http_body(response, _MAX_MODELS_RESPONSE_BYTES)
    except httpx.HTTPError as exc:
        raise AIError(
            "network error reaching AI provider: "
            + redact_command_text(str(exc))[:500],
            retryable=True,
        ) from exc
    if status_code != 200:
        detail = redact_command_text(raw.decode("utf-8", errors="replace"))[:500]
        raise AIError(
            f"models endpoint returned {status_code}: {detail}",
            status_code=status_code,
        )
    try:
        data = json.loads(raw)
    except (UnicodeDecodeError, ValueError) as exc:
        detail = redact_command_text(raw.decode("utf-8", errors="replace"))[:500]
        raise AIError(f"models endpoint returned non-JSON: {detail}") from exc
    if not isinstance(data, dict):
        raise AIError("models endpoint returned a non-object JSON response")
    items = data.get("data") or data.get("models") or []
    if not isinstance(items, list):
        return []
    out: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            out.append(item)
        elif isinstance(item, str):
            out.append({"id": item})
    return out


def _read_http_body(response: Any, limit: int) -> bytes:
    declared = response.headers.get("content-length")
    if declared:
        try:
            if int(declared) > limit:
                raise AIError(f"AI provider response exceeds {limit} bytes")
        except ValueError:
            pass
    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_bytes():
        total += len(chunk)
        if total > limit:
            raise AIError(f"AI provider response exceeds {limit} bytes")
        chunks.append(chunk)
    return b"".join(chunks)


# --------------------------------------------------------------------------- #
# Key selection
# --------------------------------------------------------------------------- #


def _now() -> datetime:
    return datetime.now(UTC)


def _is_cooled_down(key: AIProviderKey) -> bool:
    cd = key.runtime.cooldown_until
    if not cd:
        return True
    try:
        until = datetime.fromisoformat(cd)
    except ValueError:
        return True
    return _now() >= until


def _pick_key(
    provider: AIProvider,
    keys: list[AIProviderKey],
) -> tuple[AIProviderKey, int]:
    """Return (chosen_key, new_last_key_index) honouring selection_mode.

    Skips any key whose cooldown_until is still in the future. Falls back
    to ANY key if all are cooling down (better to retry than to refuse;
    upstream will tell us if we're still rate-limited).
    """
    eligible = [(i, k) for i, k in enumerate(keys) if _is_cooled_down(k)]
    pool = eligible or list(enumerate(keys))
    if not pool:
        raise AIError(
            f"provider {provider.id} has no API keys configured",
            provider_id=provider.id,
            status_code=400,
        )
    if provider.selection_mode == "random":
        idx, key = random.choice(pool)  # noqa: S311 -- not security-sensitive
        return key, idx
    # round-robin: pick the entry after `last_key_index`
    last = provider.last_key_index
    after = [(i, k) for i, k in pool if i > last]
    chosen = after[0] if after else pool[0]
    idx, key = chosen
    return key, idx


# --------------------------------------------------------------------------- #
# Public dispatcher API
# --------------------------------------------------------------------------- #


def _route_payload(
    route: AIRoute | None,
    overrides: dict[str, Any],
) -> dict[str, Any]:
    """Merge a stored route with per-call overrides into one sampling dict.

    Only keys with non-None values land in the body so the upstream
    keeps its defaults for everything else.
    """
    base: dict[str, Any] = {}
    if route is not None:
        for k in (
            "stream",
            "temperature",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "max_output_tokens",
            "seed",
            "reasoning_effort",
            "thinking_budget_tokens",
            "include_reasoning",
        ):
            v = getattr(route, k)
            if v is not None:
                base[k] = v
        if route.stop_sequences:
            base["stop"] = list(route.stop_sequences)
    for k, v in overrides.items():
        if v is not None:
            base[k] = v
    return base


def _apply_extra_body(
    body: dict[str, Any], extra_body_json: str | None
) -> dict[str, Any]:
    """Merge a JSON-encoded `extra_body` into the request body.

    Invalid JSON and core-field overrides are rejected with an actionable
    client error instead of silently changing the request contract.
    """
    if not extra_body_json:
        return body
    try:
        extra = json.loads(extra_body_json)
    except json.JSONDecodeError as exc:
        raise AIError(
            f"extraBodyJson is invalid JSON at position {exc.pos}",
            status_code=400,
        ) from exc
    if not isinstance(extra, dict):
        raise AIError("extraBodyJson must contain a JSON object", status_code=400)
    reserved = {"messages", "model"}.intersection(extra)
    if reserved:
        raise AIError(
            "extraBodyJson cannot override: " + ", ".join(sorted(reserved)),
            status_code=400,
        )
    body.update(extra)
    return body


def _build_chat_body(
    *,
    model_id: str,
    messages: list[dict[str, Any]],
    sampling: dict[str, Any],
    extra_body_json: str | None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model_id,
        "messages": messages,
    }
    # max_output_tokens -> OpenAI's max_tokens; reasoning_effort and
    # thinking_budget_tokens stay verbatim because vendors that support
    # them accept the same names (DeepSeek-Reasoner, Anthropic via
    # OpenAI-compat shim, etc).
    if "max_output_tokens" in sampling:
        body["max_tokens"] = sampling.pop("max_output_tokens")
    body.update(sampling)
    return _apply_extra_body(body, extra_body_json)


def _parse_chat_response(
    raw: dict[str, Any], provider: AIProvider, model_id: str
) -> InvokeResult:
    choices = raw.get("choices") or []
    if not isinstance(choices, list) or not choices:
        raise AIError(
            f"empty choices in response from {provider.id}",
            provider_id=provider.id,
        )
    first = choices[0]
    if not isinstance(first, dict):
        raise AIError("upstream returned an invalid choices entry")
    msg = first.get("message") or {}
    if not isinstance(msg, dict):
        raise AIError("upstream returned an invalid message object")
    content = msg.get("content")
    if isinstance(content, list):
        # Multimodal-style content list — concat text parts.
        content = "".join(
            str(p.get("text") or "") for p in content if isinstance(p, dict)
        )
    if content is None:
        content = ""
    elif not isinstance(content, str):
        raise AIError("upstream returned invalid message content")
    usage = raw.get("usage") or {}
    if not isinstance(usage, dict):
        usage = {}
    reasoning = msg.get("reasoning") or msg.get("reasoning_content")
    if reasoning is not None and not isinstance(reasoning, str):
        reasoning = str(reasoning)
    response_model = raw.get("model") or model_id
    return InvokeResult(
        content=content,
        reasoning=reasoning,
        finish_reason=first.get("finish_reason"),
        model_id=str(response_model),
        provider_id=provider.id,
        provider_name=provider.name,
        usage_input_tokens=usage.get("prompt_tokens"),
        usage_output_tokens=usage.get("completion_tokens"),
        usage_total_tokens=usage.get("total_tokens"),
    )


def invoke(
    store: AIStore,
    *,
    provider_id: str,
    model_id: str,
    messages: list[dict[str, Any]],
    route: AIRoute | None = None,
    overrides: dict[str, Any] | None = None,
    extra_body_json: str | None = None,
    timeout: float = 60.0,
    cooldown_seconds_on_rate_limit: int = 60,
) -> InvokeResult:
    """Execute one chat call against `provider_id` using `model_id`.

    Picks a key respecting cooldown + selection_mode, sends the request,
    and on 429 / upstream auth errors marks the chosen key with a cooldown
    so subsequent calls move to a different key automatically.
    """
    provider = store.get_provider(provider_id)
    if provider is None:
        raise AIError(
            f"provider {provider_id!r} not found",
            provider_id=provider_id,
            status_code=404,
        )
    if not provider.enabled:
        raise AIError(
            f"provider {provider_id!r} is disabled",
            provider_id=provider_id,
            status_code=409,
        )
    keys = store.list_keys(provider_id)
    if not keys:
        raise AIError(
            f"provider {provider_id!r} has no API keys",
            provider_id=provider_id,
            status_code=400,
        )

    key, idx = _pick_key(provider, keys)
    headers = _provider_headers(provider, key)
    sampling = _route_payload(route, overrides or {})
    body = _build_chat_body(
        model_id=model_id,
        messages=messages,
        sampling=sampling,
        extra_body_json=(extra_body_json if extra_body_json else
                         (route.extra_body_json if route else None)),
    )

    started = time.monotonic()
    try:
        raw = _post_chat(provider.base_url, headers, body, timeout=timeout)
    except AIError as exc:
        cooldown_until: str | None = None
        if exc.status_code in {401, 403, 429}:
            cooldown_until = (
                _now() + timedelta(seconds=cooldown_seconds_on_rate_limit)
            ).isoformat()
        store.update_key_runtime(
            key.id,
            success=False,
            error=redact_command_text(str(exc))[:2_000],
            cooldown_until=cooldown_until,
        )
        store.update_provider_last_index(provider.id, idx)
        raise
    elapsed = time.monotonic() - started
    _log.debug("ai invoke ok provider=%s key=%s ms=%.0f", provider.id, key.id, elapsed * 1000)
    store.update_key_runtime(key.id, success=True)
    store.update_provider_last_index(provider.id, idx)
    return _parse_chat_response(raw, provider, model_id)


def discover_models(store: AIStore, provider_id: str) -> list[AIModel]:
    """Hit `<base_url>/models`, persist results as source='discovered'.

    Manually-added rows (`source='manual'`) are preserved by AIStore's
    replace_discovered_models. Returns the new full discovered set.
    """
    provider = store.get_provider(provider_id)
    if provider is None:
        raise AIError(
            f"provider {provider_id!r} not found",
            provider_id=provider_id,
            status_code=404,
        )
    keys = store.list_keys(provider_id)
    if not keys:
        raise AIError(
            f"provider {provider_id!r} has no API keys",
            provider_id=provider_id,
            status_code=400,
        )
    key, idx = _pick_key(provider, keys)
    headers = _provider_headers(provider, key)
    try:
        items = _get_models(provider.base_url, headers, timeout=15.0)
    except AIError as exc:
        store.update_key_runtime(
            key.id,
            success=False,
            error=redact_command_text(str(exc))[:2_000],
        )
        raise
    store.update_key_runtime(key.id, success=True)
    store.update_provider_last_index(provider.id, idx)
    drafts: list[AIModel] = []
    for item in items:
        raw_id = item.get("id")
        mid = raw_id.strip() if isinstance(raw_id, str) else ""
        if not mid:
            continue
        drafts.append(
            AIModel(
                id="",
                provider_id=provider.id,
                model_id=mid,
                display_name=mid,
                source="discovered",
                enabled=True,
                raw=item,
            )
        )
    return store.replace_discovered_models(provider.id, drafts)


def test_connection(
    store: AIStore,
    *,
    provider_id: str,
    model_id: str | None = None,
    prompt: str | None = None,
    system_prompt: str | None = None,
    sampling: dict[str, Any] | None = None,
) -> ConnectionTestResult:
    """List models + optionally run a one-shot chat to confirm the key."""
    try:
        provider = store.get_provider(provider_id)
        if provider is None:
            return ConnectionTestResult(
                ok=False,
                provider_id=provider_id,
                provider_name="",
                error=f"provider {provider_id!r} not found",
            )
        models = discover_models(store, provider_id)
        completion: InvokeResult | None = None
        if model_id and prompt:
            messages: list[dict[str, Any]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            completion = invoke(
                store,
                provider_id=provider_id,
                model_id=model_id,
                messages=messages,
                overrides=sampling or {},
            )
        return ConnectionTestResult(
            ok=True,
            provider_id=provider.id,
            provider_name=provider.name,
            models=[m.raw for m in models],
            completion=completion,
        )
    except AIError as exc:
        return ConnectionTestResult(
            ok=False,
            provider_id=provider_id,
            provider_name="",
            error=str(exc),
        )


__all__ = [
    "AIError",
    "ConnectionTestResult",
    "InvokeResult",
    "discover_models",
    "invoke",
    "test_connection",
]
