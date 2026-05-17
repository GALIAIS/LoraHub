"""Anthropic Claude — uses its own messages API (not OpenAI-style)."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from lorahub.core.ai.provider_base import (
    AIProvider,
    ChatMessage,
    ChatOptions,
    ChatResult,
    ModelInfo,
    ProviderDescriptor,
    ProviderError,
    register,
)

_log = logging.getLogger(__name__)

DESCRIPTOR = ProviderDescriptor(
    id="anthropic",
    name="Anthropic Claude",
    homepage="https://www.anthropic.com",
    docs_url="https://docs.anthropic.com/en/api/messages",
    auth_help="Create a key at https://console.anthropic.com/settings/keys",
    default_base_url="https://api.anthropic.com/v1",
    default_model="claude-sonnet-4-5",
    models=(
        ModelInfo("claude-opus-4-5", "Claude Opus 4.5", vision=True, context=200_000),
        ModelInfo("claude-sonnet-4-5", "Claude Sonnet 4.5", vision=True, context=200_000),
        ModelInfo("claude-haiku-4-5", "Claude Haiku 4.5", vision=True, context=200_000),
    ),
)
register(DESCRIPTOR)


class AnthropicProvider(AIProvider):
    descriptor = DESCRIPTOR

    def chat(self, messages: list[ChatMessage], options: ChatOptions) -> ChatResult:
        import requests  # noqa: PLC0415

        body = self._build_body(messages, options, stream=False)
        try:
            r = requests.post(
                f"{self.base_url}/messages",
                headers=self._headers(),
                json=body,
                timeout=options.timeout_s,
            )
        except requests.RequestException as exc:
            raise ProviderError(
                f"network error reaching anthropic: {exc}",
                provider="anthropic",
                retryable=True,
            ) from exc

        if r.status_code != 200:
            raise ProviderError(
                f"anthropic returned {r.status_code}: {r.text[:500]}",
                provider="anthropic",
                status_code=r.status_code,
                retryable=r.status_code in {408, 429, 500, 502, 503, 504, 529},
            )
        try:
            data = r.json()
        except ValueError as exc:
            raise ProviderError(
                f"anthropic returned non-JSON: {r.text[:500]}",
                provider="anthropic",
            ) from exc

        # Anthropic returns content as a list of blocks; concat the text ones.
        blocks = data.get("content") or []
        text_parts = [b.get("text") or "" for b in blocks if b.get("type") == "text"]
        usage = data.get("usage") or {}
        return ChatResult(
            text="".join(text_parts),
            model=data.get("model") or options.model or "",
            finish_reason=data.get("stop_reason"),
            usage_input_tokens=usage.get("input_tokens"),
            usage_output_tokens=usage.get("output_tokens"),
            raw=data,
        )

    async def chat_stream(
        self, messages: list[ChatMessage], options: ChatOptions
    ) -> AsyncIterator[str]:
        import httpx  # noqa: PLC0415

        body = self._build_body(messages, options, stream=True)
        async with httpx.AsyncClient(timeout=options.timeout_s) as client:
            try:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/messages",
                    headers=self._headers(),
                    json=body,
                ) as r:
                    if r.status_code != 200:
                        text = (await r.aread()).decode("utf-8", "replace")
                        raise ProviderError(
                            f"anthropic stream returned {r.status_code}: {text[:500]}",
                            provider="anthropic",
                            status_code=r.status_code,
                            retryable=r.status_code in {408, 429, 500, 502, 503, 504, 529},
                        )
                    async for raw_line in r.aiter_lines():
                        line = raw_line.strip()
                        if not line.startswith("data:"):
                            continue
                        payload = line[len("data:") :].strip()
                        if not payload or payload == "[DONE]":
                            continue
                        try:
                            chunk = json.loads(payload)
                        except ValueError:
                            continue
                        if chunk.get("type") == "content_block_delta":
                            delta = chunk.get("delta") or {}
                            if delta.get("type") == "text_delta":
                                t = delta.get("text") or ""
                                if t:
                                    yield t
            except httpx.HTTPError as exc:
                raise ProviderError(
                    f"network error during anthropic stream: {exc}",
                    provider="anthropic",
                    retryable=True,
                ) from exc

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise ProviderError(
                "anthropic: api_key not configured",
                provider="anthropic",
                status_code=401,
            )
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    def _build_body(
        self,
        messages: list[ChatMessage],
        options: ChatOptions,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        # Anthropic separates the system prompt from the messages list.
        system_parts: list[str] = []
        user_assistant: list[ChatMessage] = []
        for m in messages:
            if m.role == "system":
                if isinstance(m.content, str):
                    system_parts.append(m.content)
                else:
                    system_parts.append(
                        " ".join(p.get("text", "") for p in m.content if p.get("type") == "text")
                    )
            else:
                user_assistant.append(m)
        body: dict[str, Any] = {
            "model": options.model or self.descriptor.default_model,
            "max_tokens": options.max_tokens or 4096,
            "messages": [self._encode_message(m) for m in user_assistant],
            "stream": stream,
            "temperature": options.temperature,
        }
        if system_parts:
            body["system"] = "\n\n".join(system_parts)
        body.update(options.extra)
        return body

    def _encode_message(self, m: ChatMessage) -> dict[str, Any]:
        # Anthropic also accepts a list of content parts; image parts use
        # `{"type": "image", "source": {"type": "base64", ...}}` rather
        # than OpenAI's image_url shape. Translate.
        if isinstance(m.content, str):
            return {"role": m.role, "content": m.content}
        out_parts: list[dict[str, Any]] = []
        for part in m.content:
            if part.get("type") == "text":
                out_parts.append({"type": "text", "text": part.get("text", "")})
            elif part.get("type") == "image_url":
                url = (part.get("image_url") or {}).get("url", "")
                if url.startswith("data:"):
                    # data:image/png;base64,XXXX
                    head, _, b64 = url.partition(",")
                    media_type = head[len("data:") :].split(";", 1)[0] or "image/png"
                    out_parts.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": b64,
                            },
                        }
                    )
                else:
                    out_parts.append(
                        {
                            "type": "image",
                            "source": {"type": "url", "url": url},
                        }
                    )
        return {"role": m.role, "content": out_parts}
