"""OpenAI-compatible chat-completions provider.

Most modern Chinese vendors (DeepSeek, Qwen DashScope, Moonshot/Kimi,
Zhipu GLM, ByteDance Doubao) ship a Bearer-token + JSON request body
that mirrors OpenAI's `/v1/chat/completions`. So one base class covers
all of them — concrete subclasses only override the descriptor and
occasionally tweak request shape.
"""

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
    ProviderDescriptor,
    ProviderError,
)

_log = logging.getLogger(__name__)


class OpenAIChatProvider(AIProvider):
    """Generic OpenAI-style chat completions client.

    Subclasses set ``descriptor``; everything else lives here. Streaming
    is parsed via Server-Sent Events (the ``data: {...}`` line format
    OpenAI documented and every clone copied).
    """

    descriptor: ProviderDescriptor  # set on subclasses

    # Endpoint suffix appended to base_url. OpenAI-style is
    # ``/chat/completions``; vendors that nest deeper (Doubao
    # ``/api/v3/chat/completions``) override this.
    chat_path: str = "/chat/completions"

    def chat(self, messages: list[ChatMessage], options: ChatOptions) -> ChatResult:
        import requests  # noqa: PLC0415

        body = self._build_body(messages, options, stream=False)
        try:
            r = requests.post(
                self.base_url + self.chat_path,
                headers=self._headers(),
                json=body,
                timeout=options.timeout_s,
            )
        except requests.RequestException as exc:
            raise ProviderError(
                f"network error reaching {self.descriptor.id}: {exc}",
                provider=self.descriptor.id,
                retryable=True,
            ) from exc

        if r.status_code != 200:
            raise ProviderError(
                f"{self.descriptor.id} returned {r.status_code}: {r.text[:500]}",
                provider=self.descriptor.id,
                status_code=r.status_code,
                retryable=r.status_code in {408, 429, 500, 502, 503, 504},
            )

        try:
            data = r.json()
        except ValueError as exc:
            raise ProviderError(
                f"{self.descriptor.id} returned non-JSON: {r.text[:500]}",
                provider=self.descriptor.id,
            ) from exc

        return self._parse_response(data, options)

    async def chat_stream(
        self, messages: list[ChatMessage], options: ChatOptions
    ) -> AsyncIterator[str]:
        import httpx  # noqa: PLC0415

        body = self._build_body(messages, options, stream=True)
        async with httpx.AsyncClient(timeout=options.timeout_s) as client:
            try:
                async with client.stream(
                    "POST",
                    self.base_url + self.chat_path,
                    headers=self._headers(),
                    json=body,
                ) as r:
                    if r.status_code != 200:
                        text = (await r.aread()).decode("utf-8", "replace")
                        raise ProviderError(
                            f"{self.descriptor.id} stream returned "
                            f"{r.status_code}: {text[:500]}",
                            provider=self.descriptor.id,
                            status_code=r.status_code,
                            retryable=r.status_code in {408, 429, 500, 502, 503, 504},
                        )
                    async for raw_line in r.aiter_lines():
                        line = raw_line.strip()
                        if not line or not line.startswith("data:"):
                            continue
                        payload = line[len("data:") :].strip()
                        if payload == "[DONE]":
                            return
                        try:
                            chunk = json.loads(payload)
                        except ValueError:
                            continue
                        for choice in chunk.get("choices", []):
                            delta = choice.get("delta") or {}
                            text = delta.get("content")
                            if text:
                                yield text
            except httpx.HTTPError as exc:
                raise ProviderError(
                    f"network error during {self.descriptor.id} stream: {exc}",
                    provider=self.descriptor.id,
                    retryable=True,
                ) from exc

    # ------------------------------------------------------------------ #
    # Hooks subclasses override
    # ------------------------------------------------------------------ #

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise ProviderError(
                f"{self.descriptor.id}: api_key not configured",
                provider=self.descriptor.id,
                status_code=401,
            )
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _build_body(
        self,
        messages: list[ChatMessage],
        options: ChatOptions,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": options.model or self.descriptor.default_model,
            "messages": [self._encode_message(m) for m in messages],
            "temperature": options.temperature,
            "stream": stream,
        }
        if options.max_tokens is not None:
            body["max_tokens"] = options.max_tokens
        if options.response_format == "json":
            body["response_format"] = {"type": "json_object"}
        body.update(options.extra)
        return body

    def _encode_message(self, m: ChatMessage) -> dict[str, Any]:
        # Pass content through as-is; OpenAI-style multimodal already
        # uses the list-of-parts shape we declared in ChatMessage.
        return {"role": m.role, "content": m.content}

    def _parse_response(
        self, data: dict[str, Any], options: ChatOptions
    ) -> ChatResult:
        choices = data.get("choices") or []
        if not choices:
            raise ProviderError(
                f"{self.descriptor.id}: empty choices in response",
                provider=self.descriptor.id,
            )
        message = choices[0].get("message") or {}
        text = message.get("content") or ""
        usage = data.get("usage") or {}
        return ChatResult(
            text=text if isinstance(text, str) else json.dumps(text),
            model=data.get("model") or options.model or "",
            finish_reason=choices[0].get("finish_reason"),
            usage_input_tokens=usage.get("prompt_tokens"),
            usage_output_tokens=usage.get("completion_tokens"),
            raw=data,
        )


__all__ = ["OpenAIChatProvider"]
