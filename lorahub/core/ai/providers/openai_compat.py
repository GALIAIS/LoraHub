"""Catch-all OpenAI-compatible provider.

For every endpoint that speaks OpenAI's chat-completions JSON shape but
isn't covered by a first-class entry: One-API, vLLM, Ollama (with the
OpenAI compatibility shim), Together, Groq, Fireworks, OpenRouter, and
so on.

The user supplies ``base_url`` and a free-form model name. We store no
catalogue — it's their responsibility to know what the endpoint serves.
"""

from __future__ import annotations

from lorahub.core.ai.openai_base import OpenAIChatProvider
from lorahub.core.ai.provider_base import (
    ProviderDescriptor,
    register,
)

DESCRIPTOR = ProviderDescriptor(
    id="openai_compat",
    name="OpenAI 兼容 (自定义)",
    homepage="",
    docs_url="https://platform.openai.com/docs/api-reference/chat/create",
    auth_help=(
        "用于一切兼容 OpenAI ChatCompletion 协议的端点: One-API, vLLM, "
        "Ollama, OpenRouter, Together, Groq, Fireworks 等。请填写 base_url"
        " (含 /v1) 和你打算用的模型 ID。"
    ),
    default_base_url="http://localhost:8000/v1",
    default_model=None,
    custom_base_url=True,
    models=(),  # user-supplied
)
register(DESCRIPTOR)


class OpenAICompatProvider(OpenAIChatProvider):
    descriptor = DESCRIPTOR
