"""Moonshot AI / Kimi — OpenAI-compatible chat at api.moonshot.cn."""

from __future__ import annotations

from lorahub.core.ai.openai_base import OpenAIChatProvider
from lorahub.core.ai.provider_base import (
    ModelInfo,
    ProviderDescriptor,
    register,
)

DESCRIPTOR = ProviderDescriptor(
    id="kimi",
    name="月之暗面 Kimi",
    homepage="https://platform.moonshot.cn",
    docs_url="https://platform.moonshot.cn/docs/api/chat",
    auth_help="在 https://platform.moonshot.cn/console/api-keys 创建 API Key",
    default_base_url="https://api.moonshot.cn/v1",
    default_model="moonshot-v1-32k",
    models=(
        ModelInfo("moonshot-v1-8k", "Kimi-8K", context=8_000),
        ModelInfo("moonshot-v1-32k", "Kimi-32K", context=32_000),
        ModelInfo("moonshot-v1-128k", "Kimi-128K", context=128_000),
        ModelInfo("moonshot-v1-32k-vision-preview", "Kimi-Vision-32K", vision=True, context=32_000),
    ),
)
register(DESCRIPTOR)


class KimiProvider(OpenAIChatProvider):
    descriptor = DESCRIPTOR
