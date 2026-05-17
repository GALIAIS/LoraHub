"""DeepSeek — fully OpenAI-compatible at api.deepseek.com."""

from __future__ import annotations

from lorahub.core.ai.openai_base import OpenAIChatProvider
from lorahub.core.ai.provider_base import (
    ModelInfo,
    ProviderDescriptor,
    register,
)

DESCRIPTOR = ProviderDescriptor(
    id="deepseek",
    name="DeepSeek",
    homepage="https://platform.deepseek.com",
    docs_url="https://api-docs.deepseek.com/",
    auth_help="Create a key at https://platform.deepseek.com/api_keys",
    default_base_url="https://api.deepseek.com/v1",
    default_model="deepseek-chat",
    models=(
        ModelInfo("deepseek-chat", "DeepSeek-V3", context=64_000),
        ModelInfo("deepseek-reasoner", "DeepSeek-R1", context=64_000),
    ),
)
register(DESCRIPTOR)


class DeepSeekProvider(OpenAIChatProvider):
    descriptor = DESCRIPTOR
