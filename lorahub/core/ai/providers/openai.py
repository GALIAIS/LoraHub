"""OpenAI provider — chat completions + GPT-4 vision."""

from __future__ import annotations

from lorahub.core.ai.openai_base import OpenAIChatProvider
from lorahub.core.ai.provider_base import (
    ModelInfo,
    ProviderDescriptor,
    register,
)

DESCRIPTOR = ProviderDescriptor(
    id="openai",
    name="OpenAI",
    homepage="https://openai.com",
    docs_url="https://platform.openai.com/docs/api-reference",
    auth_help="Create a key at https://platform.openai.com/api-keys",
    default_base_url="https://api.openai.com/v1",
    default_model="gpt-4o-mini",
    custom_base_url=False,
    models=(
        ModelInfo("gpt-4o", "GPT-4o", vision=True, context=128_000),
        ModelInfo("gpt-4o-mini", "GPT-4o mini", vision=True, context=128_000),
        ModelInfo("gpt-4.1", "GPT-4.1", vision=True, context=1_000_000),
        ModelInfo("gpt-4.1-mini", "GPT-4.1 mini", vision=True, context=1_000_000),
        ModelInfo("o3-mini", "o3-mini (reasoning)", vision=False, context=200_000),
    ),
)
register(DESCRIPTOR)


class OpenAIProvider(OpenAIChatProvider):
    descriptor = DESCRIPTOR
