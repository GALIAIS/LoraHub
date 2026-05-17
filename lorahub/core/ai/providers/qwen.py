"""Qwen via DashScope's OpenAI-compatible endpoint.

DashScope offers two surfaces: the legacy native one and an
OpenAI-compatible passthrough at
``https://dashscope.aliyuncs.com/compatible-mode/v1``. We use the
latter so we get both Qwen text models and Qwen-VL vision models for
free.
"""

from __future__ import annotations

from lorahub.core.ai.openai_base import OpenAIChatProvider
from lorahub.core.ai.provider_base import (
    ModelInfo,
    ProviderDescriptor,
    register,
)

DESCRIPTOR = ProviderDescriptor(
    id="qwen",
    name="阿里通义千问",
    homepage="https://dashscope.console.aliyun.com",
    docs_url="https://help.aliyun.com/zh/dashscope/developer-reference/compatibility-of-openai-with-dashscope",
    auth_help="在 https://dashscope.console.aliyun.com/apiKey 创建 DashScope API Key",
    default_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    default_model="qwen-plus",
    models=(
        ModelInfo("qwen-max", "Qwen-Max", context=32_768),
        ModelInfo("qwen-plus", "Qwen-Plus", context=131_072),
        ModelInfo("qwen-turbo", "Qwen-Turbo", context=131_072),
        ModelInfo("qwen3-235b-a22b", "Qwen3-235B", context=131_072),
        ModelInfo("qwen3-32b", "Qwen3-32B", context=131_072),
        ModelInfo("qwen-vl-max", "Qwen-VL-Max", vision=True, context=32_768),
        ModelInfo("qwen-vl-plus", "Qwen-VL-Plus", vision=True, context=32_768),
        ModelInfo("qwen2.5-vl-72b-instruct", "Qwen2.5-VL-72B", vision=True, context=131_072),
    ),
)
register(DESCRIPTOR)


class QwenProvider(OpenAIChatProvider):
    descriptor = DESCRIPTOR
