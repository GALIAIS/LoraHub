"""Zhipu GLM — `open.bigmodel.cn` exposes an OpenAI-compatible API."""

from __future__ import annotations

from lorahub.core.ai.openai_base import OpenAIChatProvider
from lorahub.core.ai.provider_base import (
    ModelInfo,
    ProviderDescriptor,
    register,
)

DESCRIPTOR = ProviderDescriptor(
    id="glm",
    name="智谱 GLM",
    homepage="https://open.bigmodel.cn",
    docs_url="https://bigmodel.cn/dev/api",
    auth_help="在 https://bigmodel.cn/usercenter/apikeys 创建 API Key",
    default_base_url="https://open.bigmodel.cn/api/paas/v4",
    default_model="glm-4-plus",
    models=(
        ModelInfo("glm-4-plus", "GLM-4-Plus", context=128_000),
        ModelInfo("glm-4-air", "GLM-4-Air", context=128_000),
        ModelInfo("glm-4-flash", "GLM-4-Flash", context=128_000),
        ModelInfo("glm-4v-plus", "GLM-4V-Plus", vision=True, context=8_000),
        ModelInfo("glm-4v", "GLM-4V", vision=True, context=2_000),
    ),
)
register(DESCRIPTOR)


class GLMProvider(OpenAIChatProvider):
    descriptor = DESCRIPTOR
