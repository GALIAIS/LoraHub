"""ByteDance Volcengine Doubao — OpenAI-compatible at ark.cn-beijing.volces.com.

Doubao requires that the request ``model`` field be a Volcengine
"endpoint id" (something like ``ep-20240101-xyz``) rather than the
public model name. The user creates these endpoints in the Volcengine
console and copies the ID into our model picker / default_model field.
That's why the ``models`` catalogue here lists shapes with documented
``id`` placeholders rather than fixed strings.
"""

from __future__ import annotations

from lorahub.core.ai.openai_base import OpenAIChatProvider
from lorahub.core.ai.provider_base import (
    ModelInfo,
    ProviderDescriptor,
    register,
)

DESCRIPTOR = ProviderDescriptor(
    id="doubao",
    name="字节豆包 (Volcengine)",
    homepage="https://www.volcengine.com/product/ark",
    docs_url="https://www.volcengine.com/docs/82379/1099475",
    auth_help=(
        "在 https://console.volcengine.com/ark 创建 API Key + 推理接入点 (ep-...);"
        " 把接入点 ID 填到下面的「默认模型」"
    ),
    default_base_url="https://ark.cn-beijing.volces.com/api/v3",
    default_model=None,  # user must paste their endpoint id
    custom_base_url=True,
    models=(
        ModelInfo("ep-doubao-pro", "Doubao-Pro (输入你自己的 ep- ID)", context=32_000),
        ModelInfo("ep-doubao-vision", "Doubao-Vision (输入你自己的 ep- ID)", vision=True, context=32_000),
    ),
)
register(DESCRIPTOR)


class DoubaoProvider(OpenAIChatProvider):
    descriptor = DESCRIPTOR
    chat_path = "/chat/completions"
