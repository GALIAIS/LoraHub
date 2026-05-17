"""Concrete provider implementations.

Importing this package side-effect-registers every provider's
descriptor into ``lorahub.core.ai.provider_base.PROVIDERS``. The API
router triggers this once at import time (see
``lorahub.api.routers.ai``).
"""

from __future__ import annotations

from . import (
    anthropic,
    deepseek,
    doubao,
    glm,
    google,
    kimi,
    openai,
    openai_compat,
    qwen,
)

__all__ = [
    "anthropic",
    "deepseek",
    "doubao",
    "glm",
    "google",
    "kimi",
    "openai",
    "openai_compat",
    "qwen",
]
