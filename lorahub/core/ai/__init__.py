"""AI subsystem package.

Side-effect imports the providers so the descriptor registry is
populated by the time anything else looks up ``PROVIDERS``.
"""

from __future__ import annotations

from lorahub.core.ai import providers as _providers  # noqa: F401
from lorahub.core.ai.provider_base import (
    AIProvider,
    ChatMessage,
    ChatOptions,
    ChatResult,
    ModelInfo,
    PROVIDERS,
    ProviderDescriptor,
    ProviderError,
    list_providers,
)

__all__ = [
    "AIProvider",
    "ChatMessage",
    "ChatOptions",
    "ChatResult",
    "ModelInfo",
    "PROVIDERS",
    "ProviderDescriptor",
    "ProviderError",
    "list_providers",
]
