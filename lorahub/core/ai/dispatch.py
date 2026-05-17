"""Pick the right provider class for a stored credential.

Maps the credential's ``provider`` string to the concrete
:class:`AIProvider` subclass. The store + dispatcher are kept separate
so the request-handling layer can construct providers on demand
without holding long-lived credential rows in memory.
"""

from __future__ import annotations

from typing import Any

from lorahub.api.ai_credentials_store import AICredential, AICredentialStore
from lorahub.core.ai.provider_base import AIProvider, ProviderError
from lorahub.core.ai.providers.anthropic import AnthropicProvider
from lorahub.core.ai.providers.deepseek import DeepSeekProvider
from lorahub.core.ai.providers.doubao import DoubaoProvider
from lorahub.core.ai.providers.glm import GLMProvider
from lorahub.core.ai.providers.google import GoogleProvider
from lorahub.core.ai.providers.kimi import KimiProvider
from lorahub.core.ai.providers.openai import OpenAIProvider
from lorahub.core.ai.providers.openai_compat import OpenAICompatProvider
from lorahub.core.ai.providers.qwen import QwenProvider

_REGISTRY: dict[str, type[AIProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "google": GoogleProvider,
    "deepseek": DeepSeekProvider,
    "qwen": QwenProvider,
    "kimi": KimiProvider,
    "glm": GLMProvider,
    "doubao": DoubaoProvider,
    "openai_compat": OpenAICompatProvider,
}


def build_provider(cred: AICredential) -> AIProvider:
    """Instantiate the provider class for a credential.

    Returns the live AIProvider; raises ProviderError if the credential's
    ``provider`` string is unknown.
    """
    cls = _REGISTRY.get(cred.provider)
    if cls is None:
        raise ProviderError(
            f"unknown provider {cred.provider!r}",
            provider=cred.provider,
            status_code=400,
        )
    return cls(api_key=cred.api_key, base_url=cred.base_url)


def load_provider(store: AICredentialStore, provider_id: str) -> AIProvider:
    """Look up a credential and build its provider in one step.

    Used by request handlers — keeps the credential lookup + provider
    construction in one place so we don't accidentally build providers
    against stale or missing credentials.
    """
    cred = store.get(provider_id)
    if cred is None:
        raise ProviderError(
            f"provider {provider_id!r} is not configured",
            provider=provider_id,
            status_code=404,
        )
    if not cred.enabled:
        raise ProviderError(
            f"provider {provider_id!r} is disabled in settings",
            provider=provider_id,
            status_code=409,
        )
    return build_provider(cred)


__all__ = ["build_provider", "load_provider"]
