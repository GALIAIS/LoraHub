"""LoraHub AI subsystem.

Modelled after ShiroManager: every provider is OpenAI-compatible with a
custom base_url + headers + optional org/project; each provider can hold
multiple API keys with runtime stats and cooldowns; models are either
manual or auto-discovered; tasks are routed to a (provider, model,
sampling, system_prompt) tuple via per-task records in ai.sqlite.

Public API:

    from lorahub.api.ai_store import AIStore, AIProvider, AIProviderKey,
                                       AIModel, AIRoute
    from lorahub.core.ai import client

    client.invoke(store, provider_id=..., model_id=..., messages=...)
    client.discover_models(store, provider_id=...)
    client.test_connection(store, provider_id=..., model_id=..., prompt=...)
"""

from __future__ import annotations

from lorahub.core.ai import client

__all__ = ["client"]
